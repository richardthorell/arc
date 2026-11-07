#include <arc/editor/arc_host.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/editor_history.h>
#include <arc/editor/entity_material_service.h>
#include <arc/editor/material_inspection.h>
#include <arc/editor/material_library.h>
#include <arc/editor/material_preview_realizer.h>
#include <arc/editor/procedural_mesh.h>
#include <arc/render/renderer.h>
#include <arc/render_tools/render_tools.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

TEST_CASE("compiled coat variants survive typed clearing reset and scene history")
{
    const arc::render::tools::slang_shader_compiler compiler;
    if (!compiler.available()) SKIP("Slang is required for compiled coat switching regression");
    const std::string source = R"({
      "version":4,"name":"Inherited coat","domain":"surface","blendMode":"opaque","shadingModel":"standard",
      "graph":{"version":1,"nodes":[
        {"id":"out","type":"output","values":{}},
        {"id":"tex","type":"textureSample2D","values":{"texture":"coat.png","semantic":"normal"},
         "parameter":{"exposed":true,"name":"Coat source"}},
        {"id":"coat","type":"normalMap","values":{}},
        {"id":"weight","type":"constant","values":{"value":0},
         "parameter":{"exposed":true,"name":"Coat weight"}}
      ],"connections":[
        {"id":"a","from":{"nodeId":"tex","pin":"rgb"},"to":{"nodeId":"coat","pin":"texture"}},
        {"id":"b","from":{"nodeId":"coat","pin":"normal"},"to":{"nodeId":"out","pin":"clearCoatNormal"}},
        {"id":"c","from":{"nodeId":"weight","pin":"value"},"to":{"nodeId":"out","pin":"clearCoat"}}
      ]}})";
    const auto realized = arc::editor::realize_material_preview_descriptor(source);
    INFO(realized.message);
    REQUIRE(realized.succeeded);
    REQUIRE(realized.material.coat_normal_variants);
    const auto variants = realized.material.coat_normal_variants;
    REQUIRE(variants->neutral);
    REQUIRE(variants->independent);
    arc::render::renderer renderer;
    arc::editor::editor_scene_state scene;
    auto asset = arc::editor::make_default_material_asset("Inherited coat");
    asset.material = realized.material;
    asset.path = "inherited_coat.arcmat";
    const auto shared = renderer.create_material(asset.material);
    scene.material_library.materials.push_back({asset.path, asset, shared});
    const auto first = scene.scene.create();
    const auto second = scene.scene.create();
    for (const auto entity : {first, second})
        scene.scene.emplace<arc::scene::mesh_renderer_component>(
            entity, arc::scene::mesh_renderer_component{.material = shared});
    arc::editor::ensure_scene_authoring_metadata(scene);
    const auto first_guid = arc::editor::entity_guid_of(scene, first);
    const auto second_guid = arc::editor::entity_guid_of(scene, second);
    auto read = [&](arc::ecs::entity_guid guid)
    {
        const auto entity = arc::editor::find_entity_by_guid(scene, guid);
        const auto* mesh = scene.scene.try_get<arc::scene::mesh_renderer_component>(entity);
        REQUIRE(mesh);
        return nlohmann::json::parse(arc::editor::inspect_applied_material(renderer.material_data_for(mesh->material)));
    };
    const auto neutral_generation = std::to_string(variants->neutral->generation);
    const auto active_generation = std::to_string(variants->independent->generation);
    REQUIRE(neutral_generation != active_generation);
    CHECK(read(first_guid)["programGeneration"] == active_generation);
    arc::editor::editor_history history;
    history.clear(scene);
    REQUIRE(history.begin(1, "Clear coat texture", scene));
    std::string error;
    REQUIRE(arc::editor::set_entity_material_parameter(scene, renderer, first, variants->texture_parameter, {}, nullptr,
                                                       false, error, {}, true));
    REQUIRE(history.commit(1, scene));
    CHECK(read(first_guid)["renderPath"] == "deferred");
    CHECK(read(first_guid)["programGeneration"] == neutral_generation);
    CHECK(read(second_guid)["programGeneration"] == active_generation);
    REQUIRE(history.undo(scene));
    arc::editor::synchronize_procedural_and_material_instances(scene, renderer);
    CHECK(read(first_guid)["renderPath"] == "clusteredForward");
    CHECK(read(first_guid)["programGeneration"] == active_generation);
    REQUIRE(history.redo(scene));
    arc::editor::synchronize_procedural_and_material_instances(scene, renderer);
    CHECK(read(first_guid)["programGeneration"] == neutral_generation);
    const auto target = arc::editor::find_entity_by_guid(scene, first_guid);
    REQUIRE(arc::editor::set_entity_material_parameter(scene, renderer, target, variants->texture_parameter, {},
                                                       nullptr, true, error));
    CHECK(read(first_guid)["programGeneration"] == active_generation);
    CHECK(arc::editor::entity_material_overrides_json(scene, target) == "[]");
    for (const float weight : {1.0f, 0.0f, 0.5f})
    {
        REQUIRE(arc::editor::set_entity_material_parameter(scene, renderer, target,
                                                           arc::render::make_shader_parameter_id("weight"),
                                                           std::span<const float>{&weight, 1}, nullptr, false, error));
        CHECK(read(first_guid)["renderPath"] == "clusteredForward");
        CHECK(read(first_guid)["programGeneration"] == active_generation);
        CHECK(read(second_guid)["programGeneration"] == active_generation);
    }
    const auto last_good = read(first_guid);
    const auto last_overrides = arc::editor::entity_material_overrides_json(scene, target);
    const arc::assets::asset_reference missing_texture{
        .guid = *arc::assets::parse_asset_guid("11111111-1111-4111-8111-111111111111"),
        .expected_type = arc::assets::asset_types::texture_2d};
    CHECK_FALSE(arc::editor::set_entity_material_parameter(scene, renderer, target, variants->texture_parameter, {},
                                                           &missing_texture, false, error));
    CHECK_FALSE(error.empty());
    CHECK(read(first_guid) == last_good);
    CHECK(arc::editor::entity_material_overrides_json(scene, target) == last_overrides);
}

TEST_CASE("built-in material routing is resolved by native material realization")
{
    const auto root = std::filesystem::path(ARC_SOURCE_ROOT) / "assets";
    for (const auto* file : {"standard_lit.arcmat", "unlit.arcmat", "glass.arcmat", "subsurface.arcmat"})
    {
        CAPTURE(file);
        arc::render::renderer renderer;
        arc::editor::editor_material_library library;
        arc::editor::material_asset loaded;
        REQUIRE(
            arc::editor::load_material_for_editor(library, renderer, root, root / "materials" / file, &loaded).valid());
        CHECK(arc::render::resolve_material_render_path(loaded.material) ==
              (std::string_view(file) == "standard_lit.arcmat" ? arc::render::material_render_path::deferred
                                                               : arc::render::material_render_path::clustered_forward));
    }
}

TEST_CASE("material compile responses report native coat variant routing from the unsaved document")
{
    const arc::render::tools::slang_shader_compiler compiler;
    if (!compiler.available()) SKIP("Slang is required for native material compile routing");

    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc_material_compile_routing_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    struct cleanup
    {
        std::filesystem::path root;
        ~cleanup()
        {
            std::error_code error;
            std::filesystem::remove_all(root, error);
        }
    } remove_root{root};
    arc::editor::arc_host host(std::make_unique<arc::render::renderer>());
    REQUIRE(host.open_project({.name = "Routing", .root = root}, {}).succeeded);

    const std::string graph = R"({"version":1,"nodes":[
      {"id":"out","type":"output","values":{}},
      {"id":"tex","type":"textureSample2D","values":{"texture":"","semantic":"normal"}},
      {"id":"coat","type":"normalMap","values":{}}
    ],"connections":[
      {"id":"a","from":{"nodeId":"tex","pin":"rgb"},"to":{"nodeId":"coat","pin":"texture"}},
      {"id":"b","from":{"nodeId":"coat","pin":"normal"},"to":{"nodeId":"out","pin":"clearCoatNormal"}}
    ]})";
    std::string document_graph = graph;
    std::string blend = "opaque";
    std::string expected = "deferred";
    SECTION("Unbound normal is deferred despite its connected graph") {}
    SECTION("Bound normal is forward")
    {
        const std::string empty = "\"texture\":\"\"";
        document_graph.replace(document_graph.find(empty), empty.size(), "\"texture\":\"coat.png\"");
        expected = "clusteredForward";
    }
    SECTION("Unsaved blend settings take precedence over standalone graph defaults")
    {
        blend = "blend";
        expected = "clusteredForward";
    }
    arc::editor::host_shader_compile_command command{
        .path = "Preview.generated.slang",
        .source = document_graph,
        .domain = "materialGraph",
        .preview_source = "{\"version\":4,\"name\":\"Preview\",\"domain\":\"surface\",\"blendMode\":\"" + blend +
                          "\",\"shadingModel\":\"standard\",\"graph\":" + document_graph + "}",
    };
    const auto response = host.execute(command);
    INFO(response.error);
    INFO(response.payload_json);
    REQUIRE(response.succeeded);
    CHECK(response.payload_json.find("\"materialRenderPath\":\"" + expected + "\"") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(root / command.path));

    command.preview_source = "malformed document";
    const auto invalid = host.execute(command);
    CHECK(invalid.payload_json.find("materialRenderPath") == std::string::npos);
}
