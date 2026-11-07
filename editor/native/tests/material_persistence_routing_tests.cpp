#include <arc/editor/scene_document.h>
#include <arc/editor/entity_material_service.h>
#include <arc/editor/material_library.h>
#include <arc/render/renderer.h>
#include <arc/render_tools/render_tools.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <fstream>

namespace
{
struct project_fixture
{
    std::filesystem::path root = std::filesystem::temp_directory_path() /
                                 ("arc-coat-reopen-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    project_fixture()
    {
        std::filesystem::create_directories(root / "Content");
    }
    ~project_fixture()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
    void write(std::string_view name, std::string_view bytes)
    {
        std::ofstream stream(root / "Content" / name, std::ios::binary);
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        REQUIRE(stream.good());
    }
};

struct registry_fixture
{
    arc::memory::memory_system memory;
    arc::jobs::job_system jobs{
        {.worker_count = 2, .io_worker_count = 1, .enable_render_thread = false, .memory = &memory}};
    arc::io::async_file_service files{jobs};
    arc::assets::asset_manager manager;
    arc::framework::runtime_service_registry services;
    arc::framework::runtime_service_context context{services};
    explicit registry_fixture(const std::filesystem::path& root)
        : manager({.project_root = root,
                   .asset_root = root / "Content",
                   .cache_root = root / ".arc/cache",
                   .enable_source_monitor = false},
                  jobs, files, memory)
    {
        manager.on_start(context);
        (void)manager.scan();
    }
    ~registry_fixture()
    {
        manager.on_shutdown(context);
    }
};
} // namespace

TEST_CASE("coat texture assignments clear and inheritance survive scene save and fresh reopen")
{
    using namespace arc;
    const render::tools::slang_shader_compiler compiler;
    if (!compiler.available()) SKIP("Slang is required for persisted compiled material routing");
    project_fixture project;
    const std::array<unsigned char, 21> tga{0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 24, 32, 255, 128, 128};
    project.write("normal.tga", {reinterpret_cast<const char*>(tga.data()), tga.size()});
    const std::string source =
        R"({"version":4,"name":"Coat parent","domain":"surface","blendMode":"opaque","shadingModel":"standard",
      "graph":{"version":1,"nodes":[
        {"id":"out","type":"output","values":{}},
        {"id":"tex","type":"textureSample2D","values":{"texture":"","semantic":"normal"},"parameter":{"exposed":true,"name":"Coat texture"}},
        {"id":"coat","type":"normalMap","values":{}}
      ],"connections":[
        {"id":"a","from":{"nodeId":"tex","pin":"rgb"},"to":{"nodeId":"coat","pin":"texture"}},
        {"id":"b","from":{"nodeId":"coat","pin":"normal"},"to":{"nodeId":"out","pin":"clearCoatNormal"}}
      ]}})";
    project.write("parent.arcmat", source);
    const auto parameter = render::make_shader_parameter_id("tex");
    bool instance = false;
    SECTION("Direct material binding") {}
    SECTION("Authored instance inherits an active normal")
    {
        instance = true;
    }
    std::array<ecs::entity_guid, 2> entities;
    assets::asset_guid material_guid, texture_guid;
    std::array<std::string, 3> overrides;
    {
        registry_fixture registry(project.root);
        const auto parent = registry.manager.find(std::string_view{"Content/parent.arcmat"});
        const auto texture = registry.manager.find(std::string_view{"Content/normal.tga"});
        REQUIRE(parent);
        REQUIRE(texture);
        texture_guid = texture->guid;
        const auto filename = instance ? "instance.arcmatinst" : "parent.arcmat";
        if (instance)
        {
            project.write(
                filename,
                nlohmann::json{{"version", 1},
                               {"name", "Inherited coat"},
                               {"parent", {{"guid", assets::to_string(parent->guid)}, {"pathHint", "parent.arcmat"}}},
                               {"parameterOverrides",
                                nlohmann::json::array(
                                    {{{"parameterId", std::to_string(parameter.value)}, {"value", "normal.tga"}}})},
                               {"functionOverrides", nlohmann::json::array()}}
                    .dump());
            (void)registry.manager.scan();
        }
        const auto registered = registry.manager.find(std::string("Content/") + filename);
        REQUIRE(registered);
        material_guid = registered->guid;
        render::renderer renderer;
        editor::editor_scene_state scene;
        scene.project_root = project.root;
        scene.asset_registry = &registry.manager;
        const auto material = editor::load_material_for_editor(
            scene.material_library, renderer, project.root / "Content", project.root / "Content" / filename);
        REQUIRE(material.valid());
        for (auto& guid : entities)
        {
            const auto entity = scene.scene.create();
            scene.scene.emplace<scene::transform_component>(entity);
            scene.scene.emplace<scene::mesh_renderer_component>(entity,
                                                                scene::mesh_renderer_component{.material = material});
            editor::ensure_scene_authoring_metadata(scene);
            guid = editor::entity_guid_of(scene, entity);
            scene.asset_bindings.push_back({.entity = guid,
                                            .source_kind = "primitive",
                                            .subresource = "Cube",
                                            .material = {.guid = material_guid,
                                                         .expected_type = registered->type,
                                                         .path_hint = std::string("Content/") + filename}});
        }
        const auto target = editor::find_entity_by_guid(scene, entities[0]);
        const assets::asset_reference reference{.guid = texture_guid, .expected_type = assets::asset_types::texture_2d};
        std::string error;
        for (int stage = 0; stage < 3; ++stage)
        {
            CAPTURE(instance, stage);
            REQUIRE(editor::set_entity_material_parameter(scene, renderer, target, parameter, {},
                                                          stage == 0 ? &reference : nullptr, stage == 2, error, {},
                                                          stage == 1));
            overrides[stage] = editor::entity_material_overrides_json(scene, target);
            if (stage == 0) CHECK(overrides[stage].find(assets::to_string(texture_guid)) != std::string::npos);
            if (stage == 2) CHECK(overrides[stage] == "[]");
            const auto saved = editor::save_scene_document(
                scene, project.root, project.root / ("stage" + std::to_string(stage) + ".arcscene"), &registry.manager);
            INFO(saved.message);
            REQUIRE(saved.succeeded);
        }
    }
    // No registry, scene, renderer, material handles or texture cache survives this boundary.
    for (int stage = 0; stage < 3; ++stage)
    {
        CAPTURE(instance, stage);
        registry_fixture registry(project.root);
        render::renderer renderer;
        editor::editor_scene_state scene;
        scene.project_root = project.root;
        scene.asset_registry = &registry.manager;
        const auto loaded = editor::load_scene_document(scene, renderer, project.root,
                                                        project.root / ("stage" + std::to_string(stage) + ".arcscene"),
                                                        &registry.manager);
        INFO(loaded.message);
        for (const auto& diagnostic : loaded.diagnostics)
            INFO(diagnostic);
        REQUIRE(loaded.succeeded);
        editor::synchronize_procedural_and_material_instances(scene, renderer);
        for (std::size_t index = 0; index < entities.size(); ++index)
        {
            const auto entity = editor::find_entity_by_guid(scene, entities[index]);
            REQUIRE(scene.scene.alive(entity));
            const auto* binding = editor::find_asset_binding(scene, entities[index]);
            REQUIRE(binding);
            CHECK(binding->material.guid == material_guid);
            const auto* mesh = scene.scene.try_get<scene::mesh_renderer_component>(entity);
            REQUIRE(mesh);
            const auto* material = renderer.material_data_for(mesh->material);
            REQUIRE(material);
            REQUIRE(material->runtime_program);
            const bool active = index == 1 ? instance : (stage == 0 || (stage == 2 && instance));
            CHECK(material->render_path ==
                  (active ? render::material_render_path::clustered_forward : render::material_render_path::deferred));
            CHECK(material->independent_clear_coat_normal == active);
            CHECK(editor::entity_material_overrides_json(scene, entity) == (index == 0 ? overrides[stage] : "[]"));
            if (active)
            {
                REQUIRE_FALSE(material->runtime_textures.empty());
                CHECK(material->runtime_textures[0].valid());
            }
        }
        const auto packet = renderer.frame_queue().commit(1);
        bool normal_uploaded = false;
        for (const auto& event : packet.events)
            if (const auto* upload = std::get_if<render::texture_upload_event>(&event.payload);
                upload && upload->texture->semantic == render::texture_semantic::normal)
            {
                normal_uploaded = true;
                CHECK(upload->texture->color_space == render::required_color_space(render::texture_semantic::normal));
            }
        if (stage == 0 || instance) CHECK(normal_uploaded);
    }
}
