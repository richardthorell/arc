#include <arc/editor/arc_host.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/material_library.h>
#include <arc/render/renderer.h>
#include <arc/render_tools/render_tools.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

TEST_CASE("built-in material routing is resolved by native material realization")
{
    const auto root = std::filesystem::path(ARC_SOURCE_ROOT) / "assets";
    for (const auto* file : {"standard_lit.arcmat", "unlit.arcmat", "transmission.arcmat", "subsurface.arcmat"})
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
