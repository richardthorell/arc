#include <arc/project/project.h>
#include "arc_project_test.reflection.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <string>

namespace
{
class temporary_directory
{
public:
    temporary_directory()
    {
        path_ = std::filesystem::temp_directory_path() /
                ("arc-project-tests-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
    }
    ~temporary_directory()
    {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};
} // namespace

TEST_CASE("generated project reflection supplies stable ECS component metadata")
{
    using component = arc::project::tests::reflected_stats;
    const auto& metadata = arc::ecs::component_metadata<component>();
    CHECK(arc::ecs::to_string(metadata.id) == "cb3208e9cd18443693a80cbed1099ccd");
    CHECK(metadata.schema_version == 3);
    REQUIRE(metadata.fields.size() == 2);
    CHECK(metadata.fields[0].id == 0x2b880a80f9e8fd40ull);
    CHECK(arc::ecs::has_flag(metadata.fields[0].flags, arc::ecs::reflected_field_flags::replicated));
    CHECK(metadata.fields[0].minimum == 0.0);
    CHECK(metadata.fields[0].maximum == 100.0);
    CHECK(arc::project::tests::generated::components.front().stable_id ==
          std::string_view{"cb3208e9cd18443693a80cbed1099ccd"});
}

TEST_CASE("version two project descriptors round trip and resolve project local paths")
{
    temporary_directory temporary;
    const auto path = temporary.path() / "Game.arcproject";
    arc::project::project_descriptor descriptor;
    descriptor.guid = "12345678-1234-4234-8234-123456789abc";
    descriptor.name = "Game";
    descriptor.engine_version = "0.1.0";
    descriptor.modules.push_back({.id = "Game.runtime",
                                  .kind = arc::project::module_kind::runtime,
                                  .target = "GameRuntime",
                                  .source_root = "Source/GameRuntime",
                                  .dependencies = {{.kind = arc::project::dependency_kind::engine,
                                                    .id = "ARC.Runtime",
                                                    .version = "0.1.0"}}});
    descriptor.target_platforms.push_back({.id = "windows-x64-vulkan"});

    REQUIRE(arc::project::save_descriptor(path, descriptor));
    const auto loaded = arc::project::load_descriptor(path);
    REQUIRE(loaded);
    CHECK(loaded.value().name == "Game");
    CHECK(loaded.value().modules.front().target == "GameRuntime");

    const auto context = arc::project::resolve_context(path, loaded.value());
    REQUIRE(context);
    CHECK(context.value().asset_cache_root == temporary.path() / "Intermediate" / "Cache");
    CHECK(context.value().recovery_root == temporary.path() / "Saved" / "Recovery");
}

TEST_CASE("project validation rejects missing typed module dependencies")
{
    temporary_directory temporary;
    arc::project::project_descriptor descriptor;
    descriptor.guid = "12345678-1234-4234-8234-123456789abc";
    descriptor.name = "Game";
    descriptor.engine_version = "0.1.0";
    descriptor.modules.push_back({.id = "Game.editor",
                                  .kind = arc::project::module_kind::editor,
                                  .target = "GameEditor",
                                  .source_root = "Source/GameEditor",
                                  .dependencies = {{.kind = arc::project::dependency_kind::project,
                                                    .id = "Game.runtime"}}});
    const auto result = arc::project::validate_descriptor(temporary.path() / "Game.arcproject", descriptor);
    REQUIRE_FALSE(result);
    CHECK(result.error().code == arc::project::project_error_code::missing_module);
}

TEST_CASE("engine installation registry keeps manifests as its authority")
{
    temporary_directory temporary;
    const auto manifest = temporary.path() / "sdk" / "arc-installation.json";
    std::filesystem::create_directories(manifest.parent_path());
    std::ofstream(manifest) << R"({
      "format":"arc-installation","formatVersion":1,"installationId":"arc-0.1.0-test",
      "engineVersion":"0.1.0","editor":"bin/arc-editor","sdk":".","cooker":"bin/arc-cook",
      "projectTool":"bin/arc-project","supportedPlatforms":["linux-x64-headless"],
      "configurations":["Debug","RelWithDebInfo","Shipping"],"plugins":[],"templates":[],
      "toolchain":{"compiler":"auto","generator":"auto","architecture":"x86_64","cppStandard":20}
    })";
    const auto registry = temporary.path() / "installations.json";
    REQUIRE(arc::project::register_installation(registry, manifest));
    const auto installations = arc::project::discover_installations(registry);
    REQUIRE(installations);
    REQUIRE(installations.value().size() == 1);
    CHECK(installations.value().front().engine_version == "0.1.0");
    REQUIRE(arc::project::unregister_installation(registry, "arc-0.1.0-test"));
    CHECK(arc::project::discover_installations(registry).value().empty());
}

TEST_CASE("all installed project templates generate complete repositories")
{
    temporary_directory temporary;
    const auto templates = arc::project::discover_templates(ARC_TEST_TEMPLATE_ROOT);
    REQUIRE(templates);
    REQUIRE(templates.value().size() == 4);
    for (const auto& project_template : templates.value())
    {
        const auto destination = temporary.path() / project_template.id;
        INFO(project_template.id);
        REQUIRE(arc::project::create_project({.name = "Generated Game",
                                              .destination = destination,
                                              .template_id = project_template.id,
                                              .templates_root = ARC_TEST_TEMPLATE_ROOT,
                                              .engine_version = "0.1.0"}));
        for (const auto* directory : {"Source", "Content", "Config", "Plugins", "Saved", "Intermediate", "Build"})
            CHECK(std::filesystem::is_directory(destination / directory));
        const auto descriptor = arc::project::load_descriptor(destination / "GeneratedGame.arcproject");
        REQUIRE(descriptor);
        CHECK(arc::project::validate_descriptor(destination / "GeneratedGame.arcproject", descriptor.value(),
                                                 {.engine_version = "0.1.0", .require_exact_engine = true,
                                                  .require_paths = true}));
        if (std::any_of(descriptor.value().modules.begin(), descriptor.value().modules.end(),
                        [](const auto& module) { return module.kind == arc::project::module_kind::editor; }))
        {
            const auto component_header = destination / "Source" / "GeneratedGameRuntime" / "Components.h";
            REQUIRE(std::filesystem::is_regular_file(component_header));
            std::ifstream component_input(component_header);
            const std::string component_source((std::istreambuf_iterator<char>(component_input)),
                                               std::istreambuf_iterator<char>());
            CHECK(component_source.find("ARC_COMPONENT(\"") != std::string::npos);
            CHECK(component_source.find("ARC_PROPERTY(\"") != std::string::npos);
            std::ifstream cmake_input(destination / "CMakeLists.txt");
            const std::string cmake_source((std::istreambuf_iterator<char>(cmake_input)),
                                           std::istreambuf_iterator<char>());
            CHECK(cmake_source.find("arc_generate_reflection") != std::string::npos);
        }
    }
}

TEST_CASE("version one upgrades preserve custom content roots and startup scenes")
{
    temporary_directory temporary;
    std::filesystem::create_directories(temporary.path() / "assets" / "scenes");
    const auto scene = temporary.path() / "assets" / "scenes" / "Start.arcscene";
    std::ofstream(scene) << "{}";
    std::ofstream(scene.string() + ".arcmeta")
        << R"({"guid":"12345678-1234-4234-8234-123456789abc"})";
    const auto descriptor_path = temporary.path() / "Legacy.arcproject";
    std::ofstream(descriptor_path)
        << R"({"format":"arc-project","formatVersion":1,"guid":"12345678-1234-4234-8234-123456789abd","name":"Legacy","engineVersion":"0.0.1","assetRoots":["assets"],"startupScenes":["assets/scenes/Start.arcscene"],"modules":[],"extensions":[],"settings":{}})";
    REQUIRE(arc::project::upgrade_descriptor(descriptor_path, "0.1.0"));
    const auto upgraded = arc::project::load_descriptor(descriptor_path);
    REQUIRE(upgraded);
    CHECK(upgraded.value().asset_roots == std::vector<std::filesystem::path>{"assets"});
    REQUIRE(upgraded.value().default_scene);
    CHECK(upgraded.value().default_scene->guid == "12345678-1234-4234-8234-123456789abc");
    CHECK(std::filesystem::is_regular_file(descriptor_path.string() + ".v1.bak"));
}

TEST_CASE("project validation rejects a default scene whose asset identity does not match")
{
    temporary_directory temporary;
    const auto destination = temporary.path() / "scene-identity";
    REQUIRE(arc::project::create_project({.name = "Scene Identity",
                                          .destination = destination,
                                          .template_id = "blank-3d",
                                          .templates_root = ARC_TEST_TEMPLATE_ROOT,
                                          .engine_version = "0.1.0"}));
    const auto descriptor_path = destination / "SceneIdentity.arcproject";
    auto descriptor = arc::project::load_descriptor(descriptor_path);
    REQUIRE(descriptor);
    REQUIRE(descriptor.value().default_scene);
    descriptor.value().default_scene->guid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
    const auto validation = arc::project::validate_descriptor(
        descriptor_path, descriptor.value(), {.require_paths = true});
    REQUIRE_FALSE(validation);
    CHECK(validation.error().code == arc::project::project_error_code::invalid_scene);
}
