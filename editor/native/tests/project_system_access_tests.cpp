#include <arc/editor/arc_host.h>
#include <arc/editor/scene_document.h>
#include "../src/project_module_loader.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>

namespace
{
constexpr const char* project_guid = "12345678-1234-4234-8234-123456789abc";
constexpr const char* component_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

std::filesystem::path stage_module(const std::filesystem::path& root, const char* source)
{
    std::filesystem::create_directories(root / "Content");
    std::filesystem::create_directories(root / "Build");
    const auto destination = root / "Build" / std::filesystem::path(source).filename();
    std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing);
    return destination;
}

void add_runtime_component(arc::editor::editor_scene_state& scene)
{
    const auto entity = scene.scene.create();
    scene.scene.emplace<arc::scene::name_component>(entity, "Runtime Component Probe");
    arc::editor::ensure_scene_authoring_metadata(scene);
    const auto guid = arc::editor::entity_guid_of(scene, entity);
    REQUIRE(guid.valid());
    scene.unknown_component_records.emplace_back(
        guid, R"({"runtime_probe":{"typeId":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","version":1,"value":1.0}})");
}
} // namespace

TEST_CASE("project ECS systems retain explicit project component access declarations")
{
    arc::editor::project_module_loader loader;
    const auto loaded = loader.load(ARC_TEST_GAME_MODULE_ACCESS, "0.1.0", project_guid, "fixture.editor");
    REQUIRE(loaded.succeeded);
    REQUIRE(loader.system_registrations().size() == 1);
    const auto& system = loader.system_registrations().front();
    CHECK_FALSE(system.unrestricted_native_world_access);
    REQUIRE(system.component_accesses.size() == 2);
    CHECK(system.component_accesses[0].component_id == component_id);
    CHECK(system.component_accesses[0].mode == arc::project::game_system_component_access_mode_v1::write);
    CHECK(system.component_accesses[1].component_id == "cccccccccccccccccccccccccccccccc");
    CHECK(system.component_accesses[1].mode == arc::project::game_system_component_access_mode_v1::read);
}

TEST_CASE("project module loader rejects unknown project component access declarations")
{
    arc::editor::project_module_loader loader;
    const auto loaded = loader.load(ARC_TEST_GAME_MODULE_INVALID_ACCESS, "0.1.0", project_guid, "fixture.editor");
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.message.find("unknown project component access") != std::string::npos);
}

TEST_CASE("restricted project ECS systems execute through declared component access without native world access")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-play-access-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto module_path = stage_module(root, ARC_TEST_GAME_MODULE_ACCESS);

    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::make_unique<arc::render::renderer>());
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    REQUIRE(host->open_project({.name = "Declared Component Access",
                                .root = root,
                                .project_guid = project_guid,
                                .engine_version = "0.1.0",
                                .editor_module_id = "fixture.editor",
                                .editor_module_path = module_path},
                               assets)
                .succeeded);
    add_runtime_component(host->scene_state());
    const auto authored = host->scene_state().unknown_component_records.back().second;

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::paused);
    CHECK(host->scene_state().unknown_component_records.back().second == authored);
    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("project component bridge faults systems that write through read-only declarations")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-play-access-violation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto module_path = stage_module(root, ARC_TEST_GAME_MODULE_ACCESS_VIOLATION);

    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::make_unique<arc::render::renderer>());
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    REQUIRE(host->open_project({.name = "Component Access Violation",
                                .root = root,
                                .project_guid = project_guid,
                                .engine_version = "0.1.0",
                                .editor_module_id = "fixture.editor",
                                .editor_module_path = module_path},
                               assets)
                .succeeded);
    add_runtime_component(host->scene_state());

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::faulted);
    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}
