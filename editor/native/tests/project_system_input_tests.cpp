#include <arc/editor/arc_host.h>
#include <arc/editor/editor_state.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>

namespace
{
std::filesystem::path stage_input_module(const std::filesystem::path& root)
{
    std::filesystem::create_directories(root / "Content");
    std::filesystem::create_directories(root / "Build");
    const auto destination = root / "Build" / std::filesystem::path(ARC_TEST_GAME_MODULE_INPUT).filename();
    std::filesystem::copy_file(ARC_TEST_GAME_MODULE_INPUT, destination,
                               std::filesystem::copy_options::overwrite_existing);
    return destination;
}
} // namespace

TEST_CASE("Play viewport input is sampled by project ECS systems and focus loss clears the next tick")
{
    const auto root = std::filesystem::temp_directory_path() /
                      ("arc-play-input-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto module_path = stage_input_module(root);

    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::make_unique<arc::render::renderer>());
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    REQUIRE(host->open_project({.name = "Play Input",
                                .root = root,
                                .project_guid = "12345678-1234-4234-8234-123456789abc",
                                .engine_version = "0.1.0",
                                .editor_module_id = "fixture.editor",
                                .editor_module_path = module_path},
                               assets)
                .succeeded);

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_key_command{.key = "w", .down = true, .shift = true}).succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_pointer_command{
                              .phase = arc::editor::host_viewport_pointer_phase::move, .x = 12, .y = 34})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::paused);

    REQUIRE(host->execute(arc::editor::host_viewport_pointer_command{
                              .phase = arc::editor::host_viewport_pointer_phase::cancel})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::paused);

    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);
    REQUIRE(host->execute(arc::editor::host_viewport_key_command{.key = "w", .down = true}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}
