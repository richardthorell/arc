#include <arc/editor/arc_host.h>
#include "../src/project_module_loader.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

namespace
{
constexpr const char* project_guid = "12345678-1234-4234-8234-123456789abc";

std::filesystem::path stage_module(const std::filesystem::path& root, const char* source)
{
    std::filesystem::create_directories(root / "Content");
    std::filesystem::create_directories(root / "Build");
    const auto destination = root / "Build" / std::filesystem::path(source).filename();
    std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing);
    return destination;
}
} // namespace

TEST_CASE("project module loader retains one play lifecycle registration")
{
    arc::editor::project_module_loader loader;
    const auto loaded = loader.load(ARC_TEST_GAME_MODULE_LIFECYCLE, "0.1.0", project_guid, "fixture.editor");
    REQUIRE(loaded.succeeded);
    REQUIRE(loader.play_lifecycle().has_value());
    CHECK(loader.play_lifecycle()->stable_id == "fixture.runtime.play-lifecycle");
    CHECK(loader.play_lifecycle()->begin_play != nullptr);
    CHECK(loader.play_lifecycle()->end_play != nullptr);
}

TEST_CASE("project module loader rejects incomplete play lifecycle registrations")
{
    arc::editor::project_module_loader loader;
    const auto loaded = loader.load(ARC_TEST_GAME_MODULE_INVALID_LIFECYCLE, "0.1.0", project_guid, "fixture.editor");
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.message.find("invalid play lifecycle descriptor") != std::string::npos);
}

TEST_CASE("play sessions pair BeginPlay and EndPlay across stop and restart")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-play-lifecycle-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto module_path = stage_module(root, ARC_TEST_GAME_MODULE_LIFECYCLE);

    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::make_unique<arc::render::renderer>());
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    REQUIRE(host->open_project({.name = "Play Lifecycle",
                                .root = root,
                                .project_guid = project_guid,
                                .engine_version = "0.1.0",
                                .editor_module_id = "fixture.editor",
                                .editor_module_path = module_path},
                               assets)
                .succeeded);

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    CHECK(host->runtime_snapshot().world_count == 2);
    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::paused);

    // Pause/resume belongs to the same Play session and must not invoke BeginPlay again.
    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::paused);

    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);
    CHECK(host->runtime_snapshot().world_count == 1);

    // The fixture rejects BeginPlay while its prior session is still active, so this
    // second Play proves Stop delivered the matching EndPlay before creating a new session.
    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    CHECK(host->runtime_snapshot().world_count == 2);
    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::paused);
    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);
    CHECK(host->runtime_snapshot().world_count == 1);

    // Project close uses the same Play World destruction path and must run EndPlay
    // before unloading the project module generation.
    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_close_project_command{}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);
    CHECK(host->runtime_snapshot().world_count == 1);

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}
