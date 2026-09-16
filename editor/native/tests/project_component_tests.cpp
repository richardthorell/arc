#include <arc/editor/arc_host.h>
#include <arc/editor/editor_console.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/material_asset.h>
#include <arc/editor/material_library.h>
#include <arc/editor/material_preview.h>
#include <arc/editor/scene_document.h>
#include <arc/editor/world_environment_host.h>
#include <arc/project/project.h>
#include <arc/render/primitives.h>
#include "../src/project_module_loader.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <variant>

TEST_CASE("project module generations expose stable schemas and classify compatible reloads")
{
    arc::editor::project_module_loader loader;
    const auto initial =
        loader.load(ARC_TEST_GAME_MODULE_V1, "0.1.0", "12345678-1234-4234-8234-123456789abc", "fixture.editor");
    REQUIRE(initial.succeeded);
    CHECK(initial.classification == arc::editor::module_reload_classification::initial_load);
    REQUIRE(loader.component_schemas().size() == 1);
    CHECK(loader.component_schemas().front().canonical_name == "old_component");
    REQUIRE(loader.registrations().size() == 1);
    CHECK(loader.registrations().front().stable_id == "fixture.echo");
    CHECK(loader.registrations().front().kind == arc::project::game_registration_kind_v1::console_command);

    const auto reloaded =
        loader.reload(ARC_TEST_GAME_MODULE_V2, "0.1.0", "12345678-1234-4234-8234-123456789abc", "fixture.editor");
    REQUIRE(reloaded.succeeded);
    CHECK(reloaded.classification == arc::editor::module_reload_classification::safe_hot_reload);
    CHECK(reloaded.generation == 2);
    CHECK(loader.component_schemas().front().canonical_name == "renamed_component");
    CHECK(loader.component_schemas().front().fields.front().name == "renamed_value");

    const auto play_restart =
        loader.reload(ARC_TEST_GAME_MODULE_V3, "0.1.0", "12345678-1234-4234-8234-123456789abc", "fixture.editor");
    REQUIRE(play_restart.succeeded);
    CHECK(play_restart.classification == arc::editor::module_reload_classification::play_session_restart_required);

    const auto host_restart =
        loader.reload(ARC_TEST_GAME_MODULE_V4, "0.1.0", "12345678-1234-4234-8234-123456789abc", "fixture.editor");
    CHECK_FALSE(host_restart.succeeded);
    CHECK(host_restart.classification == arc::editor::module_reload_classification::native_host_restart_required);
    CHECK(loader.generation() == 3);

    const auto rejected =
        loader.reload(ARC_TEST_GAME_MODULE_REJECTED, "0.1.0", "12345678-1234-4234-8234-123456789abc", "fixture.editor");
    CHECK_FALSE(rejected.succeeded);
    CHECK(rejected.message.find("last-good generation restored") != std::string::npos);
    CHECK(loader.loaded());
    CHECK(loader.generation() == 3);
}

TEST_CASE("project components add edit persist and migrate through the native host")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-project-component-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root / "Content");
    std::filesystem::create_directories(root / "Build");
    const auto module_v1 = root / "Build" / std::filesystem::path(ARC_TEST_GAME_MODULE_V1).filename();
    const auto module_v2 = root / "Build" / std::filesystem::path(ARC_TEST_GAME_MODULE_V2).filename();
    std::filesystem::copy_file(ARC_TEST_GAME_MODULE_V1, module_v1, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(ARC_TEST_GAME_MODULE_V2, module_v2, std::filesystem::copy_options::overwrite_existing);
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    REQUIRE(host->open_project({.name = "Project Component",
                                .root = root,
                                .project_guid = "12345678-1234-4234-8234-123456789abc",
                                .engine_version = "0.1.0",
                                .editor_module_id = "fixture.editor",
                                .editor_module_path = module_v1},
                               assets)
                .succeeded);
    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .request_id = 1,
                    .payload =
                        arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty}})
                .succeeded);
    const auto entity = host->selected_entity_snapshot().entity;
    const auto entity_guid = host->selected_entity_snapshot().guid;
    REQUIRE(entity.valid());
    REQUIRE(host->execute(
                    arc::editor::host_command_envelope{.request_id = 2,
                                                       .payload =
                                                           arc::editor::host_component_operation_command{
                                                               .operation = arc::editor::host_component_operation::add,
                                                               .component = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_command_envelope{
                              .request_id = 3,
                              .payload = arc::editor::host_patch_project_component_command{.component = "old_component",
                                                                                           .field = "old_value",
                                                                                           .value_json = "4.5"}})
                .succeeded);
    REQUIRE(host->selected_entity_snapshot().project_components.size() == 1);
    CHECK(host->selected_entity_snapshot().project_components.front().values_json.find("4.5") != std::string::npos);

    const auto scene_path = root / "Content" / "ProjectComponent.arcscene";
    REQUIRE(host->execute(arc::editor::host_save_scene_as_command{.path = scene_path}).succeeded);
    const auto reloaded =
        host->execute(arc::editor::host_command_envelope{.request_id = 4,
                                                         .payload = arc::editor::host_reload_project_module_command{
                                                             .path = module_v2,
                                                             .engine_version = "0.1.0",
                                                             .project_guid = "12345678-1234-4234-8234-123456789abc",
                                                             .module_id = "fixture.editor"}});
    REQUIRE(reloaded.succeeded);
    const auto migrated = host->selected_entity_snapshot();
    REQUIRE(migrated.project_components.size() == 1);
    CHECK(migrated.project_components.front().canonical_name == "renamed_component");
    CHECK(migrated.project_components.front().values_json.find("\"renamed_value\":4.5") != std::string::npos);
    REQUIRE(host->execute(arc::editor::host_save_scene_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_open_scene_command{.path = scene_path}).succeeded);
    const auto scene = host->scene_snapshot();
    const auto persisted = std::find_if(scene.entities.begin(), scene.entities.end(),
                                        [&](const auto& candidate) { return candidate.guid == entity_guid; });
    REQUIRE(persisted != scene.entities.end());
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = persisted->entity}).succeeded);
    REQUIRE(host->selected_entity_snapshot().project_components.size() == 1);
    CHECK(host->selected_entity_snapshot().project_components.front().values_json.find("4.5") != std::string::npos);

    std::ifstream input(scene_path, std::ios::binary);
    const std::string document((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    CHECK(document.find("renamed_component") != std::string::npos);
    CHECK(document.find("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") != std::string::npos);
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("native host keeps an editor camera throughout the project lifecycle")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    const auto move_camera = [&]
    {
        return host->execute(arc::editor::host_command_envelope{
            .request_id = 1, .payload = arc::editor::host_viewport_camera_input_command{.forward = 1.0f}});
    };
    const auto require_default_scene = [&]
    {
        const auto snapshot = host->scene_snapshot();
        CHECK(std::any_of(snapshot.entities.begin(), snapshot.entities.end(),
                          [](const auto& entity) { return entity.name == "Main Camera"; }));
        CHECK(std::any_of(snapshot.entities.begin(), snapshot.entities.end(),
                          [](const auto& entity) { return entity.name == "Default Sky"; }));
        CHECK(std::any_of(snapshot.entities.begin(), snapshot.entities.end(),
                          [](const auto& entity) { return entity.name == "Floor"; }));
    };

    require_default_scene();
    CHECK(move_camera().succeeded);
    REQUIRE(host->open_project({.name = "Camera Lifecycle"}, {}).succeeded);
    require_default_scene();
    CHECK(move_camera().succeeded);
    REQUIRE(host->execute(arc::editor::host_close_project_command{}).succeeded);
    require_default_scene();
    CHECK(move_camera().succeeded);
}
