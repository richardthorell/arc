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

#include "editor_test_support.h"

using arc::editor::tests::parse_entity_from_response;
using arc::editor::tests::pick_test_backend;

TEST_CASE("prefab authoring creates, instantiates, persists, reverts, and unpacks instances")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-prefab-authoring-test";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root / "assets" / "prefabs", error);
    REQUIRE_FALSE(error);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root / "assets";
    REQUIRE(host->open_project({.name = "Prefab Authoring", .root = root}, assets).succeeded);

    const auto created_source =
        host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube});
    REQUIRE(created_source.succeeded);
    const auto source = parse_entity_from_response(created_source.payload_json);
    REQUIRE(source.valid());
    const auto prefab_path = root / "assets" / "prefabs" / "camera_rig.arcprefab";
    REQUIRE(host->execute(arc::editor::host_create_prefab_command{.entity = source, .path = prefab_path}).succeeded);
    REQUIRE(std::filesystem::is_regular_file(prefab_path));
    REQUIRE(host->selected_entity_snapshot().prefab.has_value());
    REQUIRE(host->selected_entity_snapshot().prefab->prefab_path == "assets/prefabs/camera_rig.arcprefab");

    const auto instantiated =
        host->execute(arc::editor::host_instantiate_prefab_command{.path = "assets/prefabs/camera_rig.arcprefab"});
    REQUIRE(instantiated.succeeded);
    const auto instance_id = parse_entity_from_response(instantiated.payload_json);
    REQUIRE(instance_id.valid());
    REQUIRE(host->selected_entity_snapshot().entity == instance_id);
    REQUIRE(host->selected_entity_snapshot().prefab.has_value());
    REQUIRE(host->selected_entity_snapshot().prefab->source_missing == false);

    REQUIRE(
        host->execute(arc::editor::host_rename_entity_command{.entity = instance_id, .name = "Changed Prefab Instance"})
            .succeeded);
    REQUIRE(host->execute(arc::editor::host_apply_prefab_command{.entity = instance_id}).succeeded);
    REQUIRE(host->execute(arc::editor::host_revert_prefab_command{.entity = instance_id}).succeeded);
    const auto reverted = host->selected_entity_snapshot().entity;
    REQUIRE(reverted.valid());
    REQUIRE(host->selected_entity_snapshot().name == "Changed Prefab Instance");
    REQUIRE(host->selected_entity_snapshot().prefab.has_value());

    REQUIRE(host->execute(arc::editor::host_unpack_prefab_command{.entity = reverted}).succeeded);
    REQUIRE_FALSE(host->selected_entity_snapshot().prefab.has_value());
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    REQUIRE(host->selected_entity_snapshot().prefab.has_value());
    const auto nested_owner = arc::ecs::generate_entity_guid();
    auto& nested_instance = host->scene_state().scene.get<arc::scene::prefab_instance_component>(
        arc::ecs::entity{reverted.index, reverted.generation});
    nested_instance.nested = true;
    nested_instance.nested_owner = nested_owner;

    const auto scene_path = root / "prefab_scene.arcscene";
    REQUIRE(host->execute(arc::editor::host_save_scene_as_command{.path = scene_path}).succeeded);
    REQUIRE(host->execute(arc::editor::host_open_scene_command{.path = scene_path}).succeeded);
    const auto reopened = host->scene_snapshot();
    const auto reopened_instance =
        std::find_if(reopened.entities.begin(), reopened.entities.end(),
                     [](const auto& value) { return value.name == "Changed Prefab Instance"; });
    REQUIRE(reopened_instance != reopened.entities.end());
    const auto& nested_round_trip = host->scene_state().scene.get<arc::scene::prefab_instance_component>(
        arc::ecs::entity{reopened_instance->entity.index, reopened_instance->entity.generation});
    REQUIRE(nested_round_trip.nested);
    REQUIRE(nested_round_trip.nested_owner == nested_owner);

    std::filesystem::remove_all(root, error);
}

TEST_CASE("editor play session renders an isolated scene copy and restores the authoring viewport")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    REQUIRE(host->open_project({.name = "Play Session Isolation", .root = {}}, {}).succeeded);
    host->renderer_service().set_backend(std::make_unique<pick_test_backend>());
    host->poll_events();

    const auto selected_before = host->selected_entity_snapshot();
    REQUIRE(selected_before.entity.valid());
    const auto editor_camera = host->scene_state().camera_entity;
    const auto camera_before =
        std::as_const(host->scene_state().scene).get<arc::scene::transform_component>(editor_camera);

    auto frame = host->request_viewport({.viewport_id = "viewport-1", .frame_index = 1, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    const auto initial_renderables = host->scene_state().last_render.renderable_count;

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->runtime_snapshot().state == arc::editor::host_runtime_state::running);

    auto& authoring = host->scene_state();
    const auto authoring_only = authoring.scene.create();
    authoring.scene.emplace<arc::scene::name_component>(authoring_only, "Authoring Only During Play");
    authoring.scene.emplace<arc::scene::transform_component>(authoring_only);
    arc::scene::mesh_renderer_component mesh;
    mesh.mesh = authoring.default_mesh;
    mesh.material = authoring.default_material;
    authoring.scene.emplace<arc::scene::mesh_renderer_component>(authoring_only, mesh);
    arc::scene::update_world_transforms(authoring.scene);

    REQUIRE(host->execute(arc::editor::host_viewport_camera_input_command{.forward = 1.0f}).succeeded);
    const auto camera_during_play =
        std::as_const(host->scene_state().scene).get<arc::scene::transform_component>(editor_camera);
    CHECK(camera_during_play.position[0] == camera_before.position[0]);
    CHECK(camera_during_play.position[1] == camera_before.position[1]);
    CHECK(camera_during_play.position[2] == camera_before.position[2]);

    frame = host->request_viewport({.viewport_id = "viewport-1", .frame_index = 2, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    CHECK(host->scene_state().last_render.renderable_count == initial_renderables);

    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    const auto tick_before_step = host->runtime_snapshot().tick_id;
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().tick_id == tick_before_step + 1);

    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);
    REQUIRE(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);
    CHECK(host->selected_entity_snapshot().guid == selected_before.guid);

    frame = host->request_viewport({.viewport_id = "viewport-1", .frame_index = 3, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    CHECK(host->scene_state().last_render.renderable_count == initial_renderables + 1);
}

TEST_CASE("project module loader retains executable ECS system registrations")
{
    arc::editor::project_module_loader loader;
    const auto loaded =
        loader.load(ARC_TEST_GAME_MODULE_SYSTEM, "0.1.0", "12345678-1234-4234-8234-123456789abc", "fixture.editor");
    REQUIRE(loaded.succeeded);
    REQUIRE(loader.system_registrations().size() == 1);
    const auto& system = loader.system_registrations().front();
    CHECK(system.stable_id == "fixture.runtime.visibility");
    CHECK(system.phase == arc::project::game_system_phase_v1::gameplay_commands);
    CHECK(system.execute != nullptr);
}

TEST_CASE("project module loader rejects invalid ECS system scheduling metadata")
{
    arc::editor::project_module_loader loader;
    const auto loaded = loader.load(ARC_TEST_GAME_MODULE_INVALID_SYSTEM, "0.1.0",
                                    "12345678-1234-4234-8234-123456789abc", "fixture.editor");
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.message.find("invalid ECS system descriptor") != std::string::npos);
}

TEST_CASE("play sessions execute project ECS systems without mutating the authoring world")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-play-system-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root / "Content");
    std::filesystem::create_directories(root / "Build");
    const auto module_path = root / "Build" / std::filesystem::path(ARC_TEST_GAME_MODULE_SYSTEM).filename();
    std::filesystem::copy_file(ARC_TEST_GAME_MODULE_SYSTEM, module_path,
                               std::filesystem::copy_options::overwrite_existing);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    REQUIRE(host->open_project({.name = "Executable Play Systems",
                                .root = root,
                                .project_guid = "12345678-1234-4234-8234-123456789abc",
                                .engine_version = "0.1.0",
                                .editor_module_id = "fixture.editor",
                                .editor_module_path = module_path},
                               assets)
                .succeeded);
    host->renderer_service().set_backend(std::make_unique<pick_test_backend>());
    host->poll_events();

    auto& authoring = host->scene_state();
    const auto probe = authoring.scene.create();
    authoring.scene.emplace<arc::scene::name_component>(probe, "Runtime System Probe");
    authoring.scene.emplace<arc::scene::transform_component>(probe);
    authoring.scene.emplace<arc::scene::active_component>(probe, true);
    arc::scene::mesh_renderer_component mesh;
    mesh.mesh = authoring.default_mesh;
    mesh.material = authoring.default_material;
    authoring.scene.emplace<arc::scene::mesh_renderer_component>(probe, mesh);
    arc::editor::ensure_scene_authoring_metadata(authoring);
    const auto probe_guid = arc::editor::entity_guid_of(authoring, probe);
    REQUIRE(probe_guid.valid());
    const std::string authored_project_component =
        R"({"runtime_probe":{"typeId":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","version":1,"_arcFieldIds":{"value":"2222222222222222"},"value":1.0}})";
    authoring.unknown_component_records.emplace_back(probe_guid, authored_project_component);
    arc::scene::update_world_transforms(authoring.scene);

    auto frame = host->request_viewport({.viewport_id = "viewport-1", .frame_index = 1, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    const auto authoring_renderables = authoring.last_render.renderable_count;
    REQUIRE(authoring_renderables > 0);
    REQUIRE(authoring.scene.get<arc::scene::mesh_renderer_component>(probe).visible);
    REQUIRE(authoring.scene.get<arc::scene::active_component>(probe).active);

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->runtime_snapshot().world_count == 2);
    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    const auto tick_before = host->runtime_snapshot().tick_id;
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().tick_id == tick_before + 1);

    frame = host->request_viewport({.viewport_id = "viewport-1", .frame_index = 2, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    CHECK(authoring.last_render.renderable_count + 1 == authoring_renderables);
    CHECK(authoring.scene.get<arc::scene::mesh_renderer_component>(probe).visible);
    CHECK(authoring.scene.get<arc::scene::active_component>(probe).active);

    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);
    CHECK(host->runtime_snapshot().world_count == 1);
    frame = host->request_viewport({.viewport_id = "viewport-1", .frame_index = 3, .width = 640, .height = 360});
    REQUIRE(frame.submitted);
    CHECK(authoring.last_render.renderable_count == authoring_renderables);
    CHECK(authoring.scene.get<arc::scene::mesh_renderer_component>(probe).visible);

    const auto authored_record =
        std::find_if(authoring.unknown_component_records.begin(), authoring.unknown_component_records.end(),
                     [&](const auto& record) { return record.first == probe_guid; });
    REQUIRE(authored_record != authoring.unknown_component_records.end());
    CHECK(authored_record->second == authored_project_component);

    // Closing a project must destroy the Play World before unloading module code.
    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->runtime_snapshot().world_count == 2);
    REQUIRE(host->execute(arc::editor::host_close_project_command{}).succeeded);
    CHECK(host->runtime_snapshot().world_count == 1);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}
