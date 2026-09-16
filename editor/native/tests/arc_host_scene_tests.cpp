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

using arc::editor::tests::pick_test_backend;

TEST_CASE("arc host executes scene commands and exposes snapshots")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    arc::editor::editor_asset_state assets;
    const auto opened =
        host->open_project({.name = "Host Test", .root = std::filesystem::temp_directory_path()}, assets);
    REQUIRE(opened.succeeded);

    const auto created = host->execute(arc::editor::host_command_envelope{
        .request_id = 1,
        .payload = arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube}});
    REQUIRE(created.succeeded);
    const auto created_entity = host->selected_entity_snapshot().entity;
    REQUIRE(created_entity.valid());

    const auto renamed = host->execute(arc::editor::host_command_envelope{
        .request_id = 2,
        .payload = arc::editor::host_rename_entity_command{.entity = created_entity, .name = "Command Cube"}});
    REQUIRE(renamed.succeeded);

    const auto selected = host->selected_entity_snapshot();
    REQUIRE(selected.entity == created_entity);
    REQUIRE(selected.name == "Command Cube");
    REQUIRE(selected.transform.has_value());
    REQUIRE(std::any_of(selected.components.begin(), selected.components.end(), [](const auto& component)
                        { return component.kind == arc::editor::host_component_kind::transform; }));

    auto transform = *selected.transform;
    transform.position = {2.0f, 3.0f, 4.0f};
    REQUIRE(host->execute(arc::editor::host_command_envelope{
                              .request_id = 3,
                              .payload = arc::editor::host_set_transform_command{.entity = created_entity,
                                                                                 .transform = transform}})
                .succeeded);
    REQUIRE(host->selected_entity_snapshot().transform->position.x == Catch::Approx(2.0f));

    const auto snapshot = host->scene_snapshot();
    REQUIRE(
        std::any_of(snapshot.entities.begin(), snapshot.entities.end(), [&](const auto& entity)
                    { return entity.entity == created_entity && entity.name == "Command Cube" && entity.selected; }));

    REQUIRE(host->query({.request_id = 4, .payload = arc::editor::host_scene_hierarchy_query{}}).succeeded);
    REQUIRE(host->query({.request_id = 5, .payload = arc::editor::host_project_assets_query{}}).succeeded);
    const auto diagnostics = host->query({.request_id = 6, .payload = arc::editor::host_gateway_diagnostics_query{}});
    REQUIRE(diagnostics.succeeded);
    CHECK(diagnostics.payload_json.find("\"textureStreaming\"") != std::string::npos);
    CHECK(diagnostics.payload_json.find("\"resources\":[") != std::string::npos);
    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .request_id = 7, .payload = arc::editor::host_delete_entity_command{.entity = created_entity}})
                .succeeded);
    REQUIRE_FALSE(host->selected_entity_snapshot().entity.valid());

    const auto events = host->poll_events();
    REQUIRE(std::any_of(events.begin(), events.end(), [](const auto& event)
                        { return event.event_type == arc::editor::host_event_type::entity_created; }));
    REQUIRE(std::any_of(events.begin(), events.end(), [](const auto& event)
                        { return event.event_type == arc::editor::host_event_type::entity_deleted; }));
}

TEST_CASE("workspace documents component operations and read only projects are host authoritative")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-workspace-contract-test";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root, error);
    REQUIRE_FALSE(error);

    {
        arc::editor::arc_host_manager manager;
        auto host = manager.acquire(std::make_unique<arc::render::renderer>());
        REQUIRE(host->open_project({.name = "Workspace", .root = root}, {}).succeeded);
        REQUIRE(
            host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);

        REQUIRE(host->execute(arc::editor::host_component_operation_command{
                                  .operation = arc::editor::host_component_operation::add, .component = "camera"})
                    .succeeded);
        REQUIRE(host->selected_entity_snapshot().camera.has_value());
        REQUIRE(host->execute(arc::editor::host_component_operation_command{
                                  .operation = arc::editor::host_component_operation::reset, .component = "camera"})
                    .succeeded);
        REQUIRE(host->execute(arc::editor::host_component_operation_command{
                                  .operation = arc::editor::host_component_operation::remove, .component = "camera"})
                    .succeeded);
        REQUIRE_FALSE(host->selected_entity_snapshot().camera.has_value());
        REQUIRE_FALSE(
            host->execute(arc::editor::host_component_operation_command{
                              .operation = arc::editor::host_component_operation::remove, .component = "transform"})
                .succeeded);

        const auto workspace = host->query({.request_id = 9, .payload = arc::editor::host_workspace_documents_query{}});
        REQUIRE(workspace.succeeded);
        REQUIRE(workspace.payload_json.find("\"documents\":[{") != std::string::npos);
        REQUIRE(workspace.payload_json.find("\"dirty\":true") != std::string::npos);

        const auto recovery_path = root / "recovery" / "scene.arcscene";
        REQUIRE(host->execute(arc::editor::host_autosave_scene_command{.path = recovery_path}).succeeded);
        REQUIRE(std::filesystem::is_regular_file(recovery_path));
        REQUIRE(host->execute(arc::editor::host_open_recovery_scene_command{
                                  .path = recovery_path, .original_path = root / "assets" / "scene.arcscene"})
                    .succeeded);
        REQUIRE(host->scene_snapshot().dirty);
    }

    {
        arc::editor::arc_host_manager manager;
        auto host = manager.acquire(std::make_unique<arc::render::renderer>());
        REQUIRE(host->open_project({.name = "Read Only", .root = root, .read_only = true}, {}).succeeded);
        REQUIRE_FALSE(
            host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube})
                .succeeded);
        REQUIRE_FALSE(host->execute(arc::editor::host_reload_project_module_command{
                                        .path = root / "Build" / "ReadOnlyEditor.dll",
                                        .engine_version = "0.1.0",
                                        .project_guid = "12345678-1234-4234-8234-123456789abc",
                                        .module_id = "read-only.editor"})
                          .succeeded);
        REQUIRE(host->execute(arc::editor::host_viewport_camera_input_command{.forward = 1.0f}).succeeded);
    }

    std::filesystem::remove_all(root, error);
}

TEST_CASE("prefab property override revert is revisioned and undoable")
{
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::make_unique<arc::render::renderer>());
    REQUIRE(host->open_project({.name = "Prefab Override", .root = {}}, {}).succeeded);
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);

    const auto entity = host->selected_entity_snapshot().entity;
    const auto source = arc::ecs::generate_entity_guid();
    const auto component = arc::ecs::component_type<arc::scene::transform_component>();
    arc::ecs::prefab_instance_component instance;
    instance.prefab_guid = arc::ecs::generate_entity_guid();
    REQUIRE(arc::ecs::set_prefab_override(instance, {.key = {.source_entity = source,
                                                             .component = component,
                                                             .field = 1,
                                                             .kind = arc::ecs::prefab_override_kind::field}}));
    host->scene_state().scene.emplace<arc::ecs::prefab_instance_component>(
        arc::ecs::entity{entity.index, entity.generation}, std::move(instance));

    const auto reverted =
        host->execute(arc::editor::host_revert_prefab_override_command{.entity = entity,
                                                                       .source_entity = arc::ecs::to_string(source),
                                                                       .component_id = arc::ecs::to_string(component),
                                                                       .field_id = 1,
                                                                       .kind = "field"});
    REQUIRE(reverted.succeeded);
    REQUIRE(host->selected_entity_snapshot().prefab->override_count == 0);
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    REQUIRE(host->selected_entity_snapshot().prefab->override_count == 1);
}

TEST_CASE("multi selection property edits are atomic and apply through the host")
{
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::make_unique<arc::render::renderer>());
    REQUIRE(host->open_project({.name = "Multi Edit", .root = {}}, {}).succeeded);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);
    const auto first = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_component_operation_command{
                              .operation = arc::editor::host_component_operation::add, .component = "camera"})
                .succeeded);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);
    const auto second = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_component_operation_command{
                              .operation = arc::editor::host_component_operation::add, .component = "camera"})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = first, .additive = true, .toggle = false})
                .succeeded);
    REQUIRE(host->selected_entity_snapshot().selection_count == 2);

    arc::editor::host_camera_snapshot camera;
    camera.fov_y_degrees = 72.0f;
    REQUIRE(host->execute(
                    arc::editor::host_set_camera_command{.entity = first, .camera = camera, .apply_to_selection = true})
                .succeeded);
    REQUIRE(host->entity_snapshot(first).camera->fov_y_degrees == Catch::Approx(72.0f));
    REQUIRE(host->entity_snapshot(second).camera->fov_y_degrees == Catch::Approx(72.0f));

    arc::editor::host_transform transform;
    transform.position = {3.0f, 4.0f, 5.0f};
    REQUIRE(host->execute(arc::editor::host_set_transform_command{
                              .entity = first, .transform = transform, .apply_to_selection = true})
                .succeeded);
    REQUIRE(host->entity_snapshot(first).transform->position.x == Catch::Approx(3.0f));
    REQUIRE(host->entity_snapshot(second).transform->position.z == Catch::Approx(5.0f));

    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = second}).succeeded);
    REQUIRE(host->execute(arc::editor::host_component_operation_command{
                              .operation = arc::editor::host_component_operation::remove, .component = "camera"})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = first, .additive = true, .toggle = false})
                .succeeded);
    camera.fov_y_degrees = 80.0f;
    REQUIRE_FALSE(host->execute(arc::editor::host_set_camera_command{
                                    .entity = first, .camera = camera, .apply_to_selection = true})
                      .succeeded);
    REQUIRE(host->entity_snapshot(first).camera->fov_y_degrees == Catch::Approx(72.0f));
    REQUIRE_FALSE(host->entity_snapshot(second).camera.has_value());
}

TEST_CASE("arc host hierarchy and history preserve subtrees and group edit transactions")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({.name = "Authoring Test", .root = {}}, assets).succeeded);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube})
                .succeeded);
    const auto parent = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::sphere})
                .succeeded);
    const auto child = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_reparent_entity_command{
                              .entity = child, .parent = parent, .preserve_world = true})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty,
                                                                  .parent = parent})
                .succeeded);
    const auto empty_child = host->selected_entity_snapshot().entity;
    REQUIRE(host->selected_entity_snapshot().name == "Entity");

    const auto hierarchy = host->scene_snapshot();
    const auto parent_record = std::find_if(hierarchy.entities.begin(), hierarchy.entities.end(),
                                            [parent](const auto& value) { return value.entity == parent; });
    const auto child_record = std::find_if(hierarchy.entities.begin(), hierarchy.entities.end(),
                                           [child](const auto& value) { return value.entity == child; });
    REQUIRE(parent_record != hierarchy.entities.end());
    REQUIRE(child_record != hierarchy.entities.end());
    REQUIRE(child_record->parent_guid == parent_record->guid);
    const auto empty_record = std::find_if(hierarchy.entities.begin(), hierarchy.entities.end(),
                                           [empty_child](const auto& value) { return value.entity == empty_child; });
    REQUIRE(empty_record != hierarchy.entities.end());
    REQUIRE(empty_record->parent_guid == parent_record->guid);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);
    const auto first_root = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);
    const auto second_root = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_reorder_entity_command{.entity = second_root, .before_sibling = first_root})
                .succeeded);
    const auto reordered = host->scene_snapshot();
    const auto first_root_record = std::find_if(reordered.entities.begin(), reordered.entities.end(),
                                                [first_root](const auto& value) { return value.entity == first_root; });
    const auto second_root_record =
        std::find_if(reordered.entities.begin(), reordered.entities.end(),
                     [second_root](const auto& value) { return value.entity == second_root; });
    REQUIRE(first_root_record != reordered.entities.end());
    REQUIRE(second_root_record != reordered.entities.end());
    REQUIRE(second_root_record->parent_guid.empty());
    REQUIRE(second_root_record->sibling_order < first_root_record->sibling_order);

    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = parent}).succeeded);
    REQUIRE(host->execute(arc::editor::host_duplicate_entity_command{.entity = parent}).succeeded);
    const auto duplicate = host->selected_entity_snapshot().entity;
    REQUIRE(duplicate != parent);
    REQUIRE(host->scene_snapshot().entities.size() == reordered.entities.size() + 3);
    REQUIRE(host->execute(arc::editor::host_delete_entity_command{.entity = parent}).succeeded);
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    const auto restored_hierarchy = host->scene_snapshot();
    REQUIRE(std::any_of(restored_hierarchy.entities.begin(), restored_hierarchy.entities.end(),
                        [parent](const auto& value) { return value.entity == parent; }));

    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = duplicate}).succeeded);
    const auto original = *host->selected_entity_snapshot().transform;
    auto preview = original;
    preview.position.x += 1.0f;
    REQUIRE(
        host->execute(arc::editor::host_command_envelope{
                          .payload = arc::editor::host_set_transform_command{.entity = duplicate, .transform = preview},
                          .edit = arc::editor::host_edit_transaction{.id = 7,
                                                                     .phase = arc::editor::host_edit_phase::begin,
                                                                     .label = "Gizmo Drag"}})
            .succeeded);
    preview.position.x += 1.0f;
    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .payload = arc::editor::host_set_transform_command{.entity = duplicate, .transform = preview},
                    .edit = arc::editor::host_edit_transaction{.id = 7, .phase = arc::editor::host_edit_phase::update}})
                .succeeded);
    preview.position.x += 1.0f;
    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .payload = arc::editor::host_set_transform_command{.entity = duplicate, .transform = preview},
                    .edit = arc::editor::host_edit_transaction{.id = 7, .phase = arc::editor::host_edit_phase::commit}})
                .succeeded);
    REQUIRE(host->scene_snapshot().undo_label == "Gizmo Drag");
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    REQUIRE(host->selected_entity_snapshot().transform->position.x == Catch::Approx(original.position.x));
    REQUIRE(host->execute(arc::editor::host_history_redo_command{}).succeeded);
    REQUIRE(host->selected_entity_snapshot().transform->position.x == Catch::Approx(preview.position.x));

    auto cancelled = preview;
    cancelled.position.y += 8.0f;
    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .payload = arc::editor::host_set_transform_command{.entity = duplicate, .transform = cancelled},
                    .edit = arc::editor::host_edit_transaction{.id = 8,
                                                               .phase = arc::editor::host_edit_phase::begin,
                                                               .label = "Cancelled Drag"}})
                .succeeded);
    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .payload = arc::editor::host_set_transform_command{.entity = duplicate, .transform = cancelled},
                    .edit = arc::editor::host_edit_transaction{.id = 8, .phase = arc::editor::host_edit_phase::cancel}})
                .succeeded);
    REQUIRE(host->selected_entity_snapshot().transform->position.y == Catch::Approx(preview.position.y));
}

TEST_CASE("viewport navigation and repeated selection do not emit scene refresh events")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({.name = "Event Dedup Test", .root = {}}, assets).succeeded);
    host->poll_events();

    const auto selected = host->selected_entity_snapshot().entity;
    REQUIRE(selected.valid());
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = selected}).succeeded);
    REQUIRE(host->poll_events().empty());

    REQUIRE(host->execute(arc::editor::host_viewport_camera_input_command{.look_x = 4.0f, .look_y = -2.0f}).succeeded);
    REQUIRE(host->poll_events().empty());

    REQUIRE(host->execute(arc::editor::host_clear_selection_command{}).succeeded);
    const auto cleared = host->poll_events();
    REQUIRE(cleared.size() == 1);
    REQUIRE(cleared.front().event_type == arc::editor::host_event_type::entity_selected);
    REQUIRE(host->execute(arc::editor::host_clear_selection_command{}).succeeded);
    REQUIRE(host->poll_events().empty());
}

TEST_CASE("viewport picking resolves the asynchronous ObjectID result before CPU bounds fallback")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({.name = "Async Pick Test", .root = {}}, assets).succeeded);
    auto backend = std::make_unique<pick_test_backend>();
    auto* backend_ptr = backend.get();
    host->renderer_service().set_backend(std::move(backend));
    host->poll_events();

    const arc::editor::host_entity_id original{host->scene_state().camera_entity.index,
                                               host->scene_state().camera_entity.generation};
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::water})
                .succeeded);
    const auto target = host->scene_state().water_entity;
    REQUIRE(host->scene_state().scene.alive(target));
    REQUIRE(target != (arc::ecs::entity{original.index, original.generation}));
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = original}).succeeded);
    host->poll_events();
    REQUIRE(host->execute(arc::editor::host_viewport_pick_command{.x = 24, .y = 36}).succeeded);
    REQUIRE(host->selected_entity_snapshot().entity == original);
    REQUIRE(backend_ptr->request_.x == 24);
    REQUIRE(backend_ptr->request_.y == 36);

    backend_ptr->result = {.request_id = backend_ptr->request_.request_id,
                           .available = true,
                           .hit = true,
                           .object = {target.index, target.generation},
                           .x = 24,
                           .y = 36,
                           .frame_index = 1};
    REQUIRE(host->request_viewport({.frame_index = 1, .width = 640, .height = 480}).submitted);
    REQUIRE(host->selected_entity_snapshot().entity == (arc::editor::host_entity_id{target.index, target.generation}));
    const auto events = host->poll_events();
    REQUIRE(std::count_if(events.begin(), events.end(), [](const auto& event)
                          { return event.event_type == arc::editor::host_event_type::entity_selected; }) == 1);
}

TEST_CASE("viewport captures remain asynchronous and map ObjectIDs to persistent GUIDs")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({.name = "Capture Test", .root = {}}, assets).succeeded);
    auto backend = std::make_unique<pick_test_backend>();
    auto* backend_ptr = backend.get();
    host->renderer_service().set_backend(std::move(backend));

    REQUIRE(host->execute(arc::editor::host_viewport_capture_command{.capture_id = 77,
                                                                     .color = true,
                                                                     .depth = false,
                                                                     .object_id = true,
                                                                     .normals = false,
                                                                     .scene_color = true,
                                                                     .base_color = true,
                                                                     .material_properties = true,
                                                                     .emissive = true})
                .succeeded);
    REQUIRE(backend_ptr->capture_request.capture_id == 77);
    REQUIRE(backend_ptr->capture_request.channels.size() == 6);
    REQUIRE(std::find(backend_ptr->capture_request.channels.begin(), backend_ptr->capture_request.channels.end(),
                      arc::render::render_capture_channel::scene_color) != backend_ptr->capture_request.channels.end());

    arc::editor::host_query_envelope query{.request_id = 8,
                                           .payload = arc::editor::host_viewport_capture_query{.capture_id = 77}};
    REQUIRE(host->query(query).payload_json.find("\"pending\":true") != std::string::npos);

    const auto selected = host->selected_entity_snapshot();
    backend_ptr->capture_result = {.capture_id = 77,
                                   .frame_index = 4,
                                   .available = true,
                                   .succeeded = true,
                                   .camera = {.position = {1.0f, 2.0f, 3.0f},
                                              .forward = {0.0f, 0.0f, -1.0f},
                                              .up = {0.0f, 1.0f, 0.0f},
                                              .near_plane = 0.1f,
                                              .far_plane = 500.0f,
                                              .render_width = 640,
                                              .render_height = 360,
                                              .output_width = 1280,
                                              .output_height = 720},
                                   .images = {{.channel = arc::render::render_capture_channel::output_color,
                                               .format = arc::render::render_capture_format::rgba8_unorm,
                                               .width = 1,
                                               .height = 1,
                                               .data = {std::byte{255}, std::byte{0}, std::byte{0}, std::byte{255}}}},
                                   .objects = {{.encoded_id = selected.entity.index + 1u,
                                                .object = {selected.entity.index, selected.entity.generation}}}};
    const auto completed = host->query(query);
    REQUIRE(completed.succeeded);
    REQUIRE(completed.payload_json.find("\"pending\":false") != std::string::npos);
    REQUIRE(completed.payload_json.find(selected.guid) != std::string::npos);
    REQUIRE(completed.payload_json.find("\"data\":\"/wAA/w==\"") != std::string::npos);
    REQUIRE(completed.payload_json.find("\"position\":[1.000000,2.000000,3.000000]") != std::string::npos);
    REQUIRE(completed.payload_json.find("\"outputExtent\":[1280,720]") != std::string::npos);
}

TEST_CASE("ARC scene documents save atomically, round trip hierarchy, and reject invalid loads")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-scene-document-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code error;
    std::filesystem::create_directories(root / "assets", error);
    REQUIRE_FALSE(error);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root / "assets";
    REQUIRE(host->open_project({.name = "Persistence Test", .root = root}, assets).succeeded);
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_component_operation_command{
                              .operation = arc::editor::host_component_operation::add, .component = "camera"})
                .succeeded);
    const auto selected = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_rename_entity_command{.entity = selected, .name = "Persisted Entity"})
                .succeeded);
    const auto initial_snapshot = host->scene_snapshot();
    const auto selected_record = std::find_if(initial_snapshot.entities.begin(), initial_snapshot.entities.end(),
                                              [selected](const auto& value) { return value.entity == selected; });
    REQUIRE(selected_record != initial_snapshot.entities.end());
    const auto selected_guid = selected_record->guid;
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = selected}).succeeded);

    const auto path = root / "scenes" / "mountain.arcscene";
    REQUIRE(host->execute(arc::editor::host_save_scene_as_command{.path = path}).succeeded);
    REQUIRE(std::filesystem::exists(path));
    REQUIRE_FALSE(host->scene_snapshot().dirty);

    REQUIRE(host->execute(arc::editor::host_rename_entity_command{.entity = selected, .name = "Changed"}).succeeded);
    REQUIRE(host->scene_snapshot().dirty);
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    REQUIRE_FALSE(host->scene_snapshot().dirty);
    REQUIRE(host->selected_entity_snapshot().name == "Persisted Entity");

    std::ifstream input(path, std::ios::binary);
    std::string document((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();
    const std::string component_marker = "\"components\": {";
    const auto component_position = document.find(component_marker);
    REQUIRE(component_position != std::string::npos);
    document.insert(component_position + component_marker.size(),
                    "\n        \"FutureRenderer\": {\"version\": 7, \"opaque\": {\"quality\": \"future\"}},");
    const std::string transform_marker = "\"Transform\": {";
    const auto transform_position = document.find(transform_marker);
    REQUIRE(transform_position != std::string::npos);
    document.insert(transform_position + transform_marker.size(),
                    "\n          \"futureTransformState\": {\"author\": \"future-editor\"},");
    const auto resealed = arc::persistence::seal_json_document(document, true);
    REQUIRE(resealed.succeeded());
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << resealed.text;
    }
    REQUIRE(host->execute(arc::editor::host_open_scene_command{.path = path}).succeeded);
    REQUIRE_FALSE(host->scene_snapshot().dirty);
    const auto loaded_snapshot = host->scene_snapshot();
    REQUIRE(std::any_of(loaded_snapshot.entities.begin(), loaded_snapshot.entities.end(), [&](const auto& value)
                        { return value.guid == selected_guid && value.name == "Persisted Entity"; }));
    REQUIRE(host->execute(arc::editor::host_save_scene_command{}).succeeded);
    std::ifstream resaved_input(path, std::ios::binary);
    const std::string resaved((std::istreambuf_iterator<char>(resaved_input)), std::istreambuf_iterator<char>());
    resaved_input.close();
    REQUIRE(resaved.find("FutureRenderer") != std::string::npos);
    REQUIRE(resaved.find("\"quality\": \"future\"") != std::string::npos);
    REQUIRE(resaved.find("futureTransformState") != std::string::npos);
    REQUIRE(resaved.find("future-editor") != std::string::npos);

    auto invalid = resaved;
    const auto current_version = "\"formatVersion\": " + std::to_string(arc::editor::arc_scene_format_version);
    const auto version = invalid.find(current_version);
    REQUIRE(version != std::string::npos);
    invalid.replace(version, current_version.size(), "\"formatVersion\": 99");
    const auto invalid_path = root / "scenes" / "unsupported.arcscene";
    {
        std::ofstream output(invalid_path, std::ios::binary | std::ios::trunc);
        output << invalid;
    }
    const auto before_invalid_load = arc::editor::to_json(host->scene_snapshot());
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{.path = invalid_path}).succeeded);
    REQUIRE(arc::editor::to_json(host->scene_snapshot()) == before_invalid_load);

    auto malformed_camera = resaved;
    const auto camera_component = malformed_camera.find("\"Camera\": {");
    REQUIRE(camera_component != std::string::npos);
    const auto near_key = malformed_camera.find("\"near\":", camera_component);
    REQUIRE(near_key != std::string::npos);
    const auto near_value = malformed_camera.find_first_not_of(" \t", near_key + std::string("\"near\":").size());
    const auto near_end = malformed_camera.find_first_of(",\r\n", near_value);
    REQUIRE(near_value != std::string::npos);
    REQUIRE(near_end != std::string::npos);
    malformed_camera.replace(near_value, near_end - near_value, "0.0");
    const auto malformed_camera_path = root / "scenes" / "malformed-camera.arcscene";
    {
        std::ofstream output(malformed_camera_path, std::ios::binary | std::ios::trunc);
        output << malformed_camera;
    }
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{.path = malformed_camera_path}).succeeded);
    REQUIRE(arc::editor::to_json(host->scene_snapshot()) == before_invalid_load);

    auto unsafe_asset = resaved;
    const auto first_components = unsafe_asset.find("\"components\": {");
    REQUIRE(first_components != std::string::npos);
    unsafe_asset.insert(first_components,
                        "\"assetBinding\": {\"kind\": \"imported\", \"path\": \"../outside.glb\"},\n        ");
    const auto unsafe_asset_path = root / "scenes" / "unsafe-asset.arcscene";
    {
        std::ofstream output(unsafe_asset_path, std::ios::binary | std::ios::trunc);
        output << unsafe_asset;
    }
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{.path = unsafe_asset_path}).succeeded);
    REQUIRE(arc::editor::to_json(host->scene_snapshot()) == before_invalid_load);

    const auto first_id_key = resaved.find("\"id\": \"");
    REQUIRE(first_id_key != std::string::npos);
    const auto first_id_begin = first_id_key + std::string("\"id\": \"").size();
    const auto first_id_end = resaved.find('"', first_id_begin);
    const auto second_id_key = resaved.find("\"id\": \"", first_id_end);
    REQUIRE(first_id_end != std::string::npos);
    REQUIRE(second_id_key != std::string::npos);
    const auto second_id_begin = second_id_key + std::string("\"id\": \"").size();
    const auto second_id_end = resaved.find('"', second_id_begin);
    REQUIRE(second_id_end != std::string::npos);
    const auto first_id = resaved.substr(first_id_begin, first_id_end - first_id_begin);
    const auto second_id = resaved.substr(second_id_begin, second_id_end - second_id_begin);

    auto duplicate_guid = resaved;
    duplicate_guid.replace(second_id_begin, second_id_end - second_id_begin, first_id);
    const auto duplicate_guid_path = root / "scenes" / "duplicate-guid.arcscene";
    {
        std::ofstream output(duplicate_guid_path, std::ios::binary | std::ios::trunc);
        output << duplicate_guid;
    }
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{.path = duplicate_guid_path}).succeeded);
    REQUIRE(arc::editor::to_json(host->scene_snapshot()) == before_invalid_load);

    auto cyclic_hierarchy = resaved;
    const auto replace_parent = [&](std::size_t id_key, const std::string& parent_id)
    {
        const auto parent_key = cyclic_hierarchy.find("\"parent\":", id_key);
        REQUIRE(parent_key != std::string::npos);
        const auto value_begin =
            cyclic_hierarchy.find_first_not_of(" \t", parent_key + std::string("\"parent\":").size());
        const auto value_end = cyclic_hierarchy.find_first_of(",\r\n", value_begin);
        REQUIRE(value_begin != std::string::npos);
        REQUIRE(value_end != std::string::npos);
        cyclic_hierarchy.replace(value_begin, value_end - value_begin, '"' + parent_id + '"');
    };
    // Replace later record first so the earlier byte offset remains valid.
    replace_parent(second_id_key, first_id);
    replace_parent(first_id_key, second_id);
    const auto cyclic_path = root / "scenes" / "cyclic.arcscene";
    {
        std::ofstream output(cyclic_path, std::ios::binary | std::ios::trunc);
        output << cyclic_hierarchy;
    }
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{.path = cyclic_path}).succeeded);
    REQUIRE(arc::editor::to_json(host->scene_snapshot()) == before_invalid_load);

    std::filesystem::remove_all(root, error);
}

TEST_CASE("Water component version 2 survives scene save and reload")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-water-scene-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code error;
    std::filesystem::create_directories(root / "assets", error);
    REQUIRE_FALSE(error);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root / "assets";
    REQUIRE(host->open_project({.name = "Water Persistence Test", .root = root}, assets).succeeded);
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::water})
                .succeeded);

    auto& authored = host->scene_state().scene.get<arc::scene::water_component>(host->scene_state().water_entity);
    authored.preset.path_hint = "assets/water/Open Ocean.arcwater";
    authored.water_level = 2.75f;
    authored.visible_distance = 32000.0f;
    authored.settings.simulation.wind_speed = 17.0f;
    authored.settings.simulation.wind_direction = {0.82f, 0.57f};
    authored.settings.simulation.seed = 1337;
    authored.settings.appearance.absorption = {0.20f, 0.07f, 0.03f};
    authored.settings.appearance.refraction_strength = 0.08f;
    authored.settings.quality = arc::water::water_quality::ultra;

    const auto path = root / "scenes" / "water.arcscene";
    REQUIRE(host->execute(arc::editor::host_save_scene_as_command{.path = path}).succeeded);
    {
        std::ifstream input(path, std::ios::binary);
        const std::string document((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        CHECK(document.find("\"Water\"") != std::string::npos);
        CHECK(document.find("\"version\": 2") != std::string::npos);
        CHECK(document.find("Open Ocean.arcwater") != std::string::npos);
    }

    REQUIRE(host->execute(arc::editor::host_open_scene_command{.path = path}).succeeded);
    REQUIRE(host->scene_state().scene.alive(host->scene_state().water_entity));
    const auto& loaded = host->scene_state().scene.get<arc::scene::water_component>(host->scene_state().water_entity);
    CHECK(loaded.type == arc::water::water_body_type::ocean);
    CHECK(loaded.preset.path_hint == "assets/water/Open Ocean.arcwater");
    CHECK(loaded.water_level == Catch::Approx(2.75f));
    CHECK(loaded.visible_distance == Catch::Approx(32000.0f));
    CHECK(loaded.settings.simulation.wind_speed == Catch::Approx(17.0f));
    CHECK(loaded.settings.simulation.wind_direction[1] == Catch::Approx(0.57f));
    CHECK(loaded.settings.simulation.seed == 1337);
    CHECK(loaded.settings.appearance.absorption[0] == Catch::Approx(0.20f));
    CHECK(loaded.settings.appearance.refraction_strength == Catch::Approx(0.08f));
    CHECK(loaded.settings.quality == arc::water::water_quality::ultra);

    std::filesystem::remove_all(root, error);
}

TEST_CASE("Water Inspector snapshots and validated edits round trip through the host protocol")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({.name = "Water Inspector Test", .root = {}}, assets).succeeded);
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::water})
                .succeeded);

    const auto created = host->selected_entity_snapshot();
    REQUIRE(created.name == "Ocean");
    REQUIRE(created.water.has_value());
    REQUIRE(created.water->body_type == 0u);
    auto updated = *created.water;
    updated.water_level = 4.25f;
    updated.wind_speed = 18.0f;
    updated.wind_direction_x = 0.6f;
    updated.wind_direction_y = 0.8f;
    updated.wave_amplitude = 2.0f;
    updated.absorption = {0.3f, 0.08f, 0.025f};
    updated.quality = 3u;
    updated.priority = 7;

    const auto command = arc::editor::host_set_water_command{.entity = created.entity, .water = updated};
    REQUIRE(host->execute(command).succeeded);
    const auto configured = host->selected_entity_snapshot();
    REQUIRE(configured.water.has_value());
    CHECK(configured.water->water_level == Catch::Approx(4.25f));
    CHECK(configured.water->wind_speed == Catch::Approx(18.0f));
    CHECK(configured.water->absorption.x == Catch::Approx(0.3f));
    CHECK(configured.water->quality == 3u);
    CHECK(configured.water->priority == 7);
    CHECK(arc::editor::to_json(configured).find("\"water\":{") != std::string::npos);

    arc::editor::host_command_envelope source{
        .request_id = 91, .command_type = arc::editor::command_type(command), .payload = command};
    arc::editor::host_command_envelope parsed;
    std::string protocol_error;
    REQUIRE(arc::editor::from_json(arc::editor::to_json(source), parsed, protocol_error));
    REQUIRE(arc::editor::command_type(parsed.payload) == "water.update");
    const auto& parsed_command = std::get<arc::editor::host_set_water_command>(parsed.payload);
    CHECK(parsed_command.entity == command.entity);
    CHECK(parsed_command.water == command.water);

    auto invalid = updated;
    invalid.wind_direction_x = 0.0f;
    invalid.wind_direction_y = 0.0f;
    REQUIRE_FALSE(
        host->execute(arc::editor::host_set_water_command{.entity = created.entity, .water = invalid}).succeeded);
    CHECK(host->selected_entity_snapshot().water->water_level == Catch::Approx(4.25f));
}

TEST_CASE("built-in Water presets are discovered and drive Ocean defaults")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-water-presets-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code error;
    std::filesystem::create_directories(root / "Content", error);
    REQUIRE_FALSE(error);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    const auto builtin_root = std::filesystem::path{ARC_SOURCE_ROOT} / "assets";
    REQUIRE(host->open_project({.name = "Water Presets Test",
                                .root = root,
                                .content_roots = {assets.root},
                                .builtin_content_roots = {builtin_root}},
                               assets)
                .succeeded);
    const auto project_assets = host->project_assets_snapshot();
    CHECK(std::count_if(project_assets.assets.begin(), project_assets.assets.end(),
                        [](const auto& asset)
                        {
                            return asset.kind == "water" &&
                                   asset.type_id == arc::assets::to_string(arc::assets::asset_types::water_preset) &&
                                   asset.importer_id == arc::assets::to_string(arc::assets::importer_ids::water_preset);
                        }) >= 5);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::water})
                .succeeded);
    const auto ocean = host->selected_entity_snapshot();
    REQUIRE(ocean.water.has_value());
    CHECK_FALSE(ocean.water->preset_guid.empty());
    CHECK(ocean.water->preset_path == "builtin/water/presets/open_ocean.arcwater");
    CHECK(ocean.water->wind_speed == Catch::Approx(12.0f));

    auto storm = *ocean.water;
    storm.preset_guid.clear();
    storm.preset_path = "builtin/water/presets/storm.arcwater";
    REQUIRE(host->execute(arc::editor::host_set_water_command{.entity = ocean.entity, .water = storm}).succeeded);
    const auto configured = host->selected_entity_snapshot();
    REQUIRE(configured.water.has_value());
    CHECK(configured.water->preset_path == "builtin/water/presets/storm.arcwater");
    CHECK(configured.water->wind_speed == Catch::Approx(28.0f));
    CHECK(configured.water->wave_amplitude == Catch::Approx(5.0f));
    CHECK(configured.water->quality == 3u);

    std::filesystem::remove_all(root, error);
}

TEST_CASE("selected camera snapshots and entity-specific edits round trip atomically")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(
        host->open_project({.name = "Inspector Camera Test", .root = std::filesystem::temp_directory_path()}, assets)
            .succeeded);
    const auto editor_camera = host->scene_state().camera_entity;
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_component_operation_command{
                              .operation = arc::editor::host_component_operation::add, .component = "camera"})
                .succeeded);
    const auto game_camera = arc::ecs::entity{host->selected_entity_snapshot().entity.index,
                                              host->selected_entity_snapshot().entity.generation};
    const arc::editor::host_entity_id game_camera_id{game_camera.index, game_camera.generation};

    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .request_id = 1, .payload = arc::editor::host_select_entity_command{.entity = game_camera_id}})
                .succeeded);

    const auto selected = host->selected_entity_snapshot();
    REQUIRE(selected.entity == game_camera_id);
    REQUIRE(selected.camera.has_value());
    REQUIRE(selected.render_layer_mask == arc::editor::host_default_render_layer);
    REQUIRE(std::any_of(selected.components.begin(), selected.components.end(), [](const auto& component)
                        { return component.kind == arc::editor::host_component_kind::camera && component.editable; }));

    const auto editor_before = host->scene_state().scene.get<arc::scene::camera_component>(editor_camera);
    auto updated = *selected.camera;
    updated.projection = arc::editor::host_camera_projection::orthographic;
    updated.fov_y_degrees = 72.0f;
    updated.orthographic_height = 24.0f;
    updated.near_plane = 0.25f;
    updated.far_plane = 4096.0f;
    updated.active = true;
    updated.clear_color = {0.1f, 0.2f, 0.3f, 0.8f};
    REQUIRE(
        host->execute(arc::editor::host_command_envelope{
                          .request_id = 2,
                          .payload = arc::editor::host_set_camera_command{.entity = game_camera_id, .camera = updated}})
            .succeeded);

    const auto round_trip = host->selected_entity_snapshot();
    REQUIRE(round_trip.camera.has_value());
    REQUIRE(*round_trip.camera == updated);
    const auto& editor_after = host->scene_state().scene.get<arc::scene::camera_component>(editor_camera);
    REQUIRE(editor_after.projection == editor_before.projection);
    REQUIRE(editor_after.fov_y_radians == Catch::Approx(editor_before.fov_y_radians));
    REQUIRE(editor_after.near_plane == Catch::Approx(editor_before.near_plane));

    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .request_id = 3,
                    .payload =
                        arc::editor::host_set_render_layer_command{
                            .entity = game_camera_id, .render_layer_mask = arc::editor::host_environment_render_layer}})
                .succeeded);
    REQUIRE(host->selected_entity_snapshot().render_layer_mask == arc::editor::host_environment_render_layer);

    const auto confirmed = *host->selected_entity_snapshot().camera;
    for (const auto invalid :
         {arc::editor::host_camera_snapshot{confirmed.projection, 1.0f, confirmed.orthographic_height,
                                            confirmed.near_plane, confirmed.far_plane, confirmed.active,
                                            confirmed.clear_color},
          arc::editor::host_camera_snapshot{confirmed.projection, confirmed.fov_y_degrees, 0.0f, confirmed.near_plane,
                                            confirmed.far_plane, confirmed.active, confirmed.clear_color},
          arc::editor::host_camera_snapshot{confirmed.projection, confirmed.fov_y_degrees,
                                            confirmed.orthographic_height, 0.0f, confirmed.far_plane, confirmed.active,
                                            confirmed.clear_color},
          arc::editor::host_camera_snapshot{confirmed.projection, confirmed.fov_y_degrees,
                                            confirmed.orthographic_height, confirmed.near_plane, confirmed.near_plane,
                                            confirmed.active, confirmed.clear_color},
          arc::editor::host_camera_snapshot{confirmed.projection, std::numeric_limits<float>::infinity(),
                                            confirmed.orthographic_height, confirmed.near_plane, confirmed.far_plane,
                                            confirmed.active, confirmed.clear_color},
          arc::editor::host_camera_snapshot{confirmed.projection,
                                            confirmed.fov_y_degrees,
                                            confirmed.orthographic_height,
                                            confirmed.near_plane,
                                            confirmed.far_plane,
                                            confirmed.active,
                                            {-0.1f, 0.2f, 0.3f, 1.0f}}})
    {
        REQUIRE_FALSE(host->execute(arc::editor::host_command_envelope{
                                        .request_id = 4,
                                        .payload = arc::editor::host_set_camera_command{.entity = game_camera_id,
                                                                                        .camera = invalid}})
                          .succeeded);
        REQUIRE(*host->selected_entity_snapshot().camera == confirmed);
    }

    const auto json = arc::editor::to_json(host->selected_entity_snapshot());
    REQUIRE(json.find("\"camera\":{") != std::string::npos);
    REQUIRE(json.find("\"renderLayerMask\":2") != std::string::npos);
}
