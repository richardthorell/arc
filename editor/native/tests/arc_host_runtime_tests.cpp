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

TEST_CASE("camera and render layer JSON commands preserve their typed payloads")
{
    const arc::editor::host_entity_id entity{12, 4};
    arc::editor::host_camera_snapshot camera;
    camera.fov_y_degrees = 80.0f;
    camera.near_plane = 0.5f;
    camera.far_plane = 5000.0f;
    camera.clear_color = {0.2f, 0.3f, 0.4f, 1.0f};

    const std::array<arc::editor::host_command_payload, 2> payloads{
        arc::editor::host_command_payload{arc::editor::host_set_camera_command{.entity = entity, .camera = camera}},
        arc::editor::host_command_payload{
            arc::editor::host_set_render_layer_command{.entity = entity, .render_layer_mask = 2u}}};
    for (const auto& payload : payloads)
    {
        const arc::editor::host_command_envelope source{.request_id = 91, .payload = payload};
        arc::editor::host_command_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(arc::editor::to_json(source), parsed, error));
        REQUIRE(parsed.request_id == 91);
        REQUIRE(arc::editor::command_type(parsed.payload) == arc::editor::command_type(payload));
    }
}

TEST_CASE("physical light snapshots edit atomically and round trip through JSON")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({.name = "Inspector Light Test", .root = std::filesystem::temp_directory_path()}, assets)
                .succeeded);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::empty})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_component_operation_command{
                              .operation = arc::editor::host_component_operation::add, .component = "directionalLight"})
                .succeeded);
    const auto sun = arc::ecs::entity{host->selected_entity_snapshot().entity.index,
                                      host->selected_entity_snapshot().entity.generation};
    const arc::editor::host_entity_id sun_id{sun.index, sun.generation};
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = sun_id}).succeeded);
    const auto selected = host->selected_entity_snapshot();
    REQUIRE(selected.light.has_value());
    REQUIRE(selected.light->kind == arc::editor::host_light_kind::directional);
    REQUIRE(selected.light->unit == arc::editor::host_light_unit::lux);

    auto updated = *selected.light;
    updated.intensity = 85000.0f;
    updated.color = {1.0f, 0.82f, 0.63f};
    updated.use_color_temperature = true;
    updated.temperature_kelvin = 5200.0f;
    updated.shadow_resolution = 3072;
    updated.shadow_priority = 900;
    updated.shadow_strength = 0.9f;
    updated.contact_shadows = true;
    updated.contact_shadow_length = 1.25f;
    updated.shadow_map_method = 2;
    updated.cascade_count = 3;
    updated.shadow_distance = 180.0f;
    updated.cascade_split_lambda = 0.7f;
    updated.cascade_blend_fraction = 0.1f;
    REQUIRE(host->execute(arc::editor::host_set_light_command{.entity = sun_id, .light = updated}).succeeded);
    REQUIRE(*host->selected_entity_snapshot().light == updated);
    REQUIRE(host->execute(arc::editor::host_set_mobility_command{.entity = sun_id,
                                                                 .mobility = arc::editor::host_mobility::static_object})
                .succeeded);
    REQUIRE(host->selected_entity_snapshot().mobility == arc::editor::host_mobility::static_object);

    auto invalid = updated;
    invalid.unit = arc::editor::host_light_unit::lumen;
    REQUIRE_FALSE(host->execute(arc::editor::host_set_light_command{.entity = sun_id, .light = invalid}).succeeded);
    REQUIRE(*host->selected_entity_snapshot().light == updated);

    const arc::editor::host_command_envelope source{
        .request_id = 92, .payload = arc::editor::host_set_light_command{.entity = sun_id, .light = updated}};
    arc::editor::host_command_envelope parsed;
    std::string error;
    REQUIRE(arc::editor::from_json(arc::editor::to_json(source), parsed, error));
    REQUIRE(arc::editor::command_type(parsed.payload) == "entity.setLight");
    const auto selected_json = arc::editor::to_json(host->selected_entity_snapshot());
    REQUIRE(selected_json.find("\"light\":{") != std::string::npos);
    REQUIRE(selected_json.find("\"shadowResolution\":3072") != std::string::npos);
    REQUIRE(selected_json.find("\"mobility\":\"static\"") != std::string::npos);
}

TEST_CASE("scene authoring protocol commands and edit transactions round trip")
{
    const arc::editor::host_entity_id entity{8, 3};
    const arc::editor::host_entity_id parent{4, 2};
    const std::array<arc::editor::host_command_payload, 9> payloads{
        arc::editor::host_new_scene_command{.name = "New World"},
        arc::editor::host_save_scene_command{},
        arc::editor::host_save_scene_as_command{.path = "scenes/world.arcscene"},
        arc::editor::host_duplicate_entity_command{.entity = entity},
        arc::editor::host_reparent_entity_command{.entity = entity, .parent = parent, .preserve_world = true},
        arc::editor::host_reorder_entity_command{.entity = entity, .before_sibling = parent},
        arc::editor::host_history_undo_command{},
        arc::editor::host_history_redo_command{},
        arc::editor::host_viewport_set_tool_command{.tool = arc::editor::host_viewport_tool::rotate,
                                                    .coordinate_space = arc::editor::host_coordinate_space::local,
                                                    .snapping = true}};
    for (const auto& payload : payloads)
    {
        const arc::editor::host_command_envelope source{
            .request_id = 71,
            .command_type = arc::editor::command_type(payload),
            .payload = payload,
            .edit = arc::editor::host_edit_transaction{
                .id = 44, .phase = arc::editor::host_edit_phase::commit, .label = "Quoted \"edit\""}};
        arc::editor::host_command_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(arc::editor::to_json(source), parsed, error));
        REQUIRE(parsed.request_id == source.request_id);
        REQUIRE(arc::editor::command_type(parsed.payload) == arc::editor::command_type(payload));
        REQUIRE(parsed.edit.has_value());
        REQUIRE(parsed.edit->id == 44);
        REQUIRE(parsed.edit->phase == arc::editor::host_edit_phase::commit);
        REQUIRE(parsed.edit->label == "Quoted \"edit\"");
    }

    arc::editor::host_query_envelope history_query;
    std::string error;
    REQUIRE(arc::editor::from_json("{\"kind\":\"query\",\"requestId\":5,\"type\":\"history.state\",\"payload\":{}}",
                                   history_query, error));
    REQUIRE(std::holds_alternative<arc::editor::host_history_state_query>(history_query.payload));
}

TEST_CASE("runtime protocol commands and state query round trip")
{
    const std::array<arc::editor::host_command_payload, 7> payloads{
        arc::editor::host_runtime_resume_command{},
        arc::editor::host_runtime_pause_command{},
        arc::editor::host_runtime_stop_command{},
        arc::editor::host_runtime_step_command{.ticks = 3},
        arc::editor::host_runtime_set_time_scale_command{.value = 2.0},
        arc::editor::host_runtime_capture_snapshot_command{.label = "Before ability"},
        arc::editor::host_runtime_restore_snapshot_command{.snapshot_id = 9}};
    for (const auto& payload : payloads)
    {
        const arc::editor::host_command_envelope source{
            .request_id = 88, .command_type = arc::editor::command_type(payload), .payload = payload};
        arc::editor::host_command_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(arc::editor::to_json(source), parsed, error));
        REQUIRE(parsed.request_id == 88);
        REQUIRE(arc::editor::command_type(parsed.payload) == arc::editor::command_type(payload));
    }

    arc::editor::host_query_envelope query;
    std::string error;
    REQUIRE(arc::editor::from_json("{\"kind\":\"query\",\"requestId\":12,\"type\":\"runtime.state\",\"payload\":{}}",
                                   query, error));
    REQUIRE(std::holds_alternative<arc::editor::host_runtime_state_query>(query.payload));

    REQUIRE(arc::editor::from_json(
        "{\"kind\":\"query\",\"requestId\":13,\"type\":\"runtime.hierarchy\",\"payload\":{}}", query, error));
    REQUIRE(std::holds_alternative<arc::editor::host_runtime_hierarchy_query>(query.payload));
    REQUIRE(arc::editor::from_json("{\"kind\":\"query\",\"requestId\":14,\"type\":\"runtime.entity\",\"payload\":{"
                                   "\"entity\":{\"index\":7,\"generation\":2}}}",
                                   query, error));
    const auto parsed_runtime_entity = std::get<arc::editor::host_runtime_entity_query>(query.payload).entity;
    REQUIRE(parsed_runtime_entity.index == 7);
    REQUIRE(parsed_runtime_entity.generation == 2);

    arc::editor::host_runtime_snapshot snapshot{.state = arc::editor::host_runtime_state::paused,
                                                .tick_id = 42,
                                                .revision = 7,
                                                .discarded_ticks = 3,
                                                .time_scale = 0.5,
                                                .interpolation_alpha = 0.25,
                                                .world_count = 2,
                                                .error = "project system fault"};
    const auto json = arc::editor::to_json(snapshot);
    REQUIRE(json.find("\"state\":\"paused\"") != std::string::npos);
    REQUIRE(json.find("\"tickId\":42") != std::string::npos);
    REQUIRE(json.find("\"error\":\"project system fault\"") != std::string::npos);
}

TEST_CASE("active play worlds expose read-only hierarchy and entity inspection queries")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    REQUIRE(host->open_project({.name = "Runtime Inspection", .root = std::filesystem::temp_directory_path()}, {})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube})
                .succeeded);
    const auto authored = host->selected_entity_snapshot().entity;

    REQUIRE_FALSE(host->query({.request_id = 20, .payload = arc::editor::host_runtime_hierarchy_query{}}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);

    const auto hierarchy = host->query({.request_id = 21, .payload = arc::editor::host_runtime_hierarchy_query{}});
    REQUIRE(hierarchy.succeeded);
    CHECK(hierarchy.payload_json.find("(Play World)") != std::string::npos);
    CHECK(hierarchy.payload_json.find("Cube 1") != std::string::npos);

    const auto inspected =
        host->query({.request_id = 22, .payload = arc::editor::host_runtime_entity_query{.entity = authored}});
    REQUIRE(inspected.succeeded);
    CHECK(inspected.payload_json.find("\"name\":\"Cube 1\"") != std::string::npos);
    CHECK(inspected.payload_json.find("\"transform\":{") != std::string::npos);

    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);
    REQUIRE_FALSE(host->query({.request_id = 23, .payload = arc::editor::host_runtime_entity_query{.entity = authored}})
                      .succeeded);
}

TEST_CASE("arc host runtime controls are authoritative and revisioned")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    auto state = host->runtime_snapshot();
    REQUIRE(state.state == arc::editor::host_runtime_state::stopped);
    REQUIRE(state.tick_id == 0);
    const auto initial_revision = state.revision;

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    state = host->runtime_snapshot();
    REQUIRE(state.state == arc::editor::host_runtime_state::running);
    REQUIRE(state.revision > initial_revision);

    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 2}).succeeded);
    state = host->runtime_snapshot();
    REQUIRE(state.state == arc::editor::host_runtime_state::paused);
    REQUIRE(state.tick_id == 2);

    REQUIRE(host->execute(arc::editor::host_runtime_set_time_scale_command{.value = 4.0}).succeeded);
    REQUIRE(host->runtime_snapshot().time_scale == Catch::Approx(4.0));
    REQUIRE_FALSE(host->execute(arc::editor::host_runtime_set_time_scale_command{
                                    .value = std::numeric_limits<double>::infinity()})
                      .succeeded);

    const auto response = host->query({.request_id = 99, .payload = arc::editor::host_runtime_state_query{}});
    REQUIRE(response.succeeded);
    REQUIRE(response.payload_json.find("\"tickId\":2") != std::string::npos);

    const auto events = host->poll_events();
    REQUIRE(std::any_of(events.begin(), events.end(), [](const auto& event)
                        { return event.event_type == arc::editor::host_event_type::runtime_state_changed; }));
    REQUIRE(std::any_of(events.begin(), events.end(), [](const auto& event)
                        { return event.event_type == arc::editor::host_event_type::runtime_tick_completed; }));
}

TEST_CASE("editor preview checkpoints operate on the authoritative authoring world")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    REQUIRE(host->open_project({.name = "Runtime Authoring World", .root = std::filesystem::temp_directory_path()}, {})
                .succeeded);

    const std::size_t original_count = host->scene_snapshot().entities.size();
    const auto captured =
        host->execute(arc::editor::host_runtime_capture_snapshot_command{.label = "Authoring checkpoint"});
    REQUIRE(captured.succeeded);
    const std::string marker = "\"snapshotId\":";
    const std::size_t marker_offset = captured.payload_json.find(marker);
    REQUIRE(marker_offset != std::string::npos);
    const std::uint64_t snapshot_id = std::stoull(captured.payload_json.substr(marker_offset + marker.size()));

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube})
                .succeeded);
    REQUIRE(host->scene_snapshot().entities.size() == original_count + 1);
    REQUIRE(host->execute(arc::editor::host_runtime_restore_snapshot_command{.snapshot_id = snapshot_id}).succeeded);
    REQUIRE(host->scene_snapshot().entities.size() == original_count);
    const auto events = host->poll_events();
    REQUIRE(std::any_of(events.begin(), events.end(),
                        [](const auto& event)
                        {
                            return event.event_type == arc::editor::host_event_type::scene_changed &&
                                   event.message == "Preview runtime snapshot restored";
                        }));
}

TEST_CASE("arc host resolves the default Content directory for descriptor-free test projects")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-host-project-assets-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "Content", ec);
    REQUIRE_FALSE(ec);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    const auto response = host->execute(arc::editor::host_command_envelope{
        .request_id = 1, .payload = arc::editor::host_open_project_command{.name = "Asset Root Test", .root = root}});

    REQUIRE(response.succeeded);
    REQUIRE(host->project_assets_snapshot().asset_root == root / "Content");
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("arc host derives project roots from the validated descriptor")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-host-project-context-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "Content", ec);
    REQUIRE_FALSE(ec);
    const auto descriptor = root / "Context.arcproject";
    std::ofstream(descriptor)
        << R"({"format":"arc-project","formatVersion":3,"guid":"12345678-1234-4234-8234-123456789abc","name":"Context","engineVersion":"0.1.0","assetRoots":["Content"],"modules":[],"plugins":[],"targetPlatforms":[{"id":"windows-x64-vulkan","enabled":true}],"renderer":{"backend":"vulkan","api":"1.2","quality":"standard"}})";

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    const auto response = host->execute(arc::editor::host_command_envelope{
        .request_id = 1,
        .payload = arc::editor::host_open_project_command{.name = "Forged Name",
                                                          .root = root / "wrong-root",
                                                          .descriptor_path = descriptor,
                                                          .content_roots = {root / "wrong-content"},
                                                          .cache_root = root / "wrong-cache"}});

    INFO(response.error);
    REQUIRE(response.succeeded);
    REQUIRE(host->project_assets_snapshot().asset_root == root / "Content");
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("blank 3D project template opens its native authoring scene")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-blank-3d-template-host-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    const auto created =
        arc::project::create_project({.name = "TemplateHost",
                                      .destination = root,
                                      .template_id = "blank-3d",
                                      .templates_root = std::filesystem::path(ARC_SOURCE_ROOT) / "templates",
                                      .engine_version = "0.1.0"});
    REQUIRE(created.has_value());
    const auto descriptor_path = root / "TemplateHost.arcproject";
    const auto descriptor = arc::project::load_descriptor(descriptor_path);
    REQUIRE(descriptor.has_value());
    REQUIRE_FALSE(descriptor.value().default_scene.has_value());

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    const auto response = host->execute(arc::editor::host_command_envelope{
        .request_id = 1,
        .payload = arc::editor::host_open_project_command{.name = descriptor.value().name,
                                                          .root = root,
                                                          .descriptor_path = descriptor_path,
                                                          .content_roots = {root / "Content"},
                                                          .cache_root = root / "Intermediate" / "Cache",
                                                          .project_guid = descriptor.value().guid,
                                                          .engine_version = descriptor.value().engine_version}});

    INFO(response.error);
    REQUIRE(response.succeeded);
    auto snapshot = host->scene_snapshot();
    REQUIRE(snapshot.entities.size() == 3u);
    const auto camera = std::find_if(snapshot.entities.begin(), snapshot.entities.end(),
                                     [](const auto& entity) { return entity.name == "Main Camera"; });
    REQUIRE(camera != snapshot.entities.end());
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = camera->entity}).succeeded);
    const auto selected_camera = host->selected_entity_snapshot();
    REQUIRE(selected_camera.camera.has_value());
    snapshot = host->scene_snapshot();
    REQUIRE(snapshot.entities.size() == 3u);
    REQUIRE(std::any_of(snapshot.entities.begin(), snapshot.entities.end(),
                        [](const auto& entity) { return entity.name == "Main Camera"; }));
    const auto sky = std::find_if(snapshot.entities.begin(), snapshot.entities.end(),
                                  [](const auto& entity) { return entity.name == "Default Sky"; });
    REQUIRE(sky != snapshot.entities.end());
    REQUIRE(sky->kind == arc::editor::host_entity_kind::environment);
    const auto sky_id = sky->entity;
    const auto sky_entity = arc::ecs::entity{sky_id.index, sky_id.generation};
    REQUIRE(host->scene_state().scene.has<arc::scene::directional_light_component>(sky_entity));
    const auto environment = host->world_environment_snapshot(sky_id);
    REQUIRE(environment.has_value());
    REQUIRE(environment->enabled);
    REQUIRE(environment->sky_visible);
    REQUIRE(environment->affect_lighting);
    const auto floor = std::find_if(snapshot.entities.begin(), snapshot.entities.end(),
                                    [](const auto& entity) { return entity.name == "Floor"; });
    REQUIRE(floor != snapshot.entities.end());
    const auto floor_entity = arc::ecs::entity{floor->entity.index, floor->entity.generation};
    REQUIRE(host->scene_state().scene.has<arc::scene::mesh_renderer_component>(floor_entity));
    const auto& floor_renderer = host->scene_state().scene.get<arc::scene::mesh_renderer_component>(floor_entity);
    REQUIRE_FALSE(floor_renderer.casts_shadows);
    REQUIRE(floor_renderer.receives_shadows);
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = floor->entity}).succeeded);
    REQUIRE(host->execute(arc::editor::host_delete_entity_command{.entity = floor->entity}).succeeded);
    snapshot = host->scene_snapshot();
    REQUIRE(snapshot.entities.size() == 2u);
    REQUIRE(std::none_of(snapshot.entities.begin(), snapshot.entities.end(),
                         [](const auto& entity) { return entity.name == "Floor"; }));
    REQUIRE_FALSE(host->selected_entity_snapshot().entity.valid());
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    snapshot = host->scene_snapshot();
    REQUIRE(std::any_of(snapshot.entities.begin(), snapshot.entities.end(),
                        [](const auto& entity) { return entity.name == "Floor"; }));
    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = sky_id}).succeeded);
    REQUIRE(host->execute(arc::editor::host_set_active_command{.entity = sky_id, .active = false}).succeeded);
    snapshot = host->scene_snapshot();
    REQUIRE(snapshot.entities.size() == 3u);
    const auto edited_sky = std::find_if(snapshot.entities.begin(), snapshot.entities.end(),
                                         [](const auto& entity) { return entity.name == "Default Sky"; });
    REQUIRE(edited_sky != snapshot.entities.end());
    REQUIRE_FALSE(edited_sky->active);
    REQUIRE(host->world_environment_snapshot(sky_id).has_value());
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("world environment host snapshots round trip every settings group and preserve runtime handles")
{
    arc::scene::world_environment_settings settings;
    settings.world.hdri_texture = {.index = 1, .generation = 2};
    settings.celestial.sun_light = {.index = 3, .generation = 4};
    settings.celestial.animation_time_seconds = 91.0f;
    settings.lighting.environment = {.index = 5, .generation = 6};
    settings.lighting.hdri_texture = {.index = 7, .generation = 8};
    const arc::editor::host_entity_id entity{12, 2};
    auto snapshot = arc::editor::to_host_world_environment_snapshot(entity, settings, "environments/studio.hdr");

    snapshot.enabled = false;
    snapshot.sky_visible = false;
    snapshot.sky_source = arc::editor::host_sky_source::solid_color;
    snapshot.solid_color = {0.1f, 0.2f, 0.3f};
    snapshot.hdri_rotation_degrees = 42.0f;
    snapshot.radiance_intensity = 1.7f;
    snapshot.rayleigh_strength = 1.2f;
    snapshot.mie_strength = 0.22f;
    snapshot.sun_mode = arc::editor::host_sun_position_mode::manual_light;
    snapshot.time_mode = arc::editor::host_celestial_time_mode::simulated;
    snapshot.local_time_hours = 18.5f;
    snapshot.star_density = 0.33f;
    snapshot.clouds_enabled = false;
    snapshot.cumulus.coverage = 0.41f;
    snapshot.fog_density = 0.012f;
    snapshot.lighting_source = arc::editor::host_environment_lighting_source::constant_color;
    snapshot.diffuse_intensity = 0.75f;

    const auto converted = arc::editor::apply_host_world_environment_snapshot(snapshot, settings);
    REQUIRE_FALSE(converted.world.enabled);
    REQUIRE(converted.world.source == arc::scene::sky_source::solid_color);
    REQUIRE(converted.world.solid_color[1] == Catch::Approx(0.2f));
    REQUIRE(converted.atmosphere.rayleigh_strength == Catch::Approx(1.2f));
    REQUIRE(converted.celestial.time_mode == arc::scene::celestial_time_mode::simulated);
    REQUIRE(converted.celestial.local_time_hours == Catch::Approx(18.5f));
    REQUIRE_FALSE(converted.clouds.enabled);
    REQUIRE(converted.clouds.cumulus.coverage == Catch::Approx(0.41f));
    REQUIRE(converted.fog.density == Catch::Approx(0.012f));
    REQUIRE(converted.lighting.source == arc::scene::environment_lighting_source::constant_color);
    REQUIRE(converted.lighting.diffuse_intensity == Catch::Approx(0.75f));
    REQUIRE(converted.world.hdri_texture.index == 1);
    REQUIRE(converted.celestial.sun_light.index == 3);
    REQUIRE(converted.celestial.animation_time_seconds == Catch::Approx(91.0f));
    REQUIRE(converted.lighting.environment.index == 5);
    REQUIRE(converted.lighting.hdri_texture.index == 7);

    const auto round_trip = arc::editor::to_host_world_environment_snapshot(entity, converted, snapshot.hdri_path);
    REQUIRE(round_trip.entity == entity);
    REQUIRE(round_trip.sky_source == snapshot.sky_source);
    REQUIRE(round_trip.hdri_path == "environments/studio.hdr");
    REQUIRE(round_trip.local_time_hours == Catch::Approx(snapshot.local_time_hours));
    REQUIRE(round_trip.cumulus.coverage == Catch::Approx(snapshot.cumulus.coverage));
}

TEST_CASE("arc host validates and applies world environment commands")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({.name = "Environment Host Test", .root = {}}, assets).succeeded);
    REQUIRE(host->execute(arc::editor::host_create_entity_command{
                              .kind = arc::editor::host_create_entity_kind::world_environment})
                .succeeded);

    const auto hierarchy = host->scene_snapshot();
    const auto found = std::find_if(hierarchy.entities.begin(), hierarchy.entities.end(), [](const auto& entity)
                                    { return entity.kind == arc::editor::host_entity_kind::environment; });
    REQUIRE(found != hierarchy.entities.end());
    const auto initial = host->world_environment_snapshot(found->entity);
    REQUIRE(initial.has_value());
    REQUIRE(initial->enabled);
    REQUIRE(initial->sky_source == arc::editor::host_sky_source::physical_atmosphere);

    auto edited = *initial;
    edited.sky_visible = false;
    edited.affect_lighting = true;
    edited.local_time_hours = 19.25f;
    edited.sun_temperature_multiplier = 0.85f;
    edited.moon_angular_radius_degrees = 0.31f;
    const auto updated = host->execute(arc::editor::host_command_envelope{
        .request_id = 1, .payload = arc::editor::host_set_world_environment_command{.environment = edited}});
    REQUIRE(updated.succeeded);
    const auto current = host->world_environment_snapshot(found->entity);
    REQUIRE(current.has_value());
    REQUIRE_FALSE(current->sky_visible);
    REQUIRE(current->affect_lighting);
    REQUIRE(current->local_time_hours == Catch::Approx(19.25f));
    REQUIRE(current->sun_temperature_multiplier == Catch::Approx(0.85f));
    REQUIRE(current->moon_angular_radius_degrees == Catch::Approx(0.31f));

    edited.local_time_hours = 25.0f;
    REQUIRE_FALSE(host->execute(arc::editor::host_command_envelope{
                                    .request_id = 2,
                                    .payload = arc::editor::host_set_world_environment_command{.environment = edited}})
                      .succeeded);
    REQUIRE(host->world_environment_snapshot(found->entity)->local_time_hours == Catch::Approx(19.25f));

    REQUIRE(host
                ->execute(arc::editor::host_command_envelope{
                    .request_id = 3,
                    .payload =
                        arc::editor::host_apply_world_environment_preset_command{
                            .entity = found->entity, .preset = arc::editor::host_world_environment_preset::night}})
                .succeeded);
    REQUIRE(host->world_environment_snapshot(found->entity)->local_time_hours == Catch::Approx(23.0f));
    REQUIRE(
        host->query({.request_id = 4, .payload = arc::editor::host_world_environment_query{.entity = found->entity}})
            .succeeded);
}
