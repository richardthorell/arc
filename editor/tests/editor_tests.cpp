#include <arc/editor/arc_host.h>
#include <arc/editor/asset_drag_drop.h>
#include <arc/editor/editor_console.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/material_asset.h>
#include <arc/editor/material_library.h>
#include <arc/editor/sdl_events.h>
#include <arc/editor/world_environment_host.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <variant>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace
{

arc::editor::host_entity_id parse_entity_from_response(const std::string& line)
{
    arc::editor::host_entity_id entity;
    const auto index_pos = line.find("\"index\":");
    const auto generation_pos = line.find("\"generation\":");
    if (index_pos == std::string::npos || generation_pos == std::string::npos)
        return entity;

    std::sscanf(line.c_str() + index_pos, "\"index\":%u", &entity.index);
    std::sscanf(line.c_str() + generation_pos, "\"generation\":%u", &entity.generation);
    return entity;
}

} // namespace

TEST_CASE("editor viewport tracks size and focus state")
{
    arc::editor::editor_viewport viewport;
    REQUIRE_FALSE(viewport.valid());

    viewport.set_size(640.8f, 480.2f);
    viewport.set_focused(true);
    viewport.set_hovered(true);

    REQUIRE(viewport.valid());
    REQUIRE(viewport.width() == 640);
    REQUIRE(viewport.height() == 480);
    REQUIRE(viewport.focused());
    REQUIRE(viewport.hovered());

    viewport.set_screen_rect(100.0f, 50.0f, 320.0f, 240.0f);
    REQUIRE(viewport.width() == 320);
    REQUIRE(viewport.height() == 240);
    REQUIRE(viewport.contains_screen_point(100.0f, 50.0f));
    REQUIRE(viewport.contains_screen_point(419.0f, 289.0f));
    REQUIRE_FALSE(viewport.contains_screen_point(420.0f, 290.0f));
    REQUIRE(viewport.local_x(124.0f) == Catch::Approx(24.0f));
    REQUIRE(viewport.local_y(74.0f) == Catch::Approx(24.0f));
}

TEST_CASE("editor console sink captures bounded log records")
{
    arc::editor::editor_console_sink sink(2);

    sink.write({ .level = arc::log_level::info, .category = "one", .message = "first" });
    sink.write({ .level = arc::log_level::warn, .category = "two", .message = "second" });
    sink.write({ .level = arc::log_level::error, .category = "three", .message = "third" });

    const auto entries = sink.entries();
    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].category == "two");
    REQUIRE(entries[1].level == arc::log_level::error);
    REQUIRE(entries[1].message == "third");
}

TEST_CASE("SDL events translate to ARC events")
{
    SDL_Event source{};
    arc::event destination{};

    source.type = SDL_EVENT_WINDOW_RESIZED;
    source.window.data1 = 1280;
    source.window.data2 = 720;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::resized);
    REQUIRE(destination.width == 1280);
    REQUIRE(destination.height == 720);

    source = {};
    source.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
    source.button.button = SDL_BUTTON_LEFT;
    source.button.x = 10.0f;
    source.button.y = 20.0f;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::mouse_button_down);
    REQUIRE(destination.button == arc::mouse_button::left);
    REQUIRE(destination.x == 10);
    REQUIRE(destination.y == 20);

    source = {};
    source.type = SDL_EVENT_KEY_DOWN;
    source.key.key = SDLK_A;
    source.key.mod = SDL_KMOD_CTRL;
    source.key.repeat = true;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::key_down);
    REQUIRE(destination.key_code == SDLK_A);
    REQUIRE(destination.modifiers == SDL_KMOD_CTRL);
    REQUIRE(destination.repeat);
}

TEST_CASE("editor selection keeps one selected entity")
{
    arc::scene::registry scene;
    auto first = scene.create();
    auto second = scene.create();
    scene.emplace<arc::scene::selection_component>(first, true);

    arc::scene::entity selected = first;
    REQUIRE(arc::editor::select_entity(scene, second, selected));
    REQUIRE(selected == second);
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(first).selected);
    REQUIRE(scene.get<arc::scene::selection_component>(second).selected);

    arc::editor::clear_selection(scene, selected);
    REQUIRE_FALSE(selected.valid());
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(first).selected);
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(second).selected);
}

TEST_CASE("editor picking hits bounded entities")
{
    arc::scene::registry scene;
    const auto entity = scene.create();
    scene.emplace<arc::scene::transform_component>(entity);
    scene.emplace<arc::scene::bounds_component>(
        entity,
        arc::geometric::box3f{ arc::geometric::point3f{ -1.0f, -1.0f, -1.0f }, arc::geometric::point3f{ 1.0f, 1.0f, 1.0f } },
        arc::geometric::box3f{},
        true);

    const arc::editor::editor_ray ray{
        .origin = arc::math::vector3f{ 0.0f, 0.0f, 5.0f },
        .direction = arc::math::vector3f{ 0.0f, 0.0f, -1.0f }
    };
    REQUIRE(arc::editor::pick_bounded_entity(scene, ray) == entity);

    float distance{};
    REQUIRE(arc::editor::intersect_ray_box(
        ray,
        arc::geometric::box3f{ arc::geometric::point3f{ -1.0f, -1.0f, -1.0f }, arc::geometric::point3f{ 1.0f, 1.0f, 1.0f } },
        distance));
    REQUIRE(distance == Catch::Approx(4.0f));
}

TEST_CASE("editor camera controller orbits pans and zooms")
{
    arc::editor::editor_camera_controller camera;
    arc::scene::transform_component transform;

    camera.focus({ 0.0f, 0.0f, 0.0f }, 2.0f);
    const float focused_distance = camera.distance();
    camera.zoom(1.0f);
    REQUIRE(camera.distance() < focused_distance);

    camera.orbit(24.0f, -12.0f);
    camera.pan(10.0f, 5.0f);
    camera.apply_to(transform);
    REQUIRE(transform.dirty);
    REQUIRE(arc::math::length(transform.position) > 0.01f);
}

TEST_CASE("editor euler conversion keeps pure y rotation stable")
{
    const auto rotation = arc::editor::quaternion_from_euler_degrees({ 0.0f, 135.0f, 0.0f });
    const auto euler = arc::editor::euler_degrees_from_quaternion(rotation);

    REQUIRE(euler[0] == Catch::Approx(0.0f).margin(0.001f));
    REQUIRE(euler[1] == Catch::Approx(135.0f).margin(0.001f));
    REQUIRE(euler[2] == Catch::Approx(0.0f).margin(0.001f));
}

TEST_CASE("editor euler conversion round trips mixed rotation")
{
    const arc::math::vector3f input{ 20.0f, 135.0f, 10.0f };
    const auto rotation = arc::editor::quaternion_from_euler_degrees(input);
    const auto euler = arc::editor::euler_degrees_from_quaternion(rotation);

    REQUIRE(euler[0] == Catch::Approx(input[0]).margin(0.001f));
    REQUIRE(euler[1] == Catch::Approx(input[1]).margin(0.001f));
    REQUIRE(euler[2] == Catch::Approx(input[2]).margin(0.001f));
}

TEST_CASE("editor default sun rotation points downward at an angle")
{
    arc::scene::transform_component transform;
    transform.rotation = arc::editor::quaternion_from_euler_degrees({ -50.0f, -35.0f, 0.0f });

    const auto direction = arc::scene::forward_direction(transform);

    REQUIRE(direction[1] < -0.25f);
    REQUIRE(std::abs(direction[0]) > 0.10f);
    REQUIRE(direction[2] < -0.10f);
}

TEST_CASE("editor tool shortcuts update active tool")
{
    arc::input::input_manager input;
    input.bind_action("tool.select", { .device = arc::input::input_device_type::keyboard, .code = SDLK_Q });
    input.bind_action("tool.translate", { .device = arc::input::input_device_type::keyboard, .code = SDLK_W });
    input.bind_action("tool.rotate", { .device = arc::input::input_device_type::keyboard, .code = SDLK_E });
    input.bind_action("tool.scale", { .device = arc::input::input_device_type::keyboard, .code = SDLK_R });

    auto tool = arc::editor::editor_tool::select;
    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = SDLK_W });
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::translate);

    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = SDLK_R });
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::scale);
}

TEST_CASE("editor can add a selected primitive mesh entity")
{
    arc::editor::editor_scene_state scene;
    arc::render::renderer renderer;
    arc::render::material_desc imported_material;
    imported_material.name = "Imported Mesh Material";
    imported_material.base_color = { 1.0f, 0.72f, 0.05f, 1.0f };
    scene.default_material = renderer.create_material(imported_material);

    const auto entity = arc::editor::add_primitive_to_scene(scene, renderer, arc::editor::editor_primitive_type::plane);

    REQUIRE(scene.scene.alive(entity));
    REQUIRE(scene.selected_entity == entity);
    REQUIRE(scene.primitive_entities.size() == 1);
    REQUIRE(scene.scene.has<arc::scene::transform_component>(entity));
    REQUIRE(scene.scene.has<arc::scene::bounds_component>(entity));
    REQUIRE(scene.scene.has<arc::scene::mesh_renderer_component>(entity));
    REQUIRE(scene.scene.get<arc::scene::selection_component>(entity).selected);
    REQUIRE(scene.scene.get<arc::scene::mesh_renderer_component>(entity).mesh.valid());
    REQUIRE(scene.scene.get<arc::scene::mesh_renderer_component>(entity).material.valid());
    REQUIRE(scene.primitive_material.valid());
    REQUIRE(scene.primitive_material != scene.default_material);
    REQUIRE(scene.scene.get<arc::scene::mesh_renderer_component>(entity).material == scene.primitive_material);
}

TEST_CASE("arc host protocol serializes command and query envelopes")
{
    const arc::editor::host_entity_id entity{ .index = 7, .generation = 3 };
    const arc::editor::host_transform transform{
        .position = { 1.0f, 2.0f, 3.0f },
        .rotation = { 0.0f, 0.0f, 0.707f, 0.707f },
        .scale = { 2.0f, 2.0f, 2.0f }
    };

    const arc::editor::host_command_envelope commands[]{
        { .request_id = 1, .payload = arc::editor::host_open_project_command{ .name = "Protocol", .root = "D:/Protocol" } },
        { .request_id = 2, .payload = arc::editor::host_open_scene_command{ .path = "D:/Protocol/assets/test.glb", .append = true } },
        { .request_id = 3, .payload = arc::editor::host_create_entity_command{ .kind = arc::editor::host_create_entity_kind::cube } },
        { .request_id = 4, .payload = arc::editor::host_select_entity_command{ .entity = entity } },
        { .request_id = 5, .payload = arc::editor::host_rename_entity_command{ .entity = entity, .name = "Renamed" } },
        { .request_id = 6, .payload = arc::editor::host_delete_entity_command{ .entity = entity } },
        { .request_id = 7, .payload = arc::editor::host_set_transform_command{ .entity = entity, .transform = transform } },
        { .request_id = 8, .payload = arc::editor::host_viewport_attach_command{ .native_handle = 1234, .x = 16, .y = 32, .width = 1280, .height = 720 } },
        { .request_id = 9, .payload = arc::editor::host_viewport_resize_command{ .x = 24, .y = 48, .width = 640, .height = 360 } },
        { .request_id = 10, .payload = arc::editor::host_viewport_set_camera_mode_command{ .projection = arc::editor::host_camera_projection::orthographic } },
        { .request_id = 11, .payload = arc::editor::host_viewport_set_render_options_command{
            .render_mode = arc::editor::host_render_mode::wireframe,
            .visualization = arc::editor::host_visualization_mode::world_normal,
            .overlay = arc::editor::host_overlay_mode::all_wireframe,
            .shadows = false } },
        { .request_id = 12, .payload = arc::editor::host_viewport_camera_input_command{ .orbit_x = 4.0f, .orbit_y = -2.0f, .zoom = 1.0f } },
        { .request_id = 13, .payload = arc::editor::host_set_world_environment_command{
            .environment = { .entity = entity, .sky_source = arc::editor::host_sky_source::solid_color } } },
        { .request_id = 14, .payload = arc::editor::host_apply_world_environment_preset_command{
            .entity = entity, .preset = arc::editor::host_world_environment_preset::night } },
        { .request_id = 15, .payload = arc::editor::host_set_environment_hdri_command{
            .entity = entity, .path = "assets/environment/studio.hdr" } }
    };

    for (const auto& command : commands)
    {
        const auto json = arc::editor::to_json(command);
        arc::editor::host_command_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(json, parsed, error));
        REQUIRE(parsed.request_id == command.request_id);
        REQUIRE(parsed.command_type == arc::editor::command_type(command.payload));
    }

    const arc::editor::host_query_envelope queries[]{
        { .request_id = 16, .payload = arc::editor::host_scene_hierarchy_query{} },
        { .request_id = 17, .payload = arc::editor::host_selected_entity_query{} },
        { .request_id = 18, .payload = arc::editor::host_project_assets_query{} },
        { .request_id = 19, .payload = arc::editor::host_viewport_state_query{} },
        { .request_id = 20, .payload = arc::editor::host_world_environment_query{ .entity = entity } }
    };

    for (const auto& query : queries)
    {
        const auto json = arc::editor::to_json(query);
        arc::editor::host_query_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(json, parsed, error));
        REQUIRE(parsed.request_id == query.request_id);
        REQUIRE(parsed.query_type == arc::editor::query_type(query.payload));
    }
}

TEST_CASE("world environment JSON round trips every field and enum")
{
    arc::editor::host_world_environment_snapshot environment;
    environment.entity = { 17, 4 };
    environment.enabled = false;
    environment.sky_visible = false;
    environment.affect_lighting = false;
    environment.sky_source = arc::editor::host_sky_source::hdri;
    environment.solid_color = { 0.11f, 0.22f, 0.33f };
    environment.hdri_path = "environments/night.hdr";
    environment.hdri_rotation_degrees = 37.0f;
    environment.radiance_intensity = 1.25f;
    environment.planet_radius = 6001.0f;
    environment.atmosphere_radius = 6102.0f;
    environment.rayleigh_strength = 1.1f;
    environment.mie_strength = 0.2f;
    environment.ozone_strength = 0.3f;
    environment.atmosphere_tint = { 0.44f, 0.55f, 0.66f };
    environment.ground_albedo = { 0.12f, 0.13f, 0.14f };
    environment.mie_anisotropy = 0.7f;
    environment.rayleigh_scale_height = 7.0f;
    environment.mie_scale_height = 2.0f;
    environment.multi_scattering_factor = 0.8f;
    environment.exposure = 1.4f;
    environment.sun_disk_size = 0.03f;
    environment.sun_disk_intensity = 2.0f;
    environment.sun_mode = arc::editor::host_sun_position_mode::geographic;
    environment.time_mode = arc::editor::host_celestial_time_mode::system_clock;
    environment.latitude_degrees = -12.5f;
    environment.longitude_degrees = 130.25f;
    environment.north_offset_degrees = 15.0f;
    environment.year = 2032;
    environment.month = 2;
    environment.day = 29;
    environment.local_time_hours = 21.25f;
    environment.utc_offset_hours = -7.0f;
    environment.playing = true;
    environment.loop_day = false;
    environment.time_scale = 120.0f;
    environment.automatic_sun_light = false;
    environment.sun_intensity_multiplier = 0.75f;
    environment.sun_temperature_multiplier = 1.2f;
    environment.moon_enabled = false;
    environment.automatic_moon_phase = false;
    environment.moon_phase = 0.45f;
    environment.moon_intensity = 0.4f;
    environment.moon_angular_radius_degrees = 0.31f;
    environment.stars_enabled = false;
    environment.star_density = 0.5f;
    environment.star_intensity = 1.5f;
    environment.star_twinkle = 0.2f;
    environment.clouds_enabled = false;
    environment.cloud_shadows = false;
    environment.cumulus = { false, 0.1f, 0.2f, 1000.0f, 200.0f, 0.3f, 0.4f, 0.5f,
        -1.0f, 0.25f, 3.0f, 0.6f, 0.7f };
    environment.cirrus = { true, 0.8f, 0.7f, 7000.0f, 300.0f, 0.6f, 0.5f, 0.4f,
        0.5f, -0.5f, 9.0f, 0.3f, 0.2f };
    environment.fog_enabled = false;
    environment.fog_color = { 0.15f, 0.25f, 0.35f };
    environment.fog_density = 0.02f;
    environment.fog_height_falloff = 0.3f;
    environment.fog_start_distance = 12.0f;
    environment.fog_max_opacity = 0.6f;
    environment.fog_sun_scattering = 0.4f;
    environment.lighting_enabled = false;
    environment.lighting_source = arc::editor::host_environment_lighting_source::constant_color;
    environment.lighting_color = { 0.2f, 0.3f, 0.4f };
    environment.diffuse_intensity = 0.9f;
    environment.specular_intensity = 0.8f;

    const arc::editor::host_command_envelope command{
        .request_id = 42,
        .payload = arc::editor::host_set_world_environment_command{ environment }
    };
    const auto json = arc::editor::to_json(command);
    arc::editor::host_command_envelope parsed;
    std::string error;
    REQUIRE(arc::editor::from_json(json, parsed, error));
    const auto& round_trip = std::get<arc::editor::host_set_world_environment_command>(parsed.payload);
    REQUIRE(round_trip.environment == environment);
}

TEST_CASE("arc host executes scene commands and exposes snapshots")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    arc::editor::editor_asset_state assets;
    const auto opened = host->open_project(
        { .name = "Host Test", .root = std::filesystem::temp_directory_path() },
        assets);
    REQUIRE(opened.succeeded);

    const auto created = host->execute(arc::editor::host_command_envelope{
        .request_id = 1,
        .payload = arc::editor::host_create_entity_command{ .kind = arc::editor::host_create_entity_kind::cube } });
    REQUIRE(created.succeeded);
    const auto created_entity = host->selected_entity_snapshot().entity;
    REQUIRE(created_entity.valid());

    const auto renamed = host->execute(arc::editor::host_command_envelope{
        .request_id = 2,
        .payload = arc::editor::host_rename_entity_command{
            .entity = created_entity,
            .name = "Command Cube" } });
    REQUIRE(renamed.succeeded);

    const auto selected = host->selected_entity_snapshot();
    REQUIRE(selected.entity == created_entity);
    REQUIRE(selected.name == "Command Cube");
    REQUIRE(selected.transform.has_value());
    REQUIRE(std::any_of(selected.components.begin(), selected.components.end(), [](const auto& component) {
        return component.kind == arc::editor::host_component_kind::transform;
    }));

    auto transform = *selected.transform;
    transform.position = { 2.0f, 3.0f, 4.0f };
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .request_id = 3,
        .payload = arc::editor::host_set_transform_command{
            .entity = created_entity,
            .transform = transform } }).succeeded);
    REQUIRE(host->selected_entity_snapshot().transform->position.x == Catch::Approx(2.0f));

    const auto snapshot = host->scene_snapshot();
    REQUIRE(std::any_of(snapshot.entities.begin(), snapshot.entities.end(), [&](const auto& entity) {
        return entity.entity == created_entity && entity.name == "Command Cube" && entity.selected;
    }));

    REQUIRE(host->query({ .request_id = 4, .payload = arc::editor::host_scene_hierarchy_query{} }).succeeded);
    REQUIRE(host->query({ .request_id = 5, .payload = arc::editor::host_project_assets_query{} }).succeeded);
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .request_id = 6,
        .payload = arc::editor::host_delete_entity_command{ .entity = created_entity } }).succeeded);
    REQUIRE_FALSE(host->selected_entity_snapshot().entity.valid());

    const auto events = host->poll_events();
    REQUIRE(std::any_of(events.begin(), events.end(), [](const auto& event) {
        return event.event_type == arc::editor::host_event_type::entity_created;
    }));
    REQUIRE(std::any_of(events.begin(), events.end(), [](const auto& event) {
        return event.event_type == arc::editor::host_event_type::entity_deleted;
    }));
}

TEST_CASE("selected camera snapshots and entity-specific edits round trip atomically")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project(
        { .name = "Inspector Camera Test", .root = std::filesystem::temp_directory_path() },
        assets).succeeded);
    const auto game_camera = host->scene_state().game_camera_entity;
    const auto editor_camera = host->scene_state().camera_entity;
    const arc::editor::host_entity_id game_camera_id{ game_camera.index, game_camera.generation };

    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .request_id = 1,
        .payload = arc::editor::host_select_entity_command{ .entity = game_camera_id } }).succeeded);

    const auto selected = host->selected_entity_snapshot();
    REQUIRE(selected.entity == game_camera_id);
    REQUIRE(selected.camera.has_value());
    REQUIRE(selected.render_layer_mask == arc::editor::host_default_render_layer);
    REQUIRE(std::any_of(selected.components.begin(), selected.components.end(), [](const auto& component) {
        return component.kind == arc::editor::host_component_kind::camera && component.editable;
    }));

    const auto editor_before = host->scene_state().scene.get<arc::scene::camera_component>(editor_camera);
    auto updated = *selected.camera;
    updated.projection = arc::editor::host_camera_projection::orthographic;
    updated.fov_y_degrees = 72.0f;
    updated.orthographic_height = 24.0f;
    updated.near_plane = 0.25f;
    updated.far_plane = 4096.0f;
    updated.active = true;
    updated.clear_color = { 0.1f, 0.2f, 0.3f, 0.8f };
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .request_id = 2,
        .payload = arc::editor::host_set_camera_command{
            .entity = game_camera_id,
            .camera = updated } }).succeeded);

    const auto round_trip = host->selected_entity_snapshot();
    REQUIRE(round_trip.camera.has_value());
    REQUIRE(*round_trip.camera == updated);
    const auto& editor_after = host->scene_state().scene.get<arc::scene::camera_component>(editor_camera);
    REQUIRE(editor_after.projection == editor_before.projection);
    REQUIRE(editor_after.fov_y_radians == Catch::Approx(editor_before.fov_y_radians));
    REQUIRE(editor_after.near_plane == Catch::Approx(editor_before.near_plane));

    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .request_id = 3,
        .payload = arc::editor::host_set_render_layer_command{
            .entity = game_camera_id,
            .render_layer_mask = arc::editor::host_environment_render_layer } }).succeeded);
    REQUIRE(host->selected_entity_snapshot().render_layer_mask == arc::editor::host_environment_render_layer);

    const auto confirmed = *host->selected_entity_snapshot().camera;
    for (const auto invalid : {
        arc::editor::host_camera_snapshot{ confirmed.projection, 1.0f, confirmed.orthographic_height, confirmed.near_plane, confirmed.far_plane, confirmed.active, confirmed.clear_color },
        arc::editor::host_camera_snapshot{ confirmed.projection, confirmed.fov_y_degrees, 0.0f, confirmed.near_plane, confirmed.far_plane, confirmed.active, confirmed.clear_color },
        arc::editor::host_camera_snapshot{ confirmed.projection, confirmed.fov_y_degrees, confirmed.orthographic_height, 0.0f, confirmed.far_plane, confirmed.active, confirmed.clear_color },
        arc::editor::host_camera_snapshot{ confirmed.projection, confirmed.fov_y_degrees, confirmed.orthographic_height, confirmed.near_plane, confirmed.near_plane, confirmed.active, confirmed.clear_color },
        arc::editor::host_camera_snapshot{ confirmed.projection, std::numeric_limits<float>::infinity(), confirmed.orthographic_height, confirmed.near_plane, confirmed.far_plane, confirmed.active, confirmed.clear_color },
        arc::editor::host_camera_snapshot{ confirmed.projection, confirmed.fov_y_degrees, confirmed.orthographic_height, confirmed.near_plane, confirmed.far_plane, confirmed.active, { -0.1f, 0.2f, 0.3f, 1.0f } }
    })
    {
        REQUIRE_FALSE(host->execute(arc::editor::host_command_envelope{
            .request_id = 4,
            .payload = arc::editor::host_set_camera_command{
                .entity = game_camera_id,
                .camera = invalid } }).succeeded);
        REQUIRE(*host->selected_entity_snapshot().camera == confirmed);
    }

    const auto json = arc::editor::to_json(host->selected_entity_snapshot());
    REQUIRE(json.find("\"camera\":{") != std::string::npos);
    REQUIRE(json.find("\"renderLayerMask\":2") != std::string::npos);
}

TEST_CASE("camera and render layer JSON commands preserve their typed payloads")
{
    const arc::editor::host_entity_id entity{ 12, 4 };
    arc::editor::host_camera_snapshot camera;
    camera.fov_y_degrees = 80.0f;
    camera.near_plane = 0.5f;
    camera.far_plane = 5000.0f;
    camera.clear_color = { 0.2f, 0.3f, 0.4f, 1.0f };

    const std::array<arc::editor::host_command_payload, 2> payloads{
        arc::editor::host_command_payload{ arc::editor::host_set_camera_command{ .entity = entity, .camera = camera } },
        arc::editor::host_command_payload{ arc::editor::host_set_render_layer_command{ .entity = entity, .render_layer_mask = 2u } }
    };
    for (const auto& payload : payloads)
    {
        const arc::editor::host_command_envelope source{ .request_id = 91, .payload = payload };
        arc::editor::host_command_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(arc::editor::to_json(source), parsed, error));
        REQUIRE(parsed.request_id == 91);
        REQUIRE(arc::editor::command_type(parsed.payload) == arc::editor::command_type(payload));
    }
}

TEST_CASE("arc host resolves a project assets directory for protocol-opened projects")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-host-project-assets-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "assets", ec);
    REQUIRE_FALSE(ec);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    const auto response = host->execute(arc::editor::host_command_envelope{
        .request_id = 1,
        .payload = arc::editor::host_open_project_command{
            .name = "Asset Root Test",
            .root = root } });

    REQUIRE(response.succeeded);
    REQUIRE(host->project_assets_snapshot().asset_root == root / "assets");
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("world environment host snapshots round trip every settings group and preserve runtime handles")
{
    arc::scene::world_environment_settings settings;
    settings.world.hdri_texture = { .index = 1, .generation = 2 };
    settings.celestial.sun_light = { .index = 3, .generation = 4 };
    settings.celestial.animation_time_seconds = 91.0f;
    settings.lighting.environment = { .index = 5, .generation = 6 };
    settings.lighting.hdri_texture = { .index = 7, .generation = 8 };
    const arc::editor::host_entity_id entity{ 12, 2 };
    auto snapshot = arc::editor::to_host_world_environment_snapshot(
        entity, settings, "environments/studio.hdr");

    snapshot.enabled = false;
    snapshot.sky_visible = false;
    snapshot.sky_source = arc::editor::host_sky_source::solid_color;
    snapshot.solid_color = { 0.1f, 0.2f, 0.3f };
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

    const auto round_trip = arc::editor::to_host_world_environment_snapshot(
        entity, converted, snapshot.hdri_path);
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
    REQUIRE(host->open_project({ .name = "Environment Host Test", .root = {} }, assets).succeeded);

    const auto hierarchy = host->scene_snapshot();
    const auto found = std::find_if(hierarchy.entities.begin(), hierarchy.entities.end(), [](const auto& entity) {
        return entity.kind == arc::editor::host_entity_kind::environment;
    });
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
        .request_id = 1,
        .payload = arc::editor::host_set_world_environment_command{ .environment = edited } });
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
        .payload = arc::editor::host_set_world_environment_command{ .environment = edited } }).succeeded);
    REQUIRE(host->world_environment_snapshot(found->entity)->local_time_hours == Catch::Approx(19.25f));

    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .request_id = 3,
        .payload = arc::editor::host_apply_world_environment_preset_command{
            .entity = found->entity,
            .preset = arc::editor::host_world_environment_preset::night } }).succeeded);
    REQUIRE(host->world_environment_snapshot(found->entity)->local_time_hours == Catch::Approx(23.0f));
    REQUIRE(host->query({ .request_id = 4, .payload = arc::editor::host_world_environment_query{
        .entity = found->entity } }).succeeded);
}

TEST_CASE("arc host process speaks newline delimited json over stdio")
{
#if defined(_WIN32) && defined(ARC_HOST_PROCESS_PATH)
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;

    HANDLE child_stdin_read = nullptr;
    HANDLE child_stdin_write = nullptr;
    HANDLE child_stdout_read = nullptr;
    HANDLE child_stdout_write = nullptr;
    REQUIRE(CreatePipe(&child_stdin_read, &child_stdin_write, &security, 0));
    REQUIRE(CreatePipe(&child_stdout_read, &child_stdout_write, &security, 0));
    REQUIRE(SetHandleInformation(child_stdin_write, HANDLE_FLAG_INHERIT, 0));
    REQUIRE(SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0));

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = child_stdin_read;
    startup.hStdOutput = child_stdout_write;
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION process{};
    std::string command_line = "\"" ARC_HOST_PROCESS_PATH "\"";
    REQUIRE(CreateProcessA(
        nullptr,
        command_line.data(),
        nullptr,
        nullptr,
        TRUE,
        0,
        nullptr,
        nullptr,
        &startup,
        &process));

    CloseHandle(child_stdin_read);
    CloseHandle(child_stdout_write);

    const auto read_line = [&]() {
        std::string line;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline)
        {
            DWORD available = 0;
            REQUIRE(PeekNamedPipe(child_stdout_read, nullptr, 0, nullptr, &available, nullptr));
            if (available == 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            char ch{};
            DWORD read = 0;
            REQUIRE(ReadFile(child_stdout_read, &ch, 1, &read, nullptr));
            if (read == 0)
                continue;
            if (ch == '\n')
                return line;
            if (ch != '\r')
                line.push_back(ch);
        }
        return line;
    };

    std::vector<std::string> observed_events;
    const auto request = [&](std::uint64_t request_id, const std::string& json) {
        const std::string line = json + '\n';
        DWORD written = 0;
        REQUIRE(WriteFile(child_stdin_write, line.data(), static_cast<DWORD>(line.size()), &written, nullptr));
        REQUIRE(written == line.size());

        for (;;)
        {
            auto response = read_line();
            REQUIRE_FALSE(response.empty());
            if (response.find("\"kind\":\"response\"") != std::string::npos &&
                response.find("\"requestId\":" + std::to_string(request_id)) != std::string::npos)
            {
                REQUIRE(response.find("\"succeeded\":true") != std::string::npos);
                return response;
            }
            observed_events.push_back(std::move(response));
        }
    };

    request(1, arc::editor::to_json(arc::editor::host_command_envelope{
        .request_id = 1,
        .payload = arc::editor::host_open_project_command{
            .name = "Process Test",
            .root = std::filesystem::temp_directory_path() } }));

    const auto create_response = request(2, arc::editor::to_json(arc::editor::host_command_envelope{
        .request_id = 2,
        .payload = arc::editor::host_create_entity_command{ .kind = arc::editor::host_create_entity_kind::cube } }));
    const auto created_entity = parse_entity_from_response(create_response);
    REQUIRE(created_entity.valid());

    const auto hierarchy_response = request(3, arc::editor::to_json(arc::editor::host_query_envelope{
        .request_id = 3,
        .payload = arc::editor::host_scene_hierarchy_query{} }));
    REQUIRE(hierarchy_response.find("\"entities\"") != std::string::npos);

    request(4, arc::editor::to_json(arc::editor::host_command_envelope{
        .request_id = 4,
        .payload = arc::editor::host_rename_entity_command{
            .entity = created_entity,
            .name = "Process Cube" } }));

    const auto renamed_hierarchy = request(5, arc::editor::to_json(arc::editor::host_query_envelope{
        .request_id = 5,
        .payload = arc::editor::host_scene_hierarchy_query{} }));
    REQUIRE(renamed_hierarchy.find("Process Cube") != std::string::npos);

    request(6, arc::editor::to_json(arc::editor::host_command_envelope{
        .request_id = 6,
        .payload = arc::editor::host_close_project_command{} }));

    CloseHandle(child_stdin_write);
    WaitForSingleObject(process.hProcess, 5000);
    CloseHandle(child_stdout_read);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);

    REQUIRE(std::any_of(observed_events.begin(), observed_events.end(), [](const auto& event) {
        return event.find("\"type\":\"entity.created\"") != std::string::npos;
    }));
#else
    SUCCEED("arc_host_process integration is enabled on Windows builds with ARC_HOST_PROCESS_PATH");
#endif
}

TEST_CASE("editor material assets load save and round trip")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_asset_tests";
    std::filesystem::create_directories(root / "materials");

    auto asset = arc::editor::make_default_material_asset("Bronze");
    asset.path = root / "materials" / "bronze.arcmat";
    asset.material.base_color = { 0.8f, 0.42f, 0.18f, 1.0f };
    asset.material.metallic = 0.75f;
    asset.material.roughness = 0.32f;
    asset.material.normal_scale = 0.85f;
    asset.textures.base_color = "textures/bronze_base.png";
    asset.textures.normal = "textures/bronze_n.png";

    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));

    arc::editor::material_asset loaded;
    REQUIRE(arc::editor::load_material_asset(asset.path, root, loaded, message));
    REQUIRE(loaded.name == "Bronze");
    REQUIRE(loaded.shader == "arc/default_phong");
    REQUIRE(loaded.material.base_color[0] == Catch::Approx(0.8f));
    REQUIRE(loaded.material.metallic == Catch::Approx(0.75f));
    REQUIRE(loaded.material.roughness == Catch::Approx(0.32f));
    REQUIRE(loaded.material.normal_scale == Catch::Approx(0.85f));
    REQUIRE(loaded.textures.base_color == "textures/bronze_base.png");
    REQUIRE(loaded.textures.normal == "textures/bronze_n.png");
    REQUIRE(loaded.graph_reserved);

    const auto resolved = arc::editor::resolve_material_texture_path(root, loaded.textures.base_color);
    REQUIRE(resolved.lexically_normal() == (root / "textures" / "bronze_base.png").lexically_normal());
}

TEST_CASE("editor material assets tolerate missing and future fields")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_asset_defaults";
    std::filesystem::create_directories(root);
    const auto path = root / "future.arcmat";

    {
        std::ofstream stream(path, std::ios::binary);
        stream << R"({
  "version": 1,
  "name": "Future",
  "unknownFutureBlock": { "enabled": true },
  "surface": { "metallic": 0.2 },
  "textures": { "emissive": "textures/glow.png" },
  "graph": null
})";
    }

    arc::editor::material_asset loaded;
    std::string message;
    REQUIRE(arc::editor::load_material_asset(path, root, loaded, message));
    REQUIRE(loaded.name == "Future");
    REQUIRE(loaded.shader == "arc/default_phong");
    REQUIRE(loaded.material.metallic == Catch::Approx(0.2f));
    REQUIRE(loaded.material.roughness == Catch::Approx(0.62f));
    REQUIRE(loaded.textures.emissive == "textures/glow.png");
    REQUIRE(arc::editor::is_material_asset_path(path));
    REQUIRE_FALSE(arc::editor::is_material_asset_path(root / "mesh.glb"));
}

TEST_CASE("editor material library applies materials to selected mesh renderer")
{
    arc::scene::registry scene;
    const auto selected = scene.create();
    scene.emplace<arc::scene::mesh_renderer_component>(selected);

    const arc::render::material_handle material{ .index = 42, .generation = 7 };
    REQUIRE(arc::editor::apply_material_to_selected(scene, selected, material));
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(selected).material == material);

    const auto empty = scene.create();
    REQUIRE_FALSE(arc::editor::apply_material_to_selected(scene, empty, material));
    REQUIRE_FALSE(arc::editor::apply_material_to_selected(scene, {}, material));
}

TEST_CASE("editor asset drag payloads classify and stay asset relative")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_payload_assets";

    REQUIRE(arc::editor::classify_asset_path(root / "materials" / "matte.arcmat") == arc::editor::editor_asset_kind::material);
    REQUIRE(arc::editor::classify_asset_path(root / "textures" / "albedo.png") == arc::editor::editor_asset_kind::texture);
    REQUIRE(arc::editor::classify_asset_path(root / "textures" / "normal.dds") == arc::editor::editor_asset_kind::texture);
    REQUIRE(arc::editor::classify_asset_path(root / "textures" / "studio.hdr") == arc::editor::editor_asset_kind::environment);
    REQUIRE(arc::editor::classify_asset_path(root / "models" / "import.fbx") == arc::editor::editor_asset_kind::scene);
    REQUIRE(arc::editor::classify_asset_path(root / "models" / "level.glb") == arc::editor::editor_asset_kind::scene);
    REQUIRE(arc::editor::is_texture_asset_kind(arc::editor::editor_asset_kind::texture));
    REQUIRE(arc::editor::is_texture_asset_kind(arc::editor::editor_asset_kind::environment));

    arc::editor::editor_asset_payload payload;
    REQUIRE(arc::editor::make_asset_payload(root, root / "materials" / "matte.arcmat", false, payload));
    REQUIRE(payload.kind == arc::editor::editor_asset_kind::material);
    REQUIRE(payload.relative_path.generic_string() == "materials/matte.arcmat");

    REQUIRE(arc::editor::asset_relative_path(root, root / "textures" / "albedo.jpg").generic_string() == "textures/albedo.jpg");
    REQUIRE(arc::editor::is_texture_asset_path(root / "textures" / "normal.dds"));
}

TEST_CASE("editor material texture slots accept texture assets and reject wrong types")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_slot_tests";
    arc::editor::material_editor_state editor;
    editor.open = true;
    editor.working = arc::editor::make_default_material_asset("Slot Test");

    std::string message;
    REQUIRE(arc::editor::assign_texture_to_material_slot(
        editor,
        arc::editor::material_texture_slot::base_color,
        root,
        root / "textures" / "base.png",
        &message));
    REQUIRE(editor.dirty);
    REQUIRE(editor.working.textures.base_color == "textures/base.png");

    REQUIRE(arc::editor::assign_texture_to_material_slot(
        editor,
        arc::editor::material_texture_slot::normal,
        root,
        std::filesystem::path{ "textures/normal.dds" },
        &message));
    REQUIRE(editor.working.textures.normal == "textures/normal.dds");

    REQUIRE_FALSE(arc::editor::assign_texture_to_material_slot(
        editor,
        arc::editor::material_texture_slot::ao,
        root,
        std::filesystem::path{ "materials/not_a_texture.arcmat" },
        &message));
}

TEST_CASE("editor material library reuses material handles and saves live updates")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_material_library_reuse";
    std::filesystem::create_directories(root / "materials");
    const auto path = root / "materials" / "reused.arcmat";

    auto asset = arc::editor::make_default_material_asset("Reusable");
    asset.path = path;
    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));

    arc::render::renderer renderer;
    arc::editor::editor_material_library library;
    const auto first = arc::editor::load_material_for_editor(library, renderer, root, path);
    const auto second = arc::editor::load_material_for_editor(library, renderer, root, path);
    REQUIRE(first.valid());
    REQUIRE(first == second);
    renderer.frame_queue().commit(1);

    arc::editor::material_editor_state editor;
    REQUIRE(arc::editor::open_material_editor(editor, library, renderer, root, path, message));
    const auto opened = editor.material;
    renderer.frame_queue().commit(2);

    editor.working.material.roughness = 0.21f;
    editor.dirty = true;
    REQUIRE(arc::editor::save_material_editor(editor, library, renderer, root, message));
    REQUIRE(editor.material == opened);
    const auto packet = renderer.frame_queue().commit(3);
    REQUIRE_FALSE(packet.events.empty());
    REQUIRE(packet.events.back().type() == arc::render::render_event_type::material_upload);
    const auto& upload = std::get<arc::render::material_upload_event>(packet.events.back().payload);
    REQUIRE(upload.handle == opened);
    REQUIRE(upload.material->roughness == Catch::Approx(0.21f));
}

TEST_CASE("editor viewport material drop applies to hit entity and ignores misses")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_viewport_material_drop";
    std::filesystem::create_directories(root / "materials");
    const auto path = root / "materials" / "drop.arcmat";

    auto asset = arc::editor::make_default_material_asset("Drop Material");
    asset.path = path;
    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));

    arc::render::renderer renderer;
    arc::editor::editor_material_library library;
    arc::scene::registry scene;
    const auto entity = scene.create();
    scene.emplace<arc::scene::transform_component>(entity);
    scene.emplace<arc::scene::mesh_renderer_component>(entity);
    scene.emplace<arc::scene::bounds_component>(
        entity,
        arc::geometric::box3f{ arc::geometric::point3f{ -1.0f, -1.0f, -1.0f }, arc::geometric::point3f{ 1.0f, 1.0f, 1.0f } },
        arc::geometric::box3f{},
        true);

    arc::scene::entity selected{};
    const arc::editor::editor_ray hit_ray{
        .origin = arc::math::vector3f{ 0.0f, 0.0f, 5.0f },
        .direction = arc::math::vector3f{ 0.0f, 0.0f, -1.0f }
    };
    const auto hit = arc::editor::apply_material_asset_to_viewport_hit(
        library,
        renderer,
        root,
        std::filesystem::path{ "materials/drop.arcmat" },
        scene,
        hit_ray,
        selected,
        &message);
    REQUIRE(hit == entity);
    REQUIRE(selected == entity);
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(entity).material.valid());

    const auto assigned = scene.get<arc::scene::mesh_renderer_component>(entity).material;
    const arc::editor::editor_ray miss_ray{
        .origin = arc::math::vector3f{ 4.0f, 4.0f, 5.0f },
        .direction = arc::math::vector3f{ 0.0f, 0.0f, -1.0f }
    };
    const auto missed = arc::editor::apply_material_asset_to_viewport_hit(
        library,
        renderer,
        root,
        std::filesystem::path{ "materials/drop.arcmat" },
        scene,
        miss_ray,
        selected,
        &message);
    REQUIRE_FALSE(missed.valid());
    REQUIRE(selected == entity);
    REQUIRE(scene.get<arc::scene::mesh_renderer_component>(entity).material == assigned);
}
