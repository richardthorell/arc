#include <arc/editor/arc_host.h>
#include <arc/editor/editor_console.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/material_asset.h>
#include <arc/editor/material_library.h>
#include <arc/editor/material_preview.h>
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

class pick_test_backend final : public arc::render::render_backend
{
public:
    arc::render::render_backend_type type() const noexcept override { return arc::render::render_backend_type::vulkan; }
    const arc::render::render_capabilities& capabilities() const noexcept override { return capabilities_; }
    arc::render::render_submit_result submit(
        const arc::render::render_frame_packet&, const arc::render::compiled_render_graph&) override
    {
        return { .submitted = true };
    }
    void request_object_pick(arc::render::render_object_pick_request request) override { request_ = request; }
    arc::render::render_object_pick_result last_object_pick() const override { return result; }

    arc::render::render_object_pick_request request_{};
    arc::render::render_object_pick_result result{};

private:
    arc::render::render_capabilities capabilities_{
        .api_major = 1,
        .api_minor = 2,
        .graphics_queue = true,
        .compute_queue = true,
        .presentation = true
    };
};

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

TEST_CASE("editor gizmos keep constant screen size and hit test colored axes")
{
    arc::scene::registry registry;
    const auto camera_entity = registry.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = { 0.0f, 0.0f, 5.0f };
    registry.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    registry.emplace<arc::scene::camera_component>(camera_entity);
    const auto selected = registry.create();
    registry.emplace<arc::scene::transform_component>(selected);
    registry.emplace<arc::scene::bounds_component>(selected,
        arc::geometric::box3f{ arc::geometric::point3f{ -0.5f, -0.5f, -0.5f },
            arc::geometric::point3f{ 0.5f, 0.5f, 0.5f } });
    arc::scene::update_world_transforms(registry);

    const arc::editor::editor_gizmo_context context{
        .tool = arc::editor::editor_tool::translate,
        .viewport_width = 800,
        .viewport_height = 600
    };
    const auto overlay = arc::editor::build_editor_gizmo_overlay(registry, selected, camera_entity, context);
    REQUIRE(overlay.lines.size() == 15);
    REQUIRE(overlay.lines[12].color[0] > overlay.lines[12].color[1]);
    REQUIRE(arc::editor::hit_test_editor_gizmo(registry, selected, camera_entity, context, 450.0f, 300.0f) ==
        arc::editor::gizmo_axis::x);
    REQUIRE(arc::editor::hit_test_editor_gizmo(registry, selected, camera_entity, context, 780.0f, 580.0f) ==
        arc::editor::gizmo_axis::none);

    const auto& camera = registry.get<arc::scene::camera_component>(camera_entity);
    const float near_scale = arc::editor::editor_gizmo_world_scale(camera,
        registry.get<arc::scene::transform_component>(camera_entity), {}, 600);
    registry.get<arc::scene::transform_component>(camera_entity).set_position({ 0.0f, 0.0f, 10.0f });
    arc::scene::update_world_transforms(registry);
    const float far_scale = arc::editor::editor_gizmo_world_scale(camera,
        registry.get<arc::scene::transform_component>(camera_entity), {}, 600);
    REQUIRE(far_scale == Catch::Approx(near_scale * 2.0f));
}

TEST_CASE("editor picking hits bounded entities")
{
    arc::scene::registry scene;
    const auto terrain = scene.create();
    scene.emplace<arc::scene::transform_component>(terrain);
    scene.emplace<arc::scene::terrain_component>(terrain);
    scene.emplace<arc::scene::bounds_component>(
        terrain,
        arc::geometric::box3f{ arc::geometric::point3f{ -100.0f, -100.0f, -100.0f },
            arc::geometric::point3f{ 100.0f, 100.0f, 100.0f } },
        arc::geometric::box3f{},
        true);
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

    scene.emplace<arc::scene::active_component>(entity, false);
    REQUIRE(arc::editor::pick_bounded_entity(scene, ray) == terrain);

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

    camera.orbit(40.0f, 0.0f);
    camera.apply_to(transform);
    const auto right_after_yaw = arc::math::rotate(transform.rotation, arc::math::vector3f{ 1.0f, 0.0f, 0.0f });
    REQUIRE(right_after_yaw[1] == Catch::Approx(0.0f).margin(0.00001f));
    camera.orbit(0.0f, 30.0f);
    camera.apply_to(transform);
    const auto right_after_pitch = arc::math::rotate(transform.rotation, arc::math::vector3f{ 1.0f, 0.0f, 0.0f });
    REQUIRE(arc::math::dot(right_after_yaw, right_after_pitch) == Catch::Approx(1.0f).margin(0.00001f));
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
    input.bind_action("tool.select", { .device = arc::input::input_device_type::keyboard, .code = 'Q' });
    input.bind_action("tool.translate", { .device = arc::input::input_device_type::keyboard, .code = 'W' });
    input.bind_action("tool.rotate", { .device = arc::input::input_device_type::keyboard, .code = 'E' });
    input.bind_action("tool.scale", { .device = arc::input::input_device_type::keyboard, .code = 'R' });

    auto tool = arc::editor::editor_tool::select;
    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = 'W' });
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::translate);

    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = 'R' });
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
        { .request_id = 3, .payload = arc::editor::host_create_entity_command{
            .kind = arc::editor::host_create_entity_kind::empty, .parent = entity } },
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
            .entity = entity, .path = "assets/environment/studio.hdr" } },
        { .request_id = 16, .payload = arc::editor::host_set_mesh_renderer_command{
            .entity = entity, .visible = false, .base_color_tint = { 0.8f, 0.7f, 0.6f, 1.0f } } },
        { .request_id = 17, .payload = arc::editor::host_set_entity_material_command{
            .entity = entity, .path = "materials/stone.arcmat" } }
    };

    for (const auto& command : commands)
    {
        const auto json = arc::editor::to_json(command);
        arc::editor::host_command_envelope parsed;
        std::string error;
        REQUIRE(arc::editor::from_json(json, parsed, error));
        REQUIRE(parsed.request_id == command.request_id);
        REQUIRE(parsed.command_type == arc::editor::command_type(command.payload));
        if (command.request_id == 3)
        {
            const auto& create = std::get<arc::editor::host_create_entity_command>(parsed.payload);
            REQUIRE(create.kind == arc::editor::host_create_entity_kind::empty);
            REQUIRE(create.parent == entity);
        }
    }

    const arc::editor::host_query_envelope queries[]{
        { .request_id = 16, .payload = arc::editor::host_scene_hierarchy_query{} },
        { .request_id = 17, .payload = arc::editor::host_selected_entity_query{} },
        { .request_id = 18, .payload = arc::editor::host_project_assets_query{} },
        { .request_id = 19, .payload = arc::editor::host_asset_thumbnail_query{ .path = "textures/checker.png", .max_size = 128 } },
        { .request_id = 20, .payload = arc::editor::host_viewport_state_query{} },
        { .request_id = 21, .payload = arc::editor::host_world_environment_query{ .entity = entity } }
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

TEST_CASE("arc host catalogs textures and generates safe lazy thumbnails")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-thumbnail-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "textures");
    const auto texture_path = root / "textures" / "preview.tga";
    std::array<unsigned char, 34> tga{};
    tga[2] = 2;
    tga[12] = 2;
    tga[14] = 2;
    tga[16] = 32;
    tga[17] = 0x20;
    const std::array<unsigned char, 16> pixels{
        0, 0, 255, 255, 0, 255, 0, 255,
        255, 0, 0, 255, 255, 255, 255, 255
    };
    std::copy(pixels.begin(), pixels.end(), tga.begin() + 18);
    {
        std::ofstream output(texture_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(tga.data()), static_cast<std::streamsize>(tga.size()));
    }

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root;
    REQUIRE(host->open_project({ .name = "Thumbnail Test", .root = root }, assets).succeeded);

    const auto catalog = host->project_assets_snapshot();
    REQUIRE(std::any_of(catalog.assets.begin(), catalog.assets.end(), [](const auto& asset) {
        return asset.path == "textures/preview.tga" && asset.kind == "texture";
    }));

    const auto thumbnail = host->asset_thumbnail("textures/preview.tga", 64);
    REQUIRE(thumbnail.has_value());
    REQUIRE(thumbnail->width == 2);
    REQUIRE(thumbnail->height == 2);
    REQUIRE(thumbnail->data_url.starts_with("data:image/bmp;base64,Qk"));
    REQUIRE_FALSE(host->asset_thumbnail("../outside.tga", 64).has_value());

    const auto response = host->query({
        .request_id = 91,
        .payload = arc::editor::host_asset_thumbnail_query{ .path = "textures/preview.tga", .max_size = 64 } });
    REQUIRE(response.succeeded);
    REQUIRE(response.payload_json.find("data:image/bmp;base64,") != std::string::npos);

    const auto hierarchy = host->scene_snapshot();
    const auto environment = std::find_if(hierarchy.entities.begin(), hierarchy.entities.end(), [](const auto& entity) {
        return entity.kind == arc::editor::host_entity_kind::environment;
    });
    REQUIRE(environment != hierarchy.entities.end());
    REQUIRE(host->execute({ .request_id = 92, .payload = arc::editor::host_set_environment_hdri_command{
        .entity = environment->entity, .path = "textures/preview.tga" } }).succeeded);
    const auto assigned_environment = host->world_environment_snapshot(environment->entity);
    REQUIRE(assigned_environment.has_value());
    REQUIRE(assigned_environment->hdri_path == "textures/preview.tga");
    REQUIRE(assigned_environment->sky_source == arc::editor::host_sky_source::physical_atmosphere);
    REQUIRE_FALSE(host->execute({ .request_id = 93, .payload = arc::editor::host_set_environment_hdri_command{
        .entity = environment->entity, .path = "../outside.tga" } }).succeeded);
    REQUIRE(host->execute({ .request_id = 94, .payload = arc::editor::host_set_environment_hdri_command{
        .entity = environment->entity, .path = {} } }).succeeded);
    REQUIRE(host->world_environment_snapshot(environment->entity)->hdri_path.empty());
    std::filesystem::remove_all(root);
}

TEST_CASE("material preview renderer produces a deterministic PBR sphere")
{
    auto material = arc::editor::make_default_material_asset("Preview Bronze");
    material.material.base_color = { 0.72f, 0.24f, 0.07f, 1.0f };
    material.material.metallic = 0.85f;
    material.material.roughness = 0.28f;
    material.material.emissive_factor = { 0.01f, 0.0f, 0.0f };
    const auto first = arc::editor::render_material_preview(material, {}, 64);
    const auto second = arc::editor::render_material_preview(material, {}, 64);
    REQUIRE(first.succeeded());
    REQUIRE(first.texture.width == 64);
    REQUIRE(first.texture.height == 64);
    REQUIRE(first.texture.format == arc::render::texture_format::rgba8_srgb);
    REQUIRE(first.texture.pixels == second.texture.pixels);
    const auto center = (32u * 64u + 32u) * 4u;
    REQUIRE(first.texture.pixels[center] != first.texture.pixels[0]);
}

TEST_CASE("mesh renderer host snapshot edits and material assignment round trip")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-editor-mesh-material-host";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "materials");
    auto material = arc::editor::make_default_material_asset("Inspector Stone");
    material.path = root / "materials" / "inspector_stone.arcmat";
    material.material.base_color = { 0.3f, 0.34f, 0.38f, 1.0f };
    std::string message;
    REQUIRE(arc::editor::save_material_asset(material, root, message));

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root;
    REQUIRE(host->open_project({ .name = "Mesh Material Host", .root = root }, assets).succeeded);
    REQUIRE(host->execute({ .request_id = 1, .payload = arc::editor::host_create_entity_command{
        .kind = arc::editor::host_create_entity_kind::sphere } }).succeeded);
    const auto entity = host->selected_entity_snapshot().entity;
    const auto initial = host->selected_entity_snapshot();
    REQUIRE(initial.mesh_renderer.has_value());
    REQUIRE(initial.mesh_renderer->visible);
    REQUIRE(initial.mesh_renderer->has_material);

    REQUIRE(host->execute({ .request_id = 2, .payload = arc::editor::host_set_mesh_renderer_command{
        .entity = entity, .visible = false, .base_color_tint = { 0.5f, 0.6f, 0.7f, 0.8f } } }).succeeded);
    REQUIRE_FALSE(host->selected_entity_snapshot().mesh_renderer->visible);
    REQUIRE(host->selected_entity_snapshot().mesh_renderer->base_color_tint.z == Catch::Approx(0.7f));
    REQUIRE_FALSE(host->execute({ .request_id = 3, .payload = arc::editor::host_set_mesh_renderer_command{
        .entity = entity, .visible = true,
        .base_color_tint = { std::numeric_limits<float>::infinity(), 1.0f, 1.0f, 1.0f } } }).succeeded);

    REQUIRE(host->execute({ .request_id = 4, .payload = arc::editor::host_set_entity_material_command{
        .entity = entity, .path = "materials/inspector_stone.arcmat" } }).succeeded);
    const auto assigned = host->selected_entity_snapshot();
    REQUIRE(assigned.mesh_renderer->asset_backed_material);
    REQUIRE(assigned.mesh_renderer->material_name == "Inspector Stone");
    REQUIRE(assigned.mesh_renderer->material_path == "materials/inspector_stone.arcmat");
    REQUIRE_FALSE(host->execute({ .request_id = 5, .payload = arc::editor::host_set_entity_material_command{
        .entity = entity, .path = "../outside.arcmat" } }).succeeded);

    const auto catalog = host->project_assets_snapshot();
    REQUIRE(std::any_of(catalog.assets.begin(), catalog.assets.end(), [](const auto& asset) {
        return asset.path == "materials/inspector_stone.arcmat" && asset.kind == "material";
    }));
    const auto preview = host->asset_thumbnail("materials/inspector_stone.arcmat", 96);
    REQUIRE(preview.has_value());
    REQUIRE(preview->width == 96);
    REQUIRE(preview->height == 96);
    REQUIRE(preview->data_url.starts_with("data:image/bmp;base64,Qk"));
    std::filesystem::remove_all(root);
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

TEST_CASE("arc host hierarchy and history preserve subtrees and group edit transactions")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({ .name = "Authoring Test", .root = {} }, assets).succeeded);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{
        .kind = arc::editor::host_create_entity_kind::cube }).succeeded);
    const auto parent = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_create_entity_command{
        .kind = arc::editor::host_create_entity_kind::sphere }).succeeded);
    const auto child = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_reparent_entity_command{
        .entity = child, .parent = parent, .preserve_world = true }).succeeded);
    REQUIRE(host->execute(arc::editor::host_create_entity_command{
        .kind = arc::editor::host_create_entity_kind::empty, .parent = parent }).succeeded);
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

    REQUIRE(host->execute(arc::editor::host_create_entity_command{
        .kind = arc::editor::host_create_entity_kind::empty }).succeeded);
    const auto first_root = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_create_entity_command{
        .kind = arc::editor::host_create_entity_kind::empty }).succeeded);
    const auto second_root = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_reorder_entity_command{
        .entity = second_root, .before_sibling = first_root }).succeeded);
    const auto reordered = host->scene_snapshot();
    const auto first_root_record = std::find_if(reordered.entities.begin(), reordered.entities.end(),
        [first_root](const auto& value) { return value.entity == first_root; });
    const auto second_root_record = std::find_if(reordered.entities.begin(), reordered.entities.end(),
        [second_root](const auto& value) { return value.entity == second_root; });
    REQUIRE(first_root_record != reordered.entities.end());
    REQUIRE(second_root_record != reordered.entities.end());
    REQUIRE(second_root_record->parent_guid.empty());
    REQUIRE(second_root_record->sibling_order < first_root_record->sibling_order);

    REQUIRE(host->execute(arc::editor::host_select_entity_command{ .entity = parent }).succeeded);
    REQUIRE(host->execute(arc::editor::host_duplicate_entity_command{ .entity = parent }).succeeded);
    const auto duplicate = host->selected_entity_snapshot().entity;
    REQUIRE(duplicate != parent);
    REQUIRE(host->scene_snapshot().entities.size() == reordered.entities.size() + 3);
    REQUIRE(host->execute(arc::editor::host_delete_entity_command{ .entity = parent }).succeeded);
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    const auto restored_hierarchy = host->scene_snapshot();
    REQUIRE(std::any_of(restored_hierarchy.entities.begin(), restored_hierarchy.entities.end(),
        [parent](const auto& value) { return value.entity == parent; }));

    REQUIRE(host->execute(arc::editor::host_select_entity_command{ .entity = duplicate }).succeeded);
    const auto original = *host->selected_entity_snapshot().transform;
    auto preview = original;
    preview.position.x += 1.0f;
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .payload = arc::editor::host_set_transform_command{ .entity = duplicate, .transform = preview },
        .edit = arc::editor::host_edit_transaction{ .id = 7, .phase = arc::editor::host_edit_phase::begin, .label = "Gizmo Drag" } }).succeeded);
    preview.position.x += 1.0f;
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .payload = arc::editor::host_set_transform_command{ .entity = duplicate, .transform = preview },
        .edit = arc::editor::host_edit_transaction{ .id = 7, .phase = arc::editor::host_edit_phase::update } }).succeeded);
    preview.position.x += 1.0f;
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .payload = arc::editor::host_set_transform_command{ .entity = duplicate, .transform = preview },
        .edit = arc::editor::host_edit_transaction{ .id = 7, .phase = arc::editor::host_edit_phase::commit } }).succeeded);
    REQUIRE(host->scene_snapshot().undo_label == "Gizmo Drag");
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    REQUIRE(host->selected_entity_snapshot().transform->position.x == Catch::Approx(original.position.x));
    REQUIRE(host->execute(arc::editor::host_history_redo_command{}).succeeded);
    REQUIRE(host->selected_entity_snapshot().transform->position.x == Catch::Approx(preview.position.x));

    auto cancelled = preview;
    cancelled.position.y += 8.0f;
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .payload = arc::editor::host_set_transform_command{ .entity = duplicate, .transform = cancelled },
        .edit = arc::editor::host_edit_transaction{ .id = 8, .phase = arc::editor::host_edit_phase::begin, .label = "Cancelled Drag" } }).succeeded);
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .payload = arc::editor::host_set_transform_command{ .entity = duplicate, .transform = cancelled },
        .edit = arc::editor::host_edit_transaction{ .id = 8, .phase = arc::editor::host_edit_phase::cancel } }).succeeded);
    REQUIRE(host->selected_entity_snapshot().transform->position.y == Catch::Approx(preview.position.y));
}

TEST_CASE("viewport navigation and repeated selection do not emit scene refresh events")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({ .name = "Event Dedup Test", .root = {} }, assets).succeeded);
    host->poll_events();

    const auto selected = host->selected_entity_snapshot().entity;
    REQUIRE(selected.valid());
    REQUIRE(host->execute(arc::editor::host_select_entity_command{ .entity = selected }).succeeded);
    REQUIRE(host->poll_events().empty());

    REQUIRE(host->execute(arc::editor::host_viewport_camera_input_command{
        .orbit_x = 4.0f, .orbit_y = -2.0f }).succeeded);
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
    REQUIRE(host->open_project({ .name = "Async Pick Test", .root = {} }, assets).succeeded);
    auto backend = std::make_unique<pick_test_backend>();
    auto* backend_ptr = backend.get();
    host->renderer_service().set_backend(std::move(backend));
    host->poll_events();

    const auto original = host->selected_entity_snapshot().entity;
    const auto target = host->scene_state().water_entity;
    REQUIRE(host->scene_state().scene.alive(target));
    REQUIRE(target != (arc::scene::entity{ original.index, original.generation }));
    REQUIRE(host->execute(arc::editor::host_viewport_pick_command{ .x = 24, .y = 36 }).succeeded);
    REQUIRE(host->selected_entity_snapshot().entity == original);
    REQUIRE(backend_ptr->request_.x == 24);
    REQUIRE(backend_ptr->request_.y == 36);

    backend_ptr->result = {
        .available = true,
        .hit = true,
        .object = { target.index, target.generation },
        .x = 24,
        .y = 36,
        .frame_index = 1
    };
    REQUIRE(host->request_viewport({ .frame_index = 1, .width = 640, .height = 480 }).submitted);
    REQUIRE(host->selected_entity_snapshot().entity == (arc::editor::host_entity_id{ target.index, target.generation }));
    const auto events = host->poll_events();
    REQUIRE(std::count_if(events.begin(), events.end(), [](const auto& event) {
        return event.event_type == arc::editor::host_event_type::entity_selected;
    }) == 1);
}

TEST_CASE("ARC scene documents save atomically, round trip hierarchy, and reject invalid loads")
{
    const auto root = std::filesystem::temp_directory_path() /
        ("arc-scene-document-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code error;
    std::filesystem::create_directories(root / "assets", error);
    REQUIRE_FALSE(error);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root / "assets";
    REQUIRE(host->open_project({ .name = "Persistence Test", .root = root }, assets).succeeded);
    const auto selected = host->selected_entity_snapshot().entity;
    REQUIRE(host->execute(arc::editor::host_rename_entity_command{ .entity = selected, .name = "Persisted Entity" }).succeeded);
    const auto initial_snapshot = host->scene_snapshot();
    const auto selected_record = std::find_if(initial_snapshot.entities.begin(), initial_snapshot.entities.end(),
        [selected](const auto& value) { return value.entity == selected; });
    REQUIRE(selected_record != initial_snapshot.entities.end());
    const auto selected_guid = selected_record->guid;

    const auto path = root / "scenes" / "mountain.arcscene";
    REQUIRE(host->execute(arc::editor::host_save_scene_as_command{ .path = path }).succeeded);
    REQUIRE(std::filesystem::exists(path));
    REQUIRE_FALSE(host->scene_snapshot().dirty);

    REQUIRE(host->execute(arc::editor::host_rename_entity_command{ .entity = selected, .name = "Changed" }).succeeded);
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
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << document;
    }
    REQUIRE(host->execute(arc::editor::host_open_scene_command{ .path = path }).succeeded);
    REQUIRE_FALSE(host->scene_snapshot().dirty);
    const auto loaded_snapshot = host->scene_snapshot();
    REQUIRE(std::any_of(loaded_snapshot.entities.begin(), loaded_snapshot.entities.end(),
        [&](const auto& value) { return value.guid == selected_guid && value.name == "Persisted Entity"; }));
    REQUIRE(host->execute(arc::editor::host_save_scene_command{}).succeeded);
    std::ifstream resaved_input(path, std::ios::binary);
    const std::string resaved((std::istreambuf_iterator<char>(resaved_input)), std::istreambuf_iterator<char>());
    resaved_input.close();
    REQUIRE(resaved.find("FutureRenderer") != std::string::npos);
    REQUIRE(resaved.find("\"quality\": \"future\"") != std::string::npos);

    auto invalid = resaved;
    const auto version = invalid.find("\"formatVersion\": 1");
    REQUIRE(version != std::string::npos);
    invalid.replace(version, std::string("\"formatVersion\": 1").size(), "\"formatVersion\": 99");
    const auto invalid_path = root / "scenes" / "unsupported.arcscene";
    {
        std::ofstream output(invalid_path, std::ios::binary | std::ios::trunc);
        output << invalid;
    }
    const auto before_invalid_load = arc::editor::to_json(host->scene_snapshot());
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{ .path = invalid_path }).succeeded);
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
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{ .path = malformed_camera_path }).succeeded);
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
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{ .path = unsafe_asset_path }).succeeded);
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
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{ .path = duplicate_guid_path }).succeeded);
    REQUIRE(arc::editor::to_json(host->scene_snapshot()) == before_invalid_load);

    auto cyclic_hierarchy = resaved;
    const auto replace_parent = [&](std::size_t id_key, const std::string& parent_id) {
        const auto parent_key = cyclic_hierarchy.find("\"parent\":", id_key);
        REQUIRE(parent_key != std::string::npos);
        const auto value_begin = cyclic_hierarchy.find_first_not_of(" \t", parent_key + std::string("\"parent\":").size());
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
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{ .path = cyclic_path }).succeeded);
    REQUIRE(arc::editor::to_json(host->scene_snapshot()) == before_invalid_load);

    std::filesystem::remove_all(root, error);
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

TEST_CASE("scene authoring protocol commands and edit transactions round trip")
{
    const arc::editor::host_entity_id entity{ 8, 3 };
    const arc::editor::host_entity_id parent{ 4, 2 };
    const std::array<arc::editor::host_command_payload, 9> payloads{
        arc::editor::host_new_scene_command{ .name = "New World" },
        arc::editor::host_save_scene_command{},
        arc::editor::host_save_scene_as_command{ .path = "scenes/world.arcscene" },
        arc::editor::host_duplicate_entity_command{ .entity = entity },
        arc::editor::host_reparent_entity_command{ .entity = entity, .parent = parent, .preserve_world = true },
        arc::editor::host_reorder_entity_command{ .entity = entity, .before_sibling = parent },
        arc::editor::host_history_undo_command{},
        arc::editor::host_history_redo_command{},
        arc::editor::host_viewport_set_tool_command{ .tool = arc::editor::host_viewport_tool::rotate,
            .coordinate_space = arc::editor::host_coordinate_space::local, .snapping = true }
    };
    for (const auto& payload : payloads)
    {
        const arc::editor::host_command_envelope source{
            .request_id = 71,
            .command_type = arc::editor::command_type(payload),
            .payload = payload,
            .edit = arc::editor::host_edit_transaction{ .id = 44, .phase = arc::editor::host_edit_phase::commit, .label = "Quoted \"edit\"" }
        };
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

TEST_CASE("terrain material version two round trips fixed layer descriptors")
{
    const auto root = std::filesystem::temp_directory_path() / "arc_editor_terrain_material_tests";
    std::filesystem::create_directories(root / "materials");

    auto asset = arc::editor::make_default_material_asset("Layered Terrain");
    asset.version = 2;
    asset.path = root / "materials" / "layered.arcmat";
    asset.domain = "terrain";
    asset.material.domain = arc::render::material_domain::terrain;
    asset.material.terrain_layers[0].name = "Grass";
    asset.material.terrain_layers[0].tint = { 0.65f, 0.8f, 0.5f, 1.0f };
    asset.material.terrain_layers[0].world_scale = 2.75f;
    asset.material.terrain_layers[0].roughness = 0.84f;
    asset.terrain_layers[0].base_color = "textures/terrain/grass/base.jpg";
    asset.terrain_layers[0].normal = "textures/terrain/grass/normal.png";
    asset.terrain_layers[0].roughness = "textures/terrain/grass/roughness.jpg";
    asset.terrain_layers[0].ao = "textures/terrain/grass/ao.jpg";
    asset.terrain_layers[0].height = "textures/terrain/grass/height.png";
    asset.terrain_layers[0].packed_aorh = "textures/terrain/grass/aorh.png";

    std::string message;
    REQUIRE(arc::editor::save_material_asset(asset, root, message));
    arc::editor::material_asset loaded;
    REQUIRE(arc::editor::load_material_asset(asset.path, root, loaded, message));
    REQUIRE(loaded.version == 2);
    REQUIRE(loaded.material.domain == arc::render::material_domain::terrain);
    REQUIRE(loaded.material.terrain_layers[0].name == "Grass");
    REQUIRE(loaded.material.terrain_layers[0].world_scale == Catch::Approx(2.75f));
    REQUIRE(loaded.material.terrain_layers[0].roughness == Catch::Approx(0.84f));
    REQUIRE(loaded.material.terrain_layers[0].tint[1] == Catch::Approx(0.8f));
    REQUIRE(loaded.terrain_layers[0].base_color == "textures/terrain/grass/base.jpg");
    REQUIRE(loaded.terrain_layers[0].normal == "textures/terrain/grass/normal.png");
    REQUIRE(loaded.terrain_layers[0].roughness == "textures/terrain/grass/roughness.jpg");
    REQUIRE(loaded.terrain_layers[0].ao == "textures/terrain/grass/ao.jpg");
    REQUIRE(loaded.terrain_layers[0].height == "textures/terrain/grass/height.png");
    REQUIRE(loaded.terrain_layers[0].packed_aorh == "textures/terrain/grass/aorh.png");
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

TEST_CASE("terrain host snapshots validate brush settings and group a stroke into one history entry")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    REQUIRE(host->open_project({ .name = "Terrain Authoring", .root = {} }, assets).succeeded);
    const auto terrain_entity = host->scene_state().terrain_entity;
    REQUIRE(host->execute(arc::editor::host_select_entity_command{
        .entity = { terrain_entity.index, terrain_entity.generation } }).succeeded);
    const auto terrain_id = host->selected_entity_snapshot().entity;
    REQUIRE(host->selected_entity_snapshot().terrain.has_value());
    REQUIRE(host->selected_entity_snapshot().terrain->resolution == 257u);
    REQUIRE(host->selected_entity_snapshot().terrain->chunk_quads == 128u);

    REQUIRE(host->execute(arc::editor::host_set_terrain_brush_command{
        .entity = terrain_id,
        .tool = arc::editor::host_terrain_brush_tool::paint,
        .radius = 8.0f,
        .strength = 0.4f,
        .falloff = 0.75f,
        .active_layer = 2u }).succeeded);
    const auto configured = *host->selected_entity_snapshot().terrain;
    REQUIRE(configured.brush_tool == arc::editor::host_terrain_brush_tool::paint);
    REQUIRE(configured.brush_radius == Catch::Approx(8.0f));
    REQUIRE(configured.active_layer == 2u);
    REQUIRE_FALSE(host->execute(arc::editor::host_set_terrain_brush_command{
        .entity = terrain_id,
        .radius = std::numeric_limits<float>::infinity(),
        .strength = 0.4f,
        .falloff = 0.75f }).succeeded);
    REQUIRE(host->selected_entity_snapshot().terrain->brush_radius == Catch::Approx(8.0f));

    REQUIRE(host->execute(arc::editor::host_viewport_set_tool_command{
        .tool = arc::editor::host_viewport_tool::terrain }).succeeded);
    host->request_viewport({ .frame_index = 1u, .width = 800u, .height = 600u });
    auto& terrain = host->scene_state().scene.get<arc::scene::terrain_component>(terrain_entity);
    const auto before = terrain.layer_weights;
    const auto begin = host->execute(arc::editor::host_command_envelope{
        .payload = arc::editor::host_terrain_stroke_command{
            terrain_id, 400u, 300u, arc::editor::host_edit_phase::begin, false },
        .edit = arc::editor::host_edit_transaction{ 901u, arc::editor::host_edit_phase::begin, "Terrain Stroke" } });
    REQUIRE(begin.succeeded);
    REQUIRE(begin.payload_json.find("\"hit\":true") != std::string::npos);
    REQUIRE(host->execute(arc::editor::host_command_envelope{
        .payload = arc::editor::host_terrain_stroke_command{
            terrain_id, 405u, 300u, arc::editor::host_edit_phase::commit, false },
        .edit = arc::editor::host_edit_transaction{ 901u, arc::editor::host_edit_phase::commit, "Terrain Stroke" } }).succeeded);
    REQUIRE(host->scene_snapshot().undo_label == "Terrain Stroke");
    REQUIRE(terrain.layer_weights != before);
    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    REQUIRE(host->scene_state().scene.get<arc::scene::terrain_component>(terrain_entity).layer_weights == before);
}

TEST_CASE("terrain scene version 2 payload round trips quantized heights and rejects corrupt data atomically")
{
    const auto root = std::filesystem::temp_directory_path() / "arc-terrain-scene-v2-test";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root, error);
    REQUIRE_FALSE(error);

    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));
    arc::editor::editor_asset_state assets;
    assets.root = root;
    REQUIRE(host->open_project({ .name = "Terrain Persistence", .root = root }, assets).succeeded);
    auto& terrain = host->scene_state().scene.get<arc::scene::terrain_component>(host->scene_state().terrain_entity);
    arc::scene::terrain_brush_settings brush;
    brush.tool = arc::scene::terrain_brush_tool::sculpt;
    brush.radius = 9.0f;
    brush.strength = 0.65f;
    arc::scene::apply_terrain_brush(terrain, { 4.0f, 0.0f, -7.0f }, brush, 0.5f);
    const auto expected_heights = terrain.heights;
    const auto expected_weights = terrain.layer_weights;
    const auto path = root / "terrain.arcscene";
    REQUIRE(host->execute(arc::editor::host_save_scene_as_command{ .path = path }).succeeded);

    std::ifstream saved_stream(path, std::ios::binary);
    const std::string saved((std::istreambuf_iterator<char>(saved_stream)), std::istreambuf_iterator<char>());
    REQUIRE(saved.find("\"Terrain\"") != std::string::npos);
    REQUIRE(saved.find("\"version\": 2") != std::string::npos);
    REQUIRE(saved.find("\"heights\"") != std::string::npos);
    REQUIRE(saved.find("\"weights\"") != std::string::npos);

    terrain.heights.assign(terrain.heights.size(), -100.0f);
    REQUIRE(host->execute(arc::editor::host_open_scene_command{ .path = path }).succeeded);
    const auto& loaded = host->scene_state().scene.get<arc::scene::terrain_component>(host->scene_state().terrain_entity);
    REQUIRE(loaded.layer_weights == expected_weights);
    const auto [minimum, maximum] = std::minmax_element(expected_heights.begin(), expected_heights.end());
    const float tolerance = (*maximum - *minimum) / 65535.0f + 0.0001f;
    for (std::size_t index = 0; index < loaded.heights.size(); index += 997u)
        REQUIRE(loaded.heights[index] == Catch::Approx(expected_heights[index]).margin(tolerance));

    auto corrupt = saved;
    const auto payload = corrupt.find("\"heights\": \"");
    REQUIRE(payload != std::string::npos);
    const auto data_begin = payload + std::string("\"heights\": \"").size();
    const auto data_end = corrupt.find('"', data_begin);
    corrupt.replace(data_begin, data_end - data_begin, "AAAA");
    const auto corrupt_path = root / "corrupt.arcscene";
    { std::ofstream stream(corrupt_path, std::ios::binary | std::ios::trunc); stream << corrupt; }
    const auto revision_before = loaded.content_revision;
    REQUIRE_FALSE(host->execute(arc::editor::host_open_scene_command{ .path = corrupt_path }).succeeded);
    REQUIRE(host->scene_state().scene.get<arc::scene::terrain_component>(host->scene_state().terrain_entity).content_revision == revision_before);
    std::filesystem::remove_all(root, error);
}
