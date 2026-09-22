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
#include <arc/scene/hierarchy.h>

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

    sink.write({.level = arc::diagnostics::log_level::info, .category = "one", .message = "first"});
    sink.write({.level = arc::diagnostics::log_level::warn, .category = "two", .message = "second"});
    sink.write({.level = arc::diagnostics::log_level::error, .category = "three", .message = "third"});

    const auto entries = sink.entries();
    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].category == "two");
    REQUIRE(entries[1].level == arc::diagnostics::log_level::error);
    REQUIRE(entries[1].message == "third");
}

TEST_CASE("editor selection keeps one selected entity")
{
    arc::ecs::world scene;
    auto first = scene.create();
    auto second = scene.create();
    scene.emplace<arc::scene::selection_component>(first, true);

    arc::ecs::entity selected = first;
    REQUIRE(arc::editor::select_entity(scene, second, selected));
    REQUIRE(selected == second);
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(first).selected);
    REQUIRE(scene.get<arc::scene::selection_component>(second).selected);

    arc::editor::clear_selection(scene, selected);
    REQUIRE_FALSE(selected.valid());
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(first).selected);
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(second).selected);
}

TEST_CASE("editor perspective grid uses adaptive hierarchy and semantic XZ axes")
{
    arc::ecs::world registry;
    const auto camera_entity = registry.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 5.0f, 8.0f};
    registry.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    registry.emplace<arc::scene::camera_component>(camera_entity);
    arc::scene::update_world_transforms(registry);

    arc::render::debug_overlay_stream overlay;
    const arc::math::vector3f grid_color{0.2f, 0.21568628f, 0.23921569f};
    arc::editor::append_editor_grid_overlay(overlay, registry.get<arc::scene::camera_component>(camera_entity),
                                            registry.get<arc::scene::transform_component>(camera_entity), 720u,
                                            grid_color);
    REQUIRE_FALSE(overlay.lines.empty());

    float maximum_extent = 0.0f;
    bool saw_minor{};
    bool saw_major{};
    bool saw_x_axis{};
    bool saw_z_axis{};
    for (const auto& line : overlay.lines)
    {
        CHECK(line.depth == arc::render::debug_overlay_depth_mode::tested);
        maximum_extent = std::max(maximum_extent, std::abs(line.start[0]));
        maximum_extent = std::max(maximum_extent, std::abs(line.start[2]));
        maximum_extent = std::max(maximum_extent, std::abs(line.end[0]));
        maximum_extent = std::max(maximum_extent, std::abs(line.end[2]));
        if (line.color[0] == Catch::Approx(0.7882353f).margin(0.0001f) &&
            line.color[1] == Catch::Approx(0.3372549f).margin(0.0001f))
            saw_x_axis = true;
        else if (line.color[0] == Catch::Approx(0.2980392f).margin(0.0001f) &&
                 line.color[2] == Catch::Approx(0.8196079f).margin(0.0001f))
            saw_z_axis = true;
        else
        {
            saw_minor = saw_minor || line.color[3] < 0.45f;
            saw_major = saw_major || line.color[3] > 0.50f;
        }
    }
    CHECK(maximum_extent >= 50.0f);
    CHECK(saw_minor);
    CHECK(saw_major);
    CHECK(saw_x_axis);
    CHECK(saw_z_axis);
}

TEST_CASE("editor orthographic grid rotates onto the camera plane and emphasizes its axes")
{
    arc::ecs::world registry;
    const auto camera_entity = registry.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 0.0f, 10.0f};
    registry.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    arc::scene::camera_component camera;
    camera.projection = arc::scene::camera_projection::orthographic;
    camera.orthographic_height = 67.5f;
    registry.emplace<arc::scene::camera_component>(camera_entity, camera);
    arc::scene::update_world_transforms(registry);

    arc::render::debug_overlay_stream overlay;
    arc::editor::append_editor_grid_overlay(overlay, registry.get<arc::scene::camera_component>(camera_entity),
                                            registry.get<arc::scene::transform_component>(camera_entity), 720u);
    REQUIRE_FALSE(overlay.lines.empty());

    bool saw_x_axis{};
    bool saw_y_axis{};
    float minimum_grid_alpha = 1.0f;
    float maximum_grid_alpha = 0.0f;
    for (const auto& line : overlay.lines)
    {
        CHECK(std::abs(line.start[2]) < 0.01f);
        CHECK(std::abs(line.end[2]) < 0.01f);
        const bool x_axis = line.color[0] == Catch::Approx(0.7882353f).margin(0.0001f) &&
                            line.color[1] == Catch::Approx(0.3372549f).margin(0.0001f);
        const bool y_axis = line.color[0] == Catch::Approx(0.4117647f).margin(0.0001f) &&
                            line.color[1] == Catch::Approx(0.7098039f).margin(0.0001f);
        saw_x_axis = saw_x_axis || x_axis;
        saw_y_axis = saw_y_axis || y_axis;
        if (!x_axis && !y_axis)
        {
            minimum_grid_alpha = std::min(minimum_grid_alpha, line.color[3]);
            maximum_grid_alpha = std::max(maximum_grid_alpha, line.color[3]);
        }
    }
    CHECK(saw_x_axis);
    CHECK(saw_y_axis);
    CHECK(minimum_grid_alpha < 0.45f);
    CHECK(maximum_grid_alpha > minimum_grid_alpha + 0.1f);
}

TEST_CASE("editor gizmos keep constant screen size and hit test colored axes")
{
    arc::ecs::world registry;
    const auto camera_entity = registry.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 0.0f, 5.0f};
    registry.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    registry.emplace<arc::scene::camera_component>(camera_entity);
    const auto selected = registry.create();
    registry.emplace<arc::scene::transform_component>(selected);
    registry.emplace<arc::scene::bounds_component>(
        selected,
        arc::geometric::box3f{arc::geometric::point3f{-0.5f, -0.5f, -0.5f}, arc::geometric::point3f{0.5f, 0.5f, 0.5f}});
    arc::scene::update_world_transforms(registry);

    const arc::editor::editor_gizmo_context context{
        .tool = arc::editor::editor_tool::translate, .viewport_width = 800, .viewport_height = 600};
    const auto overlay = arc::editor::build_editor_gizmo_overlay(registry, selected, camera_entity, context);
    REQUIRE(overlay.lines.empty());
    REQUIRE(overlay.triangles.size() == 120);
    REQUIRE(overlay.triangles.front().color[0] > overlay.triangles.front().color[1]);
    REQUIRE(arc::editor::hit_test_editor_gizmo(registry, selected, camera_entity, context, 450.0f, 300.0f) ==
            arc::editor::gizmo_axis::x);
    REQUIRE(arc::editor::hit_test_editor_gizmo(registry, selected, camera_entity, context, 780.0f, 580.0f) ==
            arc::editor::gizmo_axis::none);

    const auto& camera = registry.get<arc::scene::camera_component>(camera_entity);
    const float near_scale = arc::editor::editor_gizmo_world_scale(
        camera, registry.get<arc::scene::transform_component>(camera_entity), {}, 600);
    const float off_axis_scale = arc::editor::editor_gizmo_world_scale(
        camera, registry.get<arc::scene::transform_component>(camera_entity), {8.0f, 0.0f, 0.0f}, 600);
    REQUIRE(off_axis_scale == Catch::Approx(near_scale));
    registry.get<arc::scene::transform_component>(camera_entity).set_position({0.0f, 0.0f, 10.0f});
    arc::scene::update_world_transforms(registry);
    const float far_scale = arc::editor::editor_gizmo_world_scale(
        camera, registry.get<arc::scene::transform_component>(camera_entity), {}, 600);
    REQUIRE(far_scale == Catch::Approx(near_scale * 2.0f));

    const arc::editor::editor_gizmo_context highlighted_context{.tool = arc::editor::editor_tool::translate,
                                                                .highlighted_axis = arc::editor::gizmo_axis::x,
                                                                .viewport_width = 800,
                                                                .viewport_height = 600};
    const auto highlighted =
        arc::editor::build_editor_gizmo_overlay(registry, selected, camera_entity, highlighted_context);
    REQUIRE(highlighted.triangles.size() == overlay.triangles.size());
    REQUIRE(highlighted.triangles.front().color[0] > 0.99f);
    REQUIRE(highlighted.triangles.front().color[1] > 0.8f);

    const auto rotation = arc::editor::build_editor_gizmo_overlay(
        registry, selected, camera_entity,
        {.tool = arc::editor::editor_tool::rotate, .viewport_width = 800, .viewport_height = 600});
    REQUIRE(rotation.triangles.size() == 384);

    const auto scaling = arc::editor::build_editor_gizmo_overlay(
        registry, selected, camera_entity,
        {.tool = arc::editor::editor_tool::scale, .viewport_width = 800, .viewport_height = 600});
    REQUIRE(scaling.triangles.size() == 108);
    REQUIRE(arc::editor::hit_test_editor_gizmo(
                registry, selected, camera_entity,
                {.tool = arc::editor::editor_tool::scale, .viewport_width = 800, .viewport_height = 600}, 400.0f,
                300.0f) == arc::editor::gizmo_axis::all);

    arc::math::vector2f uniform_direction;
    REQUIRE(arc::editor::editor_gizmo_drag_direction(
        registry, selected, camera_entity,
        {.tool = arc::editor::editor_tool::scale, .viewport_width = 800, .viewport_height = 600},
        arc::editor::gizmo_axis::all, 400.0f, 300.0f, uniform_direction));
    REQUIRE(arc::math::length(uniform_direction) == Catch::Approx(1.0f));

    const auto bounds_overlay = arc::editor::build_editor_gizmo_overlay(registry, selected, camera_entity,
                                                                        {.tool = arc::editor::editor_tool::select,
                                                                         .show_selection_bounds = true,
                                                                         .viewport_width = 800,
                                                                         .viewport_height = 600});
    REQUIRE(bounds_overlay.lines.size() == 12);
}

TEST_CASE("editor viewport only shows component gizmos for the selected entity")
{
    arc::ecs::world registry;
    const auto view_camera = registry.create();
    arc::scene::transform_component view_transform;
    view_transform.position = {0.0f, 2.0f, 8.0f};
    registry.emplace<arc::scene::transform_component>(view_camera, view_transform);
    registry.emplace<arc::scene::camera_component>(view_camera);

    const auto scene_camera = registry.create();
    arc::scene::transform_component scene_camera_transform;
    scene_camera_transform.position = {0.0f, 1.0f, 0.0f};
    registry.emplace<arc::scene::transform_component>(scene_camera, scene_camera_transform);
    registry.emplace<arc::scene::camera_component>(scene_camera);

    const auto light = registry.create();
    arc::scene::transform_component light_transform;
    light_transform.position = {2.0f, 3.0f, 1.0f};
    registry.emplace<arc::scene::transform_component>(light, light_transform);
    registry.emplace<arc::scene::point_light_component>(light);
    arc::scene::update_world_transforms(registry);

    const auto overlay = arc::editor::build_editor_gizmo_overlay(
        registry, scene_camera, view_camera,
        {.tool = arc::editor::editor_tool::select, .viewport_width = 1280, .viewport_height = 720});
    REQUIRE(overlay.lines.size() == 19);
    const auto frustum_lines = std::count_if(overlay.lines.begin(), overlay.lines.end(),
                                             [](const auto& line)
                                             {
                                                 return line.depth == arc::render::debug_overlay_depth_mode::tested &&
                                                        line.color[0] > 0.9f && line.color[1] > 0.5f &&
                                                        line.color[1] < 0.8f;
                                             });
    REQUIRE(frustum_lines == 12);

    const auto hidden = arc::editor::build_editor_gizmo_overlay(registry, scene_camera, view_camera,
                                                                {.tool = arc::editor::editor_tool::select,
                                                                 .show_component_gizmos = false,
                                                                 .viewport_width = 1280,
                                                                 .viewport_height = 720});
    REQUIRE(hidden.lines.empty());
}

TEST_CASE("editor grid is adaptive and remains anchored to world axes")
{
    arc::scene::camera_component camera;
    arc::scene::transform_component camera_transform;
    camera_transform.position = {12.0f, 8.0f, -17.0f};
    arc::render::debug_overlay_stream near_grid;
    arc::editor::append_editor_grid_overlay(near_grid, camera, camera_transform, 600);
    REQUIRE(near_grid.lines.size() >= 4);

    camera_transform.position[1] = 800.0f;
    arc::render::debug_overlay_stream far_grid;
    arc::editor::append_editor_grid_overlay(far_grid, camera, camera_transform, 600);
    REQUIRE(far_grid.lines.size() >= 4);
    const float near_spacing = std::abs(near_grid.lines[2].start[2] - near_grid.lines[0].start[2]);
    const float far_spacing = std::abs(far_grid.lines[2].start[2] - far_grid.lines[0].start[2]);
    REQUIRE(far_spacing > near_spacing);
}

TEST_CASE("editor gizmo drags follow each projected positive axis")
{
    arc::ecs::world registry;
    const auto camera_entity = registry.create();
    arc::scene::transform_component camera_transform;
    arc::editor::editor_camera_controller camera_controller;
    REQUIRE(camera_controller.place({4.0f, 3.0f, 5.0f}, arc::math::vector3f::zero));
    camera_controller.apply_to(camera_transform);
    registry.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    registry.emplace<arc::scene::camera_component>(camera_entity);
    const auto selected = registry.create();
    registry.emplace<arc::scene::transform_component>(selected);
    arc::scene::update_world_transforms(registry);

    for (const auto tool : {arc::editor::editor_tool::translate, arc::editor::editor_tool::scale})
    {
        const arc::editor::editor_gizmo_context context{.tool = tool, .viewport_width = 800, .viewport_height = 600};
        arc::math::vector2f direction;
        REQUIRE(arc::editor::editor_gizmo_drag_direction(registry, selected, camera_entity, context,
                                                         arc::editor::gizmo_axis::z, 400.0f, 300.0f, direction));
        REQUIRE(direction[0] < 0.0f);
        REQUIRE(arc::math::dot(arc::math::vector2f{-10.0f, 0.0f}, direction) > 0.0f);
        REQUIRE(arc::math::length(direction) == Catch::Approx(1.0f));
    }

    const arc::editor::editor_gizmo_context rotation_context{
        .tool = arc::editor::editor_tool::rotate, .viewport_width = 800, .viewport_height = 600};
    arc::math::vector2f rotation_direction;
    REQUIRE(arc::editor::editor_gizmo_drag_direction(registry, selected, camera_entity, rotation_context,
                                                     arc::editor::gizmo_axis::z, 400.0f, 200.0f, rotation_direction));
    REQUIRE(arc::math::length(rotation_direction) == Catch::Approx(1.0f));
}

TEST_CASE("editor picking hits bounded entities")
{
    arc::ecs::world scene;
    const auto terrain = scene.create();
    scene.emplace<arc::scene::transform_component>(terrain);
    scene.emplace<arc::scene::terrain_component>(terrain);
    scene.emplace<arc::scene::bounds_component>(
        terrain,
        arc::geometric::box3f{arc::geometric::point3f{-100.0f, -100.0f, -100.0f},
                              arc::geometric::point3f{100.0f, 100.0f, 100.0f}},
        arc::geometric::box3f{}, true);
    const auto entity = scene.create();
    scene.emplace<arc::scene::transform_component>(entity);
    scene.emplace<arc::scene::bounds_component>(
        entity,
        arc::geometric::box3f{arc::geometric::point3f{-1.0f, -1.0f, -1.0f}, arc::geometric::point3f{1.0f, 1.0f, 1.0f}},
        arc::geometric::box3f{}, true);

    const arc::editor::editor_ray ray{.origin = arc::math::vector3f{0.0f, 0.0f, 5.0f},
                                      .direction = arc::math::vector3f{0.0f, 0.0f, -1.0f}};
    REQUIRE(arc::editor::pick_bounded_entity(scene, ray) == terrain);

    scene.emplace<arc::scene::active_component>(entity, false);
    REQUIRE(arc::editor::pick_bounded_entity(scene, ray) == terrain);

    float distance{};
    REQUIRE(arc::editor::intersect_ray_box(
        ray,
        arc::geometric::box3f{arc::geometric::point3f{-1.0f, -1.0f, -1.0f}, arc::geometric::point3f{1.0f, 1.0f, 1.0f}},
        distance));
    REQUIRE(distance == Catch::Approx(4.0f));
}

TEST_CASE("frame selected focuses parented entities in world space and keeps the orbit pivot")
{
    arc::ecs::world scene;
    const auto parent = scene.create();
    const auto selected = scene.create();

    arc::scene::transform_component parent_transform;
    parent_transform.position = {10.0f, 4.0f, -6.0f};
    scene.emplace<arc::scene::transform_component>(parent, parent_transform);

    arc::scene::transform_component child_transform;
    child_transform.position = {2.0f, -1.0f, 3.0f};
    scene.emplace<arc::scene::transform_component>(selected, child_transform);
    scene.emplace<arc::scene::bounds_component>(
        selected,
        arc::geometric::box3f{arc::geometric::point3f{-1.0f, -2.0f, -3.0f},
                              arc::geometric::point3f{3.0f, 2.0f, 1.0f}});

    REQUIRE(arc::scene::reparent(scene, selected, parent, {}, arc::scene::reparent_transform_policy::preserve_local));
    REQUIRE(scene.get<arc::scene::transform_component>(selected).dirty);

    arc::editor::editor_camera_controller camera;
    REQUIRE(arc::editor::focus_selected_entity(scene, selected, camera));

    const arc::math::vector3f expected_focus{13.0f, 3.0f, -4.0f};
    REQUIRE(camera.focus_point()[0] == Catch::Approx(expected_focus[0]));
    REQUIRE(camera.focus_point()[1] == Catch::Approx(expected_focus[1]));
    REQUIRE(camera.focus_point()[2] == Catch::Approx(expected_focus[2]));
    REQUIRE_FALSE(scene.get<arc::scene::transform_component>(selected).dirty);

    const float orbit_radius = camera.distance();
    camera.orbit(37.0f, -19.0f);

    arc::scene::transform_component camera_transform;
    camera.apply_to(camera_transform);
    const auto to_focus = arc::math::sub(expected_focus, camera_transform.position);
    REQUIRE(arc::math::length(to_focus) == Catch::Approx(orbit_radius).margin(0.0001f));
    REQUIRE(arc::math::dot(arc::scene::forward_direction(camera_transform), arc::math::normalize(to_focus)) ==
            Catch::Approx(1.0f).margin(0.0001f));
}

TEST_CASE("editor camera controller orbits pans and zooms")
{
    arc::editor::editor_camera_controller camera;
    arc::scene::transform_component transform;

    camera.focus({0.0f, 0.0f, 0.0f}, 2.0f);
    const float focused_distance = camera.distance();
    arc::scene::transform_component before_zoom;
    camera.apply_to(before_zoom);
    const auto before_zoom_forward = arc::scene::forward_direction(before_zoom);
    camera.zoom(1.0f);
    arc::scene::transform_component after_zoom;
    camera.apply_to(after_zoom);
    REQUIRE(camera.distance() == Catch::Approx(focused_distance));
    const auto zoom_translation = arc::math::sub(after_zoom.position, before_zoom.position);
    REQUIRE(arc::math::length(zoom_translation) == Catch::Approx(1.5f));
    REQUIRE(arc::math::dot(zoom_translation, before_zoom_forward) == Catch::Approx(1.5f));

    camera.orbit(24.0f, -12.0f);
    camera.pan(10.0f, 5.0f);
    camera.apply_to(transform);
    REQUIRE(transform.dirty);
    REQUIRE(arc::math::length(transform.position) > 0.01f);

    camera.orbit(40.0f, 0.0f);
    camera.apply_to(transform);
    const auto right_after_yaw = arc::math::rotate(transform.rotation, arc::math::vector3f{1.0f, 0.0f, 0.0f});
    REQUIRE(right_after_yaw[1] == Catch::Approx(0.0f).margin(0.00001f));
    camera.orbit(0.0f, 30.0f);
    camera.apply_to(transform);
    const auto right_after_pitch = arc::math::rotate(transform.rotation, arc::math::vector3f{1.0f, 0.0f, 0.0f});
    REQUIRE(arc::math::dot(right_after_yaw, right_after_pitch) == Catch::Approx(1.0f).margin(0.00001f));

    arc::editor::editor_camera_controller direction_test;
    arc::scene::transform_component before_up;
    direction_test.focus({0.0f, 0.0f, 0.0f}, 2.0f);
    direction_test.apply_to(before_up);
    direction_test.orbit(20.0f, 0.0f);
    arc::scene::transform_component after_right;
    direction_test.apply_to(after_right);
    REQUIRE(after_right.position[0] < before_up.position[0]);
    direction_test.orbit(0.0f, 20.0f);
    arc::scene::transform_component after_up;
    direction_test.apply_to(after_up);
    REQUIRE(after_up.position[1] > before_up.position[1]);
    const auto right = arc::math::rotate(after_up.rotation, arc::math::vector3f{1.0f, 0.0f, 0.0f});
    REQUIRE(right[1] == Catch::Approx(0.0f).margin(0.00001f));

    arc::editor::editor_camera_controller synchronized;
    synchronized.synchronize_from(after_up);
    arc::scene::transform_component synchronized_transform;
    synchronized.apply_to(synchronized_transform);
    REQUIRE(arc::math::dot(arc::scene::forward_direction(after_up),
                           arc::scene::forward_direction(synchronized_transform)) ==
            Catch::Approx(1.0f).margin(0.00001f));

    arc::editor::editor_camera_controller vertical_direction;
    vertical_direction.focus({0.0f, 0.0f, 0.0f}, 2.0f);
    arc::scene::transform_component before_vertical_drag;
    vertical_direction.apply_to(before_vertical_drag);
    const auto before_upward_drag_forward = arc::scene::forward_direction(before_vertical_drag);
    vertical_direction.orbit(0.0f, -20.0f);
    arc::scene::transform_component after_upward_drag;
    vertical_direction.apply_to(after_upward_drag);
    REQUIRE(after_upward_drag.position[1] < before_vertical_drag.position[1]);
    const auto upward_drag_forward = arc::scene::forward_direction(after_upward_drag);
    REQUIRE(upward_drag_forward[1] > before_upward_drag_forward[1]);
}

TEST_CASE("editor camera free look rotates in place without roll")
{
    arc::editor::editor_camera_controller camera;
    camera.focus({3.0f, 2.0f, -4.0f}, 3.0f);

    arc::scene::transform_component before;
    camera.apply_to(before);
    const auto initial_position = before.position;
    const auto initial_focus = camera.focus_point();

    camera.look(36.0f, -24.0f);
    arc::scene::transform_component after;
    camera.apply_to(after);

    REQUIRE(arc::math::length(arc::math::sub(after.position, initial_position)) == Catch::Approx(0.0f).margin(0.0001f));
    REQUIRE(arc::math::length(arc::math::sub(camera.focus_point(), initial_focus)) ==
            Catch::Approx(0.0f).margin(0.0001f));

    const auto looked_forward = arc::scene::forward_direction(after);
    const auto position_before_dolly = after.position;
    camera.zoom(2.0f);
    camera.apply_to(after);
    const auto dolly_translation = arc::math::sub(after.position, position_before_dolly);
    REQUIRE(arc::math::dot(dolly_translation, looked_forward) == Catch::Approx(3.0f).margin(0.0001f));
    REQUIRE(arc::math::length(arc::math::cross(dolly_translation, looked_forward)) ==
            Catch::Approx(0.0f).margin(0.0001f));
    REQUIRE(arc::math::length(arc::math::sub(camera.focus_point(), initial_focus)) ==
            Catch::Approx(0.0f).margin(0.0001f));

    const float orbit_radius = arc::math::length(arc::math::sub(initial_focus, after.position));
    camera.orbit(12.0f, -8.0f);
    camera.apply_to(after);
    REQUIRE(arc::math::length(arc::math::sub(initial_focus, after.position)) ==
            Catch::Approx(orbit_radius).margin(0.0001f));
    const auto direction_to_focus = arc::math::normalize(arc::math::sub(initial_focus, after.position));
    REQUIRE(arc::math::dot(arc::scene::forward_direction(after), direction_to_focus) ==
            Catch::Approx(1.0f).margin(0.0001f));
    const auto position_after_orbit = after.position;

    for (int index = 0; index < 128; ++index)
        camera.look(index % 2 == 0 ? 17.0f : -9.0f, index % 3 == 0 ? 31.0f : -14.0f);
    camera.look(0.0f, -100000.0f);
    camera.apply_to(after);

    const auto right = arc::math::rotate(after.rotation, arc::math::vector3f{1.0f, 0.0f, 0.0f});
    const auto up = arc::scene::up_direction(after);
    REQUIRE(right[1] == Catch::Approx(0.0f).margin(0.0001f));
    REQUIRE(up[1] > 0.0f);
    REQUIRE(arc::math::length(arc::math::sub(after.position, position_after_orbit)) ==
            Catch::Approx(0.0f).margin(0.0001f));
}

TEST_CASE("camera transform forward matches the rendered view projection")
{
    arc::editor::editor_camera_controller camera_controller;
    camera_controller.focus({4.0f, 3.0f, -7.0f}, 5.0f);
    camera_controller.orbit(53.0f, -31.0f);

    arc::scene::transform_component transform;
    camera_controller.apply_to(transform);
    const auto forward = arc::scene::forward_direction(transform);
    const auto view = arc::scene::world_view_matrix(transform);
    const auto point_ahead = arc::math::add(transform.position, arc::math::mul(forward, 10.0f));
    const auto view_point_ahead = arc::math::transform_point(view, point_ahead);
    REQUIRE(view_point_ahead[0] == Catch::Approx(0.0f).margin(0.0001f));
    REQUIRE(view_point_ahead[1] == Catch::Approx(0.0f).margin(0.0001f));
    REQUIRE(view_point_ahead[2] == Catch::Approx(-10.0f).margin(0.0001f));

    arc::scene::camera_component camera;
    const auto view_projection = arc::scene::view_projection(camera, transform, 16.0f / 9.0f);
    arc::math::matrix4f inverse_view_projection;
    REQUIRE(arc::math::try_inverse(view_projection, inverse_view_projection));
    const auto near_center = arc::math::transform_point(inverse_view_projection, arc::math::vector3f{0.0f, 0.0f, 0.0f});
    const auto far_center = arc::math::transform_point(inverse_view_projection, arc::math::vector3f{0.0f, 0.0f, 1.0f});
    const auto rendered_forward = arc::math::normalize(arc::math::sub(far_center, near_center));
    REQUIRE(arc::math::dot(rendered_forward, forward) == Catch::Approx(1.0f).margin(0.0001f));
}

TEST_CASE("viewport rays use camera world space and pixel centers")
{
    arc::editor::editor_viewport viewport;
    viewport.set_size(101.0f, 101.0f);
    arc::scene::camera_component camera;
    arc::scene::transform_component transform;
    transform.world = arc::math::translation(arc::math::vector3f{8.0f, 3.0f, 2.0f});
    transform.dirty = false;

    const auto center = arc::editor::screen_ray_from_camera(camera, transform, viewport, 50.0f, 50.0f);
    REQUIRE(center.origin[0] == Catch::Approx(8.0f));
    REQUIRE(center.origin[1] == Catch::Approx(3.0f));
    REQUIRE(center.direction[0] == Catch::Approx(0.0f).margin(0.00001f));
    REQUIRE(center.direction[1] == Catch::Approx(0.0f).margin(0.00001f));
    REQUIRE(center.direction[2] == Catch::Approx(-1.0f).margin(0.00001f));

    camera.projection = arc::scene::camera_projection::orthographic;
    camera.orthographic_height = 10.0f;
    const auto corner = arc::editor::screen_ray_from_camera(camera, transform, viewport, 100.0f, 0.0f);
    REQUIRE(corner.origin[0] > center.origin[0]);
    REQUIRE(corner.origin[1] > center.origin[1]);
    REQUIRE(corner.direction[2] == Catch::Approx(-1.0f).margin(0.00001f));
}

TEST_CASE("exact scene picking selects the nearest surface instead of terrain bounds")
{
    arc::render::renderer renderer;
    const auto cube_mesh = renderer.create_mesh(arc::render::make_cube_mesh(1.0f));
    arc::ecs::world scene;

    const auto terrain_entity = scene.create();
    scene.emplace<arc::scene::transform_component>(terrain_entity);
    auto& terrain = scene.emplace<arc::scene::terrain_component>(terrain_entity);
    terrain.size = 10.0f;
    terrain.subdivisions = 2;
    terrain.heights.assign(9, 0.0f);
    terrain.layer_weights.assign(9, std::array<std::uint8_t, 4>{255, 0, 0, 0});
    scene.emplace<arc::scene::bounds_component>(terrain_entity,
                                                arc::geometric::box3f{arc::geometric::point3f{-5.0f, -10.0f, -5.0f},
                                                                      arc::geometric::point3f{5.0f, 10.0f, 5.0f}},
                                                arc::geometric::box3f{}, true);

    const auto cube = scene.create();
    arc::scene::transform_component cube_transform;
    cube_transform.set_position({0.0f, 2.0f, 0.0f});
    scene.emplace<arc::scene::transform_component>(cube, cube_transform);
    scene.emplace<arc::scene::mesh_renderer_component>(cube, cube_mesh);
    scene.emplace<arc::scene::bounds_component>(
        cube,
        arc::geometric::box3f{arc::geometric::point3f{-0.5f, -0.5f, -0.5f}, arc::geometric::point3f{0.5f, 0.5f, 0.5f}},
        arc::geometric::box3f{}, true);
    arc::scene::update_world_transforms(scene);

    const arc::editor::editor_ray ray{.origin = {0.0f, 10.0f, 0.0f}, .direction = {0.0f, -1.0f, 0.0f}};
    const auto foreground = arc::editor::pick_scene_entity(scene, renderer, ray);
    REQUIRE(foreground.entity == cube);
    REQUIRE(foreground.exact);
    REQUIRE_FALSE(foreground.background);

    scene.get<arc::scene::transform_component>(cube).set_position({0.0f, -2.0f, 0.0f});
    arc::scene::update_world_transforms(scene);
    const auto background = arc::editor::pick_scene_entity(scene, renderer, ray);
    REQUIRE(background.entity == terrain_entity);
    REQUIRE(background.exact);
    REQUIRE(background.background);
}

TEST_CASE("editor euler conversion keeps pure y rotation stable")
{
    const auto rotation = arc::editor::quaternion_from_euler_degrees({0.0f, 135.0f, 0.0f});
    const auto euler = arc::editor::euler_degrees_from_quaternion(rotation);

    REQUIRE(euler[0] == Catch::Approx(0.0f).margin(0.001f));
    REQUIRE(euler[1] == Catch::Approx(135.0f).margin(0.001f));
    REQUIRE(euler[2] == Catch::Approx(0.0f).margin(0.001f));
}

TEST_CASE("editor euler conversion round trips mixed rotation")
{
    const arc::math::vector3f input{20.0f, 135.0f, 10.0f};
    const auto rotation = arc::editor::quaternion_from_euler_degrees(input);
    const auto euler = arc::editor::euler_degrees_from_quaternion(rotation);

    REQUIRE(euler[0] == Catch::Approx(input[0]).margin(0.001f));
    REQUIRE(euler[1] == Catch::Approx(input[1]).margin(0.001f));
    REQUIRE(euler[2] == Catch::Approx(input[2]).margin(0.001f));
}

TEST_CASE("editor default sun rotation points downward at an angle")
{
    arc::scene::transform_component transform;
    transform.rotation = arc::editor::quaternion_from_euler_degrees({-50.0f, -35.0f, 0.0f});

    const auto direction = arc::scene::forward_direction(transform);

    REQUIRE(direction[1] < -0.25f);
    REQUIRE(std::abs(direction[0]) > 0.10f);
    REQUIRE(direction[2] < -0.10f);
}

TEST_CASE("snap to floor drops a bounded entity and participates in undo")
{
    auto host = std::make_shared<arc::editor::arc_host>(std::make_unique<arc::render::renderer>());
    REQUIRE(host);
    REQUIRE(host->execute(arc::editor::host_new_scene_command{.name = "Snap Floor"}).succeeded);
    const auto created =
        host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube});
    REQUIRE(created.succeeded);
    const auto selected = host->selected_entity_snapshot();
    REQUIRE(selected.entity.valid());
    REQUIRE(selected.transform.has_value());

    auto raised = *selected.transform;
    raised.position.y = 4.0f;
    REQUIRE(host->execute(arc::editor::host_set_transform_command{.entity = selected.entity, .transform = raised})
                .succeeded);
    const auto before_snap = host->selected_entity_snapshot();
    REQUIRE(before_snap.transform.has_value());
    REQUIRE(before_snap.transform->position.y == Catch::Approx(4.0f));

    const auto snapped = host->execute(arc::editor::host_snap_to_floor_command{.entity = selected.entity});
    REQUIRE(snapped.succeeded);
    const auto after_snap = host->selected_entity_snapshot();
    REQUIRE(after_snap.transform.has_value());
    REQUIRE(after_snap.transform->position.y < 1.0f);

    REQUIRE(host->execute(arc::editor::host_history_undo_command{}).succeeded);
    const auto restored = host->selected_entity_snapshot();
    REQUIRE(restored.transform.has_value());
    REQUIRE(restored.transform->position.y == Catch::Approx(4.0f));
}

TEST_CASE("editor tool shortcuts update active tool")
{
    arc::input::input_manager input;
    input.bind_action("tool.select", {.device = arc::input::input_device_type::keyboard, .code = 'Q'});
    input.bind_action("tool.translate", {.device = arc::input::input_device_type::keyboard, .code = 'W'});
    input.bind_action("tool.rotate", {.device = arc::input::input_device_type::keyboard, .code = 'E'});
    input.bind_action("tool.scale", {.device = arc::input::input_device_type::keyboard, .code = 'R'});

    auto tool = arc::editor::editor_tool::select;
    input.begin_frame();
    input.process_event({.type = arc::framework::event_type::key_down, .key_code = 'W'});
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::translate);

    input.begin_frame();
    input.process_event({.type = arc::framework::event_type::key_down, .key_code = 'R'});
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::scale);

    REQUIRE(arc::editor::editor_tool_from_shortcut('Q') == arc::editor::editor_tool::select);
    REQUIRE(arc::editor::editor_tool_from_shortcut('W') == arc::editor::editor_tool::translate);
    REQUIRE(arc::editor::editor_tool_from_shortcut('E') == arc::editor::editor_tool::rotate);
    REQUIRE(arc::editor::editor_tool_from_shortcut('R') == arc::editor::editor_tool::scale);
    REQUIRE_FALSE(arc::editor::editor_tool_from_shortcut('T').has_value());
}

TEST_CASE("editor sun controller rotates around stable yaw and pitch axes")
{
    arc::scene::transform_component transform;
    transform.rotation = arc::editor::quaternion_from_euler_degrees({-40.0f, 25.0f, 0.0f});

    arc::editor::editor_sun_controller controller;
    controller.synchronize_from(transform);
    controller.rotate(30.0f, -15.0f);
    controller.apply_to(transform);

    const auto first_direction = arc::scene::forward_direction(transform);
    REQUIRE(first_direction[1] < 0.0f);

    for (int iteration = 0; iteration < 100; ++iteration)
        controller.rotate(4.0f, -3.0f);
    controller.apply_to(transform);
    const auto euler = arc::editor::euler_degrees_from_quaternion(transform.rotation);
    REQUIRE(std::abs(euler[2]) < 0.001f);
    REQUIRE(arc::math::length(arc::scene::forward_direction(transform)) == Catch::Approx(1.0f));
}

TEST_CASE("editor can add a selected primitive mesh entity")
{
    arc::editor::editor_scene_state scene;
    arc::render::renderer renderer;
    arc::render::material_descriptor imported_material;
    imported_material.name = "Imported Mesh Material";
    imported_material.base_color = {1.0f, 0.72f, 0.05f, 1.0f};
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

TEST_CASE("editor primitives bind the authored built-in default phong material")
{
    const auto root = std::filesystem::temp_directory_path() /
                      ("arc_editor_default_primitive_material_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto builtin_root = root / "builtin";
    std::filesystem::create_directories(builtin_root / "materials");

    auto authored = arc::editor::make_default_material_asset("Default Phong");
    authored.path = builtin_root / "materials" / "default_phong.arcmat";
    authored.material.base_color = {0.82f, 0.84f, 0.78f, 1.0f};
    authored.material.roughness = 0.62f;
    std::string message;
    REQUIRE(arc::editor::save_material_asset(authored, builtin_root, message));

    arc::editor::editor_asset_state assets;
    assets.builtin_roots.push_back(builtin_root);
    arc::editor::editor_scene_state scene;
    arc::render::renderer renderer;

    const auto default_material = arc::editor::create_default_primitive_material(scene, renderer, assets);
    REQUIRE(default_material.valid());
    REQUIRE(scene.primitive_material == default_material);
    REQUIRE(scene.primitive_material_asset.expected_type == arc::assets::asset_types::material);
    REQUIRE(scene.primitive_material_asset.path_hint == "builtin/materials/default_phong.arcmat");
    REQUIRE(scene.material_library.materials.size() == 1);
    REQUIRE(scene.material_library.materials.front().asset.name == "Default Phong");

    const auto entity = arc::editor::add_primitive_to_scene(scene, renderer, arc::editor::editor_primitive_type::cube);
    REQUIRE(scene.scene.alive(entity));
    REQUIRE(scene.scene.get<arc::scene::mesh_renderer_component>(entity).material == default_material);
    const auto* binding = arc::editor::find_asset_binding(scene, arc::editor::entity_guid_of(scene, entity));
    REQUIRE(binding != nullptr);
    REQUIRE(binding->material.expected_type == arc::assets::asset_types::material);
    REQUIRE(binding->material.path_hint == "builtin/materials/default_phong.arcmat");

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("editor Ocean creation uses the Water clipmap and optical transmission material")
{
    arc::editor::editor_scene_state scene;
    arc::render::renderer renderer;
    const auto ocean = arc::editor::add_water_to_scene(scene, renderer);
    REQUIRE(scene.scene.alive(ocean));
    REQUIRE(scene.scene.has<arc::scene::water_component>(ocean));
    REQUIRE(scene.scene.has<arc::scene::mesh_renderer_component>(ocean));
    const auto& mesh_renderer = scene.scene.get<arc::scene::mesh_renderer_component>(ocean);
    const auto* mesh = renderer.mesh_data_for(mesh_renderer.mesh.conventional);
    REQUIRE(mesh != nullptr);
    CHECK(mesh->name == "Water Ocean Clipmap");
    CHECK(mesh->vertices.size() > 4u);
    CHECK_FALSE(mesh_renderer.casts_shadows);

    const auto packet = renderer.frame_queue().commit(1);
    const auto upload_event = std::ranges::find_if(
        packet.events,
        [&](const auto& event)
        {
            if (event.type() != arc::render::render_event_type::material_upload) return false;
            return std::get<arc::render::material_upload_event>(event.payload).handle == mesh_renderer.material;
        });
    REQUIRE(upload_event != packet.events.end());
    const auto& material = *std::get<arc::render::material_upload_event>(upload_event->payload).material;
    CHECK(material.shading_model == arc::render::material_shading_model::transmission);
    CHECK(material.render_path == arc::render::material_render_path::clustered_forward);
    CHECK(material.alpha_mode == arc::render::material_alpha_mode::blend);
    CHECK(material.index_of_refraction == Catch::Approx(1.333f));
    CHECK(material.transmission_factor > 0.0f);
    CHECK(material.attenuation_distance > 0.0f);

    auto& water = scene.scene.get<arc::scene::water_component>(ocean);
    water.settings.appearance.absorption = {0.4f, 0.2f, 0.1f};
    water.settings.appearance.refraction_strength = 0.2f;
    REQUIRE(arc::editor::synchronize_water_render_material(scene, renderer, ocean));
    const auto update_packet = renderer.frame_queue().commit(2);
    const auto update_event = std::ranges::find_if(
        update_packet.events,
        [&](const auto& event)
        {
            if (event.type() != arc::render::render_event_type::material_upload) return false;
            return std::get<arc::render::material_upload_event>(event.payload).handle == mesh_renderer.material;
        });
    REQUIRE(update_event != update_packet.events.end());
    const auto& updated = *std::get<arc::render::material_upload_event>(update_event->payload).material;
    CHECK(updated.transmission_factor == Catch::Approx(0.7f));
    CHECK(updated.attenuation_color[0] < updated.attenuation_color[2]);

    const auto first_material = mesh_renderer.material;
    const auto second_ocean = arc::editor::add_water_to_scene(scene, renderer);
    REQUIRE(scene.scene.alive(second_ocean));
    CHECK(scene.scene.get<arc::scene::mesh_renderer_component>(second_ocean).material != first_material);
}
