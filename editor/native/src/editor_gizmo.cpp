#include <arc/editor/editor_gizmo.h>

#include <arc/scene/transforms.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace arc::editor
{
namespace
{
constexpr float gizmo_hit_radius = 11.0f;
constexpr std::uint32_t rotation_segments = 64;
constexpr std::uint32_t radial_segments = 10;
constexpr float shaft_length = 0.76f;
constexpr float shaft_radius = 0.022f;
constexpr float arrow_radius = 0.075f;
constexpr float scale_handle_extent = 0.075f;
constexpr float uniform_scale_handle_extent = 0.09f;
constexpr float ring_half_width = 0.025f;
constexpr float highlighted_width_scale = 1.45f;
constexpr int grid_half_line_count = 50;
constexpr int grid_major_interval = 10;
constexpr float grid_target_pixel_spacing = 32.0f;
constexpr float grid_height = 0.003f;

constexpr std::array<math::vector3f, 3> canonical_axes{
    math::vector3f{1.0f, 0.0f, 0.0f}, math::vector3f{0.0f, 1.0f, 0.0f}, math::vector3f{0.0f, 0.0f, 1.0f}};
constexpr std::array<math::vector4f, 3> axis_colors{math::vector4f{0.95f, 0.12f, 0.10f, 1.0f},
                                                    math::vector4f{0.18f, 0.86f, 0.24f, 1.0f},
                                                    math::vector4f{0.14f, 0.48f, 1.0f, 1.0f}};
constexpr math::vector4f highlighted_color{1.0f, 0.86f, 0.20f, 1.0f};
constexpr math::vector4f bounds_color{1.0f, 0.55f, 0.08f, 0.82f};
constexpr math::vector4f uniform_scale_color{0.86f, 0.88f, 0.91f, 1.0f};

math::vector3f matrix_axis(const math::matrix4f& matrix, std::size_t column) noexcept
{
    return math::normalize(math::vector3f{matrix(0, column), matrix(1, column), matrix(2, column)});
}

std::array<math::vector3f, 3> gizmo_axes(const scene::transform_component& transform,
                                         gizmo_coordinate_space space) noexcept
{
    if (space == gizmo_coordinate_space::world) return canonical_axes;
    return {matrix_axis(transform.world, 0), matrix_axis(transform.world, 1), matrix_axis(transform.world, 2)};
}

math::vector4f color_for_axis(std::size_t index, gizmo_axis highlighted) noexcept
{
    return highlighted == static_cast<gizmo_axis>(index + 1) ? highlighted_color : axis_colors[index];
}

std::array<math::vector3f, 2> perpendicular_basis(const math::vector3f& axis) noexcept
{
    const auto reference =
        std::abs(axis[1]) < 0.9f ? math::vector3f{0.0f, 1.0f, 0.0f} : math::vector3f{1.0f, 0.0f, 0.0f};
    const auto tangent = math::normalize(math::cross(axis, reference));
    return {tangent, math::normalize(math::cross(axis, tangent))};
}

void append_triangle(render::debug_overlay_stream& stream, const math::vector3f& first, const math::vector3f& second,
                     const math::vector3f& third, const math::vector4f& color)
{
    stream.triangles.push_back({.first = first,
                                .second = second,
                                .third = third,
                                .color = color,
                                .depth = render::debug_overlay_depth_mode::always});
}

void append_axis_shaft(render::debug_overlay_stream& stream, const math::vector3f& origin, const math::vector3f& axis,
                       float scale, float radius_scale, const math::vector4f& color)
{
    const auto basis = perpendicular_basis(axis);
    const auto end = math::add(origin, math::mul(axis, scale * shaft_length));
    const float radius = scale * shaft_radius * radius_scale;
    for (std::uint32_t segment = 0; segment < radial_segments; ++segment)
    {
        const float first_angle = math::tau<float> * static_cast<float>(segment) / static_cast<float>(radial_segments);
        const float second_angle =
            math::tau<float> * static_cast<float>(segment + 1) / static_cast<float>(radial_segments);
        const auto radial = [&](float angle) {
            return math::mul(math::add(math::mul(basis[0], std::cos(angle)), math::mul(basis[1], std::sin(angle))),
                             radius);
        };
        const auto start_first = math::add(origin, radial(first_angle));
        const auto start_second = math::add(origin, radial(second_angle));
        const auto end_first = math::add(end, radial(first_angle));
        const auto end_second = math::add(end, radial(second_angle));
        append_triangle(stream, start_first, end_first, end_second, color);
        append_triangle(stream, start_first, end_second, start_second, color);
    }
}

void append_arrow_head(render::debug_overlay_stream& stream, const math::vector3f& origin, const math::vector3f& axis,
                       float scale, float radius_scale, const math::vector4f& color)
{
    const auto basis = perpendicular_basis(axis);
    const auto base = math::add(origin, math::mul(axis, scale * shaft_length));
    const auto tip = math::add(origin, math::mul(axis, scale));
    const float radius = scale * arrow_radius * radius_scale;
    for (std::uint32_t segment = 0; segment < radial_segments; ++segment)
    {
        const float first_angle = math::tau<float> * static_cast<float>(segment) / static_cast<float>(radial_segments);
        const float second_angle =
            math::tau<float> * static_cast<float>(segment + 1) / static_cast<float>(radial_segments);
        const auto radial = [&](float angle) {
            return math::mul(math::add(math::mul(basis[0], std::cos(angle)), math::mul(basis[1], std::sin(angle))),
                             radius);
        };
        const auto first = math::add(base, radial(first_angle));
        const auto second = math::add(base, radial(second_angle));
        append_triangle(stream, first, tip, second, color);
        append_triangle(stream, first, second, base, color);
    }
}

void append_scale_handle(render::debug_overlay_stream& stream, const math::vector3f& origin, const math::vector3f& axis,
                         float scale, float radius_scale, const math::vector4f& color)
{
    const auto basis = perpendicular_basis(axis);
    const auto center = math::add(origin, math::mul(axis, scale * (1.0f - scale_handle_extent)));
    const auto axis_extent = math::mul(axis, scale * scale_handle_extent * radius_scale);
    const auto tangent_extent = math::mul(basis[0], scale * scale_handle_extent * radius_scale);
    const auto bitangent_extent = math::mul(basis[1], scale * scale_handle_extent * radius_scale);
    std::array<math::vector3f, 8> corners{};
    for (std::size_t index = 0; index < corners.size(); ++index)
    {
        const auto axis_offset = math::mul(axis_extent, (index & 1u) != 0u ? 1.0f : -1.0f);
        const auto tangent_offset = math::mul(tangent_extent, (index & 2u) != 0u ? 1.0f : -1.0f);
        const auto bitangent_offset = math::mul(bitangent_extent, (index & 4u) != 0u ? 1.0f : -1.0f);
        corners[index] = math::add(center, math::add(axis_offset, math::add(tangent_offset, bitangent_offset)));
    }
    constexpr std::array<std::array<std::size_t, 3>, 12> faces{std::array<std::size_t, 3>{0, 2, 3},
                                                               {0, 3, 1},
                                                               {4, 5, 7},
                                                               {4, 7, 6},
                                                               {0, 1, 5},
                                                               {0, 5, 4},
                                                               {2, 6, 7},
                                                               {2, 7, 3},
                                                               {0, 4, 6},
                                                               {0, 6, 2},
                                                               {1, 3, 7},
                                                               {1, 7, 5}};
    for (const auto& face : faces)
        append_triangle(stream, corners[face[0]], corners[face[1]], corners[face[2]], color);
}

void append_uniform_scale_handle(render::debug_overlay_stream& stream, const math::vector3f& origin,
                                 const std::array<math::vector3f, 3>& axes, float scale, bool highlighted)
{
    const float extent = scale * uniform_scale_handle_extent * (highlighted ? highlighted_width_scale : 1.0f);
    std::array<math::vector3f, 8> corners{};
    for (std::size_t index = 0; index < corners.size(); ++index)
    {
        auto offset = math::vector3f::zero;
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
            offset =
                math::add(offset, math::mul(axes[axis], (index & (std::size_t{1} << axis)) != 0u ? extent : -extent));
        corners[index] = math::add(origin, offset);
    }
    constexpr std::array<std::array<std::size_t, 3>, 12> faces{std::array<std::size_t, 3>{0, 2, 3},
                                                               {0, 3, 1},
                                                               {4, 5, 7},
                                                               {4, 7, 6},
                                                               {0, 1, 5},
                                                               {0, 5, 4},
                                                               {2, 6, 7},
                                                               {2, 7, 3},
                                                               {0, 4, 6},
                                                               {0, 6, 2},
                                                               {1, 3, 7},
                                                               {1, 7, 5}};
    const auto color = highlighted ? highlighted_color : uniform_scale_color;
    for (const auto& face : faces)
        append_triangle(stream, corners[face[0]], corners[face[1]], corners[face[2]], color);
}

void append_rotation_ring(render::debug_overlay_stream& stream, const math::vector3f& origin,
                          const math::vector3f& tangent, const math::vector3f& bitangent, float scale,
                          float width_scale, const math::vector4f& color)
{
    const float half_width = ring_half_width * width_scale;
    const float inner_radius = scale * (1.0f - half_width);
    const float outer_radius = scale * (1.0f + half_width);
    const auto point = [&](float angle, float radius)
    {
        return math::add(
            origin,
            math::mul(math::add(math::mul(tangent, std::cos(angle)), math::mul(bitangent, std::sin(angle))), radius));
    };
    for (std::uint32_t segment = 0; segment < rotation_segments; ++segment)
    {
        const float first = math::tau<float> * static_cast<float>(segment) / static_cast<float>(rotation_segments);
        const float second = math::tau<float> * static_cast<float>(segment + 1) / static_cast<float>(rotation_segments);
        const auto inner_first = point(first, inner_radius);
        const auto outer_first = point(first, outer_radius);
        const auto inner_second = point(second, inner_radius);
        const auto outer_second = point(second, outer_radius);
        append_triangle(stream, inner_first, outer_first, outer_second, color);
        append_triangle(stream, inner_first, outer_second, inner_second, color);
    }
}

void append_bounds(render::debug_overlay_stream& stream, const geometric::box3f& bounds)
{
    const std::array<math::vector3f, 8> corners{math::vector3f{bounds.min[0], bounds.min[1], bounds.min[2]},
                                                math::vector3f{bounds.max[0], bounds.min[1], bounds.min[2]},
                                                math::vector3f{bounds.max[0], bounds.max[1], bounds.min[2]},
                                                math::vector3f{bounds.min[0], bounds.max[1], bounds.min[2]},
                                                math::vector3f{bounds.min[0], bounds.min[1], bounds.max[2]},
                                                math::vector3f{bounds.max[0], bounds.min[1], bounds.max[2]},
                                                math::vector3f{bounds.max[0], bounds.max[1], bounds.max[2]},
                                                math::vector3f{bounds.min[0], bounds.max[1], bounds.max[2]}};
    constexpr std::array<std::array<std::size_t, 2>, 12> edges{std::array<std::size_t, 2>{0, 1},
                                                               {1, 2},
                                                               {2, 3},
                                                               {3, 0},
                                                               {4, 5},
                                                               {5, 6},
                                                               {6, 7},
                                                               {7, 4},
                                                               {0, 4},
                                                               {1, 5},
                                                               {2, 6},
                                                               {3, 7}};
    for (const auto edge : edges)
        stream.lines.push_back(
            {corners[edge[0]], corners[edge[1]], bounds_color, render::debug_overlay_depth_mode::tested});
}

bool project_to_screen(const math::matrix4f& view_projection, const math::vector3f& point, std::uint32_t width,
                       std::uint32_t height, math::vector2f& screen) noexcept
{
    const float x = view_projection(0, 0) * point[0] + view_projection(0, 1) * point[1] +
                    view_projection(0, 2) * point[2] + view_projection(0, 3);
    const float y = view_projection(1, 0) * point[0] + view_projection(1, 1) * point[1] +
                    view_projection(1, 2) * point[2] + view_projection(1, 3);
    const float w = view_projection(3, 0) * point[0] + view_projection(3, 1) * point[1] +
                    view_projection(3, 2) * point[2] + view_projection(3, 3);
    if (!(w > 1.0e-5f)) return false;
    screen = {(x / w * 0.5f + 0.5f) * static_cast<float>(width), (0.5f - y / w * 0.5f) * static_cast<float>(height)};
    return true;
}

float distance_to_segment(const math::vector2f& point, const math::vector2f& start, const math::vector2f& end) noexcept
{
    const auto segment = math::sub(end, start);
    const float length_squared = math::length_squared(segment);
    if (length_squared <= 1.0e-6f) return std::numeric_limits<float>::max();
    const float amount = std::clamp(math::dot(math::sub(point, start), segment) / length_squared, 0.0f, 1.0f);
    return math::length(math::sub(point, math::add(start, math::mul(segment, amount))));
}

math::matrix4f gizmo_view_projection(const scene::camera_component& camera,
                                     const scene::transform_component& camera_transform,
                                     const editor_gizmo_context& context) noexcept
{
    const float aspect = static_cast<float>(std::max(1u, context.viewport_width)) /
                         static_cast<float>(std::max(1u, context.viewport_height));
    const auto projection =
        camera.projection == scene::camera_projection::orthographic
            ? scene::orthographic_rh_zo(camera.orthographic_height, aspect, camera.near_plane, camera.far_plane)
            : scene::perspective_rh_zo(camera.fov_y_radians, aspect, camera.near_plane, camera.far_plane);
    return math::matmul(projection, scene::world_view_matrix(camera_transform));
}
} // namespace

void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,
                                const scene::transform_component& camera_transform, std::uint32_t viewport_height)
{
    const float height = static_cast<float>(std::max(1u, viewport_height));
    const auto camera_position = scene::world_position(camera_transform);
    const float visible_height =
        camera.projection == scene::camera_projection::orthographic
            ? camera.orthographic_height
            : 2.0f * std::max(std::abs(camera_position[1]), 1.0f) * std::tan(camera.fov_y_radians * 0.5f);
    const float desired_spacing = std::max(0.001f, visible_height * grid_target_pixel_spacing / height);
    const float magnitude = std::pow(10.0f, std::floor(std::log10(desired_spacing)));
    const float normalized = desired_spacing / magnitude;
    const float step = normalized > 5.0f ? 10.0f : normalized > 2.0f ? 5.0f : normalized > 1.0f ? 2.0f : 1.0f;
    const float spacing = magnitude * step;
    const float center_x = std::floor(camera_position[0] / spacing) * spacing;
    const float center_z = std::floor(camera_position[2] / spacing) * spacing;
    const float extent = spacing * static_cast<float>(grid_half_line_count);

    constexpr math::vector4f minor_color{0.30f, 0.33f, 0.37f, 0.42f};
    constexpr math::vector4f major_color{0.42f, 0.45f, 0.50f, 0.62f};
    constexpr math::vector4f x_axis_color{0.76f, 0.22f, 0.18f, 0.90f};
    constexpr math::vector4f z_axis_color{0.18f, 0.40f, 0.78f, 0.90f};
    const auto grid_color = [&](float coordinate, float center, const math::vector4f& axis_color)
    {
        if (std::abs(coordinate) <= spacing * 0.25f) return axis_color;
        const auto world_line = static_cast<long long>(std::llround(coordinate / spacing));
        auto color = world_line % grid_major_interval == 0 ? major_color : minor_color;
        const float edge = std::clamp(std::abs(coordinate - center) / extent, 0.0f, 1.0f);
        color[3] *= 1.0f - edge * edge;
        return color;
    };

    for (int line = -grid_half_line_count; line <= grid_half_line_count; ++line)
    {
        const float x = center_x + static_cast<float>(line) * spacing;
        const float z = center_z + static_cast<float>(line) * spacing;
        stream.lines.push_back({.start = {center_x - extent, grid_height, z},
                                .end = {center_x + extent, grid_height, z},
                                .color = grid_color(z, center_z, x_axis_color),
                                .depth = render::debug_overlay_depth_mode::tested});
        stream.lines.push_back({.start = {x, grid_height, center_z - extent},
                                .end = {x, grid_height, center_z + extent},
                                .color = grid_color(x, center_x, z_axis_color),
                                .depth = render::debug_overlay_depth_mode::tested});
    }
}

float editor_gizmo_world_scale(const scene::camera_component& camera,
                               const scene::transform_component& camera_transform, const math::vector3f& world_position,
                               std::uint32_t viewport_height) noexcept
{
    const float height = static_cast<float>(std::max(1u, viewport_height));
    if (camera.projection == scene::camera_projection::orthographic)
        return std::max(0.001f, camera.orthographic_height * editor_gizmo_pixel_length / height);
    const auto camera_space = math::transform_point(scene::world_view_matrix(camera_transform), world_position);
    const float view_depth = std::max(0.01f, -camera_space[2]);
    return std::max(0.001f,
                    2.0f * view_depth * std::tan(camera.fov_y_radians * 0.5f) * editor_gizmo_pixel_length / height);
}

render::debug_overlay_stream build_editor_gizmo_overlay(const ecs::world& registry, ecs::entity selected,
                                                        ecs::entity camera_entity, const editor_gizmo_context& context)
{
    render::debug_overlay_stream stream;
    const auto* transform = registry.try_get<scene::transform_component>(selected);
    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);
    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);
    if (!transform || !camera || !camera_transform) return stream;
    // A terrain's scene-wide AABB projects as enormous lines through the
    // viewport and looks like geometry clipping. Terrain selection is already
    // represented by the surface/ObjectID and its dedicated brush overlay.
    if (!registry.has<scene::terrain_component>(selected))
        if (const auto* bounds = registry.try_get<scene::bounds_component>(selected))
            append_bounds(stream, transformed_bounds(bounds->local_bounds, *transform));
    if (context.tool == editor_tool::select) return stream;

    const auto origin = scene::world_position(*transform);
    const auto axes = gizmo_axes(*transform, context.coordinate_space);
    const float scale = editor_gizmo_world_scale(*camera, *camera_transform, origin, context.viewport_height);
    if (context.tool == editor_tool::rotate)
    {
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
        {
            const auto tangent = axes[(axis + 1) % 3];
            const auto bitangent = axes[(axis + 2) % 3];
            const bool highlighted = context.highlighted_axis == static_cast<gizmo_axis>(axis + 1);
            append_rotation_ring(stream, origin, tangent, bitangent, scale,
                                 highlighted ? highlighted_width_scale : 1.0f,
                                 color_for_axis(axis, context.highlighted_axis));
        }
        return stream;
    }
    for (std::size_t axis = 0; axis < axes.size(); ++axis)
    {
        const bool highlighted = context.highlighted_axis == static_cast<gizmo_axis>(axis + 1);
        const float width_scale = highlighted ? highlighted_width_scale : 1.0f;
        const auto color = color_for_axis(axis, context.highlighted_axis);
        append_axis_shaft(stream, origin, axes[axis], scale, width_scale, color);
        if (context.tool == editor_tool::translate)
            append_arrow_head(stream, origin, axes[axis], scale, width_scale, color);
        else
            append_scale_handle(stream, origin, axes[axis], scale, width_scale, color);
    }
    if (context.tool == editor_tool::scale)
        append_uniform_scale_handle(stream, origin, axes, scale, context.highlighted_axis == gizmo_axis::all);
    return stream;
}

gizmo_axis hit_test_editor_gizmo(const ecs::world& registry, ecs::entity selected, ecs::entity camera_entity,
                                 const editor_gizmo_context& context, float screen_x, float screen_y) noexcept
{
    const auto* transform = registry.try_get<scene::transform_component>(selected);
    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);
    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);
    if (!transform || !camera || !camera_transform || context.tool == editor_tool::select) return gizmo_axis::none;
    const auto view_projection = gizmo_view_projection(*camera, *camera_transform, context);
    const auto origin = scene::world_position(*transform);
    const auto axes = gizmo_axes(*transform, context.coordinate_space);
    const float scale = editor_gizmo_world_scale(*camera, *camera_transform, origin, context.viewport_height);
    math::vector2f projected_origin;
    if (!project_to_screen(view_projection, origin, context.viewport_width, context.viewport_height, projected_origin))
        return gizmo_axis::none;
    const math::vector2f pointer{screen_x, screen_y};
    if (context.tool == editor_tool::scale && math::length(math::sub(pointer, projected_origin)) <= gizmo_hit_radius)
        return gizmo_axis::all;
    float nearest = gizmo_hit_radius;
    gizmo_axis result = gizmo_axis::none;
    if (context.tool == editor_tool::rotate)
    {
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
        {
            const auto tangent = axes[(axis + 1) % 3];
            const auto bitangent = axes[(axis + 2) % 3];
            for (std::uint32_t segment = 0; segment < rotation_segments; ++segment)
            {
                const float first_angle =
                    math::tau<float> * static_cast<float>(segment) / static_cast<float>(rotation_segments);
                const float second_angle =
                    math::tau<float> * static_cast<float>(segment + 1) / static_cast<float>(rotation_segments);
                const auto ring_point = [&](float angle)
                {
                    return math::add(origin, math::mul(math::add(math::mul(tangent, std::cos(angle)),
                                                                 math::mul(bitangent, std::sin(angle))),
                                                       scale));
                };
                math::vector2f first, second;
                if (!project_to_screen(view_projection, ring_point(first_angle), context.viewport_width,
                                       context.viewport_height, first) ||
                    !project_to_screen(view_projection, ring_point(second_angle), context.viewport_width,
                                       context.viewport_height, second))
                    continue;
                const float distance = distance_to_segment(pointer, first, second);
                if (distance < nearest)
                {
                    nearest = distance;
                    result = static_cast<gizmo_axis>(axis + 1);
                }
            }
        }
        return result;
    }
    for (std::size_t axis = 0; axis < axes.size(); ++axis)
    {
        math::vector2f projected_end;
        if (!project_to_screen(view_projection, math::add(origin, math::mul(axes[axis], scale)), context.viewport_width,
                               context.viewport_height, projected_end))
            continue;
        const float distance = distance_to_segment(pointer, projected_origin, projected_end);
        if (distance < nearest)
        {
            nearest = distance;
            result = static_cast<gizmo_axis>(axis + 1);
        }
    }
    return result;
}

bool editor_gizmo_drag_direction(const ecs::world& registry, ecs::entity selected, ecs::entity camera_entity,
                                 const editor_gizmo_context& context, gizmo_axis axis, float screen_x, float screen_y,
                                 math::vector2f& direction) noexcept
{
    const auto* transform = registry.try_get<scene::transform_component>(selected);
    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);
    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);
    if (!transform || !camera || !camera_transform || axis == gizmo_axis::none || context.tool == editor_tool::select)
        return false;

    if (axis == gizmo_axis::all)
    {
        if (context.tool != editor_tool::scale) return false;
        direction = math::normalize(math::vector2f{1.0f, -1.0f});
        return true;
    }

    const std::size_t axis_index = static_cast<std::size_t>(axis) - 1u;
    const auto origin = scene::world_position(*transform);
    const auto axes = gizmo_axes(*transform, context.coordinate_space);
    const float scale = editor_gizmo_world_scale(*camera, *camera_transform, origin, context.viewport_height);
    const auto view_projection = gizmo_view_projection(*camera, *camera_transform, context);
    math::vector2f projected_origin;
    if (!project_to_screen(view_projection, origin, context.viewport_width, context.viewport_height, projected_origin))
        return false;

    if (context.tool != editor_tool::rotate)
    {
        math::vector2f projected_end;
        if (!project_to_screen(view_projection, math::add(origin, math::mul(axes[axis_index], scale)),
                               context.viewport_width, context.viewport_height, projected_end))
            return false;
        const auto projected_axis = math::sub(projected_end, projected_origin);
        const float length_squared = math::length_squared(projected_axis);
        if (length_squared <= 1.0e-4f) return false;
        direction = math::mul(projected_axis, 1.0f / std::sqrt(length_squared));
        return true;
    }

    const auto tangent = axes[(axis_index + 1) % 3];
    const auto bitangent = axes[(axis_index + 2) % 3];
    const math::vector2f pointer{screen_x, screen_y};
    float nearest = std::numeric_limits<float>::max();
    bool found{};
    for (std::uint32_t segment = 0; segment < rotation_segments; ++segment)
    {
        const float first_angle =
            math::tau<float> * static_cast<float>(segment) / static_cast<float>(rotation_segments);
        const float second_angle =
            math::tau<float> * static_cast<float>(segment + 1) / static_cast<float>(rotation_segments);
        const auto ring_point = [&](float angle)
        {
            return math::add(
                origin, math::mul(math::add(math::mul(tangent, std::cos(angle)), math::mul(bitangent, std::sin(angle))),
                                  scale));
        };
        math::vector2f first;
        math::vector2f second;
        if (!project_to_screen(view_projection, ring_point(first_angle), context.viewport_width,
                               context.viewport_height, first) ||
            !project_to_screen(view_projection, ring_point(second_angle), context.viewport_width,
                               context.viewport_height, second))
            continue;
        const auto segment_direction = math::sub(second, first);
        const float segment_length_squared = math::length_squared(segment_direction);
        if (segment_length_squared <= 1.0e-4f) continue;
        const float distance = distance_to_segment(pointer, first, second);
        if (distance >= nearest) continue;
        nearest = distance;
        direction = math::mul(segment_direction, 1.0f / std::sqrt(segment_length_squared));
        found = true;
    }
    return found;
}

} // namespace arc::editor
