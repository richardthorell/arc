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
constexpr float gizmo_hit_radius = 9.0f;
constexpr std::uint32_t rotation_segments = 48;

constexpr std::array<math::vector3f, 3> canonical_axes{
    math::vector3f{ 1.0f, 0.0f, 0.0f },
    math::vector3f{ 0.0f, 1.0f, 0.0f },
    math::vector3f{ 0.0f, 0.0f, 1.0f }
};
constexpr std::array<math::vector4f, 3> axis_colors{
    math::vector4f{ 0.95f, 0.18f, 0.16f, 1.0f },
    math::vector4f{ 0.30f, 0.82f, 0.24f, 1.0f },
    math::vector4f{ 0.20f, 0.52f, 1.0f, 1.0f }
};
constexpr math::vector4f highlighted_color{ 1.0f, 0.82f, 0.18f, 1.0f };
constexpr math::vector4f bounds_color{ 0.25f, 0.62f, 1.0f, 0.72f };

math::vector3f matrix_axis(const math::matrix4f& matrix, std::size_t column) noexcept
{
    return math::normalize(math::vector3f{ matrix(0, column), matrix(1, column), matrix(2, column) });
}

std::array<math::vector3f, 3> gizmo_axes(const scene::transform_component& transform, gizmo_coordinate_space space) noexcept
{
    if (space == gizmo_coordinate_space::world)
        return canonical_axes;
    return { matrix_axis(transform.world, 0), matrix_axis(transform.world, 1), matrix_axis(transform.world, 2) };
}

math::vector4f color_for_axis(std::size_t index, gizmo_axis highlighted) noexcept
{
    return highlighted == static_cast<gizmo_axis>(index + 1) ? highlighted_color : axis_colors[index];
}

void append_bounds(render::debug_overlay_stream& stream, const geometric::box3f& bounds)
{
    const std::array<math::vector3f, 8> corners{
        math::vector3f{ bounds.min[0], bounds.min[1], bounds.min[2] }, math::vector3f{ bounds.max[0], bounds.min[1], bounds.min[2] },
        math::vector3f{ bounds.max[0], bounds.max[1], bounds.min[2] }, math::vector3f{ bounds.min[0], bounds.max[1], bounds.min[2] },
        math::vector3f{ bounds.min[0], bounds.min[1], bounds.max[2] }, math::vector3f{ bounds.max[0], bounds.min[1], bounds.max[2] },
        math::vector3f{ bounds.max[0], bounds.max[1], bounds.max[2] }, math::vector3f{ bounds.min[0], bounds.max[1], bounds.max[2] }
    };
    constexpr std::array<std::array<std::size_t, 2>, 12> edges{
        std::array<std::size_t, 2>{0,1}, {1,2}, {2,3}, {3,0}, {4,5}, {5,6}, {6,7}, {7,4}, {0,4}, {1,5}, {2,6}, {3,7}
    };
    for (const auto edge : edges)
        stream.lines.push_back({ corners[edge[0]], corners[edge[1]], bounds_color, render::debug_overlay_depth_mode::tested });
}

bool project_to_screen(const math::matrix4f& view_projection, const math::vector3f& point,
    std::uint32_t width, std::uint32_t height, math::vector2f& screen) noexcept
{
    const float x = view_projection(0, 0) * point[0] + view_projection(0, 1) * point[1] + view_projection(0, 2) * point[2] + view_projection(0, 3);
    const float y = view_projection(1, 0) * point[0] + view_projection(1, 1) * point[1] + view_projection(1, 2) * point[2] + view_projection(1, 3);
    const float w = view_projection(3, 0) * point[0] + view_projection(3, 1) * point[1] + view_projection(3, 2) * point[2] + view_projection(3, 3);
    if (!(w > 1.0e-5f)) return false;
    screen = { (x / w * 0.5f + 0.5f) * static_cast<float>(width),
        (0.5f - y / w * 0.5f) * static_cast<float>(height) };
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
}

float editor_gizmo_world_scale(const scene::camera_component& camera, const scene::transform_component& camera_transform,
    const math::vector3f& world_position, std::uint32_t viewport_height) noexcept
{
    const float height = static_cast<float>(std::max(1u, viewport_height));
    if (camera.projection == scene::camera_projection::orthographic)
        return std::max(0.001f, camera.orthographic_height * editor_gizmo_pixel_length / height);
    const float distance = std::max(0.01f, math::length(math::sub(world_position, scene::world_position(camera_transform))));
    return std::max(0.001f, 2.0f * distance * std::tan(camera.fov_y_radians * 0.5f) * editor_gizmo_pixel_length / height);
}

render::debug_overlay_stream build_editor_gizmo_overlay(const scene::registry& registry, scene::entity selected,
    scene::entity camera_entity, const editor_gizmo_context& context)
{
    render::debug_overlay_stream stream;
    const auto* transform = registry.try_get<scene::transform_component>(selected);
    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);
    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);
    if (!transform || !camera || !camera_transform) return stream;
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
            for (std::uint32_t segment = 0; segment < rotation_segments; ++segment)
            {
                const float first = math::tau<float> * static_cast<float>(segment) / static_cast<float>(rotation_segments);
                const float second = math::tau<float> * static_cast<float>(segment + 1) / static_cast<float>(rotation_segments);
                const auto point = [&](float angle) { return math::add(origin, math::mul(math::add(
                    math::mul(tangent, std::cos(angle)), math::mul(bitangent, std::sin(angle))), scale)); };
                stream.lines.push_back({ point(first), point(second), color_for_axis(axis, context.highlighted_axis),
                    render::debug_overlay_depth_mode::always });
            }
        }
        return stream;
    }
    for (std::size_t axis = 0; axis < axes.size(); ++axis)
        stream.lines.push_back({ origin, math::add(origin, math::mul(axes[axis], scale)),
            color_for_axis(axis, context.highlighted_axis), render::debug_overlay_depth_mode::always });
    return stream;
}

gizmo_axis hit_test_editor_gizmo(const scene::registry& registry, scene::entity selected, scene::entity camera_entity,
    const editor_gizmo_context& context, float screen_x, float screen_y) noexcept
{
    const auto* transform = registry.try_get<scene::transform_component>(selected);
    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);
    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);
    if (!transform || !camera || !camera_transform || context.tool == editor_tool::select) return gizmo_axis::none;
    const float aspect = static_cast<float>(std::max(1u, context.viewport_width)) / static_cast<float>(std::max(1u, context.viewport_height));
    const auto view_projection = scene::view_projection(*camera, *camera_transform, aspect);
    const auto origin = scene::world_position(*transform);
    const auto axes = gizmo_axes(*transform, context.coordinate_space);
    const float scale = editor_gizmo_world_scale(*camera, *camera_transform, origin, context.viewport_height);
    math::vector2f projected_origin;
    if (!project_to_screen(view_projection, origin, context.viewport_width, context.viewport_height, projected_origin)) return gizmo_axis::none;
    const math::vector2f pointer{ screen_x, screen_y };
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
                const float first_angle = math::tau<float> * static_cast<float>(segment) / static_cast<float>(rotation_segments);
                const float second_angle = math::tau<float> * static_cast<float>(segment + 1) / static_cast<float>(rotation_segments);
                const auto ring_point = [&](float angle) { return math::add(origin, math::mul(math::add(
                    math::mul(tangent, std::cos(angle)), math::mul(bitangent, std::sin(angle))), scale)); };
                math::vector2f first, second;
                if (!project_to_screen(view_projection, ring_point(first_angle), context.viewport_width, context.viewport_height, first) ||
                    !project_to_screen(view_projection, ring_point(second_angle), context.viewport_width, context.viewport_height, second))
                    continue;
                const float distance = distance_to_segment(pointer, first, second);
                if (distance < nearest) { nearest = distance; result = static_cast<gizmo_axis>(axis + 1); }
            }
        }
        return result;
    }
    for (std::size_t axis = 0; axis < axes.size(); ++axis)
    {
        math::vector2f projected_end;
        if (!project_to_screen(view_projection, math::add(origin, math::mul(axes[axis], scale)),
                context.viewport_width, context.viewport_height, projected_end)) continue;
        const float distance = distance_to_segment(pointer, projected_origin, projected_end);
        if (distance < nearest) { nearest = distance; result = static_cast<gizmo_axis>(axis + 1); }
    }
    return result;
}

} // namespace arc::editor
