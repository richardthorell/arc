#include <arc/editor/editor_interaction.h>
#include <arc/render/renderer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace arc::editor
{
namespace
{

math::quatf multiply_quaternion(const math::quatf& lhs, const math::quatf& rhs) noexcept
{
    return math::quatf{
        lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[3] * rhs[1] - lhs[0] * rhs[2] + lhs[1] * rhs[3] + lhs[2] * rhs[0],
        lhs[3] * rhs[2] + lhs[0] * rhs[1] - lhs[1] * rhs[0] + lhs[2] * rhs[3],
        lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2]
    };
}

math::quatf quaternion_from_yaw_pitch(float yaw, float pitch) noexcept
{
    // Turntable orbit: yaw is always around the rig's stable +Y up axis,
    // followed by pitch around the yawed camera-local +X axis. Rebuilding the
    // orientation from these two scalars prevents any accumulated roll.
    const auto yaw_rotation = math::from_axis_angle(math::vector3f{ 0.0f, 1.0f, 0.0f }, yaw);
    const auto pitch_rotation = math::from_axis_angle(math::vector3f{ 1.0f, 0.0f, 0.0f }, pitch);
    return math::normalize(multiply_quaternion(yaw_rotation, pitch_rotation));
}

math::vector3f point_to_vector(const geometric::point3f& point) noexcept
{
    return math::vector3f{ point[0], point[1], point[2] };
}

geometric::point3f vector_to_point(const math::vector3f& value) noexcept
{
    return geometric::point3f{ value };
}

float bounds_radius(const geometric::box3f& bounds) noexcept
{
    return std::max(0.1f, math::length(geometric::size(bounds)) * 0.5f);
}

bool intersect_ray_triangle(
    const editor_ray& ray,
    const math::vector3f& a,
    const math::vector3f& b,
    const math::vector3f& c,
    float& distance) noexcept
{
    constexpr float epsilon = 1.0e-7f;
    const auto edge1 = math::sub(b, a);
    const auto edge2 = math::sub(c, a);
    const auto p = math::cross(ray.direction, edge2);
    const float determinant = math::dot(edge1, p);
    if (std::abs(determinant) <= epsilon)
        return false;
    const float inverse_determinant = 1.0f / determinant;
    const auto offset = math::sub(ray.origin, a);
    const float u = math::dot(offset, p) * inverse_determinant;
    if (u < 0.0f || u > 1.0f)
        return false;
    const auto q = math::cross(offset, edge1);
    const float v = math::dot(ray.direction, q) * inverse_determinant;
    if (v < 0.0f || u + v > 1.0f)
        return false;
    const float hit = math::dot(edge2, q) * inverse_determinant;
    if (hit < 0.0f)
        return false;
    distance = hit;
    return true;
}

float world_hit_distance(
    const editor_ray& world_ray,
    const math::matrix4f& world,
    const math::vector3f& local_position) noexcept
{
    const auto world_position = math::transform_point(world, local_position);
    return math::dot(math::sub(world_position, world_ray.origin), world_ray.direction);
}

} // namespace

const char* editor_tool_label(editor_tool tool) noexcept
{
    switch (tool)
    {
    case editor_tool::select:
        return "Select";
    case editor_tool::translate:
        return "Translate";
    case editor_tool::rotate:
        return "Rotate";
    case editor_tool::scale:
        return "Scale";
    }
    return "Select";
}

void editor_camera_controller::focus(const math::vector3f& point, float radius) noexcept
{
    focus_ = point;
    distance_ = std::clamp(radius * 3.2f, 0.35f, 500.0f);
}

void editor_camera_controller::synchronize_from(const scene::transform_component& transform) noexcept
{
    const auto forward = scene::world_forward_direction(transform);
    yaw_ = std::atan2(-forward[0], -forward[2]);
    pitch_ = std::asin(std::clamp(forward[1], -1.0f, 1.0f));
    const auto position = scene::world_position(transform);
    focus_ = math::add(position, math::mul(forward, distance_));
}

void editor_camera_controller::orbit(float delta_x, float delta_y) noexcept
{
    yaw_ = std::remainder(yaw_ - delta_x * 0.008f, math::tau<float>);
    pitch_ = std::clamp(pitch_ + delta_y * 0.008f, -1.45f, 1.45f);
}

void editor_camera_controller::pan(float delta_x, float delta_y) noexcept
{
    const auto rotation = quaternion_from_yaw_pitch(yaw_, pitch_);
    const auto right = math::rotate(rotation, math::vector3f{ 1.0f, 0.0f, 0.0f });
    const auto up = math::rotate(rotation, math::vector3f{ 0.0f, 1.0f, 0.0f });
    constexpr float speed = 0.012f;
    focus_ = math::add(focus_, math::mul(right, -delta_x * speed));
    focus_ = math::add(focus_, math::mul(up, delta_y * speed));
}

void editor_camera_controller::move_forward(float delta_y) noexcept
{
    const auto rotation = quaternion_from_yaw_pitch(yaw_, pitch_);
    const auto forward = math::rotate(rotation, math::vector3f{ 0.0f, 0.0f, -1.0f });
    const float speed = std::clamp(distance_ * 0.006f, 0.015f, 1.5f);
    focus_ = math::add(focus_, math::mul(forward, -delta_y * speed));
}

void editor_camera_controller::zoom(float wheel_delta) noexcept
{
    distance_ = std::clamp(distance_ * std::pow(0.86f, wheel_delta), 0.15f, 500.0f);
}

void editor_camera_controller::apply_to(scene::transform_component& transform) const noexcept
{
    const auto rotation = quaternion_from_yaw_pitch(yaw_, pitch_);
    const auto forward = math::rotate(rotation, math::vector3f{ 0.0f, 0.0f, -1.0f });
    transform.set_rotation(rotation);
    transform.set_position(math::sub(focus_, math::mul(forward, distance_)));
}

const math::vector3f& editor_camera_controller::focus_point() const noexcept
{
    return focus_;
}

float editor_camera_controller::distance() const noexcept
{
    return distance_;
}

void apply_tool_shortcuts(const input::input_manager& input, editor_tool& tool) noexcept
{
    if (input.pressed("tool.select"))
        tool = editor_tool::select;
    if (input.pressed("tool.translate"))
        tool = editor_tool::translate;
    if (input.pressed("tool.rotate"))
        tool = editor_tool::rotate;
    if (input.pressed("tool.scale"))
        tool = editor_tool::scale;
}

void clear_selection(scene::registry& registry, scene::entity& selected) noexcept
{
    std::vector<scene::entity> selected_entities;
    registry.view<scene::selection_component>().each(
        [&](scene::entity value, const scene::selection_component&) {
            selected_entities.push_back(value);
        });
    for (const scene::entity value : selected_entities)
    {
        if (auto* selection = registry.try_get<scene::selection_component>(value))
            selection->selected = false;
    }
    selected = {};
}

bool select_entity(scene::registry& registry, scene::entity entity, scene::entity& selected)
{
    clear_selection(registry, selected);
    if (!registry.alive(entity))
        return false;

    if (auto* selection = registry.try_get<scene::selection_component>(entity))
        selection->selected = true;
    else
        registry.emplace<scene::selection_component>(entity, true);
    selected = entity;
    return true;
}

scene::entity pick_bounded_entity(const scene::registry& registry, const editor_ray& ray) noexcept
{
    scene::entity picked{};
    float picked_distance = std::numeric_limits<float>::max();

    registry.view<scene::transform_component, scene::bounds_component>().each(
        [&](scene::entity value, const scene::transform_component& transform, const scene::bounds_component& bounds) {
            const auto* active = registry.try_get<scene::active_component>(value);
            if (active && !active->active)
                return;

            float hit_distance{};
            if (!intersect_ray_box(ray, transformed_bounds(bounds.local_bounds, transform), hit_distance))
                return;

            if (hit_distance < picked_distance)
            {
                picked = value;
                picked_distance = hit_distance;
            }
        });

    return picked;
}

editor_pick_result pick_scene_entity(
    const scene::registry& registry,
    const render::renderer& renderer,
    const editor_ray& ray) noexcept
{
    editor_pick_result picked{ .distance = std::numeric_limits<float>::max() };

    registry.view<scene::transform_component, scene::bounds_component>().each(
        [&](scene::entity value, const scene::transform_component& transform, const scene::bounds_component& bounds) {
            const auto* active = registry.try_get<scene::active_component>(value);
            if (active && !active->active)
                return;
            if (const auto* mesh_renderer = registry.try_get<scene::mesh_renderer_component>(value);
                mesh_renderer && !mesh_renderer->visible)
                return;

            float broad_distance{};
            if (!intersect_ray_box(ray, transformed_bounds(bounds.local_bounds, transform), broad_distance))
                return;

            const auto world = transform.dirty ? scene::local_matrix(transform) : transform.world;
            math::matrix4f inverse_world;
            if (!scene::inverse_affine(world, inverse_world))
                return;
            const editor_ray local_ray{
                .origin = math::transform_point(inverse_world, ray.origin),
                .direction = math::normalize(math::transform_vector(inverse_world, ray.direction))
            };

            float hit_distance = broad_distance;
            bool exact = false;
            if (const auto* terrain = registry.try_get<scene::terrain_component>(value))
            {
                const auto terrain_hit = scene::raycast_terrain(*terrain, local_ray.origin, local_ray.direction);
                if (!terrain_hit.hit)
                    return;
                hit_distance = world_hit_distance(ray, world, terrain_hit.position);
                exact = true;
            }
            else if (const auto* mesh_renderer = registry.try_get<scene::mesh_renderer_component>(value))
            {
                const auto* mesh = renderer.mesh_data_for(mesh_renderer->mesh);
                if (mesh && mesh->indices.size() >= 3)
                {
                    float local_distance = std::numeric_limits<float>::max();
                    for (std::size_t index = 0; index + 2 < mesh->indices.size(); index += 3)
                    {
                        const auto ia = mesh->indices[index];
                        const auto ib = mesh->indices[index + 1];
                        const auto ic = mesh->indices[index + 2];
                        if (ia >= mesh->vertices.size() || ib >= mesh->vertices.size() || ic >= mesh->vertices.size())
                            continue;
                        const auto position = [](const render::mesh_vertex& vertex) {
                            return math::vector3f{ vertex.position[0], vertex.position[1], vertex.position[2] };
                        };
                        float triangle_distance{};
                        if (intersect_ray_triangle(
                                local_ray,
                                position(mesh->vertices[ia]),
                                position(mesh->vertices[ib]),
                                position(mesh->vertices[ic]),
                                triangle_distance))
                            local_distance = std::min(local_distance, triangle_distance);
                    }
                    if (local_distance == std::numeric_limits<float>::max())
                        return;
                    hit_distance = world_hit_distance(
                        ray,
                        world,
                        math::add(local_ray.origin, math::mul(local_ray.direction, local_distance)));
                    exact = true;
                }
            }

            if (hit_distance < 0.0f || hit_distance >= picked.distance)
                return;
            picked = {
                .entity = value,
                .distance = hit_distance,
                .exact = exact,
                .background = registry.has<scene::terrain_component>(value) ||
                    registry.has<scene::water_component>(value) ||
                    registry.has<scene::world_environment_component>(value)
            };
        });
    return picked;
}

editor_ray screen_ray_from_camera(
    const scene::camera_component& camera,
    const scene::transform_component& camera_transform,
    const editor_viewport& viewport,
    float local_x,
    float local_y) noexcept
{
    const float width = static_cast<float>(std::max(1u, viewport.width()));
    const float height = static_cast<float>(std::max(1u, viewport.height()));
    const float aspect = width / height;
    const float ndc_x = ((local_x + 0.5f) / width) * 2.0f - 1.0f;
    const float ndc_y = 1.0f - ((local_y + 0.5f) / height) * 2.0f;
    const auto world = camera_transform.dirty ? scene::local_matrix(camera_transform) : camera_transform.world;
    const auto right = math::normalize(math::transform_vector(world, math::vector3f{ 1.0f, 0.0f, 0.0f }));
    const auto up = math::normalize(math::transform_vector(world, math::vector3f{ 0.0f, 1.0f, 0.0f }));
    const auto forward = math::normalize(math::transform_vector(world, math::vector3f{ 0.0f, 0.0f, -1.0f }));
    const auto origin = scene::world_position(camera_transform);

    if (camera.projection == scene::camera_projection::orthographic)
    {
        const float half_height = std::max(0.01f, camera.orthographic_height) * 0.5f;
        const float half_width = half_height * aspect;
        const auto offset = math::add(math::mul(right, ndc_x * half_width), math::mul(up, ndc_y * half_height));
        return {
            .origin = math::add(origin, offset),
            .direction = math::normalize(forward)
        };
    }

    const float tan_half_fov = std::tan(camera.fov_y_radians * 0.5f);
    const math::vector3f camera_direction{
        ndc_x * aspect * tan_half_fov,
        ndc_y * tan_half_fov,
        -1.0f
    };
    return {
        .origin = origin,
        .direction = math::normalize(math::add(
            math::add(math::mul(right, camera_direction[0]), math::mul(up, camera_direction[1])),
            math::mul(forward, -camera_direction[2])))
    };
}

bool intersect_ray_box(const editor_ray& ray, const geometric::box3f& bounds, float& distance) noexcept
{
    float t_min = 0.0f;
    float t_max = std::numeric_limits<float>::max();

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const float origin = ray.origin[axis];
        const float direction = ray.direction[axis];
        const float min_value = bounds.min[axis];
        const float max_value = bounds.max[axis];

        if (std::abs(direction) < 0.000001f)
        {
            if (origin < min_value || origin > max_value)
                return false;
            continue;
        }

        float t1 = (min_value - origin) / direction;
        float t2 = (max_value - origin) / direction;
        if (t1 > t2)
            std::swap(t1, t2);
        t_min = std::max(t_min, t1);
        t_max = std::min(t_max, t2);
        if (t_min > t_max)
            return false;
    }

    distance = t_min;
    return true;
}

geometric::box3f transformed_bounds(const geometric::box3f& local_bounds, const scene::transform_component& transform) noexcept
{
    const auto matrix = transform.dirty ? scene::local_matrix(transform) : transform.world;
    const std::array<math::vector3f, 8> corners{
        math::vector3f{ local_bounds.min[0], local_bounds.min[1], local_bounds.min[2] },
        math::vector3f{ local_bounds.max[0], local_bounds.min[1], local_bounds.min[2] },
        math::vector3f{ local_bounds.min[0], local_bounds.max[1], local_bounds.min[2] },
        math::vector3f{ local_bounds.max[0], local_bounds.max[1], local_bounds.min[2] },
        math::vector3f{ local_bounds.min[0], local_bounds.min[1], local_bounds.max[2] },
        math::vector3f{ local_bounds.max[0], local_bounds.min[1], local_bounds.max[2] },
        math::vector3f{ local_bounds.min[0], local_bounds.max[1], local_bounds.max[2] },
        math::vector3f{ local_bounds.max[0], local_bounds.max[1], local_bounds.max[2] }
    };

    auto min_value = math::transform_point(matrix, corners[0]);
    auto max_value = min_value;
    for (std::size_t index = 1; index < corners.size(); ++index)
    {
        const auto point = math::transform_point(matrix, corners[index]);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            min_value[axis] = std::min(min_value[axis], point[axis]);
            max_value[axis] = std::max(max_value[axis], point[axis]);
        }
    }

    return geometric::box3f{ vector_to_point(min_value), vector_to_point(max_value) };
}

bool focus_selected_entity(
    const scene::registry& registry,
    scene::entity selected,
    editor_camera_controller& camera) noexcept
{
    if (!registry.alive(selected))
        return false;

    const auto* transform = registry.try_get<scene::transform_component>(selected);
    if (!transform)
        return false;

    if (const auto* bounds = registry.try_get<scene::bounds_component>(selected))
    {
        const auto world = transformed_bounds(bounds->local_bounds, *transform);
        camera.focus(point_to_vector(geometric::center(world)), bounds_radius(world));
        return true;
    }

    camera.focus(scene::world_position(*transform), 1.0f);
    return true;
}

math::quatf quaternion_from_euler_degrees(const math::vector3f& degrees) noexcept
{
    const float x = math::to_radians(degrees[0]);
    const float y = math::to_radians(degrees[1]);
    const float z = math::to_radians(degrees[2]);
    const auto qx = math::from_axis_angle(math::vector3f{ 1.0f, 0.0f, 0.0f }, x);
    const auto qy = math::from_axis_angle(math::vector3f{ 0.0f, 1.0f, 0.0f }, y);
    const auto qz = math::from_axis_angle(math::vector3f{ 0.0f, 0.0f, 1.0f }, z);
    return math::normalize(multiply_quaternion(multiply_quaternion(qy, qx), qz));
}

math::vector3f euler_degrees_from_quaternion(const math::quatf& rotation) noexcept
{
    const auto q = math::normalize(rotation);
    const float x = q[0];
    const float y = q[1];
    const float z = q[2];
    const float w = q[3];

    const float m00 = 1.0f - 2.0f * y * y - 2.0f * z * z;
    const float m02 = 2.0f * x * z + 2.0f * y * w;
    const float m10 = 2.0f * x * y + 2.0f * z * w;
    const float m11 = 1.0f - 2.0f * x * x - 2.0f * z * z;
    const float m12 = 2.0f * y * z - 2.0f * x * w;
    const float m20 = 2.0f * x * z - 2.0f * y * w;
    const float m22 = 1.0f - 2.0f * x * x - 2.0f * y * y;

    const float x_angle = std::asin(std::clamp(-m12, -1.0f, 1.0f));
    const float cx = std::cos(x_angle);
    float y_angle{};
    float z_angle{};
    if (std::abs(cx) > 0.0001f)
    {
        y_angle = std::atan2(m02, m22);
        z_angle = std::atan2(m10, m11);
    }
    else
    {
        y_angle = std::atan2(-m20, m00);
    }

    return math::vector3f{
        math::to_degrees(x_angle),
        math::to_degrees(y_angle),
        math::to_degrees(z_angle)
    };
}

} // namespace arc::editor
