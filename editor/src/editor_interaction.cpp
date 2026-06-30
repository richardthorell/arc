#include <arc/editor/editor_interaction.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace arc::editor
{
namespace
{

constexpr float pi = 3.14159265358979323846f;
constexpr float degrees_to_radians = pi / 180.0f;
constexpr float radians_to_degrees = 180.0f / pi;

math::quatf quaternion_from_yaw_pitch(float yaw, float pitch) noexcept
{
    const float half_yaw = yaw * 0.5f;
    const float half_pitch = pitch * 0.5f;
    const float sy = std::sin(half_yaw);
    const float cy = std::cos(half_yaw);
    const float sp = std::sin(half_pitch);
    const float cp = std::cos(half_pitch);
    return math::normalize(math::quatf{ cy * sp, sy * cp, -sy * sp, cy * cp });
}

math::quatf multiply_quaternion(const math::quatf& lhs, const math::quatf& rhs) noexcept
{
    return math::quatf{
        lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[3] * rhs[1] - lhs[0] * rhs[2] + lhs[1] * rhs[3] + lhs[2] * rhs[0],
        lhs[3] * rhs[2] + lhs[0] * rhs[1] - lhs[1] * rhs[0] + lhs[2] * rhs[3],
        lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2]
    };
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

void editor_camera_controller::orbit(float delta_x, float delta_y) noexcept
{
    yaw_ -= delta_x * 0.008f;
    pitch_ = std::clamp(pitch_ - delta_y * 0.008f, -1.45f, 1.45f);
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
    const float ndc_x = (local_x / width) * 2.0f - 1.0f;
    const float ndc_y = 1.0f - (local_y / height) * 2.0f;
    const auto right = math::rotate(camera_transform.rotation, math::vector3f{ 1.0f, 0.0f, 0.0f });
    const auto up = math::rotate(camera_transform.rotation, math::vector3f{ 0.0f, 1.0f, 0.0f });
    const auto forward = math::rotate(camera_transform.rotation, math::vector3f{ 0.0f, 0.0f, -1.0f });

    if (camera.projection == scene::camera_projection::orthographic)
    {
        const float half_height = std::max(0.01f, camera.orthographic_height) * 0.5f;
        const float half_width = half_height * aspect;
        const auto offset = math::add(math::mul(right, ndc_x * half_width), math::mul(up, ndc_y * half_height));
        return {
            .origin = math::add(camera_transform.position, offset),
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
        .origin = camera_transform.position,
        .direction = math::normalize(math::rotate(camera_transform.rotation, math::normalize(camera_direction)))
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
    const auto matrix = scene::local_matrix(transform);
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

    camera.focus(transform->position, 1.0f);
    return true;
}

math::quatf quaternion_from_euler_degrees(const math::vector3f& degrees) noexcept
{
    const float x = degrees[0] * degrees_to_radians;
    const float y = degrees[1] * degrees_to_radians;
    const float z = degrees[2] * degrees_to_radians;
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
        x_angle * radians_to_degrees,
        y_angle * radians_to_degrees,
        z_angle * radians_to_degrees
    };
}

} // namespace arc::editor
