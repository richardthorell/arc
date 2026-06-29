#pragma once

#include <arc/editor/editor_viewport.h>
#include <arc/input/input.h>
#include <arc/scene/scene.h>
#include <arc/math/math.h>

#include <cstdint>

namespace arc::editor
{

/**
 * @brief Active editor manipulation tool.
 */
enum class editor_tool : std::uint8_t
{
    select,
    translate,
    rotate,
    scale
};

/**
 * @brief Return a short label for a tool.
 */
const char* editor_tool_label(editor_tool tool) noexcept;

/**
 * @brief Ray used for viewport picking.
 */
struct editor_ray
{
    math::vector3f origin{};
    math::vector3f direction{ 0.0f, 0.0f, -1.0f };
};

/**
 * @brief Orbit camera state used by the editor viewport.
 */
class editor_camera_controller
{
public:
    /**
     * @brief Set the orbit focus point and distance from a bounding radius.
     */
    void focus(const math::vector3f& point, float radius) noexcept;

    /**
     * @brief Orbit around the focus point by mouse delta in pixels.
     */
    void orbit(float delta_x, float delta_y) noexcept;

    /**
     * @brief Pan focus point by mouse delta in pixels.
     */
    void pan(float delta_x, float delta_y) noexcept;

    /**
     * @brief Dolly camera by wheel delta.
     */
    void zoom(float wheel_delta) noexcept;

    /**
     * @brief Apply the controller state to a scene transform.
     */
    void apply_to(scene::transform_component& transform) const noexcept;

    /**
     * @brief Return current focus point.
     */
    const math::vector3f& focus_point() const noexcept;

    /**
     * @brief Return current orbit distance.
     */
    float distance() const noexcept;

private:
    math::vector3f focus_{};
    float yaw_{};
    float pitch_{ -0.18f };
    float distance_{ 4.0f };
};

/**
 * @brief Switch editor tool from configured keyboard shortcuts.
 */
void apply_tool_shortcuts(const input::input_manager& input, editor_tool& tool) noexcept;

/**
 * @brief Clear the current selection and all scene selection components.
 */
void clear_selection(scene::registry& registry, scene::entity& selected) noexcept;

/**
 * @brief Select a live entity and synchronize `selection_component`.
 */
bool select_entity(scene::registry& registry, scene::entity entity, scene::entity& selected);

/**
 * @brief Pick the nearest bounded entity hit by a ray.
 */
scene::entity pick_bounded_entity(const scene::registry& registry, const editor_ray& ray) noexcept;

/**
 * @brief Build a world-space picking ray from camera and viewport coordinates.
 */
editor_ray screen_ray_from_camera(
    const scene::camera_component& camera,
    const scene::transform_component& camera_transform,
    const editor_viewport& viewport,
    float local_x,
    float local_y) noexcept;

/**
 * @brief Return whether a ray intersects a box, writing nearest distance when hit.
 */
bool intersect_ray_box(const editor_ray& ray, const geometric::box3f& bounds, float& distance) noexcept;

/**
 * @brief Transform local bounds by a transform into world-space AABB bounds.
 */
geometric::box3f transformed_bounds(const geometric::box3f& local_bounds, const scene::transform_component& transform) noexcept;

/**
 * @brief Focus a camera controller on the selected entity.
 */
bool focus_selected_entity(
    const scene::registry& registry,
    scene::entity selected,
    editor_camera_controller& camera) noexcept;

/**
 * @brief Convert Euler degrees to a quaternion.
 */
math::quatf quaternion_from_euler_degrees(const math::vector3f& degrees) noexcept;

/**
 * @brief Convert a quaternion to Euler degrees.
 */
math::vector3f euler_degrees_from_quaternion(const math::quatf& rotation) noexcept;

} // namespace arc::editor
