#pragma once

#include <arc/editor/editor_viewport.h>
#include <arc/input/input.h>
#include <arc/scene/scene.h>
#include <arc/math/math.h>

#include <cstdint>
#include <optional>

namespace arc::render
{
class renderer;
}

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
 * @brief Resolve a viewport transform-tool shortcut.
 *
 * @param key_code Uppercase platform-independent key
 * code.
 * @return The requested tool, or no value when the key is not a tool shortcut.
 */
std::optional<editor_tool> editor_tool_from_shortcut(std::uint32_t key_code) noexcept;

/**
 * @brief Ray used for viewport picking.
 */
struct editor_ray
{
    math::vector3f origin{};
    math::vector3f direction{0.0f, 0.0f, -1.0f};
};

struct editor_pick_result
{
    ecs::entity entity{};
    float distance{};
    bool exact{};
    bool background{};
};

/**
 * @brief Orbit camera state used by the editor viewport.
 */
class editor_camera_controller
{
public:
    editor_camera_controller() noexcept;

    /**
     * @brief Set the orbit focus point and distance from a bounding radius.
     */
    void focus(const math::vector3f& point, float radius) noexcept;

    /**
     * @brief Synchronize the orbit rig from an externally changed camera transform.
     */
    void synchronize_from(const scene::transform_component& transform) noexcept;

    /**
     * @brief Place the orbit rig from an absolute camera position and focus point.
     */
    bool place(const math::vector3f& position, const math::vector3f& focus) noexcept;

    /**
     * @brief Orbit around the persistent focus point by mouse delta in pixels.
     *
     * Free-look and
     * dolly motion do not move this pivot. Call `focus()` (the
     * editor's F shortcut) or `place()` to choose a new
     * one.
     */
    void orbit(float delta_x, float delta_y) noexcept;

    /**
     * @brief Rotate the camera in place using stable world-yaw/local-pitch axes.
     *
     * The persistent
     * orbit focus remains unchanged.
     */
    void look(float delta_x, float delta_y) noexcept;

    /**
     * @brief Pan focus point by mouse delta in pixels.
     */
    void pan(float delta_x, float delta_y) noexcept;

    /**
     * @brief Move the camera forward along the current look direction.
     *
     * The persistent orbit focus
     * remains unchanged.
     */
    void move_forward(float delta_y) noexcept;

    /**
     * @brief Translate the camera forward or backward without changing orbit distance.
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
    math::vector3f position_{};
    float yaw_{};
    float pitch_{-0.18f};
    float distance_{4.0f};
};

/**
 * @brief Roll-free directional-light rotation controlled by viewport mouse deltas.
 *
 * Yaw is applied around
 * world +Y and pitch around the yawed local +X axis,
 * matching ARC's turntable camera convention. The resulting
 * transform points
 * its local -Z axis along the authored sunlight direction.
 */
class editor_sun_controller
{
public:
    /** @brief Initialize yaw and pitch from a directional-light transform. */
    void synchronize_from(const scene::transform_component& transform) noexcept;

    /** @brief Apply a mouse delta in pixels while keeping the light roll-free. */
    void rotate(float delta_x, float delta_y) noexcept;

    /** @brief Write the current roll-free rotation to a light transform. */
    void apply_to(scene::transform_component& transform) const noexcept;

private:
    float yaw_{};
    float pitch_{-0.75f};
};

/**
 * @brief Switch editor tool from configured keyboard shortcuts.
 */
void apply_tool_shortcuts(const input::input_manager& input, editor_tool& tool) noexcept;

/**
 * @brief Clear the current selection and all scene selection components.
 */
void clear_selection(ecs::world& registry, ecs::entity& selected) noexcept;

/**
 * @brief Select a live entity and synchronize `selection_component`.
 */
bool select_entity(ecs::world& registry, ecs::entity entity, ecs::entity& selected);

/**
 * @brief Pick the nearest bounded entity hit by a ray.
 */
ecs::entity pick_bounded_entity(const ecs::world& registry, const editor_ray& ray) noexcept;

/**
 * @brief Pick the nearest exact terrain or static-mesh surface, with bounds fallback.
 */
editor_pick_result pick_scene_entity(const ecs::world& registry, const render::renderer& renderer,
                                     const editor_ray& ray) noexcept;

/**
 * @brief Build a world-space picking ray from camera and viewport coordinates.
 */
editor_ray screen_ray_from_camera(const scene::camera_component& camera,
                                  const scene::transform_component& camera_transform, const editor_viewport& viewport,
                                  float local_x, float local_y) noexcept;

/**
 * @brief Return whether a ray intersects a box, writing nearest distance when hit.
 */
bool intersect_ray_box(const editor_ray& ray, const geometric::box3f& bounds, float& distance) noexcept;

/**
 * @brief Transform local bounds by a transform into world-space AABB bounds.
 */
geometric::box3f transformed_bounds(const geometric::box3f& local_bounds,
                                    const scene::transform_component& transform) noexcept;

/**
 * @brief Focus a camera controller on the selected entity.
 */
bool focus_selected_entity(const ecs::world& registry, ecs::entity selected, editor_camera_controller& camera) noexcept;

/**
 * @brief Convert Euler degrees to a quaternion.
 */
math::quatf quaternion_from_euler_degrees(const math::vector3f& degrees) noexcept;

/**
 * @brief Convert a quaternion to Euler degrees.
 */
math::vector3f euler_degrees_from_quaternion(const math::quatf& rotation) noexcept;

} // namespace arc::editor
