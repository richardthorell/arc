#pragma once

#include <arc/editor/editor_interaction.h>
#include <arc/render/render_world.h>

namespace arc::editor
{

enum class gizmo_coordinate_space : std::uint8_t
{
    world,
    local
};
enum class gizmo_axis : std::uint8_t
{
    none,
    x,
    y,
    z,
    all
};
/** Desired axis length in output pixels, independent of camera distance. */
inline constexpr float editor_gizmo_pixel_length = 112.0f;

struct editor_gizmo_context
{
    editor_tool tool{editor_tool::translate};
    gizmo_coordinate_space coordinate_space{gizmo_coordinate_space::world};
    gizmo_axis highlighted_axis{gizmo_axis::none};
    std::uint32_t viewport_width{};
    std::uint32_t viewport_height{};
};

float editor_gizmo_world_scale(const scene::camera_component& camera,
                               const scene::transform_component& camera_transform, const math::vector3f& world_position,
                               std::uint32_t viewport_height) noexcept;

/**
 * @brief Append an adaptive camera-centered XZ editor grid.
 *
 * Grid spacing follows powers of ten so minor lines retain a useful on-screen
 * density while zooming. World axes remain anchored at the origin.
 */
void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,
                                const scene::transform_component& camera_transform,
                                std::uint32_t viewport_height);

render::debug_overlay_stream build_editor_gizmo_overlay(const ecs::world& registry, ecs::entity selected,
                                                        ecs::entity camera_entity, const editor_gizmo_context& context);

gizmo_axis hit_test_editor_gizmo(const ecs::world& registry, ecs::entity selected, ecs::entity camera_entity,
                                 const editor_gizmo_context& context, float screen_x, float screen_y) noexcept;

/**
 * @brief Resolve the positive screen-space drag direction for one visible gizmo axis.
 * @param registry World containing the selected entity and editor camera.
 * @param selected Entity manipulated by the gizmo.
 * @param camera_entity Active editor camera entity.
 * @param context Current tool, coordinate space, and viewport extent.
 * @param axis Axis selected by hit testing.
 * @param screen_x Pointer x-coordinate at drag start, in output pixels.
 * @param screen_y Pointer y-coordinate at drag start, in output pixels.
 * @param direction Receives a normalized screen-space direction where positive motion increases the value.
 * @return `true` when a stable projected drag direction could be resolved.
 */
bool editor_gizmo_drag_direction(const ecs::world& registry, ecs::entity selected, ecs::entity camera_entity,
                                 const editor_gizmo_context& context, gizmo_axis axis, float screen_x, float screen_y,
                                 math::vector2f& direction) noexcept;

} // namespace arc::editor
