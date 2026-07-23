#pragma once

#include <arc/editor/editor_interaction.h>
#include <arc/render/render_world.h>

namespace arc::editor
{

enum class gizmo_coordinate_space : std::uint8_t { world, local };
enum class gizmo_axis : std::uint8_t { none, x, y, z };
inline constexpr float editor_gizmo_pixel_length = 92.0f;

struct editor_gizmo_context
{
    editor_tool tool{ editor_tool::translate };
    gizmo_coordinate_space coordinate_space{ gizmo_coordinate_space::world };
    gizmo_axis highlighted_axis{ gizmo_axis::none };
    std::uint32_t viewport_width{};
    std::uint32_t viewport_height{};
};

float editor_gizmo_world_scale(
    const scene::camera_component& camera,
    const scene::transform_component& camera_transform,
    const math::vector3f& world_position,
    std::uint32_t viewport_height) noexcept;

render::debug_overlay_stream build_editor_gizmo_overlay(
    const scene::registry& registry,
    scene::entity selected,
    scene::entity camera_entity,
    const editor_gizmo_context& context);

gizmo_axis hit_test_editor_gizmo(
    const scene::registry& registry,
    scene::entity selected,
    scene::entity camera_entity,
    const editor_gizmo_context& context,
    float screen_x,
    float screen_y) noexcept;

} // namespace arc::editor
