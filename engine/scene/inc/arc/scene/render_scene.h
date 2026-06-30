#pragma once

#include <arc/render/renderer.h>
#include <arc/render/render_world.h>
#include <arc/scene/components.h>
#include <arc/scene/registry.h>

#include <cstdint>

namespace arc::scene
{

/**
 * @brief Summary of one scene render extraction.
 */
struct render_scene_result
{
    bool camera_found{};
    std::size_t renderable_count{};
    std::size_t submitted_draw_count{};
    std::size_t sky_atmosphere_count{};
    std::size_t height_fog_count{};
    std::size_t terrain_count{};
    std::size_t water_count{};
    std::size_t vegetation_count{};
    std::size_t vegetation_instance_count{};
    std::size_t decal_count{};
    std::size_t directional_light_count{};
    std::size_t point_light_count{};
    std::size_t spot_light_count{};
    std::size_t skipped_directional_light_count{};
    std::size_t skipped_point_light_count{};
    std::size_t skipped_spot_light_count{};
    std::size_t reflection_probe_count{};
    std::size_t irradiance_probe_count{};
    std::size_t selected_count{};
    std::size_t culled_count{};
    std::size_t instance_batch_count{};
    std::size_t indirect_draw_count{};
};

/**
 * @brief Editor/runtime visibility filters for optional environment systems.
 */
struct render_environment_visibility
{
    bool sky{ true };
    bool fog{ true };
    bool terrain{ true };
    bool water{ true };
    bool vegetation{ true };
    bool decals{ true };
};

/**
 * @brief Extract visible scene renderers into renderer frame events.
 */
render_scene_result render_scene(
    registry& scene,
    render::renderer& renderer,
    std::uint32_t viewport_width,
    std::uint32_t viewport_height,
    render::render_mode mode = render::render_mode::shaded,
    render::mesh_visualization_mode visualization = render::mesh_visualization_mode::standard,
    render::editor_overlay_mode overlay = render::editor_overlay_mode::selected_wireframe,
    bool shadows_enabled = true,
    render_environment_visibility environment_visibility = {});

} // namespace arc::scene
