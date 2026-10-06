#pragma once

#include <arc/render/handles.h>
#include <arc/render/events.h>
#include <arc/math/vector.h>
#include <arc/water/water_types.h>

#include <cstdint>
#include <string>

namespace arc::render
{

/**
 * @brief Geometry/simulation source attached to a fluid surface.
 *
 * Fluid Surface intentionally does not define one universal simulation model. A source describes
 * which specialization owns geometry/deformation while material shading remains authoritative
 * through the normal Material ABI and canonical forward pass.
 */
enum class fluid_surface_source_kind : std::uint8_t
{
    static_surface,
    water
};

/** @brief Surface products a fluid source can provide to rendering. */
struct fluid_surface_channels
{
    bool displacement{};
    bool normals{true};
    bool velocity{};
    bool foam{};
    bool thickness{};
};

/**
 * @brief Water-specific provider data carried by a generic Fluid Surface.
 *
 * This remains Water-owned on purpose: FFT/JONSWAP, Ocean/Lake/River policy, shoreline and
 * underwater behavior are not generalized into the Fluid Surface contract.
 */
struct water_fluid_surface_source
{
    water::water_body_type type{water::water_body_type::ocean};
    water::water_runtime_settings settings;
    float water_level{};
    float visible_distance{20000.0f};
    float finest_grid_cell_size{};
    std::uint32_t grid_ring_count{8u};
    std::int32_t priority{};
    bool follow_camera{true};
    bool shoreline_enabled{true};
    bool underwater_enabled{true};
};

/**
 * @brief Backend-neutral, frame-immutable fluid surface resolved from scene authoring state.
 *
 * The common contract owns identity, placement and material. Specialized sources only provide
 * geometry/simulation products. Surface lighting always comes from the assigned ARC material.
 */
struct fluid_surface_render_instance
{
    render_object_id object_id{};
    material_handle material{};
    /** Authored entity position before any source-specific camera-relative placement. */
    math::vector3f position{};
    /** World-space origin of the rendered surface for this view. */
    math::vector3f surface_origin{};
    fluid_surface_source_kind source_kind{fluid_surface_source_kind::static_surface};
    fluid_surface_channels channels{};
    water_fluid_surface_source water;
    std::string label;

    [[nodiscard]] constexpr bool is_water() const noexcept
    {
        return source_kind == fluid_surface_source_kind::water;
    }
};

} // namespace arc::render
