#pragma once

#include <arc/scene/terrain_surface_ir.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace arc::scene
{

/**
 * @brief Renderer-independent per-surface terrain attributes kept separate from triangle geometry.
 *
 * The four RGBA8 channels represent authored terrain layer weights. Mesh-authored terrain currently has no
 * per-surface weight source in TerrainSurfaceIR, so it deterministically falls back to layer zero until a richer
 * attribute source is supplied by the evaluator.
 */
struct terrain_render_attributes
{
    std::uint32_t width{1u};
    std::uint32_t height{1u};
    std::vector<std::array<std::uint8_t, 4>> material_weights{{255u, 0u, 0u, 0u}};
    bool default_layer_only{true};
};

/**
 * @brief Compile terrain material attributes without introducing terrain semantics into generic mesh vertices.
 */
[[nodiscard]] std::optional<terrain_render_attributes>
build_terrain_render_attributes(const terrain_surface_ir& surface);

} // namespace arc::scene
