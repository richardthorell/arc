#pragma once

#include <arc/scene/components.h>
#include <arc/scene/terrain_asset.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>

namespace arc::scene
{

/** @brief Backend-independent heightfield view produced by terrain evaluation. */
struct terrain_surface_heightfield_ir
{
    std::uint32_t sample_width{};
    std::uint32_t sample_height{};
    float width{};
    float depth{};
    std::span<const float> heights;
    std::span<const std::array<std::uint8_t, 4>> material_weights;
};

/** @brief Backend-independent triangle surface view for future mesh-authored terrain evaluation. */
struct terrain_surface_mesh_ir
{
    std::span<const math::vector3f> positions;
    std::span<const std::uint32_t> indices;
};

using terrain_surface_geometry_ir = std::variant<terrain_surface_heightfield_ir, terrain_surface_mesh_ir>;

/**
 * @brief Ephemeral evaluated terrain surface consumed by runtime systems.
 *
 * The spans reference evaluator-owned memory and are valid only for the lifetime of that evaluated source. Renderer,
 * collision, navigation, and destruction compilers consume this contract rather than the authored source type.
 */
struct terrain_surface_ir
{
    static constexpr std::uint32_t current_schema_version = 1;

    std::uint32_t schema_version{current_schema_version};
    std::uint64_t source_revision{};
    terrain_world_bounds local_bounds{};
    terrain_surface_geometry_ir geometry;
};

/** @brief Validate topology and referenced sample/index data without depending on a renderer backend. */
[[nodiscard]] bool validate_terrain_surface_ir(const terrain_surface_ir& surface) noexcept;

/**
 * @brief Adapt the current inline heightfield terrain cache into TerrainSurfaceIR.
 *
 * This is the compatibility boundary used while authored terrain sources migrate to the unified terrain evaluator.
 */
[[nodiscard]] std::optional<terrain_surface_ir>
make_legacy_terrain_surface_ir(const terrain_component& terrain) noexcept;

} // namespace arc::scene
