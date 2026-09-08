#pragma once

#include <arc/scene/components.h>
#include <arc/scene/terrain_asset.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

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

/** @brief Backend-independent triangle surface view for mesh-authored or topology-changing terrain evaluation. */
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
 * collision, navigation, destruction, and virtual-geometry compilers consume this contract rather than authored source
 * types.
 */
struct terrain_surface_ir
{
    static constexpr std::uint32_t current_schema_version = 1;

    std::uint32_t schema_version{current_schema_version};
    std::uint64_t source_revision{};
    terrain_world_bounds local_bounds{};
    terrain_surface_geometry_ir geometry;
};

/** @brief Owning heightfield storage used by terrain evaluators before exposing an IR view. */
struct terrain_evaluated_heightfield
{
    std::uint32_t sample_width{};
    std::uint32_t sample_height{};
    float width{};
    float depth{};
    std::vector<float> heights;
    std::vector<std::array<std::uint8_t, 4>> material_weights;
};

/** @brief Owning arbitrary-topology surface storage used by terrain evaluators. */
struct terrain_evaluated_mesh
{
    std::vector<math::vector3f> positions;
    std::vector<std::uint32_t> indices;
};

using terrain_evaluated_geometry = std::variant<terrain_evaluated_heightfield, terrain_evaluated_mesh>;

/** @brief Owning evaluated terrain surface from which short-lived TerrainSurfaceIR views are produced. */
struct terrain_evaluated_surface
{
    std::uint32_t schema_version{terrain_surface_ir::current_schema_version};
    std::uint64_t source_revision{};
    terrain_world_bounds local_bounds{};
    terrain_evaluated_geometry geometry;

    [[nodiscard]] terrain_surface_ir view() const noexcept;
};

/** @brief Validate topology and referenced sample/index data without depending on a renderer backend. */
[[nodiscard]] bool validate_terrain_surface_ir(const terrain_surface_ir& surface) noexcept;

/** @brief Produce an owning copy of an existing IR view. */
[[nodiscard]] std::optional<terrain_evaluated_surface> copy_terrain_surface_ir(const terrain_surface_ir& surface);

/** @brief Stable process/platform-independent fingerprint of evaluated surface content. */
[[nodiscard]] std::uint64_t terrain_surface_fingerprint(const terrain_surface_ir& surface) noexcept;

/**
 * @brief Adapt the current inline heightfield terrain cache into TerrainSurfaceIR.
 *
 * This is the compatibility boundary used while authored terrain sources migrate to the unified terrain evaluator.
 */
[[nodiscard]] std::optional<terrain_surface_ir>
make_legacy_terrain_surface_ir(const terrain_component& terrain) noexcept;

} // namespace arc::scene
