#pragma once

#include <arc/scene/terrain_asset.h>

#include <cstdint>
#include <optional>

namespace arc::scene
{

enum class terrain_region_edge : std::uint8_t
{
    negative_x,
    positive_x,
    negative_z,
    positive_z
};

/** @brief Return the region directly across one canonical authoring edge. */
[[nodiscard]] terrain_region_id terrain_region_neighbor(terrain_region_id region, terrain_region_edge edge) noexcept;

/** @brief True only for four-connected authoring regions sharing exactly one full edge. */
[[nodiscard]] bool terrain_regions_share_edge(terrain_region_id lhs, terrain_region_id rhs) noexcept;

/**
 * @brief Deterministic owner for shared boundary conditions.
 *
 * The lexicographically smaller region owns the canonical seam. Non-neighboring inputs return no owner. This ownership
 * belongs to terrain evaluation/builds and is intentionally independent from virtual-geometry cluster/page boundaries.
 */
[[nodiscard]] std::optional<terrain_region_id> terrain_shared_seam_owner(terrain_region_id lhs,
                                                                        terrain_region_id rhs) noexcept;

} // namespace arc::scene
