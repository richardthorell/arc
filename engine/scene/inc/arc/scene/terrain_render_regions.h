#pragma once

#include <arc/scene/terrain_asset.h>
#include <arc/scene/terrain_surface_ir.h>

#include <cstdint>
#include <vector>

namespace arc::scene
{

/** Default world-space edge length used to group terrain render artifacts. */
inline constexpr double default_terrain_render_region_size = 256.0;

/**
 * @brief One independently compilable terrain render region.
 *
 * Region geometry and material attributes are owning so asynchronous/cached compilation can
 * outlive the source view. vertex_normals contains normals evaluated from the complete source
 * surface so duplicated boundary vertices remain shading-compatible across neighboring regions.
 */
struct terrain_render_region
{
    terrain_region_id id{};
    terrain_evaluated_surface surface{};
    std::vector<math::vector3f> vertex_normals;
    std::uint64_t geometry_fingerprint{};
    std::uint64_t attribute_fingerprint{};
};

/**
 * @brief Partition an evaluated terrain surface into stable sample-aligned render regions.
 *
 * Heightfields are split only on source quad boundaries and duplicate the shared boundary row/
 * column, guaranteeing identical seam positions. Mesh surfaces remain a single region until the
 * mesh-native terrain milestone introduces topology-aware spatial partitioning.
 */
[[nodiscard]] std::vector<terrain_render_region>
build_terrain_render_regions(const terrain_surface_ir& surface,
                             double target_region_size = default_terrain_render_region_size);

/** Stable non-zero render-instance discriminator for one terrain region. */
[[nodiscard]] std::uint64_t terrain_render_region_instance_id(terrain_region_id id) noexcept;

} // namespace arc::scene
