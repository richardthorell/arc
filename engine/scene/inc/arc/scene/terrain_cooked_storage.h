#pragma once

#include <arc/assets/cook.h>
#include <arc/core/result.h>
#include <arc/scene/terrain_artifacts.h>
#include <arc/scene/terrain_render_regions.h>

#include <string>
#include <string_view>
#include <vector>

namespace arc::scene
{

struct terrain_cooked_storage_error
{
    std::string message;
};

/** Cook output for one TerrainAsset: a lightweight region manifest plus independently stored payload artifacts. */
struct terrain_cooked_storage
{
    terrain_cooked_manifest manifest;
    std::vector<assets::cooked_artifact> artifacts;
};

using terrain_cooked_storage_result = core::result<terrain_cooked_storage, terrain_cooked_storage_error>;

/**
 * @brief Compile an evaluated terrain surface into independently addressable per-region cooked artifacts.
 *
 * Detailed virtual-geometry pages stay in external `.arcvg` artifacts. The lightweight terrain manifest records
 * bounds, generations, metadata ranges, root/page offsets, sizes and hashes so a runtime can discover and range-read
 * one region without realizing every region. Asynchronous scheduling is deliberately left to M2.3.
 */
[[nodiscard]] terrain_cooked_storage_result
build_terrain_cooked_storage(assets::asset_guid terrain, const terrain_surface_ir& surface,
                             std::uint64_t authoring_revision, std::string_view target_profile = "default",
                             double target_region_size = default_terrain_render_region_size);

} // namespace arc::scene
