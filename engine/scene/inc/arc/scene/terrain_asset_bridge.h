#pragma once

#include <arc/assets/assets.h>
#include <arc/scene/components.h>

#include <cstdint>
#include <string>

namespace arc::scene
{

/** @brief Result of resolving one scene terrain component against its authored terrain asset. */
struct terrain_asset_binding_result
{
    bool succeeded{};
    bool bound{};
    bool using_legacy_surface{true};
    std::uint64_t generation{};
    std::uint64_t authoring_revision{};
    std::string message;
};

/**
 * @brief Resolve and load the terrain asset referenced by a scene component.
 *
 * The current inline heightfield remains the compatibility surface cache. Loading an asset records its generation and
 * authoring revision without changing rendered samples; the TerrainSurfaceIR evaluator will replace that cache in the
 * next stage.
 */
[[nodiscard]] terrain_asset_binding_result refresh_terrain_asset_binding(terrain_component& terrain,
                                                                         assets::asset_manager& manager);

} // namespace arc::scene
