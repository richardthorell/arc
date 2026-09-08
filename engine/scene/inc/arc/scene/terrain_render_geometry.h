#pragma once

#include <arc/render/virtual_mesh.h>
#include <arc/scene/terrain_surface_ir.h>

#include <optional>

namespace arc::scene
{

/**
 * @brief Compile one evaluated terrain surface into ARC's generic render-geometry artifact.
 *
 * The resulting payload contains the normal virtual-geometry hierarchy/pages plus conventional cooked LOD fallback
 * generated from the same canonical triangle source. Terrain material/attribute data intentionally remains outside
 * this geometry-only artifact.
 *
 * @return Compiled render geometry, or std::nullopt when the evaluated surface is invalid or produces no renderable
 * triangles.
 */
[[nodiscard]] std::optional<render::virtual_mesh_data>
build_terrain_render_geometry(const terrain_surface_ir& surface,
                              const render::virtual_mesh_build_options& options = {});

} // namespace arc::scene
