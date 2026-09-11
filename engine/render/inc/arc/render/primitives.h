#pragma once

#include <arc/render/mesh.h>

#include <cstdint>

namespace arc::render
{

/** @brief Topology controls for the flat camera-relative W0 Ocean surface. */
struct water_ocean_grid_descriptor
{
    float visible_distance{20000.0f};
    std::uint32_t inner_grid_cells{32u};
    std::uint32_t ring_count{8u};
};

/**
 * @brief Create a flat XZ plane centered at the origin.
 */
mesh_data make_plane_mesh(float size = 1.0f);

/**
 * @brief Create a flat square center patch surrounded by progressively coarser rings.
 *
 * The mesh is authored around the origin and is intended to be translated to a snapped camera-relative origin each
 * frame. Its outer half-extent equals `visible_distance`.
 */
mesh_data make_water_ocean_grid(const water_ocean_grid_descriptor& descriptor = {});

/** @brief Finest cell size used to quantize movement of a camera-relative Ocean grid. */
[[nodiscard]] float water_ocean_grid_cell_size(const water_ocean_grid_descriptor& descriptor = {}) noexcept;

/** @brief Snap an Ocean grid to the camera on XZ while preserving the authored water level on Y. */
[[nodiscard]] math::vector3f water_ocean_grid_origin(const math::vector3f& camera_position, float water_level,
                                                     float cell_size) noexcept;

/**
 * @brief Create a cube centered at the origin.
 */
mesh_data make_cube_mesh(float size = 1.0f);

/**
 * @brief Create a UV sphere centered at the origin.
 */
mesh_data make_uv_sphere_mesh(float radius = 0.5f, std::uint32_t slices = 32, std::uint32_t stacks = 16);

/**
 * @brief Create a capped cylinder centered at the origin.
 */
mesh_data make_cylinder_mesh(float radius = 0.5f, float height = 1.0f, std::uint32_t segments = 32);

/**
 * @brief Create a capped cone centered at the origin.
 */
mesh_data make_cone_mesh(float radius = 0.5f, float height = 1.0f, std::uint32_t segments = 32);

/**
 * @brief Create a Y-axis capsule centered at the origin.
 */
mesh_data make_capsule_mesh(float radius = 0.5f, float cylinder_height = 1.0f, std::uint32_t segments = 32,
                            std::uint32_t hemisphere_segments = 8);

/**
 * @brief Create a generated XZ terrain grid centered at the origin.
 *
 * The surface is a deterministic mountain-and-valley composition. Texture
 * coordinates are world-scaled so tileable landscape materials retain detail
 * as the terrain size grows.
 */
mesh_data make_terrain_grid_mesh(float size = 24.0f, std::uint32_t subdivisions = 32, float height_scale = 0.35f);

/**
 * @brief Sample the same deterministic terrain height used by the grid mesh.
 *
 * This lets editor scenery, water, and gameplay markers sit on the generated
 * surface without duplicating its generation formula.
 */
[[nodiscard]] float sample_terrain_height(float x, float z, float size, float height_scale) noexcept;

/**
 * @brief Create a deterministic clump of simple crossed grass blades.
 */
mesh_data make_grass_patch_mesh(float patch_size = 8.0f, std::uint32_t blade_count = 96, float height = 0.65f);

} // namespace arc::render
