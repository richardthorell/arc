#pragma once

#include <arc/render/mesh.h>

#include <cstdint>

namespace arc::render
{

/**
 * @brief Create a flat XZ plane centered at the origin.
 */
mesh_data make_plane_mesh(float size = 1.0f);

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
float sample_terrain_height(float x, float z, float size, float height_scale) noexcept;

/**
 * @brief Create a deterministic clump of simple crossed grass blades.
 */
mesh_data make_grass_patch_mesh(float patch_size = 8.0f, std::uint32_t blade_count = 96, float height = 0.65f);

} // namespace arc::render
