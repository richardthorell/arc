#pragma once

#include <arc/math/vector.h>
#include <arc/render/handles.h>
#include <arc/render/mesh.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace arc::render
{

using virtual_mesh_handle = resource_handle;

/**
 * @brief Fixed-size virtual geometry cluster built from source mesh triangles.
 */
struct virtual_mesh_cluster
{
    std::uint32_t first_index{};
    std::uint32_t index_count{};
    std::uint32_t first_triangle{};
    std::uint32_t triangle_count{};
    std::uint32_t first_vertex{};
    std::uint32_t vertex_count{};
    std::size_t material_index{ std::numeric_limits<std::size_t>::max() };
    math::vector3f bounds_min{};
    math::vector3f bounds_max{};
    math::vector3f sphere_center{};
    float sphere_radius{};
};

/**
 * @brief Placeholder hierarchy node for future virtual mesh LOD construction.
 */
struct virtual_mesh_lod_node
{
    std::uint32_t first_cluster{};
    std::uint32_t cluster_count{};
    std::uint32_t first_child{};
    std::uint32_t child_count{};
    float error{};
    math::vector3f bounds_min{};
    math::vector3f bounds_max{};
    math::vector3f sphere_center{};
    float sphere_radius{};
};

/**
 * @brief Deterministic statistics from building a virtual mesh asset.
 */
struct virtual_mesh_build_stats
{
    std::uint32_t source_vertex_count{};
    std::uint32_t source_triangle_count{};
    std::uint32_t cluster_count{};
    float average_triangles_per_cluster{};
    std::uint32_t material_group_count{};
    std::uint32_t invalid_triangle_count{};
};

/**
 * @brief Options controlling the first fixed-size virtual mesh builder.
 */
struct virtual_mesh_build_options
{
    std::uint32_t max_triangles_per_cluster{ 128 };
};

/**
 * @brief CPU-side virtual mesh asset data independent of classic mesh uploads.
 */
struct virtual_mesh_data
{
    std::vector<mesh_vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<virtual_mesh_cluster> clusters;
    std::vector<virtual_mesh_lod_node> lod_nodes;
    virtual_mesh_build_stats stats;
};

/**
 * @brief Build fixed-size virtual mesh clusters from classic mesh data.
 */
virtual_mesh_data build_virtual_mesh(const mesh_data& source, const virtual_mesh_build_options& options = {});

} // namespace arc::render
