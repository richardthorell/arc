#pragma once

#include <arc/math/vector.h>
#include <arc/render/handles.h>
#include <arc/render/mesh.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace arc::render
{

using virtual_mesh_handle = resource_handle;

inline constexpr std::uint32_t virtual_geometry_max_vertices_per_cluster = 64;
inline constexpr std::uint32_t virtual_geometry_max_triangles_per_cluster = 124;
inline constexpr std::uint32_t virtual_geometry_page_bytes = 64u * 1024u;
inline constexpr std::uint32_t virtual_geometry_decoded_vertex_bytes = 24u;
inline constexpr std::uint32_t virtual_geometry_decoded_cluster_header_bytes = 16u;
inline constexpr std::uint32_t invalid_virtual_geometry_index = std::numeric_limits<std::uint32_t>::max();

/** @brief Runtime representation policy authored on an ordinary mesh renderer. */
enum class geometry_representation_policy : std::uint8_t
{
    auto_select,
    conventional,
    virtualized
};

/** @brief Raster implementation selected for virtual geometry. */
enum class virtual_geometry_raster_path : std::uint8_t
{
    unavailable,
    compute,
    mesh_shader
};

/**
 * @brief Unified runtime binding for conventional LODs and virtual geometry from one source asset.
 *
 * The first conventional handle remains the compatibility LOD0 handle. Asset realization may
 * populate up to four cooked LODs and one virtual artifact without changing scene representation.
 */
struct geometry_resource_handle
{
    mesh_handle conventional{};
    std::array<mesh_handle, 4> conventional_lods{};
    std::array<float, 4> conventional_lod_errors{};
    virtual_mesh_handle virtualized{};
    std::uint8_t conventional_lod_count{};
    std::uint32_t asset_generation{};

    constexpr geometry_resource_handle() noexcept = default;
    constexpr geometry_resource_handle(mesh_handle value) noexcept
        : conventional(value), conventional_lods{value}, conventional_lod_count(value.valid() ? 1u : 0u)
    {
    }

    constexpr geometry_resource_handle& operator=(mesh_handle value) noexcept
    {
        conventional = value;
        conventional_lods = {value};
        conventional_lod_errors = {};
        conventional_lod_count = value.valid() ? 1u : 0u;
        return *this;
    }

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return conventional.valid();
    }
    [[nodiscard]] constexpr operator mesh_handle() const noexcept
    {
        return conventional;
    }

    /** @brief Select the coarsest valid cooked LOD whose object-space error remains acceptable. */
    [[nodiscard]] constexpr mesh_handle select_conventional_lod(float maximum_object_space_error) const noexcept
    {
        const auto count = std::min<std::size_t>(conventional_lod_count, conventional_lods.size());
        for (std::size_t index = count; index > 0; --index)
            if (conventional_lods[index - 1].valid() &&
                conventional_lod_errors[index - 1] <= maximum_object_space_error)
                return conventional_lods[index - 1];
        return conventional;
    }
};

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
    std::size_t material_index{std::numeric_limits<std::size_t>::max()};
    math::vector3f bounds_min{};
    math::vector3f bounds_max{};
    math::vector3f sphere_center{};
    float sphere_radius{};
    math::vector3f cone_axis{};
    float cone_cutoff{-1.0f};
    float geometric_error{};
    std::uint32_t hierarchy_node{invalid_virtual_geometry_index};
    std::uint32_t page_index{invalid_virtual_geometry_index};
    std::uint32_t page_byte_offset{};
    std::uint16_t hierarchy_level{};
    std::uint16_t flags{};
};

/**
 * @brief One deterministic node in a virtual-geometry cluster hierarchy.
 */
struct virtual_mesh_lod_node
{
    std::uint32_t first_cluster{};
    std::uint32_t cluster_count{};
    std::uint32_t first_child{};
    std::uint32_t child_count{};
    std::uint32_t parent{invalid_virtual_geometry_index};
    std::uint32_t page_index{invalid_virtual_geometry_index};
    float error{};
    math::vector3f bounds_min{};
    math::vector3f bounds_max{};
    math::vector3f sphere_center{};
    float sphere_radius{};
    math::vector3f cone_axis{};
    float cone_cutoff{-1.0f};
    std::uint16_t level{};
    std::uint16_t flags{};
};

/** @brief One independently streamable virtual-geometry page. */
struct virtual_geometry_page
{
    std::uint32_t first_cluster{};
    std::uint32_t cluster_count{};
    std::uint32_t uncompressed_offset{};
    std::uint32_t uncompressed_size{};
    std::uint32_t compressed_offset{};
    std::uint32_t compressed_size{};
    std::uint64_t content_hash{};
    bool root{};
};

/** @brief Conventional mesh data for one generated fallback LOD. */
struct conventional_mesh_lod
{
    float ratio{1.0f};
    float geometric_error{};
    std::vector<mesh_vertex> vertices;
    std::vector<std::uint32_t> indices;
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
    std::uint32_t hierarchy_level_count{};
    std::uint32_t page_count{};
    std::uint32_t root_page_count{};
    std::uint64_t uncompressed_page_bytes{};
    std::uint64_t compressed_page_bytes{};
    std::uint32_t boundary_edge_count{};
};

/**
 * @brief Options controlling the first fixed-size virtual mesh builder.
 */
struct virtual_mesh_build_options
{
    std::uint32_t max_vertices_per_cluster{virtual_geometry_max_vertices_per_cluster};
    std::uint32_t max_triangles_per_cluster{virtual_geometry_max_triangles_per_cluster};
    std::uint32_t minimum_group_size{4};
    std::uint32_t maximum_group_size{8};
    float parent_triangle_ratio{0.5f};
    std::uint32_t maximum_root_clusters{4};
    bool build_conventional_lods{true};
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
    std::vector<std::uint32_t> hierarchy_children;
    std::vector<std::uint32_t> root_nodes;
    std::vector<virtual_geometry_page> pages;
    std::vector<std::byte> page_payload;
    std::vector<conventional_mesh_lod> conventional_lods;
    virtual_mesh_build_stats stats;
};

/**
 * @brief Build topology-aware clusters, hierarchy pages, and fallback LODs.
 */
virtual_mesh_data build_virtual_mesh(const mesh_data& source, const virtual_mesh_build_options& options = {});

/** @brief Decode one independently compressed page into caller-owned storage. */
[[nodiscard]] bool decode_virtual_geometry_page(const virtual_mesh_data& mesh, std::uint32_t page_index,
                                                std::vector<std::byte>& output);

} // namespace arc::render
