#include <arc/render/virtual_mesh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace arc::render
{
namespace
{

math::vector3f vertex_position(const mesh_vertex& vertex) noexcept
{
    return { vertex.position[0], vertex.position[1], vertex.position[2] };
}

math::vector3f min_vector(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return {
        std::min(lhs[0], rhs[0]),
        std::min(lhs[1], rhs[1]),
        std::min(lhs[2], rhs[2])
    };
}

math::vector3f max_vector(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return {
        std::max(lhs[0], rhs[0]),
        std::max(lhs[1], rhs[1]),
        std::max(lhs[2], rhs[2])
    };
}

void include_vertex(virtual_mesh_cluster& cluster, const mesh_vertex& vertex, bool& has_bounds) noexcept
{
    const auto position = vertex_position(vertex);
    if (!has_bounds)
    {
        cluster.bounds_min = position;
        cluster.bounds_max = position;
        has_bounds = true;
        return;
    }

    cluster.bounds_min = min_vector(cluster.bounds_min, position);
    cluster.bounds_max = max_vector(cluster.bounds_max, position);
}

void finish_cluster(virtual_mesh_data& result, virtual_mesh_cluster& cluster)
{
    if (cluster.triangle_count == 0)
        return;

    cluster.index_count = cluster.triangle_count * 3u;
    cluster.sphere_center = math::mul(math::add(cluster.bounds_min, cluster.bounds_max), 0.5f);
    cluster.sphere_radius = 0.0f;

    const auto index_end = cluster.first_index + cluster.index_count;
    for (auto index = cluster.first_index; index < index_end; ++index)
    {
        const auto vertex_index = result.indices[index];
        if (vertex_index >= result.vertices.size())
            continue;
        const auto delta = math::sub(vertex_position(result.vertices[vertex_index]), cluster.sphere_center);
        cluster.sphere_radius = std::max(cluster.sphere_radius, math::length(delta));
    }

    result.clusters.push_back(cluster);
    cluster = {};
}

} // namespace

virtual_mesh_data build_virtual_mesh(const mesh_data& source, const virtual_mesh_build_options& options)
{
    virtual_mesh_data result;
    result.vertices = source.vertices;
    result.stats.source_vertex_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(source.vertices.size(), std::numeric_limits<std::uint32_t>::max()));
    result.stats.source_triangle_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(source.indices.size() / 3u, std::numeric_limits<std::uint32_t>::max()));

    const auto max_triangles_per_cluster = std::max(1u, options.max_triangles_per_cluster);
    const auto trailing_index_count = source.indices.size() % 3u;
    if (trailing_index_count != 0)
        ++result.stats.invalid_triangle_count;

    virtual_mesh_cluster cluster{};
    cluster.material_index = source.material_index;
    std::uint32_t valid_triangle_count{};
    std::uint32_t min_vertex = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t max_vertex{};
    bool cluster_has_bounds{};

    const auto flush_cluster = [&]() {
        if (cluster.triangle_count == 0)
            return;
        cluster.first_vertex = min_vertex;
        cluster.vertex_count = max_vertex >= min_vertex ? (max_vertex - min_vertex + 1u) : 0u;
        finish_cluster(result, cluster);
        min_vertex = std::numeric_limits<std::uint32_t>::max();
        max_vertex = 0;
        cluster_has_bounds = false;
        cluster.material_index = source.material_index;
    };

    for (std::size_t index = 0; index + 2 < source.indices.size(); index += 3)
    {
        const std::uint32_t i0 = source.indices[index + 0];
        const std::uint32_t i1 = source.indices[index + 1];
        const std::uint32_t i2 = source.indices[index + 2];
        if (i0 >= source.vertices.size() || i1 >= source.vertices.size() || i2 >= source.vertices.size())
        {
            ++result.stats.invalid_triangle_count;
            continue;
        }

        if (cluster.triangle_count == 0)
        {
            cluster.first_index = static_cast<std::uint32_t>(result.indices.size());
            cluster.first_triangle = valid_triangle_count;
            cluster.material_index = source.material_index;
        }

        const std::array<std::uint32_t, 3> triangle{ i0, i1, i2 };
        for (const auto vertex_index : triangle)
        {
            result.indices.push_back(vertex_index);
            include_vertex(cluster, source.vertices[vertex_index], cluster_has_bounds);
            min_vertex = std::min(min_vertex, vertex_index);
            max_vertex = std::max(max_vertex, vertex_index);
        }
        ++cluster.triangle_count;
        ++valid_triangle_count;

        if (cluster.triangle_count >= max_triangles_per_cluster)
            flush_cluster();
    }
    flush_cluster();

    result.stats.cluster_count = static_cast<std::uint32_t>(result.clusters.size());
    result.stats.material_group_count = result.clusters.empty() ? 0u : 1u;
    result.stats.average_triangles_per_cluster = result.stats.cluster_count == 0
        ? 0.0f
        : static_cast<float>(valid_triangle_count) / static_cast<float>(result.stats.cluster_count);
    return result;
}

} // namespace arc::render
