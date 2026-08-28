#include <arc/render/virtual_mesh.h>

#include <meshoptimizer.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace arc::render
{
namespace
{

struct packed_virtual_vertex
{
    std::uint16_t position[3]{};
    std::int16_t normal[2]{};
    std::int16_t tangent[2]{};
    std::uint16_t texcoord[2]{};
    std::uint8_t color[4]{};
    std::int8_t tangent_sign{1};
    std::uint8_t padding[1]{};
};

static_assert(sizeof(packed_virtual_vertex) == 24);

struct encoded_cluster_header
{
    std::uint32_t cluster_index{};
    std::uint16_t vertex_count{};
    std::uint16_t triangle_count{};
    std::uint32_t encoded_vertex_bytes{};
    std::uint32_t triangle_bytes{};
};

static_assert(sizeof(encoded_cluster_header) == virtual_geometry_decoded_cluster_header_bytes);

struct hierarchy_work_node
{
    std::uint32_t node_index{};
    std::vector<std::uint32_t> source_indices;
    std::vector<std::uint32_t> unique_vertices;
    math::vector3f center{};
    float error{};
};

math::vector3f vertex_position(const mesh_vertex& vertex) noexcept
{
    return {vertex.position[0], vertex.position[1], vertex.position[2]};
}

math::vector3f minimum(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return {std::min(lhs[0], rhs[0]), std::min(lhs[1], rhs[1]), std::min(lhs[2], rhs[2])};
}

math::vector3f maximum(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return {std::max(lhs[0], rhs[0]), std::max(lhs[1], rhs[1]), std::max(lhs[2], rhs[2])};
}

float squared_distance(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return math::length_squared(math::sub(lhs, rhs));
}

std::uint64_t edge_key(std::uint32_t lhs, std::uint32_t rhs) noexcept
{
    if (rhs < lhs) std::swap(lhs, rhs);
    return (static_cast<std::uint64_t>(lhs) << 32u) | rhs;
}

bool valid_triangle(const mesh_data& source, std::uint32_t i0, std::uint32_t i1, std::uint32_t i2) noexcept
{
    if (i0 >= source.vertices.size() || i1 >= source.vertices.size() || i2 >= source.vertices.size()) return false;
    if (i0 == i1 || i1 == i2 || i0 == i2) return false;
    const auto edge0 = math::sub(vertex_position(source.vertices[i1]), vertex_position(source.vertices[i0]));
    const auto edge1 = math::sub(vertex_position(source.vertices[i2]), vertex_position(source.vertices[i0]));
    return math::length_squared(math::cross(edge0, edge1)) > 1.0e-16f;
}

std::vector<std::uint32_t> sanitize_indices(const mesh_data& source, virtual_mesh_build_stats& stats)
{
    std::vector<std::uint32_t> result;
    result.reserve(source.indices.size());
    for (std::size_t index = 0; index + 2 < source.indices.size(); index += 3)
    {
        const auto i0 = source.indices[index + 0];
        const auto i1 = source.indices[index + 1];
        const auto i2 = source.indices[index + 2];
        if (!valid_triangle(source, i0, i1, i2))
        {
            ++stats.invalid_triangle_count;
            continue;
        }
        result.insert(result.end(), {i0, i1, i2});
    }
    if (source.indices.size() % 3u != 0) ++stats.invalid_triangle_count;
    return result;
}

std::vector<std::uint32_t> unique_vertices(std::span<const std::uint32_t> indices)
{
    std::vector<std::uint32_t> result(indices.begin(), indices.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::size_t shared_vertex_count(const hierarchy_work_node& lhs, const hierarchy_work_node& rhs) noexcept
{
    std::size_t count{};
    auto left = lhs.unique_vertices.begin();
    auto right = rhs.unique_vertices.begin();
    while (left != lhs.unique_vertices.end() && right != rhs.unique_vertices.end())
    {
        if (*left == *right)
        {
            ++count;
            ++left;
            ++right;
        }
        else if (*left < *right)
            ++left;
        else
            ++right;
    }
    return count;
}

void calculate_bounds(virtual_mesh_cluster& cluster, const std::vector<mesh_vertex>& vertices,
                      std::span<const std::uint32_t> indices)
{
    bool initialized{};
    math::vector3f normal_sum{};
    float minimum_axis_dot = 1.0f;
    for (const auto index : indices)
    {
        if (index >= vertices.size()) continue;
        const auto position = vertex_position(vertices[index]);
        if (!initialized)
        {
            cluster.bounds_min = position;
            cluster.bounds_max = position;
            initialized = true;
        }
        else
        {
            cluster.bounds_min = minimum(cluster.bounds_min, position);
            cluster.bounds_max = maximum(cluster.bounds_max, position);
        }
    }
    cluster.sphere_center = math::mul(math::add(cluster.bounds_min, cluster.bounds_max), 0.5f);
    for (const auto index : indices)
    {
        if (index >= vertices.size()) continue;
        cluster.sphere_radius = std::max(
            cluster.sphere_radius, math::length(math::sub(vertex_position(vertices[index]), cluster.sphere_center)));
    }
    for (std::size_t index = 0; index + 2 < indices.size(); index += 3)
    {
        const auto p0 = vertex_position(vertices[indices[index + 0]]);
        const auto p1 = vertex_position(vertices[indices[index + 1]]);
        const auto p2 = vertex_position(vertices[indices[index + 2]]);
        const auto face = math::normalize(math::cross(math::sub(p1, p0), math::sub(p2, p0)));
        normal_sum = math::add(normal_sum, face);
    }
    if (math::length_squared(normal_sum) > 1.0e-12f)
    {
        cluster.cone_axis = math::normalize(normal_sum);
        for (std::size_t index = 0; index + 2 < indices.size(); index += 3)
        {
            const auto p0 = vertex_position(vertices[indices[index + 0]]);
            const auto p1 = vertex_position(vertices[indices[index + 1]]);
            const auto p2 = vertex_position(vertices[indices[index + 2]]);
            const auto face = math::normalize(math::cross(math::sub(p1, p0), math::sub(p2, p0)));
            minimum_axis_dot = std::min(minimum_axis_dot, math::dot(cluster.cone_axis, face));
        }
        cluster.cone_cutoff = minimum_axis_dot > 0.0f ? minimum_axis_dot : -1.0f;
    }
}

std::uint32_t append_cluster(virtual_mesh_data& result, const mesh_data& source,
                             std::span<const std::uint32_t> source_indices, float error, std::uint16_t level)
{
    virtual_mesh_cluster cluster{};
    cluster.first_index = static_cast<std::uint32_t>(result.indices.size());
    cluster.first_triangle = cluster.first_index / 3u;
    cluster.first_vertex = static_cast<std::uint32_t>(result.vertices.size());
    cluster.material_index = source.material_index;
    cluster.geometric_error = error;
    cluster.hierarchy_level = level;

    const auto vertices = unique_vertices(source_indices);
    std::unordered_map<std::uint32_t, std::uint32_t> remap;
    remap.reserve(vertices.size());
    for (const auto source_vertex : vertices)
    {
        remap.emplace(source_vertex, static_cast<std::uint32_t>(result.vertices.size()));
        result.vertices.push_back(source.vertices[source_vertex]);
    }
    for (const auto source_index : source_indices)
        result.indices.push_back(remap.at(source_index));

    cluster.vertex_count = static_cast<std::uint32_t>(vertices.size());
    cluster.index_count = static_cast<std::uint32_t>(source_indices.size());
    cluster.triangle_count = cluster.index_count / 3u;
    calculate_bounds(cluster, result.vertices,
                     std::span<const std::uint32_t>(result.indices).subspan(cluster.first_index, cluster.index_count));
    result.clusters.push_back(cluster);
    return static_cast<std::uint32_t>(result.clusters.size() - 1u);
}

std::vector<hierarchy_work_node> build_clusters(virtual_mesh_data& result, const mesh_data& source,
                                                std::span<const std::uint32_t> indices,
                                                const virtual_mesh_build_options& options, float error,
                                                std::uint16_t level, bool create_nodes = true)
{
    std::vector<hierarchy_work_node> output;
    if (indices.empty()) return output;
    const auto max_vertices = std::clamp(options.max_vertices_per_cluster, 3u, 255u);
    const auto max_triangles = std::clamp(options.max_triangles_per_cluster, 1u, 255u);
    const auto bound = meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles);
    std::vector<meshopt_Meshlet> meshlets(bound);
    std::vector<std::uint32_t> meshlet_vertices(bound * max_vertices);
    std::vector<std::uint8_t> meshlet_triangles(bound * max_triangles * 3u);
    const auto count =
        meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(),
                              indices.size(), &source.vertices[0].position[0], source.vertices.size(),
                              sizeof(mesh_vertex), max_vertices, max_triangles, 0.5f);
    meshlets.resize(count);
    for (const auto& meshlet : meshlets)
        meshopt_optimizeMeshletLevel(meshlet_vertices.data() + meshlet.vertex_offset, meshlet.vertex_count,
                                     meshlet_triangles.data() + meshlet.triangle_offset, meshlet.triangle_count, 3);
    output.reserve(count);
    for (const auto& meshlet : meshlets)
    {
        std::vector<std::uint32_t> cluster_indices;
        cluster_indices.reserve(meshlet.triangle_count * 3u);
        for (std::uint32_t triangle = 0; triangle < meshlet.triangle_count; ++triangle)
        {
            for (std::uint32_t corner = 0; corner < 3u; ++corner)
            {
                const auto local = meshlet_triangles[meshlet.triangle_offset + triangle * 3u + corner];
                cluster_indices.push_back(meshlet_vertices[meshlet.vertex_offset + local]);
            }
        }
        const auto cluster_index = append_cluster(result, source, cluster_indices, error, level);
        auto node_index = invalid_virtual_geometry_index;
        if (create_nodes)
        {
            virtual_mesh_lod_node node{};
            node.first_cluster = cluster_index;
            node.cluster_count = 1;
            node.error = error;
            node.bounds_min = result.clusters[cluster_index].bounds_min;
            node.bounds_max = result.clusters[cluster_index].bounds_max;
            node.sphere_center = result.clusters[cluster_index].sphere_center;
            node.sphere_radius = result.clusters[cluster_index].sphere_radius;
            node.cone_axis = result.clusters[cluster_index].cone_axis;
            node.cone_cutoff = result.clusters[cluster_index].cone_cutoff;
            node.level = level;
            result.lod_nodes.push_back(node);
            node_index = static_cast<std::uint32_t>(result.lod_nodes.size() - 1u);
            result.clusters[cluster_index].hierarchy_node = node_index;
        }
        output.push_back({.node_index = node_index,
                          .source_indices = std::move(cluster_indices),
                          .unique_vertices = {},
                          .center = result.clusters[cluster_index].sphere_center,
                          .error = error});
        output.back().unique_vertices = unique_vertices(output.back().source_indices);
    }
    return output;
}

float source_extent(const mesh_data& source) noexcept
{
    if (source.vertices.empty()) return 0.0f;
    auto bounds_min = vertex_position(source.vertices.front());
    auto bounds_max = bounds_min;
    for (const auto& vertex : source.vertices)
    {
        bounds_min = minimum(bounds_min, vertex_position(vertex));
        bounds_max = maximum(bounds_max, vertex_position(vertex));
    }
    return math::length(math::sub(bounds_max, bounds_min));
}

std::vector<std::uint32_t> simplify(const mesh_data& source, std::span<const std::uint32_t> indices,
                                    std::size_t target_index_count, float& absolute_error, bool lock_boundaries)
{
    std::vector<std::uint32_t> result(indices.size());
    float relative_error{};
    const auto count = meshopt_simplify(
        result.data(), indices.data(), indices.size(), &source.vertices[0].position[0], source.vertices.size(),
        sizeof(mesh_vertex), std::max<std::size_t>(3u, target_index_count / 3u * 3u), 1.0f,
        lock_boundaries ? static_cast<unsigned int>(meshopt_SimplifyLockBorder) : 0u, &relative_error);
    result.resize(count >= 3 ? count : indices.size());
    if (count < 3) std::copy(indices.begin(), indices.end(), result.begin());
    absolute_error = relative_error * source_extent(source);
    return result;
}

std::vector<std::vector<std::size_t>> make_groups(const std::vector<hierarchy_work_node>& nodes,
                                                  const virtual_mesh_build_options& options)
{
    std::vector<std::vector<std::size_t>> groups;
    std::vector<bool> used(nodes.size());
    const auto maximum_size = std::max(options.minimum_group_size, options.maximum_group_size);
    for (std::size_t seed = 0; seed < nodes.size(); ++seed)
    {
        if (used[seed]) continue;
        std::vector<std::size_t> candidates;
        candidates.reserve(nodes.size());
        for (std::size_t candidate = 0; candidate < nodes.size(); ++candidate)
            if (!used[candidate] && candidate != seed) candidates.push_back(candidate);
        std::stable_sort(candidates.begin(), candidates.end(),
                         [&](std::size_t lhs, std::size_t rhs)
                         {
                             const auto left_shared = shared_vertex_count(nodes[seed], nodes[lhs]);
                             const auto right_shared = shared_vertex_count(nodes[seed], nodes[rhs]);
                             if (left_shared != right_shared) return left_shared > right_shared;
                             const auto left_distance = squared_distance(nodes[seed].center, nodes[lhs].center);
                             const auto right_distance = squared_distance(nodes[seed].center, nodes[rhs].center);
                             return left_distance != right_distance ? left_distance < right_distance : lhs < rhs;
                         });
        auto& group = groups.emplace_back();
        group.push_back(seed);
        used[seed] = true;
        for (const auto candidate : candidates)
        {
            if (group.size() >= maximum_size) break;
            group.push_back(candidate);
            used[candidate] = true;
        }
    }
    return groups;
}

void build_hierarchy(virtual_mesh_data& result, const mesh_data& source, std::vector<hierarchy_work_node> current,
                     const virtual_mesh_build_options& options)
{
    std::uint16_t level = 1;
    while (current.size() > std::max(1u, options.maximum_root_clusters))
    {
        std::vector<hierarchy_work_node> parents;
        for (const auto& group : make_groups(current, options))
        {
            std::vector<std::uint32_t> combined;
            float child_error{};
            for (const auto child : group)
            {
                combined.insert(combined.end(), current[child].source_indices.begin(),
                                current[child].source_indices.end());
                child_error = std::max(child_error, current[child].error);
            }
            float simplify_error{};
            auto simplified = simplify(source, combined,
                                       static_cast<std::size_t>(static_cast<float>(combined.size()) *
                                                                std::clamp(options.parent_triangle_ratio, 0.1f, 0.9f)),
                                       simplify_error, true);
            const auto parent_error = std::max(child_error, simplify_error);
            const auto first_cluster = static_cast<std::uint32_t>(result.clusters.size());
            auto parent_clusters = build_clusters(result, source, simplified, options, parent_error, level, false);
            if (parent_clusters.empty()) continue;

            virtual_mesh_lod_node parent{};
            parent.first_cluster = first_cluster;
            parent.cluster_count = static_cast<std::uint32_t>(result.clusters.size()) - first_cluster;
            parent.first_child = static_cast<std::uint32_t>(result.hierarchy_children.size());
            parent.child_count = static_cast<std::uint32_t>(group.size());
            parent.error = parent_error;
            parent.level = level;
            bool initialized{};
            math::vector3f cone_sum{};
            for (const auto child : group)
            {
                const auto child_node = current[child].node_index;
                result.hierarchy_children.push_back(child_node);
                const auto& node = result.lod_nodes[child_node];
                if (!initialized)
                {
                    parent.bounds_min = node.bounds_min;
                    parent.bounds_max = node.bounds_max;
                    initialized = true;
                }
                else
                {
                    parent.bounds_min = minimum(parent.bounds_min, node.bounds_min);
                    parent.bounds_max = maximum(parent.bounds_max, node.bounds_max);
                }
                cone_sum = math::add(cone_sum, node.cone_axis);
            }
            parent.sphere_center = math::mul(math::add(parent.bounds_min, parent.bounds_max), 0.5f);
            for (const auto child : group)
            {
                const auto& node = result.lod_nodes[current[child].node_index];
                parent.sphere_radius =
                    std::max(parent.sphere_radius,
                             math::length(math::sub(node.sphere_center, parent.sphere_center)) + node.sphere_radius);
            }
            if (math::length_squared(cone_sum) > 1.0e-12f) parent.cone_axis = math::normalize(cone_sum);
            parent.cone_cutoff = -1.0f;

            result.lod_nodes.push_back(parent);
            const auto parent_index = static_cast<std::uint32_t>(result.lod_nodes.size() - 1u);
            for (const auto child : group)
                result.lod_nodes[current[child].node_index].parent = parent_index;
            for (std::uint32_t cluster = first_cluster; cluster < result.clusters.size(); ++cluster)
                result.clusters[cluster].hierarchy_node = parent_index;

            parents.push_back({.node_index = parent_index,
                               .source_indices = std::move(simplified),
                               .unique_vertices = unique_vertices(combined),
                               .center = parent.sphere_center,
                               .error = parent_error});
        }
        if (parents.empty() || parents.size() >= current.size()) break;
        current = std::move(parents);
        ++level;
    }
    result.root_nodes.reserve(current.size());
    for (const auto& root : current)
        result.root_nodes.push_back(root.node_index);
    result.stats.hierarchy_level_count = current.empty() ? 0u : static_cast<std::uint32_t>(level + 1u);
}

std::int16_t snorm16(float value) noexcept
{
    return static_cast<std::int16_t>(std::round(std::clamp(value, -1.0f, 1.0f) * 32767.0f));
}

std::array<std::int16_t, 2> encode_octahedral(math::vector3f value) noexcept
{
    value = math::normalize(value);
    const float denominator = std::abs(value[0]) + std::abs(value[1]) + std::abs(value[2]);
    if (denominator <= 1.0e-8f) return {};
    float x = value[0] / denominator;
    float y = value[1] / denominator;
    if (value[2] < 0.0f)
    {
        const float old_x = x;
        x = (1.0f - std::abs(y)) * (old_x >= 0.0f ? 1.0f : -1.0f);
        y = (1.0f - std::abs(old_x)) * (y >= 0.0f ? 1.0f : -1.0f);
    }
    return {snorm16(x), snorm16(y)};
}

std::uint16_t float_to_half(float value) noexcept
{
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16u) & 0x8000u);
    const int exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;
    const auto mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return sign;
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(exponent) << 10u) | (mantissa >> 13u));
}

packed_virtual_vertex pack_vertex(const mesh_vertex& source, const virtual_mesh_cluster& cluster) noexcept
{
    packed_virtual_vertex result{};
    for (std::size_t component = 0; component < 3; ++component)
    {
        const auto extent = cluster.bounds_max[component] - cluster.bounds_min[component];
        const auto normalized =
            extent > 1.0e-8f ? (source.position[component] - cluster.bounds_min[component]) / extent : 0.0f;
        result.position[component] =
            static_cast<std::uint16_t>(std::round(std::clamp(normalized, 0.0f, 1.0f) * 65535.0f));
    }
    const auto normal = encode_octahedral({source.normal[0], source.normal[1], source.normal[2]});
    const auto tangent = encode_octahedral({source.tangent[0], source.tangent[1], source.tangent[2]});
    result.normal[0] = normal[0];
    result.normal[1] = normal[1];
    result.tangent[0] = tangent[0];
    result.tangent[1] = tangent[1];
    result.tangent_sign = source.tangent[3] < 0.0f ? -1 : 1;
    result.texcoord[0] = float_to_half(source.texcoord[0]);
    result.texcoord[1] = float_to_half(source.texcoord[1]);
    for (std::size_t component = 0; component < 4; ++component)
        result.color[component] =
            static_cast<std::uint8_t>(std::round(std::clamp(source.color[component], 0.0f, 1.0f) * 255.0f));
    return result;
}

template <class T> void append_value(std::vector<std::byte>& target, const T& value)
{
    const auto bytes = std::as_bytes(std::span(&value, 1));
    target.insert(target.end(), bytes.begin(), bytes.end());
}

void assign_pages(virtual_mesh_data& result)
{
    std::unordered_set<std::uint32_t> root_clusters;
    for (const auto root : result.root_nodes)
    {
        if (root >= result.lod_nodes.size()) continue;
        const auto& node = result.lod_nodes[root];
        for (std::uint32_t offset = 0; offset < node.cluster_count; ++offset)
            root_clusters.insert(node.first_cluster + offset);
    }

    std::vector<std::vector<std::byte>> cluster_payloads(result.clusters.size());
    for (std::uint32_t cluster_index = 0; cluster_index < result.clusters.size(); ++cluster_index)
    {
        const auto& cluster = result.clusters[cluster_index];
        std::vector<packed_virtual_vertex> vertices;
        vertices.reserve(cluster.vertex_count);
        for (std::uint32_t vertex = 0; vertex < cluster.vertex_count; ++vertex)
            vertices.push_back(pack_vertex(result.vertices[cluster.first_vertex + vertex], cluster));

        const auto bound = meshopt_encodeVertexBufferBound(vertices.size(), sizeof(packed_virtual_vertex));
        std::vector<std::byte> encoded_vertices(bound);
        const auto encoded_size = meshopt_encodeVertexBuffer(reinterpret_cast<unsigned char*>(encoded_vertices.data()),
                                                             encoded_vertices.size(), vertices.data(), vertices.size(),
                                                             sizeof(packed_virtual_vertex));
        encoded_vertices.resize(encoded_size);
        std::vector<std::uint8_t> triangles;
        triangles.reserve(cluster.index_count);
        for (std::uint32_t index = 0; index < cluster.index_count; ++index)
        {
            const auto global = result.indices[cluster.first_index + index];
            triangles.push_back(static_cast<std::uint8_t>(global - cluster.first_vertex));
        }
        encoded_cluster_header header{.cluster_index = cluster_index,
                                      .vertex_count = static_cast<std::uint16_t>(cluster.vertex_count),
                                      .triangle_count = static_cast<std::uint16_t>(cluster.triangle_count),
                                      .encoded_vertex_bytes = static_cast<std::uint32_t>(encoded_vertices.size()),
                                      .triangle_bytes = static_cast<std::uint32_t>(triangles.size())};
        auto& payload = cluster_payloads[cluster_index];
        append_value(payload, header);
        payload.insert(payload.end(), encoded_vertices.begin(), encoded_vertices.end());
        payload.insert(payload.end(), reinterpret_cast<const std::byte*>(triangles.data()),
                       reinterpret_cast<const std::byte*>(triangles.data() + triangles.size()));
    }

    virtual_geometry_page page{};
    page.first_cluster = 0;
    std::vector<std::byte> page_bytes;
    std::uint32_t decoded_page_bytes{};
    auto flush = [&]()
    {
        if (page.cluster_count == 0) return;
        page.compressed_offset = static_cast<std::uint32_t>(result.page_payload.size());
        page.compressed_size = static_cast<std::uint32_t>(page_bytes.size());
        page.uncompressed_size = decoded_page_bytes;
        std::uint64_t hash = 1469598103934665603ull;
        for (const auto byte : page_bytes)
        {
            hash ^= std::to_integer<std::uint8_t>(byte);
            hash *= 1099511628211ull;
        }
        page.content_hash = hash;
        result.page_payload.insert(result.page_payload.end(), page_bytes.begin(), page_bytes.end());
        const auto page_index = static_cast<std::uint32_t>(result.pages.size());
        for (std::uint32_t cluster = page.first_cluster; cluster < page.first_cluster + page.cluster_count; ++cluster)
            result.clusters[cluster].page_index = page_index;
        result.pages.push_back(page);
        page = {};
        page.first_cluster = static_cast<std::uint32_t>(
            result.pages.empty() ? 0 : result.pages.back().first_cluster + result.pages.back().cluster_count);
        page_bytes.clear();
        decoded_page_bytes = 0;
    };

    for (std::uint32_t cluster = 0; cluster < cluster_payloads.size(); ++cluster)
    {
        const auto& payload = cluster_payloads[cluster];
        if (!page_bytes.empty() && page_bytes.size() + payload.size() > virtual_geometry_page_bytes) flush();
        if (page.cluster_count == 0) page.first_cluster = cluster;
        result.clusters[cluster].page_byte_offset = decoded_page_bytes;
        decoded_page_bytes += virtual_geometry_decoded_cluster_header_bytes +
                              result.clusters[cluster].vertex_count * virtual_geometry_decoded_vertex_bytes +
                              result.clusters[cluster].index_count;
        page.root = page.root || root_clusters.contains(cluster);
        page_bytes.insert(page_bytes.end(), payload.begin(), payload.end());
        ++page.cluster_count;
    }
    flush();
    for (auto& node : result.lod_nodes)
        if (node.cluster_count != 0 && node.first_cluster < result.clusters.size())
            node.page_index = result.clusters[node.first_cluster].page_index;
    result.stats.page_count = static_cast<std::uint32_t>(result.pages.size());
    result.stats.root_page_count = static_cast<std::uint32_t>(
        std::count_if(result.pages.begin(), result.pages.end(), [](const auto& candidate) { return candidate.root; }));
    result.stats.compressed_page_bytes = result.page_payload.size();
    for (const auto& cluster : result.clusters)
        result.stats.uncompressed_page_bytes +=
            static_cast<std::uint64_t>(cluster.vertex_count) * sizeof(packed_virtual_vertex) + cluster.index_count;
}

void build_conventional_lods(virtual_mesh_data& result, const mesh_data& source,
                             std::span<const std::uint32_t> valid_indices)
{
    constexpr std::array<float, 4> ratios{1.0f, 0.5f, 0.25f, 0.125f};
    result.conventional_lods.reserve(ratios.size());
    for (const auto ratio : ratios)
    {
        conventional_mesh_lod lod;
        lod.ratio = ratio;
        lod.vertices = source.vertices;
        if (ratio == 1.0f)
            lod.indices.assign(valid_indices.begin(), valid_indices.end());
        else
        {
            const auto target = std::max<std::size_t>(3u, static_cast<std::size_t>(valid_indices.size() * ratio));
            lod.indices = simplify(source, valid_indices, target, lod.geometric_error, false);
        }
        std::vector<std::uint32_t> cache_optimized(lod.indices.size());
        meshopt_optimizeVertexCache(cache_optimized.data(), lod.indices.data(), lod.indices.size(),
                                    lod.vertices.size());
        std::vector<std::uint32_t> overdraw_optimized(lod.indices.size());
        meshopt_optimizeOverdraw(overdraw_optimized.data(), cache_optimized.data(), cache_optimized.size(),
                                 &lod.vertices[0].position[0], lod.vertices.size(), sizeof(mesh_vertex), 1.05f);
        lod.indices = std::move(overdraw_optimized);
        std::vector<mesh_vertex> fetch_optimized(lod.vertices.size());
        const auto used_vertex_count =
            meshopt_optimizeVertexFetch(fetch_optimized.data(), lod.indices.data(), lod.indices.size(),
                                        lod.vertices.data(), lod.vertices.size(), sizeof(mesh_vertex));
        fetch_optimized.resize(used_vertex_count);
        lod.vertices = std::move(fetch_optimized);
        result.conventional_lods.push_back(std::move(lod));
    }
}

} // namespace

virtual_mesh_data build_virtual_mesh(const mesh_data& source, const virtual_mesh_build_options& options)
{
    virtual_mesh_data result;
    result.stats.source_vertex_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(source.vertices.size(), std::numeric_limits<std::uint32_t>::max()));
    result.stats.source_triangle_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(source.indices.size() / 3u, std::numeric_limits<std::uint32_t>::max()));
    if (source.vertices.empty()) return result;

    auto valid_indices = sanitize_indices(source, result.stats);
    if (valid_indices.empty()) return result;

    std::unordered_map<std::uint64_t, std::uint32_t> edges;
    for (std::size_t index = 0; index < valid_indices.size(); index += 3)
    {
        ++edges[edge_key(valid_indices[index + 0], valid_indices[index + 1])];
        ++edges[edge_key(valid_indices[index + 1], valid_indices[index + 2])];
        ++edges[edge_key(valid_indices[index + 2], valid_indices[index + 0])];
    }
    result.stats.boundary_edge_count = static_cast<std::uint32_t>(
        std::count_if(edges.begin(), edges.end(), [](const auto& edge) { return edge.second == 1u; }));

    auto leaves = build_clusters(result, source, valid_indices, options, 0.0f, 0);
    build_hierarchy(result, source, std::move(leaves), options);
    assign_pages(result);
    if (options.build_conventional_lods) build_conventional_lods(result, source, valid_indices);

    result.stats.cluster_count = static_cast<std::uint32_t>(result.clusters.size());
    result.stats.material_group_count = result.clusters.empty() ? 0u : 1u;
    result.stats.average_triangles_per_cluster =
        result.stats.cluster_count == 0
            ? 0.0f
            : static_cast<float>(std::accumulate(result.clusters.begin(), result.clusters.end(), std::uint64_t{},
                                                 [](std::uint64_t total, const auto& cluster)
                                                 { return total + cluster.triangle_count; })) /
                  static_cast<float>(result.stats.cluster_count);
    return result;
}

bool decode_virtual_geometry_page(const virtual_mesh_data& mesh, std::uint32_t page_index,
                                  std::vector<std::byte>& output)
{
    if (page_index >= mesh.pages.size()) return false;
    const auto& page = mesh.pages[page_index];
    if (static_cast<std::uint64_t>(page.compressed_offset) + page.compressed_size > mesh.page_payload.size())
        return false;
    auto cursor = page.compressed_offset;
    const auto end = cursor + page.compressed_size;
    output.clear();
    for (std::uint32_t cluster = 0; cluster < page.cluster_count; ++cluster)
    {
        if (cursor + sizeof(encoded_cluster_header) > end) return false;
        encoded_cluster_header header{};
        std::memcpy(&header, mesh.page_payload.data() + cursor, sizeof(header));
        cursor += sizeof(header);
        if (cursor + header.encoded_vertex_bytes + header.triangle_bytes > end) return false;
        std::vector<packed_virtual_vertex> vertices(header.vertex_count);
        if (meshopt_decodeVertexBuffer(vertices.data(), vertices.size(), sizeof(packed_virtual_vertex),
                                       reinterpret_cast<const unsigned char*>(mesh.page_payload.data() + cursor),
                                       header.encoded_vertex_bytes) != 0)
            return false;
        cursor += header.encoded_vertex_bytes;
        append_value(output, header);
        const auto vertex_bytes = std::as_bytes(std::span(vertices));
        output.insert(output.end(), vertex_bytes.begin(), vertex_bytes.end());
        output.insert(output.end(), mesh.page_payload.begin() + cursor,
                      mesh.page_payload.begin() + cursor + header.triangle_bytes);
        cursor += header.triangle_bytes;
    }
    return cursor == end;
}

} // namespace arc::render
