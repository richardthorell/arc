#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace
{
arc::render::mesh_data make_corpus_grid(std::uint32_t side)
{
    arc::render::mesh_data mesh;
    mesh.name = "virtual geometry CI corpus";
    mesh.material_index = 1u;
    mesh.vertices.resize(static_cast<std::size_t>(side) * side);
    for (std::uint32_t z = 0; z < side; ++z)
        for (std::uint32_t x = 0; x < side; ++x)
        {
            auto& vertex = mesh.vertices[static_cast<std::size_t>(z) * side + x];
            const auto u = static_cast<float>(x) / static_cast<float>(side - 1u);
            const auto v = static_cast<float>(z) / static_cast<float>(side - 1u);
            vertex.position[0] = u * 64.0f;
            vertex.position[1] = std::sin(u * 17.0f) * std::cos(v * 19.0f);
            vertex.position[2] = v * 64.0f;
            vertex.normal[1] = 1.0f;
            vertex.tangent[0] = 1.0f;
            vertex.tangent[3] = 1.0f;
            vertex.texcoord[0] = u;
            vertex.texcoord[1] = v;
            std::fill(std::begin(vertex.color), std::end(vertex.color), 1.0f);
        }
    for (std::uint32_t z = 0; z + 1u < side; ++z)
        for (std::uint32_t x = 0; x + 1u < side; ++x)
        {
            const auto i0 = z * side + x;
            const auto i1 = i0 + 1u;
            const auto i2 = i0 + side;
            const auto i3 = i2 + 1u;
            mesh.indices.insert(mesh.indices.end(), {i0, i2, i1, i1, i2, i3});
        }
    return mesh;
}

using position_key = std::array<std::uint32_t, 3>;
using edge_key = std::pair<position_key, position_key>;

position_key key(const arc::render::mesh_vertex& vertex)
{
    return {std::bit_cast<std::uint32_t>(vertex.position[0]), std::bit_cast<std::uint32_t>(vertex.position[1]),
            std::bit_cast<std::uint32_t>(vertex.position[2])};
}

edge_key edge(position_key lhs, position_key rhs)
{
    if (rhs < lhs) std::swap(lhs, rhs);
    return {lhs, rhs};
}

void add_node_edges(const arc::render::virtual_mesh_data& geometry, const arc::render::virtual_mesh_lod_node& node,
                    std::map<edge_key, std::uint32_t>& counts)
{
    for (std::uint32_t cluster_offset = 0; cluster_offset < node.cluster_count; ++cluster_offset)
    {
        const auto cluster_index = node.first_cluster + cluster_offset;
        REQUIRE(cluster_index < geometry.clusters.size());
        const auto& cluster = geometry.clusters[cluster_index];
        for (std::uint32_t index = 0; index < cluster.index_count; index += 3u)
        {
            REQUIRE(cluster.first_index + index + 2u < geometry.indices.size());
            const auto i0 = geometry.indices[cluster.first_index + index + 0u];
            const auto i1 = geometry.indices[cluster.first_index + index + 1u];
            const auto i2 = geometry.indices[cluster.first_index + index + 2u];
            REQUIRE(i0 < geometry.vertices.size());
            REQUIRE(i1 < geometry.vertices.size());
            REQUIRE(i2 < geometry.vertices.size());
            ++counts[edge(key(geometry.vertices[i0]), key(geometry.vertices[i1]))];
            ++counts[edge(key(geometry.vertices[i1]), key(geometry.vertices[i2]))];
            ++counts[edge(key(geometry.vertices[i2]), key(geometry.vertices[i0]))];
        }
    }
}

std::set<edge_key> boundary_edges(const std::map<edge_key, std::uint32_t>& counts)
{
    std::set<edge_key> result;
    for (const auto& [candidate, count] : counts)
        if (count == 1u) result.insert(candidate);
    return result;
}
} // namespace

TEST_CASE("virtual geometry CI corpus is deterministic and hierarchy error is monotonic")
{
    using namespace arc::render;
    const auto source = make_corpus_grid(65u);
    const auto first = build_virtual_mesh(source, {.build_conventional_lods = false});
    const auto second = build_virtual_mesh(source, {.build_conventional_lods = false});

    REQUIRE(first.stats.source_triangle_count == 8192u);
    REQUIRE(first.page_payload == second.page_payload);
    REQUIRE(first.hierarchy_children == second.hierarchy_children);
    REQUIRE(first.root_nodes == second.root_nodes);
    REQUIRE(first.pages.size() == second.pages.size());
    REQUIRE(first.stats.root_page_count > 0u);
    REQUIRE(std::all_of(first.clusters.begin(), first.clusters.end(),
                        [](const auto& cluster)
                        {
                            return cluster.vertex_count <= virtual_geometry_max_vertices_per_cluster &&
                                   cluster.triangle_count <= virtual_geometry_max_triangles_per_cluster;
                        }));
    for (std::uint32_t node_index = 0; node_index < first.lod_nodes.size(); ++node_index)
    {
        const auto& node = first.lod_nodes[node_index];
        for (std::uint32_t child_offset = 0; child_offset < node.child_count; ++child_offset)
        {
            REQUIRE(node.first_child + child_offset < first.hierarchy_children.size());
            const auto child = first.hierarchy_children[node.first_child + child_offset];
            REQUIRE(child < first.lod_nodes.size());
            REQUIRE(first.lod_nodes[child].parent == node_index);
            REQUIRE(node.error >= first.lod_nodes[child].error);
        }
    }

    const std::array artifact_sources{virtual_geometry_artifact_source{
        .name = source.name, .material_index = source.material_index, .geometry = &first}};
    const auto artifact = encode_virtual_geometry_artifact(artifact_sources);
    REQUIRE(artifact);
    const auto index = inspect_virtual_geometry_artifact(artifact.value());
    REQUIRE(index);
    REQUIRE(index.value().meshes.size() == 1u);
    REQUIRE(index.value().meshes.front().pages.size() == first.pages.size());
}

TEST_CASE("adjacency hierarchy preserves locked group boundaries and retains legacy comparison")
{
    using namespace arc::render;
    const auto source = make_corpus_grid(33u);
    const auto geometry = build_virtual_mesh(source, {.minimum_group_size = 3u,
                                                      .maximum_group_size = 6u,
                                                      .maximum_root_clusters = 2u,
                                                      .build_conventional_lods = false});
    const auto legacy = build_virtual_mesh(source, {.minimum_group_size = 3u,
                                                    .maximum_group_size = 6u,
                                                    .maximum_root_clusters = 2u,
                                                    .build_conventional_lods = false,
                                                    .hierarchy_builder = virtual_mesh_hierarchy_builder::legacy_seed});

    REQUIRE_FALSE(geometry.root_nodes.empty());
    REQUIRE_FALSE(legacy.root_nodes.empty());
    REQUIRE(geometry.stats.source_triangle_count == legacy.stats.source_triangle_count);
    REQUIRE(geometry.stats.boundary_edge_count == legacy.stats.boundary_edge_count);
    REQUIRE(std::all_of(geometry.clusters.begin(), geometry.clusters.end(),
                        [&](const auto& cluster) { return cluster.material_index == source.material_index; }));

    for (std::uint32_t node_index = 0; node_index < geometry.lod_nodes.size(); ++node_index)
    {
        const auto& parent = geometry.lod_nodes[node_index];
        if (parent.child_count == 0u) continue;
        REQUIRE(parent.child_count <= 6u);
        std::map<edge_key, std::uint32_t> child_edge_counts;
        for (std::uint32_t child_offset = 0; child_offset < parent.child_count; ++child_offset)
        {
            const auto child_index = geometry.hierarchy_children[parent.first_child + child_offset];
            REQUIRE(child_index < geometry.lod_nodes.size());
            add_node_edges(geometry, geometry.lod_nodes[child_index], child_edge_counts);
        }
        std::map<edge_key, std::uint32_t> parent_edge_counts;
        add_node_edges(geometry, parent, parent_edge_counts);
        const auto child_boundary = boundary_edges(child_edge_counts);
        const auto parent_boundary = boundary_edges(parent_edge_counts);
        for (const auto& required_edge : child_boundary)
            REQUIRE(parent_boundary.contains(required_edge));
    }
}
