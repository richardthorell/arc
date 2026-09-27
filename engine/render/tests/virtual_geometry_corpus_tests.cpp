#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
