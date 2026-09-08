#include <arc/scene/terrain_render_geometry.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

TEST_CASE("heightfield terrain compiles into generic virtual geometry with conventional fallback")
{
    const std::vector<float> heights(6u, 0.0f);
    const std::vector<std::array<std::uint8_t, 4>> weights{{255u, 0u, 0u, 0u},   {0u, 255u, 0u, 0u},
                                                           {0u, 0u, 255u, 0u},   {0u, 0u, 0u, 255u},
                                                           {64u, 64u, 64u, 63u}, {128u, 127u, 0u, 0u}};

    arc::scene::terrain_surface_ir surface;
    surface.source_revision = 7u;
    surface.local_bounds = {-2.0, 0.0, -1.0, 2.0, 0.0, 1.0};
    surface.geometry = arc::scene::terrain_surface_heightfield_ir{
        .sample_width = 3u,
        .sample_height = 2u,
        .width = 4.0f,
        .depth = 2.0f,
        .heights = heights,
        .material_weights = weights,
    };

    const auto artifact = arc::scene::build_terrain_render_geometry(surface);
    REQUIRE(artifact.has_value());
    CHECK(artifact->stats.source_vertex_count == 6u);
    CHECK(artifact->stats.source_triangle_count == 4u);
    CHECK(artifact->stats.invalid_triangle_count == 0u);
    CHECK(artifact->stats.boundary_edge_count == 6u);
    REQUIRE_FALSE(artifact->clusters.empty());
    REQUIRE_FALSE(artifact->root_nodes.empty());
    REQUIRE_FALSE(artifact->pages.empty());
    REQUIRE(artifact->conventional_lods.size() == 4u);
    REQUIRE_FALSE(artifact->conventional_lods.front().vertices.empty());
    REQUIRE_FALSE(artifact->conventional_lods.front().indices.empty());

    for (const auto& vertex : artifact->conventional_lods.front().vertices)
    {
        CHECK(vertex.normal[0] == Catch::Approx(0.0f).margin(0.0001f));
        CHECK(vertex.normal[1] == Catch::Approx(1.0f).margin(0.0001f));
        CHECK(vertex.normal[2] == Catch::Approx(0.0f).margin(0.0001f));
        CHECK(vertex.texcoord[0] >= 0.0f);
        CHECK(vertex.texcoord[0] <= 1.0f);
        CHECK(vertex.texcoord[1] >= 0.0f);
        CHECK(vertex.texcoord[1] <= 1.0f);
        for (const float channel : vertex.color)
            CHECK(channel == Catch::Approx(1.0f));
    }
}

TEST_CASE("terrain render geometry compilation is deterministic")
{
    const std::vector<float> heights{0.0f, 0.5f, 1.0f, -0.25f, 0.25f, 0.75f};
    const std::vector<std::array<std::uint8_t, 4>> weights(heights.size(),
                                                           std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});

    arc::scene::terrain_surface_ir surface;
    surface.source_revision = 19u;
    surface.local_bounds = {-3.0, -0.25, -2.0, 3.0, 1.0, 2.0};
    surface.geometry = arc::scene::terrain_surface_heightfield_ir{
        .sample_width = 3u,
        .sample_height = 2u,
        .width = 6.0f,
        .depth = 4.0f,
        .heights = heights,
        .material_weights = weights,
    };

    const auto first = arc::scene::build_terrain_render_geometry(surface);
    const auto second = arc::scene::build_terrain_render_geometry(surface);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    CHECK(first->stats.cluster_count == second->stats.cluster_count);
    CHECK(first->stats.page_count == second->stats.page_count);
    CHECK(first->stats.root_page_count == second->stats.root_page_count);
    REQUIRE(first->indices == second->indices);
    REQUIRE(first->page_payload == second->page_payload);
    REQUIRE(first->pages.size() == second->pages.size());
    for (std::size_t index = 0; index < first->pages.size(); ++index)
        CHECK(first->pages[index].content_hash == second->pages[index].content_hash);
}

TEST_CASE("mesh terrain uses the same generic render geometry compiler")
{
    const std::vector<arc::math::vector3f> positions{
        {-1.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 1.0f}};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u, 2u, 1u, 3u};

    arc::scene::terrain_surface_ir surface;
    surface.local_bounds = {-1.0, 0.0, -1.0, 1.0, 0.0, 1.0};
    surface.geometry = arc::scene::terrain_surface_mesh_ir{.positions = positions, .indices = indices};

    const auto artifact = arc::scene::build_terrain_render_geometry(surface);
    REQUIRE(artifact.has_value());
    CHECK(artifact->stats.source_vertex_count == 4u);
    CHECK(artifact->stats.source_triangle_count == 2u);
    REQUIRE_FALSE(artifact->clusters.empty());
    REQUIRE_FALSE(artifact->pages.empty());
    REQUIRE_FALSE(artifact->conventional_lods.empty());
}

TEST_CASE("terrain render geometry rejects surfaces without renderable triangles")
{
    const std::vector<arc::math::vector3f> positions{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u};

    arc::scene::terrain_surface_ir surface;
    surface.local_bounds = {0.0, -1.0, -1.0, 2.0, 1.0, 1.0};
    surface.geometry = arc::scene::terrain_surface_mesh_ir{.positions = positions, .indices = indices};

    REQUIRE(arc::scene::validate_terrain_surface_ir(surface));
    REQUIRE_FALSE(arc::scene::build_terrain_render_geometry(surface).has_value());
}
