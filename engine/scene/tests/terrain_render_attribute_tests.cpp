#include <arc/scene/terrain_render_attributes.h>
#include <arc/scene/terrain_render_geometry.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

TEST_CASE("heightfield terrain compiles layer weights as a separate render attribute artifact")
{
    const std::vector<float> heights{0.0f, 0.25f, 0.5f, 0.75f};
    const std::vector<std::array<std::uint8_t, 4>> weights{
        {255u, 0u, 0u, 0u}, {128u, 127u, 0u, 0u}, {0u, 255u, 0u, 0u}, {0u, 0u, 255u, 0u}};

    arc::scene::terrain_surface_ir surface;
    surface.source_revision = 7u;
    surface.local_bounds = {.min_x = -1.0, .min_y = 0.0, .min_z = -1.0, .max_x = 1.0, .max_y = 0.75, .max_z = 1.0};
    surface.geometry = arc::scene::terrain_surface_heightfield_ir{.sample_width = 2u,
                                                                  .sample_height = 2u,
                                                                  .width = 2.0f,
                                                                  .depth = 2.0f,
                                                                  .heights = heights,
                                                                  .material_weights = weights};

    const auto attributes = arc::scene::build_terrain_render_attributes(surface);
    REQUIRE(attributes.has_value());
    CHECK(attributes->width == 2u);
    CHECK(attributes->height == 2u);
    CHECK_FALSE(attributes->default_layer_only);
    REQUIRE(attributes->material_weights.size() == weights.size());
    CHECK(attributes->material_weights == weights);

    const auto geometry = arc::scene::build_terrain_render_geometry(surface);
    REQUIRE(geometry.has_value());
    REQUIRE_FALSE(geometry->conventional_lods.empty());
    for (const auto& vertex : geometry->conventional_lods.front().vertices)
    {
        CHECK(vertex.color[0] == 1.0f);
        CHECK(vertex.color[1] == 1.0f);
        CHECK(vertex.color[2] == 1.0f);
        CHECK(vertex.color[3] == 1.0f);
    }
}

TEST_CASE("mesh terrain uses deterministic layer-zero fallback attributes")
{
    const std::vector<arc::math::vector3f> positions{{-1.0f, 0.0f, -1.0f}, {1.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u};

    arc::scene::terrain_surface_ir surface;
    surface.source_revision = 11u;
    surface.local_bounds = {.min_x = -1.0, .min_y = 0.0, .min_z = -1.0, .max_x = 1.0, .max_y = 0.0, .max_z = 1.0};
    surface.geometry = arc::scene::terrain_surface_mesh_ir{.positions = positions, .indices = indices};

    const auto attributes = arc::scene::build_terrain_render_attributes(surface);
    REQUIRE(attributes.has_value());
    CHECK(attributes->width == 1u);
    CHECK(attributes->height == 1u);
    CHECK(attributes->default_layer_only);
    REQUIRE(attributes->material_weights.size() == 1u);
    CHECK(attributes->material_weights.front() == std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
}
