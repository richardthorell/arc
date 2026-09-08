#include <arc/scene/terrain.h>
#include <arc/scene/terrain_surface_ir.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <variant>
#include <vector>

TEST_CASE("legacy heightfields adapt to backend independent terrain surface IR")
{
    arc::scene::terrain_component terrain;
    terrain.size = 20.0f;
    terrain.subdivisions = 2u;
    terrain.content_revision = 17u;
    terrain.heights = {-2.0f, -1.0f, 0.0f, -1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 4.0f};
    terrain.layer_weights.assign(9u, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});

    const auto surface = arc::scene::make_legacy_terrain_surface_ir(terrain);
    REQUIRE(surface.has_value());
    REQUIRE(arc::scene::validate_terrain_surface_ir(*surface));
    REQUIRE(surface->source_revision == 17u);
    CHECK(surface->local_bounds.min_x == Catch::Approx(-10.0));
    CHECK(surface->local_bounds.max_x == Catch::Approx(10.0));
    CHECK(surface->local_bounds.min_y == Catch::Approx(-2.0));
    CHECK(surface->local_bounds.max_y == Catch::Approx(4.0));

    const auto* heightfield = std::get_if<arc::scene::terrain_surface_heightfield_ir>(&surface->geometry);
    REQUIRE(heightfield != nullptr);
    REQUIRE(heightfield->sample_width == 3u);
    REQUIRE(heightfield->sample_height == 3u);
    REQUIRE(heightfield->heights.data() == terrain.heights.data());
    REQUIRE(heightfield->material_weights.data() == terrain.layer_weights.data());
}

TEST_CASE("heightfield terrain surfaces canonicalize to deterministic triangle geometry")
{
    const std::vector<float> heights{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const std::vector<std::array<std::uint8_t, 4>> weights(heights.size(),
                                                           std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});

    arc::scene::terrain_surface_ir surface;
    surface.source_revision = 9u;
    surface.local_bounds = {-2.0, 0.0, -1.0, 2.0, 5.0, 1.0};
    surface.geometry = arc::scene::terrain_surface_heightfield_ir{
        .sample_width = 3u,
        .sample_height = 2u,
        .width = 4.0f,
        .depth = 2.0f,
        .heights = heights,
        .material_weights = weights,
    };

    const auto geometry = arc::scene::canonicalize_terrain_surface_geometry(surface);
    REQUIRE(geometry.has_value());
    REQUIRE(geometry->positions.size() == 6u);
    REQUIRE(geometry->indices == std::vector<std::uint32_t>{0u, 3u, 1u, 1u, 3u, 4u, 1u, 4u, 2u, 2u, 4u, 5u});

    CHECK(geometry->positions[0][0] == Catch::Approx(-2.0f));
    CHECK(geometry->positions[0][1] == Catch::Approx(0.0f));
    CHECK(geometry->positions[0][2] == Catch::Approx(-1.0f));
    CHECK(geometry->positions[1][0] == Catch::Approx(0.0f));
    CHECK(geometry->positions[1][1] == Catch::Approx(1.0f));
    CHECK(geometry->positions[1][2] == Catch::Approx(-1.0f));
    CHECK(geometry->positions[5][0] == Catch::Approx(2.0f));
    CHECK(geometry->positions[5][1] == Catch::Approx(5.0f));
    CHECK(geometry->positions[5][2] == Catch::Approx(1.0f));

    const auto repeated = arc::scene::canonicalize_terrain_surface_geometry(surface);
    REQUIRE(repeated.has_value());
    REQUIRE(repeated->indices == geometry->indices);
    REQUIRE(repeated->positions.size() == geometry->positions.size());
    for (std::size_t index = 0; index < geometry->positions.size(); ++index)
        for (std::size_t component = 0; component < 3u; ++component)
            CHECK(repeated->positions[index][component] == Catch::Approx(geometry->positions[index][component]));
}

TEST_CASE("terrain surface IR validates mesh topology independently from the renderer")
{
    const std::vector<arc::math::vector3f> positions{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u};
    arc::scene::terrain_surface_ir surface;
    surface.local_bounds = {0.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    surface.geometry = arc::scene::terrain_surface_mesh_ir{.positions = positions, .indices = indices};
    REQUIRE(arc::scene::validate_terrain_surface_ir(surface));

    const auto geometry = arc::scene::canonicalize_terrain_surface_geometry(surface);
    REQUIRE(geometry.has_value());
    REQUIRE(geometry->indices == indices);
    REQUIRE(geometry->positions.size() == positions.size());
    for (std::size_t index = 0; index < positions.size(); ++index)
        for (std::size_t component = 0; component < 3u; ++component)
            CHECK(geometry->positions[index][component] == Catch::Approx(positions[index][component]));

    const std::vector<std::uint32_t> invalid_indices{0u, 1u, 3u};
    surface.geometry = arc::scene::terrain_surface_mesh_ir{.positions = positions, .indices = invalid_indices};
    REQUIRE_FALSE(arc::scene::validate_terrain_surface_ir(surface));
    REQUIRE_FALSE(arc::scene::canonicalize_terrain_surface_geometry(surface).has_value());
}

TEST_CASE("invalid legacy heightfields do not cross the terrain surface IR boundary")
{
    arc::scene::terrain_component terrain;
    terrain.subdivisions = 2u;
    terrain.heights.assign(8u, 0.0f);
    terrain.layer_weights.assign(9u, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    REQUIRE_FALSE(arc::scene::make_legacy_terrain_surface_ir(terrain).has_value());
}
