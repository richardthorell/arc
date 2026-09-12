#include <arc/scene/terrain.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("M3.4 terrain raycast traverses the regular heightfield grid")
{
    using namespace arc;
    scene::terrain_component terrain;
    terrain.size = 1024.0f;
    terrain.subdivisions = 1024u;
    const auto resolution = static_cast<std::size_t>(terrain.subdivisions) + 1u;
    const auto count = resolution * resolution;
    terrain.heights.assign(count, 0.0f);
    terrain.layer_weights.assign(count, {255u, 0u, 0u, 0u});

    const auto vertical = scene::raycast_terrain(terrain, {0.0f, 100.0f, 0.0f}, {0.0f, -1.0f, 0.0f});
    REQUIRE(vertical.hit);
    CHECK(vertical.distance == Catch::Approx(100.0f));
    CHECK(vertical.position[1] == Catch::Approx(0.0f));

    const auto direction = math::normalize(math::vector3f{1.0f, -0.01f, 0.0f});
    const auto grazing = scene::raycast_terrain(terrain, {-600.0f, 10.0f, 0.0f}, direction);
    REQUIRE(grazing.hit);
    CHECK(grazing.position[0] == Catch::Approx(400.0f).margin(0.1f));
    CHECK(grazing.position[1] == Catch::Approx(0.0f).margin(0.01f));
}
