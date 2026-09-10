#include <arc/scene/scene.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace
{

arc::scene::terrain_surface_ir make_region_test_surface(std::uint64_t revision = 1u)
{
    constexpr std::uint32_t sample_width = 9u;
    constexpr std::uint32_t sample_height = 9u;
    static std::vector<float> heights(sample_width * sample_height);
    static std::vector<std::array<std::uint8_t, 4>> weights(sample_width * sample_height,
                                                            std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    for (std::uint32_t z = 0; z < sample_height; ++z)
        for (std::uint32_t x = 0; x < sample_width; ++x)
            heights[static_cast<std::size_t>(z) * sample_width + x] = static_cast<float>(x + z) * 0.25f;

    arc::scene::terrain_surface_ir surface;
    surface.source_revision = revision;
    surface.local_bounds = {-256.0, 0.0, -256.0, 256.0, 4.0, 256.0};
    surface.geometry = arc::scene::terrain_surface_heightfield_ir{
        .sample_width = sample_width,
        .sample_height = sample_height,
        .width = 512.0f,
        .depth = 512.0f,
        .heights = heights,
        .material_weights = weights,
    };
    return surface;
}

arc::scene::terrain_component make_region_test_terrain()
{
    constexpr std::uint32_t sample_width = 9u;
    constexpr std::uint32_t sample_height = 9u;
    arc::scene::terrain_component terrain;
    terrain.size = 512.0f;
    terrain.subdivisions = 8u;
    terrain.content_revision = 1u;
    terrain.heights.resize(sample_width * sample_height);
    terrain.layer_weights.assign(sample_width * sample_height, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    for (std::uint32_t z = 0; z < sample_height; ++z)
        for (std::uint32_t x = 0; x < sample_width; ++x)
            terrain.heights[static_cast<std::size_t>(z) * sample_width + x] = static_cast<float>(x + z) * 0.25f;
    return terrain;
}

} // namespace

TEST_CASE("heightfield render partition is sample aligned and seam compatible")
{
    const auto surface = make_region_test_surface();
    const auto regions = arc::scene::build_terrain_render_regions(surface, 256.0);
    REQUIRE(regions.size() == 4u);
    CHECK(regions[0].id == arc::scene::terrain_region_id{0, 0});
    CHECK(regions[1].id == arc::scene::terrain_region_id{1, 0});
    CHECK(regions[2].id == arc::scene::terrain_region_id{0, 1});
    CHECK(regions[3].id == arc::scene::terrain_region_id{1, 1});

    const auto left = regions[0].surface.view();
    const auto right = regions[1].surface.view();
    const auto left_geometry = arc::scene::canonicalize_terrain_surface_geometry(left);
    const auto right_geometry = arc::scene::canonicalize_terrain_surface_geometry(right);
    REQUIRE(left_geometry.has_value());
    REQUIRE(right_geometry.has_value());

    const auto left_width = std::get<arc::scene::terrain_surface_heightfield_ir>(left.geometry).sample_width;
    const auto right_width = std::get<arc::scene::terrain_surface_heightfield_ir>(right.geometry).sample_width;
    REQUIRE(left_width == 5u);
    REQUIRE(right_width == 5u);
    for (std::uint32_t z = 0; z < 5u; ++z)
    {
        const auto& left_edge = left_geometry->positions[static_cast<std::size_t>(z) * left_width + 4u];
        const auto& right_edge = right_geometry->positions[static_cast<std::size_t>(z) * right_width];
        CHECK(left_edge[0] == Catch::Approx(right_edge[0]));
        CHECK(left_edge[1] == Catch::Approx(right_edge[1]));
        CHECK(left_edge[2] == Catch::Approx(right_edge[2]));

        const auto& left_normal = regions[0].vertex_normals[static_cast<std::size_t>(z) * left_width + 4u];
        const auto& right_normal = regions[1].vertex_normals[static_cast<std::size_t>(z) * right_width];
        CHECK(left_normal[0] == Catch::Approx(right_normal[0]));
        CHECK(left_normal[1] == Catch::Approx(right_normal[1]));
        CHECK(left_normal[2] == Catch::Approx(right_normal[2]));
    }
}

TEST_CASE("terrain proxy replaces only the region whose geometry changed")
{
    arc::render::renderer renderer;
    arc::scene::terrain_render_proxy_cache cache;
    auto terrain = make_region_test_terrain();
    const auto guid = arc::ecs::generate_entity_guid();

    REQUIRE(cache.synchronize(guid, terrain, renderer));
    const auto* initial = cache.find(guid);
    REQUIRE(initial != nullptr);
    REQUIRE(initial->regions.size() == 4u);
    const auto region0 = initial->regions[0].geometry;
    const auto region1 = initial->regions[1].geometry;
    const auto region2 = initial->regions[2].geometry;
    const auto region3 = initial->regions[3].geometry;

    constexpr std::size_t interior_sample = 10u; // (1,1), outside neighboring regions' normal dependency halo.
    terrain.heights[interior_sample] += 3.0f;
    ++terrain.content_revision;
    const arc::scene::terrain_dirty_region dirty{
        .min_x = 1u, .min_z = 1u, .max_x = 1u, .max_z = 1u, .valid = true, .heights_changed = true};
    REQUIRE(cache.synchronize(guid, terrain, renderer, &dirty));

    const auto* rebuilt = cache.find(guid);
    REQUIRE(rebuilt != nullptr);
    REQUIRE(rebuilt->regions.size() == 4u);
    CHECK(rebuilt->regions[0].geometry != region0);
    CHECK(rebuilt->regions[1].geometry == region1);
    CHECK(rebuilt->regions[2].geometry == region2);
    CHECK(rebuilt->regions[3].geometry == region3);
    CHECK_FALSE(renderer.mesh_alive(region0.conventional));
    CHECK(renderer.mesh_alive(region1.conventional));
    CHECK(renderer.mesh_alive(region2.conventional));
    CHECK(renderer.mesh_alive(region3.conventional));
}

TEST_CASE("terrain paint changes attributes without replacing region geometry")
{
    arc::render::renderer renderer;
    arc::scene::terrain_render_proxy_cache cache;
    auto terrain = make_region_test_terrain();
    const auto guid = arc::ecs::generate_entity_guid();

    REQUIRE(cache.synchronize(guid, terrain, renderer));
    const auto* initial = cache.find(guid);
    REQUIRE(initial != nullptr);
    REQUIRE(initial->regions.size() == 4u);
    const auto geometry0 = initial->regions[0].geometry;
    const auto attribute_fingerprint = initial->regions[0].attribute_fingerprint;

    constexpr std::size_t interior_sample = 10u;
    terrain.layer_weights[interior_sample] = {0u, 255u, 0u, 0u};
    ++terrain.content_revision;
    const arc::scene::terrain_dirty_region dirty{
        .min_x = 1u, .min_z = 1u, .max_x = 1u, .max_z = 1u, .valid = true, .weights_changed = true};
    REQUIRE(cache.synchronize(guid, terrain, renderer, &dirty));

    const auto* painted = cache.find(guid);
    REQUIRE(painted != nullptr);
    CHECK(painted->regions[0].geometry == geometry0);
    CHECK(painted->regions[0].attribute_fingerprint != attribute_fingerprint);
    CHECK(renderer.mesh_alive(geometry0.conventional));
}
