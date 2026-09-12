#include <arc/scene/terrain_evaluator.h>
#include <arc/scene/terrain_render_regions.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>

TEST_CASE("M3.4 adjacent evaluation halos share sculpt samples and boundary normals")
{
    using namespace arc::scene;
    terrain_asset asset;
    asset.source.id = generate_terrain_stable_id();
    asset.partition.authoring_region_size = 16.0;
    const auto layer = add_terrain_sculpt_layer(asset).id;
    const auto address = terrain_modifier_sample_at(asset.coordinates, asset.partition, 0.0, 8.0);
    REQUIRE(accumulate_terrain_sculpt_samples(
                asset, layer, std::array{terrain_sculpt_sample_edit{address.region, {address.x, address.z, 7.0f}}})
                .revision != 0u);
    const std::vector<float> heights(30u, 0.0f);
    const std::vector<std::array<std::uint8_t, 4>> weights(30u, {255u, 0u, 0u, 0u});
    const terrain_heightfield_source_view source{6u, 5u, 20.0f, 16.0f, heights, weights, asset.authoring_revision};
    auto evaluator = make_default_terrain_evaluator();
    const auto left =
        evaluator.evaluate(asset, {.region = {-1, 0},
                                   .heightfield_source = source,
                                   .source_bounds = terrain_world_bounds{-8.0, 0.0, -8.0, 12.0, 0.0, 8.0}});
    const auto right =
        evaluator.evaluate(asset, {.region = {0, 0},
                                   .heightfield_source = source,
                                   .source_bounds = terrain_world_bounds{-12.0, 0.0, -8.0, 8.0, 0.0, 8.0}});
    REQUIRE(left.succeeded);
    REQUIRE(right.succeeded);
    CHECK(std::get<terrain_evaluated_heightfield>(left.surface.geometry).heights[16u] == Catch::Approx(7.0f));
    CHECK(std::get<terrain_evaluated_heightfield>(right.surface.geometry).heights[13u] == Catch::Approx(7.0f));
    const auto left_regions = build_terrain_render_regions(left.surface.view(), std::numeric_limits<double>::max());
    const auto right_regions = build_terrain_render_regions(right.surface.view(), std::numeric_limits<double>::max());
    REQUIRE(left_regions.size() == 1u);
    REQUIRE(right_regions.size() == 1u);
    for (std::size_t z = 0; z < 5u; ++z)
        for (std::size_t axis = 0; axis < 3u; ++axis)
            CHECK(left_regions.front().vertex_normals[z * 6u + 4u][axis] ==
                  Catch::Approx(right_regions.front().vertex_normals[z * 6u + 1u][axis]));
}
