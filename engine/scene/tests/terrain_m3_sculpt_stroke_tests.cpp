#include <arc/scene/terrain.h>
#include <arc/scene/terrain_asset.h>
#include <arc/scene/terrain_evaluator.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

namespace
{

arc::scene::terrain_asset make_asset()
{
    arc::scene::terrain_asset asset;
    asset.source.id = arc::scene::generate_terrain_stable_id();
    asset.source.kind = arc::scene::terrain_source_kind::flat;
    asset.coordinates = {.origin_x = 0.0, .origin_y = 0.0, .origin_z = 0.0, .meters_per_unit = 1.0};
    asset.partition = {.authoring_region_size = 100.0, .dependency_halo = 4.0};
    return asset;
}

} // namespace

TEST_CASE("M3.3 sculpt edits fold into stable sparse region payloads with one revision")
{
    using namespace arc::scene;
    auto asset = make_asset();
    const auto layer = add_terrain_sculpt_layer(asset, "Detail").id;
    const auto start_revision = asset.authoring_revision;
    const auto left = terrain_modifier_sample_at(asset.coordinates, asset.partition, -25.0, 10.0);
    const auto right = terrain_modifier_sample_at(asset.coordinates, asset.partition, 125.0, 10.0);

    const std::array edits{
        terrain_sculpt_sample_edit{left.region, {left.x, left.z, 1.25f}},
        terrain_sculpt_sample_edit{left.region, {left.x, left.z, -0.25f}},
        terrain_sculpt_sample_edit{right.region, {right.x, right.z, 2.0f}},
    };
    const auto update = accumulate_terrain_sculpt_samples(asset, layer, edits);

    CHECK(update.revision == start_revision + 1u);
    REQUIRE(update.regions.size() == 2u);
    const auto* modifier = find_terrain_modifier(asset, layer);
    REQUIRE(modifier != nullptr);
    REQUIRE(modifier->region_payloads.size() == 2u);
    const auto* left_payload = find_terrain_modifier_payload(*modifier, left.region);
    REQUIRE(left_payload != nullptr);
    const auto& left_samples = std::get<terrain_sculpt_region_payload>(left_payload->data).samples;
    REQUIRE(left_samples.size() == 1u);
    CHECK(left_samples.front().delta == 1.0f);
    CHECK(asset.regions.size() == 2u);
    CHECK(validate_terrain_asset(asset).valid());
}

TEST_CASE("M3.3 sculpt edits remove sparse samples when accumulated delta returns to zero")
{
    using namespace arc::scene;
    auto asset = make_asset();
    const auto layer = add_terrain_sculpt_layer(asset).id;
    const auto address = terrain_modifier_sample_at(asset.coordinates, asset.partition, 10.0, 20.0);
    REQUIRE(accumulate_terrain_sculpt_samples(
                asset, layer, std::array{terrain_sculpt_sample_edit{address.region, {address.x, address.z, 2.0f}}})
                .revision != 0u);
    REQUIRE(accumulate_terrain_sculpt_samples(
                asset, layer, std::array{terrain_sculpt_sample_edit{address.region, {address.x, address.z, -2.0f}}})
                .revision != 0u);

    const auto* modifier = find_terrain_modifier(asset, layer);
    REQUIRE(modifier != nullptr);
    CHECK(find_terrain_modifier_payload(*modifier, address.region) == nullptr);
    CHECK_FALSE(modifier->affected_bounds.has_value());
}

TEST_CASE("M3.3 compatibility brush reports exact per-sample geometry deltas")
{
    using namespace arc::scene;
    terrain_component terrain;
    terrain.size = 4.0f;
    terrain.subdivisions = 4u;
    terrain.heights.assign(25u, 0.0f);
    terrain.layer_weights.assign(25u, {255u, 0u, 0u, 0u});
    std::vector<terrain_sculpt_brush_delta> deltas;
    terrain_brush_settings brush;
    brush.tool = terrain_brush_tool::sculpt;
    brush.radius = 0.25f;
    brush.strength = 0.5f;

    const auto dirty = apply_terrain_brush(terrain, {0.0f, 0.0f, 0.0f}, brush, 1.0f, &deltas);
    REQUIRE(dirty.valid);
    REQUIRE(deltas.size() == 1u);
    CHECK(deltas.front().x == 2u);
    CHECK(deltas.front().z == 2u);
    CHECK(deltas.front().delta == 6.0f);
}

TEST_CASE("M3.5 compatibility brush reports exact per-sample paint deltas")
{
    using namespace arc::scene;
    terrain_component terrain;
    terrain.size = 4.0f;
    terrain.subdivisions = 4u;
    terrain.heights.assign(25u, 0.0f);
    terrain.layer_weights.assign(25u, {255u, 0u, 0u, 0u});
    std::vector<terrain_paint_brush_delta> deltas;
    terrain_brush_settings brush;
    brush.tool = terrain_brush_tool::paint;
    brush.radius = 0.25f;
    brush.strength = 0.5f;
    brush.active_layer = 1u;

    const auto dirty = apply_terrain_brush(terrain, {0.0f, 0.0f, 0.0f}, brush, 1.0f, nullptr, &deltas);
    REQUIRE(dirty.valid);
    REQUIRE(dirty.weights_changed);
    REQUIRE(deltas.size() == 1u);
    CHECK(deltas.front().x == 2u);
    CHECK(deltas.front().z == 2u);
    CHECK(deltas.front().delta == std::array<std::int16_t, 4>{-255, 255, 0, 0});
}

TEST_CASE("M3.3 default evaluator applies sculpt payload for the requested authoring region")
{
    using namespace arc::scene;
    auto asset = make_asset();
    asset.source.kind = terrain_source_kind::heightfield;
    asset.source.asset.guid = arc::assets::generate_asset_guid();
    const auto layer = add_terrain_sculpt_layer(asset).id;
    const auto center = terrain_modifier_sample_coordinate_max / 2u;
    REQUIRE(set_terrain_sculpt_region_samples(asset, layer, {0, 0}, {{center, center, 2.5f}}).revision != 0u);

    std::array<float, 9> heights{};
    std::array<std::array<std::uint8_t, 4>, 9> weights{};
    for (auto& weight : weights)
        weight = {255u, 0u, 0u, 0u};
    terrain_evaluation_request request;
    request.region = {0, 0};
    request.heightfield_source = terrain_heightfield_source_view{.sample_width = 3u,
                                                                 .sample_height = 3u,
                                                                 .width = 100.0f,
                                                                 .depth = 100.0f,
                                                                 .heights = heights,
                                                                 .material_weights = weights,
                                                                 .source_revision = 1u};

    const auto evaluator = make_default_terrain_evaluator();
    const auto result = evaluator.evaluate(asset, request);
    REQUIRE(result.succeeded);
    const auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&result.surface.geometry);
    REQUIRE(heightfield != nullptr);
    REQUIRE(heightfield->heights.size() == 9u);
    CHECK(heightfield->heights[4] == 2.5f);
}

TEST_CASE("M3.5 default evaluator applies ordered paint payloads to material weights")
{
    using namespace arc::scene;
    auto asset = make_asset();
    asset.source.kind = terrain_source_kind::heightfield;
    asset.source.asset.guid = arc::assets::generate_asset_guid();
    const auto layer = add_terrain_paint_layer(asset).id;
    const auto center = terrain_modifier_sample_coordinate_max / 2u;
    REQUIRE(set_terrain_paint_region_samples(asset, layer, {0, 0}, {{center, center, {-96, 96, 0, 0}}}).revision != 0u);

    std::array<float, 9> heights{};
    std::array<std::array<std::uint8_t, 4>, 9> weights{};
    for (auto& weight : weights)
        weight = {255u, 0u, 0u, 0u};
    terrain_evaluation_request request;
    request.region = {0, 0};
    request.heightfield_source = terrain_heightfield_source_view{.sample_width = 3u,
                                                                 .sample_height = 3u,
                                                                 .width = 100.0f,
                                                                 .depth = 100.0f,
                                                                 .heights = heights,
                                                                 .material_weights = weights,
                                                                 .source_revision = 1u};

    const auto result = make_default_terrain_evaluator().evaluate(asset, request);
    REQUIRE(result.succeeded);
    const auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&result.surface.geometry);
    REQUIRE(heightfield != nullptr);
    REQUIRE(heightfield->material_weights.size() == 9u);
    CHECK(heightfield->material_weights[4] == std::array<std::uint8_t, 4>{159u, 96u, 0u, 0u});
}
