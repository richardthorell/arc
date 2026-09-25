#include <arc/scene/terrain_asset.h>
#include <arc/scene/terrain_asset_io.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <variant>
#include <vector>

namespace
{

arc::scene::terrain_asset make_asset()
{
    arc::scene::terrain_asset asset;
    asset.source.id = {0x7465727261696e00ull, 1u};
    asset.source.kind = arc::scene::terrain_source_kind::flat;
    asset.partition.authoring_region_size = 256.0;
    return asset;
}

} // namespace

TEST_CASE("M3.1 sculpt and paint layers own sparse per-region edit payloads")
{
    using namespace arc::scene;

    auto asset = make_asset();
    const auto source = asset.source;
    const auto sculpt_id = add_terrain_sculpt_layer(asset, "Large Forms").id;
    const auto paint_id = add_terrain_paint_layer(asset, "Ground Paint").id;

    const auto sculpt_dirty = set_terrain_sculpt_region_samples(asset, sculpt_id, {2, -1},
                                                                {{7u, 3u, 1.25f}, {4u, 2u, -0.5f}, {9u, 3u, 0.0f}});
    REQUIRE(sculpt_dirty.revision != 0u);
    REQUIRE((sculpt_dirty.regions == std::vector<terrain_region_id>{{2, -1}}));

    const auto* sculpt = find_terrain_modifier(asset, sculpt_id);
    REQUIRE(sculpt != nullptr);
    REQUIRE(sculpt->type_id == terrain_builtin_modifier_types::sculpt_layer);
    REQUIRE(sculpt->domains == terrain_domain::geometry);
    REQUIRE(sculpt->affected_bounds.has_value());
    const auto* sculpt_payload = find_terrain_modifier_payload(*sculpt, {2, -1});
    REQUIRE(sculpt_payload != nullptr);
    REQUIRE(std::holds_alternative<terrain_sculpt_region_payload>(sculpt_payload->data));
    const auto& sculpt_samples = std::get<terrain_sculpt_region_payload>(sculpt_payload->data).samples;
    REQUIRE(sculpt_samples.size() == 2u);
    CHECK(sculpt_samples[0] == terrain_sculpt_sample_delta{4u, 2u, -0.5f});
    CHECK(sculpt_samples[1] == terrain_sculpt_sample_delta{7u, 3u, 1.25f});

    const auto sculpt_region =
        std::ranges::find_if(asset.regions, [](const auto& value) { return value.id == terrain_region_id{2, -1}; });
    REQUIRE(sculpt_region != asset.regions.end());
    CHECK(terrain_domain_contains(sculpt_region->dirty_domains, terrain_domain::geometry));
    CHECK_FALSE(terrain_domain_contains(sculpt_region->dirty_domains, terrain_domain::attributes));

    const auto paint_dirty = set_terrain_paint_region_samples(
        asset, paint_id, {-3, 4},
        {{3u, 8u, std::array<std::int16_t, 4>{32, -32, 0, 0}}, {1u, 1u, std::array<std::int16_t, 4>{0, 0, 0, 0}}});
    REQUIRE(paint_dirty.revision != 0u);
    REQUIRE((paint_dirty.regions == std::vector<terrain_region_id>{{-3, 4}}));

    const auto* paint = find_terrain_modifier(asset, paint_id);
    REQUIRE(paint != nullptr);
    REQUIRE(paint->type_id == terrain_builtin_modifier_types::paint_layer);
    REQUIRE(paint->domains == terrain_domain::attributes);
    const auto* paint_payload = find_terrain_modifier_payload(*paint, {-3, 4});
    REQUIRE(paint_payload != nullptr);
    REQUIRE(std::holds_alternative<terrain_paint_region_payload>(paint_payload->data));
    REQUIRE(std::get<terrain_paint_region_payload>(paint_payload->data).samples.size() == 1u);

    CHECK(asset.source.id == source.id);
    CHECK(asset.source.kind == source.kind);
    CHECK(asset.source.asset.guid == source.asset.guid);
    CHECK(asset.source.asset.path_hint == source.asset.path_hint);
    REQUIRE(validate_terrain_asset(asset).valid());
}

TEST_CASE("M3.1 sparse edit payloads survive terrain asset save and reload")
{
    using namespace arc::scene;

    auto asset = make_asset();
    const auto sculpt_id = add_terrain_sculpt_layer(asset, "Trail").id;
    const auto paint_id = add_terrain_paint_layer(asset, "Mud").id;
    REQUIRE(set_terrain_sculpt_region_samples(asset, sculpt_id, {0, 0}, {{8u, 9u, 2.5f}}).revision != 0u);
    REQUIRE(set_terrain_paint_region_samples(asset, paint_id, {1, 0},
                                             {{12u, 2u, std::array<std::int16_t, 4>{-16, 16, 0, 0}}})
                .revision != 0u);

    const auto encoded = write_terrain_asset_json(asset, false);
    REQUIRE(encoded);
    REQUIRE(encoded.value().find("regionPayloads") != std::string::npos);
    REQUIRE(encoded.value().find("sculpt-height-delta") != std::string::npos);
    REQUIRE(encoded.value().find("paint-weight-delta") != std::string::npos);

    const auto decoded = read_terrain_asset_json(encoded.value());
    REQUIRE(decoded);
    REQUIRE(validate_terrain_asset(decoded.value()).valid());

    const auto* sculpt = find_terrain_modifier(decoded.value(), sculpt_id);
    const auto* paint = find_terrain_modifier(decoded.value(), paint_id);
    REQUIRE(sculpt != nullptr);
    REQUIRE(paint != nullptr);
    const auto* sculpt_payload = find_terrain_modifier_payload(*sculpt, {0, 0});
    const auto* paint_payload = find_terrain_modifier_payload(*paint, {1, 0});
    REQUIRE(sculpt_payload != nullptr);
    REQUIRE(paint_payload != nullptr);
    CHECK(std::get<terrain_sculpt_region_payload>(sculpt_payload->data).samples.front() ==
          terrain_sculpt_sample_delta{8u, 9u, 2.5f});
    CHECK(std::get<terrain_paint_region_payload>(paint_payload->data).samples.front() ==
          terrain_paint_sample_delta{12u, 2u, std::array<std::int16_t, 4>{-16, 16, 0, 0}});
}

TEST_CASE("M3 production workflow duplicates modifier payloads with a new stable identity")
{
    using namespace arc::scene;

    auto asset = make_asset();
    const auto source_id = add_terrain_sculpt_layer(asset, "Cliff Detail").id;
    REQUIRE(set_terrain_sculpt_region_samples(asset, source_id, {2, -1}, {{7u, 3u, 1.25f}}).revision != 0u);
    for (auto& region : asset.regions)
    {
        region.compiled_revision = region.dirty_revision;
        region.dirty_domains = terrain_domain::none;
    }
    const auto revision = asset.authoring_revision;

    const auto* duplicate = duplicate_terrain_modifier(asset, source_id);
    REQUIRE(duplicate != nullptr);
    CHECK(duplicate->id != source_id);
    CHECK(duplicate->name == "Cliff Detail Copy");
    CHECK(duplicate->type_id == terrain_builtin_modifier_types::sculpt_layer);
    CHECK(duplicate->region_payloads.size() == 1u);
    CHECK(asset.authoring_revision == revision + 1u);
    REQUIRE(asset.modifiers.size() == 2u);
    CHECK(asset.modifiers[0].id == source_id);
    CHECK(asset.modifiers[1].id == duplicate->id);

    const auto region = std::ranges::find_if(asset.regions, [](const auto& value)
                                             { return value.id == terrain_region_id{2, -1}; });
    REQUIRE(region != asset.regions.end());
    CHECK(terrain_domain_contains(region->dirty_domains, terrain_domain::geometry));
    CHECK(validate_terrain_asset(asset).valid());
}

TEST_CASE("M3.1 clearing the last sparse sample removes the region payload")
{
    using namespace arc::scene;

    auto asset = make_asset();
    const auto sculpt_id = add_terrain_sculpt_layer(asset).id;
    REQUIRE(set_terrain_sculpt_region_samples(asset, sculpt_id, {0, 0}, {{1u, 1u, 1.0f}}).revision != 0u);
    REQUIRE(set_terrain_sculpt_region_samples(asset, sculpt_id, {0, 0}, {}).revision != 0u);

    const auto* sculpt = find_terrain_modifier(asset, sculpt_id);
    REQUIRE(sculpt != nullptr);
    CHECK(sculpt->region_payloads.empty());
    CHECK_FALSE(sculpt->affected_bounds.has_value());
}

TEST_CASE("M3.5 paint edits fold into sparse attribute payloads with one revision")
{
    using namespace arc::scene;

    auto asset = make_asset();
    const auto layer = add_terrain_paint_layer(asset, "Mud").id;
    const auto start_revision = asset.authoring_revision;
    const auto address = terrain_modifier_sample_at(asset.coordinates, asset.partition, 40.0, -12.0);
    const std::array edits{
        terrain_paint_sample_edit{address.region, {address.x, address.z, {-10, 10, 0, 0}}},
        terrain_paint_sample_edit{address.region, {address.x, address.z, {-15, 15, 0, 0}}},
    };

    const auto update = accumulate_terrain_paint_samples(asset, layer, edits);

    CHECK(update.revision == start_revision + 1u);
    REQUIRE(update.regions == std::vector<terrain_region_id>{address.region});
    const auto* modifier = find_terrain_modifier(asset, layer);
    REQUIRE(modifier != nullptr);
    const auto* payload = find_terrain_modifier_payload(*modifier, address.region);
    REQUIRE(payload != nullptr);
    const auto& samples = std::get<terrain_paint_region_payload>(payload->data).samples;
    REQUIRE(samples.size() == 1u);
    CHECK(samples.front().delta == std::array<std::int16_t, 4>{-25, 25, 0, 0});
    const auto region =
        std::ranges::find_if(asset.regions, [&](const auto& value) { return value.id == address.region; });
    REQUIRE(region != asset.regions.end());
    CHECK(region->dirty_domains == terrain_domain::attributes);
    CHECK(validate_terrain_asset(asset).valid());
}

TEST_CASE("M3.5 paint edits remove a sparse sample when its accumulated delta returns to zero")
{
    using namespace arc::scene;

    auto asset = make_asset();
    const auto layer = add_terrain_paint_layer(asset).id;
    const auto address = terrain_modifier_sample_at(asset.coordinates, asset.partition, 4.0, 8.0);
    REQUIRE(accumulate_terrain_paint_samples(
                asset, layer,
                std::array{terrain_paint_sample_edit{address.region, {address.x, address.z, {-32, 32, 0, 0}}}})
                .revision != 0u);
    REQUIRE(accumulate_terrain_paint_samples(
                asset, layer,
                std::array{terrain_paint_sample_edit{address.region, {address.x, address.z, {32, -32, 0, 0}}}})
                .revision != 0u);

    const auto* modifier = find_terrain_modifier(asset, layer);
    REQUIRE(modifier != nullptr);
    CHECK(find_terrain_modifier_payload(*modifier, address.region) == nullptr);
    CHECK_FALSE(modifier->affected_bounds.has_value());
}

TEST_CASE("M3.1 rejects sparse payloads attached to incompatible modifier types")
{
    using namespace arc::scene;

    auto asset = make_asset();
    auto& modifier = add_terrain_sculpt_layer(asset);
    terrain_modifier_region_payload payload;
    payload.region = {0, 0};
    payload.data = terrain_paint_region_payload{{{1u, 1u, {1, -1, 0, 0}}}};
    modifier.region_payloads.push_back(std::move(payload));

    const auto validation = validate_terrain_asset(asset);
    REQUIRE_FALSE(validation.valid());
    REQUIRE(std::ranges::any_of(validation.issues, [](const auto& issue)
                                { return issue.code == terrain_asset_validation_code::invalid_modifier_payload; }));
}
