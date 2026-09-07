#include <arc/scene/terrain_asset.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

namespace
{

arc::scene::terrain_stable_id id(std::uint64_t low)
{
    return {0x7465727261696e00ull, low};
}

arc::scene::terrain_asset make_valid_asset()
{
    arc::scene::terrain_asset asset;
    asset.source.id = id(1);
    asset.source.kind = arc::scene::terrain_source_kind::flat;
    return asset;
}

} // namespace

TEST_CASE("terrain asset stable IDs round trip")
{
    const auto value = id(42);
    const auto text = arc::scene::to_string(value);
    REQUIRE(text.size() == 36);
    REQUIRE(arc::scene::parse_terrain_stable_id(text) == value);
    REQUIRE_FALSE(arc::scene::parse_terrain_stable_id("invalid").has_value());
}

TEST_CASE("terrain domains compose without coupling derived products")
{
    using arc::scene::terrain_domain;
    const auto tunnel = terrain_domain::geometry | terrain_domain::topology | terrain_domain::collision |
                        terrain_domain::navigation;
    REQUIRE(arc::scene::terrain_domain_contains(tunnel, terrain_domain::geometry));
    REQUIRE(arc::scene::terrain_domain_contains(tunnel, terrain_domain::topology));
    REQUIRE_FALSE(arc::scene::terrain_domain_contains(tunnel, terrain_domain::attributes));
    REQUIRE_FALSE(arc::scene::terrain_domain_contains(tunnel, terrain_domain::destruction));
}

TEST_CASE("terrain authoring regions use stable half open world partitions")
{
    const arc::scene::terrain_coordinate_system coordinates{1000.0, 25.0, -500.0, 1.0};
    const arc::scene::terrain_partition_settings partition{256.0, 8.0};

    REQUIRE(arc::scene::terrain_region_at(coordinates, partition, 1000.0, -500.0) ==
            arc::scene::terrain_region_id{0, 0});
    REQUIRE(arc::scene::terrain_region_at(coordinates, partition, 1255.999, -244.001) ==
            arc::scene::terrain_region_id{0, 0});
    REQUIRE(arc::scene::terrain_region_at(coordinates, partition, 1256.0, -244.0) ==
            arc::scene::terrain_region_id{1, 1});
    REQUIRE(arc::scene::terrain_region_at(coordinates, partition, 999.999, -500.001) ==
            arc::scene::terrain_region_id{-1, -1});

    const auto bounds = arc::scene::terrain_region_bounds(coordinates, partition, {-1, 2});
    REQUIRE(bounds.min_x == Catch::Approx(744.0));
    REQUIRE(bounds.max_x == Catch::Approx(1000.0));
    REQUIRE(bounds.min_z == Catch::Approx(12.0));
    REQUIRE(bounds.max_z == Catch::Approx(268.0));
    REQUIRE(bounds.min_y == Catch::Approx(25.0));
    REQUIRE(bounds.max_y == Catch::Approx(25.0));
}

TEST_CASE("terrain overlapping regions are deterministic and do not double count shared edges")
{
    const arc::scene::terrain_coordinate_system coordinates{};
    const arc::scene::terrain_partition_settings partition{100.0, 12.0};
    const arc::scene::terrain_world_bounds bounds{0.0, -20.0, 0.0, 200.0, 80.0, 100.0};

    const auto regions = arc::scene::terrain_regions_overlapping(coordinates, partition, bounds);
    REQUIRE(regions == std::vector<arc::scene::terrain_region_id>{{0, 0}, {1, 0}});

    const auto dependency = arc::scene::expand_terrain_bounds(bounds, partition.dependency_halo);
    REQUIRE(dependency.min_x == Catch::Approx(-12.0));
    REQUIRE(dependency.max_x == Catch::Approx(212.0));
    REQUIRE(dependency.min_z == Catch::Approx(-12.0));
    REQUIRE(dependency.max_z == Catch::Approx(112.0));
}

TEST_CASE("terrain asset validates source modifiers attributes and runtime policy")
{
    auto asset = make_valid_asset();

    arc::scene::terrain_modifier_descriptor road;
    road.id = id(2);
    road.type_id = "arc.terrain.road";
    road.name = "Road";
    road.domains = arc::scene::terrain_domain::geometry | arc::scene::terrain_domain::attributes |
                   arc::scene::terrain_domain::navigation;
    asset.modifiers.push_back(road);

    arc::scene::terrain_attribute_definition grass;
    grass.id = id(3);
    grass.name = "GrassDensity";
    grass.type = arc::scene::terrain_attribute_type::floating_point;
    grass.semantic = arc::scene::terrain_attribute_semantic::foliage_density;
    grass.default_value = 0.0;
    asset.attributes.push_back(grass);

    REQUIRE(arc::scene::validate_terrain_asset(asset).valid());

    asset.attributes.push_back(grass);
    const auto duplicate = arc::scene::validate_terrain_asset(asset);
    REQUIRE_FALSE(duplicate.valid());
    REQUIRE(std::any_of(duplicate.issues.begin(), duplicate.issues.end(), [](const auto& issue)
                        { return issue.code == arc::scene::terrain_asset_validation_code::duplicate_stable_id; }));
    REQUIRE(std::any_of(duplicate.issues.begin(), duplicate.issues.end(), [](const auto& issue)
                        { return issue.code == arc::scene::terrain_asset_validation_code::duplicate_attribute_name; }));
}

TEST_CASE("terrain validation keeps fracture derived data independent from asset validity")
{
    auto asset = make_valid_asset();
    asset.runtime.mutability = arc::scene::terrain_runtime_mutability::fractureable;
    asset.runtime.persistent_runtime_changes = true;
    asset.build.build_destruction = false;

    const auto validation = arc::scene::validate_terrain_asset(asset);
    REQUIRE(validation.valid());
    REQUIRE(std::any_of(validation.issues.begin(), validation.issues.end(), [](const auto& issue)
                        { return issue.severity == arc::scene::terrain_asset_validation_severity::warning; }));

    asset.runtime.mutability = arc::scene::terrain_runtime_mutability::immutable;
    REQUIRE_FALSE(arc::scene::validate_terrain_asset(asset).valid());
}

TEST_CASE("heightfield and mesh sources are authoring inputs rather than terrain types")
{
    auto asset = make_valid_asset();
    asset.source.kind = arc::scene::terrain_source_kind::heightfield;
    asset.source.asset.path_hint = "Assets/Terrain/mountain.r16";
    asset.source.asset.expected_type = arc::assets::asset_types::texture_2d;
    REQUIRE(arc::scene::validate_terrain_asset(asset).valid());

    asset.source.kind = arc::scene::terrain_source_kind::mesh;
    asset.source.asset.path_hint = "Assets/Terrain/cliff.fbx";
    asset.source.asset.expected_type = arc::assets::asset_types::static_mesh;
    REQUIRE(arc::scene::validate_terrain_asset(asset).valid());
}
