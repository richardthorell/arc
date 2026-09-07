#include <arc/scene/terrain_asset_io.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>

namespace
{

arc::scene::terrain_stable_id terrain_id(std::uint64_t low)
{
    return {0x7465727261696e00ull, low};
}

arc::scene::terrain_asset make_heightfield_asset()
{
    arc::scene::terrain_asset asset;
    asset.coordinates = {4096.0, 32.0, -2048.0, 1.0};
    asset.partition = {128.0, 12.0};
    asset.source.id = terrain_id(1);
    asset.source.kind = arc::scene::terrain_source_kind::heightfield;
    asset.source.asset.expected_type = arc::assets::asset_types::texture_2d;
    asset.source.asset.path_hint = "Content/Terrain/island-height.png";
    asset.source.transform.translation = {4.0f, 2.0f, -8.0f};

    arc::scene::terrain_modifier_descriptor sculpt;
    sculpt.id = terrain_id(2);
    sculpt.type_id = "arc.terrain.sculpt";
    sculpt.name = "Ridge";
    sculpt.domains = arc::scene::terrain_domain::geometry | arc::scene::terrain_domain::collision |
                     arc::scene::terrain_domain::navigation;
    sculpt.affected_bounds = arc::scene::terrain_world_bounds{4096.0, 0.0, -2048.0, 4160.0, 80.0, -1984.0};
    sculpt.canonical_parameters = R"({"strength":0.75,"brush":"ridge"})";
    asset.modifiers.push_back(std::move(sculpt));

    arc::scene::terrain_attribute_definition hardness;
    hardness.id = terrain_id(3);
    hardness.name = "Hardness";
    hardness.type = arc::scene::terrain_attribute_type::floating_point;
    hardness.semantic = arc::scene::terrain_attribute_semantic::hardness;
    hardness.default_value = 0.5;
    asset.attributes.push_back(std::move(hardness));

    asset.runtime.mutability = arc::scene::terrain_runtime_mutability::deformable;
    asset.runtime.persistent_runtime_changes = true;
    return asset;
}

const arc::scene::terrain_region_dependency* find_dependency(const arc::scene::terrain_build_region_snapshot& snapshot,
                                                             arc::scene::terrain_region_id region)
{
    const auto found = std::find_if(snapshot.dependencies.begin(), snapshot.dependencies.end(),
                                    [&](const auto& dependency) { return dependency.region == region; });
    return found == snapshot.dependencies.end() ? nullptr : &*found;
}

} // namespace

TEST_CASE("terrain assets round trip heightfield authoring and region state")
{
    auto asset = make_heightfield_asset();
    const auto dirty =
        arc::scene::mark_terrain_dirty(asset, {4096.0, 0.0, -2048.0, 4224.0, 100.0, -1920.0},
                                       arc::scene::terrain_domain::geometry | arc::scene::terrain_domain::collision);
    REQUIRE(dirty.revision == 2);
    REQUIRE(dirty.regions.size() == 1);

    auto& region = arc::scene::ensure_terrain_region(asset, {0, 0});
    region.dependencies.push_back({{2, -1}, arc::scene::terrain_domain::attributes});

    const auto encoded = arc::scene::write_terrain_asset_json(asset);
    REQUIRE(encoded.has_value());
    REQUIRE(encoded.value().find("\"format\": \"arc.terrain\"") != std::string::npos);

    const auto decoded = arc::scene::read_terrain_asset_json(encoded.value());
    REQUIRE(decoded.has_value());
    const auto& round_trip = decoded.value();
    REQUIRE(round_trip.source.kind == arc::scene::terrain_source_kind::heightfield);
    REQUIRE(round_trip.source.id == asset.source.id);
    REQUIRE(round_trip.source.asset.path_hint == "Content/Terrain/island-height.png");
    REQUIRE(round_trip.coordinates.origin_x == Catch::Approx(4096.0));
    REQUIRE(round_trip.partition.authoring_region_size == Catch::Approx(128.0));
    REQUIRE(round_trip.modifiers.size() == 1);
    REQUIRE(round_trip.modifiers.front().type_id == "arc.terrain.sculpt");
    REQUIRE(round_trip.attributes.size() == 1);
    REQUIRE(round_trip.attributes.front().semantic == arc::scene::terrain_attribute_semantic::hardness);
    REQUIRE(round_trip.authoring_revision == 2);
    REQUIRE(round_trip.regions.size() == 1);
    REQUIRE(round_trip.regions.front().dirty_revision == 2);
    REQUIRE(arc::scene::terrain_domain_contains(round_trip.regions.front().dirty_domains,
                                                arc::scene::terrain_domain::geometry));
    REQUIRE(round_trip.regions.front().dependencies.size() == 1);
    REQUIRE((round_trip.regions.front().dependencies.front().region == arc::scene::terrain_region_id{2, -1}));
}

TEST_CASE("terrain dirty tracking rejects stale region builds and publishes complete revisions")
{
    arc::scene::terrain_asset asset;
    asset.source.id = terrain_id(1);
    asset.partition = {100.0, 10.0};

    const auto geometry =
        arc::scene::mark_terrain_dirty(asset, {0.0, 0.0, 0.0, 50.0, 10.0, 50.0}, arc::scene::terrain_domain::geometry);
    REQUIRE(geometry.revision == 2);
    REQUIRE((geometry.regions == std::vector<arc::scene::terrain_region_id>{{0, 0}}));

    const auto attributes = arc::scene::mark_terrain_dirty(asset, {0.0, 0.0, 0.0, 50.0, 10.0, 50.0},
                                                           arc::scene::terrain_domain::attributes);
    REQUIRE(attributes.revision == 3);

    REQUIRE_FALSE(arc::scene::mark_terrain_region_compiled(asset, {0, 0}, arc::scene::terrain_domain::geometry,
                                                           geometry.revision));
    REQUIRE(arc::scene::mark_terrain_region_compiled(asset, {0, 0}, arc::scene::terrain_domain::geometry,
                                                     attributes.revision));

    const auto& partially_compiled = asset.regions.front();
    REQUIRE(partially_compiled.compiled_revision == 0);
    REQUIRE(partially_compiled.dirty_domains == arc::scene::terrain_domain::attributes);

    REQUIRE(arc::scene::mark_terrain_region_compiled(asset, {0, 0}, arc::scene::terrain_domain::attributes,
                                                     attributes.revision));
    REQUIRE(asset.regions.front().dirty_domains == arc::scene::terrain_domain::none);
    REQUIRE(asset.regions.front().compiled_revision == attributes.revision);
}

TEST_CASE("terrain build snapshots combine halo and explicit dependencies deterministically")
{
    arc::scene::terrain_asset asset;
    asset.source.id = terrain_id(1);
    asset.partition = {100.0, 10.0};

    auto& target = arc::scene::ensure_terrain_region(asset, {0, 0});
    target.dependencies.push_back({{5, 5}, arc::scene::terrain_domain::destruction});
    const auto dirty =
        arc::scene::mark_terrain_dirty(asset, {0.0, 0.0, 0.0, 100.0, 1.0, 100.0}, arc::scene::terrain_domain::geometry);
    REQUIRE(dirty.revision == 2);

    const auto snapshot = arc::scene::make_terrain_build_region_snapshot(asset, {0, 0});
    REQUIRE(snapshot.authoring_revision == 2);
    REQUIRE(snapshot.target_dirty_revision == 2);
    REQUIRE(snapshot.evaluation_bounds.min_x == Catch::Approx(-10.0));
    REQUIRE(snapshot.evaluation_bounds.max_x == Catch::Approx(110.0));
    REQUIRE(snapshot.evaluation_bounds.min_z == Catch::Approx(-10.0));
    REQUIRE(snapshot.evaluation_bounds.max_z == Catch::Approx(110.0));
    REQUIRE(find_dependency(snapshot, {0, 0}) == nullptr);

    const auto* west = find_dependency(snapshot, {-1, 0});
    REQUIRE(west != nullptr);
    REQUIRE(arc::scene::terrain_domain_contains(west->domains, arc::scene::terrain_domain::geometry));
    REQUIRE(arc::scene::terrain_domain_contains(west->domains, arc::scene::terrain_domain::topology));

    const auto* explicit_dependency = find_dependency(snapshot, {5, 5});
    REQUIRE(explicit_dependency != nullptr);
    REQUIRE(explicit_dependency->domains == arc::scene::terrain_domain::destruction);

    REQUIRE(std::is_sorted(
        snapshot.dependencies.begin(), snapshot.dependencies.end(), [](const auto& lhs, const auto& rhs)
        { return lhs.region.z < rhs.region.z || (lhs.region.z == rhs.region.z && lhs.region.x < rhs.region.x); }));
}

TEST_CASE("terrain importer materializes typed assets and reports authoring dependencies")
{
    auto asset = make_heightfield_asset();
    const auto encoded = arc::scene::write_terrain_asset_json(asset, false);
    REQUIRE(encoded.has_value());

    auto importer = arc::scene::make_terrain_asset_importer();
    REQUIRE(importer != nullptr);
    REQUIRE(importer->descriptor().id == arc::assets::importer_ids::terrain);
    REQUIRE(importer->descriptor().output_types ==
            std::vector<arc::assets::asset_type_id>{arc::assets::asset_types::terrain});
    REQUIRE(importer->descriptor().extensions == std::vector<std::string>{".terrain"});

    const auto bytes = std::as_bytes(std::span<const char>{encoded.value().data(), encoded.value().size()});
    arc::assets::asset_import_context context;
    context.reference.guid = arc::assets::generate_asset_guid();
    context.reference.expected_type = arc::assets::asset_types::terrain;
    context.metadata.type = arc::assets::asset_types::terrain;
    context.source_path = "Content/Terrain/island.terrain";
    context.source_bytes = bytes;

    const auto imported = importer->import(context);
    REQUIRE(imported.succeeded());
    const auto* payload = imported.payload.get<arc::scene::terrain_asset>();
    REQUIRE(payload != nullptr);
    REQUIRE(payload->source.kind == arc::scene::terrain_source_kind::heightfield);
    REQUIRE(imported.dependencies.size() == 1);
    REQUIRE(imported.dependencies.front().path_hint == "Content/Terrain/island-height.png");
}

TEST_CASE("terrain codec rejects documents outside its versioned contract")
{
    const auto missing_format = arc::scene::read_terrain_asset_json("{}");
    REQUIRE_FALSE(missing_format.has_value());
    REQUIRE(missing_format.error().code == arc::scene::terrain_asset_io_error_code::invalid_document);

    const auto unsupported =
        arc::scene::read_terrain_asset_json(R"({"format":"arc.terrain","formatVersion":99,"terrain":{}})");
    REQUIRE_FALSE(unsupported.has_value());
    REQUIRE(unsupported.error().code == arc::scene::terrain_asset_io_error_code::unsupported_format_version);
}
