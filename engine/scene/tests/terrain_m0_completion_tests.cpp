#include <arc/scene/terrain_artifacts.h>
#include <arc/scene/terrain_evaluator.h>
#include <arc/scene/terrain_region_seams.h>
#include <arc/scene/terrain_runtime_journal.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <variant>
#include <vector>

namespace
{

arc::scene::terrain_asset make_flat_asset()
{
    arc::scene::terrain_asset asset;
    asset.source.id = arc::scene::generate_terrain_stable_id();
    asset.source.kind = arc::scene::terrain_source_kind::flat;
    return asset;
}

arc::scene::terrain_modifier_descriptor height_offset(float value)
{
    arc::scene::terrain_modifier_descriptor modifier;
    modifier.id = arc::scene::generate_terrain_stable_id();
    modifier.type_id = "arc.height-offset";
    modifier.name = "Height offset";
    modifier.domains = arc::scene::terrain_domain::geometry;
    modifier.canonical_parameters = "{\"offset\":" + std::to_string(value) + "}";
    return modifier;
}

} // namespace

TEST_CASE("flat TerrainAsset evaluates deterministically into owning TerrainSurfaceIR")
{
    auto asset = make_flat_asset();
    asset.partition.authoring_region_size = 64.0;
    asset.source.transform.translation[1] = 3.0f;

    const auto evaluator = arc::scene::make_default_terrain_evaluator();
    const auto first = evaluator.evaluate(asset, {.region = {2, -1}});
    const auto second = evaluator.evaluate(asset, {.region = {2, -1}});
    REQUIRE(first.succeeded);
    REQUIRE(second.succeeded);
    REQUIRE(first.content_fingerprint == second.content_fingerprint);
    REQUIRE(first.content_fingerprint != 0u);
    REQUIRE(arc::scene::validate_terrain_surface_ir(first.surface.view()));
    CHECK(first.world_origin_x == Catch::Approx(160.0));
    CHECK(first.world_origin_z == Catch::Approx(-32.0));

    const auto* heightfield = std::get_if<arc::scene::terrain_evaluated_heightfield>(&first.surface.geometry);
    REQUIRE(heightfield != nullptr);
    REQUIRE(heightfield->sample_width == 2u);
    REQUIRE(heightfield->sample_height == 2u);
    REQUIRE(heightfield->heights.size() == 4u);
    CHECK(heightfield->heights.front() == Catch::Approx(3.0f));
}

TEST_CASE("TerrainAsset heightfield source data crosses the evaluator boundary without renderer dependencies")
{
    auto asset = make_flat_asset();
    asset.source.kind = arc::scene::terrain_source_kind::heightfield;
    asset.source.asset.path_hint = "terrain/source.r16";

    const std::vector<float> heights{-2.0f, 0.0f, 1.0f, 4.0f};
    const std::vector<std::array<std::uint8_t, 4>> weights(4u, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    arc::scene::terrain_evaluation_request request;
    request.heightfield_source = arc::scene::terrain_heightfield_source_view{
        .sample_width = 2u,
        .sample_height = 2u,
        .width = 20.0f,
        .depth = 20.0f,
        .heights = heights,
        .material_weights = weights,
        .source_revision = 9u,
    };

    const auto result = arc::scene::make_default_terrain_evaluator().evaluate(asset, request);
    REQUIRE(result.succeeded);
    REQUIRE(result.surface.source_revision == 9u);
    const auto view = result.surface.view();
    REQUIRE(arc::scene::validate_terrain_surface_ir(view));
    CHECK(view.local_bounds.min_y == Catch::Approx(-2.0));
    CHECK(view.local_bounds.max_y == Catch::Approx(4.0));

    const auto copy = arc::scene::copy_terrain_surface_ir(view);
    REQUIRE(copy.has_value());
    REQUIRE(arc::scene::terrain_surface_fingerprint(copy->view()) == result.content_fingerprint);
}

TEST_CASE("terrain modifiers are ordered bounded and never silently ignored")
{
    auto asset = make_flat_asset();
    asset.modifiers.push_back(height_offset(2.0f));
    asset.modifiers.push_back(height_offset(3.0f));
    auto disabled = height_offset(100.0f);
    disabled.enabled = false;
    asset.modifiers.push_back(disabled);
    auto outside = height_offset(200.0f);
    outside.affected_bounds = arc::scene::terrain_world_bounds{1000.0, -100.0, 1000.0, 1100.0, 100.0, 1100.0};
    asset.modifiers.push_back(outside);

    const auto evaluator = arc::scene::make_default_terrain_evaluator();
    const auto result = evaluator.evaluate(asset, {});
    REQUIRE(result.succeeded);
    const auto* heightfield = std::get_if<arc::scene::terrain_evaluated_heightfield>(&result.surface.geometry);
    REQUIRE(heightfield != nullptr);
    CHECK(heightfield->heights.front() == Catch::Approx(5.0f));

    auto unsupported_asset = make_flat_asset();
    auto unsupported = height_offset(1.0f);
    unsupported.type_id = "test.not-registered";
    unsupported_asset.modifiers.push_back(unsupported);
    const auto failed = evaluator.evaluate(unsupported_asset, {});
    REQUIRE_FALSE(failed.succeeded);
    REQUIRE_FALSE(failed.diagnostics.empty());
    CHECK(failed.diagnostics.back().code == arc::scene::terrain_evaluation_diagnostic_code::unsupported_modifier);
}

TEST_CASE("terrain artifact keys are content addressed and independent of dependency ordering")
{
    arc::scene::terrain_artifact_build_input first;
    first.region = {4, -2};
    first.surface_fingerprint = 0x12345678u;
    first.authoring_revision = 12u;
    first.source_revision = 9u;
    first.target_profile = "desktop-high";
    first.dependencies = {
        {{3, -2}, arc::scene::terrain_domain::geometry, 8u},
        {{4, -1}, arc::scene::terrain_domain::attributes, 7u},
    };
    auto reordered = first;
    std::reverse(reordered.dependencies.begin(), reordered.dependencies.end());

    const auto render_key =
        arc::scene::make_terrain_artifact_key(first, arc::scene::terrain_artifact_kind::render_geometry, 1u);
    const auto reordered_key =
        arc::scene::make_terrain_artifact_key(reordered, arc::scene::terrain_artifact_kind::render_geometry, 1u);
    const auto collision_key =
        arc::scene::make_terrain_artifact_key(first, arc::scene::terrain_artifact_kind::collision, 1u);
    REQUIRE(render_key.valid());
    REQUIRE(render_key == reordered_key);
    REQUIRE(render_key != collision_key);
    REQUIRE(arc::scene::to_string(render_key).size() == 32u);
}

TEST_CASE("terrain cooked manifests keep derived systems independent")
{
    arc::scene::terrain_artifact_build_input input;
    input.surface_fingerprint = 77u;
    input.authoring_revision = 4u;
    input.source_revision = 4u;
    const auto key =
        arc::scene::make_terrain_artifact_key(input, arc::scene::terrain_artifact_kind::render_geometry, 1u);

    arc::scene::terrain_cooked_manifest manifest;
    manifest.terrain = arc::assets::generate_asset_guid();
    manifest.authoring_revision = 4u;
    manifest.regions.push_back({
        .region = {},
        .bounds = {0.0, -1.0, 0.0, 256.0, 12.0, 256.0},
        .source_revision = 4u,
        .compiled_revision = 4u,
        .artifacts = {{arc::scene::terrain_artifact_kind::render_geometry, key, 1u, "terrain/render/region-0", 1u, 1u}},
    });
    REQUIRE(arc::scene::validate_terrain_cooked_manifest(manifest));

    manifest.regions.front().artifacts.push_back(
        {arc::scene::terrain_artifact_kind::render_geometry, key, 1u, "duplicate", 1u, 1u});
    REQUIRE_FALSE(arc::scene::validate_terrain_cooked_manifest(manifest));
}

TEST_CASE("terrain authoring seams have one deterministic owner independent of query order")
{
    const arc::scene::terrain_region_id left{4, -3};
    const arc::scene::terrain_region_id right{5, -3};
    REQUIRE(arc::scene::terrain_regions_share_edge(left, right));
    REQUIRE(arc::scene::terrain_regions_share_edge(right, left));
    REQUIRE(arc::scene::terrain_region_neighbor(left, arc::scene::terrain_region_edge::positive_x) == right);
    REQUIRE(arc::scene::terrain_shared_seam_owner(left, right) == left);
    REQUIRE(arc::scene::terrain_shared_seam_owner(right, left) == left);
    REQUIRE_FALSE(arc::scene::terrain_shared_seam_owner(left, {6, -3}).has_value());
}

TEST_CASE("terrain runtime journal is deterministic and stores operations instead of generated meshes")
{
    arc::scene::terrain_runtime_journal journal;
    journal.base_authoring_revision = 11u;
    journal.operations.push_back({
        .id = arc::scene::generate_terrain_stable_id(),
        .kind = arc::scene::terrain_runtime_operation_kind::boolean_subtract,
        .bounds = {-2.0, -2.0, -2.0, 2.0, 2.0, 2.0},
        .seed = 42u,
        .schema_version = 1u,
        .canonical_payload = "{\"shape\":\"sphere\",\"radius\":2.0}",
    });
    REQUIRE(arc::scene::validate_terrain_runtime_journal(journal));
    const auto first = arc::scene::terrain_runtime_journal_fingerprint(journal);
    const auto second = arc::scene::terrain_runtime_journal_fingerprint(journal);
    REQUIRE(first != 0u);
    REQUIRE(first == second);

    journal.operations.front().canonical_payload = "{\"shape\":\"sphere\",\"radius\":3.0}";
    REQUIRE(arc::scene::terrain_runtime_journal_fingerprint(journal) != first);
}
