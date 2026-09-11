#include <arc/editor/terrain_rebuild.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
bool rebuild(arc::editor::terrain_rebuild_session& session, arc::scene::terrain_render_proxy_cache& cache,
             arc::ecs::entity_guid guid, arc::scene::terrain_component& terrain, arc::render::renderer& renderer)
{
    arc::jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    (void)session.pump(jobs, cache, guid, terrain, renderer);
    jobs.shutdown();
    return session.pump(jobs, cache, guid, terrain, renderer);
}
}

TEST_CASE("M3.4 sculpt preview transitions to asset-owned regions and preserves distant generations")
{
    using namespace arc;
    scene::terrain_component terrain;
    terrain.size = 1024.0f;
    terrain.subdivisions = 16u;
    terrain.heights.assign(289u, 0.0f);
    terrain.layer_weights.assign(289u, {255u, 0u, 0u, 0u});
    scene::terrain_asset asset;
    asset.source.id = scene::generate_terrain_stable_id();
    const auto layer = scene::add_terrain_sculpt_layer(asset).id;
    render::renderer renderer;
    scene::terrain_render_proxy_cache cache;
    const auto guid = ecs::generate_entity_guid();
    editor::terrain_rebuild_session session(asset, terrain);
    REQUIRE(rebuild(session, cache, guid, terrain, renderer));
    REQUIRE(cache.find(guid)->asset_owned);
    REQUIRE(cache.find(guid)->regions.size() == 16u);
    const auto distant = cache.find(guid)->regions.back().geometry;
    const auto generation = cache.find(guid)->generation;
    const auto address = scene::terrain_modifier_sample_at(asset.coordinates, asset.partition, -448.0, -448.0);
    REQUIRE(scene::accumulate_terrain_sculpt_samples(asset, layer,
        std::array{scene::terrain_sculpt_sample_edit{address.region, {address.x, address.z, 3.0f}}}).revision != 0u);
    terrain.heights[18u] = 3.0f;
    ++terrain.content_revision;
    const scene::terrain_dirty_region dirty{.min_x = 1u, .min_z = 1u, .max_x = 1u, .max_z = 1u,
        .valid = true, .heights_changed = true};
    REQUIRE(cache.synchronize(guid, terrain, renderer, &dirty));
    CHECK_FALSE(cache.find(guid)->asset_owned);
    CHECK(renderer.mesh_alive(distant.conventional));
    session.update(asset);
    REQUIRE(rebuild(session, cache, guid, terrain, renderer));
    CHECK(cache.find(guid)->asset_owned);
    CHECK(cache.find(guid)->generation == generation + 1u);
    CHECK(cache.find(guid)->regions.back().geometry == distant);
    CHECK(terrain.heights[18u] == Catch::Approx(3.0f));
    cache.clear(renderer);
}

TEST_CASE("M3.4 obsolete builds cannot overwrite newer sculpt edits")
{
    using namespace arc;
    scene::terrain_component terrain;
    terrain.size = 16.0f;
    terrain.subdivisions = 4u;
    terrain.heights.assign(25u, 0.0f);
    terrain.layer_weights.assign(25u, {255u, 0u, 0u, 0u});
    scene::terrain_asset asset;
    asset.source.id = scene::generate_terrain_stable_id();
    const auto layer = scene::add_terrain_sculpt_layer(asset).id;
    render::renderer renderer;
    scene::terrain_render_proxy_cache cache;
    const auto guid = ecs::generate_entity_guid();
    editor::terrain_rebuild_session session(asset, terrain);
    jobs::job_system first_jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    CHECK_FALSE(session.pump(first_jobs, cache, guid, terrain, renderer));
    first_jobs.shutdown();
    const auto address = scene::terrain_modifier_sample_at(asset.coordinates, asset.partition, 0.0, 0.0);
    REQUIRE(scene::accumulate_terrain_sculpt_samples(asset, layer,
        std::array{scene::terrain_sculpt_sample_edit{address.region, {address.x, address.z, 7.0f}}}).revision != 0u);
    session.update(asset);
    REQUIRE(rebuild(session, cache, guid, terrain, renderer));
    CHECK(terrain.heights[12u] == Catch::Approx(7.0f));
    CHECK(terrain.asset_authoring_revision == asset.authoring_revision);
    cache.clear(renderer);
}
