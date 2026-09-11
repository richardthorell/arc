#include <arc/scene/terrain.h>
#include <arc/scene/terrain_region_build.h>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <stdexcept>

namespace
{
arc::scene::terrain_asset build_asset()
{
    arc::scene::terrain_asset asset;
    asset.source.id = arc::scene::generate_terrain_stable_id();
    for (std::int64_t x = 0; x < 3; ++x)
    {
        auto& region = arc::scene::ensure_terrain_region(asset, {x, 0});
        region.dirty_domains = x == 1 ? arc::scene::terrain_domain::attributes : arc::scene::terrain_domain::geometry;
    }
    return asset;
}

arc::scene::terrain_evaluation_result evaluate(const arc::scene::terrain_asset& asset, arc::scene::terrain_region_id id)
{
    return arc::scene::make_default_terrain_evaluator().evaluate(asset, {.region = id});
}

arc::scene::terrain_region_build_batch compile(const arc::scene::terrain_asset& asset)
{
    arc::jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    arc::scene::terrain_region_build_queue queue;
    REQUIRE(queue.schedule(jobs, asset, evaluate));
    jobs.shutdown();
    auto result = queue.take_ready(asset);
    REQUIRE(result.has_value());
    return std::move(*result);
}
}

TEST_CASE("M3.4 workers build only dirty geometry regions from immutable snapshots")
{
    auto asset = build_asset();
    arc::jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    arc::scene::terrain_region_build_queue queue;
    std::atomic<unsigned> calls{};
    REQUIRE(queue.schedule(jobs, asset, [&](const auto& snapshot, auto id)
        { ++calls; return evaluate(snapshot, id); }));
    CHECK_FALSE(queue.schedule(jobs, asset, evaluate));
    jobs.shutdown();
    auto result = queue.take_ready(asset);
    REQUIRE(result);
    REQUIRE(result->succeeded);
    CHECK(calls == 2u);
    REQUIRE(result->regions.size() == 2u);
    CHECK(result->regions[0].evaluation.region.x == 0);
    CHECK(result->regions[1].evaluation.region.x == 2);
    CHECK_FALSE(queue.pending());
    CHECK(asset.regions[0].dirty_domains == arc::scene::terrain_domain::geometry);
    CHECK_FALSE(queue.take_ready(asset));
}

TEST_CASE("M3.4 authoring changes reject completed builds before publication")
{
    auto asset = build_asset();
    arc::jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    arc::scene::terrain_region_build_queue queue;
    REQUIRE(queue.schedule(jobs, asset, evaluate));
    ++asset.authoring_revision;
    jobs.shutdown();
    const auto result = queue.take_ready(asset);
    REQUIRE(result);
    CHECK(result->stale);
    CHECK_FALSE(result->succeeded);
    CHECK(result->authoring_revision + 1u == asset.authoring_revision);
}

TEST_CASE("M3.4 worker failure and abandoned queues cannot publish partial batches")
{
    auto asset = build_asset();
    arc::jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    arc::scene::terrain_region_build_queue queue;
    REQUIRE(queue.schedule(jobs, asset, [](const auto&, auto) -> arc::scene::terrain_evaluation_result
        { throw std::runtime_error("source unavailable"); }));
    {
        arc::scene::terrain_region_build_queue abandoned;
        REQUIRE(abandoned.schedule(jobs, asset, evaluate));
    }
    jobs.shutdown();
    const auto result = queue.take_ready(asset);
    REQUIRE(result);
    CHECK_FALSE(result->succeeded);
}

TEST_CASE("M3.4 publication retains unaffected handles and rolls back incomplete staging")
{
    auto asset = build_asset();
    auto initial = compile(asset);
    REQUIRE(initial.succeeded);
    arc::render::renderer renderer;
    arc::scene::terrain_render_proxy_cache cache;
    arc::scene::terrain_component terrain;
    const auto guid = arc::ecs::generate_entity_guid();
    REQUIRE(cache.publish(guid, initial, terrain, renderer));
    const auto first = cache.find(guid)->regions[0].geometry;
    const auto unaffected = cache.find(guid)->regions[1].geometry;
    const auto generation = cache.find(guid)->generation;
    auto broken = compile(asset);
    broken.regions.back().attributes.reset();
    CHECK_FALSE(cache.publish(guid, broken, terrain, renderer));
    CHECK(cache.find(guid)->generation == generation);
    CHECK(cache.find(guid)->regions[0].geometry == first);
    CHECK(renderer.mesh_alive(first.conventional));
    CHECK(renderer.mesh_alive(unaffected.conventional));
    asset.regions.back().dirty_domains = arc::scene::terrain_domain::none;
    ++asset.authoring_revision;
    asset.regions.front().dirty_revision = asset.authoring_revision;
    auto next = compile(asset);
    REQUIRE(cache.publish(guid, next, terrain, renderer));
    CHECK(cache.find(guid)->generation == generation + 1u);
    CHECK(cache.find(guid)->regions[0].geometry != first);
    CHECK(cache.find(guid)->regions[1].geometry == unaffected);
    CHECK_FALSE(renderer.mesh_alive(first.conventional));
    cache.clear(renderer);
    CHECK_FALSE(renderer.mesh_alive(unaffected.conventional));
}
