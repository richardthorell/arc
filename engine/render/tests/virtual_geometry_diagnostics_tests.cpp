#include <arc/render/virtual_geometry.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <span>

TEST_CASE("M2.6 exposes request overflow and post eviction reload pressure")
{
    using namespace arc::render;

    virtual_mesh_data geometry;
    geometry.pages = {{.uncompressed_size = 8u, .compressed_size = 4u, .root = true},
                      {.uncompressed_size = 8u, .compressed_size = 4u},
                      {.uncompressed_size = 8u, .compressed_size = 4u}};
    const virtual_mesh_handle resource{17u, 3u};
    virtual_geometry_residency_manager residency({.gpu_budget_bytes = 16u,
                                                  .compressed_cpu_budget_bytes = 8u,
                                                  .maximum_requests_per_frame = 1u,
                                                  .protected_frame_count = 0u,
                                                  .reload_cooldown_frames = 4u});
    residency.register_resource(resource, geometry, 9u);

    residency.begin_frame(1u);
    const std::array<virtual_geometry_page_request, 2> initial{virtual_geometry_page_request{.resource = resource,
                                                                                             .resource_generation = 9u,
                                                                                             .page_index = 1u,
                                                                                             .projected_error = 10.0f,
                                                                                             .screen_coverage = 1.0f},
                                                               virtual_geometry_page_request{.resource = resource,
                                                                                             .resource_generation = 9u,
                                                                                             .page_index = 2u,
                                                                                             .projected_error = 5.0f,
                                                                                             .screen_coverage = 1.0f}};
    residency.request(initial);
    auto loads = residency.take_load_requests();
    REQUIRE(loads.size() == 1u);
    REQUIRE(loads.front().page_index == 1u);
    CHECK(residency.snapshot().request_budget_overflow == 1u);
    residency.publish(resource, 9u, 1u, 8u, 4u);

    residency.begin_frame(2u);
    loads = residency.take_load_requests();
    REQUIRE(loads.size() == 1u);
    REQUIRE(loads.front().page_index == 2u);
    residency.publish(resource, 9u, 2u, 8u, 4u);
    const auto evictions = residency.take_evictions();
    REQUIRE(evictions.size() == 1u);
    REQUIRE(evictions.front().page_index == 1u);

    residency.begin_frame(3u);
    const virtual_geometry_page_request retry{.resource = resource,
                                              .resource_generation = 9u,
                                              .page_index = 1u,
                                              .projected_error = 10.0f,
                                              .screen_coverage = 0.5f};
    residency.request(std::span(&retry, 1u));
    const auto snapshot = residency.snapshot();
    CHECK(snapshot.reload_pressure_requests == 1u);
    CHECK(snapshot.cooldown_suppressed_requests == 1u);
    CHECK(residency.take_load_requests().empty());
}
