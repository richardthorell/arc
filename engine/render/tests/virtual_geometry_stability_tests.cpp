#include <arc/render/virtual_geometry_stability.h>

#include <catch2/catch_test_macros.hpp>

#include <limits>

TEST_CASE("virtual geometry invalidates previous HZB history for discontinuous views")
{
    using arc::render::virtual_geometry_history_invalidation;
    using arc::render::virtual_geometry_history_valid;

    CHECK(virtual_geometry_history_valid(virtual_geometry_history_invalidation::none));
    CHECK_FALSE(virtual_geometry_history_valid(virtual_geometry_history_invalidation::camera_cut));
    CHECK_FALSE(virtual_geometry_history_valid(virtual_geometry_history_invalidation::teleport));
    CHECK_FALSE(virtual_geometry_history_valid(virtual_geometry_history_invalidation::viewport_resize));
    CHECK_FALSE(virtual_geometry_history_valid(virtual_geometry_history_invalidation::projection_change));
    CHECK_FALSE(virtual_geometry_history_valid(virtual_geometry_history_invalidation::newly_visible_instance));
    CHECK_FALSE(virtual_geometry_history_valid(virtual_geometry_history_invalidation::world_reset));

    const auto combined = virtual_geometry_history_invalidation::teleport |
                          virtual_geometry_history_invalidation::projection_change;
    CHECK(contains(combined, virtual_geometry_history_invalidation::teleport));
    CHECK(contains(combined, virtual_geometry_history_invalidation::projection_change));
    CHECK_FALSE(contains(combined, virtual_geometry_history_invalidation::viewport_resize));
    CHECK_FALSE(virtual_geometry_history_valid(combined));
}

TEST_CASE("virtual geometry projected error uses hysteresis around refinement threshold")
{
    using arc::render::should_refine_virtual_geometry;

    CHECK_FALSE(should_refine_virtual_geometry(1.05f, 1.0f, false));
    CHECK(should_refine_virtual_geometry(1.11f, 1.0f, false));

    CHECK(should_refine_virtual_geometry(0.95f, 1.0f, true));
    CHECK_FALSE(should_refine_virtual_geometry(0.89f, 1.0f, true));
}

TEST_CASE("virtual geometry refinement policy rejects invalid inputs deterministically")
{
    using arc::render::should_refine_virtual_geometry;

    CHECK_FALSE(should_refine_virtual_geometry(std::numeric_limits<float>::infinity(), 1.0f, false));
    CHECK_FALSE(should_refine_virtual_geometry(2.0f, 0.0f, false));
    CHECK_FALSE(should_refine_virtual_geometry(2.0f, -1.0f, true));
}

TEST_CASE("virtual geometry uses the nominal threshold when refinement history is invalid")
{
    using arc::render::should_refine_virtual_geometry;

    CHECK(should_refine_virtual_geometry(1.05f, 1.0f, false, false));
    CHECK_FALSE(should_refine_virtual_geometry(0.95f, 1.0f, true, false));
    CHECK_FALSE(should_refine_virtual_geometry(1.05f, 1.0f, false, true));
    CHECK(should_refine_virtual_geometry(0.95f, 1.0f, true, true));
}

TEST_CASE("virtual geometry refinement history is bounded double buffered and generation safe")
{
    using namespace arc::render;
    const virtual_geometry_refinement_key first{.instance_index = 3u,
                                                 .instance_generation = 7u,
                                                 .resource_generation = 11u,
                                                 .hierarchy_node = 5u};
    auto replaced = first;
    ++replaced.resource_generation;

    virtual_geometry_refinement_history history(4u);
    history.begin_frame();
    CHECK(history.record(first));
    CHECK_FALSE(history.refined_last_frame(first));
    history.begin_frame();
    CHECK(history.refined_last_frame(first));
    CHECK_FALSE(history.refined_last_frame(replaced));
    CHECK(history.record(replaced));
    history.begin_frame();
    CHECK(history.refined_last_frame(replaced));
    CHECK_FALSE(history.refined_last_frame(first));

    virtual_geometry_refinement_history saturated(1u);
    saturated.begin_frame();
    CHECK(saturated.record(first));
    CHECK_FALSE(saturated.record(replaced));
    CHECK(saturated.current_overflowed());
    saturated.begin_frame();
    CHECK(saturated.previous_overflowed());
    CHECK_FALSE(saturated.refined_last_frame(first));
}
