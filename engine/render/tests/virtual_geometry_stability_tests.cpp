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
