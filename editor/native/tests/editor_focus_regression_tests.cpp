#include <arc/editor/editor_interaction.h>
#include <arc/scene/hierarchy.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("focused selection remains the orbit pivot across repeated focus commands")
{
    arc::ecs::world registry;
    const auto selected = registry.create();

    arc::scene::transform_component transform;
    transform.position = {6.0f, 2.0f, -3.0f};
    registry.emplace<arc::scene::transform_component>(selected, transform);
    registry.emplace<arc::scene::bounds_component>(
        selected,
        arc::geometric::box3f{arc::geometric::point3f{-1.0f, -2.0f, -0.5f},
                              arc::geometric::point3f{1.0f, 2.0f, 0.5f}});

    arc::editor::editor_camera_controller camera;
    REQUIRE(arc::editor::focus_selected_entity(registry, selected, camera));

    const auto first_focus = camera.focus_point();
    const float first_distance = camera.distance();
    CHECK(first_focus[0] == Catch::Approx(6.0f));
    CHECK(first_focus[1] == Catch::Approx(2.0f));
    CHECK(first_focus[2] == Catch::Approx(-3.0f));

    camera.orbit(31.0f, -14.0f);
    CHECK(camera.focus_point()[0] == Catch::Approx(first_focus[0]));
    CHECK(camera.focus_point()[1] == Catch::Approx(first_focus[1]));
    CHECK(camera.focus_point()[2] == Catch::Approx(first_focus[2]));

    REQUIRE(arc::editor::focus_selected_entity(registry, selected, camera));
    CHECK(camera.focus_point()[0] == Catch::Approx(first_focus[0]));
    CHECK(camera.focus_point()[1] == Catch::Approx(first_focus[1]));
    CHECK(camera.focus_point()[2] == Catch::Approx(first_focus[2]));
    CHECK(camera.distance() == Catch::Approx(first_distance));

    camera.orbit(-18.0f, 9.0f);
    CHECK(camera.focus_point()[0] == Catch::Approx(first_focus[0]));
    CHECK(camera.focus_point()[1] == Catch::Approx(first_focus[1]));
    CHECK(camera.focus_point()[2] == Catch::Approx(first_focus[2]));
}

TEST_CASE("focus selection refreshes dirty parent transforms before choosing the pivot")
{
    arc::ecs::world registry;
    const auto parent = registry.create();
    arc::scene::transform_component parent_transform;
    parent_transform.position = {10.0f, 1.0f, -4.0f};
    registry.emplace<arc::scene::transform_component>(parent, parent_transform);

    const auto selected = registry.create();
    arc::scene::transform_component child_transform;
    child_transform.position = {3.0f, 2.0f, 0.0f};
    registry.emplace<arc::scene::transform_component>(selected, child_transform);
    registry.emplace<arc::scene::bounds_component>(
        selected,
        arc::geometric::box3f{arc::geometric::point3f{-0.5f, -0.5f, -0.5f},
                              arc::geometric::point3f{0.5f, 0.5f, 0.5f}});

    REQUIRE(arc::scene::reparent(registry, selected, parent, {},
                                 arc::scene::reparent_transform_policy::preserve_local));
    REQUIRE(registry.get<arc::scene::transform_component>(selected).dirty);

    arc::editor::editor_camera_controller camera;
    REQUIRE(arc::editor::focus_selected_entity(registry, selected, camera));

    CHECK(camera.focus_point()[0] == Catch::Approx(13.0f));
    CHECK(camera.focus_point()[1] == Catch::Approx(3.0f));
    CHECK(camera.focus_point()[2] == Catch::Approx(-4.0f));

    camera.orbit(24.0f, 11.0f);
    CHECK(camera.focus_point()[0] == Catch::Approx(13.0f));
    CHECK(camera.focus_point()[1] == Catch::Approx(3.0f));
    CHECK(camera.focus_point()[2] == Catch::Approx(-4.0f));
}
