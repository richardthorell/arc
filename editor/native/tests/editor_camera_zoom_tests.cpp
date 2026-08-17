#include <arc/editor/editor_interaction.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("editor camera wheel zoom stops before crossing the focused pivot")
{
    arc::editor::editor_camera_controller camera;
    camera.focus({0.0f, 0.0f, 0.0f}, 2.0f);

    arc::scene::transform_component before;
    camera.apply_to(before);
    const auto focus = camera.focus_point();
    const auto initial_from_focus = arc::math::sub(before.position, focus);
    const float orbit_distance = camera.distance();

    for (int index = 0; index < 256; ++index) camera.zoom(1.0f);

    arc::scene::transform_component zoomed_in;
    camera.apply_to(zoomed_in);
    const auto final_from_focus = arc::math::sub(zoomed_in.position, focus);
    CHECK(arc::math::length(final_from_focus) == Catch::Approx(0.35f).margin(0.0001f));
    CHECK(arc::math::dot(initial_from_focus, final_from_focus) > 0.0f);

    // Dolly keeps the persistent orbit radius untouched until an orbit gesture
    // reconciles it with the current camera-to-focus distance.
    CHECK(camera.distance() == Catch::Approx(orbit_distance));
    camera.orbit(0.0f, 0.0f);
    CHECK(camera.distance() == Catch::Approx(0.35f).margin(0.0001f));
}

TEST_CASE("editor camera wheel zoom clamps pathological wheel deltas")
{
    arc::editor::editor_camera_controller zoom_in;
    zoom_in.focus({0.0f, 0.0f, 0.0f}, 2.0f);
    zoom_in.zoom(100000.0f);
    arc::scene::transform_component near_focus;
    zoom_in.apply_to(near_focus);
    CHECK(arc::math::length(arc::math::sub(near_focus.position, zoom_in.focus_point())) ==
          Catch::Approx(0.35f).margin(0.0001f));

    arc::editor::editor_camera_controller zoom_out;
    zoom_out.focus({0.0f, 0.0f, 0.0f}, 2.0f);
    arc::scene::transform_component before;
    zoom_out.apply_to(before);
    zoom_out.zoom(-100000.0f);
    arc::scene::transform_component after;
    zoom_out.apply_to(after);
    CHECK(arc::math::length(arc::math::sub(after.position, before.position)) == Catch::Approx(12.0f).margin(0.0001f));
}
