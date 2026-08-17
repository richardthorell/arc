#include <arc/editor/editor_interaction.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("editor camera wheel zoom approaches focus without crossing it")
{
    arc::editor::editor_camera_controller camera;
    camera.focus({0.0f, 0.0f, 0.0f}, 2.0f);

    arc::scene::transform_component before;
    camera.apply_to(before);
    const auto focus = camera.focus_point();
    const auto initial_from_focus = arc::math::sub(before.position, focus);
    const float initial_distance = arc::math::length(initial_from_focus);

    camera.zoom(1.0f);
    arc::scene::transform_component after_one_step;
    camera.apply_to(after_one_step);
    CHECK(camera.distance() < initial_distance);
    CHECK(arc::math::length(arc::math::sub(after_one_step.position, focus)) == Catch::Approx(camera.distance()));

    for (int index = 0; index < 256; ++index) camera.zoom(1.0f);

    arc::scene::transform_component zoomed_in;
    camera.apply_to(zoomed_in);
    const auto final_from_focus = arc::math::sub(zoomed_in.position, focus);
    CHECK(camera.distance() == Catch::Approx(0.35f).margin(0.0001f));
    CHECK(arc::math::length(final_from_focus) == Catch::Approx(0.35f).margin(0.0001f));
    CHECK(arc::math::dot(initial_from_focus, final_from_focus) > 0.0f);
}

TEST_CASE("editor camera wheel zoom out remains bounded")
{
    arc::editor::editor_camera_controller camera;
    camera.focus({0.0f, 0.0f, 0.0f}, 2.0f);

    for (int index = 0; index < 256; ++index) camera.zoom(-1.0f);

    arc::scene::transform_component zoomed_out;
    camera.apply_to(zoomed_out);
    CHECK(camera.distance() == Catch::Approx(500.0f).margin(0.001f));
    CHECK(arc::math::length(arc::math::sub(zoomed_out.position, camera.focus_point())) ==
          Catch::Approx(500.0f).margin(0.001f));
}

TEST_CASE("editor camera wheel zoom clamps pathological input")
{
    arc::editor::editor_camera_controller camera;
    camera.focus({0.0f, 0.0f, 0.0f}, 2.0f);

    camera.zoom(100000.0f);
    CHECK(camera.distance() >= 0.35f);

    camera.zoom(-100000.0f);
    CHECK(camera.distance() <= 500.0f);
}
