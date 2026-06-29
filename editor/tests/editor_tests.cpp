#include <arc/editor/editor_console.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/sdl_events.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("editor viewport tracks size and focus state")
{
    arc::editor::editor_viewport viewport;
    REQUIRE_FALSE(viewport.valid());

    viewport.set_size(640.8f, 480.2f);
    viewport.set_focused(true);
    viewport.set_hovered(true);

    REQUIRE(viewport.valid());
    REQUIRE(viewport.width() == 640);
    REQUIRE(viewport.height() == 480);
    REQUIRE(viewport.focused());
    REQUIRE(viewport.hovered());

    viewport.set_screen_rect(100.0f, 50.0f, 320.0f, 240.0f);
    REQUIRE(viewport.width() == 320);
    REQUIRE(viewport.height() == 240);
    REQUIRE(viewport.contains_screen_point(100.0f, 50.0f));
    REQUIRE(viewport.contains_screen_point(419.0f, 289.0f));
    REQUIRE_FALSE(viewport.contains_screen_point(420.0f, 290.0f));
    REQUIRE(viewport.local_x(124.0f) == Catch::Approx(24.0f));
    REQUIRE(viewport.local_y(74.0f) == Catch::Approx(24.0f));
}

TEST_CASE("editor console sink captures bounded log records")
{
    arc::editor::editor_console_sink sink(2);

    sink.write({ .level = arc::log_level::info, .category = "one", .message = "first" });
    sink.write({ .level = arc::log_level::warn, .category = "two", .message = "second" });
    sink.write({ .level = arc::log_level::error, .category = "three", .message = "third" });

    const auto entries = sink.entries();
    REQUIRE(entries.size() == 2);
    REQUIRE(entries[0].category == "two");
    REQUIRE(entries[1].level == arc::log_level::error);
    REQUIRE(entries[1].message == "third");
}

TEST_CASE("SDL events translate to ARC events")
{
    SDL_Event source{};
    arc::event destination{};

    source.type = SDL_EVENT_WINDOW_RESIZED;
    source.window.data1 = 1280;
    source.window.data2 = 720;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::resized);
    REQUIRE(destination.width == 1280);
    REQUIRE(destination.height == 720);

    source = {};
    source.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
    source.button.button = SDL_BUTTON_LEFT;
    source.button.x = 10.0f;
    source.button.y = 20.0f;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::mouse_button_down);
    REQUIRE(destination.button == arc::mouse_button::left);
    REQUIRE(destination.x == 10);
    REQUIRE(destination.y == 20);

    source = {};
    source.type = SDL_EVENT_KEY_DOWN;
    source.key.key = SDLK_A;
    source.key.mod = SDL_KMOD_CTRL;
    source.key.repeat = true;
    REQUIRE(arc::editor::translate_sdl_event(source, destination));
    REQUIRE(destination.type == arc::event_type::key_down);
    REQUIRE(destination.key_code == SDLK_A);
    REQUIRE(destination.modifiers == SDL_KMOD_CTRL);
    REQUIRE(destination.repeat);
}

TEST_CASE("editor selection keeps one selected entity")
{
    arc::scene::registry scene;
    auto first = scene.create();
    auto second = scene.create();
    scene.emplace<arc::scene::selection_component>(first, true);

    arc::scene::entity selected = first;
    REQUIRE(arc::editor::select_entity(scene, second, selected));
    REQUIRE(selected == second);
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(first).selected);
    REQUIRE(scene.get<arc::scene::selection_component>(second).selected);

    arc::editor::clear_selection(scene, selected);
    REQUIRE_FALSE(selected.valid());
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(first).selected);
    REQUIRE_FALSE(scene.get<arc::scene::selection_component>(second).selected);
}

TEST_CASE("editor picking hits bounded entities")
{
    arc::scene::registry scene;
    const auto entity = scene.create();
    scene.emplace<arc::scene::transform_component>(entity);
    scene.emplace<arc::scene::bounds_component>(
        entity,
        arc::geometric::box3f{ arc::geometric::point3f{ -1.0f, -1.0f, -1.0f }, arc::geometric::point3f{ 1.0f, 1.0f, 1.0f } },
        arc::geometric::box3f{},
        true);

    const arc::editor::editor_ray ray{
        .origin = arc::math::vector3f{ 0.0f, 0.0f, 5.0f },
        .direction = arc::math::vector3f{ 0.0f, 0.0f, -1.0f }
    };
    REQUIRE(arc::editor::pick_bounded_entity(scene, ray) == entity);

    float distance{};
    REQUIRE(arc::editor::intersect_ray_box(
        ray,
        arc::geometric::box3f{ arc::geometric::point3f{ -1.0f, -1.0f, -1.0f }, arc::geometric::point3f{ 1.0f, 1.0f, 1.0f } },
        distance));
    REQUIRE(distance == Catch::Approx(4.0f));
}

TEST_CASE("editor camera controller orbits pans and zooms")
{
    arc::editor::editor_camera_controller camera;
    arc::scene::transform_component transform;

    camera.focus({ 0.0f, 0.0f, 0.0f }, 2.0f);
    const float focused_distance = camera.distance();
    camera.zoom(1.0f);
    REQUIRE(camera.distance() < focused_distance);

    camera.orbit(24.0f, -12.0f);
    camera.pan(10.0f, 5.0f);
    camera.apply_to(transform);
    REQUIRE(transform.dirty);
    REQUIRE(arc::math::length(transform.position) > 0.01f);
}

TEST_CASE("editor tool shortcuts update active tool")
{
    arc::input::input_manager input;
    input.bind_action("tool.select", { .device = arc::input::input_device_type::keyboard, .code = SDLK_Q });
    input.bind_action("tool.translate", { .device = arc::input::input_device_type::keyboard, .code = SDLK_W });
    input.bind_action("tool.rotate", { .device = arc::input::input_device_type::keyboard, .code = SDLK_E });
    input.bind_action("tool.scale", { .device = arc::input::input_device_type::keyboard, .code = SDLK_R });

    auto tool = arc::editor::editor_tool::select;
    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = SDLK_W });
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::translate);

    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = SDLK_R });
    arc::editor::apply_tool_shortcuts(input, tool);
    REQUIRE(tool == arc::editor::editor_tool::scale);
}
