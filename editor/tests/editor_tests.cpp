#include <arc/editor/editor_console.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/sdl_events.h>

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
