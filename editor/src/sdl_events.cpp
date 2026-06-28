#include <arc/editor/sdl_events.h>

#include <SDL3/SDL_mouse.h>

namespace arc::editor
{
namespace
{

mouse_button translate_mouse_button(unsigned char button) noexcept
{
    switch (button)
    {
    case SDL_BUTTON_LEFT:
        return mouse_button::left;
    case SDL_BUTTON_RIGHT:
        return mouse_button::right;
    case SDL_BUTTON_MIDDLE:
        return mouse_button::middle;
    case SDL_BUTTON_X1:
        return mouse_button::x1;
    case SDL_BUTTON_X2:
        return mouse_button::x2;
    default:
        return mouse_button::unknown;
    }
}

} // namespace

bool translate_sdl_event(const SDL_Event& source, event& destination) noexcept
{
    destination = {};

    switch (source.type)
    {
    case SDL_EVENT_QUIT:
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        destination.type = event_type::close_requested;
        return true;

    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        destination.type = event_type::resized;
        destination.width = static_cast<std::uint32_t>(source.window.data1);
        destination.height = static_cast<std::uint32_t>(source.window.data2);
        return true;

    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        destination.type = event_type::focus_gained;
        return true;

    case SDL_EVENT_WINDOW_FOCUS_LOST:
        destination.type = event_type::focus_lost;
        return true;

    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        destination.type = source.type == SDL_EVENT_KEY_DOWN ? event_type::key_down : event_type::key_up;
        destination.key_code = static_cast<int>(source.key.key);
        destination.modifiers = static_cast<std::uint32_t>(source.key.mod);
        destination.repeat = source.key.repeat;
        return true;

    case SDL_EVENT_MOUSE_MOTION:
        destination.type = event_type::mouse_moved;
        destination.x = static_cast<int>(source.motion.x);
        destination.y = static_cast<int>(source.motion.y);
        return true;

    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        destination.type = source.type == SDL_EVENT_MOUSE_BUTTON_DOWN ? event_type::mouse_button_down : event_type::mouse_button_up;
        destination.button = translate_mouse_button(source.button.button);
        destination.x = static_cast<int>(source.button.x);
        destination.y = static_cast<int>(source.button.y);
        return true;

    case SDL_EVENT_MOUSE_WHEEL:
        destination.type = event_type::mouse_wheel;
        destination.wheel_delta = source.wheel.y;
        destination.x = static_cast<int>(source.wheel.mouse_x);
        destination.y = static_cast<int>(source.wheel.mouse_y);
        return true;

    default:
        return false;
    }
}

} // namespace arc::editor
