#include "android_lifecycle.h"

namespace arc::platform::android
{
void lifecycle_state::apply(lifecycle_event event, std::uint32_t width, std::uint32_t height) noexcept
{
    if (state_.destroy_requested) return;

    switch (event)
    {
        case lifecycle_event::start:
            state_.started = true;
            break;
        case lifecycle_event::resume:
            state_.resumed = true;
            break;
        case lifecycle_event::pause:
            state_.resumed = false;
            break;
        case lifecycle_event::stop:
            state_.started = false;
            state_.resumed = false;
            break;
        case lifecycle_event::gain_focus:
            state_.focused = true;
            break;
        case lifecycle_event::lose_focus:
            state_.focused = false;
            break;
        case lifecycle_event::window_created:
            // APP_CMD_INIT_WINDOW can be reissued for the same native window.
            if (!state_.window_available) ++state_.window_generation;
            state_.window_available = true;
            state_.width = width;
            state_.height = height;
            break;
        case lifecycle_event::window_destroyed:
            state_.window_available = false;
            state_.width = 0;
            state_.height = 0;
            break;
        case lifecycle_event::window_resized:
            if (state_.window_available)
            {
                state_.width = width;
                state_.height = height;
            }
            break;
        case lifecycle_event::config_changed:
        case lifecycle_event::low_memory:
            // Host callbacks notify the future runtime adapter; no implicit reset.
            break;
        case lifecycle_event::destroy:
            state_.destroy_requested = true;
            state_.started = false;
            state_.resumed = false;
            state_.focused = false;
            state_.window_available = false;
            state_.width = 0;
            state_.height = 0;
            break;
    }
}
} // namespace arc::platform::android
