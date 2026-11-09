#include "../app/src/main/cpp/android_vulkan_window.h"

#include <cassert>
#include <cstdint>

using arc::platform::android::active_vulkan_window;
using arc::platform::android::lifecycle_event;
using arc::platform::android::lifecycle_state;

int main()
{
    lifecycle_state state;
    auto* window = reinterpret_cast<ANativeWindow*>(static_cast<std::uintptr_t>(1));
    assert(active_vulkan_window(state, window).window == nullptr);
    state.apply(lifecycle_event::start);
    state.apply(lifecycle_event::resume);
    state.apply(lifecycle_event::gain_focus);
    state.apply(lifecycle_event::window_created, 640, 480);
    auto first = active_vulkan_window(state, window);
    assert(first.window == window && first.generation == 1);
    state.apply(lifecycle_event::pause);
    assert(active_vulkan_window(state, window).window == nullptr);
    state.apply(lifecycle_event::resume);
    state.apply(lifecycle_event::window_destroyed);
    assert(active_vulkan_window(state, window).window == nullptr);
    state.apply(lifecycle_event::window_created, 480, 640);
    auto second = active_vulkan_window(state, window);
    assert(second.window == window && second.generation == 2);
    state.apply(lifecycle_event::destroy);
    assert(active_vulkan_window(state, window).window == nullptr);
}
