#pragma once

#include "android_lifecycle.h"

#include <cstdint>

struct ANativeWindow;

namespace arc::platform::android
{
// Borrowed from native_app_glue; only valid until the next window teardown.
struct vulkan_native_window
{
    ANativeWindow* window{};
    std::uint64_t generation{};
};

[[nodiscard]] inline vulkan_native_window active_vulkan_window(const lifecycle_state& lifecycle,
                                                               ANativeWindow* attached_window) noexcept
{
    if (!lifecycle.can_present() || !attached_window) return {};
    return {.window = attached_window, .generation = lifecycle.snapshot().window_generation};
}
} // namespace arc::platform::android
