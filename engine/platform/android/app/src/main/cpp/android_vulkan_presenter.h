#pragma once

#include "android_lifecycle.h"
#include "android_vulkan_window.h"

#include <arc/render/render_backend.h>

#include <cstdint>
#include <memory>

namespace arc::platform::android
{
// All operations run on the NativeActivity event-loop thread. The Vulkan
// backend and swapchain die before the borrowed native window is released.
class android_vulkan_presenter
{
public:
    void synchronize(const lifecycle_state& lifecycle, ANativeWindow* window);
    void present(const lifecycle_state& lifecycle);
    void shutdown() noexcept;
    [[nodiscard]] bool active() const noexcept
    {
        return backend_ != nullptr;
    }

private:
    std::unique_ptr<arc::render::render_backend> backend_;
    std::uint64_t generation_{};
    std::uint64_t failed_generation_{};
};
} // namespace arc::platform::android
