#pragma once

#include "android_lifecycle.h"
#include "android_vulkan_window.h"

#include <arc/render/renderer.h>

namespace arc::ecs { class world; }

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
    void present(const lifecycle_state& lifecycle, arc::ecs::world* scene);
    void shutdown() noexcept;
    [[nodiscard]] bool active() const noexcept
    {
        return renderer_ != nullptr;
    }

private:
    std::unique_ptr<arc::render::renderer> renderer_;
    std::uint64_t generation_{};
    std::uint64_t failed_generation_{};
    std::uint32_t submitted_width_{};
    std::uint32_t submitted_height_{};
    std::uint64_t frame_index_{};
};
} // namespace arc::platform::android
