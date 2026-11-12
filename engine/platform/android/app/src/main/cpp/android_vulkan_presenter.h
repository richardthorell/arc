#pragma once

#include "android_lifecycle.h"
#include "android_vulkan_window.h"

#include <arc/render/renderer.h>
#include <arc/scene/cooked_mesh_hydration.h>

namespace arc::ecs
{
class world;
}

#include <cstdint>
#include <memory>

namespace arc::platform::android
{
class android_runtime_bridge;

// All operations run on the NativeActivity event-loop thread. The Vulkan
// backend and swapchain die before the borrowed native window is released.
class android_vulkan_presenter
{
public:
    void synchronize(const lifecycle_state& lifecycle, ANativeWindow* window);
    void present(const lifecycle_state& lifecycle, android_runtime_bridge& runtime);
    void shutdown() noexcept;
    [[nodiscard]] bool active() const noexcept
    {
        return renderer_ != nullptr;
    }

private:
    [[nodiscard]] bool hydrate_scene(android_runtime_bridge& runtime);
    std::unique_ptr<arc::render::renderer> renderer_;
    arc::ecs::world* bound_scene_{};
    arc::scene::cooked_mesh_residencies mesh_residency_;
    std::uint64_t hydrated_generation_{};
    std::uint64_t generation_{};
    std::uint64_t failed_generation_{};
    std::uint32_t submitted_width_{};
    std::uint32_t submitted_height_{};
    std::uint64_t frame_index_{};
};
} // namespace arc::platform::android
