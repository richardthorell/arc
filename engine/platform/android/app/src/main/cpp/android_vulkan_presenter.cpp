#include "android_vulkan_presenter.h"
#include "android_vulkan_surface.h"

#include <arc/diagnostics/log.h>
#include <arc/render/vulkan/vulkan_backend.h>

#include <utility>

namespace arc::platform::android
{
void android_vulkan_presenter::shutdown() noexcept
{
    // Renderer destructor destroys GPU presentation resources and VkSurfaceKHR
    // while the Activity's ANativeWindow is still valid.
    backend_.reset();
    generation_ = 0;
}

void android_vulkan_presenter::synchronize(const lifecycle_state& lifecycle, ANativeWindow* window)
{
    const auto& state = lifecycle.snapshot();
    if (!state.window_available || state.destroy_requested || !window)
    {
        shutdown();
        return;
    }
    if (backend_ && generation_ == state.window_generation) return;
    shutdown();

    const auto binding = active_vulkan_window(lifecycle, window);
    if (!binding.window || failed_generation_ == binding.generation) return;

    arc::render::vulkan::vulkan_backend_config config;
    config.instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
    config.create_surface = create_android_vulkan_surface;
    // Backend invokes the callback synchronously and does not retain user_data.
    auto window_binding = binding;
    config.surface_user_data = &window_binding;
    config.viewport_output = arc::render::viewport_output_type::native_window;
    config.force_disable_optional_features = true;

    auto created = arc::render::vulkan::create_vulkan_backend(config);
    if (!created)
    {
        failed_generation_ = binding.generation;
        arc::diagnostics::error("Android", created.error().message);
        return;
    }
    backend_ = std::move(created).value();
    generation_ = binding.generation;
    arc::diagnostics::info("Android", "ARC Vulkan renderer attached to native window");
}

void android_vulkan_presenter::present(const lifecycle_state& lifecycle)
{
    if (!backend_ || !lifecycle.can_present()) return;
    const auto& state = lifecycle.snapshot();
    if (state.window_generation != generation_)
    {
        shutdown();
        return;
    }
    const auto result = backend_->present_surface_frame(state.width, state.height);
    if (result) return;

    if (result.error().code == arc::render::surface_frame_error_code::out_of_date)
        return; // Existing swapchain automatically retries after resize.

    arc::diagnostics::error("Android", result.error().message);
    failed_generation_ = generation_;
    shutdown(); // Avoid repeated device/surface failures until a new window.
}
} // namespace arc::platform::android
