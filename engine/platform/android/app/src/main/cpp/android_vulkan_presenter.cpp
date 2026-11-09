#include "android_vulkan_presenter.h"
#include "android_vulkan_surface.h"

#include <arc/diagnostics/log.h>
#include <arc/framework/capabilities.h>
#include <arc/render/render_graph.h>
#include <arc/render/render_world.h>
#include <arc/render/vulkan/vulkan_backend.h>
#include <arc/scene/render_scene.h>

#include <string>
#include <utility>

namespace arc::platform::android
{
void android_vulkan_presenter::shutdown() noexcept
{
    // Renderer owns the backend, swapchain and surface; release them while the
    // Activity's borrowed ANativeWindow is still valid.
    renderer_.reset();
    generation_ = 0;
    submitted_width_ = 0;
    submitted_height_ = 0;
}

void android_vulkan_presenter::synchronize(const lifecycle_state& lifecycle, ANativeWindow* window)
{
    const auto& state = lifecycle.snapshot();
    if (!state.window_available || state.destroy_requested || !window)
    {
        shutdown();
        return;
    }
    if (renderer_ && generation_ == state.window_generation) return;
    shutdown();

    const auto binding = active_vulkan_window(lifecycle, window);
    if (!binding.window || failed_generation_ == binding.generation) return;

    arc::render::vulkan::vulkan_backend_config backend_config;
    backend_config.instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
    backend_config.create_surface = create_android_vulkan_surface;
    auto window_binding = binding;
    backend_config.surface_user_data = &window_binding;
    backend_config.viewport_output = arc::render::viewport_output_type::native_window;
    backend_config.force_disable_optional_features = true;

    auto created = arc::render::vulkan::create_vulkan_backend(backend_config);
    if (!created)
    {
        failed_generation_ = binding.generation;
        arc::diagnostics::error("Android", created.error().message);
        return;
    }
    auto backend = std::move(created).value();
    const auto& capabilities = backend->capabilities();
    arc::diagnostics::info("Android", std::string("Vulkan adapter: ") + capabilities.adapter_name);
    arc::diagnostics::info("Android", std::string("Vulkan API ") +
                                      std::to_string(capabilities.api_major) + "." +
                                      std::to_string(capabilities.api_minor));
    if (!capabilities.presentation || !capabilities.graphics_queue || !capabilities.compute_queue)
    {
        arc::diagnostics::error("Android", "Vulkan adapter lacks required graphics/compute/presentation queues");
        failed_generation_ = binding.generation;
        return;
    }

    arc::render::renderer_config renderer_config;
    renderer_config.quality = arc::render::render_quality_tier::low;
    renderer_config.path = arc::render::render_path::forward_plus;
    renderer_config.force_disable_optional_features = true;
    renderer_config.force_disable_gpu_driven = true;
    renderer_config.force_disable_async_compute = true;
    renderer_config.force_disable_dynamic_gi = true;
    renderer_config.force_disable_temporal = true;
    renderer_config.enable_dynamic_resolution = false;

    renderer_ = std::make_unique<arc::render::renderer>(renderer_config);
    renderer_->set_platform_capabilities({.family = arc::framework::platform_family::android,
                                          .form_factor = arc::framework::device_form_factor::handheld,
                                          .window_system = true,
                                          .high_dpi = true});
    renderer_->set_backend(std::move(backend));
    generation_ = binding.generation;
    arc::diagnostics::info("Android", "ARC Vulkan renderer attached to native window");
}

void android_vulkan_presenter::present(const lifecycle_state& lifecycle, arc::ecs::world* scene)
{
    if (!renderer_ || !lifecycle.can_present()) return;
    const auto& state = lifecycle.snapshot();
    if (state.window_generation != generation_)
    {
        shutdown();
        return;
    }

    bool submitted_scene = false;
    if (scene)
    {
        // Render extraction owns camera, scene geometry and light event
        // generation; the Android platform must not synthesize render items.
        const auto extracted = arc::scene::render_scene(*scene, *renderer_, state.width, state.height,
                                                        arc::render::render_mode::shaded,
                                                        arc::render::mesh_visualization_mode::standard,
                                                        arc::render::editor_overlay_mode::none, false);
        if (extracted.camera_found)
        {
            const auto graph = arc::render::make_scene_draw_graph(
                "viewport", renderer_->resolved_config(), false, extracted.environment);
            const auto submitted = renderer_->render_frame(++frame_index_, graph);
            if (!submitted)
            {
                arc::diagnostics::error("Android", submitted.error().message);
                failed_generation_ = generation_;
                shutdown();
                return;
            }
            submitted_scene = true;
        }
    }

    if (!submitted_scene && (state.width != submitted_width_ || state.height != submitted_height_))
    {
        const auto graph = arc::render::make_clear_present_graph("viewport");
        arc::render::render_graph_compile_options options;
        options.output_extent = {.width = state.width, .height = state.height};
        options.render_extent = options.output_extent;
        options.frame_index = frame_index_;
        auto compiled = graph.compile(options);
        if (!compiled)
        {
            arc::diagnostics::error("Android", compiled.error().message);
            failed_generation_ = generation_;
            shutdown();
            return;
        }
        arc::render::render_frame_packet packet;
        packet.frame_index = ++frame_index_;
        const auto submitted = renderer_->backend()->submit(packet, compiled.value());
        if (!submitted)
        {
            arc::diagnostics::error("Android", submitted.error().message);
            failed_generation_ = generation_;
            shutdown();
            return;
        }
        submitted_width_ = state.width;
        submitted_height_ = state.height;
        arc::diagnostics::info("Android", "ARC clear/present fallback graph submitted");
    }

    const auto result = renderer_->backend()->present_surface_frame(state.width, state.height);
    if (result || result.error().code == arc::render::surface_frame_error_code::out_of_date) return;
    arc::diagnostics::error("Android", result.error().message);
    failed_generation_ = generation_;
    shutdown();
}
} // namespace arc::platform::android
