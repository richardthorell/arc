#include "android_lifecycle.h"
#include "android_log_sink.h"
#include "android_package_assets.h"
#include "android_runtime_bridge.h"
#include "android_vulkan_window.h"

#include <arc/diagnostics/log.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>

#include <cstdint>

namespace
{
using arc::platform::android::lifecycle_event;
using arc::platform::android::lifecycle_state;

struct android_host_context
{
    lifecycle_state lifecycle;
    ANativeWindow* attached_window{}; // Borrowed from native_app_glue; never release here.
    std::uint64_t last_presentable_generation{};
};

std::uint32_t dimension(int value) noexcept
{
    return value > 0 ? static_cast<std::uint32_t>(value) : 0;
}

void update_window(android_app* app, android_host_context& host, lifecycle_event event)
{
    const auto width = app->window ? dimension(ANativeWindow_getWidth(app->window)) : 0;
    const auto height = app->window ? dimension(ANativeWindow_getHeight(app->window)) : 0;
    host.lifecycle.apply(event, width, height);
    const auto& state = host.lifecycle.snapshot();
    arc::diagnostics::info("Android", event == lifecycle_event::window_created ? "Native surface attached"
                                                                               : "Native surface size changed");
    // The Vulkan backend must recreate its surface for each new generation.
    // Surface loss does not tear down the packaged runtime.
    (void)state;
}

void on_app_command(android_app* app, std::int32_t command)
{
    auto& host = *static_cast<android_host_context*>(app->userData);
    switch (command)
    {
        case APP_CMD_START:
            host.lifecycle.apply(lifecycle_event::start);
            arc::diagnostics::info("Android", "Activity started");
            break;
        case APP_CMD_RESUME:
            host.lifecycle.apply(lifecycle_event::resume);
            arc::diagnostics::info("Android", "Activity resumed");
            break;
        case APP_CMD_PAUSE:
            host.lifecycle.apply(lifecycle_event::pause);
            arc::diagnostics::info("Android", "Activity paused");
            break;
        case APP_CMD_STOP:
            host.lifecycle.apply(lifecycle_event::stop);
            arc::diagnostics::info("Android", "Activity stopped");
            break;
        case APP_CMD_GAINED_FOCUS:
            host.lifecycle.apply(lifecycle_event::gain_focus);
            break;
        case APP_CMD_LOST_FOCUS:
            host.lifecycle.apply(lifecycle_event::lose_focus);
            break;
        case APP_CMD_INIT_WINDOW:
            if (host.attached_window != app->window)
            {
                if (host.attached_window) host.lifecycle.apply(lifecycle_event::window_destroyed);
                host.attached_window = app->window;
                update_window(app, host, lifecycle_event::window_created);
            }
            else
                update_window(app, host, lifecycle_event::window_resized);
            break;
        case APP_CMD_TERM_WINDOW:
            host.attached_window = nullptr;
            host.lifecycle.apply(lifecycle_event::window_destroyed);
            arc::diagnostics::info("Android", "Native surface detached");
            break;
        case APP_CMD_WINDOW_RESIZED:
        case APP_CMD_CONTENT_RECT_CHANGED:
            update_window(app, host, lifecycle_event::window_resized);
            break;
        case APP_CMD_CONFIG_CHANGED:
            host.lifecycle.apply(lifecycle_event::config_changed);
            arc::diagnostics::info("Android", "Device configuration changed");
            break;
        case APP_CMD_LOW_MEMORY:
            host.lifecycle.apply(lifecycle_event::low_memory);
            arc::diagnostics::warn("Android", "Android reported low memory");
            break;
        case APP_CMD_DESTROY:
            host.lifecycle.apply(lifecycle_event::destroy);
            break;
        default:
            break;
    }
    const auto binding = arc::platform::android::active_vulkan_window(host.lifecycle, host.attached_window);
    if (binding.window && binding.generation != host.last_presentable_generation)
    {
        host.last_presentable_generation = binding.generation;
        arc::diagnostics::info("Android", "Vulkan-capable native window generation available");
    }
}
} // namespace

extern "C" void android_main(android_app* app)
{
    app_dummy(); // Retain NativeActivity entrypoint from the NDK glue static library.
    install_android_log_sink();
    android_host_context host;
    app->userData = &host;
    app->onAppCmd = on_app_command;

    // These paths are owned by Android. A3 will feed them to the shared launch
    // context; the asset manager is an Android asset-provider handle, not a path.
    if (app->activity && app->activity->internalDataPath)
        arc::diagnostics::info("Android", "App-private writable storage available");
    arc::platform::android::package_asset_source package_assets(
        app->activity ? static_cast<void*>(app->activity->assetManager) : nullptr);
    if (package_assets.available())
        arc::diagnostics::info("Android", "APK asset manager available (no extraction to writable storage)");

    // Keep the VFS, job scheduler and validated launch alive across window
    // destruction/recreation. Only the Android process owns this lifetime.
    arc::platform::android::android_runtime_bridge runtime(package_assets,
                                                           app->activity ? app->activity->internalDataPath : nullptr);
    const bool booted = runtime.boot();
    if (!booted) arc::diagnostics::warn("Android", "Host remains alive; no staged distribution was loaded");

    arc::diagnostics::info("Android", "Native host started (A3 shared boot)");

    while (!app->destroyRequested && !host.lifecycle.snapshot().destroy_requested)
    {
        android_poll_source* source = nullptr;
        // NativeActivity callbacks are delivered on this thread; block when
        // inactive. Rendering (#402) will later use a bounded active wait.
        const auto result = ALooper_pollOnce(-1, nullptr, nullptr, reinterpret_cast<void**>(&source));
        if (result == ALOOPER_POLL_ERROR)
        {
            arc::diagnostics::error("Android", "Native host event loop failed");
            break;
        }
        if (source) source->process(app, source);
    }

    host.lifecycle.apply(lifecycle_event::destroy);
    app->onAppCmd = nullptr;
    app->userData = nullptr;
    arc::diagnostics::info("Android", "Native host stopped");
}
