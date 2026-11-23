#include "android_lifecycle.h"
#include "android_log_sink.h"
#include "android_package_assets.h"
#include "android_project_runtime_module.h"
#include "android_runtime_bridge.h"
#include "android_vulkan_window.h"
#include "android_vulkan_presenter.h"

#include <arc/diagnostics/log.h>
#include <arc/http/android_http.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <jni.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace
{
using arc::platform::android::lifecycle_event;
using arc::platform::android::lifecycle_state;

/** Only non-secret development settings are read from Activity Intent extras.
 * Untrusted/invalid overrides fail closed in the shared C9.3 validator and
 * Shipping packages reject all such overrides. No URLs are logged here.
 */
struct content_intent
{
    bool valid{true};
    std::optional<std::string> server;
    std::optional<std::string> mode;
};

bool read_string_extra(JNIEnv* env, jobject intent, jmethodID getter, const char* key,
                       std::optional<std::string>& target)
{
    const auto name = env->NewStringUTF(key);
    if (!name) return false;
    const auto value = static_cast<jstring>(env->CallObjectMethod(intent, getter, name));
    env->DeleteLocalRef(name);
    if (env->ExceptionCheck()) return false;
    if (!value) return true; // Absent extra, not an empty supplied value.
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars)
    {
        env->DeleteLocalRef(value);
        return false;
    }
    target = chars;
    env->ReleaseStringUTFChars(value, chars);
    env->DeleteLocalRef(value);
    return true;
}

content_intent read_content_intent(ANativeActivity* activity)
{
    content_intent result;
    if (!activity || !activity->vm || !activity->clazz) return result;

    JNIEnv* env = nullptr;
    bool attached = false;
    const auto status = activity->vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (status == JNI_EDETACHED)
    {
        if (activity->vm->AttachCurrentThread(&env, nullptr) != JNI_OK)
            return {.valid = false};
        attached = true;
    }
    else if (status != JNI_OK)
        return {.valid = false};

    const auto cls = env->GetObjectClass(activity->clazz);
    const auto get_intent = cls ? env->GetMethodID(cls, "getIntent", "()Landroid/content/Intent;") : nullptr;
    const auto intent = get_intent ? env->CallObjectMethod(activity->clazz, get_intent) : nullptr;
    const auto intent_cls = intent ? env->GetObjectClass(intent) : nullptr;
    const auto get_extra = intent_cls ? env->GetMethodID(
        intent_cls, "getStringExtra", "(Ljava/lang/String;)Ljava/lang/String;") : nullptr;
    result.valid = get_extra &&
                   read_string_extra(env, intent, get_extra, "org.arcengine.CONTENT_SERVER", result.server) &&
                   read_string_extra(env, intent, get_extra, "org.arcengine.CONTENT_MODE", result.mode) &&
                   !env->ExceptionCheck();
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (intent_cls) env->DeleteLocalRef(intent_cls);
    if (intent) env->DeleteLocalRef(intent);
    if (cls) env->DeleteLocalRef(cls);
    if (attached) activity->vm->DetachCurrentThread();
    return result;
}

struct android_host_context
{
    lifecycle_state lifecycle;
    ANativeWindow* attached_window{}; // Borrowed from native_app_glue; never release here.
    arc::platform::android::android_vulkan_presenter presenter;
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
            host.presenter.shutdown();
            host.lifecycle.apply(lifecycle_event::destroy);
            break;
        default:
            break;
    }
    // Reconcile surface ownership while the borrowed window is still valid.
    host.presenter.synchronize(host.lifecycle, host.attached_window);
}
} // namespace

extern "C" void android_main(android_app* app)
{
    app_dummy(); // Retain NativeActivity entrypoint from the NDK glue static library.
    install_android_log_sink();
    android_host_context host;
    app->userData = &host;
    app->onAppCmd = on_app_command;

    // Register the app ClassLoader for arc-http's Android TLS bridge before
    // runtime jobs may request any remote cooked artifacts. The guard is
    // declared before runtime so its teardown runs *after* jobs have drained.
    const bool http_ready =
        app->activity && arc::http::register_android_http_transport(app->activity->vm, app->activity->clazz);
    if (!http_ready) arc::diagnostics::warn("Android", "Platform HTTPS bridge unavailable; remote content disabled");
    struct http_registration_guard
    {
        ~http_registration_guard()
        {
            arc::http::unregister_android_http_transport();
        }
    } http_guard;
    (void)http_guard;

    // These paths are owned by Android. A3 will feed them to the shared launch
    // context; the asset manager is an Android asset-provider handle, not a path.
    if (app->activity && app->activity->internalDataPath)
        arc::diagnostics::info("Android", "App-private writable storage available");
    arc::platform::android::package_asset_source package_assets(
        app->activity ? static_cast<void*>(app->activity->assetManager) : nullptr);
    if (package_assets.available())
        arc::diagnostics::info("Android", "APK asset manager available (no extraction to writable storage)");

    // A staged game must contain its project-native module. The plain host
    // fixture without cooked content remains useful for Vulkan/NDK smoke.
    std::uint64_t manifest_size = 0;
    const bool packaged_game = package_assets.file_size("arc-distribution.json", manifest_size);
    arc::platform::android::android_project_runtime_module project_module;
    const bool module_ready = project_module.load(packaged_game);

    // Keep the project module loaded for the full runtime process. It is not
    // unloaded when Android destroys and recreates a native window.
    arc::platform::android::android_runtime_bridge runtime(package_assets,
                                                           app->activity ? app->activity->internalDataPath : nullptr);
    const auto extras = read_content_intent(app->activity);
    const auto view = [](const std::optional<std::string>& value) -> std::optional<std::string_view>
    {
        return value ? std::optional<std::string_view>(*value) : std::nullopt;
    };
    const auto selected = arc::project::parse_android_content_metadata(view(extras.server), view(extras.mode));
    if (!extras.valid || !selected)
        arc::diagnostics::error("Android", "Invalid Development content Intent metadata");
    const bool booted = module_ready && extras.valid && selected &&
                        runtime.boot(project_module.project_guid(), selected.value()) && project_module.start();
    if (!booted) arc::diagnostics::warn("Android", "Host remains alive; packaged startup did not complete");

    arc::diagnostics::info("Android", "Native host started (A3 shared boot)");

    while (!app->destroyRequested && !host.lifecycle.snapshot().destroy_requested)
    {
        android_poll_source* source = nullptr;
        // Block when inactive; poll without delay while presenting so every
        // Android lifecycle command is processed on the render thread.
        const int timeout = host.presenter.active() && host.lifecycle.can_present() ? 0 : -1;
        const auto result = ALooper_pollOnce(timeout, nullptr, nullptr, reinterpret_cast<void**>(&source));
        if (result == ALOOPER_POLL_ERROR)
        {
            arc::diagnostics::error("Android", "Native host event loop failed");
            break;
        }
        if (source) source->process(app, source);
        if (!app->destroyRequested) host.presenter.present(host.lifecycle, runtime);
    }

    host.presenter.shutdown();
    host.lifecycle.apply(lifecycle_event::destroy);
    project_module.shutdown();
    app->onAppCmd = nullptr;
    app->userData = nullptr;
    arc::diagnostics::info("Android", "Native host stopped");
}
