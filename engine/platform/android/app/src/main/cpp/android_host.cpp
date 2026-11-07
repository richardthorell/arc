#include <android/log.h>
#include <android_native_app_glue.h>

namespace
{
constexpr const char* log_tag = "ARC.Android";
}

extern "C" void android_main(android_app* app)
{
    app_dummy(); // Retain NativeActivity entrypoint from the NDK glue static library.
    __android_log_print(ANDROID_LOG_INFO, log_tag, "Native host started (A1 scaffold)");

    // A1 keeps the NativeActivity process alive without starting ARC's runtime,
    // renderer, or allocating a Vulkan surface. A2 will add lifecycle callbacks.
    while (!app->destroyRequested)
    {
        int events = 0;
        android_poll_source* source = nullptr;
        const int result = ALooper_pollOnce(-1, nullptr, &events,
                                            reinterpret_cast<void**>(&source));
        if (result == ALOOPER_POLL_ERROR)
        {
            __android_log_print(ANDROID_LOG_ERROR, log_tag, "Native host event loop failed");
            break;
        }
        if (source != nullptr)
            source->process(app, source);
    }
    __android_log_print(ANDROID_LOG_INFO, log_tag, "Native host stopped");
}
