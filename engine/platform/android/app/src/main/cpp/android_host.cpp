#include "android_log_sink.h"

#include <arc/diagnostics/log.h>
#include <android_native_app_glue.h>

extern "C" void android_main(android_app* app)
{
    app_dummy(); // Retain NativeActivity entrypoint from the NDK glue static library.
    install_android_log_sink();
    arc::diagnostics::info("Android", "Native host started (A1 scaffold)");

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
            arc::diagnostics::error("Android", "Native host event loop failed");
            break;
        }
        if (source != nullptr)
            source->process(app, source);
    }
    arc::diagnostics::info("Android", "Native host stopped");
}
