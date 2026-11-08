#include <arc/diagnostics/log.h>
#include <android/log.h>
#include <memory>
#include <string>
#include <android_native_app_glue.h>

namespace
{
constexpr const char* log_tag = "ARC";
int android_priority(arc::diagnostics::log_level level) noexcept
{
    using arc::diagnostics::log_level;
    switch (level)
    {
        case log_level::trace: return ANDROID_LOG_VERBOSE;
        case log_level::debug: return ANDROID_LOG_DEBUG;
        case log_level::info: return ANDROID_LOG_INFO;
        case log_level::warn: return ANDROID_LOG_WARN;
        case log_level::error: return ANDROID_LOG_ERROR;
        case log_level::fatal: return ANDROID_LOG_FATAL;
    }
    return ANDROID_LOG_INFO;
}
class android_log_sink final : public arc::diagnostics::log_sink
{
public:
    void write(const arc::diagnostics::log_record& record) override
    {
        std::string message;
        if (!record.category.empty())
        {
            message.reserve(record.category.size() + record.message.size() + 3);
            message += '[';
            message.append(record.category);
            message += "] ";
        }
        message.append(record.message);
        __android_log_write(android_priority(record.level), log_tag, message.c_str());
    }
};
}

extern "C" void android_main(android_app* app)
{
    app_dummy(); // Retain NativeActivity entrypoint from the NDK glue static library.
    arc::diagnostics::clear_log_sinks();
    arc::diagnostics::add_log_sink(std::make_shared<android_log_sink>());
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
