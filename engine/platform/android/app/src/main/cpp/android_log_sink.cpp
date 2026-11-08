#include "android_log_sink.h"

#include <arc/diagnostics/log.h>
#include <android/log.h>

#include <algorithm>
#include <climits>
#include <memory>

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

// printf's precision parameter is an int; clamp string_view lengths before
// passing them to %.*s, without copying or allocating temporary strings.
int precision(std::size_t length) noexcept
{
    return static_cast<int>(std::min(length, static_cast<std::size_t>(INT_MAX)));
}

class android_log_sink final : public arc::diagnostics::log_sink
{
public:
    void write(const arc::diagnostics::log_record& record) override
    {
        const auto priority = android_priority(record.level);
        if (record.category.empty())
        {
            __android_log_print(priority, log_tag, "%.*s",
                                precision(record.message.size()), record.message.data());
        }
        else
        {
            __android_log_print(priority, log_tag, "[%.*s] %.*s",
                                precision(record.category.size()), record.category.data(),
                                precision(record.message.size()), record.message.data());
        }
    }
};
} // namespace

void install_android_log_sink()
{
    arc::diagnostics::clear_log_sinks();
    arc::diagnostics::add_log_sink(std::make_shared<android_log_sink>());
}
