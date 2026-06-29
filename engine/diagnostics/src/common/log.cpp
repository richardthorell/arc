#include <arc/diagnostics/log.h>

#include <iostream>
#include <utility>

namespace arc
{
namespace
{

class stderr_log_sink final : public log_sink
{
public:
    void write(const log_record& record) override
    {
        std::clog << '[' << to_string(record.level) << ']';
        if (!record.category.empty())
            std::clog << '[' << record.category << ']';
        std::clog << ' ' << record.message << '\n';
    }
};

std::shared_ptr<logger> make_default_logger()
{
    auto result = std::make_shared<logger>();
    result->add_sink(std::make_shared<stderr_log_sink>());
    return result;
}

std::mutex& active_logger_mutex()
{
    static std::mutex value;
    return value;
}

std::shared_ptr<logger>& active_logger_storage()
{
    static std::shared_ptr<logger> value;
    return value;
}

std::shared_ptr<logger>& default_logger_storage()
{
    static std::shared_ptr<logger> value = make_default_logger();
    return value;
}

bool should_log(log_level record_level, log_level min_level) noexcept
{
    return static_cast<int>(record_level) >= static_cast<int>(min_level);
}

} // namespace

log_sink::~log_sink() = default;

logger::logger() = default;

void logger::set_min_level(log_level level) noexcept
{
    std::lock_guard lock(mutex_);
    min_level_ = level;
}

log_level logger::min_level() const noexcept
{
    std::lock_guard lock(mutex_);
    return min_level_;
}

void logger::add_sink(std::shared_ptr<log_sink> sink)
{
    if (!sink)
        return;

    std::lock_guard lock(mutex_);
    sinks_.push_back(std::move(sink));
}

void logger::clear_sinks()
{
    std::lock_guard lock(mutex_);
    sinks_.clear();
}

std::size_t logger::sink_count() const
{
    std::lock_guard lock(mutex_);
    return sinks_.size();
}

void logger::write(const log_record& record)
{
    std::vector<std::shared_ptr<log_sink>> sinks;
    {
        std::lock_guard lock(mutex_);
        if (!should_log(record.level, min_level_))
            return;
        sinks = sinks_;
    }

    for (const auto& sink : sinks)
        sink->write(record);
}

logger& default_logger()
{
    return *default_logger_storage();
}

void set_logger(std::shared_ptr<logger> value)
{
    std::lock_guard lock(active_logger_mutex());
    active_logger_storage() = std::move(value);
}

std::shared_ptr<logger> get_logger()
{
    std::lock_guard lock(active_logger_mutex());
    if (auto current = active_logger_storage())
        return current;
    return default_logger_storage();
}

void add_log_sink(std::shared_ptr<log_sink> sink)
{
    get_logger()->add_sink(std::move(sink));
}

void clear_log_sinks()
{
    get_logger()->clear_sinks();
}

void log(log_level level, std::string_view category, std::string_view message, std::source_location source)
{
    log_record record{};
    record.level = level;
    record.category = category;
    record.message = message;
    record.source = source;
    record.timestamp = std::chrono::system_clock::now();
    record.thread_id = std::this_thread::get_id();
    get_logger()->write(record);
}

std::string_view to_string(log_level level) noexcept
{
    switch (level)
    {
    case log_level::trace:
        return "trace";
    case log_level::debug:
        return "debug";
    case log_level::info:
        return "info";
    case log_level::warn:
        return "warn";
    case log_level::error:
        return "error";
    case log_level::fatal:
        return "fatal";
    }

    return "unknown";
}

void trace(std::string_view category, std::string_view message, std::source_location source)
{
    log(log_level::trace, category, message, source);
}

void debug(std::string_view category, std::string_view message, std::source_location source)
{
    log(log_level::debug, category, message, source);
}

void info(std::string_view category, std::string_view message, std::source_location source)
{
    log(log_level::info, category, message, source);
}

void warn(std::string_view category, std::string_view message, std::source_location source)
{
    log(log_level::warn, category, message, source);
}

void error(std::string_view category, std::string_view message, std::source_location source)
{
    log(log_level::error, category, message, source);
}

void fatal(std::string_view category, std::string_view message, std::source_location source)
{
    log(log_level::fatal, category, message, source);
}

} // namespace arc
