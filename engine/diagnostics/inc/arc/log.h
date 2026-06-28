#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace arc
{

/**
 * @brief Severity assigned to a diagnostic log message.
 */
enum class log_level
{
    trace,
    debug,
    info,
    warn,
    error,
    fatal
};

/**
 * @brief Immutable log payload delivered to sinks.
 */
struct log_record
{
    log_level level{ log_level::info };
    std::string_view category{};
    std::string_view message{};
    std::source_location source{};
    std::chrono::system_clock::time_point timestamp{};
    std::thread::id thread_id{};
};

/**
 * @brief Destination for diagnostic log records.
 */
class log_sink
{
public:
    virtual ~log_sink();

    /**
     * @brief Write one log record.
     */
    virtual void write(const log_record& record) = 0;
};

/**
 * @brief Thread-safe fanout logger with level filtering.
 */
class logger
{
public:
    logger();

    /**
     * @brief Set the minimum severity accepted by this logger.
     */
    void set_min_level(log_level level) noexcept;

    /**
     * @brief Return the minimum severity accepted by this logger.
     */
    log_level min_level() const noexcept;

    /**
     * @brief Add a sink that receives accepted records.
     */
    void add_sink(std::shared_ptr<log_sink> sink);

    /**
     * @brief Remove all sinks from this logger.
     */
    void clear_sinks();

    /**
     * @brief Return the current number of sinks.
     */
    std::size_t sink_count() const;

    /**
     * @brief Write a fully-built record to all sinks if it passes filtering.
     */
    void write(const log_record& record);

private:
    mutable std::mutex mutex_;
    log_level min_level_{ log_level::trace };
    std::vector<std::shared_ptr<log_sink>> sinks_;
};

/**
 * @brief Return the process-wide default logger.
 */
logger& default_logger();

/**
 * @brief Replace the active process-wide logger.
 *
 * Passing `nullptr` restores the default logger.
 */
void set_logger(std::shared_ptr<logger> value);

/**
 * @brief Return the active process-wide logger.
 */
std::shared_ptr<logger> get_logger();

/**
 * @brief Add a sink to the active process-wide logger.
 */
void add_log_sink(std::shared_ptr<log_sink> sink);

/**
 * @brief Remove all sinks from the active process-wide logger.
 */
void clear_log_sinks();

/**
 * @brief Write a message to the active process-wide logger.
 */
void log(
    log_level level,
    std::string_view category,
    std::string_view message,
    std::source_location source = std::source_location::current());

/**
 * @brief Return a stable lowercase name for a log level.
 */
std::string_view to_string(log_level level) noexcept;

/**
 * @brief Log a trace message.
 */
void trace(std::string_view category, std::string_view message, std::source_location source = std::source_location::current());

/**
 * @brief Log a debug message.
 */
void debug(std::string_view category, std::string_view message, std::source_location source = std::source_location::current());

/**
 * @brief Log an informational message.
 */
void info(std::string_view category, std::string_view message, std::source_location source = std::source_location::current());

/**
 * @brief Log a warning message.
 */
void warn(std::string_view category, std::string_view message, std::source_location source = std::source_location::current());

/**
 * @brief Log an error message.
 */
void error(std::string_view category, std::string_view message, std::source_location source = std::source_location::current());

/**
 * @brief Log a fatal message.
 */
void fatal(std::string_view category, std::string_view message, std::source_location source = std::source_location::current());

} // namespace arc
