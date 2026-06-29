#pragma once

#include <arc/log.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace arc::editor
{

/**
 * @brief Log entry captured for display in the editor console.
 */
struct console_log_entry
{
    log_level level{ log_level::info };
    std::string category;
    std::string message;
    std::chrono::system_clock::time_point timestamp{};
};

/**
 * @brief Thread-safe log sink used by the editor console panel.
 */
class editor_console_sink final : public log_sink
{
public:
    explicit editor_console_sink(std::size_t max_entries = 1000);

    /**
     * @brief Capture one diagnostic record.
     */
    void write(const log_record& record) override;

    /**
     * @brief Return a snapshot of captured entries.
     */
    std::vector<console_log_entry> entries() const;

    /**
     * @brief Remove all captured entries.
     */
    void clear();

private:
    std::size_t max_entries_{};
    mutable std::mutex mutex_;
    std::vector<console_log_entry> entries_;
};

using editor_console_sink_ptr = std::shared_ptr<editor_console_sink>;

} // namespace arc::editor
