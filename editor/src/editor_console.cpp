#include <arc/editor/editor_console.h>

namespace arc::editor
{

editor_console_sink::editor_console_sink(std::size_t max_entries)
    : max_entries_(max_entries == 0 ? 1 : max_entries)
{
}

void editor_console_sink::write(const log_record& record)
{
    std::lock_guard lock(mutex_);
    if (entries_.size() == max_entries_)
        entries_.erase(entries_.begin());

    entries_.push_back({
        .level = record.level,
        .category = std::string(record.category),
        .message = std::string(record.message),
        .timestamp = record.timestamp
    });
}

std::vector<console_log_entry> editor_console_sink::entries() const
{
    std::lock_guard lock(mutex_);
    return entries_;
}

void editor_console_sink::clear()
{
    std::lock_guard lock(mutex_);
    entries_.clear();
}

} // namespace arc::editor
