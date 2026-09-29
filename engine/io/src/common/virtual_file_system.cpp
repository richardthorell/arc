#include <arc/io/virtual_file_system.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <system_error>

namespace arc::io
{
namespace
{

file_error logical_error(file_error_code code, std::string_view path, std::string message)
{
    return {.code = code, .message = std::move(message), .logical_path = std::string(path)};
}

bool valid_uri_token(std::string_view value) noexcept
{
    if (value.empty()) return false;
    return std::all_of(value.begin(), value.end(),
                       [](char character)
                       {
                           return std::isalnum(static_cast<unsigned char>(character)) || character == '-' ||
                                  character == '_' || character == '.';
                       });
}

std::string lowercase_ascii(std::string_view value)
{
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](char character)
                   { return static_cast<char>(std::tolower(static_cast<unsigned char>(character))); });
    return result;
}

file_result<std::string> normalize_relative(std::string_view input)
{
    std::string result;
    std::size_t cursor{};
    while (cursor < input.size())
    {
        while (cursor < input.size() && input[cursor] == '/')
            ++cursor;
        const auto begin = cursor;
        while (cursor < input.size() && input[cursor] != '/')
        {
            if (input[cursor] == '\\')
                return file_result<std::string>::failure(
                    logical_error(file_error_code::invalid_request, input, "Virtual paths cannot contain backslashes"));
            ++cursor;
        }
        if (begin == cursor) continue;
        const auto component = input.substr(begin, cursor - begin);
        if (component == ".") continue;
        if (component == "..")
            return file_result<std::string>::failure(logical_error(
                file_error_code::invalid_request, input, "Virtual paths cannot traverse above their mount root"));
        if (!result.empty()) result.push_back('/');
        result.append(component);
    }
    return file_result<std::string>::success(std::move(result));
}

std::uint64_t fingerprint(std::uint64_t size, std::filesystem::file_time_type modified) noexcept
{
    const auto stamp = static_cast<std::uint64_t>(modified.time_since_epoch().count());
    std::uint64_t result = size + 0x9e3779b97f4a7c15ull;
    result ^= stamp + 0x9e3779b97f4a7c15ull + (result << 6u) + (result >> 2u);
    return result == 0 ? 1 : result;
}

file_result<file_buffer> read_physical_range(const std::filesystem::path& path, std::uint64_t offset, std::size_t bytes,
                                             std::uint64_t expected_generation,
                                             const jobs::cancellation_token& cancellation)
{
    std::error_code error;
    const auto before_size = std::filesystem::file_size(path, error);
    if (error)
        return file_result<file_buffer>::failure(
            {.code = file_error_code::not_found, .path = path, .message = "File is no longer available"});
    const auto before_time = std::filesystem::last_write_time(path, error);
    if (error || fingerprint(before_size, before_time) != expected_generation)
        return file_result<file_buffer>::failure(
            {.code = file_error_code::stale_handle, .path = path, .message = "Resolved file generation is stale"});
    if (offset > before_size || bytes > before_size - offset)
        return file_result<file_buffer>::failure(
            {.code = file_error_code::invalid_range, .path = path, .message = "Read range is outside the file"});

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return file_result<file_buffer>::failure(
            {.code = file_error_code::permission_denied, .path = path, .message = "File could not be opened"});
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    file_buffer result(bytes);
    constexpr std::size_t chunk_size = 1024u * 1024u;
    std::size_t completed{};
    while (completed < result.size())
    {
        if (cancellation.stop_requested())
            return file_result<file_buffer>::failure(
                {.code = file_error_code::cancelled, .path = path, .message = "Read was cancelled"});
        const auto count = std::min(chunk_size, result.size() - completed);
        stream.read(reinterpret_cast<char*>(result.data() + completed), static_cast<std::streamsize>(count));
        if (stream.gcount() != static_cast<std::streamsize>(count))
            return file_result<file_buffer>::failure(
                {.code = file_error_code::read_failed, .path = path, .message = "File read ended unexpectedly"});
        completed += count;
    }

    const auto after_size = std::filesystem::file_size(path, error);
    const auto after_time = error ? std::filesystem::file_time_type{} : std::filesystem::last_write_time(path, error);
    if (error || fingerprint(after_size, after_time) != expected_generation)
        return file_result<file_buffer>::failure(
            {.code = file_error_code::stale_handle, .path = path, .message = "File changed while it was read"});
    return file_result<file_buffer>::success(std::move(result));
}

std::string join_virtual(const virtual_path& root, std::string_view relative)
{
    std::string result(root.string());
    if (!relative.empty())
    {
        if (!result.empty() && result.back() != '/') result.push_back('/');
        result.append(relative);
    }
    return result;
}

} // namespace

virtual_path::virtual_path(std::string value, std::uint32_t authority_offset, std::uint32_t path_offset)
    : value_(std::move(value)), authority_offset_(authority_offset), path_offset_(path_offset)
{
}

file_result<virtual_path> virtual_path::parse(std::string_view value)
{
    const auto separator = value.find("://");
    if (separator == std::string_view::npos || separator == 0)
        return file_result<virtual_path>::failure(
            logical_error(file_error_code::invalid_request, value, "Virtual path must use scheme://authority syntax"));
    const auto authority_begin = separator + 3;
    const auto path_begin = value.find('/', authority_begin);
    const auto authority_end = path_begin == std::string_view::npos ? value.size() : path_begin;
    const auto scheme = value.substr(0, separator);
    const auto authority = value.substr(authority_begin, authority_end - authority_begin);
    if (!valid_uri_token(scheme) || !valid_uri_token(authority))
        return file_result<virtual_path>::failure(
            logical_error(file_error_code::invalid_request, value, "Virtual path scheme or authority is invalid"));

    auto relative =
        normalize_relative(path_begin == std::string_view::npos ? std::string_view{} : value.substr(path_begin + 1));
    if (!relative) return file_result<virtual_path>::failure(relative.error());
    return from_parts(scheme, authority, relative.value());
}

file_result<virtual_path> virtual_path::from_parts(std::string_view scheme, std::string_view authority,
                                                   std::string_view relative_path)
{
    if (!valid_uri_token(scheme) || !valid_uri_token(authority))
        return file_result<virtual_path>::failure(logical_error(
            file_error_code::invalid_request, {}, "Virtual path scheme and authority must be non-empty URI tokens"));
    auto relative = normalize_relative(relative_path);
    if (!relative) return file_result<virtual_path>::failure(relative.error());

    std::string value = lowercase_ascii(scheme);
    value.append("://");
    const auto authority_offset = static_cast<std::uint32_t>(value.size());
    value.append(lowercase_ascii(authority));
    const auto path_offset = relative.value().empty() ? static_cast<std::uint32_t>(value.size())
                                                      : static_cast<std::uint32_t>(value.size() + 1);
    if (!relative.value().empty())
    {
        value.push_back('/');
        value.append(relative.value());
    }
    return file_result<virtual_path>::success(virtual_path(std::move(value), authority_offset, path_offset));
}

bool virtual_path::empty() const noexcept
{
    return value_.empty();
}

std::string_view virtual_path::string() const noexcept
{
    return value_;
}

std::string_view virtual_path::scheme() const noexcept
{
    return value_.empty() ? std::string_view{} : std::string_view(value_).substr(0, authority_offset_ - 3u);
}

std::string_view virtual_path::authority() const noexcept
{
    if (value_.empty()) return {};
    const auto end = path_offset_ == value_.size() ? value_.size() : path_offset_ - 1u;
    return std::string_view(value_).substr(authority_offset_, end - authority_offset_);
}

std::string_view virtual_path::relative_path() const noexcept
{
    if (value_.empty() || path_offset_ == value_.size()) return {};
    return std::string_view(value_).substr(path_offset_);
}

file_result<std::vector<provider_directory_entry>> virtual_file_provider::enumerate(std::string_view)
{
    return file_result<std::vector<provider_directory_entry>>::failure(
        logical_error(file_error_code::provider_unavailable, {}, "Provider does not support enumeration"));
}

std::vector<provider_change> virtual_file_provider::poll_changes()
{
    return {};
}

bool resolved_virtual_file::valid() const noexcept
{
    return provider_ != nullptr;
}

const virtual_path& resolved_virtual_file::path() const noexcept
{
    return path_;
}

mount_id resolved_virtual_file::mount() const noexcept
{
    return mount_;
}

std::uint64_t resolved_virtual_file::size() const noexcept
{
    return provider_file_.size;
}

std::uint64_t resolved_virtual_file::provider_generation() const noexcept
{
    return provider_generation_;
}

std::uint64_t resolved_virtual_file::content_generation() const noexcept
{
    return provider_file_.content_generation;
}

virtual_file_system::virtual_file_system(jobs::job_system& jobs, virtual_file_system_config config)
    : jobs_(&jobs), config_(config)
{
    config_.event_history_capacity = std::max<std::size_t>(1, config_.event_history_capacity);
}

file_result<mount_id> virtual_file_system::mount(mount_descriptor descriptor)
{
    if (descriptor.root.empty() || !descriptor.provider)
        return file_result<mount_id>::failure(
            logical_error(file_error_code::invalid_request, descriptor.root.string(), "Mount is incomplete"));

    mount_id id{};
    const auto root = descriptor.root;
    {
        std::unique_lock lock(mount_mutex_);
        const auto duplicate = std::find_if(
            mounts_.begin(), mounts_.end(), [&](const mount_record& record)
            { return record.descriptor.root == descriptor.root && record.descriptor.priority == descriptor.priority; });
        if (duplicate != mounts_.end())
            return file_result<mount_id>::failure(logical_error(
                file_error_code::invalid_request, descriptor.root.string(), "Mount priority is already occupied"));
        id = next_mount_id_++;
        mounts_.push_back({.id = id, .descriptor = std::move(descriptor)});
        std::sort(mounts_.begin(), mounts_.end(),
                  [](const mount_record& lhs, const mount_record& rhs)
                  {
                      if (lhs.descriptor.priority != rhs.descriptor.priority)
                          return lhs.descriptor.priority > rhs.descriptor.priority;
                      return lhs.id < rhs.id;
                  });
    }
    {
        std::lock_guard lock(event_mutex_);
        telemetry_.mounted_providers++;
    }
    static_cast<void>(
        publish({virtual_change_event{.kind = virtual_change_kind::mount_changed, .path = root, .mount = id}}));
    return file_result<mount_id>::success(id);
}

file_result<void> virtual_file_system::unmount(mount_id id)
{
    virtual_path root;
    {
        std::unique_lock lock(mount_mutex_);
        const auto found =
            std::find_if(mounts_.begin(), mounts_.end(), [id](const mount_record& record) { return record.id == id; });
        if (found == mounts_.end())
            return file_result<void>::failure(
                logical_error(file_error_code::not_found, {}, "Virtual filesystem mount was not found"));
        root = found->descriptor.root;
        mounts_.erase(found);
    }
    {
        std::lock_guard lock(event_mutex_);
        telemetry_.mounted_providers--;
    }
    static_cast<void>(publish(
        {virtual_change_event{.kind = virtual_change_kind::mount_changed, .path = std::move(root), .mount = id}}));
    return file_result<void>::success();
}

file_result<resolved_virtual_file> virtual_file_system::resolve(const virtual_path& path)
{
    std::shared_lock lock(mount_mutex_);
    {
        std::lock_guard event_lock(event_mutex_);
        telemetry_.resolutions++;
    }
    for (const auto& mount : mounts_)
    {
        const auto& root = mount.descriptor.root;
        if (root.scheme() != path.scheme() || root.authority() != path.authority()) continue;
        const auto root_relative = root.relative_path();
        const auto path_relative = path.relative_path();
        if (!root_relative.empty() &&
            (path_relative.size() < root_relative.size() ||
             path_relative.substr(0, root_relative.size()) != root_relative ||
             (path_relative.size() > root_relative.size() && path_relative[root_relative.size()] != '/')))
            continue;
        auto provider_relative = path_relative.substr(root_relative.size());
        if (!provider_relative.empty() && provider_relative.front() == '/') provider_relative.remove_prefix(1);
        auto lookup = mount.descriptor.provider->resolve(provider_relative);
        if (lookup.status == provider_lookup_status::not_found)
        {
            std::lock_guard event_lock(event_mutex_);
            telemetry_.fallthroughs++;
            continue;
        }
        if (lookup.status == provider_lookup_status::tombstone)
            return file_result<resolved_virtual_file>::failure(
                logical_error(file_error_code::tombstoned, path.string(), "Virtual file is hidden by an overlay"));
        if (lookup.status == provider_lookup_status::failure)
        {
            lookup.error.logical_path = std::string(path.string());
            return file_result<resolved_virtual_file>::failure(std::move(lookup.error));
        }
        resolved_virtual_file result;
        result.path_ = path;
        result.mount_ = mount.id;
        result.provider_file_ = std::move(lookup.file);
        result.provider_generation_ = mount.descriptor.provider->provider_generation();
        result.provider_ = mount.descriptor.provider;
        return file_result<resolved_virtual_file>::success(std::move(result));
    }
    return file_result<resolved_virtual_file>::failure(
        logical_error(file_error_code::not_found, path.string(), "No mounted provider contains the virtual file"));
}

file_result<resolved_virtual_file> virtual_file_system::resolve(std::string_view path)
{
    auto parsed = virtual_path::parse(path);
    if (!parsed) return file_result<resolved_virtual_file>::failure(parsed.error());
    return resolve(parsed.value());
}

jobs::job_future<file_result<file_buffer>> virtual_file_system::failed_read(file_error error)
{
    return jobs_->submit_future([error = std::move(error)]() mutable
                                { return file_result<file_buffer>::failure(std::move(error)); });
}

jobs::job_future<file_result<file_buffer>> virtual_file_system::read_all(const resolved_virtual_file& file,
                                                                         jobs::cancellation_token cancellation)
{
    if (file.size() > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return failed_read(logical_error(file_error_code::invalid_range, file.path().string(),
                                         "Virtual file is too large for this process"));
    return read_range(file, 0, static_cast<std::size_t>(file.size()), std::move(cancellation));
}

jobs::job_future<file_result<file_buffer>> virtual_file_system::read_range(const resolved_virtual_file& file,
                                                                           std::uint64_t offset, std::size_t bytes,
                                                                           jobs::cancellation_token cancellation)
{
    if (!file.valid())
        return failed_read(logical_error(file_error_code::invalid_request, {}, "Resolved virtual file is invalid"));
    if (offset > file.size() || bytes > file.size() - offset)
        return failed_read(
            logical_error(file_error_code::invalid_range, file.path().string(), "Read range is outside the file"));
    {
        std::lock_guard lock(event_mutex_);
        telemetry_.read_operations++;
        telemetry_.read_bytes += bytes;
    }
    return file.provider_->read_range(file.provider_file_, offset, bytes, std::move(cancellation));
}

jobs::job_future<file_result<file_buffer>> virtual_file_system::read_all(const virtual_path& path,
                                                                         jobs::cancellation_token cancellation)
{
    auto file = resolve(path);
    if (!file) return failed_read(file.error());
    return read_all(file.value(), std::move(cancellation));
}

file_result<std::vector<provider_directory_entry>> virtual_file_system::enumerate(const virtual_path& prefix)
{
    std::shared_lock lock(mount_mutex_);
    std::vector<provider_directory_entry> result;
    std::unordered_map<std::string, bool> seen;
    for (const auto& mount : mounts_)
    {
        if (mount.descriptor.root.scheme() != prefix.scheme() ||
            mount.descriptor.root.authority() != prefix.authority())
            continue;
        if (!has_capability(mount.descriptor.provider->capabilities(), provider_capability::enumeration)) continue;
        auto entries = mount.descriptor.provider->enumerate(prefix.relative_path());
        if (!entries) return entries;
        for (auto& entry : entries.value())
        {
            if (seen.contains(entry.relative_path)) continue;
            seen.emplace(entry.relative_path, true);
            if (!entry.tombstone) result.push_back(std::move(entry));
        }
    }
    std::sort(result.begin(), result.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.relative_path < rhs.relative_path; });
    return file_result<std::vector<provider_directory_entry>>::success(std::move(result));
}

virtual_change_batch virtual_file_system::publish(std::vector<virtual_change_event> events)
{
    virtual_change_batch batch;
    std::vector<virtual_change_callback> callbacks;
    {
        std::lock_guard lock(event_mutex_);
        for (auto& event : events)
        {
            event.sequence = next_sequence_++;
            if (batch.first_sequence == 0) batch.first_sequence = event.sequence;
            batch.last_sequence = event.sequence;
            history_.push_back(event);
            batch.events.push_back(std::move(event));
        }
        if (history_.size() > config_.event_history_capacity)
        {
            const auto remove_count = history_.size() - config_.event_history_capacity;
            history_.erase(history_.begin(), history_.begin() + static_cast<std::ptrdiff_t>(remove_count));
            batch.history_overflow = true;
            telemetry_.event_overflows++;
        }
        callbacks.reserve(callbacks_.size());
        for (const auto& [subscription, callback] : callbacks_)
        {
            static_cast<void>(subscription);
            callbacks.push_back(callback);
        }
    }
    for (const auto& callback : callbacks)
        callback(batch);
    return batch;
}

virtual_change_batch virtual_file_system::poll_changes()
{
    std::vector<mount_record> mounts;
    {
        std::shared_lock lock(mount_mutex_);
        mounts = mounts_;
    }
    std::vector<virtual_change_event> events;
    for (auto& mount : mounts)
    {
        if (!has_capability(mount.descriptor.provider->capabilities(), provider_capability::change_polling)) continue;
        for (auto& change : mount.descriptor.provider->poll_changes())
        {
            auto path = virtual_path::parse(join_virtual(mount.descriptor.root, change.relative_path));
            if (!path) continue;
            auto kind = virtual_change_kind::modified;
            switch (change.kind)
            {
                case provider_change_kind::added:
                    kind = virtual_change_kind::added;
                    break;
                case provider_change_kind::modified:
                    kind = virtual_change_kind::modified;
                    break;
                case provider_change_kind::removed:
                    kind = virtual_change_kind::removed;
                    break;
                case provider_change_kind::reset:
                    kind = virtual_change_kind::reset;
                    break;
            }
            events.push_back({.kind = kind,
                              .path = std::move(path.value()),
                              .mount = mount.id,
                              .content_generation = change.content_generation});
        }
    }
    std::sort(events.begin(), events.end(),
              [](const auto& lhs, const auto& rhs)
              {
                  if (lhs.path != rhs.path) return lhs.path < rhs.path;
                  if (lhs.mount != rhs.mount) return lhs.mount < rhs.mount;
                  return lhs.kind < rhs.kind;
              });
    return publish(std::move(events));
}

virtual_change_batch virtual_file_system::events_since(std::uint64_t sequence) const
{
    std::lock_guard lock(event_mutex_);
    virtual_change_batch result;
    if (!history_.empty() && sequence + 1u < history_.front().sequence) result.history_overflow = true;
    for (const auto& event : history_)
    {
        if (event.sequence <= sequence) continue;
        if (result.first_sequence == 0) result.first_sequence = event.sequence;
        result.last_sequence = event.sequence;
        result.events.push_back(event);
    }
    return result;
}

virtual_change_subscription virtual_file_system::subscribe(virtual_change_callback callback)
{
    std::lock_guard lock(event_mutex_);
    const auto subscription = next_subscription_++;
    callbacks_.emplace(subscription, std::move(callback));
    return subscription;
}

void virtual_file_system::unsubscribe(virtual_change_subscription subscription)
{
    std::lock_guard lock(event_mutex_);
    callbacks_.erase(subscription);
}

virtual_file_system_telemetry virtual_file_system::telemetry() const noexcept
{
    std::lock_guard lock(event_mutex_);
    return telemetry_;
}

filesystem_file_provider::filesystem_file_provider(async_file_service& files, std::filesystem::path root,
                                                   filesystem_provider_config config)
    : files_(&files), config_(config)
{
    root = std::filesystem::absolute(root).lexically_normal();
    root_ = std::move(root);
}

provider_capabilities filesystem_file_provider::capabilities() const noexcept
{
    return provider_capability::full_read | provider_capability::range_read | provider_capability::metadata |
           provider_capability::enumeration | provider_capability::change_polling;
}

std::uint64_t filesystem_file_provider::provider_generation() const noexcept
{
    std::lock_guard lock(mutex_);
    return generation_;
}

file_result<std::filesystem::path> filesystem_file_provider::physical_path(std::string_view relative_path) const
{
    auto normalized = normalize_relative(relative_path);
    if (!normalized) return file_result<std::filesystem::path>::failure(normalized.error());
    auto candidate = (root_ / std::filesystem::path(normalized.value())).lexically_normal();
    const auto relative = candidate.lexically_relative(root_);
    if (relative.empty() && candidate != root_)
        return file_result<std::filesystem::path>::failure(
            logical_error(file_error_code::invalid_request, relative_path, "Path is outside the provider root"));
    for (const auto& component : relative)
        if (component == "..")
            return file_result<std::filesystem::path>::failure(
                logical_error(file_error_code::invalid_request, relative_path, "Path is outside the provider root"));
    return file_result<std::filesystem::path>::success(std::move(candidate));
}

provider_lookup_result filesystem_file_provider::resolve(std::string_view relative_path)
{
    auto path = physical_path(relative_path);
    if (!path) return {.status = provider_lookup_status::failure, .error = path.error()};
    std::error_code error;
    if (!std::filesystem::is_regular_file(path.value(), error))
        return {.status = error ? provider_lookup_status::failure : provider_lookup_status::not_found,
                .error = error ? file_error{.code = file_error_code::provider_unavailable,
                                            .path = path.value(),
                                            .message = error.message()}
                               : file_error{}};
    const auto size = std::filesystem::file_size(path.value(), error);
    const auto modified =
        error ? std::filesystem::file_time_type{} : std::filesystem::last_write_time(path.value(), error);
    if (error)
        return {.status = provider_lookup_status::failure,
                .error = {.code = file_error_code::read_failed, .path = path.value(), .message = error.message()}};
    return {.status = provider_lookup_status::found,
            .file = {
                .key = path.value().generic_string(), .size = size, .content_generation = fingerprint(size, modified)}};
}

jobs::job_future<file_result<file_buffer>> filesystem_file_provider::read_range(const provider_file& file,
                                                                                std::uint64_t offset, std::size_t bytes,
                                                                                jobs::cancellation_token cancellation)
{
    const auto path = std::filesystem::path(file.key);
    return files_->scheduler().submit_future(
        {.name = "vfs.filesystem.read",
         .priority = jobs::job_priority::normal,
         .affinity = jobs::job_affinity::io_thread,
         .cancellation = cancellation,
         .dependency_policy = jobs::job_dependency_policy::cancel_on_failure},
        [path, offset, bytes, generation = file.content_generation, cancellation]
        { return read_physical_range(path, offset, bytes, generation, cancellation); });
}

std::unordered_map<std::string, filesystem_file_provider::snapshot_entry> filesystem_file_provider::scan() const
{
    std::unordered_map<std::string, snapshot_entry> result;
    std::error_code error;
    if (!std::filesystem::is_directory(root_, error)) return result;
    for (std::filesystem::recursive_directory_iterator iterator(root_, error), end; iterator != end && !error;
         iterator.increment(error))
    {
        if (!iterator->is_regular_file(error) || error) continue;
        auto relative = iterator->path().lexically_relative(root_).generic_string();
        if (!config_.case_sensitive) relative = lowercase_ascii(relative);
        const auto size = iterator->file_size(error);
        if (error) continue;
        const auto modified = iterator->last_write_time(error);
        if (error) continue;
        result.emplace(std::move(relative), snapshot_entry{.size = size, .modified = modified});
    }
    return result;
}

file_result<std::vector<provider_directory_entry>> filesystem_file_provider::enumerate(std::string_view relative_prefix)
{
    const auto files = scan();
    std::vector<provider_directory_entry> result;
    for (const auto& [path, entry] : files)
    {
        if (!relative_prefix.empty() && !path.starts_with(relative_prefix)) continue;
        result.push_back(
            {.relative_path = path, .size = entry.size, .content_generation = fingerprint(entry.size, entry.modified)});
    }
    std::sort(result.begin(), result.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.relative_path < rhs.relative_path; });
    return file_result<std::vector<provider_directory_entry>>::success(std::move(result));
}

std::vector<provider_change> filesystem_file_provider::poll_changes()
{
    auto current = scan();
    std::vector<provider_change> changes;
    std::lock_guard lock(mutex_);
    if (!snapshot_initialized_)
    {
        snapshot_ = std::move(current);
        snapshot_initialized_ = true;
        return changes;
    }
    if (current == snapshot_)
    {
        pending_snapshot_.clear();
        pending_snapshot_valid_ = false;
        return changes;
    }
    const auto now = std::chrono::steady_clock::now();
    if (config_.debounce.count() > 0)
    {
        if (!pending_snapshot_valid_ || current != pending_snapshot_)
        {
            pending_snapshot_ = std::move(current);
            pending_since_ = now;
            pending_snapshot_valid_ = true;
            return changes;
        }
        if (now - pending_since_ < config_.debounce) return changes;
        current = std::move(pending_snapshot_);
    }
    pending_snapshot_.clear();
    pending_snapshot_valid_ = false;
    for (const auto& [path, entry] : current)
    {
        const auto previous = snapshot_.find(path);
        if (previous == snapshot_.end())
            changes.push_back({.kind = provider_change_kind::added,
                               .relative_path = path,
                               .content_generation = fingerprint(entry.size, entry.modified)});
        else if (previous->second != entry)
            changes.push_back({.kind = provider_change_kind::modified,
                               .relative_path = path,
                               .content_generation = fingerprint(entry.size, entry.modified)});
    }
    for (const auto& [path, entry] : snapshot_)
    {
        static_cast<void>(entry);
        if (!current.contains(path)) changes.push_back({.kind = provider_change_kind::removed, .relative_path = path});
    }
    if (!changes.empty()) ++generation_;
    snapshot_ = std::move(current);
    std::sort(changes.begin(), changes.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.relative_path < rhs.relative_path; });
    return changes;
}

const std::filesystem::path& filesystem_file_provider::root() const noexcept
{
    return root_;
}

memory_file_provider::memory_file_provider(jobs::job_system& jobs) : jobs_(&jobs) {}

provider_capabilities memory_file_provider::capabilities() const noexcept
{
    return provider_capability::full_read | provider_capability::range_read | provider_capability::metadata |
           provider_capability::enumeration | provider_capability::change_polling;
}

std::uint64_t memory_file_provider::provider_generation() const noexcept
{
    std::shared_lock lock(mutex_);
    return generation_;
}

provider_lookup_result memory_file_provider::resolve(std::string_view relative_path)
{
    auto normalized = normalize_relative(relative_path);
    if (!normalized) return {.status = provider_lookup_status::failure, .error = normalized.error()};
    std::shared_lock lock(mutex_);
    const auto found = entries_.find(normalized.value());
    if (found == entries_.end()) return {};
    if (found->second.tombstone) return {.status = provider_lookup_status::tombstone};
    return {.status = provider_lookup_status::found,
            .file = {.key = found->first,
                     .size = found->second.bytes->size(),
                     .content_generation = found->second.generation}};
}

jobs::job_future<file_result<file_buffer>> memory_file_provider::read_range(const provider_file& file,
                                                                            std::uint64_t offset, std::size_t bytes,
                                                                            jobs::cancellation_token cancellation)
{
    return jobs_->submit_future(
        {.name = "vfs.memory.read",
         .priority = jobs::job_priority::normal,
         .affinity = jobs::job_affinity::io_thread,
         .cancellation = cancellation,
         .dependency_policy = jobs::job_dependency_policy::cancel_on_failure},
        [this, file, offset, bytes, cancellation]
        {
            if (cancellation.stop_requested())
                return file_result<file_buffer>::failure(
                    logical_error(file_error_code::cancelled, file.key, "Read was cancelled"));
            std::shared_ptr<const file_buffer> source;
            {
                std::shared_lock lock(mutex_);
                const auto found = entries_.find(file.key);
                if (found == entries_.end() || found->second.tombstone ||
                    found->second.generation != file.content_generation)
                    return file_result<file_buffer>::failure(
                        logical_error(file_error_code::stale_handle, file.key, "Resolved memory file is stale"));
                source = found->second.bytes;
            }
            if (offset > source->size() || bytes > source->size() - offset)
                return file_result<file_buffer>::failure(
                    logical_error(file_error_code::invalid_range, file.key, "Read range is outside the file"));
            return file_result<file_buffer>::success(
                file_buffer(source->begin() + static_cast<std::ptrdiff_t>(offset),
                            source->begin() + static_cast<std::ptrdiff_t>(offset + bytes)));
        });
}

file_result<std::vector<provider_directory_entry>> memory_file_provider::enumerate(std::string_view relative_prefix)
{
    std::shared_lock lock(mutex_);
    std::vector<provider_directory_entry> result;
    for (const auto& [path, value] : entries_)
    {
        if (!relative_prefix.empty() && !path.starts_with(relative_prefix)) continue;
        result.push_back({.relative_path = path,
                          .size = value.bytes ? value.bytes->size() : 0,
                          .content_generation = value.generation,
                          .tombstone = value.tombstone});
    }
    std::sort(result.begin(), result.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.relative_path < rhs.relative_path; });
    return file_result<std::vector<provider_directory_entry>>::success(std::move(result));
}

std::vector<provider_change> memory_file_provider::poll_changes()
{
    std::unique_lock lock(mutex_);
    auto result = std::move(pending_changes_);
    pending_changes_.clear();
    return result;
}

file_result<void> memory_file_provider::publish(std::span<const memory_provider_update> updates)
{
    struct prepared_update
    {
        std::string path;
        std::shared_ptr<const file_buffer> bytes;
        bool tombstone{};
    };
    std::vector<prepared_update> prepared;
    prepared.reserve(updates.size());
    for (const auto& update : updates)
    {
        auto normalized = normalize_relative(update.relative_path);
        if (!normalized || normalized.value().empty())
            return file_result<void>::failure(normalized
                                                  ? logical_error(file_error_code::invalid_request,
                                                                  update.relative_path, "Memory provider path is empty")
                                                  : normalized.error());
        prepared.push_back({.path = std::move(normalized.value()),
                            .bytes = std::make_shared<const file_buffer>(update.bytes),
                            .tombstone = update.tombstone});
    }

    std::unique_lock lock(mutex_);
    const auto generation = ++generation_;
    for (auto& update : prepared)
    {
        const auto found = entries_.find(update.path);
        const auto kind = found == entries_.end() ? provider_change_kind::added : provider_change_kind::modified;
        entries_[update.path] = {
            .bytes = std::move(update.bytes), .generation = generation, .tombstone = update.tombstone};
        pending_changes_.push_back({.kind = update.tombstone ? provider_change_kind::removed : kind,
                                    .relative_path = update.path,
                                    .content_generation = generation});
    }
    return file_result<void>::success();
}

file_result<void> memory_file_provider::clear()
{
    std::unique_lock lock(mutex_);
    entries_.clear();
    ++generation_;
    pending_changes_.push_back({.kind = provider_change_kind::reset, .content_generation = generation_});
    return file_result<void>::success();
}

} // namespace arc::io
