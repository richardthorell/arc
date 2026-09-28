#pragma once

/** @file
 * @brief Provider-based, read-only virtual filesystem for runtime content.
 */

#include <arc/io/io.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace arc::io
{

class virtual_path
{
public:
    virtual_path() = default;

    [[nodiscard]] static file_result<virtual_path> parse(std::string_view value);
    [[nodiscard]] static file_result<virtual_path> from_parts(std::string_view scheme, std::string_view authority,
                                                               std::string_view relative_path = {});

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::string_view string() const noexcept;
    [[nodiscard]] std::string_view scheme() const noexcept;
    [[nodiscard]] std::string_view authority() const noexcept;
    [[nodiscard]] std::string_view relative_path() const noexcept;

    auto operator<=>(const virtual_path&) const = default;

private:
    std::string value_;
    std::uint32_t authority_offset_{};
    std::uint32_t path_offset_{};

    virtual_path(std::string value, std::uint32_t authority_offset, std::uint32_t path_offset);
};

enum class provider_capability : std::uint32_t
{
    none = 0,
    full_read = 1u << 0u,
    range_read = 1u << 1u,
    metadata = 1u << 2u,
    enumeration = 1u << 3u,
    change_polling = 1u << 4u
};

using provider_capabilities = std::uint32_t;

[[nodiscard]] constexpr provider_capabilities operator|(provider_capability lhs, provider_capability rhs) noexcept
{
    return static_cast<provider_capabilities>(lhs) | static_cast<provider_capabilities>(rhs);
}

[[nodiscard]] constexpr provider_capabilities operator|(provider_capabilities lhs, provider_capability rhs) noexcept
{
    return lhs | static_cast<provider_capabilities>(rhs);
}

[[nodiscard]] constexpr bool has_capability(provider_capabilities capabilities, provider_capability capability) noexcept
{
    return (capabilities & static_cast<provider_capabilities>(capability)) != 0;
}

struct provider_file
{
    std::string key;
    std::uint64_t size{};
    std::uint64_t content_generation{};
};

enum class provider_lookup_status : std::uint8_t
{
    found,
    not_found,
    tombstone,
    failure
};

struct provider_lookup_result
{
    provider_lookup_status status{provider_lookup_status::not_found};
    provider_file file;
    file_error error;
};

struct provider_directory_entry
{
    std::string relative_path;
    std::uint64_t size{};
    std::uint64_t content_generation{};
    bool tombstone{};
};

enum class provider_change_kind : std::uint8_t
{
    added,
    modified,
    removed,
    reset
};

struct provider_change
{
    provider_change_kind kind{provider_change_kind::modified};
    std::string relative_path;
    std::uint64_t content_generation{};
};

class virtual_file_provider
{
public:
    virtual ~virtual_file_provider() = default;

    [[nodiscard]] virtual provider_capabilities capabilities() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t provider_generation() const noexcept = 0;
    [[nodiscard]] virtual provider_lookup_result resolve(std::string_view relative_path) = 0;
    [[nodiscard]] virtual jobs::job_future<file_result<file_buffer>>
    read_range(const provider_file& file, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {}) = 0;
    [[nodiscard]] virtual file_result<std::vector<provider_directory_entry>>
    enumerate(std::string_view relative_prefix);
    [[nodiscard]] virtual std::vector<provider_change> poll_changes();
};

using mount_id = std::uint64_t;

struct mount_descriptor
{
    virtual_path root;
    std::int32_t priority{};
    std::shared_ptr<virtual_file_provider> provider;
    std::string debug_name;
};

class resolved_virtual_file
{
public:
    resolved_virtual_file() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const virtual_path& path() const noexcept;
    [[nodiscard]] mount_id mount() const noexcept;
    [[nodiscard]] std::uint64_t size() const noexcept;
    [[nodiscard]] std::uint64_t provider_generation() const noexcept;
    [[nodiscard]] std::uint64_t content_generation() const noexcept;

private:
    virtual_path path_;
    mount_id mount_{};
    provider_file provider_file_;
    std::uint64_t provider_generation_{};
    std::shared_ptr<virtual_file_provider> provider_;

    friend class virtual_file_system;
};

enum class virtual_change_kind : std::uint8_t
{
    added,
    modified,
    removed,
    reset,
    mount_changed
};

struct virtual_change_event
{
    std::uint64_t sequence{};
    virtual_change_kind kind{virtual_change_kind::modified};
    virtual_path path;
    mount_id mount{};
    std::uint64_t content_generation{};
};

struct virtual_change_batch
{
    std::uint64_t first_sequence{};
    std::uint64_t last_sequence{};
    bool history_overflow{};
    std::vector<virtual_change_event> events;
};

struct virtual_file_system_config
{
    std::size_t event_history_capacity{4096};
};

struct virtual_file_system_telemetry
{
    std::uint64_t mounted_providers{};
    std::uint64_t resolutions{};
    std::uint64_t fallthroughs{};
    std::uint64_t read_operations{};
    std::uint64_t read_bytes{};
    std::uint64_t stale_completions{};
    std::uint64_t event_overflows{};
};

using virtual_change_callback = std::function<void(const virtual_change_batch&)>;
using virtual_change_subscription = std::uint64_t;

class virtual_file_system
{
public:
    explicit virtual_file_system(jobs::job_system& jobs, virtual_file_system_config config = {});

    [[nodiscard]] file_result<mount_id> mount(mount_descriptor descriptor);
    [[nodiscard]] file_result<void> unmount(mount_id id);

    [[nodiscard]] file_result<resolved_virtual_file> resolve(const virtual_path& path);
    [[nodiscard]] file_result<resolved_virtual_file> resolve(std::string_view path);
    [[nodiscard]] jobs::job_future<file_result<file_buffer>>
    read_all(const resolved_virtual_file& file, jobs::cancellation_token cancellation = {});
    [[nodiscard]] jobs::job_future<file_result<file_buffer>>
    read_range(const resolved_virtual_file& file, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {});
    [[nodiscard]] jobs::job_future<file_result<file_buffer>>
    read_all(const virtual_path& path, jobs::cancellation_token cancellation = {});
    [[nodiscard]] file_result<std::vector<provider_directory_entry>> enumerate(const virtual_path& prefix);

    [[nodiscard]] virtual_change_batch poll_changes();
    [[nodiscard]] virtual_change_batch events_since(std::uint64_t sequence) const;
    [[nodiscard]] virtual_change_subscription subscribe(virtual_change_callback callback);
    void unsubscribe(virtual_change_subscription subscription);

    [[nodiscard]] virtual_file_system_telemetry telemetry() const noexcept;

private:
    struct mount_record
    {
        mount_id id{};
        mount_descriptor descriptor;
    };

    [[nodiscard]] jobs::job_future<file_result<file_buffer>> failed_read(file_error error);
    [[nodiscard]] virtual_change_batch publish(std::vector<virtual_change_event> events);

    jobs::job_system* jobs_{};
    virtual_file_system_config config_{};
    mutable std::shared_mutex mount_mutex_;
    std::vector<mount_record> mounts_;
    mutable std::mutex event_mutex_;
    std::vector<virtual_change_event> history_;
    std::unordered_map<virtual_change_subscription, virtual_change_callback> callbacks_;
    std::uint64_t next_mount_id_{1};
    std::uint64_t next_sequence_{1};
    std::uint64_t next_subscription_{1};
    virtual_file_system_telemetry telemetry_{};
};

struct filesystem_provider_config
{
    bool case_sensitive{true};
    std::chrono::milliseconds debounce{100};
};

class filesystem_file_provider final : public virtual_file_provider
{
public:
    filesystem_file_provider(async_file_service& files, std::filesystem::path root,
                             filesystem_provider_config config = {});

    [[nodiscard]] provider_capabilities capabilities() const noexcept override;
    [[nodiscard]] std::uint64_t provider_generation() const noexcept override;
    [[nodiscard]] provider_lookup_result resolve(std::string_view relative_path) override;
    [[nodiscard]] jobs::job_future<file_result<file_buffer>>
    read_range(const provider_file& file, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {}) override;
    [[nodiscard]] file_result<std::vector<provider_directory_entry>>
    enumerate(std::string_view relative_prefix) override;
    [[nodiscard]] std::vector<provider_change> poll_changes() override;

    [[nodiscard]] const std::filesystem::path& root() const noexcept;

private:
    struct snapshot_entry
    {
        std::uint64_t size{};
        std::filesystem::file_time_type modified{};

        auto operator<=>(const snapshot_entry&) const = default;
    };

    [[nodiscard]] file_result<std::filesystem::path> physical_path(std::string_view relative_path) const;
    [[nodiscard]] std::unordered_map<std::string, snapshot_entry> scan() const;

    async_file_service* files_{};
    std::filesystem::path root_;
    filesystem_provider_config config_{};
    mutable std::mutex mutex_;
    std::uint64_t generation_{1};
    bool snapshot_initialized_{};
    std::unordered_map<std::string, snapshot_entry> snapshot_;
};

struct memory_provider_update
{
    std::string relative_path;
    file_buffer bytes;
    bool tombstone{};
};

class memory_file_provider final : public virtual_file_provider
{
public:
    explicit memory_file_provider(jobs::job_system& jobs);

    [[nodiscard]] provider_capabilities capabilities() const noexcept override;
    [[nodiscard]] std::uint64_t provider_generation() const noexcept override;
    [[nodiscard]] provider_lookup_result resolve(std::string_view relative_path) override;
    [[nodiscard]] jobs::job_future<file_result<file_buffer>>
    read_range(const provider_file& file, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {}) override;
    [[nodiscard]] file_result<std::vector<provider_directory_entry>>
    enumerate(std::string_view relative_prefix) override;
    [[nodiscard]] std::vector<provider_change> poll_changes() override;

    [[nodiscard]] file_result<void> publish(std::span<const memory_provider_update> updates);
    [[nodiscard]] file_result<void> clear();

private:
    struct entry
    {
        std::shared_ptr<const file_buffer> bytes;
        std::uint64_t generation{};
        bool tombstone{};
    };

    jobs::job_system* jobs_{};
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, entry> entries_;
    std::vector<provider_change> pending_changes_;
    std::uint64_t generation_{1};
};

} // namespace arc::io
