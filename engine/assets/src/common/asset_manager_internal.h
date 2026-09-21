#pragma once

#include <arc/assets/assets.h>

#include <sqlite3.h>

#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace arc::assets
{
namespace manager_detail
{
std::string path_key(std::string value);
std::string path_key(const std::filesystem::path& value);
std::optional<asset_guid> authored_asset_guid(const std::filesystem::path& source, asset_type_id type);
bool publish_artifact(const std::filesystem::path& destination, std::span<const std::byte> bytes,
                      const asset_hash& expected_hash, std::string& error);

class sqlite_statement
{
public:
    sqlite_statement(sqlite3* database, const char* sql);
    ~sqlite_statement();
    sqlite_statement(const sqlite_statement&) = delete;
    sqlite_statement& operator=(const sqlite_statement&) = delete;
    sqlite3_stmt* get() const noexcept;
    explicit operator bool() const noexcept;

private:
    sqlite3_stmt* statement_{};
};

void bind_text(sqlite3_stmt* statement, int index, std::string_view value);
} // namespace manager_detail

struct asset_manager::implementation
{
    struct record
    {
        asset_snapshot snapshot;
        asset_source_metadata metadata;
        std::filesystem::path absolute_path;
        std::filesystem::file_time_type modified{};
        std::uint64_t file_size{};
        std::shared_ptr<detail::asset_slot> slot = std::make_shared<detail::asset_slot>();
        jobs::job_handle active_import;
        jobs::cancellation_source import_cancellation;
        std::chrono::steady_clock::time_point last_used{};
        std::filesystem::file_time_type pending_modified{};
        std::uint64_t pending_file_size{};
        std::chrono::steady_clock::time_point pending_since{};
        asset_residency requested_residency{asset_residency::cpu};
        bool pending_source_change{};
        bool virtual_asset{};
    };

    asset_manager_config config;
    jobs::job_system* jobs{};
    io::async_file_service* files{};
    memory::memory_system* memory{};
    std::unique_ptr<memory::streaming_heap> streaming;
    std::uint64_t pressure_handler{};
    mutable std::shared_mutex mutex;
    sqlite3* database{};
    std::unordered_map<asset_guid, record, asset_guid_hash> records;
    std::unordered_map<std::string, asset_guid> paths;
    std::unordered_map<asset_importer_id, std::unique_ptr<asset_importer>, asset_importer_id_hash> importers;
    std::unordered_map<asset_type_id, asset_guid, asset_type_id_hash> fallbacks;
    std::vector<missing_asset_reference> missing_references;
    std::vector<asset_event> events;
    std::unordered_map<std::uint64_t, asset_event_callback> subscribers;
    std::uint64_t next_subscription{1};
    std::uint64_t next_event{1};
    std::uint64_t revision{};
    std::uint64_t next_diagnostic{1};
    std::chrono::steady_clock::time_point next_poll{};
    bool started{};

    implementation(asset_manager_config value, jobs::job_system& scheduler, io::async_file_service& file_service,
                   memory::memory_system& memory_system);
    ~implementation();

    bool managed_source_path(const std::filesystem::path& path) const;
    void emit(asset_event_kind kind, asset_guid guid, asset_state state, std::string message, float progress = 0.0f);
    asset_diagnostic diagnostic(asset_guid guid, asset_diagnostic_severity severity, std::string category,
                                std::string message);
    bool open_database(std::string& error);
    bool rebuild_database(std::string& error);
    bool load_database(std::string& error);
    bool persist_record(const record& value);
    void persist_dependencies(const record& value);
    void persist_artifacts(const record& value);
    void clear_tombstone(asset_guid guid);
    void persist_tombstone(const record& value);
    bool dependency_reaches(asset_guid current, asset_guid target,
                            std::unordered_set<asset_guid, asset_guid_hash>& visited) const;
    void mark_reverse_stale(asset_guid guid, std::string_view reason);
    jobs::job_handle ensure_import(asset_guid guid, asset_streaming_priority priority,
                                   jobs::cancellation_token cancellation,
                                   asset_residency requested_residency = asset_residency::cpu);
};
} // namespace arc::assets
