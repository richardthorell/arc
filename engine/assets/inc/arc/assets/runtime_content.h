#pragma once

/** @file
 * @brief Logical cooked-content providers, catalog, and verified live-update ingestion.
 */

#include <arc/assets/package_artifact_reader.h>
#include <arc/io/virtual_file_system.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::assets
{

[[nodiscard]] io::file_result<io::virtual_path>
cooked_artifact_virtual_path(const cooked_artifact_address& address, std::string_view mount = "artifact://game");
[[nodiscard]] std::optional<cooked_artifact_address> parse_cooked_artifact_virtual_path(const io::virtual_path& path);

class package_artifact_provider final : public io::virtual_file_provider
{
public:
    using create_result = core::result<std::shared_ptr<package_artifact_provider>, asset_error>;

    [[nodiscard]] static create_result create(io::virtual_file_system& files, const io::virtual_path& manifest_file);
    ~package_artifact_provider() override;

    package_artifact_provider(const package_artifact_provider&) = delete;
    package_artifact_provider& operator=(const package_artifact_provider&) = delete;

    [[nodiscard]] io::provider_capabilities capabilities() const noexcept override;
    [[nodiscard]] std::uint64_t provider_generation() const noexcept override;
    [[nodiscard]] io::provider_lookup_result resolve(std::string_view relative_path) override;
    [[nodiscard]] jobs::job_future<io::file_result<io::file_buffer>>
    read_range(const io::provider_file& file, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {}) override;
    [[nodiscard]] io::file_result<std::vector<io::provider_directory_entry>>
    enumerate(std::string_view relative_prefix) override;

    [[nodiscard]] const cook_manifest& manifest() const noexcept;

private:
    struct implementation;
    explicit package_artifact_provider(std::unique_ptr<implementation> implementation);
    std::unique_ptr<implementation> implementation_;
};

enum class cooked_artifact_change_kind : std::uint8_t
{
    added,
    updated,
    removed,
    catalog_reset
};

struct cooked_artifact_change_event
{
    std::uint64_t sequence{};
    cooked_artifact_change_kind kind{cooked_artifact_change_kind::updated};
    cooked_artifact_address address;
    std::uint64_t content_generation{};
};

struct cooked_artifact_change_batch
{
    std::uint64_t first_sequence{};
    std::uint64_t last_sequence{};
    bool catalog_reset{};
    std::vector<cooked_artifact_change_event> events;
};

struct cooked_asset_catalog_telemetry
{
    std::uint64_t resolves{};
    std::uint64_t resolve_misses{};
    std::uint64_t change_batches{};
    std::uint64_t resets{};
};

class cooked_asset_catalog
{
public:
    explicit cooked_asset_catalog(io::virtual_file_system& files, io::virtual_path artifact_root);
    ~cooked_asset_catalog();

    [[nodiscard]] asset_status mount_package(const io::virtual_path& manifest_file, std::int32_t priority = 0);
    [[nodiscard]] asset_status mount_provider(std::shared_ptr<io::virtual_file_provider> provider,
                                              std::int32_t priority, std::string debug_name);
    [[nodiscard]] io::file_result<io::resolved_virtual_file> resolve(const cooked_artifact_address& address);
    [[nodiscard]] cooked_artifact_change_batch poll_changes();
    [[nodiscard]] cooked_asset_catalog_telemetry telemetry() const noexcept;

private:
    io::virtual_file_system* files_{};
    io::virtual_path artifact_root_;
    std::vector<io::mount_id> mounts_;
    std::uint64_t last_vfs_sequence_{};
    cooked_asset_catalog_telemetry telemetry_{};
};

struct runtime_update_artifact
{
    cooked_artifact_address address;
    asset_type_id type{};
    std::uint32_t schema_version{};
    content_hash hash{};
    std::uint64_t size{};
    bool tombstone{};
};

struct runtime_update_manifest
{
    static constexpr std::uint32_t current_format_version = 1;
    std::uint32_t format_version{current_format_version};
    std::string target_profile;
    std::string base_build_id;
    std::uint64_t update_sequence{};
    std::vector<runtime_update_artifact> artifacts;
};

struct runtime_update_receiver_config
{
    std::filesystem::path cache_root;
    std::string target_profile;
    std::string base_build_id;
};

struct runtime_update_telemetry
{
    std::uint64_t accepted_updates{};
    std::uint64_t rejected_updates{};
    std::uint64_t staged_blobs{};
    std::uint64_t staged_bytes{};
    std::uint64_t reused_blobs{};
    std::uint64_t active_sequence{};
    std::uint64_t active_artifacts{};
};

class cas_overlay_provider final : public io::virtual_file_provider
{
public:
    cas_overlay_provider(io::async_file_service& files, std::filesystem::path cache_root);
    ~cas_overlay_provider() override;

    cas_overlay_provider(const cas_overlay_provider&) = delete;
    cas_overlay_provider& operator=(const cas_overlay_provider&) = delete;

    [[nodiscard]] asset_status load(std::string_view expected_target_profile, std::string_view expected_base_build_id);
    [[nodiscard]] asset_status activate(const runtime_update_manifest& manifest);

    [[nodiscard]] io::provider_capabilities capabilities() const noexcept override;
    [[nodiscard]] std::uint64_t provider_generation() const noexcept override;
    [[nodiscard]] io::provider_lookup_result resolve(std::string_view relative_path) override;
    [[nodiscard]] jobs::job_future<io::file_result<io::file_buffer>>
    read_range(const io::provider_file& file, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {}) override;
    [[nodiscard]] io::file_result<std::vector<io::provider_directory_entry>>
    enumerate(std::string_view relative_prefix) override;
    [[nodiscard]] std::vector<io::provider_change> poll_changes() override;

    [[nodiscard]] const std::filesystem::path& cache_root() const noexcept;
    [[nodiscard]] std::uint64_t active_sequence() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
    friend class runtime_update_receiver;
};

class runtime_update_receiver
{
public:
    runtime_update_receiver(std::shared_ptr<cas_overlay_provider> overlay, runtime_update_receiver_config config);
    ~runtime_update_receiver();

    runtime_update_receiver(const runtime_update_receiver&) = delete;
    runtime_update_receiver& operator=(const runtime_update_receiver&) = delete;

    [[nodiscard]] asset_status begin(runtime_update_manifest manifest);
    [[nodiscard]] asset_status begin_blob(content_hash hash, std::uint64_t size);
    [[nodiscard]] asset_status stage_blob(std::span<const std::byte> bytes);
    [[nodiscard]] asset_status finish_blob();
    [[nodiscard]] asset_status finish_verify();
    [[nodiscard]] asset_status commit();
    void abort() noexcept;

    [[nodiscard]] asset_status ingest(const runtime_update_manifest& manifest,
                                      std::span<const std::pair<content_hash, io::file_buffer>> blobs);
    [[nodiscard]] runtime_update_telemetry telemetry() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace arc::assets
