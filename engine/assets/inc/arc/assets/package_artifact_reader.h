#pragma once

#include <arc/assets/cook.h>
#include <arc/io/virtual_file_system.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::assets
{

/** @brief Stable address for one named artifact inside a cooked asset package. */
struct cooked_artifact_address
{
    asset_guid asset{};
    artifact_schema_id schema{};
    std::string name;

    [[nodiscard]] bool valid() const noexcept
    {
        return asset.valid() && schema.valid() && !name.empty();
    }

    friend bool operator==(const cooked_artifact_address&, const cooked_artifact_address&) = default;
};

/// Validated logical location and byte range for one named cooked artifact.
/// `path` is retained temporarily for callers using the native-path compatibility mount. Runtime mounts populate
/// `file` and do not expose their provider's physical package path or offset.
struct cooked_artifact_location
{
    io::resolved_virtual_file file;
    std::filesystem::path path;
    std::uint64_t offset{};
    std::uint64_t size{};

    [[nodiscard]] bool valid() const noexcept
    {
        return (file.valid() || !path.empty()) && size != 0u;
    }
};

/// Metadata-first reader for independently addressable cooked artifacts.
/// Mounting reads only the cook manifest and filesystem metadata. Artifact bytes remain on disk until an explicit
/// read or range read is requested. Runtime mounts return immutable VFS handles; the native-path overload remains a
/// temporary compatibility boundary for authoring and downstream code that has not migrated yet.
class package_artifact_reader
{
public:
    ~package_artifact_reader();

    [[nodiscard]] asset_status mount(const std::filesystem::path& manifest_path);
    [[nodiscard]] asset_status mount(io::virtual_file_system& files, const io::virtual_path& manifest_path,
                                     io::virtual_path artifact_root, std::int32_t priority = 0);

    [[nodiscard]] const cook_manifest_artifact* find(const cooked_artifact_address& address) const noexcept;
    [[nodiscard]] std::optional<cooked_artifact_location> locate(const cooked_artifact_address& address) const noexcept;
    [[nodiscard]] core::result<std::vector<std::byte>, asset_error> read(const cooked_artifact_address& address) const;
    [[nodiscard]] core::result<std::vector<std::byte>, asset_error>
    read_range(const cooked_artifact_address& address, std::uint64_t relative_offset, std::uint64_t size) const;

    [[nodiscard]] const cook_manifest& manifest() const noexcept;
    [[nodiscard]] std::uint64_t bytes_read() const noexcept;
    void reset_statistics() noexcept;

private:
    void release_virtual_mount() noexcept;

    std::filesystem::path root_;
    cook_manifest manifest_;
    io::virtual_file_system* virtual_files_{};
    io::virtual_path artifact_root_;
    io::mount_id virtual_mount_{};
    std::shared_ptr<io::virtual_file_provider> virtual_provider_;
    mutable std::atomic<std::uint64_t> bytes_read_{};
};

} // namespace arc::assets
