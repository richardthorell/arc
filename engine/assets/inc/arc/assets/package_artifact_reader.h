#pragma once

#include <arc/assets/cook.h>

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
};

/** @brief Validated physical package location for one named cooked artifact. */
struct cooked_artifact_location
{
    std::filesystem::path path;
    std::uint64_t offset{};
    std::uint64_t size{};

    [[nodiscard]] bool valid() const noexcept
    {
        return !path.empty() && size != 0u;
    }
};

/**
 * @brief Metadata-first reader for independently addressable cooked artifacts.
 *
 * Mounting reads only the cook manifest and filesystem metadata. Artifact bytes remain on disk until an explicit
 * read or range read is requested. The validated physical location can also be handed to an async IO service without
 * performing synchronous reads on the render thread.
 */
class package_artifact_reader
{
public:
    [[nodiscard]] asset_status mount(const std::filesystem::path& manifest_path);

    [[nodiscard]] const cook_manifest_artifact* find(const cooked_artifact_address& address) const noexcept;
    [[nodiscard]] std::optional<cooked_artifact_location>
    locate(const cooked_artifact_address& address) const noexcept;
    [[nodiscard]] core::result<std::vector<std::byte>, asset_error> read(const cooked_artifact_address& address) const;
    [[nodiscard]] core::result<std::vector<std::byte>, asset_error>
    read_range(const cooked_artifact_address& address, std::uint64_t relative_offset, std::uint64_t size) const;

    [[nodiscard]] const cook_manifest& manifest() const noexcept;
    [[nodiscard]] std::uint64_t bytes_read() const noexcept;
    void reset_statistics() noexcept;

private:
    std::filesystem::path root_;
    cook_manifest manifest_;
    mutable std::atomic<std::uint64_t> bytes_read_{};
};

} // namespace arc::assets
