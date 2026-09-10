#pragma once

#include <arc/assets/cook.h>

#include <cstdint>
#include <filesystem>
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

/**
 * @brief Metadata-first reader for independently addressable cooked artifacts.
 *
 * Mounting reads only the cook manifest and filesystem metadata. Artifact bytes remain on disk until an explicit
 * read or range read is requested. This is intentionally synchronous; scheduling and predictive IO policy belong to
 * higher-level streaming systems.
 */
class package_artifact_reader
{
public:
    [[nodiscard]] asset_status mount(const std::filesystem::path& manifest_path);

    [[nodiscard]] const cook_manifest_artifact* find(const cooked_artifact_address& address) const noexcept;
    [[nodiscard]] core::result<std::vector<std::byte>, asset_error>
    read(const cooked_artifact_address& address) const;
    [[nodiscard]] core::result<std::vector<std::byte>, asset_error>
    read_range(const cooked_artifact_address& address, std::uint64_t relative_offset, std::uint64_t size) const;

    [[nodiscard]] const cook_manifest& manifest() const noexcept;
    [[nodiscard]] std::uint64_t bytes_read() const noexcept;
    void reset_statistics() noexcept;

private:
    std::filesystem::path root_;
    cook_manifest manifest_;
    mutable std::uint64_t bytes_read_{};
};

} // namespace arc::assets
