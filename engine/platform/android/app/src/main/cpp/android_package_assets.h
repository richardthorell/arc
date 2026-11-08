#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace arc::platform::android
{
enum class asset_read_error : std::uint8_t { none, unavailable, invalid_path, invalid_range, read_failed };

struct asset_read_result
{
    asset_read_error error{asset_read_error::none};
    std::vector<std::byte> bytes;
    [[nodiscard]] explicit operator bool() const noexcept { return error == asset_read_error::none; }
};

[[nodiscard]] bool valid_package_asset_path(std::string_view path) noexcept;

// Borrows the Activity's AAssetManager. Packaged assets are not filesystem files.
class package_asset_source
{
public:
    explicit package_asset_source(void* manager) noexcept : manager_(manager) {}
    [[nodiscard]] bool available() const noexcept { return manager_ != nullptr; }
    [[nodiscard]] bool file_size(std::string_view path, std::uint64_t& size) const;
    [[nodiscard]] asset_read_result read_range(std::string_view path, std::uint64_t offset,
                                               std::size_t bytes) const;
private:
    void* manager_{};
};
} // namespace arc::platform::android
