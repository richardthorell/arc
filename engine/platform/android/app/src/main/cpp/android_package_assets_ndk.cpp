#include "android_package_assets.h"
#include <android/asset_manager.h>

#include <cstdio>
#include <limits>
#include <memory>
#include <string>

namespace arc::platform::android
{
asset_read_result package_asset_source::read_range(std::string_view path, std::uint64_t offset,
                                                   std::size_t bytes) const
{
    if (!valid_package_asset_path(path)) return {.error = asset_read_error::invalid_path};
    if (!manager_) return {.error = asset_read_error::unavailable};

    // Android's C API takes a NUL-terminated filename.
    const std::string name(path);
    auto* native = AAssetManager_open(static_cast<AAssetManager*>(manager_), name.c_str(), AASSET_MODE_RANDOM);
    if (!native) return {.error = asset_read_error::unavailable};
    const std::unique_ptr<AAsset, decltype(&AAsset_close)> asset(native, &AAsset_close);

    const auto length = AAsset_getLength64(asset.get());
    if (length < 0 || offset > static_cast<std::uint64_t>(length) ||
        bytes > static_cast<std::uint64_t>(length) - offset ||
        bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return {.error = asset_read_error::invalid_range};

    if (AAsset_seek64(asset.get(), static_cast<off64_t>(offset), SEEK_SET) < 0)
        return {.error = asset_read_error::read_failed};
    asset_read_result result;
    result.bytes.resize(bytes);
    std::size_t read = 0;
    while (read < bytes)
    {
        const auto count = AAsset_read(asset.get(), result.bytes.data() + read, static_cast<int>(bytes - read));
        if (count <= 0) return {.error = asset_read_error::read_failed};
        read += static_cast<std::size_t>(count);
    }
    return result;
}
} // namespace arc::platform::android
