#pragma once

#include "android_package_assets.h"

#include <arc/io/virtual_file_system.h>

namespace arc::platform::android
{
// AssetManager is borrowed; host must unmount and destroy the provider
// before NativeActivity destroys its AAssetManager.
class android_apk_provider final : public arc::io::virtual_file_provider
{
public:
    android_apk_provider(arc::jobs::job_system& jobs, const package_asset_source& source)
        : jobs_(&jobs), source_(&source) {}

    [[nodiscard]] arc::io::provider_capabilities capabilities() const noexcept override;
    [[nodiscard]] std::uint64_t provider_generation() const noexcept override { return 1; }
    [[nodiscard]] arc::io::provider_lookup_result resolve(std::string_view relative_path) override;
    [[nodiscard]] arc::jobs::job_future<arc::io::file_result<arc::io::file_buffer>>
    read_range(const arc::io::provider_file& file, std::uint64_t offset, std::size_t bytes,
               arc::jobs::cancellation_token cancellation = {}) override;

private:
    arc::jobs::job_system* jobs_;
    const package_asset_source* source_;
};
} // namespace arc::platform::android
