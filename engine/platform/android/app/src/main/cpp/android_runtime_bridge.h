#pragma once

#include "android_package_assets.h"

#include <arc/io/virtual_file_system.h>
#include <arc/jobs/jobs.h>
#include <arc/project/runtime_boot.h>
#include <arc/project/runtime_catalog.h>

#include <memory>

namespace arc::platform::android
{
// Bootstraps exactly once per Activity process. Surface loss doesn't unmount
// content; the host releases this session only on process teardown.
class android_runtime_bridge final
{
public:
    android_runtime_bridge(const package_asset_source& assets, const char* storage_path);
    ~android_runtime_bridge();

    android_runtime_bridge(const android_runtime_bridge&) = delete;
    android_runtime_bridge& operator=(const android_runtime_bridge&) = delete;

    [[nodiscard]] bool boot();

private:
    // Order is intentional: mount and boot context are destroyed before the
    // VFS/job scheduler; all submitted reads complete before AssetManager dies.
    const package_asset_source& assets_;
    arc::jobs::job_system jobs_;
    arc::io::virtual_file_system vfs_;
    std::filesystem::path writable_storage_;
    arc::io::mount_id mount_{};
    std::shared_ptr<arc::io::virtual_file_provider> provider_;
    std::unique_ptr<arc::project::runtime_boot_context> boot_;
    std::unique_ptr<arc::project::runtime_catalog_session> catalog_;
};
} // namespace arc::platform::android
