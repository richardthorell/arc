#include "android_runtime_bridge.h"
#include "android_apk_provider.h"

#include <arc/diagnostics/log.h>
#include <arc/framework/capabilities.h>
#include <arc/scene/runtime_scene.h>

#include <filesystem>
#include <string>
#include <utility>

namespace arc::platform::android
{
namespace
{
constexpr auto package_root = "package://runtime";
}

android_runtime_bridge::android_runtime_bridge(const package_asset_source& assets, const char* storage_path)
    : assets_(assets), jobs_(arc::jobs::job_system_config{.worker_count = 1, .io_worker_count = 1}), vfs_(jobs_),
      writable_storage_(storage_path ? std::filesystem::path(storage_path).lexically_normal() : std::filesystem::path{})
{
}

android_runtime_bridge::~android_runtime_bridge()
{
    scene_.reset();
    catalog_.reset();
    boot_.reset();
    if (mount_)
    {
        auto result = vfs_.unmount(mount_);
        if (!result) arc::diagnostics::warn("Android", "Could not unmount APK distribution");
    }
    provider_.reset();
    jobs_.shutdown(arc::jobs::job_shutdown_mode::drain);
}

bool android_runtime_bridge::boot()
{
    if (catalog_) return true;
    if (!assets_.available() || writable_storage_.empty() || !writable_storage_.is_absolute())
    {
        arc::diagnostics::error("Android", "Missing APK asset provider or app-private writable storage");
        return false;
    }

    auto root = arc::io::virtual_path::parse(package_root);
    if (!root)
    {
        arc::diagnostics::error("Android", "Invalid ARC package mount root");
        return false;
    }

    if (!mount_)
    {
        provider_ = std::make_shared<android_apk_provider>(jobs_, assets_);
        auto mounted = vfs_.mount({.root = root.value(), .provider = provider_, .debug_name = "Android APK assets"});
        if (!mounted)
        {
            arc::diagnostics::error("Android", mounted.error().message);
            provider_.reset();
            return false;
        }
        mount_ = mounted.value();
    }

    arc::project::runtime_boot_request request;
    request.role = arc::project::runtime_launch_role::client;
    request.host.distribution = {.kind = arc::project::runtime_distribution_location_kind::mounted_vfs,
                                 .files = &vfs_,
                                 .virtual_root = root.value()};
    request.host.writable_storage_root = writable_storage_;
    request.host.capabilities = {.family = arc::framework::platform_family::android,
                                 .form_factor = arc::framework::device_form_factor::handheld,
                                 .window_system = true,
                                 .high_dpi = true,
                                 .dynamic_libraries = true,
                                 .persistent_local_storage = true,
                                 .native_package_assets = true};
    request.expected.engine_version = ARC_ANDROID_ENGINE_VERSION;
    request.expected.sdk_version = ARC_ANDROID_SDK_VERSION;
    request.expected.architecture = "arm64";
    request.expected.renderer = "vulkan";
    request.expected.configuration = ARC_ANDROID_BUILD_CONFIGURATION;
    request.expected.target_id = "android-arm64-vulkan";

    auto discovered = arc::project::discover_runtime_boot(std::move(request));
    if (!discovered)
    {
        const auto& failure = discovered.error();
        arc::diagnostics::error("Android", "Packaged runtime boot failed:");
        arc::diagnostics::error("Android", failure.field);
        arc::diagnostics::error("Android", failure.message);
        return false;
    }
    boot_ = std::make_unique<arc::project::runtime_boot_context>(std::move(discovered).value());
    arc::diagnostics::info("Android", "ARC packaged distribution manifest verified");
    auto mounted_catalog = arc::project::bootstrap_runtime_catalog({.boot = *boot_, .files = vfs_});
    if (!mounted_catalog)
    {
        const auto& failure = mounted_catalog.error();
        arc::diagnostics::error("Android", "Packaged cooked catalog bootstrap failed:");
        arc::diagnostics::error("Android", failure.field);
        arc::diagnostics::error("Android", failure.message);
        boot_.reset();
        return false;
    }
    catalog_ = std::move(mounted_catalog).value();
    arc::diagnostics::info("Android", "ARC cooked startup catalog mounted");
    if (catalog_->startup_scenes().empty())
    {
        arc::diagnostics::warn("Android", "Distribution has no startup scene");
        return true;
    }

    const auto guid = catalog_->startup_scenes().front();
    constexpr auto type = arc::assets::asset_types::scene;
    constexpr auto schema = arc::assets::artifact_schemas::scene;
    auto bytes = catalog_->read_active_artifact(guid, type, schema);
    if (!bytes)
    {
        arc::diagnostics::error("Android", "Cannot load active cooked startup scene: " + bytes.error().message);
        return false;
    }
    auto next = std::make_unique<arc::ecs::world>();
    const auto failure = arc::scene::materialize_cooked_scene(*next, bytes.value());
    if (!failure.empty())
    {
        arc::diagnostics::error("Android", failure);
        return false;
    }
    scene_ = std::move(next);
    arc::diagnostics::info("Android", "Cooked startup scene materialized into ECS world");
    return true;
}
} // namespace arc::platform::android
