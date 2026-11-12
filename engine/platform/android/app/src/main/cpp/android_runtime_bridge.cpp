#include "android_runtime_bridge.h"
#include "android_apk_provider.h"

#include <arc/diagnostics/log.h>
#include <arc/framework/capabilities.h>
#include <arc/scene/runtime_scene.h>
#include <arc/render/mesh_artifact.h>
#include <arc/render/cooked_material.h>
#include <arc/render/cooked_texture.h>

#include <filesystem>
#include <string>
#include <set>
#include <vector>
#include <utility>

namespace arc::platform::android
{
namespace
{
constexpr auto package_root = "package://runtime";

using decoded_result = arc::core::result<std::vector<decoded_scene_mesh_resources>, std::string>;

/** Pure CPU payload preparation. The session enforces active-provider authority
 * and resolves each GUID through the packaged catalog, never an APK path hint.
 * No ECS handles or renderer resources are modified on a partial failure.
 */
decoded_result decode_scene_resources(arc::project::runtime_catalog_session& catalog,
                                      const arc::scene::cooked_scene_mesh_bindings& bindings)
{
    namespace assets = arc::assets;
    namespace render = arc::render;
    std::vector<decoded_scene_mesh_resources> decoded;
    decoded.reserve(bindings.size());

    for (const auto& binding : bindings)
    {
        const auto owner = assets::to_string(binding.source.guid);
        if (!binding.source.resolved())
            return decoded_result::failure("Scene mesh binding has no authored source GUID");

        auto mesh_bytes = catalog.read_active_artifact(binding.source.guid, binding.source.expected_type,
                                                       assets::artifact_schemas::mesh);
        if (!mesh_bytes) return decoded_result::failure("Mesh " + owner + ": " + mesh_bytes.error().message);
        auto mesh = render::decode_conventional_mesh_artifact(mesh_bytes.value(), binding.subresource);
        if (!mesh) return decoded_result::failure("Mesh " + owner + ": " + mesh.error());

        decoded_scene_mesh_resources resources;
        resources.entity = binding.entity;
        resources.mesh = std::move(mesh).value().mesh;

        if (binding.material.resolved())
        {
            const auto material_guid = binding.material.guid;
            if (binding.material.expected_type != assets::asset_types::material)
                return decoded_result::failure("Material " + assets::to_string(material_guid) +
                                               ": only conventional cooked Material v4 is supported");

            auto material_bytes = catalog.read_active_artifact(material_guid, assets::asset_types::material,
                                                               assets::artifact_schemas::material);
            if (!material_bytes)
                return decoded_result::failure("Material " + assets::to_string(material_guid) + ": " +
                                               material_bytes.error().message);
            auto material = render::decode_cooked_material_metadata(material_bytes.value());
            if (!material)
                return decoded_result::failure("Material " + assets::to_string(material_guid) + ": " + material.error());
            resources.material = std::move(material).value();

            // The existing cooker tracks material dependencies by stable GUID.
            // Only texture dependencies are decoded here; shaders remain metadata
            // until V3.4d creates a renderer-owned shader/material program.
            const auto record = catalog.catalog().database().query(material_guid, assets::asset_types::material);
            if (!record) return decoded_result::failure("Material disappeared from the active cooked catalog");

            std::set<assets::asset_guid> visited_textures;
            for (const auto& dependency : record->dependencies)
            {
                if (!catalog.catalog().database().query(dependency, assets::asset_types::texture_2d)) continue;
                if (!visited_textures.insert(dependency).second) continue;

                auto texture_bytes = catalog.read_active_artifact(dependency, assets::asset_types::texture_2d,
                                                                  assets::artifact_schemas::texture);
                if (!texture_bytes)
                    return decoded_result::failure("Texture " + assets::to_string(dependency) + ": " +
                                                   texture_bytes.error().message);
                auto texture = render::decode_cooked_texture_data(texture_bytes.value());
                if (!texture)
                    return decoded_result::failure("Texture " + assets::to_string(dependency) + ": " + texture.error());
                resources.textures.emplace_back(dependency, std::move(texture).value());
            }
        }
        decoded.push_back(std::move(resources));
    }
    return decoded_result::success(std::move(decoded));
}
} // namespace

android_runtime_bridge::android_runtime_bridge(const package_asset_source& assets, const char* storage_path)
    : assets_(assets), jobs_(arc::jobs::job_system_config{.worker_count = 1, .io_worker_count = 1}), vfs_(jobs_),
      writable_storage_(storage_path ? std::filesystem::path(storage_path).lexically_normal() : std::filesystem::path{})
{
}

android_runtime_bridge::~android_runtime_bridge()
{
    decoded_resources_.clear();
    scene_.reset();
    mesh_bindings_.clear();
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
        catalog_.reset();
        boot_.reset();
        return false;
    }
    auto next = std::make_unique<arc::ecs::world>();
    arc::scene::cooked_scene_mesh_bindings next_bindings;
    const auto failure = arc::scene::materialize_cooked_scene(*next, bytes.value(), &next_bindings);
    if (!failure.empty())
    {
        arc::diagnostics::error("Android", failure);
        catalog_.reset();
        boot_.reset();
        return false;
    }
    auto next_resources = decode_scene_resources(*catalog_, next_bindings);
    if (!next_resources)
    {
        arc::diagnostics::error("Android", "Cooked startup resource decoding failed: " + next_resources.error());
        catalog_.reset();
        boot_.reset();
        return false;
    }
    scene_ = std::move(next);
    mesh_bindings_ = std::move(next_bindings);
    decoded_resources_ = std::move(next_resources).value();
    arc::diagnostics::info("Android", "Cooked startup scene materialized into ECS world");
    arc::diagnostics::info("Android", "Cooked startup mesh/material/texture CPU resources decoded");
    return true;
}
} // namespace arc::platform::android
