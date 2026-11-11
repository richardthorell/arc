#pragma once

#include "android_package_assets.h"

#include <arc/io/virtual_file_system.h>
#include <arc/ecs/world.h>
#include <arc/jobs/jobs.h>
#include <arc/project/runtime_boot.h>
#include <arc/project/runtime_catalog.h>
#include <arc/scene/runtime_scene.h>
#include <arc/render/cooked_material.h>
#include <arc/render/mesh.h>
#include <arc/render/material.h>

#include <optional>
#include <vector>

#include <memory>

namespace arc::platform::android
{
// Bootstraps exactly once per Activity process. Surface loss doesn't unmount
// content; the host releases this session only on process teardown.
struct decoded_scene_mesh_resources
{
    arc::ecs::entity_guid entity{};
    arc::render::mesh_data mesh;
    std::optional<arc::render::cooked_material_metadata> material;
    std::vector<std::pair<arc::assets::asset_guid, arc::render::texture_data>> textures;
};

class android_runtime_bridge final
{
public:
    android_runtime_bridge(const package_asset_source& assets, const char* storage_path);
    ~android_runtime_bridge();

    android_runtime_bridge(const android_runtime_bridge&) = delete;
    android_runtime_bridge& operator=(const android_runtime_bridge&) = delete;

    [[nodiscard]] bool boot();
    [[nodiscard]] arc::ecs::world* startup_scene() noexcept
    {
        return scene_.get();
    }
    [[nodiscard]] const std::vector<decoded_scene_mesh_resources>& decoded_mesh_resources() const noexcept
    {
        return decoded_resources_;
    }
    [[nodiscard]] const arc::scene::cooked_scene_mesh_bindings& startup_mesh_bindings() const noexcept
    {
        return mesh_bindings_;
    }

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
    std::unique_ptr<arc::ecs::world> scene_;
    arc::scene::cooked_scene_mesh_bindings mesh_bindings_;
    // CPU-side data outlives surface generations; no Vulkan handles here.
    std::vector<decoded_scene_mesh_resources> decoded_resources_;
};
} // namespace arc::platform::android
