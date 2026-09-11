#pragma once

#include <arc/scene/terrain.h>
#include <arc/scene/terrain_region_build.h>

#include <filesystem>

namespace arc::editor
{

/** Editor-owned build state, discarded with its world/asset binding. No worker captures this object. */
class terrain_rebuild_session
{
public:
    terrain_rebuild_session(scene::terrain_asset asset, const scene::terrain_component& terrain,
                            std::filesystem::path source_path = {});
    void update(scene::terrain_asset asset, bool invalidate_all = false);
    /** Poll without waiting, stage GPU resources, then acknowledge the published dirty domains. */
    bool pump(jobs::job_system& jobs, scene::terrain_render_proxy_cache& proxies, ecs::entity_guid guid,
              scene::terrain_component& terrain, render::renderer& renderer);
    [[nodiscard]] const std::string& error() const noexcept
    {
        return error_;
    }
    [[nodiscard]] const scene::terrain_asset& asset() const noexcept
    {
        return asset_;
    }

private:
    scene::terrain_asset asset_;
    scene::terrain_region_build_queue queue_;
    scene::terrain_region_evaluation_fn evaluate_;
    std::optional<scene::terrain_region_build_batch> ready_;
    float size_{};
    double spacing_{};
    std::uint64_t failed_revision_{};
    std::string error_;
};

} // namespace arc::editor
