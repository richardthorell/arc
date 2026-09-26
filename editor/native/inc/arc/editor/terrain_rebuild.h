#pragma once

#include <arc/scene/terrain.h>
#include <arc/scene/terrain_region_build.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace arc::editor
{

enum class terrain_rebuild_phase : std::uint8_t
{
    idle,
    queued,
    building,
    publishing,
    failed
};

struct terrain_rebuild_region_status
{
    scene::terrain_region_id id{};
    scene::terrain_world_bounds bounds{};
    scene::terrain_domain domains{scene::terrain_domain::none};
};

struct terrain_rebuild_status
{
    terrain_rebuild_phase phase{terrain_rebuild_phase::idle};
    std::uint64_t authoring_revision{};
    std::uint32_t geometry_regions{};
    std::uint32_t attribute_regions{};
    std::vector<terrain_rebuild_region_status> dirty_regions;
    std::string error;
};

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
    [[nodiscard]] terrain_rebuild_status status() const;

private:
    scene::terrain_asset asset_;
    scene::terrain_region_build_queue queue_;
    scene::terrain_region_evaluation_fn evaluate_;
    std::optional<scene::terrain_region_build_batch> ready_;
    float size_{};
    double spacing_{};
    std::uint64_t failed_revision_{};
    bool replace_all_regions_{true};
    std::string error_;
};

} // namespace arc::editor
