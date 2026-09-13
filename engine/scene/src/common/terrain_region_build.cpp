#include <arc/scene/terrain_region_build.h>

#include <algorithm>
#include <utility>

namespace arc::scene
{

bool terrain_region_build_queue::schedule(jobs::job_system& jobs, const terrain_asset& asset,
                                          terrain_region_evaluation_fn evaluate)
{
    if (pending() || !evaluate) return false;
    snapshot_ = std::make_shared<const terrain_asset>(asset);
    for (const auto& region : asset.regions)
    {
        const auto render_domains = terrain_domain::geometry | terrain_domain::topology | terrain_domain::attributes;
        const auto domains = region.dirty_domains & render_domains;
        if (domains == terrain_domain::none) continue;
        jobs_.push_back(jobs.submit_future(
            {.name = "terrain.region.build"},
            [snapshot = snapshot_, evaluate, id = region.id, domains]
            {
                terrain_region_build result;
                result.evaluation = evaluate(*snapshot, id);
                result.domains = domains;
                if (result.evaluation.succeeded)
                {
                    const auto geometry_domains = terrain_domain::geometry | terrain_domain::topology;
                    if ((domains & geometry_domains) != terrain_domain::none)
                        result.geometry = result.evaluation.vertex_normals.empty()
                                              ? build_terrain_render_geometry(result.evaluation.surface.view())
                                              : build_terrain_render_region_geometry(result.evaluation.surface.view(),
                                                                                     result.evaluation.vertex_normals);
                    result.attributes = build_terrain_render_attributes(result.evaluation.surface.view());
                }
                return result;
            }));
    }
    if (jobs_.empty()) snapshot_.reset();
    return pending();
}

bool terrain_region_build_queue::pending() const noexcept
{
    return !jobs_.empty();
}

std::optional<terrain_region_build_batch> terrain_region_build_queue::take_ready(const terrain_asset& current)
{
    if (!pending() || !std::all_of(jobs_.begin(), jobs_.end(), [](const auto& job) { return job.ready(); }))
        return std::nullopt;
    terrain_region_build_batch result;
    result.authoring_revision = snapshot_->authoring_revision;
    // Stack/source changes and dependency edits also invalidate results, not just the target revision.
    result.stale =
        current.authoring_revision != snapshot_->authoring_revision || current.source.id != snapshot_->source.id;
    result.succeeded = !result.stale;
    for (auto& job : jobs_)
    {
        if (job.status() != jobs::job_status::succeeded)
        {
            result.succeeded = false;
            continue;
        }
        auto region = job.get();
        const auto fresh = make_terrain_build_region_snapshot(current, region.evaluation.region);
        if (fresh.target_dirty_revision != region.evaluation.build_snapshot.target_dirty_revision) result.stale = true;
        const auto geometry_domains = terrain_domain::geometry | terrain_domain::topology;
        const bool geometry_ready = (region.domains & geometry_domains) == terrain_domain::none || region.geometry;
        const bool attributes_ready =
            (region.domains & (geometry_domains | terrain_domain::attributes)) == terrain_domain::none ||
            region.attributes;
        result.succeeded =
            result.succeeded && !result.stale && region.evaluation.succeeded && geometry_ready && attributes_ready;
        result.regions.push_back(std::move(region));
    }
    jobs_.clear();
    snapshot_.reset();
    return result;
}

} // namespace arc::scene
