#pragma once

#include <arc/jobs/jobs.h>
#include <arc/scene/terrain_evaluator.h>
#include <arc/scene/terrain_render_attributes.h>
#include <arc/scene/terrain_render_geometry.h>

#include <memory>

namespace arc::scene
{

/** Owning CPU products. No renderer or mutable authoring objects cross the worker boundary. */
struct terrain_region_build
{
    terrain_evaluation_result evaluation;
    std::optional<render::virtual_mesh_data> geometry;
    std::optional<terrain_render_attributes> attributes;
};

struct terrain_region_build_batch
{
    std::uint64_t authoring_revision{};
    bool stale{};
    bool succeeded{};
    std::vector<terrain_region_build> regions;
};

/** Providers run on workers and must own their captured source data and return terrain-local surfaces. */
using terrain_region_evaluation_fn = std::function<terrain_evaluation_result(const terrain_asset&, terrain_region_id)>;

/**
 * Single-owner queue: schedule and take_ready run on the publication thread. Workers own immutable
 * snapshots, so abandoning the queue never blocks or accesses a destroyed world. A batch is published
 * as a unit; dirty domains are acknowledged by the caller only after all GPU resources are staged.
 */
class terrain_region_build_queue
{
public:
    bool schedule(jobs::job_system& jobs, const terrain_asset& asset, terrain_region_evaluation_fn evaluate);
    [[nodiscard]] bool pending() const noexcept;
    [[nodiscard]] std::optional<terrain_region_build_batch> take_ready(const terrain_asset& current);

private:
    std::shared_ptr<const terrain_asset> snapshot_;
    std::vector<jobs::job_future<terrain_region_build>> jobs_;
};

} // namespace arc::scene
