#include <arc/editor/material_preview_jobs.h>

#include <utility>

namespace arc::editor
{

jobs::job_future<material_preview_result>
render_material_preview_async(jobs::job_system& job_system, material_asset asset, std::filesystem::path asset_root,
                              std::uint32_t size, jobs::cancellation_token cancellation)
{
    return job_system.submit_future({.name = "editor.material_thumbnail",
                                     .priority = jobs::job_priority::background,
                                     .affinity = jobs::job_affinity::any_worker,
                                     .cancellation = std::move(cancellation)},
                                    [asset = std::move(asset), asset_root = std::move(asset_root), size]()
                                    { return render_material_preview(asset, asset_root, size); });
}

} // namespace arc::editor
