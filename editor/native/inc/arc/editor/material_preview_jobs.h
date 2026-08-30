#pragma once

#include <arc/editor/material_preview.h>
#include <arc/jobs/jobs.h>

#include <cstdint>
#include <filesystem>

namespace arc::editor
{

/**
 * @brief Schedule material preview generation on the shared ARC job system.
 *
 * The existing synchronous renderer remains the primitive used by the host today.
 * This entry point gives the host a queue-ready API so thumbnail requests can move
 * off the editor/viewport command path without introducing a second worker pool.
 */
jobs::job_future<material_preview_result>
render_material_preview_async(jobs::job_system& job_system, material_asset asset, std::filesystem::path asset_root,
                              std::uint32_t size = 128, jobs::cancellation_token cancellation = {});

} // namespace arc::editor
