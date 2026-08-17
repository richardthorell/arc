#pragma once

#include <cstdint>

namespace arc::render
{
class renderer;
}

namespace arc::editor
{

struct editor_scene_state;

/**
 * @brief Lightweight render telemetry exposed by the editor viewport.
 */
struct viewport_render_stats
{
    std::uint64_t triangles{};
    std::uint64_t vertices{};
    std::uint64_t gpu_memory_used_bytes{};
    std::uint64_t gpu_memory_budget_bytes{};
    bool gpu_memory_available{};
};

/**
 * @brief Collect geometry and GPU-memory telemetry for the latest editor viewport frame.
 *
 * Conventional mesh geometry is counted from renderer-owned CPU mesh metadata.
 * Backend frame profiling contributes runtime-selected virtual-geometry and
 * terrain triangle counts. GPU memory is reported when the active backend
 * exposes a current heap-usage snapshot.
 */
viewport_render_stats collect_viewport_render_stats(const editor_scene_state& scene,
                                                    const render::renderer& renderer) noexcept;

} // namespace arc::editor
