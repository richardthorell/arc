#pragma once

#include <cstdint>

namespace arc::render
{

/** @brief Reasons previous-frame occlusion data must not reject geometry this frame. */
enum class virtual_geometry_history_invalidation : std::uint32_t
{
    none = 0u,
    camera_cut = 1u << 0u,
    teleport = 1u << 1u,
    viewport_resize = 1u << 2u,
    projection_change = 1u << 3u,
    newly_visible_instance = 1u << 4u
};

/** @brief Backend-neutral policy for temporal HZB use and projected-error hysteresis. */
struct virtual_geometry_traversal_stability
{
    float refine_hysteresis{0.10f};
    float coarsen_hysteresis{0.10f};
};

/** @brief Returns true when previous-frame HZB rejection is safe for the current view/instance. */
[[nodiscard]] bool virtual_geometry_history_valid(virtual_geometry_history_invalidation invalidation) noexcept;

/**
 * @brief Stable projected-error refinement decision shared by CPU reference and GPU implementations.
 * @param projected_error Current projected geometric error in pixels.
 * @param threshold Nominal refinement threshold in pixels.
 * @param refined_last_frame Whether this node was refined in the previous valid-history frame.
 * @param policy Hysteresis widths expressed as fractions of the nominal threshold.
 */
[[nodiscard]] bool should_refine_virtual_geometry(float projected_error, float threshold, bool refined_last_frame,
                                                  virtual_geometry_traversal_stability policy = {}) noexcept;

} // namespace arc::render
