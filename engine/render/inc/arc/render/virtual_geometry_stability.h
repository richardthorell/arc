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

[[nodiscard]] constexpr virtual_geometry_history_invalidation
operator|(virtual_geometry_history_invalidation lhs, virtual_geometry_history_invalidation rhs) noexcept
{
    return static_cast<virtual_geometry_history_invalidation>(static_cast<std::uint32_t>(lhs) |
                                                               static_cast<std::uint32_t>(rhs));
}

constexpr virtual_geometry_history_invalidation&
operator|=(virtual_geometry_history_invalidation& lhs, virtual_geometry_history_invalidation rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

[[nodiscard]] constexpr bool contains(virtual_geometry_history_invalidation value,
                                      virtual_geometry_history_invalidation requested) noexcept
{
    return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(requested)) != 0u;
}

/** @brief Which depth history a virtual-geometry traversal phase may consume. */
enum class virtual_geometry_traversal_phase : std::uint8_t
{
    single_phase,
    previous_hzb,
    current_hzb
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

/** @brief Use hysteresis when history is valid, otherwise use the nominal threshold. */
[[nodiscard]] bool should_refine_virtual_geometry(float projected_error, float threshold, bool refined_last_frame,
                                                  bool history_valid,
                                                  virtual_geometry_traversal_stability policy = {}) noexcept;

} // namespace arc::render
