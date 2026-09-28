#include <arc/render/virtual_geometry_stability.h>

#include <algorithm>
#include <cmath>

namespace arc::render
{

bool virtual_geometry_history_valid(virtual_geometry_history_invalidation invalidation) noexcept
{
    return invalidation == virtual_geometry_history_invalidation::none;
}

bool should_refine_virtual_geometry(float projected_error, float threshold, bool refined_last_frame,
                                    virtual_geometry_traversal_stability policy) noexcept
{
    return should_refine_virtual_geometry(projected_error, threshold, refined_last_frame, true, policy);
}

bool should_refine_virtual_geometry(float projected_error, float threshold, bool refined_last_frame,
                                    bool history_valid, virtual_geometry_traversal_stability policy) noexcept
{
    if (!std::isfinite(projected_error) || !std::isfinite(threshold) || threshold <= 0.0f) return false;
    if (!history_valid) return projected_error > threshold;

    const auto refine_hysteresis = std::max(policy.refine_hysteresis, 0.0f);
    const auto coarsen_hysteresis = std::max(policy.coarsen_hysteresis, 0.0f);
    const auto refine_threshold = threshold * (1.0f + refine_hysteresis);
    const auto coarsen_threshold = threshold * std::max(0.0f, 1.0f - coarsen_hysteresis);

    return refined_last_frame ? projected_error > coarsen_threshold : projected_error > refine_threshold;
}

} // namespace arc::render
