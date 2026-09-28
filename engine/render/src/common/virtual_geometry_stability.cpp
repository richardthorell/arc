#include <arc/render/virtual_geometry_stability.h>

#include <algorithm>
#include <cmath>

namespace arc::render
{
namespace
{
constexpr std::uint32_t refinement_history_probe_count = 8u;
}

std::uint32_t virtual_geometry_refinement_key_hash(virtual_geometry_refinement_key key) noexcept
{
    auto value = key.instance_index * 0x9e3779b9u;
    value ^= key.instance_generation * 0x85ebca6bu;
    value ^= key.resource_generation * 0xc2b2ae35u;
    value ^= key.hierarchy_node * 0x27d4eb2fu;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    return value | 1u;
}

virtual_geometry_refinement_history::virtual_geometry_refinement_history(std::uint32_t capacity)
{
    capacity = std::max(capacity, 1u);
    for (auto& generation : generations_)
        generation.resize(capacity);
}

void virtual_geometry_refinement_history::begin_frame()
{
    current_generation_ ^= 1u;
    std::fill(generations_[current_generation_].begin(), generations_[current_generation_].end(), slot{});
    overflowed_[current_generation_] = false;
    frame_started_ = true;
}

bool virtual_geometry_refinement_history::refined_last_frame(virtual_geometry_refinement_key key) const noexcept
{
    const auto previous = current_generation_ ^ 1u;
    if (!frame_started_ || overflowed_[previous]) return false;
    const auto capacity = static_cast<std::uint32_t>(generations_[previous].size());
    const auto start = virtual_geometry_refinement_key_hash(key) % capacity;
    for (std::uint32_t probe = 0u; probe < std::min(capacity, refinement_history_probe_count); ++probe)
    {
        const auto& candidate = generations_[previous][(start + probe) % capacity];
        if (!candidate.occupied) return false;
        if (candidate.key == key) return true;
    }
    return false;
}

bool virtual_geometry_refinement_history::record(virtual_geometry_refinement_key key) noexcept
{
    if (!frame_started_) return false;
    auto& generation = generations_[current_generation_];
    const auto capacity = static_cast<std::uint32_t>(generation.size());
    const auto start = virtual_geometry_refinement_key_hash(key) % capacity;
    for (std::uint32_t probe = 0u; probe < std::min(capacity, refinement_history_probe_count); ++probe)
    {
        auto& candidate = generation[(start + probe) % capacity];
        if (candidate.occupied && candidate.key != key) continue;
        candidate = {.key = key, .occupied = true};
        return true;
    }
    overflowed_[current_generation_] = true;
    return false;
}

bool virtual_geometry_refinement_history::previous_overflowed() const noexcept
{
    return overflowed_[current_generation_ ^ 1u];
}

bool virtual_geometry_refinement_history::current_overflowed() const noexcept
{
    return overflowed_[current_generation_];
}

std::uint32_t virtual_geometry_refinement_history::capacity() const noexcept
{
    return static_cast<std::uint32_t>(generations_[current_generation_].size());
}

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
