#include <arc/render/gpu_driven.h>

#include <algorithm>
#include <cmath>

namespace arc::render
{

std::vector<gpu_table_dirty_range> coalesce_gpu_table_dirty_ranges(std::span<const std::uint32_t> indices)
{
    if (indices.empty()) return {};

    std::vector<std::uint32_t> sorted(indices.begin(), indices.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    std::vector<gpu_table_dirty_range> ranges;
    ranges.reserve(sorted.size());
    std::uint32_t first = sorted.front();
    std::uint32_t previous = first;
    for (std::size_t index = 1; index < sorted.size(); ++index)
    {
        const auto current = sorted[index];
        if (current != previous + 1u)
        {
            ranges.push_back({.first = first, .count = previous - first + 1u});
            first = current;
        }
        previous = current;
    }
    ranges.push_back({.first = first, .count = previous - first + 1u});
    return ranges;
}

std::uint64_t make_gpu_transparent_sort_key(float normalized_depth, std::uint16_t pipeline_bin,
                                            std::uint32_t stable_instance_index) noexcept
{
    const auto finite_depth = std::isfinite(normalized_depth) ? normalized_depth : 0.0f;
    const auto depth = static_cast<std::uint32_t>(std::clamp(finite_depth, 0.0f, 1.0f) * 16'777'215.0f);
    const auto descending_depth = 0x00ffffffu - depth;
    return (static_cast<std::uint64_t>(pipeline_bin) << 48u) | (static_cast<std::uint64_t>(descending_depth) << 24u) |
           static_cast<std::uint64_t>(stable_instance_index & 0x00ffffffu);
}

} // namespace arc::render
