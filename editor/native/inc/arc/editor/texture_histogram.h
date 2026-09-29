#pragma once

#include <arc/editor/texture_preview.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace arc::editor
{

inline constexpr std::size_t texture_histogram_bin_count = 256u;

/** @brief Per-channel source-data histogram for one texture mip. */
struct texture_histogram
{
    std::uint32_t mip{};
    std::uint64_t sample_count{};
    std::array<float, 4> minimum{};
    std::array<float, 4> maximum{};
    std::array<std::array<std::uint64_t, texture_histogram_bin_count>, 4> bins{};
    bool valid{};
};

/**
 * @brief Build a histogram from exact decoded source texels for one 2D mip.
 *
 * Each channel owns its observed value range, so float/HDR values remain
 * meaningful rather than being clamped to display space. Callers should run
 * this potentially expensive operation on their existing preview/job worker;
 * the result contains no renderer state and is safe to publish afterwards.
 */
[[nodiscard]] inline texture_histogram build_texture_histogram(const render::texture_data& texture,
                                                               std::uint32_t mip = 0u)
{
    texture_histogram result{.mip = mip};
    if (!texture.has_pixels() || texture.dimension != render::texture_dimension::texture_2d) return result;

    std::uint32_t width = texture.width;
    std::uint32_t height = texture.height;
    if (!texture.mips.empty())
    {
        if (mip >= texture.mips.size()) return result;
        width = texture.mips[mip].width;
        height = texture.mips[mip].height;
    }
    else if (mip != 0u)
        return result;

    if (width == 0u || height == 0u) return result;
    result.minimum.fill(std::numeric_limits<float>::infinity());
    result.maximum.fill(-std::numeric_limits<float>::infinity());

    for (std::uint32_t y = 0u; y < height; ++y)
    {
        for (std::uint32_t x = 0u; x < width; ++x)
        {
            const auto sample = inspect_texture_texel(texture, x, y, mip);
            if (!sample.valid) return {};
            for (std::size_t channel = 0u; channel < sample.rgba.size(); ++channel)
            {
                result.minimum[channel] = std::min(result.minimum[channel], sample.rgba[channel]);
                result.maximum[channel] = std::max(result.maximum[channel], sample.rgba[channel]);
            }
            ++result.sample_count;
        }
    }

    for (std::uint32_t y = 0u; y < height; ++y)
    {
        for (std::uint32_t x = 0u; x < width; ++x)
        {
            const auto sample = inspect_texture_texel(texture, x, y, mip);
            if (!sample.valid) return {};
            for (std::size_t channel = 0u; channel < sample.rgba.size(); ++channel)
            {
                const float range = result.maximum[channel] - result.minimum[channel];
                const float normalized = range > 0.0f ? (sample.rgba[channel] - result.minimum[channel]) / range : 0.0f;
                const auto bin = static_cast<std::size_t>(
                    std::clamp(std::floor(normalized * static_cast<float>(texture_histogram_bin_count - 1u)), 0.0f,
                               static_cast<float>(texture_histogram_bin_count - 1u)));
                ++result.bins[channel][bin];
            }
        }
    }

    result.valid = true;
    return result;
}

} // namespace arc::editor
