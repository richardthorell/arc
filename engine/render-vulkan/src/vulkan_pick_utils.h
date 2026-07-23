#pragma once

#include <algorithm>
#include <cstdint>

namespace arc::render::vulkan::detail
{

constexpr std::uint32_t map_output_pixel_to_render_pixel(
    std::uint32_t coordinate,
    std::uint32_t output_extent,
    std::uint32_t render_extent) noexcept
{
    if (output_extent == 0 || render_extent == 0)
        return 0;

    // Match the pixel-center mapping used by the internal-to-output blit.
    const auto mapped = ((static_cast<std::uint64_t>(coordinate) * 2u + 1u) * render_extent) /
        (static_cast<std::uint64_t>(output_extent) * 2u);
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(mapped, render_extent - 1u));
}

}
