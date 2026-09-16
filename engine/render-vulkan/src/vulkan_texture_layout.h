#pragma once

#include <arc/render/material.h>
#include <arc/render/texture_artifact.h>

#include <cstdint>

namespace arc::render::vulkan::backend_detail
{

/** @brief Vulkan-supported topology for one sampled texture resource. */
[[nodiscard]] constexpr bool vulkan_texture_topology_supported(texture_dimension dimension, std::uint32_t width,
                                                               std::uint32_t height, std::uint32_t depth,
                                                               std::uint32_t array_layers) noexcept
{
    if (width == 0 || height == 0 || depth == 0 || array_layers == 0) return false;
    switch (dimension)
    {
        case texture_dimension::texture_2d:
            return depth == 1;
        case texture_dimension::texture_3d:
            return array_layers == 1;
        case texture_dimension::cube:
            // T2 supports one cube per texture. Cube arrays need a CUBE_ARRAY
            // view and are intentionally left for the array-texture follow-up.
            return width == height && depth == 1 && array_layers == 1;
    }
    return false;
}

[[nodiscard]] constexpr bool vulkan_texture_topology_supported(const texture_data& data) noexcept
{
    return vulkan_texture_topology_supported(data.dimension, data.width, data.height, data.depth, data.array_layers);
}

[[nodiscard]] constexpr bool vulkan_texture_topology_supported(const texture_artifact_index& artifact) noexcept
{
    return vulkan_texture_topology_supported(artifact.dimension, artifact.width, artifact.height, artifact.depth,
                                             artifact.array_layers);
}

struct streamed_texture_window_layout
{
    texture_dimension dimension{texture_dimension::texture_2d};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t depth{1};
    std::uint32_t array_layers{1};
    std::uint32_t mip_levels{};
};

/**
 * @brief Resolve the image topology for a contiguous streamed mip window.
 *
 * The artifact is authoritative here. This prevents streamed cube/volume
 * textures from being reconstructed as ordinary 2D images when the resident
 * mip window moves.
 */
[[nodiscard]] inline bool resolve_streamed_texture_window_layout(const texture_artifact_index& artifact,
                                                                 std::uint32_t base_mip,
                                                                 streamed_texture_window_layout& out) noexcept
{
    if (!vulkan_texture_topology_supported(artifact) || base_mip >= artifact.mips.size()) return false;
    const auto& base = artifact.mips[base_mip];
    if (base.width == 0 || base.height == 0 || base.depth == 0) return false;

    out.dimension = artifact.dimension;
    out.width = base.width;
    out.height = base.height;
    out.depth = base.depth;
    out.array_layers = artifact.array_layers;
    out.mip_levels = static_cast<std::uint32_t>(artifact.mips.size()) - base_mip;
    return true;
}

} // namespace arc::render::vulkan::backend_detail
