#pragma once

#include <arc/editor/material_asset.h>
#include <arc/render/texture.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace arc::editor
{

struct material_preview_result
{
    render::texture_data texture;
    std::string message;

    bool succeeded() const noexcept { return texture.has_pixels(); }
};

/**
 * @brief Render a deterministic scene-linear PBR material sphere on the CPU.
 *
 * This provides editor previews without creating another presentation surface.
 * It intentionally consumes the same material asset fields as the raster path,
 * and can later be replaced by an offscreen backend implementation without
 * changing the host or asset-picker contracts.
 */
material_preview_result render_material_preview(
    const material_asset& asset,
    const std::filesystem::path& asset_root,
    std::uint32_t size = 128);

} // namespace arc::editor
