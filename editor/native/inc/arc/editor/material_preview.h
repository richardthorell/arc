#pragma once

#include <arc/editor/material_asset.h>
#include <arc/render/texture.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace arc::editor
{

struct material_preview_result
{
    render::texture_data texture;
    std::string message;

    bool succeeded() const noexcept
    {
        return texture.has_pixels();
    }
};

struct material_graph_preview_result
{
    render::material_descriptor material;
    std::string message;

    bool succeeded() const noexcept
    {
        return message.empty();
    }
};

/**
 * @brief Resolve the statically evaluable outputs of a validated Material Graph for editor preview.
 *
 * The graph is first compiled by ARC's native Material IR compiler, so authoring validation and
 * connectivity remain native-authoritative. Dynamic inputs such as time, UVs, textures, and
 * shader-backed functions keep their renderer defaults until the preview renderer executes the
 * compiled Material ABI directly.
 */
material_graph_preview_result material_graph_preview_descriptor(std::string_view graph_json);

/**
 * @brief Render a deterministic scene-linear PBR material sphere on the CPU.
 *
 * This provides editor previews without creating another presentation surface.
 * It intentionally consumes the same material asset fields as the raster path,
 * and can later be replaced by an offscreen backend implementation without
 * changing the host or asset-picker contracts.
 */
material_preview_result render_material_preview(const material_asset& asset, const std::filesystem::path& asset_root,
                                                std::uint32_t size = 128);

} // namespace arc::editor
