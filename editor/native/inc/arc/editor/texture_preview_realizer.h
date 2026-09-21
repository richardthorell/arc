#pragma once

#include <arc/editor/material_preview_realizer.h>
#include <arc/render/texture.h>

#include <cstdint>

namespace arc::editor
{

struct texture_preview_shader_options
{
    bool red{true};
    bool green{true};
    bool blue{true};
    bool alpha{true};
    float exposure{};
    bool nearest{};
};

/**
 * @brief Rebase one source mip as mip zero without decoding its payload.
 *
 * Encoded DDS/BC payloads remain encoded so the renderer uploads the exact
 * compressed blocks that runtime sampling sees.
 */
[[nodiscard]] render::texture_data select_texture_preview_mip(render::texture_data texture, std::uint32_t mip);

/** @brief Small editor-owned checker texture used for alpha compositing. */
[[nodiscard]] render::texture_data make_texture_preview_checker();

/** @brief Build the unlit Material ABI program used by the native 2D texture preview. */
[[nodiscard]] material_preview_descriptor_result
realize_texture_preview_material(std::uint32_t width, std::uint32_t height,
                                 const texture_preview_shader_options& options);

} // namespace arc::editor
