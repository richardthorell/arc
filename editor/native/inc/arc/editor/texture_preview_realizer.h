#pragma once

#include <arc/editor/material_preview_realizer.h>
#include <arc/render/texture.h>
#include <arc/scene/environment.h>

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

/** @brief Build the emissive Material ABI program used by the native 2D texture preview. */
[[nodiscard]] material_preview_descriptor_result
realize_texture_preview_material(std::uint32_t width, std::uint32_t height,
                                 const texture_preview_shader_options& options);

/** @brief Exclude physical scene lighting from a pixel-accurate texture preview. */
void disable_texture_preview_scene_lighting(ecs::world& world, ecs::entity sun, ecs::entity environment);

/** @brief Show the source texture with the built-in G-buffer path when Slang cannot compile the preview graph. */
void apply_texture_preview_fallback(render::material_descriptor& material, render::texture_handle texture,
                                    const texture_preview_shader_options& options);

} // namespace arc::editor
