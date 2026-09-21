#pragma once

#include <arc/render/material.h>

#include <memory>
#include <string>

namespace arc::editor
{

struct texture_preview_display_options
{
    math::vector4f channels = math::vector4f::one;
    float exposure_ev{};
    bool checkerboard{true};
    bool nearest{};
};

struct texture_preview_material_result
{
    render::material_descriptor material;
    bool succeeded{};
    std::string message;
};

/** Build the editor-only Material ABI program used to inspect a Texture2D on the GPU. */
[[nodiscard]] texture_preview_material_result
realize_texture_preview_material(render::texture_handle texture, const texture_preview_display_options& options);

/** Apply display-only controls without recompiling the preview shader. */
void apply_texture_preview_options(render::material_descriptor& material,
                                   const texture_preview_display_options& options);

} // namespace arc::editor
