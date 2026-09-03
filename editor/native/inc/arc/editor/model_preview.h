#pragma once

#include <arc/render/material.h>
#include <arc/render/mesh.h>
#include <arc/render/texture.h>

#include <cstdint>
#include <optional>
#include <string>

namespace arc::editor
{

struct model_preview_options
{
    std::uint32_t size{128};

    // V1 uses the editor's neutral/default Phong-style preview material. Keeping
    // this override in the renderer contract lets callers supply a material
    // referenced by the model later without changing the thumbnail API.
    std::optional<render::material_descriptor> material_override;
};

struct model_preview_result
{
    render::texture_data texture;
    std::string message;

    bool succeeded() const noexcept
    {
        return texture.has_pixels();
    }
};

/**
 * @brief Render a deterministic CPU model thumbnail with automatic bounds framing.
 */
model_preview_result render_model_preview(const render::scene_import_result& scene,
                                          const model_preview_options& options = {});

} // namespace arc::editor
