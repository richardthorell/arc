#pragma once

#include <arc/render/texture.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace arc::editor
{

/** @brief CPU image used by Content Browser texture thumbnails. */
struct texture_preview_image
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::byte> rgba;

    [[nodiscard]] bool valid() const noexcept
    {
        return width > 0u && height > 0u &&
               rgba.size() == static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    }
};

/**
 * @brief Build a display-ready preview from decoded texture data.
 *
 * Ordinary textures remain flat. Cube textures are laid out as a horizontal
 * 4x3 cross using ARC's +X, -X, +Y, -Y, +Z, -Z face order. The decision is
 * based on texture dimension rather than source file format, so future cube
 * importers automatically receive the same Content Browser presentation.
 */
[[nodiscard]] texture_preview_image build_texture_preview_image(const render::texture_data& texture,
                                                                std::uint32_t max_size);

/** @brief Encode a preview image as a top-down 32-bit BMP. */
[[nodiscard]] std::vector<std::byte> encode_texture_preview_bmp(const texture_preview_image& preview);

} // namespace arc::editor
