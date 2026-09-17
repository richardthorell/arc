#include <arc/editor/texture_preview.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace arc::editor
{
namespace
{
constexpr std::array<std::int8_t, 12> cube_cross_faces{
    -1, 2, -1, -1,
    1, 4, 0, 5,
    -1, 3, -1, -1,
};

std::uint8_t preview_channel(float linear_value)
{
    const float clamped = std::max(0.0f, linear_value);
    const float mapped = clamped / (1.0f + clamped);
    const float srgb = mapped <= 0.0031308f ? mapped * 12.92f : 1.055f * std::pow(mapped, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint8_t>(std::clamp(std::lround(srgb * 255.0f), 0l, 255l));
}

std::uint8_t preview_alpha(float value)
{
    return static_cast<std::uint8_t>(std::clamp(std::lround(value * 255.0f), 0l, 255l));
}

void write_u16(std::vector<std::byte>& bytes, std::size_t offset, std::uint16_t value)
{
    bytes[offset] = static_cast<std::byte>(value & 0xffu);
    bytes[offset + 1u] = static_cast<std::byte>((value >> 8u) & 0xffu);
}

void write_u32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value)
{
    for (std::size_t index = 0; index < 4u; ++index)
        bytes[offset + index] = static_cast<std::byte>((value >> (index * 8u)) & 0xffu);
}

std::array<std::uint8_t, 4> source_pixel(const render::texture_data& texture, std::size_t source_offset,
                                         bool float_pixels)
{
    if (float_pixels)
    {
        std::array<float, 4> linear{};
        std::memcpy(linear.data(), texture.pixels.data() + source_offset, sizeof(linear));
        return {preview_channel(linear[0]), preview_channel(linear[1]), preview_channel(linear[2]),
                preview_alpha(linear[3])};
    }
    return {std::to_integer<std::uint8_t>(texture.pixels[source_offset]),
            std::to_integer<std::uint8_t>(texture.pixels[source_offset + 1u]),
            std::to_integer<std::uint8_t>(texture.pixels[source_offset + 2u]),
            std::to_integer<std::uint8_t>(texture.pixels[source_offset + 3u])};
}
} // namespace

texture_preview_image build_texture_preview_image(const render::texture_data& texture, std::uint32_t max_size)
{
    texture_preview_image preview;
    if (!texture.has_pixels() || texture.width == 0u || texture.height == 0u || max_size == 0u) return preview;

    const bool float_pixels = texture.format == render::texture_format::rgba32f;
    const bool byte_pixels =
        texture.format == render::texture_format::rgba8_unorm || texture.format == render::texture_format::rgba8_srgb;
    if (!float_pixels && !byte_pixels) return preview;

    const bool cube = texture.dimension == render::texture_dimension::cube;
    constexpr std::uint32_t cube_face_count = 6u;
    const std::uint32_t face_count = cube ? cube_face_count : 1u;
    const std::size_t bytes_per_pixel = float_pixels ? sizeof(float) * 4u : 4u;
    const std::uint64_t face_pixel_count = static_cast<std::uint64_t>(texture.width) * texture.height;
    const std::uint64_t required_bytes = face_pixel_count * face_count * bytes_per_pixel;
    if (required_bytes > std::numeric_limits<std::size_t>::max()) return preview;

    const std::size_t base_offset = texture.mips.empty() ? 0u : texture.mips.front().offset;
    if (base_offset > texture.pixels.size() || required_bytes > texture.pixels.size() - base_offset) return preview;
    if (!texture.mips.empty() && texture.mips.front().size < required_bytes) return preview;

    const std::uint32_t columns = cube ? 4u : 1u;
    const std::uint32_t rows = cube ? 3u : 1u;
    const std::uint64_t layout_width = static_cast<std::uint64_t>(texture.width) * columns;
    const std::uint64_t layout_height = static_cast<std::uint64_t>(texture.height) * rows;
    const auto largest = static_cast<double>(std::max(layout_width, layout_height));
    const double scale = std::min(1.0, static_cast<double>(max_size) / largest);
    const auto scaled_face_width =
        std::max(1u, static_cast<std::uint32_t>(std::floor(static_cast<double>(texture.width) * scale)));
    const auto scaled_face_height =
        std::max(1u, static_cast<std::uint32_t>(std::floor(static_cast<double>(texture.height) * scale)));
    preview.width = scaled_face_width * columns;
    preview.height = scaled_face_height * rows;
    preview.rgba.assign(static_cast<std::size_t>(preview.width) * preview.height * 4u, std::byte{0});

    for (std::uint32_t y = 0; y < preview.height; ++y)
    {
        const std::uint32_t cell_y = y / scaled_face_height;
        const std::uint32_t local_y = y % scaled_face_height;
        for (std::uint32_t x = 0; x < preview.width; ++x)
        {
            const std::uint32_t cell_x = x / scaled_face_width;
            const std::uint32_t local_x = x % scaled_face_width;
            const std::int8_t face = cube ? cube_cross_faces[cell_y * columns + cell_x] : 0;
            if (face < 0) continue;

            const auto source_x = std::min(texture.width - 1u, local_x * texture.width / scaled_face_width);
            const auto source_y = std::min(texture.height - 1u, local_y * texture.height / scaled_face_height);
            const std::uint64_t source_pixel_index = static_cast<std::uint64_t>(face) * face_pixel_count +
                                                     static_cast<std::uint64_t>(source_y) * texture.width + source_x;
            const std::size_t source_offset =
                base_offset + static_cast<std::size_t>(source_pixel_index * bytes_per_pixel);
            const auto rgba = source_pixel(texture, source_offset, float_pixels);
            const std::size_t target = (static_cast<std::size_t>(y) * preview.width + x) * 4u;
            for (std::size_t channel = 0; channel < rgba.size(); ++channel)
                preview.rgba[target + channel] = static_cast<std::byte>(rgba[channel]);
        }
    }
    return preview;
}

std::vector<std::byte> encode_texture_preview_bmp(const texture_preview_image& preview)
{
    if (!preview.valid()) return {};
    constexpr std::size_t header_size = 54u;
    const std::uint64_t pixel_bytes = static_cast<std::uint64_t>(preview.width) * preview.height * 4u;
    if (pixel_bytes > std::numeric_limits<std::uint32_t>::max() - header_size) return {};

    std::vector<std::byte> bmp(header_size + static_cast<std::size_t>(pixel_bytes));
    bmp[0] = std::byte{'B'};
    bmp[1] = std::byte{'M'};
    write_u32(bmp, 2u, static_cast<std::uint32_t>(bmp.size()));
    write_u32(bmp, 10u, static_cast<std::uint32_t>(header_size));
    write_u32(bmp, 14u, 40u);
    write_u32(bmp, 18u, preview.width);
    write_u32(bmp, 22u, static_cast<std::uint32_t>(-static_cast<std::int32_t>(preview.height)));
    write_u16(bmp, 26u, 1u);
    write_u16(bmp, 28u, 32u);
    write_u32(bmp, 34u, static_cast<std::uint32_t>(pixel_bytes));

    for (std::uint32_t y = 0; y < preview.height; ++y)
    {
        for (std::uint32_t x = 0; x < preview.width; ++x)
        {
            const std::size_t source = (static_cast<std::size_t>(y) * preview.width + x) * 4u;
            const std::size_t target = header_size + source;
            bmp[target] = preview.rgba[source + 2u];
            bmp[target + 1u] = preview.rgba[source + 1u];
            bmp[target + 2u] = preview.rgba[source];
            bmp[target + 3u] = preview.rgba[source + 3u];
        }
    }
    return bmp;
}

} // namespace arc::editor
