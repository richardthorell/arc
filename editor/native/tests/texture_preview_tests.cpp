#include <arc/editor/texture_preview.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace
{
std::array<std::uint8_t, 4> pixel_at(const arc::editor::texture_preview_image& image, std::uint32_t x, std::uint32_t y)
{
    const auto offset = (static_cast<std::size_t>(y) * image.width + x) * 4u;
    return {std::to_integer<std::uint8_t>(image.rgba[offset]), std::to_integer<std::uint8_t>(image.rgba[offset + 1u]),
            std::to_integer<std::uint8_t>(image.rgba[offset + 2u]),
            std::to_integer<std::uint8_t>(image.rgba[offset + 3u])};
}
} // namespace

TEST_CASE("cube texture previews use the generic 4x3 cross layout", "[editor][texture][preview]")
{
    arc::render::texture_data texture;
    texture.width = 1u;
    texture.height = 1u;
    texture.dimension = arc::render::texture_dimension::cube;
    texture.format = arc::render::texture_format::rgba8_unorm;
    texture.array_layers = 1u;

    // ARC cube face order: +X, -X, +Y, -Y, +Z, -Z.
    constexpr std::array<std::array<std::uint8_t, 4>, 6> faces{{
        {{255u, 0u, 0u, 255u}},
        {{0u, 255u, 0u, 255u}},
        {{0u, 0u, 255u, 255u}},
        {{255u, 255u, 0u, 255u}},
        {{255u, 0u, 255u, 255u}},
        {{0u, 255u, 255u, 255u}},
    }};
    for (const auto& face : faces)
        for (const auto channel : face)
            texture.pixels.push_back(static_cast<std::byte>(channel));
    texture.mips.push_back({.width = 1u, .height = 1u, .offset = 0u, .size = texture.pixels.size()});

    const auto preview = arc::editor::build_texture_preview_image(texture, 128u);
    REQUIRE(preview.valid());
    CHECK(preview.width == 4u);
    CHECK(preview.height == 3u);

    CHECK(pixel_at(preview, 1u, 0u) == faces[2]); // +Y
    CHECK(pixel_at(preview, 0u, 1u) == faces[1]); // -X
    CHECK(pixel_at(preview, 1u, 1u) == faces[4]); // +Z
    CHECK(pixel_at(preview, 2u, 1u) == faces[0]); // +X
    CHECK(pixel_at(preview, 3u, 1u) == faces[5]); // -Z
    CHECK(pixel_at(preview, 1u, 2u) == faces[3]); // -Y
    CHECK((pixel_at(preview, 0u, 0u) == std::array<std::uint8_t, 4>{0u, 0u, 0u, 0u}));
}

TEST_CASE("ordinary texture previews remain flat", "[editor][texture][preview]")
{
    arc::render::texture_data texture;
    texture.width = 2u;
    texture.height = 1u;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::rgba8_unorm;
    texture.pixels = {std::byte{0xff}, std::byte{0x20}, std::byte{0x10}, std::byte{0xff},
                      std::byte{0x10}, std::byte{0x20}, std::byte{0xff}, std::byte{0xff}};
    texture.mips.push_back({.width = 2u, .height = 1u, .offset = 0u, .size = texture.pixels.size()});

    const auto preview = arc::editor::build_texture_preview_image(texture, 128u);
    REQUIRE(preview.valid());
    CHECK(preview.width == 2u);
    CHECK(preview.height == 1u);
    CHECK((pixel_at(preview, 0u, 0u) == std::array<std::uint8_t, 4>{255u, 32u, 16u, 255u}));
    CHECK((pixel_at(preview, 1u, 0u) == std::array<std::uint8_t, 4>{16u, 32u, 255u, 255u}));
}

TEST_CASE("native texel inspection uses selected mip source data", "[editor][texture][inspection]")
{
    arc::render::texture_data texture;
    texture.width = 2u;
    texture.height = 2u;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::rgba8_unorm;
    texture.pixels.resize(20u, std::byte{0});
    texture.pixels[16] = std::byte{0x80};
    texture.pixels[17] = std::byte{0x40};
    texture.pixels[18] = std::byte{0x20};
    texture.pixels[19] = std::byte{0xff};
    texture.mips.push_back({.width = 2u, .height = 2u, .offset = 0u, .size = 16u});
    texture.mips.push_back({.width = 1u, .height = 1u, .offset = 16u, .size = 4u});

    const auto sample = arc::editor::inspect_texture_texel(texture, 0u, 0u, 1u);
    REQUIRE(sample.valid);
    CHECK(sample.mip == 1u);
    CHECK(sample.rgba[0] == 128.0f / 255.0f);
    CHECK(sample.rgba[1] == 64.0f / 255.0f);
    CHECK(sample.rgba[2] == 32.0f / 255.0f);
    CHECK(sample.rgba[3] == 1.0f);
    CHECK_FALSE(arc::editor::inspect_texture_texel(texture, 1u, 0u, 1u).valid);
}

TEST_CASE("native texel inspection preserves unclamped HDR values", "[editor][texture][inspection]")
{
    arc::render::texture_data texture;
    texture.width = 1u;
    texture.height = 1u;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::rgba32f;
    const std::array<float, 4> source{2.5f, -0.25f, 0.125f, 1.5f};
    texture.pixels.resize(sizeof(source));
    std::memcpy(texture.pixels.data(), source.data(), sizeof(source));

    const auto sample = arc::editor::inspect_texture_texel(texture, 0u, 0u);
    REQUIRE(sample.valid);
    CHECK(sample.rgba == source);
}
