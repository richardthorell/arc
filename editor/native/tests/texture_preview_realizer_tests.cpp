#include <arc/editor/texture_preview_realizer.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>

TEST_CASE("texture preview mip selection rebases decoded mip payload", "[editor][texture][preview][native]")
{
    arc::render::texture_data texture;
    texture.width = 4u;
    texture.height = 4u;
    texture.depth = 1u;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::rgba8_unorm;
    texture.mip_levels = 2u;
    texture.pixels.resize(80u);
    for (std::size_t index = 0; index < texture.pixels.size(); ++index)
        texture.pixels[index] = static_cast<std::byte>(index);
    texture.mips = {
        {.width = 4u, .height = 4u, .offset = 0u, .size = 64u},
        {.width = 2u, .height = 2u, .offset = 64u, .size = 16u},
    };

    const auto preview = arc::editor::select_texture_preview_mip(std::move(texture), 1u);

    REQUIRE(preview.width == 2u);
    REQUIRE(preview.height == 2u);
    REQUIRE(preview.mip_levels == 1u);
    REQUIRE(preview.mips.size() == 1u);
    CHECK(preview.mips.front().offset == 0u);
    CHECK(preview.mips.front().size == 16u);
    REQUIRE(preview.pixels.size() == 16u);
    CHECK(std::to_integer<std::uint8_t>(preview.pixels.front()) == 64u);
    CHECK(std::to_integer<std::uint8_t>(preview.pixels.back()) == 79u);
}

TEST_CASE("texture preview mip selection preserves encoded compressed payload", "[editor][texture][preview][native]")
{
    arc::render::texture_data texture;
    texture.width = 8u;
    texture.height = 8u;
    texture.depth = 1u;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::bc1_rgba_unorm;
    texture.compressed = true;
    texture.dds = true;
    texture.mip_levels = 2u;
    texture.encoded.resize(40u);
    for (std::size_t index = 0; index < texture.encoded.size(); ++index)
        texture.encoded[index] = static_cast<std::byte>(index + 1u);
    texture.mips = {
        {.width = 8u, .height = 8u, .offset = 0u, .size = 32u},
        {.width = 4u, .height = 4u, .offset = 32u, .size = 8u},
    };

    const auto preview = arc::editor::select_texture_preview_mip(std::move(texture), 1u);

    REQUIRE(preview.width == 4u);
    REQUIRE(preview.height == 4u);
    REQUIRE(preview.mips.size() == 1u);
    CHECK(preview.encoded.size() == 8u);
    CHECK(preview.pixels.empty());
    CHECK(std::to_integer<std::uint8_t>(preview.encoded.front()) == 33u);
    CHECK(std::to_integer<std::uint8_t>(preview.encoded.back()) == 40u);
}

TEST_CASE("native texture preview checker is a valid linear RGBA texture", "[editor][texture][preview][native]")
{
    const auto checker = arc::editor::make_texture_preview_checker();

    REQUIRE(checker.dimension == arc::render::texture_dimension::texture_2d);
    REQUIRE(checker.format == arc::render::texture_format::rgba8_unorm);
    REQUIRE(checker.color_space == arc::render::texture_color_space::linear);
    REQUIRE(checker.width == 64u);
    REQUIRE(checker.height == 64u);
    REQUIRE(checker.mips.size() == 1u);
    REQUIRE(checker.pixels.size() == static_cast<std::size_t>(64u * 64u * 4u));

    const auto first = std::to_integer<std::uint8_t>(checker.pixels[0]);
    const auto adjacent_cell = std::to_integer<std::uint8_t>(checker.pixels[8u * 4u]);
    CHECK(first != adjacent_cell);
    CHECK(std::to_integer<std::uint8_t>(checker.pixels[3]) == 255u);
}

TEST_CASE("native texture preview material exposes target and checker textures", "[editor][texture][preview][native]")
{
    const auto result = arc::editor::realize_texture_preview_material(
        512u, 256u, {.red = true, .green = true, .blue = true, .alpha = true, .exposure = 1.0f, .nearest = true});

    REQUIRE(result.succeeded);
    CHECK(std::ranges::find(result.texture_sources, std::string_view{"__arc_texture_preview_target__"}) !=
          result.texture_sources.end());
    CHECK(std::ranges::find(result.texture_sources, std::string_view{"__arc_texture_preview_checker__"}) !=
          result.texture_sources.end());
    CHECK(result.material.shading_model == arc::render::material_shading_model::unlit);
}
