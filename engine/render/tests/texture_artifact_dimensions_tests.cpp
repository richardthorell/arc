#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace
{

arc::render::texture_data make_dimension_texture(arc::render::texture_dimension dimension, std::uint32_t width,
                                                 std::uint32_t height, std::uint32_t depth = 1,
                                                 std::uint32_t array_layers = 1)
{
    using namespace arc::render;
    texture_data texture;
    texture.name = "dimension-artifact-test";
    texture.width = width;
    texture.height = height;
    texture.depth = depth;
    texture.dimension = dimension;
    texture.format = texture_format::rgba8_unorm;
    texture.color_space = texture_color_space::linear;
    texture.semantic = texture_semantic::generic_color;
    texture.array_layers = array_layers;
    const std::uint32_t face_count = dimension == texture_dimension::cube ? 6u : 1u;

    for (;;)
    {
        const auto offset = texture.pixels.size();
        const auto texel_count = static_cast<std::size_t>(width) * height * depth * array_layers * face_count;
        for (std::size_t texel = 0; texel < texel_count; ++texel)
        {
            texture.pixels.push_back(static_cast<std::byte>(texel & 0xffu));
            texture.pixels.push_back(static_cast<std::byte>((texel >> 1u) & 0xffu));
            texture.pixels.push_back(static_cast<std::byte>((texel >> 2u) & 0xffu));
            texture.pixels.push_back(std::byte{0xff});
        }
        texture.mips.push_back(
            {.width = width, .height = height, .offset = offset, .size = texture.pixels.size() - offset});
        if (width == 1 && height == 1 && (dimension != texture_dimension::texture_3d || depth == 1)) break;
        width = std::max(1u, width / 2u);
        height = std::max(1u, height / 2u);
        if (dimension == texture_dimension::texture_3d) depth = std::max(1u, depth / 2u);
    }
    texture.mip_levels = static_cast<std::uint32_t>(texture.mips.size());
    return texture;
}

} // namespace

TEST_CASE("texture artifacts preserve ordinary 2D topology under schema v4")
{
    using namespace arc::render;
    const auto texture = make_dimension_texture(texture_dimension::texture_2d, 8, 4);
    const auto encoded = encode_texture_artifact(texture, texture_streaming_mode::streamed_mips);
    REQUIRE(encoded.has_value());

    const auto inspected = inspect_texture_artifact(encoded.value());
    REQUIRE(inspected.has_value());
    const auto& index = inspected.value();
    CHECK(index.schema_version == texture_artifact_schema_version);
    CHECK(index.dimension == texture_dimension::texture_2d);
    CHECK(index.depth == 1);
    CHECK(index.array_layers == 1);
    CHECK(index.face_count == 1);
    CHECK(index.mips[0].depth == 1);
}

TEST_CASE("texture artifacts preserve cube topology and atomic face mips")
{
    using namespace arc::render;
    const auto texture = make_dimension_texture(texture_dimension::cube, 8, 8);
    const auto encoded = encode_texture_artifact(texture, texture_streaming_mode::streamed_mips);
    REQUIRE(encoded.has_value());

    const auto inspected = inspect_texture_artifact(encoded.value());
    REQUIRE(inspected.has_value());
    const auto& index = inspected.value();
    CHECK(index.schema_version == texture_artifact_schema_version);
    CHECK(index.dimension == texture_dimension::cube);
    CHECK(index.depth == 1);
    CHECK(index.array_layers == 1);
    CHECK(index.face_count == 6);
    REQUIRE(index.mips.size() == 4);
    CHECK(index.mips[0].width == 8);
    CHECK(index.mips[0].height == 8);
    CHECK(index.mips[0].depth == 1);
    CHECK(index.mips[0].decoded_size == 8u * 8u * 6u * 4u);

    const auto mip = read_texture_artifact_mip(encoded.value(), index, 0);
    REQUIRE(mip.has_value());
    CHECK(mip.value().size() == 8u * 8u * 6u * 4u);
}

TEST_CASE("texture artifacts preserve volume depth across the mip chain")
{
    using namespace arc::render;
    const auto texture = make_dimension_texture(texture_dimension::texture_3d, 8, 4, 4);
    const auto encoded = encode_texture_artifact(texture, texture_streaming_mode::streamed_mips);
    REQUIRE(encoded.has_value());

    const auto inspected = inspect_texture_artifact(encoded.value());
    REQUIRE(inspected.has_value());
    const auto& index = inspected.value();
    CHECK(index.dimension == texture_dimension::texture_3d);
    CHECK(index.depth == 4);
    CHECK(index.array_layers == 1);
    CHECK(index.face_count == 1);
    REQUIRE(index.mips.size() == 4);
    CHECK(index.mips[0].depth == 4);
    CHECK(index.mips[1].depth == 2);
    CHECK(index.mips[2].depth == 1);
    CHECK(index.mips[3].depth == 1);
    CHECK(index.mips[0].decoded_size == 8u * 4u * 4u * 4u);
    CHECK(index.mips[1].decoded_size == 4u * 2u * 2u * 4u);
}

TEST_CASE("texture artifacts preserve logical array layers")
{
    using namespace arc::render;
    const auto texture = make_dimension_texture(texture_dimension::texture_2d, 8, 4, 1, 3);
    const auto encoded = encode_texture_artifact(texture, texture_streaming_mode::streamed_mips);
    REQUIRE(encoded.has_value());

    const auto inspected = inspect_texture_artifact(encoded.value());
    REQUIRE(inspected.has_value());
    const auto& index = inspected.value();
    CHECK(index.dimension == texture_dimension::texture_2d);
    CHECK(index.depth == 1);
    CHECK(index.array_layers == 3);
    CHECK(index.face_count == 1);
    CHECK(index.mips[0].decoded_size == 8u * 4u * 3u * 4u);
}

TEST_CASE("texture artifact dimensional validation rejects unsupported topology")
{
    using namespace arc::render;

    auto nonsquare_cube = make_dimension_texture(texture_dimension::cube, 8, 4);
    const auto cube_result = encode_texture_artifact(nonsquare_cube, texture_streaming_mode::streamed_mips);
    REQUIRE_FALSE(cube_result.has_value());
    CHECK(cube_result.error().code == texture_artifact_error_code::unsupported_texture);

    auto array_volume = make_dimension_texture(texture_dimension::texture_3d, 8, 8, 4, 2);
    const auto volume_result = encode_texture_artifact(array_volume, texture_streaming_mode::streamed_mips);
    REQUIRE_FALSE(volume_result.has_value());
    CHECK(volume_result.error().code == texture_artifact_error_code::unsupported_texture);

    const auto cube = make_dimension_texture(texture_dimension::cube, 8, 8);
    const auto virtual_cube = encode_texture_artifact(cube, texture_streaming_mode::virtual_tiles);
    REQUIRE_FALSE(virtual_cube.has_value());
    CHECK(virtual_cube.error().code == texture_artifact_error_code::unsupported_texture);
}
