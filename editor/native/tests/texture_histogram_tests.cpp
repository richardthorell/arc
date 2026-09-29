#include <arc/editor/texture_histogram.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstring>

TEST_CASE("native histogram bins exact source values", "[editor][texture][histogram]")
{
    arc::render::texture_data texture;
    texture.width = 2u;
    texture.height = 1u;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::rgba8_unorm;
    texture.pixels = {std::byte{0x00}, std::byte{0x40}, std::byte{0x80}, std::byte{0xff},
                      std::byte{0xff}, std::byte{0x40}, std::byte{0x00}, std::byte{0xff}};

    const auto histogram = arc::editor::build_texture_histogram(texture);
    REQUIRE(histogram.valid);
    CHECK(histogram.sample_count == 2u);
    CHECK(histogram.minimum[0] == 0.0f);
    CHECK(histogram.maximum[0] == 1.0f);
    CHECK(histogram.bins[0][0] == 1u);
    CHECK(histogram.bins[0][255] == 1u);
    CHECK(histogram.bins[1][0] == 2u);
    CHECK(histogram.bins[2][0] == 1u);
    CHECK(histogram.bins[2][255] == 1u);
}

TEST_CASE("native histogram preserves HDR range and selected mip", "[editor][texture][histogram]")
{
    arc::render::texture_data texture;
    texture.width = 2u;
    texture.height = 1u;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::rgba32f;

    const std::array<float, 8> source{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const std::array<float, 4> mip{4.0f, -2.0f, 0.5f, 2.0f};
    texture.pixels.resize(sizeof(source) + sizeof(mip));
    std::memcpy(texture.pixels.data(), source.data(), sizeof(source));
    std::memcpy(texture.pixels.data() + sizeof(source), mip.data(), sizeof(mip));
    texture.mips.push_back({.width = 2u, .height = 1u, .offset = 0u, .size = sizeof(source)});
    texture.mips.push_back({.width = 1u, .height = 1u, .offset = sizeof(source), .size = sizeof(mip)});

    const auto histogram = arc::editor::build_texture_histogram(texture, 1u);
    REQUIRE(histogram.valid);
    CHECK(histogram.mip == 1u);
    CHECK(histogram.sample_count == 1u);
    CHECK(histogram.minimum == mip);
    CHECK(histogram.maximum == mip);
    for (std::size_t channel = 0u; channel < 4u; ++channel)
        CHECK(histogram.bins[channel][0] == 1u);
}

TEST_CASE("native histogram rejects unsupported or missing mip data", "[editor][texture][histogram]")
{
    arc::render::texture_data texture;
    texture.width = 1u;
    texture.height = 1u;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::rgba8_unorm;
    texture.pixels = {std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0xff}};

    CHECK_FALSE(arc::editor::build_texture_histogram(texture, 1u).valid);
    texture.dimension = arc::render::texture_dimension::cube;
    CHECK_FALSE(arc::editor::build_texture_histogram(texture).valid);
}
