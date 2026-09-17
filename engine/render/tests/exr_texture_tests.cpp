#include <arc/render/texture.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

TEST_CASE("TinyEXR decodes HDR texture data as linear RGBA32F", "[render][texture][exr]")
{
    const auto path = std::filesystem::path(__FILE__).parent_path() / "data" / "tiny_hdr.exr";
    const auto loaded = arc::render::load_exr_texture_asset(path);

    REQUIRE(loaded.succeeded());
    CHECK(loaded.texture.width == 2);
    CHECK(loaded.texture.height == 1);
    CHECK(loaded.texture.format == arc::render::texture_format::rgba32f);
    CHECK(loaded.texture.color_space == arc::render::texture_color_space::linear);
    CHECK(loaded.texture.semantic == arc::render::texture_semantic::environment);
    CHECK(loaded.texture.mime_type == "image/x-exr");
    REQUIRE(loaded.texture.mips.size() == 1);
    CHECK(loaded.texture.mips.front().size == 2u * 4u * sizeof(float));
    REQUIRE(loaded.texture.pixels.size() == 2u * 4u * sizeof(float));

    std::array<float, 8> pixels{};
    std::memcpy(pixels.data(), loaded.texture.pixels.data(), loaded.texture.pixels.size());
    CHECK(pixels[3] == 1.0f);
    CHECK(pixels[7] == 1.0f);
    CHECK(std::max({pixels[4], pixels[5], pixels[6]}) > 1.0f);
}

TEST_CASE("TinyEXR reports malformed payloads without producing texture data", "[render][texture][exr]")
{
    std::vector<std::byte> invalid(16, std::byte{0});
    const auto loaded = arc::render::load_exr_texture_asset_bytes(std::move(invalid), "invalid.exr");

    CHECK_FALSE(loaded.succeeded());
    CHECK(loaded.message.find("OpenEXR decoding failed") != std::string::npos);
}
