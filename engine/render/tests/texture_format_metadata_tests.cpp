#include <arc/render/cooked_texture.h>
#include <arc/render/texture_artifact.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace
{

arc::render::texture_data make_compressed_texture(arc::render::texture_format format, std::uint32_t width,
                                                  std::uint32_t height, std::size_t bytes)
{
    using namespace arc::render;
    const auto info = texture_format_metadata(format);
    texture_data texture;
    texture.name = "format-metadata-test";
    texture.width = width;
    texture.height = height;
    texture.format = format;
    texture.color_space = info.srgb ? texture_color_space::srgb : texture_color_space::linear;
    texture.semantic = texture_semantic::generic_color;
    texture.compressed = true;
    texture.encoded.resize(bytes);
    texture.mips = {{.width = width, .height = height, .offset = 0, .size = bytes}};
    texture.mip_levels = 1;
    return texture;
}

} // namespace

TEST_CASE("texture format metadata describes desktop and mobile compression families")
{
    using namespace arc::render;

    const auto bc7 = texture_format_metadata(texture_format::bc7_rgba_srgb);
    CHECK(bc7.family == texture_format_family::bc);
    CHECK(bc7.block_width == 4);
    CHECK(bc7.block_height == 4);
    CHECK(bc7.bytes_per_block == 16);
    CHECK(bc7.channels == 4);
    CHECK(bc7.compressed);
    CHECK(bc7.srgb);
    CHECK_FALSE(bc7.hdr);

    const auto astc = texture_format_metadata(texture_format::astc_6x6_srgb);
    CHECK(astc.family == texture_format_family::astc);
    CHECK(astc.block_width == 6);
    CHECK(astc.block_height == 6);
    CHECK(astc.bytes_per_block == 16);
    CHECK(astc.compressed);
    CHECK(astc.srgb);

    const auto etc2 = texture_format_metadata(texture_format::etc2_rgb8_unorm);
    CHECK(etc2.family == texture_format_family::etc2_eac);
    CHECK(etc2.block_width == 4);
    CHECK(etc2.block_height == 4);
    CHECK(etc2.bytes_per_block == 8);
    CHECK(etc2.channels == 3);
    CHECK_FALSE(etc2.srgb);

    const auto eac = texture_format_metadata(texture_format::eac_rg11_snorm);
    CHECK(eac.family == texture_format_family::etc2_eac);
    CHECK(eac.bytes_per_block == 16);
    CHECK(eac.channels == 2);
    CHECK(eac.signed_normalized);

    CHECK(texture_format_metadata(texture_format::bc6h_rgb_ufloat).hdr);
    CHECK(texture_format_metadata(texture_format::rgba16f).hdr);
    CHECK_FALSE(valid_texture_format(static_cast<texture_format>(255)));
}

TEST_CASE("texture artifacts validate ASTC block dimensions")
{
    using namespace arc::render;

    // 13x7 ASTC 6x6 occupies ceil(13/6) * ceil(7/6) = 3 * 2 blocks.
    const auto texture = make_compressed_texture(texture_format::astc_6x6_unorm, 13, 7, 3u * 2u * 16u);
    const auto encoded = encode_texture_artifact(texture, texture_streaming_mode::resident);
    REQUIRE(encoded.has_value());

    const auto inspected = inspect_texture_artifact(encoded.value());
    REQUIRE(inspected.has_value());
    CHECK(inspected.value().format == texture_format::astc_6x6_unorm);
    REQUIRE(inspected.value().mips.size() == 1);
    CHECK(inspected.value().mips.front().decoded_size == 96);
}

TEST_CASE("texture artifacts validate ETC2 and EAC payload sizes")
{
    using namespace arc::render;

    // 7x5 ETC2 RGB8 occupies four 8-byte blocks.
    const auto etc2 = make_compressed_texture(texture_format::etc2_rgb8_srgb, 7, 5, 32);
    const auto etc2_encoded = encode_texture_artifact(etc2, texture_streaming_mode::resident);
    REQUIRE(etc2_encoded.has_value());
    const auto etc2_index = inspect_texture_artifact(etc2_encoded.value());
    REQUIRE(etc2_index.has_value());
    CHECK(etc2_index.value().format == texture_format::etc2_rgb8_srgb);
    CHECK(etc2_index.value().mips.front().decoded_size == 32);

    // 5x5 EAC RG11 occupies four 16-byte blocks.
    const auto eac = make_compressed_texture(texture_format::eac_rg11_unorm, 5, 5, 64);
    const auto eac_encoded = encode_texture_artifact(eac, texture_streaming_mode::resident);
    REQUIRE(eac_encoded.has_value());
    const auto eac_index = inspect_texture_artifact(eac_encoded.value());
    REQUIRE(eac_index.has_value());
    CHECK(eac_index.value().format == texture_format::eac_rg11_unorm);
    CHECK(eac_index.value().mips.front().decoded_size == 64);
}

TEST_CASE("texture artifacts reject mis-sized compressed mobile payloads")
{
    using namespace arc::render;

    const auto texture = make_compressed_texture(texture_format::astc_5x5_unorm, 10, 10, 63);
    const auto encoded = encode_texture_artifact(texture, texture_streaming_mode::resident);
    REQUIRE_FALSE(encoded.has_value());
    CHECK(encoded.error().code == texture_artifact_error_code::invalid_data);
}

TEST_CASE("virtual textures reject block sizes that do not align to the current tile geometry")
{
    using namespace arc::render;

    const auto texture = make_compressed_texture(texture_format::astc_6x6_unorm, 12, 12, 64);
    const auto encoded = encode_texture_artifact(texture, texture_streaming_mode::virtual_tiles);
    REQUIRE_FALSE(encoded.has_value());
    CHECK(encoded.error().code == texture_artifact_error_code::unsupported_texture);
}


namespace
{
template <class T> void append_legacy(std::vector<std::byte>& bytes, const T& value)
{
    const auto* begin = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(T));
}

std::vector<std::byte> legacy_texture_fixture()
{
    using namespace arc::render;
    std::vector<std::byte> result;
    constexpr std::string_view magic = "ARC_TEXTURE_1";
    append_legacy(result, static_cast<std::uint64_t>(magic.size()));
    for (const char letter : magic) result.push_back(static_cast<std::byte>(letter));
    append_legacy(result, std::uint32_t{2}); // width
    append_legacy(result, std::uint32_t{2}); // height
    append_legacy(result, std::uint32_t{1}); // depth
    append_legacy(result, std::uint32_t{1}); // array layers
    append_legacy(result, std::uint32_t{1}); // mip levels
    append_legacy(result, texture_dimension::texture_2d);
    append_legacy(result, texture_format::rgba8_srgb);
    append_legacy(result, texture_color_space::srgb);
    append_legacy(result, texture_semantic::base_color);
    append_legacy(result, std::uint32_t{1}); // mip table entries
    append_legacy(result, std::uint32_t{2}); // mip width
    append_legacy(result, std::uint32_t{2}); // mip height
    append_legacy(result, std::uint64_t{0}); // mip offset
    append_legacy(result, std::uint64_t{16}); // mip size
    append_legacy(result, std::uint64_t{16}); // payload size
    result.insert(result.end(), 16, std::byte{0x7f});
    return result;
}
} // namespace

TEST_CASE("Android runtime decodes legacy ARC_TEXTURE_1 generated by the current cooker")
{
    const auto bytes = legacy_texture_fixture();
    const auto decoded = arc::render::decode_cooked_texture_data(bytes);
    REQUIRE(decoded);
    CHECK(decoded.value().width == 2);
    CHECK(decoded.value().height == 2);
    CHECK(decoded.value().format == arc::render::texture_format::rgba8_srgb);
    CHECK_FALSE(decoded.value().compressed);
    REQUIRE(decoded.value().mips.size() == 1);
    CHECK(decoded.value().mips.front().size == 16);
    CHECK(decoded.value().pixels.size() == 16);
}

TEST_CASE("Android runtime fails closed on malformed legacy texture mip and payload data")
{
    const auto bytes = legacy_texture_fixture();
    auto truncated = bytes;
    truncated.pop_back();
    CHECK_FALSE(arc::render::decode_cooked_texture_data(truncated));

    auto extra = bytes;
    extra.push_back(std::byte{0});
    CHECK_FALSE(arc::render::decode_cooked_texture_data(extra));

    auto bad_mip = bytes;
    constexpr std::size_t header_and_mip_offset = sizeof(std::uint64_t) + 13 + 5 * sizeof(std::uint32_t) +
                                                   4 * sizeof(std::uint8_t) + sizeof(std::uint32_t) +
                                                   2 * sizeof(std::uint32_t);
    bad_mip[header_and_mip_offset] = std::byte{1}; // mip offset must be zero
    CHECK_FALSE(arc::render::decode_cooked_texture_data(bad_mip));

    auto wrong_format = bytes;
    constexpr std::size_t format_offset = sizeof(std::uint64_t) + 13 + 5 * sizeof(std::uint32_t) +
                                          sizeof(std::uint8_t);
    wrong_format[format_offset] = std::byte{0xff};
    CHECK_FALSE(arc::render::decode_cooked_texture_data(wrong_format));
}

TEST_CASE("Android runtime still decodes indexed texture artifacts")
{
    using namespace arc::render;
    const auto source = make_compressed_texture(texture_format::astc_6x6_unorm, 12, 12, 64);
    const auto indexed = encode_texture_artifact(source, texture_streaming_mode::resident);
    REQUIRE(indexed);
    const auto decoded = decode_cooked_texture_data(indexed.value());
    REQUIRE(decoded);
    CHECK(decoded.value().compressed);
    CHECK(decoded.value().format == texture_format::astc_6x6_unorm);
    CHECK(decoded.value().encoded.size() == 64);
}
