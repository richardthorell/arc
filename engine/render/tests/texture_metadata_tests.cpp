#include <arc/render/texture.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{
constexpr std::uint32_t fourcc(char a, char b, char c, char d) noexcept
{
    return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8u) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16u) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24u);
}

void write_u32(std::array<std::byte, 128>& bytes, std::size_t offset, std::uint32_t value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void write_u16(std::vector<std::byte>& bytes, std::size_t offset, std::uint16_t value)
{
    bytes[offset] = static_cast<std::byte>(value & 0xffu);
    bytes[offset + 1u] = static_cast<std::byte>((value >> 8u) & 0xffu);
}

void write_u32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value)
{
    bytes[offset] = static_cast<std::byte>(value & 0xffu);
    bytes[offset + 1u] = static_cast<std::byte>((value >> 8u) & 0xffu);
    bytes[offset + 2u] = static_cast<std::byte>((value >> 16u) & 0xffu);
    bytes[offset + 3u] = static_cast<std::byte>((value >> 24u) & 0xffu);
}

std::vector<std::byte> make_tiff_fixture()
{
    constexpr std::size_t ifd_offset = 8;
    constexpr std::size_t entry_count = 10;
    constexpr std::size_t bits_per_sample_offset = 134;
    constexpr std::size_t pixel_offset = 140;

    std::vector<std::byte> bytes(143);
    bytes[0] = std::byte{0x49};
    bytes[1] = std::byte{0x49};
    write_u16(bytes, 2, 42);
    write_u32(bytes, 4, static_cast<std::uint32_t>(ifd_offset));
    write_u16(bytes, ifd_offset, static_cast<std::uint16_t>(entry_count));

    std::size_t entry = ifd_offset + 2u;
    const auto write_short_entry = [&](std::uint16_t tag, std::uint16_t value)
    {
        write_u16(bytes, entry, tag);
        write_u16(bytes, entry + 2u, 3);
        write_u32(bytes, entry + 4u, 1);
        write_u16(bytes, entry + 8u, value);
        entry += 12u;
    };
    const auto write_long_entry = [&](std::uint16_t tag, std::uint32_t value)
    {
        write_u16(bytes, entry, tag);
        write_u16(bytes, entry + 2u, 4);
        write_u32(bytes, entry + 4u, 1);
        write_u32(bytes, entry + 8u, value);
        entry += 12u;
    };

    write_short_entry(256, 1);
    write_short_entry(257, 1);
    write_u16(bytes, entry, 258);
    write_u16(bytes, entry + 2u, 3);
    write_u32(bytes, entry + 4u, 3);
    write_u32(bytes, entry + 8u, static_cast<std::uint32_t>(bits_per_sample_offset));
    entry += 12u;
    write_short_entry(259, 1);
    write_short_entry(262, 2);
    write_long_entry(273, static_cast<std::uint32_t>(pixel_offset));
    write_short_entry(277, 3);
    write_long_entry(278, 1);
    write_long_entry(279, 3);
    write_short_entry(284, 1);
    write_u32(bytes, entry, 0);

    write_u16(bytes, bits_per_sample_offset, 8);
    write_u16(bytes, bits_per_sample_offset + 2u, 8);
    write_u16(bytes, bits_per_sample_offset + 4u, 8);
    bytes[pixel_offset] = std::byte{0x12};
    bytes[pixel_offset + 1u] = std::byte{0x34};
    bytes[pixel_offset + 2u] = std::byte{0x56};
    return bytes;
}
} // namespace

TEST_CASE("texture metadata inspection does not require DDS payload", "[render][texture]")
{
    std::array<std::byte, 128> bytes{};
    write_u32(bytes, 0, fourcc('D', 'D', 'S', ' '));
    write_u32(bytes, 4, 124);
    write_u32(bytes, 12, 8);
    write_u32(bytes, 16, 16);
    write_u32(bytes, 28, 5);
    write_u32(bytes, 76, 32);
    write_u32(bytes, 80, 0x00000004u);
    write_u32(bytes, 84, fourcc('D', 'X', 'T', '1'));

    const auto path = std::filesystem::temp_directory_path() / "arc_texture_metadata_albedo.dds";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        REQUIRE(output.good());
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    const auto info = arc::render::inspect_texture_asset(path);
    CHECK(info.succeeded());
    CHECK(info.width == 16);
    CHECK(info.height == 8);
    CHECK(info.mip_count == 5);
    CHECK(info.format == arc::render::texture_format::bc1_rgba_srgb);

    const auto full_load = arc::render::load_texture_asset(path);
    CHECK_FALSE(full_load.succeeded());

    std::error_code error;
    std::filesystem::remove(path, error);
}

TEST_CASE("PSD textures are supported and decoded through stb", "[render][texture]")
{
    const std::array<std::uint8_t, 43> psd = {0x38, 0x42, 0x50, 0x53, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                                              0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                                              0x00, 0x08, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                              0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x56};

    std::vector<std::byte> bytes;
    bytes.reserve(psd.size());
    for (const auto value : psd)
        bytes.push_back(static_cast<std::byte>(value));

    CHECK(arc::render::is_supported_texture_asset("source.PSD"));
    const auto loaded = arc::render::load_texture_asset_bytes(bytes, "source.psd");
    REQUIRE(loaded.succeeded());
    CHECK(loaded.texture.width == 1);
    CHECK(loaded.texture.height == 1);
    CHECK(loaded.texture.format == arc::render::texture_format::rgba8_srgb);
    CHECK(loaded.texture.mime_type == "image/vnd.adobe.photoshop");
    CHECK(loaded.texture.mip_levels == 1);
}

TEST_CASE("TIFF textures are supported and decoded through libtiff", "[render][texture]")
{
    const auto bytes = make_tiff_fixture();

    CHECK(arc::render::is_supported_texture_asset("source.TIF"));
    CHECK(arc::render::is_supported_texture_asset("source.TIFF"));
    const auto loaded = arc::render::load_texture_asset_bytes(bytes, "source.tiff");
    REQUIRE(loaded.succeeded());
    CHECK(loaded.texture.width == 1);
    CHECK(loaded.texture.height == 1);
    CHECK(loaded.texture.format == arc::render::texture_format::rgba8_srgb);
    CHECK(loaded.texture.mime_type == "image/tiff");
    CHECK(loaded.texture.mip_levels == 1);
    REQUIRE(loaded.texture.pixels.size() == 4);
    CHECK(std::to_integer<std::uint8_t>(loaded.texture.pixels[0]) == 0x12);
    CHECK(std::to_integer<std::uint8_t>(loaded.texture.pixels[1]) == 0x34);
    CHECK(std::to_integer<std::uint8_t>(loaded.texture.pixels[2]) == 0x56);
    CHECK(std::to_integer<std::uint8_t>(loaded.texture.pixels[3]) == 0xff);
}
