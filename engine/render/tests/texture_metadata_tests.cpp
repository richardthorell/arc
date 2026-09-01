#include <arc/render/texture.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>

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
