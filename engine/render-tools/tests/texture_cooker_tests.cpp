#include <arc/render_tools/texture_cooker.h>

#include <arc/render/texture.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{

template <class T> void write(std::vector<std::byte>& bytes, std::size_t offset, T value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::vector<std::byte> checker_bmp()
{
    std::vector<std::byte> bytes(70);
    bytes[0] = std::byte{'B'};
    bytes[1] = std::byte{'M'};
    write<std::uint32_t>(bytes, 2, static_cast<std::uint32_t>(bytes.size()));
    write<std::uint32_t>(bytes, 10, 54);
    write<std::uint32_t>(bytes, 14, 40);
    write<std::int32_t>(bytes, 18, 2);
    write<std::int32_t>(bytes, 22, 2);
    write<std::uint16_t>(bytes, 26, 1);
    write<std::uint16_t>(bytes, 28, 24);
    write<std::uint32_t>(bytes, 34, 16);
    const std::uint8_t pixels[16]{0, 0, 0, 255, 255, 255, 0, 0,
                                  255, 255, 255, 0, 0, 0, 0, 0};
    std::memcpy(bytes.data() + 54, pixels, sizeof(pixels));
    return bytes;
}

} // namespace

TEST_CASE("texture import settings migrate to resident and serialize v2 modes")
{
    using namespace arc::render;
    using namespace arc::render::tools;
    const auto legacy = parse_texture_import_settings("{}", 1);
    REQUIRE(legacy.has_value());
    CHECK(legacy.value().streaming_mode == texture_streaming_mode::resident);
    const auto virtual_settings = parse_texture_import_settings(R"({"streamingMode":"virtual_tiles"})", 2);
    REQUIRE(virtual_settings.has_value());
    CHECK(virtual_settings.value().streaming_mode == texture_streaming_mode::virtual_tiles);
    CHECK(serialize_texture_import_settings(virtual_settings.value()) == R"({"streamingMode":"virtual_tiles"})");
    CHECK_FALSE(parse_texture_import_settings(R"({"streamingMode":"automatic"})", 2).has_value());
}

TEST_CASE("decoded sRGB textures downsample in linear space and cook conventional virtual companions")
{
    using namespace arc;
    auto source = checker_bmp();
    const auto loaded = render::load_texture_asset_bytes(source, "checker_albedo.bmp");
    REQUIRE(loaded.succeeded());
    REQUIRE(loaded.texture.mips.size() == 2);
    const auto& mip = loaded.texture.mips[1];
    const auto value = std::to_integer<std::uint8_t>(loaded.texture.pixels[mip.offset]);
    CHECK(value >= 187);
    CHECK(value <= 189);

    render::tools::texture_cook_processor processor;
    assets::asset_cook_context context;
    context.asset.guid = {1, 2};
    context.asset.type = assets::asset_types::texture_2d;
    context.source.source_path = "checker_albedo.bmp";
    context.source.bytes = std::move(source);
    context.settings_version = 2;
    context.canonical_settings = R"({"streamingMode":"virtual_tiles"})";
    const auto cooked = processor.cook(context);
    REQUIRE(cooked.succeeded());
    REQUIRE(cooked.artifacts.size() == 1);
    CHECK(cooked.artifacts[0].extension == ".arctex");
    const auto inspected = render::inspect_texture_artifact(cooked.artifacts[0].bytes);
    REQUIRE(inspected.has_value());
    CHECK(inspected.value().mode == render::texture_streaming_mode::virtual_tiles);
    CHECK(inspected.value().mips.size() == loaded.texture.mips.size());
}
