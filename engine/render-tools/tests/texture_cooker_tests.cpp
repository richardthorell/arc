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
    const std::uint8_t pixels[16]{0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0};
    std::memcpy(bytes.data() + 54, pixels, sizeof(pixels));
    return bytes;
}

} // namespace

TEST_CASE("texture import settings migrate and serialize authored v3 settings")
{
    using namespace arc::render;
    using namespace arc::render::tools;
    const auto legacy = parse_texture_import_settings(R"({"streamingMode":"virtual_tiles"})", 2);
    REQUIRE(legacy.has_value());
    CHECK(legacy.value().streaming_mode == texture_streaming_mode::virtual_tiles);
    CHECK(legacy.value().semantic == texture_semantic::generic_color);
    CHECK(legacy.value().color_space == texture_color_space::srgb);

    auto settings = texture_import_settings_for_preset(texture_import_preset::normal_map);
    CHECK(settings.semantic == texture_semantic::normal);
    CHECK(settings.color_space == texture_color_space::linear);
    const auto serialized = serialize_texture_import_settings(settings);
    const auto parsed = parse_texture_import_settings(serialized, texture_import_settings::current_version);
    REQUIRE(parsed.has_value());
    CHECK(parsed.value().preset == texture_import_preset::normal_map);
    CHECK(parsed.value().semantic == texture_semantic::normal);
    CHECK(parsed.value().color_space == texture_color_space::linear);
    CHECK(parsed.value().streaming_mode == texture_streaming_mode::streamed_mips);
    CHECK_FALSE(parse_texture_import_settings(R"({"semantic":"banana"})", 3).has_value());
}

TEST_CASE("decoded textures cook authored semantic and color space into artifacts")
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
    context.settings_version = 3;
    context.canonical_settings =
        R"({"colorSpace":"linear","preset":"normal_map","semantic":"normal","streamingMode":"virtual_tiles"})";
    const auto cooked = processor.cook(context);
    REQUIRE(cooked.succeeded());
    REQUIRE(cooked.artifacts.size() == 1);
    const auto inspected = render::inspect_texture_artifact(cooked.artifacts[0].bytes);
    REQUIRE(inspected.has_value());
    CHECK(inspected.value().mode == render::texture_streaming_mode::virtual_tiles);
    CHECK(inspected.value().semantic == render::texture_semantic::normal);
    CHECK(inspected.value().color_space == render::texture_color_space::linear);
    CHECK(inspected.value().format == render::texture_format::rgba8_unorm);
}
