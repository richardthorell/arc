#include <arc/render_tools/texture_cooker.h>

#include <arc/render/texture.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
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

TEST_CASE("texture import settings migrate and serialize authored v4 policy")
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

TEST_CASE("texture presets resolve deterministic group sampling policy")
{
    using namespace arc::render;
    using namespace arc::render::tools;
    const auto normal = texture_import_settings_for_preset(texture_import_preset::normal_map);
    CHECK(normal.semantic == texture_semantic::normal);
    CHECK(normal.color_space == texture_color_space::linear);
    CHECK(normal.compression == texture_compression_policy::normal);
    CHECK(normal.generate_mips);
    CHECK(normal.anisotropy == 8.0f);

    const auto ui = texture_import_settings_for_preset(texture_import_preset::ui);
    CHECK_FALSE(ui.generate_mips);
    CHECK(ui.wrap_u == texture_address_mode::clamp_to_edge);
    CHECK(ui.wrap_v == texture_address_mode::clamp_to_edge);
    CHECK(ui.streaming_mode == texture_streaming_mode::resident);
}

TEST_CASE("normal map preprocessing rebuilds and renormalizes authored mip chain")
{
    using namespace arc;
    auto loaded = render::load_texture_asset_bytes(checker_bmp(), "checker.bmp");
    REQUIRE(loaded.succeeded());
    auto settings = render::tools::texture_import_settings_for_preset(render::tools::texture_import_preset::normal_map);
    settings.max_size = 2;
    const auto processed = render::tools::preprocess_texture_for_cook(std::move(loaded.texture), settings,
                                                                      assets::windows_vulkan_cook_target());
    REQUIRE(processed.has_value());
    REQUIRE(processed.value().texture.mips.size() == 2);
    CHECK(processed.value().metadata.normal_mips_renormalized);
    const auto& mip = processed.value().texture.mips.back();
    const auto offset = mip.offset;
    const auto x =
        static_cast<float>(std::to_integer<std::uint8_t>(processed.value().texture.pixels[offset])) / 255.0f * 2.0f -
        1.0f;
    const auto y = static_cast<float>(std::to_integer<std::uint8_t>(processed.value().texture.pixels[offset + 1u])) /
                       255.0f * 2.0f -
                   1.0f;
    const auto z = static_cast<float>(std::to_integer<std::uint8_t>(processed.value().texture.pixels[offset + 2u])) /
                       255.0f * 2.0f -
                   1.0f;
    CHECK(std::sqrt(x * x + y * y + z * z) == Catch::Approx(1.0f).margin(0.02f));
}

TEST_CASE("texture preprocessing applies max size and records artifact policy")
{
    using namespace arc;
    auto loaded = render::load_texture_asset_bytes(checker_bmp(), "checker.bmp");
    REQUIRE(loaded.succeeded());
    auto settings = render::tools::texture_import_settings_for_preset(render::tools::texture_import_preset::color);
    settings.max_size = 1;
    settings.power_of_two = render::texture_power_of_two_policy::resize_down;
    settings.preserve_alpha_coverage = true;
    const auto processed = render::tools::preprocess_texture_for_cook(std::move(loaded.texture), settings,
                                                                      assets::windows_vulkan_cook_target());
    REQUIRE(processed.has_value());
    CHECK(processed.value().texture.width == 1);
    CHECK(processed.value().texture.height == 1);
    CHECK(processed.value().metadata.source_width == 2);
    CHECK(processed.value().metadata.resized);
    const auto artifact =
        render::encode_texture_artifact(processed.value().texture, settings.streaming_mode, processed.value().metadata);
    REQUIRE(artifact.has_value());
    const auto inspected = render::inspect_texture_artifact(artifact.value());
    REQUIRE(inspected.has_value());
    CHECK(inspected.value().metadata.requested_max_size == 1);
    CHECK(inspected.value().metadata.compression == render::texture_compression_policy::color);
    CHECK(inspected.value().metadata.anisotropy == 8.0f);
}
