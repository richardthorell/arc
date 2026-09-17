#include <arc/render_tools/texture_cooker.h>

#include <arc/render/texture.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace
{

template <class T> void write(std::vector<std::byte>& bytes, std::size_t offset, T value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

constexpr std::uint32_t fourcc(char a, char b, char c, char d) noexcept
{
    return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8u) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16u) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24u);
}

std::vector<std::byte> authored_bc1_dds()
{
    constexpr std::array<std::size_t, 4> mip_sizes{32u, 8u, 8u, 8u};
    std::vector<std::byte> bytes(128u + 56u);
    write<std::uint32_t>(bytes, 0, fourcc('D', 'D', 'S', ' '));
    write<std::uint32_t>(bytes, 4, 124u);
    write<std::uint32_t>(bytes, 12, 8u);
    write<std::uint32_t>(bytes, 16, 8u);
    write<std::uint32_t>(bytes, 28, 4u);
    write<std::uint32_t>(bytes, 76, 32u);
    write<std::uint32_t>(bytes, 80, 0x00000004u);
    write<std::uint32_t>(bytes, 84, fourcc('D', 'X', 'T', '1'));

    std::size_t offset = 128u;
    for (std::size_t mip = 0; mip < mip_sizes.size(); ++mip)
    {
        for (std::size_t index = 0; index < mip_sizes[mip]; ++index)
            bytes[offset + index] = static_cast<std::byte>(0x10u * (mip + 1u) + index);
        offset += mip_sizes[mip];
    }
    return bytes;
}

std::span<const std::byte> source_mip(std::span<const std::byte> bytes, std::uint32_t mip)
{
    constexpr std::array<std::size_t, 4> offsets{128u, 160u, 168u, 176u};
    constexpr std::array<std::size_t, 4> sizes{32u, 8u, 8u, 8u};
    return bytes.subspan(offsets[mip], sizes[mip]);
}

} // namespace

TEST_CASE("texture mip policy migrates legacy generateMips settings")
{
    using namespace arc::render;
    using namespace arc::render::tools;

    const auto legacy_disabled = parse_texture_import_settings(R"({"generateMips":false})", 7);
    REQUIRE(legacy_disabled.has_value());
    CHECK(legacy_disabled.value().mip_policy == texture_mip_policy::none);
    CHECK_FALSE(legacy_disabled.value().generate_mips);

    const auto legacy_enabled = parse_texture_import_settings(R"({"generateMips":true})", 7);
    REQUIRE(legacy_enabled.has_value());
    CHECK(legacy_enabled.value().mip_policy == texture_mip_policy::preserve_source);
    CHECK(legacy_enabled.value().generate_mips);

    const auto generated = parse_texture_import_settings(R"({"mipPolicy":"generate"})", 8);
    REQUIRE(generated.has_value());
    CHECK(generated.value().mip_policy == texture_mip_policy::generate);
    CHECK(generated.value().generate_mips);

    CHECK_FALSE(parse_texture_import_settings(R"({"mipPolicy":"invalid"})", 8).has_value());
}

TEST_CASE("texture mip policy round trips through current import settings")
{
    using namespace arc::render;
    using namespace arc::render::tools;

    auto settings = texture_import_settings_for_preset(texture_import_preset::color);
    settings.mip_policy = texture_mip_policy::generate;
    const auto parsed =
        parse_texture_import_settings(serialize_texture_import_settings(settings), texture_import_settings::current_version);
    REQUIRE(parsed.has_value());
    CHECK(parsed.value().mip_policy == texture_mip_policy::generate);

    const auto ui = texture_import_settings_for_preset(texture_import_preset::ui);
    CHECK(ui.mip_policy == texture_mip_policy::none);
    CHECK_FALSE(ui.generate_mips);
}

TEST_CASE("authored DDS mip chains are preserved byte for byte")
{
    using namespace arc;
    using namespace arc::render;
    using namespace arc::render::tools;

    const auto source = authored_bc1_dds();
    auto loaded = load_texture_asset_bytes(source, "authored_albedo.dds");
    REQUIRE(loaded.succeeded());
    REQUIRE(loaded.texture.mips.size() == 4u);

    auto settings = texture_import_settings_for_preset(texture_import_preset::color);
    const auto processed =
        preprocess_texture_for_cook(std::move(loaded.texture), settings, assets::windows_vulkan_cook_target());
    REQUIRE(processed.has_value());
    CHECK(processed.value().metadata.source_mip_count == 4u);
    CHECK(processed.value().metadata.mip_policy == texture_mip_policy::preserve_source);
    CHECK(processed.value().metadata.source_mips_preserved);
    CHECK_FALSE(processed.value().metadata.generated_mips);

    const auto artifact =
        encode_texture_artifact(processed.value().texture, settings.streaming_mode, processed.value().metadata);
    REQUIRE(artifact.has_value());
    const auto inspected = inspect_texture_artifact(artifact.value());
    REQUIRE(inspected.has_value());
    CHECK(inspected.value().schema_version == texture_artifact_schema_version);
    CHECK(inspected.value().format == texture_format::bc1_rgba_srgb);
    CHECK(inspected.value().mip_count == 4u);
    CHECK(inspected.value().metadata.source_mip_count == 4u);
    CHECK(inspected.value().metadata.mip_policy == texture_mip_policy::preserve_source);
    CHECK(inspected.value().metadata.source_mips_preserved);
    CHECK_FALSE(inspected.value().metadata.generated_mips);

    for (std::uint32_t mip = 0; mip < 4u; ++mip)
    {
        const auto payload = read_texture_artifact_mip(artifact.value(), inspected.value(), mip);
        REQUIRE(payload.has_value());
        const auto expected = source_mip(source, mip);
        REQUIRE(payload.value().size() == expected.size());
        CHECK(std::equal(payload.value().begin(), payload.value().end(), expected.begin(), expected.end()));
    }
}

TEST_CASE("none mip policy keeps only the authored base level")
{
    using namespace arc;
    using namespace arc::render;
    using namespace arc::render::tools;

    const auto source = authored_bc1_dds();
    auto loaded = load_texture_asset_bytes(source, "authored_albedo.dds");
    REQUIRE(loaded.succeeded());

    auto settings = texture_import_settings_for_preset(texture_import_preset::color);
    settings.streaming_mode = texture_streaming_mode::resident;
    settings.mip_policy = texture_mip_policy::none;
    settings.generate_mips = false;
    const auto processed =
        preprocess_texture_for_cook(std::move(loaded.texture), settings, assets::windows_vulkan_cook_target());
    REQUIRE(processed.has_value());
    CHECK(processed.value().metadata.source_mip_count == 4u);
    CHECK(processed.value().metadata.mip_policy == texture_mip_policy::none);
    CHECK_FALSE(processed.value().metadata.source_mips_preserved);
    REQUIRE(processed.value().texture.mips.size() == 1u);

    const auto artifact =
        encode_texture_artifact(processed.value().texture, settings.streaming_mode, processed.value().metadata);
    REQUIRE(artifact.has_value());
    const auto inspected = inspect_texture_artifact(artifact.value());
    REQUIRE(inspected.has_value());
    CHECK(inspected.value().mip_count == 1u);
    CHECK(inspected.value().metadata.source_mip_count == 4u);
    CHECK(inspected.value().metadata.mip_policy == texture_mip_policy::none);
    const auto payload = read_texture_artifact_mip(artifact.value(), inspected.value(), 0u);
    REQUIRE(payload.has_value());
    const auto expected = source_mip(source, 0u);
    CHECK(std::equal(payload.value().begin(), payload.value().end(), expected.begin(), expected.end()));
}

TEST_CASE("generate mip policy rejects encoded DDS until block decoding is available")
{
    using namespace arc;
    using namespace arc::render;
    using namespace arc::render::tools;

    auto loaded = load_texture_asset_bytes(authored_bc1_dds(), "authored_albedo.dds");
    REQUIRE(loaded.succeeded());
    auto settings = texture_import_settings_for_preset(texture_import_preset::color);
    settings.mip_policy = texture_mip_policy::generate;
    const auto processed =
        preprocess_texture_for_cook(std::move(loaded.texture), settings, assets::windows_vulkan_cook_target());
    REQUIRE_FALSE(processed.has_value());
    CHECK(processed.error().find("DDS block decoding") != std::string::npos);
}
