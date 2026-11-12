#include <arc/render/cooked_material.h>
#include <arc/render/cooked_texture.h>
#include <arc/render/texture_artifact.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <span>
#include <string_view>
#include <vector>

namespace
{
template <class T> void append(std::vector<std::byte>& buffer, const T& value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    buffer.insert(buffer.end(), data, data + sizeof(value));
}

void append_string(std::vector<std::byte>& bytes, std::string_view text)
{
    append(bytes, static_cast<std::uint64_t>(text.size()));
    auto payload = std::as_bytes(std::span(text.data(), text.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
}

std::vector<std::byte> material_bytes()
{
    using namespace arc::render;
    std::vector<std::byte> bytes;
    append_string(bytes, "ARC_MATERIAL_4");
    append(bytes, material_pass_contract_version);
    append(bytes, material_abi_version);
    append(bytes, std::uint64_t{1}); // shader package GUID high
    append(bytes, std::uint64_t{2}); // shader package GUID low
    append(bytes, std::uint32_t{1});
    append(bytes, material_pass::forward);
    append(bytes, std::uint64_t{7}); // permutation
    append(bytes, std::uint64_t{8}); // entry
    std::array<std::byte, 32> hash{};
    bytes.insert(bytes.end(), hash.begin(), hash.end());
    append(bytes, std::uint32_t{1});
    append(bytes, std::uint64_t{11});
    append_string(bytes, "roughness");
    append(bytes, shader_parameter_type::float32);
    append(bytes, std::uint32_t{0});
    append(bytes, std::uint32_t{4});
    append(bytes, true);
    append(bytes, 0.0f);
    append(bytes, 1.0f);
    append_string(bytes, R"({"version":4,"domain":"surface"})");
    return bytes;
}

arc::render::texture_data make_texture()
{
    arc::render::texture_data t;
    t.width = 4;
    t.height = 4;
    t.format = arc::render::texture_format::rgba8_unorm;
    t.color_space = arc::render::texture_color_space::linear;
    t.semantic = arc::render::texture_semantic::normal;
    t.pixels.resize(4u * 4u * 4u, std::byte{0x7f});
    t.mips.push_back({.width = 4, .height = 4, .offset = 0, .size = t.pixels.size()});
    t.mip_levels = 1;
    return t;
}
} // namespace

TEST_CASE("V3.4c.2 reads cooked material v4 without loading authoring tools")
{
    auto bytes = material_bytes();
    const auto material = arc::render::decode_cooked_material_metadata(bytes);
    REQUIRE(material);
    REQUIRE(material.value().compiled.passes.size() == 1);
    CHECK(material.value().compiled.passes[0].pass == arc::render::material_pass::forward);
    REQUIRE(material.value().parameters.size() == 1);
    CHECK(material.value().parameters[0].name == "roughness");
    CHECK(material.value().parameters[0].id.representation() == 11);
}

TEST_CASE("V3.4c.2 rejects material ABI mismatch and malformed trailing bytes")
{
    auto bytes = material_bytes();
    bytes[sizeof(std::uint64_t) + std::string_view("ARC_MATERIAL_4").size()] ^= std::byte{1};
    CHECK_FALSE(arc::render::decode_cooked_material_metadata(bytes));
    bytes = material_bytes();
    bytes.push_back(std::byte{0xff});
    CHECK_FALSE(arc::render::decode_cooked_material_metadata(bytes));
    bytes = material_bytes();
    bytes.pop_back();
    CHECK_FALSE(arc::render::decode_cooked_material_metadata(bytes));
}

TEST_CASE("V3.4c.2 round-trips integrity-checked conventional texture mip data")
{
    using namespace arc::render;
    const auto encoded = encode_texture_artifact(make_texture(), texture_streaming_mode::resident);
    REQUIRE(encoded);
    const auto decoded = decode_cooked_texture_data(encoded.value());
    REQUIRE(decoded);
    CHECK(decoded.value().width == 4);
    CHECK(decoded.value().height == 4);
    CHECK(decoded.value().format == texture_format::rgba8_unorm);
    REQUIRE(decoded.value().mips.size() == 1);
    CHECK(decoded.value().pixels.size() == 64);
    CHECK_FALSE(decoded.value().compressed);
    CHECK(decoded.value().pixels.front() == std::byte{0x7f});
}

TEST_CASE("V3.4c.2 rejects invalid cooked texture data")
{
    using namespace arc::render;
    const auto encoded = encode_texture_artifact(make_texture(), texture_streaming_mode::resident);
    REQUIRE(encoded);
    auto bytes = encoded.value();
    const auto index = inspect_texture_artifact(bytes);
    REQUIRE(index);
    REQUIRE_FALSE(index.value().mips.empty());
    const auto& mip = index.value().mips.front();
    REQUIRE(mip.stored_size > 0);
    // Corrupt the hashed mip payload, not trailing alignment padding.
    bytes[static_cast<std::size_t>(mip.offset)] ^= std::byte{1};
    CHECK_FALSE(decode_cooked_texture_data(bytes));
    bytes = encoded.value();
    // Truncate inside the required mip data, not in alignment padding.
    bytes.resize(static_cast<std::size_t>(mip.offset) + mip.stored_size - 1u);
    CHECK_FALSE(decode_cooked_texture_data(bytes));
}


TEST_CASE("V3.4d.4 restores authored shading, blend mode, sidedness and shadow policy")
{
    using namespace arc::render;
    cooked_material_metadata metadata;
    metadata.compiled.package = {.high = 1, .low = 2};
    metadata.canonical_document_json =
        R"({"version":4,"domain":"surface","shadingModel":"unlit","blendMode":"blend","doubleSided":true,"castShadows":false})";
    auto program = std::make_shared<material_runtime_program>();
    program->passes.push_back({.pass = material_pass::forward, .permutation = shader_permutation_id{7}});
    const auto material = make_cooked_material_descriptor(metadata, program);
    REQUIRE(material);
    CHECK(material.value().shading_model == material_shading_model::unlit);
    CHECK(material.value().alpha_mode == material_alpha_mode::blend);
    CHECK(material.value().double_sided);
    CHECK_FALSE(material.value().cast_shadows);
    CHECK(material.value().shader_permutation == shader_permutation_id{7});
    CHECK(material.value().runtime_program == program);
}

TEST_CASE("V3.4d.4 refuses unknown authored material state and missing executable forward pass")
{
    using namespace arc::render;
    cooked_material_metadata metadata;
    metadata.compiled.package = {.high = 1, .low = 2};
    auto program = std::make_shared<material_runtime_program>();
    program->passes.push_back({.pass = material_pass::forward, .permutation = shader_permutation_id{7}});

    for (const std::string_view document : {
             R"({"version":4,"domain":"terrain","shadingModel":"standard"})",
             R"({"version":4,"domain":"surface","shadingModel":"unknown"})",
             R"({"version":4,"domain":"surface","blendMode":"invalid"})",
             R"({"version":4,"domain":"surface","doubleSided":"yes"})",
             R"({"version":3,"domain":"surface"})",
             R"({"version":4,"domain":"surface","castShadows":2})"})
    {
        metadata.canonical_document_json = document;
        CHECK_FALSE(make_cooked_material_descriptor(metadata, program));
    }

    metadata.canonical_document_json = R"({"version":4,"domain":"surface","blendMode":"masked"})";
    auto no_forward = std::make_shared<material_runtime_program>();
    no_forward->passes.push_back({.pass = material_pass::depth, .permutation = shader_permutation_id{8}});
    CHECK_FALSE(make_cooked_material_descriptor(metadata, no_forward));
    const auto accepted = make_cooked_material_descriptor(metadata, program);
    REQUIRE(accepted);
    CHECK(accepted.value().alpha_mode == material_alpha_mode::masked);
}
