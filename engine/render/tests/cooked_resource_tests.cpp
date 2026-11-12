#include <arc/render/cooked_material.h>
#include <arc/render/cooked_texture.h>
#include <arc/render/texture_artifact.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
    bytes.back() ^= std::byte{1};
    CHECK_FALSE(decode_cooked_texture_data(bytes));
    bytes = encoded.value();
    const auto index = inspect_texture_artifact(bytes);
    REQUIRE(index);
    REQUIRE_FALSE(index.value().mips.empty());
    const auto& mip = index.value().mips.front();
    // Truncate inside the required mip data, not in alignment padding.
    bytes.resize(static_cast<std::size_t>(mip.offset) + mip.stored_size - 1u);
    CHECK_FALSE(decode_cooked_texture_data(bytes));
}
