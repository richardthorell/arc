#include <arc/render_tools/material_asset.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>

namespace
{

template <class T> T read_value(std::span<const std::byte> bytes, std::size_t& cursor)
{
    REQUIRE(cursor + sizeof(T) <= bytes.size());
    T value{};
    std::memcpy(&value, bytes.data() + cursor, sizeof(T));
    cursor += sizeof(T);
    return value;
}

std::string read_string(std::span<const std::byte> bytes, std::size_t& cursor)
{
    const auto size = read_value<std::uint64_t>(bytes, cursor);
    REQUIRE(cursor + size <= bytes.size());
    std::string value(reinterpret_cast<const char*>(bytes.data() + cursor), static_cast<std::size_t>(size));
    cursor += static_cast<std::size_t>(size);
    return value;
}

arc::render::tools::material_package_v3 compiled_package()
{
    arc::render::tools::material_package_v3 package;
    package.compiled.package = {.high = 11, .low = 22};
    package.compiled.passes = {
        {.pass = arc::render::material_pass::forward,
         .permutation = {200},
         .entry_point = arc::render::make_shader_entry_point_id("main", arc::render::shader_stage::fragment)},
        {.pass = arc::render::material_pass::depth,
         .permutation = {100},
         .entry_point = arc::render::make_shader_entry_point_id("main", arc::render::shader_stage::fragment)}};
    package.compiled.passes[0].build_hash.bytes[0] = std::byte{0x20};
    package.compiled.passes[1].build_hash.bytes[0] = std::byte{0x10};
    package.parameters.push_back({.id = arc::render::make_shader_parameter_id("roughness"),
                                  .name = "Roughness",
                                  .type = arc::render::shader_parameter_type::float32,
                                  .offset = 0,
                                  .size = 4});
    package.canonical_document_json = R"({"version":4,"graph":{"version":1,"nodes":[],"connections":[]}})";
    return package;
}

} // namespace

TEST_CASE("material authoring accepts the current graph schema without migration")
{
    constexpr std::string_view source = R"({
      "version":4,
      "name":"Current Material",
      "domain":"surface",
      "shadingModel":"transmission",
      "blendMode":"masked",
      "doubleSided":true,
      "graph":{"version":1,"nodes":[],"connections":[]},
      "futureEditorMetadata":{"keep":true}
    })";

    const auto result = arc::render::tools::parse_material_authoring_json(source);
    REQUIRE(result);
    REQUIRE(result.value().source_version == arc::render::tools::material_authoring_version);
    REQUIRE(result.value().version == arc::render::tools::material_authoring_version);
    REQUIRE_FALSE(result.value().migrated);
    REQUIRE_FALSE(result.value().graph_json.empty());
    REQUIRE(result.value().shader_path.empty());
    REQUIRE(result.value().domain == arc::render::material_domain::surface);
    REQUIRE(result.value().shading_model == arc::render::material_shading_model::transmission);
    REQUIRE(result.value().alpha_mode == arc::render::material_alpha_mode::masked);
    REQUIRE(result.value().double_sided);
    REQUIRE(result.value().canonical_json.find("\"futureEditorMetadata\":{\"keep\":true}") != std::string::npos);
}

TEST_CASE("material authoring accepts a current handwritten Material Shader")
{
    const auto result = arc::render::tools::parse_material_authoring_json(
        R"({"version":4,"shaderPath":"Shaders/custom.slang","graph":null})");
    REQUIRE(result);
    REQUIRE(result.value().graph_json.empty());
    REQUIRE(result.value().shader_path == "Shaders/custom.slang");
}

TEST_CASE("material authoring rejects legacy schemas")
{
    constexpr std::string_view missing_source = R"({"graph":{"version":1,"nodes":[],"connections":[]}})";
    const auto missing = arc::render::tools::parse_material_authoring_json(missing_source);
    REQUIRE_FALSE(missing);
    REQUIRE(missing.error().code == arc::render::tools::material_asset_error_code::invalid_document);

    for (const auto version : {1, 2, 3, 5})
    {
        auto source = "{\"version\":" + std::to_string(version);
        source += ",\"graph\":{\"version\":1,\"nodes\":[],\"connections\":[]}}";
        const auto result = arc::render::tools::parse_material_authoring_json(source);
        REQUIRE_FALSE(result);
        REQUIRE(result.error().code == arc::render::tools::material_asset_error_code::unsupported_version);
    }
}

TEST_CASE("material authoring requires exactly one compiled implementation")
{
    const auto missing = arc::render::tools::parse_material_authoring_json(R"({"version":4,"graph":null})");
    REQUIRE_FALSE(missing);
    REQUIRE(missing.error().code == arc::render::tools::material_asset_error_code::invalid_document);

    const auto both = arc::render::tools::parse_material_authoring_json(
        R"({"version":4,"shaderPath":"Shaders/custom.slang","graph":{"version":1,"nodes":[],"connections":[]}})");
    REQUIRE_FALSE(both);
    REQUIRE(both.error().code == arc::render::tools::material_asset_error_code::invalid_document);
}

TEST_CASE("material authoring validates compiled implementation field types")
{
    const auto invalid_graph =
        arc::render::tools::parse_material_authoring_json(R"({"version":4,"graph":[],"shaderPath":null})");
    REQUIRE_FALSE(invalid_graph);
    REQUIRE(invalid_graph.error().code == arc::render::tools::material_asset_error_code::invalid_document);

    const auto invalid_shader =
        arc::render::tools::parse_material_authoring_json(R"({"version":4,"graph":null,"shaderPath":42})");
    REQUIRE_FALSE(invalid_shader);
    REQUIRE(invalid_shader.error().code == arc::render::tools::material_asset_error_code::invalid_document);

    const auto empty_shader =
        arc::render::tools::parse_material_authoring_json(R"({"version":4,"graph":null,"shaderPath":""})");
    REQUIRE_FALSE(empty_shader);
    REQUIRE(empty_shader.error().code == arc::render::tools::material_asset_error_code::invalid_document);
}

TEST_CASE("material package v3 round trips deterministic compiled pass bindings")
{
    const auto package = compiled_package();
    const auto bytes = arc::render::tools::serialize_material_package_v3(package);
    std::size_t cursor{};
    REQUIRE(read_string(std::span<const std::byte>(bytes), cursor) == arc::render::tools::material_package_signature);

    const auto decoded = arc::render::tools::deserialize_material_package_v3(bytes);
    REQUIRE(decoded);
    REQUIRE(decoded.value().compiled.package == package.compiled.package);
    REQUIRE(decoded.value().compiled.passes.size() == 2);
    REQUIRE(decoded.value().compiled.passes[0].pass == arc::render::material_pass::depth);
    REQUIRE(decoded.value().compiled.passes[1].pass == arc::render::material_pass::forward);
    REQUIRE(decoded.value().compiled.passes[0].permutation == arc::render::shader_permutation_id{100});
    REQUIRE(decoded.value().compiled.passes[1].permutation == arc::render::shader_permutation_id{200});
    REQUIRE(decoded.value().parameters.size() == 1);
    REQUIRE(decoded.value().canonical_document_json == package.canonical_document_json);

    auto reversed = package;
    std::ranges::reverse(reversed.compiled.passes);
    REQUIRE(arc::render::tools::serialize_material_package_v3(reversed) == bytes);
}

TEST_CASE("surface material package v3 rejects missing compiled passes")
{
    arc::render::tools::material_package_v3 package;
    package.canonical_document_json = R"({"version":4,"domain":"surface"})";
    const auto bytes = arc::render::tools::serialize_material_package_v3(package);
    const auto decoded = arc::render::tools::deserialize_material_package_v3(bytes);
    REQUIRE_FALSE(decoded);
    REQUIRE(decoded.error().code == arc::render::tools::material_asset_error_code::corrupt_package);
}

TEST_CASE("terrain material package v3 may use the dedicated terrain renderer")
{
    arc::render::tools::material_package_v3 package;
    package.canonical_document_json = R"({"version":4,"domain":"terrain"})";
    const auto bytes = arc::render::tools::serialize_material_package_v3(package);
    const auto decoded = arc::render::tools::deserialize_material_package_v3(bytes);
    REQUIRE(decoded);
    REQUIRE(decoded.value().compiled.passes.empty());
    REQUIRE_FALSE(decoded.value().compiled.package.valid());
}
