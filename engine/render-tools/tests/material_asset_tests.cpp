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

} // namespace

TEST_CASE("material authoring schema migrates historical versions without dropping fields")
{
    constexpr std::string_view source = R"({
      "name":"Legacy Material",
      "shaderPath":"Shaders/custom.slang",
      "graph":{"version":1,"nodes":[],"connections":[]},
      "futureEditorMetadata":{"keep":true}
    })";

    const auto result = arc::render::tools::parse_material_authoring_json(source);
    REQUIRE(result);
    REQUIRE(result.value().source_version == 1);
    REQUIRE(result.value().version == arc::render::tools::material_authoring_version);
    REQUIRE(result.value().migrated);
    REQUIRE(result.value().shader_path == "Shaders/custom.slang");
    REQUIRE_FALSE(result.value().graph_json.empty());
    REQUIRE(result.value().canonical_json.find("\"version\":4") != std::string::npos);
    REQUIRE(result.value().canonical_json.find("\"name\":\"Legacy Material\"") != std::string::npos);
    REQUIRE(result.value().canonical_json.find("\"futureEditorMetadata\":{\"keep\":true}") != std::string::npos);
    REQUIRE(result.value().graph_json.find("\"version\":1") != std::string::npos);
}

TEST_CASE("material authoring migration exposes pass routing properties")
{
    const auto material = arc::render::tools::parse_material_authoring_json(
        R"({"version":3,"domain":"surface","shadingModel":"transmission","blendMode":"masked","doubleSided":true})");
    REQUIRE(material);
    REQUIRE(material.value().domain == arc::render::material_domain::surface);
    REQUIRE(material.value().shading_model == arc::render::material_shading_model::transmission);
    REQUIRE(material.value().alpha_mode == arc::render::material_alpha_mode::masked);
    REQUIRE(material.value().double_sided);
}

TEST_CASE("material authoring schema accepts current and nullable legacy documents")
{
    const auto current = arc::render::tools::parse_material_authoring_json(R"({"version":4,"name":"Current"})");
    REQUIRE(current);
    REQUIRE(current.value().source_version == 4);
    REQUIRE_FALSE(current.value().migrated);

    const auto legacy = arc::render::tools::parse_material_authoring_json(R"({"version":3,"graph":null})");
    REQUIRE(legacy);
    REQUIRE(legacy.value().migrated);
    REQUIRE(legacy.value().graph_json.empty());
    REQUIRE(legacy.value().canonical_json.find("\"graph\":null") != std::string::npos);

    const auto future = arc::render::tools::parse_material_authoring_json(R"({"version":5})");
    REQUIRE_FALSE(future);
    REQUIRE(future.error().code == arc::render::tools::material_asset_error_code::unsupported_version);
}

TEST_CASE("material authoring schema validates typed migration fields")
{
    const auto invalid_graph = arc::render::tools::parse_material_authoring_json(R"({"version":3,"graph":[]})");
    REQUIRE_FALSE(invalid_graph);
    REQUIRE(invalid_graph.error().code == arc::render::tools::material_asset_error_code::invalid_document);

    const auto invalid_shader = arc::render::tools::parse_material_authoring_json(R"({"version":2,"shaderPath":42})");
    REQUIRE_FALSE(invalid_shader);
    REQUIRE(invalid_shader.error().code == arc::render::tools::material_asset_error_code::invalid_document);
}

TEST_CASE("material package v2 preserves the established binary envelope")
{
    arc::render::tools::material_package_v2 package;
    package.shader_package = {.high = 11, .low = 22};
    package.permutation = {33};
    package.parameters.push_back({.id = arc::render::make_shader_parameter_id("roughness"),
                                  .name = "Roughness",
                                  .type = arc::render::shader_parameter_type::float32,
                                  .offset = 16,
                                  .size = 4});
    package.canonical_document_json = R"({"version":4})";

    const auto bytes = arc::render::tools::serialize_material_package_v2(package);
    std::size_t cursor{};
    const std::span<const std::byte> view(bytes);
    REQUIRE(read_string(view, cursor) == arc::render::tools::material_package_v2_signature);
    REQUIRE(read_value<std::uint64_t>(view, cursor) == 11);
    REQUIRE(read_value<std::uint64_t>(view, cursor) == 22);
    REQUIRE(read_value<std::uint64_t>(view, cursor) == 33);
    REQUIRE(read_value<std::uint32_t>(view, cursor) == 1);
    REQUIRE(read_value<std::uint64_t>(view, cursor) ==
            arc::render::make_shader_parameter_id("roughness").representation());
    REQUIRE(read_string(view, cursor) == "Roughness");
    REQUIRE(read_value<arc::render::shader_parameter_type>(view, cursor) ==
            arc::render::shader_parameter_type::float32);
    REQUIRE(read_value<std::uint32_t>(view, cursor) == 16);
    REQUIRE(read_value<std::uint32_t>(view, cursor) == 4);
    REQUIRE(read_string(view, cursor) == package.canonical_document_json);
    REQUIRE(cursor == bytes.size());
}

TEST_CASE("material package v3 round trips deterministic pass bindings")
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
    package.canonical_document_json = R"({"version":4})";

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

TEST_CASE("material package v3 supports legacy-only materials without compiled passes")
{
    arc::render::tools::material_package_v3 package;
    package.canonical_document_json = R"({"version":4,"graph":null})";
    const auto bytes = arc::render::tools::serialize_material_package_v3(package);
    const auto decoded = arc::render::tools::deserialize_material_package_v3(bytes);
    REQUIRE(decoded);
    REQUIRE_FALSE(decoded.value().compiled.package.valid());
    REQUIRE(decoded.value().compiled.passes.empty());
}
