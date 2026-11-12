#include <arc/render/cooked_material_texture_bindings.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace
{
arc::render::cooked_material_texture_manifest sample_manifest()
{
    using namespace arc::render;
    cooked_material_texture_manifest result;
    result.material = {.high = 9, .low = 10};
    result.textures.push_back({.binding = {.slot = 0,
                                          .parameter_id = shader_parameter_id{31},
                                          .type = shader_parameter_type::texture_2d,
                                          .dimension_slot = 0,
                                          .semantic = texture_semantic::base_color},
                               .asset = {.high = 1, .low = 2}});
    result.textures.push_back({.binding = {.slot = 1,
                                          .type = shader_parameter_type::texture_2d,
                                          .dimension_slot = 1,
                                          .semantic = texture_semantic::normal},
                               .asset = {.high = 3, .low = 4}});
    return result;
}

arc::render::material_descriptor sample_material()
{
    arc::render::material_descriptor material;
    material.shader_package = {.high = 9, .low = 10};
    auto program = std::make_shared<arc::render::material_runtime_program>();
    program->parameters.push_back({.id = arc::render::shader_parameter_id{31},
                                   .name = "albedo",
                                   .type = arc::render::shader_parameter_type::texture_2d});
    material.runtime_program = std::move(program);
    return material;
}
} // namespace

TEST_CASE("V3.4d.5 cooked texture manifest round trips typed slot identities and GUIDs")
{
    using namespace arc::render;
    const auto manifest = sample_manifest();
    const auto encoded = encode_cooked_material_texture_manifest(manifest);
    REQUIRE(encoded);
    const auto decoded = decode_cooked_material_texture_manifest(encoded.value());
    REQUIRE(decoded);
    REQUIRE(decoded.value().textures.size() == 2);
    CHECK(decoded.value().material == manifest.material);
    CHECK(decoded.value().textures[0].asset == manifest.textures[0].asset);
    CHECK(decoded.value().textures[0].binding.parameter_id == shader_parameter_id{31});
    CHECK(decoded.value().textures[1].binding.dimension_slot == 1);
    CHECK(decoded.value().textures[1].binding.semantic == texture_semantic::normal);
    CHECK(encoded.value() == encode_cooked_material_texture_manifest(decoded.value()).value());
}

TEST_CASE("V3.4d.5 rejects truncated, trailing and invalid texture binding slots")
{
    using namespace arc::render;
    const auto original = encode_cooked_material_texture_manifest(sample_manifest());
    REQUIRE(original);
    auto bytes = original.value();
    bytes.pop_back();
    CHECK_FALSE(decode_cooked_material_texture_manifest(bytes));
    bytes = original.value();
    bytes.push_back(std::byte{0xff});
    CHECK_FALSE(decode_cooked_material_texture_manifest(bytes));
    bytes = original.value();
    bytes[0] ^= std::byte{1};
    CHECK_FALSE(decode_cooked_material_texture_manifest(bytes));

    auto manifest = sample_manifest();
    manifest.textures[1].binding.slot = 0;
    CHECK_FALSE(encode_cooked_material_texture_manifest(manifest));
    manifest = sample_manifest();
    manifest.textures[1].binding.dimension_slot = 0;
    CHECK_FALSE(encode_cooked_material_texture_manifest(manifest));
    manifest = sample_manifest();
    manifest.textures[1].binding.type = shader_parameter_type::sampler;
    CHECK_FALSE(encode_cooked_material_texture_manifest(manifest));
}

TEST_CASE("V3.4d.5 binds exact resident texture handles without modifying source program")
{
    using namespace arc::render;
    auto original = sample_material();
    const auto original_program = original.runtime_program;
    const auto manifest = sample_manifest();
    const std::array residents{
        std::pair{arc::assets::asset_guid{3, 4}, texture_handle{4, 1}},
        std::pair{arc::assets::asset_guid{1, 2}, texture_handle{7, 1}}};
    const auto bound = bind_cooked_material_textures(original, manifest, residents);
    REQUIRE(bound);
    REQUIRE(bound.value().runtime_textures.size() == 2);
    CHECK(bound.value().runtime_textures[0] == residents[1].second);
    CHECK(bound.value().runtime_textures[1] == residents[0].second);
    REQUIRE(bound.value().runtime_program->texture_bindings.size() == 2);
    CHECK(original_program->texture_bindings.empty());
    CHECK(original.runtime_textures.empty());

    auto unavailable = sample_manifest();
    unavailable.textures[0].asset = {.high = 42, .low = 43};
    CHECK_FALSE(bind_cooked_material_textures(original, unavailable, residents));
    unavailable = sample_manifest();
    unavailable.material = {.high = 43, .low = 44};
    CHECK_FALSE(bind_cooked_material_textures(original, unavailable, residents));
    unavailable = sample_manifest();
    unavailable.textures[0].binding.parameter_id = shader_parameter_id{99};
    CHECK_FALSE(bind_cooked_material_textures(original, unavailable, residents));
}

TEST_CASE("V3.4d.5 preserves unbound texture slots for semantic fallback")
{
    using namespace arc::render;
    auto manifest = sample_manifest();
    manifest.textures[0].asset = {};
    const std::array residents{std::pair{arc::assets::asset_guid{3, 4}, texture_handle{4, 1}}};
    const auto bound = bind_cooked_material_textures(sample_material(), manifest, residents);
    REQUIRE(bound);
    REQUIRE(bound.value().runtime_textures.size() == 2);
    CHECK_FALSE(bound.value().runtime_textures[0].valid());
    CHECK(bound.value().runtime_textures[1] == residents[0].second);
}
