#include <arc/scene/cooked_texture_hydration.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>

namespace
{
arc::render::texture_data texture()
{
    arc::render::texture_data value;
    value.width = 4;
    value.height = 4;
    value.format = arc::render::texture_format::rgba8_unorm;
    value.color_space = arc::render::texture_color_space::linear;
    value.semantic = arc::render::texture_semantic::normal;
    value.pixels.assign(64, std::byte{0x7f});
    value.mips.push_back({.width = 4, .height = 4, .offset = 0, .size = 64});
    return value;
}
} // namespace

TEST_CASE("V3.4d.2 deduplicates cooked textures by GUID and releases renderer generation")
{
    arc::render::renderer renderer;
    const arc::assets::asset_guid first{1, 2};
    const arc::assets::asset_guid second{3, 4};
    const std::array uploads{arc::scene::cooked_texture_upload{.asset = first, .texture = texture()},
                             arc::scene::cooked_texture_upload{.asset = second, .texture = texture()},
                             arc::scene::cooked_texture_upload{.asset = first, .texture = texture()}};
    arc::scene::cooked_texture_residencies residency;

    CHECK(arc::scene::hydrate_cooked_scene_textures(renderer, uploads, residency).empty());
    REQUIRE(residency.size() == 2);
    const auto first_handle = arc::scene::find_cooked_scene_texture(residency, first);
    const auto second_handle = arc::scene::find_cooked_scene_texture(residency, second);
    REQUIRE(first_handle.valid());
    REQUIRE(second_handle.valid());
    CHECK(renderer.texture_alive(first_handle));
    CHECK(renderer.texture_alive(second_handle));
    CHECK_FALSE(arc::scene::find_cooked_scene_texture(residency, {9, 10}).valid());
    CHECK_FALSE(arc::scene::hydrate_cooked_scene_textures(renderer, uploads, residency).empty());

    arc::scene::release_cooked_scene_textures(renderer, residency);
    CHECK(residency.empty());
    CHECK_FALSE(renderer.texture_alive(first_handle));
    CHECK_FALSE(renderer.texture_alive(second_handle));
    CHECK(arc::scene::hydrate_cooked_scene_textures(renderer, uploads, residency).empty());
    REQUIRE(residency.size() == 2);
    arc::scene::release_cooked_scene_textures(renderer, residency);
}

TEST_CASE("V3.4d.2 rejects malformed cooked textures before publishing any handles")
{
    arc::render::renderer renderer;
    arc::scene::cooked_texture_residencies residency;
    auto invalid = texture();
    invalid.pixels.clear();
    const std::array bad_payload{arc::scene::cooked_texture_upload{.asset = {1, 2}, .texture = texture()},
                                 arc::scene::cooked_texture_upload{.asset = {3, 4}, .texture = invalid}};
    CHECK_FALSE(arc::scene::hydrate_cooked_scene_textures(renderer, bad_payload, residency).empty());
    CHECK(residency.empty());

    const std::array bad_guid{arc::scene::cooked_texture_upload{.asset = {}, .texture = texture()}};
    CHECK_FALSE(arc::scene::hydrate_cooked_scene_textures(renderer, bad_guid, residency).empty());
    CHECK(residency.empty());
}
