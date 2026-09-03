#include <arc/editor/model_preview.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

TEST_CASE("model preview renders imported geometry with default material", "[editor][thumbnail][model]")
{
    arc::render::scene_import_result scene;
    arc::render::mesh_data mesh;
    mesh.name = "triangle";
    mesh.vertices = {
        {.position = {-1.0f, -0.8f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f}},
        {.position = {1.0f, -0.8f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f}},
        {.position = {0.0f, 1.0f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f}},
    };
    mesh.indices = {0, 1, 2};
    scene.meshes.push_back(std::move(mesh));
    scene.nodes.push_back({.name = "triangle", .mesh_index = 0});

    const auto preview = arc::editor::render_model_preview(scene, {.size = 64});
    REQUIRE(preview.succeeded());
    CHECK(preview.texture.width == 64);
    CHECK(preview.texture.height == 64);

    constexpr std::uint8_t background_r = 22u;
    constexpr std::uint8_t background_g = 25u;
    constexpr std::uint8_t background_b = 30u;
    std::size_t shaded_pixels{};
    for (std::size_t offset = 0; offset + 3u < preview.texture.pixels.size(); offset += 4u)
    {
        const auto r = std::to_integer<std::uint8_t>(preview.texture.pixels[offset]);
        const auto g = std::to_integer<std::uint8_t>(preview.texture.pixels[offset + 1u]);
        const auto b = std::to_integer<std::uint8_t>(preview.texture.pixels[offset + 2u]);
        if (r != background_r || g != background_g || b != background_b) ++shaded_pixels;
    }
    CHECK(shaded_pixels > 50u);
}

TEST_CASE("model preview material can be overridden", "[editor][thumbnail][model]")
{
    arc::render::scene_import_result scene;
    arc::render::mesh_data mesh;
    mesh.vertices = {
        {.position = {-1.0f, -1.0f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f}},
        {.position = {1.0f, -1.0f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f}},
        {.position = {0.0f, 1.0f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f}},
    };
    mesh.indices = {0, 1, 2};
    scene.meshes.push_back(std::move(mesh));
    scene.nodes.push_back({.mesh_index = 0});

    arc::render::material_descriptor override_material;
    override_material.base_color = arc::math::vector4f{0.95f, 0.12f, 0.08f, 1.0f};
    override_material.roughness = 0.35f;
    const auto preview = arc::editor::render_model_preview(
        scene, {.size = 48, .material_override = override_material});
    REQUIRE(preview.succeeded());
}
