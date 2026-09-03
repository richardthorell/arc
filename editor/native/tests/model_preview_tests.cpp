#include <arc/editor/model_preview.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

TEST_CASE("model preview renders centered geometry on a transparent background", "[editor][thumbnail][model]")
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

    constexpr std::uint32_t size = 64;
    const auto preview = arc::editor::render_model_preview(scene, {.size = size});
    REQUIRE(preview.succeeded());
    CHECK(preview.texture.width == size);
    CHECK(preview.texture.height == size);

    std::size_t shaded_pixels{};
    std::uint32_t minimum_x = size;
    std::uint32_t minimum_y = size;
    std::uint32_t maximum_x{};
    std::uint32_t maximum_y{};
    for (std::uint32_t y = 0; y < size; ++y)
    {
        for (std::uint32_t x = 0; x < size; ++x)
        {
            const auto offset = (static_cast<std::size_t>(y) * size + x) * 4u;
            const auto alpha = std::to_integer<std::uint8_t>(preview.texture.pixels[offset + 3u]);
            if (alpha == 0u) continue;
            ++shaded_pixels;
            minimum_x = std::min(minimum_x, x);
            minimum_y = std::min(minimum_y, y);
            maximum_x = std::max(maximum_x, x);
            maximum_y = std::max(maximum_y, y);
        }
    }

    CHECK(shaded_pixels > 50u);
    CHECK(std::to_integer<std::uint8_t>(preview.texture.pixels[3u]) == 0u);
    REQUIRE(minimum_x <= maximum_x);
    REQUIRE(minimum_y <= maximum_y);
    const float bounds_center_x = (static_cast<float>(minimum_x) + static_cast<float>(maximum_x)) * 0.5f;
    const float bounds_center_y = (static_cast<float>(minimum_y) + static_cast<float>(maximum_y)) * 0.5f;
    const float image_center = static_cast<float>(size - 1u) * 0.5f;
    CHECK(std::abs(bounds_center_x - image_center) <= 3.0f);
    CHECK(std::abs(bounds_center_y - image_center) <= 3.0f);
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
    const auto preview = arc::editor::render_model_preview(scene, {.size = 48, .material_override = override_material});
    REQUIRE(preview.succeeded());
}
