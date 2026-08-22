#include <arc/render/material_abi.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("material ABI v1 legacy evaluator preserves descriptor defaults")
{
    const arc::render::material_descriptor material;
    const arc::render::material_inputs inputs;

    const auto surface = arc::render::evaluate_legacy_material(material, inputs);

    REQUIRE(arc::render::material_abi_version == 1);
    REQUIRE(surface.base_color[0] == Catch::Approx(1.0f));
    REQUIRE(surface.base_color[1] == Catch::Approx(1.0f));
    REQUIRE(surface.base_color[2] == Catch::Approx(1.0f));
    REQUIRE(surface.opacity == Catch::Approx(1.0f));
    REQUIRE(surface.metallic == Catch::Approx(0.0f));
    REQUIRE(surface.roughness == Catch::Approx(0.6f));
    REQUIRE(surface.ambient_occlusion == Catch::Approx(1.0f));
    REQUIRE(surface.alpha_cutoff == Catch::Approx(0.5f));
    REQUIRE(surface.index_of_refraction == Catch::Approx(1.5f));
    REQUIRE(surface.sheen_roughness == Catch::Approx(0.5f));
    REQUIRE(surface.normal_ws[2] == Catch::Approx(1.0f));
    REQUIRE(surface.clear_coat_normal_ws[2] == Catch::Approx(1.0f));
}

TEST_CASE("material ABI v1 legacy evaluator preserves texture and advanced factors")
{
    arc::render::material_descriptor material;
    material.base_color = {0.8f, 0.6f, 0.4f, 0.5f};
    material.metallic = 0.75f;
    material.roughness = 0.5f;
    material.alpha_cutoff = 0.33f;
    material.base_color_texture = {.index = 1, .generation = 1};
    material.metallic_roughness_texture = {.index = 2, .generation = 1};
    material.normal_texture = {.index = 3, .generation = 1};
    material.occlusion_texture = {.index = 4, .generation = 1};
    material.emissive_texture = {.index = 5, .generation = 1};
    material.clear_coat_texture = {.index = 6, .generation = 1};
    material.clear_coat_roughness_texture = {.index = 7, .generation = 1};
    material.clear_coat_normal_texture = {.index = 8, .generation = 1};
    material.anisotropy_texture = {.index = 9, .generation = 1};
    material.subsurface_texture = {.index = 10, .generation = 1};
    material.thickness_texture = {.index = 11, .generation = 1};
    material.transmission_texture = {.index = 12, .generation = 1};
    material.occlusion_strength = 0.5f;
    material.emissive_factor = {2.0f, 1.0f, 0.5f};
    material.emissive_strength = 3.0f;
    material.clear_coat_factor = 0.8f;
    material.clear_coat_roughness = 0.4f;
    material.anisotropy_factor = 0.6f;
    material.transmission_factor = 0.9f;
    material.thickness_factor = 0.25f;
    material.subsurface_factor = 0.7f;

    arc::render::material_inputs inputs;
    inputs.vertex_color = {0.5f, 1.0f, 0.25f, 0.8f};
    inputs.tangent_ws = {0.0f, 1.0f, 0.0f, -1.0f};

    arc::render::legacy_material_samples samples;
    samples.base_color = {0.5f, 0.25f, 1.0f, 0.5f};
    samples.metallic_roughness = {1.0f, 0.5f, 0.25f, 1.0f};
    samples.normal_ws = {0.0f, 1.0f, 0.0f};
    samples.occlusion = 0.4f;
    samples.emissive = {0.25f, 0.5f, 1.0f};
    samples.clear_coat = 0.5f;
    samples.clear_coat_roughness = 0.25f;
    samples.clear_coat_normal_ws = {1.0f, 0.0f, 0.0f};
    samples.anisotropy = 0.5f;
    samples.transmission = 0.5f;
    samples.thickness = 0.4f;
    samples.subsurface = 0.25f;

    const auto surface = arc::render::evaluate_legacy_material(material, inputs, samples);

    REQUIRE(surface.base_color[0] == Catch::Approx(0.2f));
    REQUIRE(surface.base_color[1] == Catch::Approx(0.15f));
    REQUIRE(surface.base_color[2] == Catch::Approx(0.1f));
    REQUIRE(surface.opacity == Catch::Approx(0.2f));
    REQUIRE(surface.metallic == Catch::Approx(0.1875f));
    REQUIRE(surface.roughness == Catch::Approx(0.25f));
    REQUIRE(surface.ambient_occlusion == Catch::Approx(0.7f));
    REQUIRE(surface.emissive_radiance[0] == Catch::Approx(1.5f));
    REQUIRE(surface.emissive_radiance[1] == Catch::Approx(1.5f));
    REQUIRE(surface.emissive_radiance[2] == Catch::Approx(1.5f));
    REQUIRE(surface.clear_coat == Catch::Approx(0.4f));
    REQUIRE(surface.clear_coat_roughness == Catch::Approx(0.1f));
    REQUIRE(surface.anisotropy == Catch::Approx(0.3f));
    REQUIRE(surface.transmission == Catch::Approx(0.45f));
    REQUIRE(surface.thickness == Catch::Approx(0.1f));
    REQUIRE(surface.subsurface == Catch::Approx(0.175f));
    REQUIRE(surface.normal_ws[1] == Catch::Approx(1.0f));
    REQUIRE(surface.clear_coat_normal_ws[0] == Catch::Approx(1.0f));
    REQUIRE(surface.tangent_ws[1] == Catch::Approx(1.0f));
}
