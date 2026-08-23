#include <arc/render/material.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("material descriptor production defaults remain stable")
{
    const arc::render::material_descriptor material;

    REQUIRE_FALSE(material.handle.valid());
    REQUIRE(material.name.empty());
    REQUIRE(material.domain == arc::render::material_domain::surface);
    REQUIRE(material.shading_model == arc::render::material_shading_model::standard);

    REQUIRE(material.base_color[0] == Catch::Approx(1.0f));
    REQUIRE(material.base_color[1] == Catch::Approx(1.0f));
    REQUIRE(material.base_color[2] == Catch::Approx(1.0f));
    REQUIRE(material.base_color[3] == Catch::Approx(1.0f));
    REQUIRE(material.metallic == Catch::Approx(0.0f));
    REQUIRE(material.roughness == Catch::Approx(0.6f));
    REQUIRE(material.alpha_cutoff == Catch::Approx(0.5f));
    REQUIRE(material.alpha_mode == arc::render::material_alpha_mode::opaque);
    REQUIRE_FALSE(material.double_sided);

    REQUIRE_FALSE(material.base_color_texture.valid());
    REQUIRE_FALSE(material.metallic_roughness_texture.valid());
    REQUIRE_FALSE(material.normal_texture.valid());
    REQUIRE_FALSE(material.occlusion_texture.valid());
    REQUIRE_FALSE(material.emissive_texture.valid());
    REQUIRE_FALSE(material.clear_coat_texture.valid());
    REQUIRE_FALSE(material.clear_coat_roughness_texture.valid());
    REQUIRE_FALSE(material.clear_coat_normal_texture.valid());
    REQUIRE_FALSE(material.anisotropy_texture.valid());
    REQUIRE_FALSE(material.subsurface_texture.valid());
    REQUIRE_FALSE(material.thickness_texture.valid());
    REQUIRE_FALSE(material.transmission_texture.valid());

    REQUIRE(material.normal_scale == Catch::Approx(1.0f));
    REQUIRE(material.occlusion_strength == Catch::Approx(1.0f));
    REQUIRE(material.emissive_factor[0] == Catch::Approx(0.0f));
    REQUIRE(material.emissive_factor[1] == Catch::Approx(0.0f));
    REQUIRE(material.emissive_factor[2] == Catch::Approx(0.0f));
    REQUIRE(material.emissive_strength == Catch::Approx(1.0f));
    REQUIRE(material.emissive_luminance_nits == Catch::Approx(0.0f));

    REQUIRE(material.clear_coat_factor == Catch::Approx(0.0f));
    REQUIRE(material.clear_coat_roughness == Catch::Approx(0.0f));
    REQUIRE(material.clear_coat_normal_scale == Catch::Approx(1.0f));
    REQUIRE(material.sheen_factor == Catch::Approx(0.0f));
    REQUIRE(material.transmission_factor == Catch::Approx(0.0f));
    REQUIRE(material.index_of_refraction == Catch::Approx(1.5f));
    REQUIRE(material.thickness_factor == Catch::Approx(0.0f));
    REQUIRE(material.attenuation_color[0] == Catch::Approx(1.0f));
    REQUIRE(material.attenuation_color[1] == Catch::Approx(1.0f));
    REQUIRE(material.attenuation_color[2] == Catch::Approx(1.0f));
    REQUIRE(material.attenuation_distance == Catch::Approx(1.0f));
    REQUIRE(material.subsurface_factor == Catch::Approx(0.0f));
    REQUIRE(material.anisotropy_factor == Catch::Approx(0.0f));
    REQUIRE(material.anisotropy_rotation == Catch::Approx(0.0f));
    REQUIRE(material.parallax_height_scale == Catch::Approx(0.0f));
    REQUIRE(material.displacement_mode == arc::render::material_displacement_mode::none);
    REQUIRE_FALSE(material.material_graph.valid());

    for (const auto& layer : material.terrain_layers)
    {
        REQUIRE(layer.name.empty());
        REQUIRE_FALSE(layer.base_color_texture.valid());
        REQUIRE_FALSE(layer.normal_texture.valid());
        REQUIRE_FALSE(layer.packed_surface_texture.valid());
        REQUIRE(layer.tint[0] == Catch::Approx(1.0f));
        REQUIRE(layer.tint[1] == Catch::Approx(1.0f));
        REQUIRE(layer.tint[2] == Catch::Approx(1.0f));
        REQUIRE(layer.tint[3] == Catch::Approx(1.0f));
        REQUIRE(layer.world_scale == Catch::Approx(4.0f));
        REQUIRE(layer.roughness == Catch::Approx(0.8f));
    }
}

TEST_CASE("representative advanced material preserves the shader permutation contract")
{
    arc::render::material_descriptor material;
    material.domain = arc::render::material_domain::surface;
    material.shading_model = arc::render::material_shading_model::transmission;
    material.alpha_mode = arc::render::material_alpha_mode::masked;
    material.double_sided = true;
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
    material.clear_coat_factor = 0.7f;
    material.sheen_factor = 0.3f;
    material.transmission_factor = 0.8f;
    material.subsurface_factor = 0.4f;
    material.anisotropy_factor = 0.5f;
    material.parallax_height_scale = 0.025f;
    material.displacement_mode = arc::render::material_displacement_mode::parallax;

    const auto key = arc::render::make_shader_permutation_key(material, 7, true);

    REQUIRE(key.alpha_mode == arc::render::material_alpha_mode::masked);
    REQUIRE(key.debug_view == 7);
    REQUIRE(key.has_base_color_texture);
    REQUIRE(key.has_metallic_roughness_texture);
    REQUIRE(key.has_normal_texture);
    REQUIRE(key.has_occlusion_texture);
    REQUIRE(key.has_emissive_texture);
    REQUIRE(key.has_clear_coat_texture);
    REQUIRE(key.has_clear_coat_roughness_texture);
    REQUIRE(key.has_clear_coat_normal_texture);
    REQUIRE(key.has_anisotropy_texture);
    REQUIRE(key.has_subsurface_texture);
    REQUIRE(key.has_thickness_texture);
    REQUIRE(key.has_transmission_texture);
    REQUIRE(key.double_sided);
    REQUIRE(key.wireframe);
    REQUIRE(key.clear_coat);
    REQUIRE(key.sheen);
    REQUIRE(key.transmission);
    REQUIRE(key.subsurface);
    REQUIRE(key.anisotropy);
    REQUIRE(key.parallax);
}
