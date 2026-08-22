#include <arc/render/material_abi.h>

#include <algorithm>
#include <cmath>

namespace arc::render
{
namespace
{

math::vector3f normalized_or(const math::vector3f& value, const math::vector3f& fallback) noexcept
{
    const float length_squared = value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
    if (length_squared <= 1.0e-12f) return fallback;
    const float inverse_length = 1.0f / std::sqrt(length_squared);
    return {value[0] * inverse_length, value[1] * inverse_length, value[2] * inverse_length};
}

math::vector3f multiplied(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return {lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]};
}

} // namespace

material_surface evaluate_legacy_material(const material_descriptor& material, const material_inputs& inputs,
                                          const legacy_material_samples& samples) noexcept
{
    const auto geometric_normal = normalized_or(inputs.normal_ws, {0.0f, 0.0f, 1.0f});
    const auto normal =
        material.normal_texture.valid() ? normalized_or(samples.normal_ws, geometric_normal) : geometric_normal;
    const auto clear_coat_normal =
        material.clear_coat_normal_texture.valid() ? normalized_or(samples.clear_coat_normal_ws, normal) : normal;

    const auto base_sample = material.base_color_texture.valid() ? samples.base_color : math::vector4f::one;
    const math::vector4f color{material.base_color[0] * inputs.vertex_color[0] * base_sample[0],
                               material.base_color[1] * inputs.vertex_color[1] * base_sample[1],
                               material.base_color[2] * inputs.vertex_color[2] * base_sample[2],
                               material.base_color[3] * inputs.vertex_color[3] * base_sample[3]};

    const float roughness_sample = material.metallic_roughness_texture.valid() ? samples.metallic_roughness[1] : 1.0f;
    const float metallic_sample = material.metallic_roughness_texture.valid() ? samples.metallic_roughness[2] : 1.0f;
    const float occlusion =
        material.occlusion_texture.valid() ? 1.0f + (samples.occlusion - 1.0f) * material.occlusion_strength : 1.0f;
    const auto emissive_sample = material.emissive_texture.valid() ? samples.emissive : math::vector3f::one;
    const auto emissive = multiplied(emissive_sample, material.emissive_factor);

    material_surface surface;
    surface.base_color = {color[0], color[1], color[2]};
    surface.metallic = std::clamp(material.metallic * metallic_sample, 0.0f, 1.0f);
    surface.roughness = std::clamp(material.roughness * roughness_sample, 0.04f, 1.0f);
    surface.normal_ws = normal;
    surface.clear_coat_normal_ws = clear_coat_normal;
    surface.tangent_ws = {inputs.tangent_ws[0], inputs.tangent_ws[1], inputs.tangent_ws[2]};
    surface.ambient_occlusion = occlusion;
    surface.emissive_radiance = {emissive[0] * material.emissive_strength, emissive[1] * material.emissive_strength,
                                 emissive[2] * material.emissive_strength};
    surface.opacity = color[3];
    surface.alpha_cutoff = material.alpha_cutoff;
    surface.index_of_refraction = material.index_of_refraction;
    surface.clear_coat = material.clear_coat_factor * (material.clear_coat_texture.valid() ? samples.clear_coat : 1.0f);
    surface.clear_coat_roughness =
        material.clear_coat_roughness *
        (material.clear_coat_roughness_texture.valid() ? samples.clear_coat_roughness : 1.0f);
    surface.sheen = material.sheen_factor;
    surface.sheen_color = material.sheen_color;
    surface.anisotropy = material.anisotropy_factor * (material.anisotropy_texture.valid() ? samples.anisotropy : 1.0f);
    surface.anisotropy_rotation = material.anisotropy_rotation;
    surface.transmission =
        material.transmission_factor * (material.transmission_texture.valid() ? samples.transmission : 1.0f);
    surface.thickness = material.thickness_factor * (material.thickness_texture.valid() ? samples.thickness : 1.0f);
    surface.attenuation_color = material.attenuation_color;
    surface.attenuation_distance = material.attenuation_distance;
    surface.subsurface_color = material.subsurface_color;
    surface.subsurface = material.subsurface_factor * (material.subsurface_texture.valid() ? samples.subsurface : 1.0f);
    return surface;
}

} // namespace arc::render
