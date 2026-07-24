#include <arc/render/material.h>

#include <algorithm>
#include <cmath>
#include <functional>

namespace arc::render
{

shader_permutation_key make_shader_permutation_key(
    const material_desc& material,
    std::uint8_t debug_view,
    bool wireframe) noexcept
{
    return {
        .alpha_mode = material.alpha_mode,
        .debug_view = debug_view,
        .has_base_color_texture = material.base_color_texture.valid(),
        .has_metallic_roughness_texture = material.metallic_roughness_texture.valid(),
        .has_normal_texture = material.normal_texture.valid(),
        .has_occlusion_texture = material.occlusion_texture.valid(),
        .has_emissive_texture = material.emissive_texture.valid(),
        .has_clear_coat_texture = material.clear_coat_texture.valid(),
        .has_clear_coat_roughness_texture = material.clear_coat_roughness_texture.valid(),
        .has_clear_coat_normal_texture = material.clear_coat_normal_texture.valid(),
        .has_anisotropy_texture = material.anisotropy_texture.valid(),
        .has_subsurface_texture = material.subsurface_texture.valid(),
        .has_thickness_texture = material.thickness_texture.valid(),
        .has_transmission_texture = material.transmission_texture.valid(),
        .double_sided = material.double_sided,
        .wireframe = wireframe,
        .clear_coat = material.clear_coat_factor > 0.0f,
        .sheen = material.sheen_factor > 0.0f,
        .transmission = material.transmission_factor > 0.0f,
        .subsurface = material.subsurface_factor > 0.0f,
        .anisotropy = material.anisotropy_factor != 0.0f,
        .parallax = material.parallax_height_scale != 0.0f ||
            material.displacement_mode != material_displacement_mode::none
    };
}

std::size_t hash_shader_permutation_key(const shader_permutation_key& key) noexcept
{
    auto combine = [](std::size_t seed, std::size_t value) {
        return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u));
    };

    std::size_t seed = 0;
    seed = combine(seed, std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(key.alpha_mode)));
    seed = combine(seed, std::hash<std::uint8_t>{}(key.debug_view));
    seed = combine(seed, std::hash<bool>{}(key.has_base_color_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_metallic_roughness_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_normal_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_occlusion_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_emissive_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_clear_coat_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_clear_coat_roughness_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_clear_coat_normal_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_anisotropy_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_subsurface_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_thickness_texture));
    seed = combine(seed, std::hash<bool>{}(key.has_transmission_texture));
    seed = combine(seed, std::hash<bool>{}(key.double_sided));
    seed = combine(seed, std::hash<bool>{}(key.wireframe));
    seed = combine(seed, std::hash<bool>{}(key.clear_coat));
    seed = combine(seed, std::hash<bool>{}(key.sheen));
    seed = combine(seed, std::hash<bool>{}(key.transmission));
    seed = combine(seed, std::hash<bool>{}(key.subsurface));
    seed = combine(seed, std::hash<bool>{}(key.anisotropy));
    seed = combine(seed, std::hash<bool>{}(key.parallax));
    return seed;
}

float srgb_to_linear(float value) noexcept
{
    value = std::max(value, 0.0f);
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb(float value) noexcept
{
    value = std::max(value, 0.0f);
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

math::vector3f srgb_to_linear(const math::vector3f& value) noexcept
{
    return { srgb_to_linear(value[0]), srgb_to_linear(value[1]), srgb_to_linear(value[2]) };
}

math::vector3f linear_to_srgb(const math::vector3f& value) noexcept
{
    return { linear_to_srgb(value[0]), linear_to_srgb(value[1]), linear_to_srgb(value[2]) };
}

float ggx_distribution(float n_dot_h, float roughness) noexcept
{
    const float alpha = std::max(roughness * roughness, 0.001f);
    const float alpha_squared = alpha * alpha;
    const float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0f) + 1.0f;
    return alpha_squared / std::max(3.14159265358979323846f * denominator * denominator, 1.0e-6f);
}

float smith_ggx_correlated(float n_dot_v, float n_dot_l, float roughness) noexcept
{
    n_dot_v = std::clamp(n_dot_v, 0.0f, 1.0f);
    n_dot_l = std::clamp(n_dot_l, 0.0f, 1.0f);
    const float alpha = std::max(roughness * roughness, 0.001f);
    const float alpha_squared = alpha * alpha;
    const float lambda_v = n_dot_l * std::sqrt(std::max(
        n_dot_v * n_dot_v * (1.0f - alpha_squared) + alpha_squared, 0.0f));
    const float lambda_l = n_dot_v * std::sqrt(std::max(
        n_dot_l * n_dot_l * (1.0f - alpha_squared) + alpha_squared, 0.0f));
    return 0.5f / std::max(lambda_v + lambda_l, 1.0e-6f);
}

math::vector3f fresnel_schlick(float cos_theta, const math::vector3f& f0) noexcept
{
    const float factor = std::pow(1.0f - std::clamp(cos_theta, 0.0f, 1.0f), 5.0f);
    return {
        f0[0] + (1.0f - f0[0]) * factor,
        f0[1] + (1.0f - f0[1]) * factor,
        f0[2] + (1.0f - f0[2]) * factor
    };
}

math::vector3f beer_lambert_attenuation(
    const math::vector3f& attenuation_color,
    float attenuation_distance,
    float thickness) noexcept
{
    if (!std::isfinite(attenuation_distance) || attenuation_distance <= 0.0f)
        return { 1.0f, 1.0f, 1.0f };

    const float path = std::max(thickness, 0.0f) / attenuation_distance;
    return {
        std::pow(std::clamp(attenuation_color[0], 1.0e-6f, 1.0f), path),
        std::pow(std::clamp(attenuation_color[1], 1.0e-6f, 1.0f), path),
        std::pow(std::clamp(attenuation_color[2], 1.0e-6f, 1.0f), path)
    };
}

} // namespace arc::render
