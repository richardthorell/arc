#include <arc/render/material.h>

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

} // namespace arc::render
