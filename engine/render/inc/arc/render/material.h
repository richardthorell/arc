#pragma once

#include <arc/render/handles.h>
#include <arc/math/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace arc::render
{

enum class material_domain : std::uint8_t { surface, terrain };

/**
 * @brief Physically based surface model selected by an authored material.
 */
enum class material_shading_model : std::uint8_t
{
    standard,
    skin,
    transmission
};

/**
 * @brief Declared transfer function for decoded texture values.
 */
enum class texture_color_space : std::uint8_t
{
    linear,
    srgb
};

/**
 * @brief Intended use of a texture. The semantic determines validation and
 * the default color-space interpretation when importing an asset.
 */
enum class texture_semantic : std::uint8_t
{
    generic_color,
    base_color,
    emissive,
    normal,
    metallic_roughness,
    occlusion,
    clear_coat,
    anisotropy,
    thickness,
    transmission,
    environment
};

struct terrain_layer_desc
{
    std::string name;
    texture_handle base_color_texture{};
    texture_handle normal_texture{};
    texture_handle packed_surface_texture{};
    math::vector4f tint{ 1.0f, 1.0f, 1.0f, 1.0f };
    float world_scale{ 4.0f };
    float roughness{ 0.8f };
};

/**
 * @brief Alpha/material queue mode used for sorting and pass selection.
 */
enum class material_alpha_mode : std::uint8_t
{
    opaque,
    masked,
    blend
};

/**
 * @brief CPU texture format understood by the renderer facade.
 */
enum class texture_format : std::uint8_t
{
    rgba8_unorm,
    rgba8_srgb,
    rgba16f,
    rgba32f,
    bc1_rgba_unorm,
    bc1_rgba_srgb,
    bc2_rgba_unorm,
    bc2_rgba_srgb,
    bc3_rgba_unorm,
    bc3_rgba_srgb,
    bc4_r_unorm,
    bc5_rg_unorm,
    bc6h_rgb_ufloat,
    bc7_rgba_unorm,
    bc7_rgba_srgb
};

/** @brief Backend-neutral texture dimensionality. */
enum class texture_dimension : std::uint8_t
{
    texture_2d,
    texture_3d,
    cube
};

/**
 * @brief Byte range for one texture mip level in the encoded payload.
 */
struct texture_mip_data
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::size_t offset{};
    std::size_t size{};
};

/**
 * @brief Renderer texture payload.
 *
 * Pixels are expected to be tightly packed RGBA8 when present. When `mips` is
 * populated, it describes packed offsets in `pixels` (decoded images) or
 * `encoded` (native DDS payloads). Encoded bytes are otherwise preserved for
 * asset import paths that can parse metadata before image decoding is available.
 */
struct texture_data
{
    std::string name;
    std::filesystem::path source_path;
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t depth{ 1 };
    texture_dimension dimension{ texture_dimension::texture_2d };
    texture_format format{ texture_format::rgba8_srgb };
    texture_color_space color_space{ texture_color_space::srgb };
    texture_semantic semantic{ texture_semantic::generic_color };
    std::vector<std::byte> pixels;
    std::vector<std::byte> encoded;
    std::vector<texture_mip_data> mips;
    std::string mime_type;
    std::uint32_t array_layers{ 1 };
    std::uint32_t mip_levels{ 1 };
    bool compressed{};
    bool dds{};

    /**
     * @brief Return whether this texture has decoded pixels ready for upload.
     */
    bool has_pixels() const noexcept
    {
        return width != 0 && height != 0 && !pixels.empty();
    }

    /**
     * @brief Return whether this texture carries encoded mip payloads.
     */
    bool has_encoded_mips() const noexcept
    {
        return width != 0 && height != 0 && !encoded.empty() && !mips.empty();
    }
};

/**
 * @brief Texture creation metadata.
 */
struct texture_desc
{
    std::string name;
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t depth{ 1 };
    texture_dimension dimension{ texture_dimension::texture_2d };
    std::uint32_t mip_levels{ 1 };
    texture_format format{ texture_format::rgba8_srgb };
    texture_color_space color_space{ texture_color_space::srgb };
    texture_semantic semantic{ texture_semantic::generic_color };
};

/**
 * @brief Future displacement evaluation policy.
 */
enum class material_displacement_mode : std::uint8_t
{
    none,
    parallax,
    tessellated
};

/**
 * @brief Renderer material description used by scene rendering.
 */
struct material_desc
{
    material_handle handle{};
    std::string name;
    material_domain domain{ material_domain::surface };
    material_shading_model shading_model{ material_shading_model::standard };

    math::vector4f base_color{ 1.0f, 1.0f, 1.0f, 1.0f };
    float metallic{};
    float roughness{ 0.6f };
    float alpha_cutoff{ 0.5f };
    material_alpha_mode alpha_mode{ material_alpha_mode::opaque };
    bool double_sided{};

    texture_handle base_color_texture{};
    texture_handle metallic_roughness_texture{};
    texture_handle normal_texture{};
    texture_handle occlusion_texture{};
    texture_handle emissive_texture{};
    texture_handle clear_coat_texture{};
    texture_handle clear_coat_roughness_texture{};
    texture_handle clear_coat_normal_texture{};
    texture_handle anisotropy_texture{};
    texture_handle subsurface_texture{};
    texture_handle thickness_texture{};
    texture_handle transmission_texture{};

    float normal_scale{ 1.0f };
    float occlusion_strength{ 1.0f };
    math::vector3f emissive_factor{};
    float emissive_strength{ 1.0f };
    float emissive_luminance_nits{};

    float clear_coat_factor{};
    float clear_coat_roughness{};
    float clear_coat_normal_scale{ 1.0f };
    float sheen_factor{};
    math::vector3f sheen_color{};
    float transmission_factor{};
    float index_of_refraction{ 1.5f };
    float thickness_factor{};
    math::vector3f attenuation_color{ 1.0f, 1.0f, 1.0f };
    float attenuation_distance{ 1.0f };
    float subsurface_factor{};
    math::vector3f subsurface_color{ 1.0f, 0.35f, 0.2f };
    math::vector3f subsurface_radius{ 1.0f, 0.35f, 0.2f };
    float anisotropy_factor{};
    float anisotropy_rotation{};
    float parallax_height_scale{};
    material_displacement_mode displacement_mode{ material_displacement_mode::none };

    resource_handle material_graph{};
    std::array<terrain_layer_desc, 4> terrain_layers{};
};

/**
 * @brief Compact shader permutation key for material and debug variants.
 */
struct shader_permutation_key
{
    material_alpha_mode alpha_mode{ material_alpha_mode::opaque };
    std::uint8_t debug_view{};
    bool has_base_color_texture{};
    bool has_metallic_roughness_texture{};
    bool has_normal_texture{};
    bool has_occlusion_texture{};
    bool has_emissive_texture{};
    bool has_clear_coat_texture{};
    bool has_clear_coat_roughness_texture{};
    bool has_clear_coat_normal_texture{};
    bool has_anisotropy_texture{};
    bool has_subsurface_texture{};
    bool has_thickness_texture{};
    bool has_transmission_texture{};
    bool double_sided{};
    bool wireframe{};
    bool clear_coat{};
    bool sheen{};
    bool transmission{};
    bool subsurface{};
    bool anisotropy{};
    bool parallax{};

    friend bool operator==(const shader_permutation_key& lhs, const shader_permutation_key& rhs) noexcept = default;
};

/**
 * @brief Build a shader permutation key from a material and viewport mode.
 */
shader_permutation_key make_shader_permutation_key(
    const material_desc& material,
    std::uint8_t debug_view = 0,
    bool wireframe = false) noexcept;

/**
 * @brief Return a stable hash for a shader permutation key.
 */
std::size_t hash_shader_permutation_key(const shader_permutation_key& key) noexcept;

/** @brief Return the required color space for a material texture semantic. */
constexpr texture_color_space required_color_space(texture_semantic semantic) noexcept
{
    return semantic == texture_semantic::base_color || semantic == texture_semantic::emissive ||
            semantic == texture_semantic::generic_color
        ? texture_color_space::srgb
        : texture_color_space::linear;
}

/** @brief Return whether a texture declaration matches its semantic. */
constexpr bool valid_texture_color_space(texture_semantic semantic, texture_color_space color_space) noexcept
{
    return required_color_space(semantic) == color_space;
}

constexpr bool texture_semantic_accepts(texture_semantic semantic, texture_color_space color_space) noexcept
{
    return valid_texture_color_space(semantic, color_space);
}

float srgb_to_linear(float value) noexcept;
float linear_to_srgb(float value) noexcept;
math::vector3f srgb_to_linear(const math::vector3f& value) noexcept;
math::vector3f linear_to_srgb(const math::vector3f& value) noexcept;

/** @brief CPU reference helpers used by validation and deterministic tests. */
float ggx_distribution(float n_dot_h, float roughness) noexcept;
float smith_ggx_correlated(float n_dot_v, float n_dot_l, float roughness) noexcept;
math::vector3f fresnel_schlick(float cos_theta, const math::vector3f& f0) noexcept;
math::vector3f beer_lambert_attenuation(
    const math::vector3f& attenuation_color,
    float attenuation_distance,
    float thickness) noexcept;

inline math::vector3f beer_lambert(
    const math::vector3f& attenuation_color,
    float attenuation_distance,
    float thickness) noexcept
{
    return beer_lambert_attenuation(attenuation_color, attenuation_distance, thickness);
}

} // namespace arc::render
