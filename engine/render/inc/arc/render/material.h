#pragma once

#include <arc/render/handles.h>
#include <arc/math/vector.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace arc::render
{

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
    rgba32f
};

/**
 * @brief Renderer texture payload.
 *
 * Pixels are expected to be tightly packed RGBA8 when present. Encoded bytes are
 * preserved for asset import paths that can parse material metadata before image
 * decoding is available.
 */
struct texture_data
{
    std::string name;
    std::uint32_t width{};
    std::uint32_t height{};
    texture_format format{ texture_format::rgba8_srgb };
    std::vector<std::byte> pixels;
    std::vector<std::byte> encoded;
    std::string mime_type;

    /**
     * @brief Return whether this texture has decoded pixels ready for upload.
     */
    bool has_pixels() const noexcept
    {
        return width != 0 && height != 0 && !pixels.empty();
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
    texture_format format{ texture_format::rgba8_srgb };
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

    float normal_scale{ 1.0f };
    float occlusion_strength{ 1.0f };
    math::vector3f emissive_factor{};
    float emissive_strength{ 1.0f };

    float clear_coat_factor{};
    float clear_coat_roughness{};
    float sheen_factor{};
    math::vector3f sheen_color{};
    float transmission_factor{};
    float subsurface_factor{};
    float anisotropy_factor{};
    float anisotropy_rotation{};
    float parallax_height_scale{};
    material_displacement_mode displacement_mode{ material_displacement_mode::none };

    resource_handle material_graph{};
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

} // namespace arc::render
