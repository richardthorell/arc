#pragma once

#include <arc/render/material.h>

#include <cstdint>

namespace arc::render
{

/** @brief Versioned contract shared by authored material implementations and ARC render passes. */
inline constexpr std::uint32_t material_abi_version = 1;

/** @brief Per-fragment inputs available to a surface material implementation. */
struct material_inputs
{
    math::vector3f position_ws{};
    math::vector3f normal_ws{0.0f, 0.0f, 1.0f};
    math::vector4f tangent_ws{1.0f, 0.0f, 0.0f, 1.0f};
    math::vector2f uv0{};
    math::vector2f uv1{};
    math::vector4f vertex_color = math::vector4f::one;
    math::vector3f view_ws{0.0f, 0.0f, 1.0f};
};

/** @brief Backend-neutral surface values consumed by ARC lighting and pass code. */
struct material_surface
{
    math::vector3f base_color{0.8f, 0.8f, 0.8f};
    float metallic{};
    float roughness{0.6f};
    math::vector3f normal_ws{0.0f, 0.0f, 1.0f};
    math::vector3f clear_coat_normal_ws{0.0f, 0.0f, 1.0f};
    math::vector3f tangent_ws{1.0f, 0.0f, 0.0f};
    float ambient_occlusion{1.0f};
    math::vector3f emissive_radiance{};
    float opacity{1.0f};
    float alpha_cutoff{0.5f};
    float index_of_refraction{1.5f};
    float clear_coat{};
    float clear_coat_roughness{0.1f};
    float sheen{};
    math::vector3f sheen_color{};
    float sheen_roughness{0.5f};
    float anisotropy{};
    float anisotropy_rotation{};
    float transmission{};
    float thickness{};
    math::vector3f attenuation_color = math::vector3f::one;
    float attenuation_distance{1.0f};
    math::vector3f subsurface_color{1.0f, 0.35f, 0.2f};
    float subsurface{};
};

/**
 * @brief Already-decoded texture samples used by the CPU legacy-reference evaluator.
 *
 * This is migration scaffolding, not the future material resource model. GPU legacy evaluation samples the
 * currently bound textures directly and feeds the same @ref material_surface contract.
 */
struct legacy_material_samples
{
    math::vector4f base_color = math::vector4f::one;
    math::vector4f metallic_roughness = math::vector4f::one;
    math::vector3f normal_ws{0.0f, 0.0f, 1.0f};
    float occlusion{1.0f};
    math::vector3f emissive = math::vector3f::one;
    float clear_coat{1.0f};
    float clear_coat_roughness{1.0f};
    math::vector3f clear_coat_normal_ws{0.0f, 0.0f, 1.0f};
    float anisotropy{1.0f};
    float subsurface{1.0f};
    float thickness{1.0f};
    float transmission{1.0f};
};

/** @brief CPU reference for the compatibility material implementation used by the current renderer. */
[[nodiscard]] material_surface evaluate_legacy_material(const material_descriptor& material,
                                                        const material_inputs& inputs,
                                                        const legacy_material_samples& samples = {}) noexcept;

} // namespace arc::render
