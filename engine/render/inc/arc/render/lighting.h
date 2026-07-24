#pragma once

#include <arc/render/exposure.h>
#include <arc/render/events.h>
#include <arc/render/handles.h>
#include <arc/render/material.h>
#include <arc/math/constants.h>
#include <arc/math/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace arc::render
{

inline constexpr std::uint32_t max_directional_lights = 4;
inline constexpr std::uint32_t max_point_lights = 64;
inline constexpr std::uint32_t max_spot_lights = 64;
inline constexpr std::uint32_t max_area_lights = 32;
inline constexpr std::uint32_t directional_shadow_cascade_count = 4;

/**
 * @brief Authoring unit used by scene lights.
 */
enum class light_intensity_unit : std::uint8_t
{
    unitless,
    lumen,
    candela,
    lux,
    nit
};

enum class area_light_shape : std::uint8_t
{
    rectangle,
    disk
};

enum class environment_generation_state : std::uint8_t
{
    missing,
    queued,
    generating,
    ready,
    failed,
    fallback
};

/**
 * @brief Renderer environment source and v1 prefilter outputs.
 */
struct environment_desc
{
    environment_handle handle{};
    std::string name;
    texture_handle equirectangular_texture{};
    math::vector3f fallback_color{ 0.12f, 0.12f, 0.12f };
    float intensity{ 1.0f };
    math::vector3f diffuse_irradiance{ 0.12f, 0.12f, 0.12f };
    float diffuse_intensity{ 1.0f };
    texture_handle irradiance_texture{};
    texture_handle prefiltered_specular_texture{};
    texture_handle brdf_integration_lut{};
    std::uint32_t radiance_resolution{};
    std::uint32_t irradiance_resolution{};
    std::uint32_t prefiltered_specular_resolution{};
    std::uint32_t brdf_lut_resolution{};
    std::uint32_t prefiltered_mip_count{};
    std::string cache_key;
    std::string fallback_reason;
    environment_generation_state generation_state{ environment_generation_state::missing };
    bool prefiltered{};
};

/**
 * @brief Packed directional light data suitable for GPU upload.
 */
struct directional_light_data
{
    math::vector4f direction_intensity{ 0.0f, -1.0f, 0.0f, 0.0f };
    math::vector4f color_flags{ 1.0f, 1.0f, 1.0f, 0.0f };
};

/**
 * @brief Packed point light data suitable for GPU upload.
 */
struct point_light_data
{
    math::vector4f position_range{ 0.0f, 0.0f, 0.0f, 1.0f };
    math::vector4f color_intensity{ 1.0f, 1.0f, 1.0f, 0.0f };
};

/**
 * @brief Packed spot light data suitable for GPU upload.
 */
struct spot_light_data
{
    math::vector4f position_range{ 0.0f, 0.0f, 0.0f, 1.0f };
    math::vector4f direction_inner_angle{ 0.0f, -1.0f, 0.0f, 0.35f };
    math::vector4f color_intensity{ 1.0f, 1.0f, 1.0f, 0.0f };
    math::vector4f params{ 0.75f, 0.0f, 0.0f, 0.0f };
};

/**
 * @brief Packed rectangle/disk light data.
 */
struct area_light_data
{
    math::vector4f position_shape{};
    math::vector4f direction_two_sided{ 0.0f, -1.0f, 0.0f, 0.0f };
    math::vector4f tangent_width{ 1.0f, 0.0f, 0.0f, 1.0f };
    math::vector4f color_intensity{ 1.0f, 1.0f, 1.0f, 0.0f };
    math::vector4f dimensions_shadow{ 1.0f, 1.0f, 0.0f, 0.0f };
};

/**
 * @brief Per-frame packed lighting data.
 */
struct scene_lighting_data
{
    std::array<directional_light_data, max_directional_lights> directional_lights{};
    std::array<point_light_data, max_point_lights> point_lights{};
    std::array<spot_light_data, max_spot_lights> spot_lights{};
    std::array<area_light_data, max_area_lights> area_lights{};
    math::vector4f ambient_color_intensity{ 0.12f, 0.12f, 0.12f, 1.0f };
    std::uint32_t directional_count{};
    std::uint32_t point_count{};
    std::uint32_t spot_count{};
    std::uint32_t area_count{};
    std::uint32_t skipped_directional_count{};
    std::uint32_t skipped_point_count{};
    std::uint32_t skipped_spot_count{};
    std::uint32_t skipped_area_count{};
};

/**
 * @brief One directional shadow cascade in packed renderer form.
 */
struct directional_shadow_cascade_data
{
    math::matrix4f light_view_projection{ math::identity<float, 4>() };
    float split_depth{};
};

/**
 * @brief Directional CSM settings/data prepared for a frame.
 */
struct directional_shadow_data
{
    std::array<directional_shadow_cascade_data, directional_shadow_cascade_count> cascades{};
    shadow_settings settings{};
    bool enabled{};
};

/**
 * @brief Cache key for future directional shadow reuse.
 */
struct directional_shadow_cache_key
{
    std::uint32_t light_index{};
    std::uint32_t resolution{};
    shadow_filter filter{ shadow_filter::pcf_3x3 };

    friend constexpr bool operator==(const directional_shadow_cache_key&, const directional_shadow_cache_key&) noexcept = default;
};

/**
 * @brief Cache key for future point-light cubemap shadows.
 */
struct point_shadow_cache_key
{
    std::uint32_t light_index{};
    std::uint32_t resolution{};

    friend constexpr bool operator==(const point_shadow_cache_key&, const point_shadow_cache_key&) noexcept = default;
};

/**
 * @brief Cache key for future spotlight shadows.
 */
struct spot_shadow_cache_key
{
    std::uint32_t light_index{};
    std::uint32_t resolution{};

    friend constexpr bool operator==(const spot_shadow_cache_key&, const spot_shadow_cache_key&) noexcept = default;
};

/**
 * @brief Return deterministic cascade split depths.
 */
std::array<float, directional_shadow_cascade_count> cascade_splits(float near_plane, float far_plane, float split_lambda = 0.65f) noexcept;

/**
 * @brief Convert a color temperature in Kelvin to linear RGB approximation.
 */
math::vector3f color_temperature_rgb(float kelvin) noexcept;

/**
 * @brief Convert authored intensity into the renderer's unitless shading scale.
 */
float light_intensity_scale(light_intensity_unit unit, float intensity, float range = 1.0f) noexcept;

/** @brief Physically based inverse-square attenuation with a smooth authored cutoff. */
float inverse_square_attenuation(float distance, float range, float source_radius = 0.01f) noexcept;

/** @brief Solid angle of a cone with the supplied half angle in radians. */
float cone_solid_angle(float half_angle_radians) noexcept;

/** @brief Convert EV100 into the multiplier applied to scene-linear radiance. */
float exposure_multiplier(float ev100, float compensation_ev = 0.0f) noexcept;

/** @brief Advance one exposure value toward a metered target without overshoot. */
exposure_state adapt_exposure(
    exposure_state current,
    const exposure_settings& settings,
    float metered_ev100,
    float delta_seconds,
    bool camera_cut = false) noexcept;

/**
 * @brief Estimate light importance for v1 capping/sorting.
 */
float estimate_light_contribution(const directional_light_event& light) noexcept;
float estimate_light_contribution(const point_light_event& light) noexcept;
float estimate_light_contribution(const spot_light_event& light) noexcept;
float estimate_light_contribution(const area_light_event& light) noexcept;

/**
 * @brief Pack extracted render lights into capped GPU-ready arrays.
 */
scene_lighting_data pack_scene_lighting(
    const std::vector<directional_light_event>& directional,
    const std::vector<point_light_event>& point,
    const std::vector<spot_light_event>& spot,
    const environment_desc* environment = nullptr,
    std::uint32_t point_limit = max_point_lights,
    std::uint32_t spot_limit = max_spot_lights,
    const std::vector<area_light_event>& area = {});

} // namespace arc::render
