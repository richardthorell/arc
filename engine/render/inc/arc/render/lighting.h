#pragma once

#include <arc/render/events.h>
#include <arc/render/handles.h>
#include <arc/render/material.h>
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

/**
 * @brief Authoring unit used by scene lights.
 */
enum class light_intensity_unit : std::uint8_t
{
    unitless,
    lumen,
    candela,
    lux
};

/**
 * @brief Renderer environment source. Prefiltering is a follow-up pass.
 */
struct environment_desc
{
    environment_handle handle{};
    std::string name;
    texture_handle equirectangular_texture{};
    math::vector3f fallback_color{ 0.12f, 0.12f, 0.12f };
    float intensity{ 1.0f };
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
 * @brief Per-frame packed lighting data.
 */
struct scene_lighting_data
{
    std::array<directional_light_data, max_directional_lights> directional_lights{};
    std::array<point_light_data, max_point_lights> point_lights{};
    std::array<spot_light_data, max_spot_lights> spot_lights{};
    math::vector4f ambient_color_intensity{ 0.12f, 0.12f, 0.12f, 1.0f };
    std::uint32_t directional_count{};
    std::uint32_t point_count{};
    std::uint32_t spot_count{};
    std::uint32_t skipped_directional_count{};
    std::uint32_t skipped_point_count{};
    std::uint32_t skipped_spot_count{};
};

/**
 * @brief Convert a color temperature in Kelvin to linear RGB approximation.
 */
math::vector3f color_temperature_rgb(float kelvin) noexcept;

/**
 * @brief Convert authored intensity into the renderer's unitless shading scale.
 */
float light_intensity_scale(light_intensity_unit unit, float intensity, float range = 1.0f) noexcept;

/**
 * @brief Estimate light importance for v1 capping/sorting.
 */
float estimate_light_contribution(const directional_light_event& light) noexcept;
float estimate_light_contribution(const point_light_event& light) noexcept;
float estimate_light_contribution(const spot_light_event& light) noexcept;

/**
 * @brief Pack extracted render lights into capped GPU-ready arrays.
 */
scene_lighting_data pack_scene_lighting(
    const std::vector<directional_light_event>& directional,
    const std::vector<point_light_event>& point,
    const std::vector<spot_light_event>& spot,
    const environment_desc* environment = nullptr);

} // namespace arc::render
