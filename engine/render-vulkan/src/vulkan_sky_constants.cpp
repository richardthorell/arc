#include "vulkan_sky_constants.h"

#include <arc/math/math.h>

#include <algorithm>
#include <bit>
#include <cmath>

namespace arc::render::vulkan::detail
{
namespace
{

constexpr math::vector3f default_sun_direction{ 0.35f, -0.85f, -0.40f };
constexpr float default_vertical_fov_tangent = 0.57735f;
constexpr float minimum_projection_scale = 0.0001f;
constexpr float minimum_sun_angular_radius_degrees = 0.01f;
constexpr float minimum_environment_exposure = 0.001f;
constexpr float maximum_star_density = 0.99f;
constexpr float star_twinkle_amplitude = 0.15f;
constexpr float star_twinkle_frequency = 2.17f;

std::uint32_t quantize_unsigned(float value, float maximum, std::uint32_t mask) noexcept
{
    if (!std::isfinite(value))
        value = 0.0f;
    return static_cast<std::uint32_t>(std::round(
        std::clamp(value / maximum, 0.0f, 1.0f) * static_cast<float>(mask)));
}

float packed_float(std::uint32_t bits) noexcept
{
    return std::bit_cast<float>(bits);
}

void pack_cloud_layer(
    const cloud_layer_data& layer,
    bool clouds_enabled,
    float time_seconds,
    float* destination) noexcept
{
    if (!clouds_enabled || !layer.enabled)
        return;
    destination[0] = std::clamp(layer.coverage, 0.0f, 1.0f);
    destination[1] = std::clamp(layer.density, 0.0f, 1.0f);
    auto wind = layer.wind_direction;
    if (math::length_squared(wind) > minimum_projection_scale)
        wind = math::normalize(wind);
    const float phase = time_seconds * layer.wind_speed * layer.scale;
    destination[2] = wind[0] * phase;
    destination[3] = wind[1] * phase;
}

} // namespace

std::uint32_t pack_sun_settings_bits(float intensity, float angular_radius_degrees) noexcept
{
    return quantize_unsigned(intensity, 32.0f, 0xffffu) |
        (quantize_unsigned(angular_radius_degrees, 5.0f, 0xffffu) << 16u);
}

std::uint32_t pack_moon_settings_bits(
    float phase,
    float intensity,
    float angular_radius_degrees,
    bool enabled) noexcept
{
    return quantize_unsigned(phase, 1.0f, 0x3ffu) |
        (quantize_unsigned(intensity, 8.0f, 0x3ffu) << 10u) |
        (quantize_unsigned(angular_radius_degrees, 5.0f, 0x3ffu) << 20u) |
        (enabled ? 1u << 30u : 0u);
}

std::uint32_t pack_sky_settings_bits(
    sky_source_mode source,
    float star_density,
    float star_intensity) noexcept
{
    return static_cast<std::uint32_t>(source) |
        (quantize_unsigned(star_density, 1.0f, 0x3ffu) << 2u) |
        (quantize_unsigned(star_intensity, 16.0f, 0xfffu) << 12u);
}

sky_push_constants build_sky_push_constants(
    const world_environment_data& environment,
    const render_camera& camera,
    std::uint32_t viewport_width,
    std::uint32_t viewport_height,
    bool include_cirrus,
    math::vector3f sun_direction_override) noexcept
{
    auto sun_direction = math::length_squared(sun_direction_override) >= minimum_projection_scale
        ? sun_direction_override
        : environment.celestial.sun_direction;
    if (math::length_squared(sun_direction) < minimum_projection_scale)
        sun_direction = default_sun_direction;
    sun_direction = math::normalize(sun_direction);

    sky_push_constants constants{};
    for (std::uint32_t channel = 0; channel < 3; ++channel)
    {
        constants.camera_forward_fov[channel] = camera.forward[channel];
        constants.camera_up_aspect[channel] = camera.up[channel];
        constants.sun_direction_intensity[channel] = sun_direction[channel];
        constants.moon_direction_phase[channel] = environment.celestial.moon_direction[channel];
    }
    constants.camera_forward_fov[3] = std::abs(camera.projection(1, 1)) > minimum_projection_scale
        ? 1.0f / std::abs(camera.projection(1, 1))
        : default_vertical_fov_tangent;
    constants.camera_up_aspect[3] = viewport_height == 0
        ? 1.0f
        : static_cast<float>(viewport_width) / static_cast<float>(viewport_height);
    constants.sun_direction_intensity[3] = packed_float(pack_sun_settings_bits(
        environment.atmosphere.sun_disk_intensity * environment.celestial.sun_intensity,
        std::max(minimum_sun_angular_radius_degrees, environment.celestial.sun_angular_radius_degrees)));
    constants.moon_direction_phase[3] = packed_float(pack_moon_settings_bits(
        environment.celestial.moon_phase,
        environment.celestial.moon_intensity,
        environment.celestial.moon_angular_radius_degrees,
        environment.celestial.moon_enabled));

    const auto sky_color = environment.source == sky_source_mode::solid_color
        ? environment.solid_color
        : environment.atmosphere.tint;
    for (std::uint32_t channel = 0; channel < 3; ++channel)
        constants.sky_color_source[channel] = sky_color[channel];
    const float star_density = environment.celestial.stars_enabled
        ? std::clamp(environment.celestial.star_density, 0.0f, maximum_star_density)
        : 0.0f;
    const float star_twinkle = 1.0f + environment.celestial.star_twinkle * star_twinkle_amplitude *
        std::sin(environment.celestial.time_seconds * star_twinkle_frequency);
    constants.sky_color_source[3] = packed_float(pack_sky_settings_bits(
        environment.source,
        star_density,
        environment.celestial.star_intensity * star_twinkle));
    constants.atmosphere[0] = environment.source == sky_source_mode::hdri
        ? math::to_radians(environment.hdri_rotation_degrees)
        : environment.atmosphere.rayleigh_strength;
    constants.atmosphere[1] = environment.atmosphere.mie_strength;
    constants.atmosphere[2] = environment.atmosphere.ozone_strength;
    constants.atmosphere[3] = environment.radiance_intensity *
        std::max(environment.atmosphere.exposure, minimum_environment_exposure);

    pack_cloud_layer(
        environment.clouds.cumulus,
        environment.clouds.enabled,
        environment.celestial.time_seconds,
        constants.cumulus);
    if (include_cirrus)
    {
        pack_cloud_layer(
            environment.clouds.cirrus,
            environment.clouds.enabled,
            environment.celestial.time_seconds,
            constants.cirrus);
    }
    return constants;
}

} // namespace arc::render::vulkan::detail
