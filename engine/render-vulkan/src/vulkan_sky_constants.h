#pragma once

#include <arc/render/render_world.h>

#include <cstddef>
#include <cstdint>

namespace arc::render::vulkan::detail
{

inline constexpr std::size_t vulkan_minimum_push_constant_bytes = 128;

struct sky_push_constants
{
    float camera_forward_fov[4]{ 0.0f, 0.0f, -1.0f, 0.57735f };
    float camera_up_aspect[4]{ 0.0f, 1.0f, 0.0f, 1.77778f };
    float sun_direction_intensity[4]{ 0.35f, -0.85f, -0.40f, 1.4f };
    float moon_direction_phase[4]{ -0.35f, 0.85f, 0.40f, 0.65f };
    float sky_color_source[4]{ 0.56f, 0.72f, 1.0f, 0.042f };
    float atmosphere[4]{ 1.0f, 0.35f, 0.15f, 1.0f };
    float cumulus[4]{};
    float cirrus[4]{};
};

static_assert(sizeof(sky_push_constants) == vulkan_minimum_push_constant_bytes);

std::uint32_t pack_sun_settings_bits(float intensity, float angular_radius_degrees) noexcept;
std::uint32_t pack_moon_settings_bits(
    float phase,
    float intensity,
    float angular_radius_degrees,
    bool enabled) noexcept;
std::uint32_t pack_sky_settings_bits(
    sky_source_mode source,
    float star_density,
    float star_intensity) noexcept;

sky_push_constants build_sky_push_constants(
    const world_environment_data& environment,
    const render_camera& camera,
    std::uint32_t viewport_width,
    std::uint32_t viewport_height,
    bool include_cirrus,
    math::vector3f sun_direction_override = {}) noexcept;

} // namespace arc::render::vulkan::detail
