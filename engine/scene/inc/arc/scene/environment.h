#pragma once

#include <arc/scene/components.h>
#include <arc/scene/registry.h>

#include <string>
#include <vector>

namespace arc::scene
{

struct solar_position
{
    float azimuth_degrees{};
    float elevation_degrees{};
    // Direction travelled by sunlight, matching directional_light_component.
    math::vector3f light_direction{ 0.0f, -1.0f, 0.0f };
};

struct environment_validation_result
{
    bool valid{ true };
    std::vector<std::string> errors;
};

enum class world_environment_preset : std::uint8_t
{
    clear_day,
    alpine_late_morning,
    golden_hour,
    overcast,
    night,
    indoor_neutral
};

bool is_valid_gregorian_date(std::int32_t year, std::int32_t month, std::int32_t day) noexcept;

solar_position calculate_solar_position(
    float latitude_degrees,
    float longitude_degrees,
    float north_offset_degrees,
    std::int32_t year,
    std::int32_t month,
    std::int32_t day,
    float local_time_hours,
    float utc_offset_hours) noexcept;

float calculate_moon_phase(
    std::int32_t year,
    std::int32_t month,
    std::int32_t day,
    float local_time_hours,
    float utc_offset_hours) noexcept;

environment_validation_result validate_world_environment(
    const world_environment_component& world,
    const sky_atmosphere_component& atmosphere,
    const celestial_sky_component& celestial,
    const cloud_layers_component& clouds,
    const height_fog_component& fog,
    const environment_lighting_component& lighting);

void apply_world_environment_preset(
    world_environment_preset preset,
    world_environment_component& world,
    sky_atmosphere_component& atmosphere,
    celestial_sky_component& celestial,
    cloud_layers_component& clouds,
    height_fog_component& fog,
    environment_lighting_component& lighting) noexcept;

/**
 * @brief Advance simulated clocks and drive linked geographic sun lights.
 */
void update_world_environments(registry& scene, float delta_seconds);

} // namespace arc::scene
