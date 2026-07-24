#include <arc/scene/environment.h>

#include <arc/math/math.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <limits>

namespace arc::scene
{
namespace
{

inline constexpr float clear_day_sun_illuminance_lux = 65000.0f;

constexpr float hours_per_day = 24.0f;
constexpr float minutes_per_hour = 60.0f;
constexpr float seconds_per_hour = 3600.0f;
constexpr float seconds_per_day = hours_per_day * seconds_per_hour;
constexpr float degrees_per_solar_minute = 4.0f;

bool leap_year(std::int32_t year) noexcept
{
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

std::int32_t days_in_month(std::int32_t year, std::int32_t month) noexcept
{
    constexpr std::int32_t days[]{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    if (month < 1 || month > 12)
        return 0;
    return days[month - 1] + (month == 2 && leap_year(year) ? 1 : 0);
}

std::int32_t day_of_year(std::int32_t year, std::int32_t month, std::int32_t day) noexcept
{
    std::int32_t result = day;
    for (std::int32_t value = 1; value < month; ++value)
        result += days_in_month(year, value);
    return result;
}

double julian_day(
    std::int32_t year,
    std::int32_t month,
    std::int32_t day,
    float utc_hours) noexcept
{
    if (month <= 2)
    {
        --year;
        month += 12;
    }
    const std::int32_t century = year / 100;
    const std::int32_t correction = 2 - century + century / 4;
    return std::floor(365.25 * (year + 4716)) +
        std::floor(30.6001 * (month + 1)) + day + correction - 1524.5 + utc_hours / 24.0;
}

math::quatf rotation_from_forward(const math::vector3f& direction) noexcept
{
    const math::vector3f from{ 0.0f, 0.0f, -1.0f };
    const auto to = math::normalize(direction);
    const float dot = std::clamp(math::dot(from, to), -1.0f, 1.0f);
    if (dot < -0.9999f)
        return math::quatf{ 0.0f, 1.0f, 0.0f, 0.0f };
    const auto axis = math::cross(from, to);
    return math::normalize(math::quatf{ axis[0], axis[1], axis[2], 1.0f + dot });
}

void advance_date(celestial_sky_component& celestial, std::int32_t days) noexcept
{
    while (days > 0)
    {
        ++celestial.day;
        if (celestial.day > days_in_month(celestial.year, celestial.month))
        {
            celestial.day = 1;
            if (++celestial.month > 12)
            {
                celestial.month = 1;
                ++celestial.year;
            }
        }
        --days;
    }
    while (days < 0)
    {
        --celestial.day;
        if (celestial.day < 1)
        {
            if (--celestial.month < 1)
            {
                celestial.month = 12;
                --celestial.year;
            }
            celestial.day = days_in_month(celestial.year, celestial.month);
        }
        ++days;
    }
}

void use_system_time(celestial_sky_component& celestial)
{
    const auto now = std::chrono::system_clock::now() +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::duration<double, std::ratio<3600>>(celestial.utc_offset_hours));
    const auto value = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    celestial.year = utc.tm_year + 1900;
    celestial.month = utc.tm_mon + 1;
    celestial.day = utc.tm_mday;
    celestial.local_time_hours = static_cast<float>(utc.tm_hour) +
        static_cast<float>(utc.tm_min) / minutes_per_hour +
        static_cast<float>(utc.tm_sec) / seconds_per_hour;
}

bool finite(float value) noexcept { return std::isfinite(value); }

bool finite(const math::vector2f& value) noexcept { return finite(value[0]) && finite(value[1]); }
bool finite(const math::vector3f& value) noexcept { return finite(value[0]) && finite(value[1]) && finite(value[2]); }

void validate_cloud_layer(
    const cloud_layer_settings& layer,
    const char* name,
    environment_validation_result& result)
{
    if (!finite(layer.coverage) || layer.coverage < 0.0f || layer.coverage > 1.0f)
        result.errors.emplace_back(std::string(name) + " coverage must be in [0, 1]");
    if (!finite(layer.density) || layer.density < 0.0f || layer.density > 1.0f)
        result.errors.emplace_back(std::string(name) + " density must be in [0, 1]");
    if (!finite(layer.altitude) || layer.altitude < 0.0f || !finite(layer.thickness) || layer.thickness < 0.0f)
        result.errors.emplace_back(std::string(name) + " altitude and thickness must be non-negative");
    if (!finite(layer.scale) || layer.scale <= 0.0f)
        result.errors.emplace_back(std::string(name) + " scale must be positive");
    if (!finite(layer.detail) || layer.detail < 0.0f || layer.detail > 1.0f ||
        !finite(layer.softness) || layer.softness < 0.0f || layer.softness > 1.0f ||
        !finite(layer.lighting_strength) || layer.lighting_strength < 0.0f ||
        !finite(layer.silver_lining) || layer.silver_lining < 0.0f || layer.silver_lining > 1.0f)
        result.errors.emplace_back(std::string(name) + " detail, softness, lighting, and silver lining are out of range");
    if (!finite(layer.wind_direction) || !finite(layer.wind_speed) || layer.wind_speed < 0.0f)
        result.errors.emplace_back(std::string(name) + " wind must be finite and non-negative");
}

} // namespace

bool is_valid_gregorian_date(std::int32_t year, std::int32_t month, std::int32_t day) noexcept
{
    return year >= 1 && year <= 9999 && month >= 1 && month <= 12 &&
        day >= 1 && day <= days_in_month(year, month);
}

solar_position calculate_solar_position(
    float latitude_degrees,
    float longitude_degrees,
    float north_offset_degrees,
    std::int32_t year,
    std::int32_t month,
    std::int32_t day,
    float local_time_hours,
    float utc_offset_hours) noexcept
{
    latitude_degrees = std::clamp(latitude_degrees, -90.0f, 90.0f);
    longitude_degrees = std::clamp(longitude_degrees, -180.0f, 180.0f);
    const float year_days = leap_year(year) ? 366.0f : 365.0f;
    const float gamma = math::tau<float> / year_days *
        (static_cast<float>(day_of_year(year, month, day) - 1) + (local_time_hours - 12.0f) / hours_per_day);
    const float equation_of_time = 229.18f *
        (0.000075f + 0.001868f * std::cos(gamma) - 0.032077f * std::sin(gamma) -
         0.014615f * std::cos(2.0f * gamma) - 0.040849f * std::sin(2.0f * gamma));
    const float declination =
        0.006918f - 0.399912f * std::cos(gamma) + 0.070257f * std::sin(gamma) -
        0.006758f * std::cos(2.0f * gamma) + 0.000907f * std::sin(2.0f * gamma) -
        0.002697f * std::cos(3.0f * gamma) + 0.00148f * std::sin(3.0f * gamma);
    const float true_solar_minutes = local_time_hours * minutes_per_hour + equation_of_time +
        degrees_per_solar_minute * longitude_degrees - minutes_per_hour * utc_offset_hours;
    float hour_angle_degrees = true_solar_minutes / degrees_per_solar_minute - 180.0f;
    while (hour_angle_degrees < -180.0f)
        hour_angle_degrees += 360.0f;
    while (hour_angle_degrees > 180.0f)
        hour_angle_degrees -= 360.0f;
    const float hour_angle = math::to_radians(hour_angle_degrees);
    const float latitude = math::to_radians(latitude_degrees);
    const float cos_zenith = std::clamp(
        std::sin(latitude) * std::sin(declination) +
        std::cos(latitude) * std::cos(declination) * std::cos(hour_angle),
        -1.0f,
        1.0f);
    const float zenith = std::acos(cos_zenith);
    const float elevation = math::pi<float> * 0.5f - zenith;
    const float azimuth = std::atan2(
        std::sin(hour_angle),
        std::cos(hour_angle) * std::sin(latitude) - std::tan(declination) * std::cos(latitude)) + math::pi<float>;
    const float world_azimuth = azimuth + math::to_radians(north_offset_degrees);
    const float cos_elevation = std::cos(elevation);
    const math::vector3f toward_sun{
        std::sin(world_azimuth) * cos_elevation,
        std::sin(elevation),
        -std::cos(world_azimuth) * cos_elevation
    };
    return {
        .azimuth_degrees = std::fmod(math::to_degrees(azimuth) + 360.0f, 360.0f),
        .elevation_degrees = math::to_degrees(elevation),
        .light_direction = math::normalize(toward_sun * -1.0f)
    };
}

float calculate_moon_phase(
    std::int32_t year,
    std::int32_t month,
    std::int32_t day,
    float local_time_hours,
    float utc_offset_hours) noexcept
{
    constexpr double reference_new_moon = 2451550.25972; // 2000-01-06 18:14 UTC
    constexpr double synodic_month = 29.53058867;
    double cycles = (julian_day(year, month, day, local_time_hours - utc_offset_hours) - reference_new_moon) /
        synodic_month;
    cycles -= std::floor(cycles);
    return static_cast<float>(cycles);
}

environment_validation_result validate_world_environment(
    const world_environment_settings& settings)
{
    const auto& world = settings.world;
    const auto& atmosphere = settings.atmosphere;
    const auto& celestial = settings.celestial;
    const auto& clouds = settings.clouds;
    const auto& fog = settings.fog;
    const auto& lighting = settings.lighting;
    environment_validation_result result;
    if (!finite(world.radiance_intensity) || world.radiance_intensity < 0.0f)
        result.errors.emplace_back("sky radiance intensity must be non-negative");
    if (!finite(world.solid_color) || !finite(world.hdri_rotation_degrees))
        result.errors.emplace_back("sky color and HDRI rotation must be finite");
    if (!finite(atmosphere.planet_radius) || atmosphere.planet_radius <= 0.0f ||
        !finite(atmosphere.atmosphere_radius) || atmosphere.atmosphere_radius <= atmosphere.planet_radius)
        result.errors.emplace_back("atmosphere radius must be greater than the positive planet radius");
    if (!finite(atmosphere.mie_anisotropy) || atmosphere.mie_anisotropy <= -0.99f || atmosphere.mie_anisotropy >= 0.99f)
        result.errors.emplace_back("Mie anisotropy must be in (-0.99, 0.99)");
    if (!finite(atmosphere.rayleigh_strength) || !finite(atmosphere.mie_strength) ||
        !finite(atmosphere.ozone_strength) || atmosphere.rayleigh_strength < 0.0f ||
        atmosphere.mie_strength < 0.0f || atmosphere.ozone_strength < 0.0f)
        result.errors.emplace_back("atmospheric scattering strengths must be non-negative");
    if (!finite(atmosphere.tint) || !finite(atmosphere.ground_albedo) ||
        !finite(atmosphere.rayleigh_scale_height) || atmosphere.rayleigh_scale_height <= 0.0f ||
        !finite(atmosphere.mie_scale_height) || atmosphere.mie_scale_height <= 0.0f ||
        !finite(atmosphere.multi_scattering_factor) || atmosphere.multi_scattering_factor < 0.0f ||
        !finite(atmosphere.exposure) || atmosphere.exposure < 0.0f ||
        !finite(atmosphere.sun_disk_size) || atmosphere.sun_disk_size < 0.0f ||
        !finite(atmosphere.sun_disk_intensity) || atmosphere.sun_disk_intensity < 0.0f)
        result.errors.emplace_back("atmosphere artist and scale parameters must be finite and non-negative");
    if (!is_valid_gregorian_date(celestial.year, celestial.month, celestial.day))
        result.errors.emplace_back("celestial date is not a valid Gregorian date");
    if (!finite(celestial.latitude_degrees) || celestial.latitude_degrees < -90.0f || celestial.latitude_degrees > 90.0f)
        result.errors.emplace_back("latitude must be in [-90, 90]");
    if (!finite(celestial.longitude_degrees) || celestial.longitude_degrees < -180.0f || celestial.longitude_degrees > 180.0f)
        result.errors.emplace_back("longitude must be in [-180, 180]");
    if (!finite(celestial.utc_offset_hours) || celestial.utc_offset_hours < -14.0f || celestial.utc_offset_hours > 14.0f)
        result.errors.emplace_back("UTC offset must be in [-14, 14]");
    if (!finite(celestial.local_time_hours) || celestial.local_time_hours < 0.0f || celestial.local_time_hours >= 24.0f)
        result.errors.emplace_back("local time must be in [0, 24)");
    if (!finite(celestial.north_offset_degrees) || !finite(celestial.time_scale) || celestial.time_scale < 0.0f ||
        !finite(celestial.sun_intensity_multiplier) || celestial.sun_intensity_multiplier < 0.0f ||
        !finite(celestial.sun_temperature_multiplier) || celestial.sun_temperature_multiplier <= 0.0f ||
        !finite(celestial.moon_phase) || celestial.moon_phase < 0.0f || celestial.moon_phase > 1.0f ||
        !finite(celestial.moon_intensity) || celestial.moon_intensity < 0.0f ||
        !finite(celestial.moon_angular_radius_degrees) || celestial.moon_angular_radius_degrees <= 0.0f ||
        !finite(celestial.star_density) || celestial.star_density < 0.0f || celestial.star_density > 1.0f ||
        !finite(celestial.star_intensity) || celestial.star_intensity < 0.0f ||
        !finite(celestial.star_twinkle) || celestial.star_twinkle < 0.0f || celestial.star_twinkle > 1.0f)
        result.errors.emplace_back("celestial artist and simulation parameters are out of range");
    validate_cloud_layer(clouds.cumulus, "cumulus", result);
    validate_cloud_layer(clouds.cirrus, "cirrus", result);
    if (!finite(fog.color) || !finite(fog.density) || fog.density < 0.0f ||
        !finite(fog.height_falloff) || fog.height_falloff < 0.0f ||
        !finite(fog.start_distance) || fog.start_distance < 0.0f ||
        !finite(fog.max_opacity) || fog.max_opacity < 0.0f || fog.max_opacity > 1.0f ||
        !finite(fog.sun_scattering_strength) || fog.sun_scattering_strength < 0.0f)
        result.errors.emplace_back("fog parameters must be finite and within their authored ranges");
    if (!finite(lighting.constant_color) || !finite(lighting.diffuse_intensity) ||
        !finite(lighting.specular_intensity) || lighting.diffuse_intensity < 0.0f || lighting.specular_intensity < 0.0f)
        result.errors.emplace_back("environment lighting intensities must be non-negative");
    result.valid = result.errors.empty();
    return result;
}

std::optional<world_environment_settings> read_world_environment_settings(
    const registry& scene,
    entity environment)
{
    const auto* world = scene.try_get<world_environment_component>(environment);
    const auto* atmosphere = scene.try_get<sky_atmosphere_component>(environment);
    const auto* celestial = scene.try_get<celestial_sky_component>(environment);
    const auto* clouds = scene.try_get<cloud_layers_component>(environment);
    const auto* fog = scene.try_get<height_fog_component>(environment);
    const auto* lighting = scene.try_get<environment_lighting_component>(environment);
    if (!world || !atmosphere || !celestial || !clouds || !fog || !lighting)
        return std::nullopt;
    return world_environment_settings{ *world, *atmosphere, *celestial, *clouds, *fog, *lighting };
}

bool set_world_environment_settings(
    registry& scene,
    entity environment,
    const world_environment_settings& settings)
{
    if (!scene.alive(environment) || !validate_world_environment(settings).valid)
        return false;

    scene.emplace<world_environment_component>(environment, settings.world);
    scene.emplace<sky_atmosphere_component>(environment, settings.atmosphere);
    scene.emplace<celestial_sky_component>(environment, settings.celestial);
    scene.emplace<cloud_layers_component>(environment, settings.clouds);
    scene.emplace<height_fog_component>(environment, settings.fog);
    scene.emplace<environment_lighting_component>(environment, settings.lighting);
    return true;
}

void apply_world_environment_preset(
    world_environment_preset preset,
    world_environment_settings& settings) noexcept
{
    const auto hdri_texture = settings.world.hdri_texture;
    const auto linked_sun = settings.celestial.sun_light;
    const auto animation_time_seconds = settings.celestial.animation_time_seconds;
    const auto lighting_environment = settings.lighting.environment;
    const auto lighting_hdri_texture = settings.lighting.hdri_texture;
    settings = {};
    auto& world = settings.world;
    auto& atmosphere = settings.atmosphere;
    auto& celestial = settings.celestial;
    auto& clouds = settings.clouds;
    auto& fog = settings.fog;
    auto& lighting = settings.lighting;
    celestial.sun_mode = sun_position_mode::geographic;
    switch (preset)
    {
    case world_environment_preset::clear_day:
        clouds.cumulus.coverage = 0.08f;
        clouds.cirrus.enabled = false;
        fog.density = 0.004f;
        celestial.local_time_hours = 12.5f;
        break;
    case world_environment_preset::alpine_late_morning:
        atmosphere.rayleigh_strength = 1.12f;
        atmosphere.mie_strength = 0.18f;
        atmosphere.ozone_strength = 0.24f;
        atmosphere.tint = { 0.50f, 0.70f, 1.0f };
        celestial.local_time_hours = 10.5f;
        clouds.cumulus.coverage = 0.24f;
        clouds.cumulus.density = 0.52f;
        clouds.cirrus.coverage = 0.10f;
        fog.color = { 0.67f, 0.76f, 0.84f };
        fog.density = 0.005f;
        fog.height_falloff = 0.05f;
        fog.start_distance = 42.0f;
        fog.max_opacity = 0.34f;
        break;
    case world_environment_preset::golden_hour:
        celestial.local_time_hours = 19.35f;
        atmosphere.mie_strength = 0.42f;
        atmosphere.ozone_strength = 0.32f;
        clouds.cumulus.coverage = 0.18f;
        fog.color = { 0.72f, 0.48f, 0.31f };
        fog.density = 0.012f;
        break;
    case world_environment_preset::overcast:
        clouds.cumulus.coverage = 0.82f;
        clouds.cumulus.density = 0.78f;
        clouds.cirrus.coverage = 0.36f;
        celestial.sun_intensity_multiplier = 0.45f;
        atmosphere.mie_strength = 0.5f;
        lighting.diffuse_intensity = 1.15f;
        fog.density = 0.014f;
        break;
    case world_environment_preset::night:
        celestial.local_time_hours = 23.0f;
        celestial.star_intensity = 1.2f;
        clouds.cumulus.coverage = 0.12f;
        fog.density = 0.004f;
        break;
    case world_environment_preset::indoor_neutral:
        world.sky_visible = false;
        world.source = sky_source::solid_color;
        world.solid_color = { 0.025f, 0.025f, 0.025f };
        clouds.enabled = false;
        fog.enabled = false;
        lighting.source = environment_lighting_source::constant_color;
        lighting.constant_color = { 0.18f, 0.18f, 0.18f };
        break;
    }
    world.hdri_texture = hdri_texture;
    celestial.sun_light = linked_sun;
    celestial.animation_time_seconds = animation_time_seconds;
    lighting.environment = lighting_environment;
    lighting.hdri_texture = lighting_hdri_texture;
}

void update_world_environments(registry& scene, float delta_seconds)
{
    scene.view<world_environment_component, celestial_sky_component>().each(
        [&](entity value, const world_environment_component&, const celestial_sky_component&) {
            auto* world_ptr = scene.try_get<world_environment_component>(value);
            auto* celestial_ptr = scene.try_get<celestial_sky_component>(value);
            if (!world_ptr || !celestial_ptr)
                return;
            auto& world = *world_ptr;
            auto& celestial = *celestial_ptr;
            if (!world.enabled || !celestial.enabled)
                return;
            if (std::isfinite(delta_seconds) && delta_seconds > 0.0f)
                celestial.animation_time_seconds = std::fmod(
                    celestial.animation_time_seconds + delta_seconds,
                    seconds_per_day);
            if (celestial.time_mode == celestial_time_mode::system_clock)
                use_system_time(celestial);
            else if (celestial.time_mode == celestial_time_mode::simulated && celestial.playing && std::isfinite(delta_seconds))
            {
                celestial.local_time_hours += delta_seconds * celestial.time_scale / seconds_per_hour;
                const auto whole_days = static_cast<std::int32_t>(std::floor(celestial.local_time_hours / hours_per_day));
                celestial.local_time_hours -= static_cast<float>(whole_days) * hours_per_day;
                if (celestial.local_time_hours < 0.0f)
                    celestial.local_time_hours += hours_per_day;
                if (celestial.loop_day)
                    celestial.local_time_hours = std::fmod(celestial.local_time_hours + hours_per_day, hours_per_day);
                else
                    advance_date(celestial, whole_days);
            }
            if (celestial.sun_mode != sun_position_mode::geographic || !scene.alive(celestial.sun_light))
                return;
            const auto solar = calculate_solar_position(
                celestial.latitude_degrees,
                celestial.longitude_degrees,
                celestial.north_offset_degrees,
                celestial.year,
                celestial.month,
                celestial.day,
                celestial.local_time_hours,
                celestial.utc_offset_hours);
            if (auto* transform = scene.try_get<transform_component>(celestial.sun_light))
                transform->set_rotation(rotation_from_forward(solar.light_direction));
            if (celestial.automatic_sun_light)
            {
                if (auto* light = scene.try_get<directional_light_component>(celestial.sun_light))
                {
                    const float daylight = std::clamp((solar.elevation_degrees + 6.0f) / 18.0f, 0.0f, 1.0f);
                    const float warm = 1.0f - std::clamp((solar.elevation_degrees + 2.0f) / 25.0f, 0.0f, 1.0f);
                    light->color = { 1.0f, 1.0f, 1.0f };
                    light->use_color_temperature = true;
                    light->temperature_kelvin = std::clamp(
                        (6500.0f - warm * 4000.0f) * celestial.sun_temperature_multiplier,
                        1000.0f,
                        40000.0f);
                    light->intensity =
                        clear_day_sun_illuminance_lux * daylight * celestial.sun_intensity_multiplier;
                    light->intensity_unit = render::light_intensity_unit::lux;
                    light->enabled = daylight > 0.001f;
                }
            }
        });
}

} // namespace arc::scene
