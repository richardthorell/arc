#include <arc/render/lighting.h>

#include <arc/math/constants.h>

#include <algorithm>
#include <cmath>

namespace arc::render
{
namespace
{

float clamp(float value, float min_value, float max_value) noexcept
{
    return std::max(min_value, std::min(value, max_value));
}

float srgb_channel_from_temperature(float value) noexcept
{
    return clamp(value / 255.0f, 0.0f, 1.0f);
}

template <class Light>
std::vector<Light> sorted_by_contribution(std::vector<Light> lights)
{
    std::sort(lights.begin(), lights.end(), [](const Light& lhs, const Light& rhs) {
        return estimate_light_contribution(lhs) > estimate_light_contribution(rhs);
    });
    return lights;
}

} // namespace

math::vector3f color_temperature_rgb(float kelvin) noexcept
{
    const float temperature = clamp(kelvin, 1000.0f, 40000.0f) / 100.0f;
    float red = 255.0f;
    float green = 255.0f;
    float blue = 255.0f;

    if (temperature <= 66.0f)
    {
        red = 255.0f;
        green = 99.4708025861f * std::log(temperature) - 161.1195681661f;
        blue = temperature <= 19.0f ? 0.0f : 138.5177312231f * std::log(temperature - 10.0f) - 305.0447927307f;
    }
    else
    {
        red = 329.698727446f * std::pow(temperature - 60.0f, -0.1332047592f);
        green = 288.1221695283f * std::pow(temperature - 60.0f, -0.0755148492f);
        blue = 255.0f;
    }

    return srgb_to_linear(math::vector3f{
        srgb_channel_from_temperature(red),
        srgb_channel_from_temperature(green),
        srgb_channel_from_temperature(blue)
    });
}

float light_intensity_scale(light_intensity_unit unit, float intensity, float range) noexcept
{
    (void)range;
    switch (unit)
    {
    case light_intensity_unit::unitless:
        return intensity;
    case light_intensity_unit::lumen:
        return intensity / (4.0f * math::pi<float>);
    case light_intensity_unit::candela:
        return intensity;
    case light_intensity_unit::lux:
        return intensity;
    case light_intensity_unit::nit:
        return intensity;
    }
    return intensity;
}

float inverse_square_attenuation(float distance, float range, float source_radius) noexcept
{
    distance = std::max(distance, std::max(source_radius, 0.001f));
    const float inverse_square = 1.0f / (distance * distance);
    if (range <= 0.0f)
        return inverse_square;

    const float normalized = std::clamp(distance / range, 0.0f, 1.0f);
    const float cutoff = 1.0f - normalized * normalized * normalized * normalized;
    return inverse_square * cutoff * cutoff;
}

float cone_solid_angle(float half_angle_radians) noexcept
{
    return 2.0f * math::pi<float> *
        (1.0f - std::cos(std::clamp(half_angle_radians, 0.0f, math::pi<float>)));
}

float exposure_multiplier(float ev100, float compensation_ev) noexcept
{
    if (!std::isfinite(ev100) || !std::isfinite(compensation_ev))
        return 1.0f;
    return std::exp2(compensation_ev - ev100) / 1.2f;
}

exposure_state adapt_exposure(
    exposure_state current,
    const exposure_settings& settings,
    float metered_ev100,
    float delta_seconds,
    bool camera_cut) noexcept
{
    const float target = std::clamp(
        settings.mode == exposure_mode::manual ? settings.manual_ev100 : metered_ev100,
        std::min(settings.minimum_ev100, settings.maximum_ev100),
        std::max(settings.minimum_ev100, settings.maximum_ev100));

    if (!current.valid || camera_cut || !std::isfinite(current.ev100))
        current.ev100 = target;
    else
    {
        const float speed = target < current.ev100 ? settings.brighten_speed : settings.darken_speed;
        const float alpha = 1.0f - std::exp(-std::max(speed, 0.0f) * std::max(delta_seconds, 0.0f));
        current.ev100 += (target - current.ev100) * std::clamp(alpha, 0.0f, 1.0f);
    }
    current.multiplier = exposure_multiplier(current.ev100, settings.compensation_ev);
    current.valid = true;
    return current;
}

float estimate_light_contribution(const directional_light_event& light) noexcept
{
    return std::max({ light.color[0], light.color[1], light.color[2] }) * std::max(light.intensity, 0.0f);
}

float estimate_light_contribution(const point_light_event& light) noexcept
{
    return std::max({ light.color[0], light.color[1], light.color[2] }) *
        light_intensity_scale(light.intensity_unit, std::max(light.intensity, 0.0f), light.range);
}

float estimate_light_contribution(const spot_light_event& light) noexcept
{
    const float cone = std::max(cone_solid_angle(light.outer_angle), 0.001f);
    return std::max({ light.color[0], light.color[1], light.color[2] }) *
        light_intensity_scale(light.intensity_unit, std::max(light.intensity, 0.0f), light.range) /
        cone;
}

float estimate_light_contribution(const area_light_event& light) noexcept
{
    const float area = light.shape == area_light_shape::disk
        ? math::pi<float> * 0.25f * light.width * light.width
        : light.width * light.height;
    return std::max({ light.color[0], light.color[1], light.color[2] }) *
        std::max(light.intensity, 0.0f) * std::max(area, 0.001f);
}

scene_lighting_data pack_scene_lighting(
    const std::vector<directional_light_event>& directional,
    const std::vector<point_light_event>& point,
    const std::vector<spot_light_event>& spot,
    const environment_desc* environment,
    std::uint32_t point_limit,
    std::uint32_t spot_limit,
    const std::vector<area_light_event>& area)
{
    scene_lighting_data data{};
    if (environment)
    {
        const auto ambient = environment->prefiltered ? environment->diffuse_irradiance : environment->fallback_color;
        const float intensity = environment->prefiltered ? environment->diffuse_intensity : environment->intensity;
        data.ambient_color_intensity = {
            ambient[0],
            ambient[1],
            ambient[2],
            intensity
        };
    }

    const auto sorted_directional = sorted_by_contribution(directional);
    const auto sorted_point = sorted_by_contribution(point);
    const auto sorted_spot = sorted_by_contribution(spot);
    const auto sorted_area = sorted_by_contribution(area);

    data.directional_count = static_cast<std::uint32_t>(std::min<std::size_t>(sorted_directional.size(), max_directional_lights));
    point_limit = std::min(point_limit, max_point_lights);
    spot_limit = std::min(spot_limit, max_spot_lights);
    data.point_count = static_cast<std::uint32_t>(std::min<std::size_t>(sorted_point.size(), point_limit));
    data.spot_count = static_cast<std::uint32_t>(std::min<std::size_t>(sorted_spot.size(), spot_limit));
    data.area_count = static_cast<std::uint32_t>(std::min<std::size_t>(sorted_area.size(), max_area_lights));
    data.skipped_directional_count = static_cast<std::uint32_t>(sorted_directional.size() - data.directional_count);
    data.skipped_point_count = static_cast<std::uint32_t>(sorted_point.size() - data.point_count);
    data.skipped_spot_count = static_cast<std::uint32_t>(sorted_spot.size() - data.spot_count);
    data.skipped_area_count = static_cast<std::uint32_t>(sorted_area.size() - data.area_count);

    for (std::uint32_t index = 0; index < data.directional_count; ++index)
    {
        const auto& light = sorted_directional[index];
        data.directional_lights[index] = {
            .direction_intensity = { light.direction[0], light.direction[1], light.direction[2], light.intensity },
            .color_flags = { light.color[0], light.color[1], light.color[2], light.casts_shadows ? 1.0f : 0.0f }
        };
    }

    for (std::uint32_t index = 0; index < data.point_count; ++index)
    {
        const auto& light = sorted_point[index];
        data.point_lights[index] = {
            .position_range = { light.position[0], light.position[1], light.position[2], light.range },
            .color_intensity = {
                light.color[0],
                light.color[1],
                light.color[2],
                light_intensity_scale(light.intensity_unit, light.intensity, light.range)
            }
        };
    }

    for (std::uint32_t index = 0; index < data.spot_count; ++index)
    {
        const auto& light = sorted_spot[index];
        const float intensity = light.intensity_unit == light_intensity_unit::lumen
            ? std::max(light.intensity, 0.0f) / std::max(cone_solid_angle(light.outer_angle), 0.001f)
            : light_intensity_scale(light.intensity_unit, light.intensity, light.range);
        data.spot_lights[index] = {
            .position_range = { light.position[0], light.position[1], light.position[2], light.range },
            .direction_inner_angle = { light.direction[0], light.direction[1], light.direction[2], light.inner_angle },
            .color_intensity = {
                light.color[0],
                light.color[1],
                light.color[2],
                intensity
            },
            .params = { light.outer_angle, light.casts_shadows ? 1.0f : 0.0f, 0.0f, 0.0f }
        };
    }

    for (std::uint32_t index = 0; index < data.area_count; ++index)
    {
        const auto& light = sorted_area[index];
        const float width = std::max(light.width, 0.001f);
        const float height = light.shape == area_light_shape::disk
            ? width
            : std::max(light.height, 0.001f);
        const float area_size = light.shape == area_light_shape::disk
            ? math::pi<float> * 0.25f * width * width
            : width * height;
        const float radiance = light.intensity_unit == light_intensity_unit::lumen
            ? std::max(light.intensity, 0.0f) /
                std::max(math::pi<float> * area_size * (light.two_sided ? 2.0f : 1.0f), 0.001f)
            : light_intensity_scale(light.intensity_unit, light.intensity);
        data.area_lights[index] = {
            .position_shape = {
                light.position[0], light.position[1], light.position[2],
                static_cast<float>(light.shape) },
            .direction_two_sided = {
                light.direction[0], light.direction[1], light.direction[2],
                light.two_sided ? 1.0f : 0.0f },
            .tangent_width = {
                light.tangent[0], light.tangent[1], light.tangent[2], width },
            .color_intensity = {
                light.color[0], light.color[1], light.color[2], radiance },
            .dimensions_shadow = {
                width, height, light.casts_shadows ? 1.0f : 0.0f, 0.0f }
        };
    }

    return data;
}

std::array<float, directional_shadow_cascade_count> cascade_splits(float near_plane, float far_plane, float split_lambda) noexcept
{
    near_plane = std::max(0.001f, near_plane);
    far_plane = std::max(near_plane + 0.001f, far_plane);
    split_lambda = std::clamp(split_lambda, 0.0f, 1.0f);

    std::array<float, directional_shadow_cascade_count> result{};
    const float range = far_plane - near_plane;
    const float ratio = far_plane / near_plane;
    for (std::uint32_t index = 0; index < directional_shadow_cascade_count; ++index)
    {
        const float p = static_cast<float>(index + 1) / static_cast<float>(directional_shadow_cascade_count);
        const float logarithmic = near_plane * std::pow(ratio, p);
        const float uniform = near_plane + range * p;
        result[index] = split_lambda * logarithmic + (1.0f - split_lambda) * uniform;
    }
    result.back() = far_plane;
    return result;
}

} // namespace arc::render
