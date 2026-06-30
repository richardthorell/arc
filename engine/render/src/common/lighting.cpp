#include <arc/render/lighting.h>

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

    return {
        srgb_channel_from_temperature(red),
        srgb_channel_from_temperature(green),
        srgb_channel_from_temperature(blue)
    };
}

float light_intensity_scale(light_intensity_unit unit, float intensity, float range) noexcept
{
    const float safe_range = std::max(range, 0.001f);
    switch (unit)
    {
    case light_intensity_unit::unitless:
        return intensity;
    case light_intensity_unit::lumen:
        return intensity / (4.0f * 3.1415926535f * safe_range * safe_range);
    case light_intensity_unit::candela:
        return intensity;
    case light_intensity_unit::lux:
        return intensity * safe_range * safe_range;
    }
    return intensity;
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
    const float cone = std::max(light.outer_angle - light.inner_angle, 0.001f);
    return std::max({ light.color[0], light.color[1], light.color[2] }) *
        light_intensity_scale(light.intensity_unit, std::max(light.intensity, 0.0f), light.range) /
        cone;
}

scene_lighting_data pack_scene_lighting(
    const std::vector<directional_light_event>& directional,
    const std::vector<point_light_event>& point,
    const std::vector<spot_light_event>& spot,
    const environment_desc* environment)
{
    scene_lighting_data data{};
    if (environment)
    {
        data.ambient_color_intensity = {
            environment->fallback_color[0],
            environment->fallback_color[1],
            environment->fallback_color[2],
            environment->intensity
        };
    }

    const auto sorted_directional = sorted_by_contribution(directional);
    const auto sorted_point = sorted_by_contribution(point);
    const auto sorted_spot = sorted_by_contribution(spot);

    data.directional_count = static_cast<std::uint32_t>(std::min<std::size_t>(sorted_directional.size(), max_directional_lights));
    data.point_count = static_cast<std::uint32_t>(std::min<std::size_t>(sorted_point.size(), max_point_lights));
    data.spot_count = static_cast<std::uint32_t>(std::min<std::size_t>(sorted_spot.size(), max_spot_lights));
    data.skipped_directional_count = static_cast<std::uint32_t>(sorted_directional.size() - data.directional_count);
    data.skipped_point_count = static_cast<std::uint32_t>(sorted_point.size() - data.point_count);
    data.skipped_spot_count = static_cast<std::uint32_t>(sorted_spot.size() - data.spot_count);

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
        data.spot_lights[index] = {
            .position_range = { light.position[0], light.position[1], light.position[2], light.range },
            .direction_inner_angle = { light.direction[0], light.direction[1], light.direction[2], light.inner_angle },
            .color_intensity = {
                light.color[0],
                light.color[1],
                light.color[2],
                light_intensity_scale(light.intensity_unit, light.intensity, light.range)
            },
            .params = { light.outer_angle, light.casts_shadows ? 1.0f : 0.0f, 0.0f, 0.0f }
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
