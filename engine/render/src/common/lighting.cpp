#include <arc/render/lighting.h>

#include <arc/math/constants.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <sstream>

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

template <class Light> std::vector<Light> sorted_by_contribution(std::vector<Light> lights)
{
    std::sort(lights.begin(), lights.end(), [](const Light& lhs, const Light& rhs)
              { return estimate_light_contribution(lhs) > estimate_light_contribution(rhs); });
    return lights;
}

} // namespace


bool parse_ies_profile(std::string_view source, photometric_profile& out, std::string& error)
{
    out = {};
    error.clear();

    const auto tilt_position = source.find("TILT=");
    if (tilt_position == std::string_view::npos)
    {
        error = "IES profile is missing TILT= declaration";
        return false;
    }
    const auto tilt_end = source.find_first_of("\r\n", tilt_position);
    const auto tilt = source.substr(tilt_position + 5u, tilt_end == std::string_view::npos
                                                                ? std::string_view::npos
                                                                : tilt_end - (tilt_position + 5u));
    if (tilt != "NONE")
    {
        error = "IES TILT data is not supported yet; use TILT=NONE";
        return false;
    }

    const auto numeric_start = tilt_end == std::string_view::npos ? source.size() : tilt_end + 1u;
    std::istringstream stream(std::string{source.substr(numeric_start)});
    std::uint32_t lamp_count{};
    float lumens_per_lamp{};
    float candela_multiplier{};
    std::uint32_t vertical_count{};
    std::uint32_t horizontal_count{};
    std::uint32_t photometric_type{};
    std::uint32_t units_type{};
    float width{}, length{}, height{}, ballast_factor{}, future_use{}, input_watts{};
    if (!(stream >> lamp_count >> lumens_per_lamp >> candela_multiplier >> vertical_count >> horizontal_count >>
          photometric_type >> units_type >> width >> length >> height >> ballast_factor >> future_use >> input_watts))
    {
        error = "IES photometric header is incomplete";
        return false;
    }
    (void)units_type;
    (void)width;
    (void)length;
    (void)height;
    (void)ballast_factor;
    (void)future_use;
    (void)input_watts;
    if (lamp_count == 0u || vertical_count < 2u || horizontal_count == 0u || photometric_type < 1u ||
        photometric_type > 3u || !std::isfinite(lumens_per_lamp) || !std::isfinite(candela_multiplier) ||
        candela_multiplier <= 0.0f)
    {
        error = "IES photometric header contains invalid counts or scaling";
        return false;
    }

    out.vertical_angles_degrees.resize(vertical_count);
    out.horizontal_angles_degrees.resize(horizontal_count);
    for (auto& angle : out.vertical_angles_degrees)
        if (!(stream >> angle) || !std::isfinite(angle))
        {
            error = "IES vertical angle table is incomplete";
            out = {};
            return false;
        }
    for (auto& angle : out.horizontal_angles_degrees)
        if (!(stream >> angle) || !std::isfinite(angle))
        {
            error = "IES horizontal angle table is incomplete";
            out = {};
            return false;
        }

    const auto strictly_non_decreasing = [](const std::vector<float>& values)
    {
        return std::adjacent_find(values.begin(), values.end(), std::greater<float>{}) == values.end();
    };
    if (!strictly_non_decreasing(out.vertical_angles_degrees) ||
        !strictly_non_decreasing(out.horizontal_angles_degrees))
    {
        error = "IES angle tables must be monotonically increasing";
        out = {};
        return false;
    }

    out.normalized_candela.resize(static_cast<std::size_t>(vertical_count) * horizontal_count);
    float peak{};
    for (auto& value : out.normalized_candela)
    {
        if (!(stream >> value) || !std::isfinite(value) || value < 0.0f)
        {
            error = "IES candela table is incomplete or contains invalid values";
            out = {};
            return false;
        }
        value *= candela_multiplier;
        peak = std::max(peak, value);
    }
    if (!(peak > 0.0f))
    {
        error = "IES candela distribution has no positive intensity";
        out = {};
        return false;
    }

    for (auto& value : out.normalized_candela) value /= peak;
    out.peak_candela = peak;
    out.declared_lumens = static_cast<float>(lamp_count) * std::max(lumens_per_lamp, 0.0f);
    out.photometric_type = photometric_type;
    return true;
}

float sample_photometric_profile(const photometric_profile& profile, float vertical_angle_radians,
                                 float horizontal_angle_radians) noexcept
{
    if (profile.vertical_angles_degrees.empty() || profile.horizontal_angles_degrees.empty() ||
        profile.normalized_candela.size() !=
            profile.vertical_angles_degrees.size() * profile.horizontal_angles_degrees.size())
        return 1.0f;

    const float radians_to_degrees = 180.0f / math::pi<float>;
    float vertical = std::clamp(std::abs(vertical_angle_radians) * radians_to_degrees,
                                profile.vertical_angles_degrees.front(), profile.vertical_angles_degrees.back());
    float horizontal = std::fmod(horizontal_angle_radians * radians_to_degrees, 360.0f);
    if (horizontal < 0.0f) horizontal += 360.0f;

    const float horizontal_max = profile.horizontal_angles_degrees.back();
    if (profile.horizontal_angles_degrees.size() == 1u)
        horizontal = profile.horizontal_angles_degrees.front();
    else if (horizontal_max <= 90.0001f)
    {
        if (horizontal > 180.0f) horizontal = 360.0f - horizontal;
        if (horizontal > 90.0f) horizontal = 180.0f - horizontal;
    }
    else if (horizontal_max <= 180.0001f && horizontal > 180.0f)
        horizontal = 360.0f - horizontal;
    horizontal = std::clamp(horizontal, profile.horizontal_angles_degrees.front(), horizontal_max);

    const auto bracket = [](const std::vector<float>& values, float value)
    {
        const auto upper = std::lower_bound(values.begin(), values.end(), value);
        if (upper == values.begin()) return std::pair<std::size_t, std::size_t>{0u, 0u};
        if (upper == values.end()) return std::pair<std::size_t, std::size_t>{values.size() - 1u, values.size() - 1u};
        const auto high = static_cast<std::size_t>(std::distance(values.begin(), upper));
        return std::pair<std::size_t, std::size_t>{high - 1u, high};
    };
    const auto [v0, v1] = bracket(profile.vertical_angles_degrees, vertical);
    const auto [h0, h1] = bracket(profile.horizontal_angles_degrees, horizontal);
    const auto fraction = [](float value, float low, float high)
    {
        return high > low ? std::clamp((value - low) / (high - low), 0.0f, 1.0f) : 0.0f;
    };
    const float tv = fraction(vertical, profile.vertical_angles_degrees[v0], profile.vertical_angles_degrees[v1]);
    const float th = fraction(horizontal, profile.horizontal_angles_degrees[h0], profile.horizontal_angles_degrees[h1]);
    const auto sample = [&](std::size_t h, std::size_t v)
    {
        return profile.normalized_candela[h * profile.vertical_angles_degrees.size() + v];
    };
    const float low = sample(h0, v0) + (sample(h0, v1) - sample(h0, v0)) * tv;
    const float high = sample(h1, v0) + (sample(h1, v1) - sample(h1, v0)) * tv;
    return std::clamp(low + (high - low) * th, 0.0f, 1.0f);
}

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

    return srgb_to_linear(math::vector3f{srgb_channel_from_temperature(red), srgb_channel_from_temperature(green),
                                         srgb_channel_from_temperature(blue)});
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
    if (range <= 0.0f) return inverse_square;

    const float normalized = std::clamp(distance / range, 0.0f, 1.0f);
    const float cutoff = 1.0f - normalized * normalized * normalized * normalized;
    return inverse_square * cutoff * cutoff;
}

float cone_solid_angle(float half_angle_radians) noexcept
{
    return 2.0f * math::pi<float> * (1.0f - std::cos(std::clamp(half_angle_radians, 0.0f, math::pi<float>)));
}

float exposure_multiplier(float ev100, float compensation_ev) noexcept
{
    if (!std::isfinite(ev100) || !std::isfinite(compensation_ev)) return 1.0f;
    return std::exp2(compensation_ev - ev100) / 1.2f;
}

exposure_state adapt_exposure(exposure_state current, const exposure_settings& settings, float metered_ev100,
                              float delta_seconds, bool camera_cut) noexcept
{
    const float target = std::clamp(settings.mode == exposure_mode::manual ? settings.manual_ev100 : metered_ev100,
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
    return std::max({light.color[0], light.color[1], light.color[2]}) * std::max(light.intensity, 0.0f);
}

float estimate_light_contribution(const point_light_event& light) noexcept
{
    return std::max({light.color[0], light.color[1], light.color[2]}) *
           light_intensity_scale(light.intensity_unit, std::max(light.intensity, 0.0f), light.range);
}

float estimate_light_contribution(const spot_light_event& light) noexcept
{
    const float cone = std::max(cone_solid_angle(light.outer_angle), 0.001f);
    return std::max({light.color[0], light.color[1], light.color[2]}) *
           light_intensity_scale(light.intensity_unit, std::max(light.intensity, 0.0f), light.range) / cone;
}

float estimate_light_contribution(const area_light_event& light) noexcept
{
    const float area = light.shape == area_light_shape::disk ? math::pi<float> * 0.25f * light.width * light.width
                                                             : light.width * light.height;
    return std::max({light.color[0], light.color[1], light.color[2]}) * std::max(light.intensity, 0.0f) *
           std::max(area, 0.001f);
}

scene_lighting_data pack_scene_lighting(const std::vector<directional_light_event>& directional,
                                        const std::vector<point_light_event>& point,
                                        const std::vector<spot_light_event>& spot,
                                        const environment_descriptor* environment, std::uint32_t point_limit,
                                        std::uint32_t spot_limit, const std::vector<area_light_event>& area)
{
    scene_lighting_data data{};
    if (environment)
    {
        const auto ambient = environment->prefiltered ? environment->diffuse_irradiance : environment->fallback_color;
        const float intensity = environment->prefiltered ? environment->diffuse_intensity : environment->intensity;
        data.ambient_color_intensity = {ambient[0], ambient[1], ambient[2], intensity};
    }

    const auto sorted_directional = sorted_by_contribution(directional);
    const auto sorted_point = sorted_by_contribution(point);
    const auto sorted_spot = sorted_by_contribution(spot);
    const auto sorted_area = sorted_by_contribution(area);

    data.directional_count =
        static_cast<std::uint32_t>(std::min<std::size_t>(sorted_directional.size(), max_directional_lights));
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
            .direction_intensity = {light.direction[0], light.direction[1], light.direction[2], light.intensity},
            .color_flags = {light.color[0], light.color[1], light.color[2], light.casts_shadows ? 1.0f : 0.0f},
            .source_shape = {std::max(light.source_angle, 0.0f), 0.0f, 0.0f, 0.0f}};
    }

    for (std::uint32_t index = 0; index < data.point_count; ++index)
    {
        const auto& light = sorted_point[index];
        data.point_lights[index] = {
            .position_range = {light.position[0], light.position[1], light.position[2], light.range},
            .color_intensity = {light.color[0], light.color[1], light.color[2],
                                light_intensity_scale(light.intensity_unit, light.intensity, light.range)},
            .object_id_shadow = {static_cast<float>(light.object_id.index),
                                 static_cast<float>(light.object_id.generation), -1.0f, 0.0f},
            .shadow_parameters = {-1.0f, 0.0f, 0.0f, 0.0f},
            .source_shape = {std::max(light.source_radius, 0.0f), std::max(light.source_length, 0.0f), 0.0f, 0.0f}};
    }

    for (std::uint32_t index = 0; index < data.spot_count; ++index)
    {
        const auto& light = sorted_spot[index];
        const float intensity =
            light.intensity_unit == light_intensity_unit::lumen
                ? std::max(light.intensity, 0.0f) / std::max(cone_solid_angle(light.outer_angle), 0.001f)
                : light_intensity_scale(light.intensity_unit, light.intensity, light.range);
        data.spot_lights[index] = {
            .position_range = {light.position[0], light.position[1], light.position[2], light.range},
            .direction_inner_angle = {light.direction[0], light.direction[1], light.direction[2], light.inner_angle},
            .color_intensity = {light.color[0], light.color[1], light.color[2], intensity},
            .params = {light.outer_angle, light.casts_shadows ? 1.0f : 0.0f, 0.0f, 0.0f},
            .object_id_shadow = {static_cast<float>(light.object_id.index),
                                 static_cast<float>(light.object_id.generation), -1.0f, 0.0f},
            .shadow_parameters = {-1.0f, 0.0f, 0.0f, 0.0f},
            .source_shape = {std::max(light.source_radius, 0.0f), std::max(light.source_length, 0.0f), 0.0f, 0.0f}};
    }

    for (std::uint32_t index = 0; index < data.area_count; ++index)
    {
        const auto& light = sorted_area[index];
        const float width = std::max(light.width, 0.001f);
        const float height = light.shape == area_light_shape::disk ? width : std::max(light.height, 0.001f);
        const float area_size =
            light.shape == area_light_shape::disk ? math::pi<float> * 0.25f * width * width : width * height;
        const float radiance = light.intensity_unit == light_intensity_unit::lumen
                                   ? std::max(light.intensity, 0.0f) /
                                         std::max(math::pi<float> * area_size * (light.two_sided ? 2.0f : 1.0f), 0.001f)
                                   : light_intensity_scale(light.intensity_unit, light.intensity);
        data.area_lights[index] = {.position_shape = {light.position[0], light.position[1], light.position[2],
                                                      static_cast<float>(light.shape)},
                                   .direction_two_sided = {light.direction[0], light.direction[1], light.direction[2],
                                                           light.two_sided ? 1.0f : 0.0f},
                                   .tangent_width = {light.tangent[0], light.tangent[1], light.tangent[2], width},
                                   .color_intensity = {light.color[0], light.color[1], light.color[2], radiance},
                                   .dimensions_shadow = {width, height, light.casts_shadows ? 1.0f : 0.0f, 0.0f}};
    }

    return data;
}

std::array<float, directional_shadow_cascade_count> cascade_splits(float near_plane, float far_plane,
                                                                   float split_lambda) noexcept
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

namespace
{

math::vector4f transform_cluster_point(const math::matrix4f& matrix, const math::vector3f& point) noexcept
{
    return {matrix(0, 0) * point[0] + matrix(0, 1) * point[1] + matrix(0, 2) * point[2] + matrix(0, 3),
            matrix(1, 0) * point[0] + matrix(1, 1) * point[1] + matrix(1, 2) * point[2] + matrix(1, 3),
            matrix(2, 0) * point[0] + matrix(2, 1) * point[1] + matrix(2, 2) * point[2] + matrix(2, 3),
            matrix(3, 0) * point[0] + matrix(3, 1) * point[1] + matrix(3, 2) * point[2] + matrix(3, 3)};
}

std::uint32_t cluster_depth_slice(float distance, float near_plane, float far_plane, std::uint32_t slices) noexcept
{
    if (slices <= 1u) return 0u;
    near_plane = std::max(near_plane, 0.001f);
    far_plane = std::max(far_plane, near_plane + 0.001f);
    distance = std::clamp(distance, near_plane, far_plane);
    const float normalized = std::log(distance / near_plane) / std::log(far_plane / near_plane);
    return std::min(static_cast<std::uint32_t>(normalized * static_cast<float>(slices)), slices - 1u);
}

} // namespace

clustered_light_grid build_clustered_light_grid(const scene_lighting_data& lighting,
                                                const clustered_light_grid_view& view,
                                                clustered_light_grid_config config)
{
    clustered_light_grid result{};
    result.config.tile_size_pixels = std::max(config.tile_size_pixels, 1u);
    result.config.depth_slices = std::max(config.depth_slices, 1u);
    result.config.maximum_lights_per_cluster = std::clamp(config.maximum_lights_per_cluster, 1u, 256u);

    const auto width = std::max(view.viewport_width, 1u);
    const auto height = std::max(view.viewport_height, 1u);
    result.tiles_x = (width + result.config.tile_size_pixels - 1u) / result.config.tile_size_pixels;
    result.tiles_y = (height + result.config.tile_size_pixels - 1u) / result.config.tile_size_pixels;
    result.cluster_count = result.tiles_x * result.tiles_y * result.config.depth_slices;

    const std::uint32_t record_words = 1u + result.config.maximum_lights_per_cluster;
    result.gpu_words.assign(
        clustered_light_header_words + static_cast<std::size_t>(result.cluster_count) * record_words, 0u);
    auto& words = result.gpu_words;
    words[0] = result.config.tile_size_pixels;
    words[1] = result.tiles_x;
    words[2] = result.tiles_y;
    words[3] = result.config.depth_slices;
    words[4] = result.config.maximum_lights_per_cluster;
    words[5] = result.cluster_count;
    words[6] = std::bit_cast<std::uint32_t>(std::max(view.near_plane, 0.001f));
    words[7] = std::bit_cast<std::uint32_t>(std::max(view.far_plane, view.near_plane + 0.001f));

    const auto append_reference = [&](std::uint32_t cluster, std::uint32_t reference, clustered_light_kind kind)
    {
        if (cluster >= result.cluster_count) return;
        const auto base = clustered_light_header_words + static_cast<std::size_t>(cluster) * record_words;
        auto& count = words[base];
        if (count >= result.config.maximum_lights_per_cluster)
        {
            ++result.overflow_count;
            return;
        }
        words[base + 1u + count] = reference;
        ++count;
        switch (kind)
        {
            case clustered_light_kind::point:
                ++result.point_light_references;
                break;
            case clustered_light_kind::spot:
                ++result.spot_light_references;
                break;
            case clustered_light_kind::area:
                ++result.area_light_references;
                break;
        }
    };

    const auto append_sphere =
        [&](const math::vector3f& center, float radius, std::uint32_t reference, clustered_light_kind kind)
    {
        radius = std::max(radius, 0.001f);
        const auto dx = center[0] - view.camera_position[0];
        const auto dy = center[1] - view.camera_position[1];
        const auto dz = center[2] - view.camera_position[2];
        const float camera_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float min_distance = std::max(view.near_plane, camera_distance - radius);
        const float max_distance = std::min(view.far_plane, camera_distance + radius);
        if (max_distance < view.near_plane || min_distance > view.far_plane) return;

        std::uint32_t min_tile_x = 0u;
        std::uint32_t max_tile_x = result.tiles_x - 1u;
        std::uint32_t min_tile_y = 0u;
        std::uint32_t max_tile_y = result.tiles_y - 1u;

        const auto view_position = transform_cluster_point(view.view, center);
        const float view_depth = std::max(-view_position[2], view.near_plane);
        if (camera_distance > radius && view_depth > view.near_plane)
        {
            const auto clip = transform_cluster_point(view.view_projection, center);
            if (clip[3] > 1.0e-5f)
            {
                const float ndc_x = clip[0] / clip[3];
                const float ndc_y = clip[1] / clip[3];
                const float ndc_radius_x = std::abs(view.projection(0, 0)) * radius / view_depth;
                const float ndc_radius_y = std::abs(view.projection(1, 1)) * radius / view_depth;
                const float min_px = (std::clamp(ndc_x - ndc_radius_x, -1.0f, 1.0f) * 0.5f + 0.5f) * width;
                const float max_px = (std::clamp(ndc_x + ndc_radius_x, -1.0f, 1.0f) * 0.5f + 0.5f) * width;
                const float min_py = (0.5f - std::clamp(ndc_y + ndc_radius_y, -1.0f, 1.0f) * 0.5f) * height;
                const float max_py = (0.5f - std::clamp(ndc_y - ndc_radius_y, -1.0f, 1.0f) * 0.5f) * height;
                min_tile_x =
                    std::min(static_cast<std::uint32_t>(std::max(min_px, 0.0f)) / result.config.tile_size_pixels,
                             result.tiles_x - 1u);
                max_tile_x =
                    std::min(static_cast<std::uint32_t>(std::max(max_px, 0.0f)) / result.config.tile_size_pixels,
                             result.tiles_x - 1u);
                min_tile_y =
                    std::min(static_cast<std::uint32_t>(std::max(min_py, 0.0f)) / result.config.tile_size_pixels,
                             result.tiles_y - 1u);
                max_tile_y =
                    std::min(static_cast<std::uint32_t>(std::max(max_py, 0.0f)) / result.config.tile_size_pixels,
                             result.tiles_y - 1u);
            }
        }

        const auto min_slice =
            cluster_depth_slice(min_distance, view.near_plane, view.far_plane, result.config.depth_slices);
        const auto max_slice =
            cluster_depth_slice(max_distance, view.near_plane, view.far_plane, result.config.depth_slices);
        for (std::uint32_t slice = min_slice; slice <= max_slice; ++slice)
            for (std::uint32_t y = min_tile_y; y <= max_tile_y; ++y)
                for (std::uint32_t x = min_tile_x; x <= max_tile_x; ++x)
                {
                    const auto cluster = (slice * result.tiles_y + y) * result.tiles_x + x;
                    append_reference(cluster, reference, kind);
                }
    };

    for (std::uint32_t index = 0u; index < std::min(lighting.point_count, max_point_lights); ++index)
        append_sphere(math::vector3f{lighting.point_lights[index].position_range[0],
                                     lighting.point_lights[index].position_range[1],
                                     lighting.point_lights[index].position_range[2]},
                      lighting.point_lights[index].position_range[3],
                      encode_clustered_light_reference(clustered_light_kind::point, index),
                      clustered_light_kind::point);

    for (std::uint32_t index = 0u; index < std::min(lighting.spot_count, max_spot_lights); ++index)
        append_sphere(math::vector3f{lighting.spot_lights[index].position_range[0],
                                     lighting.spot_lights[index].position_range[1],
                                     lighting.spot_lights[index].position_range[2]},
                      lighting.spot_lights[index].position_range[3],
                      encode_clustered_light_reference(clustered_light_kind::spot, index), clustered_light_kind::spot);

    for (std::uint32_t index = 0u; index < std::min(lighting.area_count, max_area_lights); ++index)
    {
        const auto reference = encode_clustered_light_reference(clustered_light_kind::area, index);
        for (std::uint32_t cluster = 0u; cluster < result.cluster_count; ++cluster)
            append_reference(cluster, reference, clustered_light_kind::area);
    }

    words[8] = result.point_light_references;
    words[9] = result.spot_light_references;
    words[10] = result.area_light_references;
    words[11] = result.overflow_count;
    return result;
}

} // namespace arc::render
