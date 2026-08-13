#include <arc/editor/host_protocol.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <nlohmann/json.hpp>

namespace arc::editor
{
namespace
{

std::string escape_json(std::string_view value)
{
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const char ch : value)
    {
        switch (ch)
        {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

std::string quote(std::string_view value)
{
    return "\"" + escape_json(value) + "\"";
}

std::string bool_json(bool value)
{
    return value ? "true" : "false";
}

std::string path_array_json(const std::vector<std::filesystem::path>& values)
{
    std::string result{"["};
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) result.push_back(',');
        result += quote(values[index].generic_string());
    }
    result.push_back(']');
    return result;
}

template <class Enum>
bool enum_from_string(std::string_view text, const std::pair<std::string_view, Enum>* values, std::size_t count,
                      Enum& out)
{
    for (std::size_t index = 0; index < count; ++index)
    {
        if (values[index].first == text)
        {
            out = values[index].second;
            return true;
        }
    }
    return false;
}

std::size_t find_key(std::string_view json, std::string_view key)
{
    const std::string needle = "\"" + std::string(key) + "\"";
    const auto key_pos = json.find(needle);
    if (key_pos == std::string_view::npos) return std::string_view::npos;
    const auto colon = json.find(':', key_pos + needle.size());
    if (colon == std::string_view::npos) return std::string_view::npos;
    return colon + 1;
}

std::size_t skip_ws(std::string_view json, std::size_t pos)
{
    while (pos < json.size())
    {
        const char ch = json[pos];
        if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') break;
        ++pos;
    }
    return pos;
}

bool string_value(std::string_view json, std::string_view key, std::string& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '"') return false;
    ++pos;

    std::string value;
    while (pos < json.size())
    {
        const char ch = json[pos++];
        if (ch == '"')
        {
            out = std::move(value);
            return true;
        }
        if (ch == '\\' && pos < json.size())
        {
            const char escaped = json[pos++];
            switch (escaped)
            {
                case 'n':
                    value.push_back('\n');
                    break;
                case 'r':
                    value.push_back('\r');
                    break;
                case 't':
                    value.push_back('\t');
                    break;
                default:
                    value.push_back(escaped);
                    break;
            }
        }
        else
        {
            value.push_back(ch);
        }
    }
    return false;
}

bool string_array_value(std::string_view json, std::string_view key, std::vector<std::filesystem::path>& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '[') return false;
    ++pos;
    std::vector<std::filesystem::path> values;
    while (true)
    {
        pos = skip_ws(json, pos);
        if (pos >= json.size()) return false;
        if (json[pos] == ']')
        {
            out = std::move(values);
            return true;
        }
        if (json[pos] != '"') return false;
        ++pos;
        std::string value;
        while (pos < json.size() && json[pos] != '"')
        {
            if (json[pos] == '\\' && pos + 1 < json.size()) ++pos;
            value.push_back(json[pos++]);
        }
        if (pos >= json.size()) return false;
        ++pos;
        values.emplace_back(std::move(value));
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ',') ++pos;
    }
}

template <class Number> bool number_value(std::string_view json, std::string_view key, Number& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size()) return false;

    const auto start = pos;
    while (pos < json.size())
    {
        const char ch = json[pos];
        if ((ch < '0' || ch > '9') && ch != '-' && ch != '+' && ch != '.' && ch != 'e' && ch != 'E') break;
        ++pos;
    }
    const auto token = json.substr(start, pos - start);
    if constexpr (std::is_floating_point_v<Number>)
    {
        std::istringstream stream{std::string(token)};
        stream >> out;
        return !stream.fail();
    }
    else
    {
        return std::from_chars(token.data(), token.data() + token.size(), out).ec == std::errc{};
    }
}

bool bool_value(std::string_view json, std::string_view key, bool& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size()) return false;
    if (json.substr(pos, 4) == "true")
    {
        out = true;
        return true;
    }
    if (json.substr(pos, 5) == "false")
    {
        out = false;
        return true;
    }
    return false;
}

bool object_value(std::string_view json, std::string_view key, std::string_view& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '{') return false;

    std::size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    const auto start = pos;
    for (; pos < json.size(); ++pos)
    {
        const char ch = json[pos];
        if (in_string)
        {
            if (escaped)
                escaped = false;
            else if (ch == '\\')
                escaped = true;
            else if (ch == '"')
                in_string = false;
            continue;
        }
        if (ch == '"')
        {
            in_string = true;
        }
        else if (ch == '{')
        {
            ++depth;
        }
        else if (ch == '}')
        {
            --depth;
            if (depth == 0)
            {
                out = json.substr(start, pos - start + 1);
                return true;
            }
        }
    }
    return false;
}

bool array3_value(std::string_view json, std::string_view key, host_vec3& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '[') return false;
    ++pos;

    std::array<float, 3> values{};
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        pos = skip_ws(json, pos);
        const auto start = pos;
        while (pos < json.size() && json[pos] != ',' && json[pos] != ']')
            ++pos;
        std::istringstream stream{std::string(json.substr(start, pos - start))};
        stream >> values[index];
        if (stream.fail()) return false;
        pos = skip_ws(json, pos);
        if (index + 1 < values.size())
        {
            if (pos >= json.size() || json[pos] != ',') return false;
            ++pos;
        }
    }
    out = {values[0], values[1], values[2]};
    return true;
}

bool quat_value(std::string_view json, std::string_view key, host_quat& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '[') return false;
    ++pos;

    std::array<float, 4> values{};
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        pos = skip_ws(json, pos);
        const auto start = pos;
        while (pos < json.size() && json[pos] != ',' && json[pos] != ']')
            ++pos;
        std::istringstream stream{std::string(json.substr(start, pos - start))};
        stream >> values[index];
        if (stream.fail()) return false;
        pos = skip_ws(json, pos);
        if (index + 1 < values.size())
        {
            if (pos >= json.size() || json[pos] != ',') return false;
            ++pos;
        }
    }
    out = {values[0], values[1], values[2], values[3]};
    return true;
}

bool array4_value(std::string_view json, std::string_view key, host_vec4& out)
{
    host_quat value;
    if (!quat_value(json, key, value)) return false;
    out = {value.x, value.y, value.z, value.w};
    return true;
}

bool entity_value(std::string_view json, host_entity_id& out)
{
    return number_value(json, "index", out.index) && number_value(json, "generation", out.generation);
}

bool entity_field_value(std::string_view json, std::string_view key, host_entity_id& out)
{
    std::string_view object;
    return object_value(json, key, object) && entity_value(object, out);
}

bool transform_value(std::string_view json, std::string_view key, host_transform& out)
{
    std::string_view object;
    if (!object_value(json, key, object)) return false;
    return array3_value(object, "position", out.position) && quat_value(object, "rotation", out.rotation) &&
           array3_value(object, "scale", out.scale);
}

template <class Enum>
bool parse_enum(std::string_view payload, std::string_view key, const std::pair<std::string_view, Enum>* values,
                std::size_t count, Enum& out);

bool camera_value(std::string_view json, std::string_view key, host_camera_snapshot& out)
{
    std::string_view object;
    if (!object_value(json, key, object)) return false;
    std::string projection;
    if (!string_value(object, "projection", projection) ||
        (projection != "perspective" && projection != "orthographic"))
        return false;
    out.projection =
        projection == "orthographic" ? host_camera_projection::orthographic : host_camera_projection::perspective;
    std::string exposure_mode;
    if (string_value(object, "exposureMode", exposure_mode))
    {
        if (exposure_mode != "manual" && exposure_mode != "automatic") return false;
        out.exposure_mode = exposure_mode == "manual" ? host_exposure_mode::manual : host_exposure_mode::automatic;
    }
    std::string metering;
    if (string_value(object, "exposureMetering", metering))
    {
        if (metering != "average" && metering != "centerWeighted") return false;
        out.exposure_metering = metering == "centerWeighted" ? host_exposure_metering_mode::center_weighted
                                                             : host_exposure_metering_mode::average;
    }
    const auto optional_number = [&](std::string_view name, float& value)
    {
        if (object.find(std::string{"\""} + std::string{name} + "\"") == std::string_view::npos) return true;
        return number_value(object, name, value);
    };
    return number_value(object, "fovYDegrees", out.fov_y_degrees) &&
           number_value(object, "orthographicHeight", out.orthographic_height) &&
           number_value(object, "nearPlane", out.near_plane) && number_value(object, "farPlane", out.far_plane) &&
           bool_value(object, "active", out.active) && array4_value(object, "clearColor", out.clear_color) &&
           optional_number("manualEV100", out.manual_ev100) &&
           optional_number("exposureCompensation", out.exposure_compensation) &&
           optional_number("minimumEV100", out.minimum_ev100) && optional_number("maximumEV100", out.maximum_ev100) &&
           optional_number("brightenSpeed", out.brighten_speed) && optional_number("darkenSpeed", out.darken_speed);
}

bool light_value(std::string_view json, std::string_view key, host_light_snapshot& out)
{
    std::string_view object;
    if (!object_value(json, key, object)) return false;
    static constexpr std::pair<std::string_view, host_light_kind> kinds[]{{"directional", host_light_kind::directional},
                                                                          {"point", host_light_kind::point},
                                                                          {"spot", host_light_kind::spot},
                                                                          {"rectangle", host_light_kind::rectangle},
                                                                          {"disk", host_light_kind::disk}};
    static constexpr std::pair<std::string_view, host_light_unit> units[]{{"unitless", host_light_unit::unitless},
                                                                          {"lumens", host_light_unit::lumen},
                                                                          {"candela", host_light_unit::candela},
                                                                          {"lux", host_light_unit::lux},
                                                                          {"nits", host_light_unit::nit}};
    if (!parse_enum(object, "kind", kinds, std::size(kinds), out.kind) ||
        !parse_enum(object, "unit", units, std::size(units), out.unit) || !array3_value(object, "color", out.color) ||
        !number_value(object, "intensity", out.intensity) || !number_value(object, "range", out.range) ||
        !number_value(object, "innerAngleDegrees", out.inner_angle_degrees) ||
        !number_value(object, "outerAngleDegrees", out.outer_angle_degrees) ||
        !number_value(object, "width", out.width) || !number_value(object, "height", out.height) ||
        !bool_value(object, "twoSided", out.two_sided) || !bool_value(object, "enabled", out.enabled) ||
        !bool_value(object, "castsShadows", out.casts_shadows) ||
        !bool_value(object, "useColorTemperature", out.use_color_temperature) ||
        !number_value(object, "temperatureKelvin", out.temperature_kelvin))
        return false;
    const auto optional_number = [&](std::string_view name, auto& value)
    {
        if (object.find(std::string{"\""} + std::string{name} + "\"") == std::string_view::npos) return true;
        return number_value(object, name, value);
    };
    const auto optional_bool = [&](std::string_view name, bool& value)
    {
        if (object.find(std::string{"\""} + std::string{name} + "\"") == std::string_view::npos) return true;
        return bool_value(object, name, value);
    };
    std::uint32_t shadow_filter = out.shadow_filter;
    std::uint32_t cache_mode = out.shadow_cache_mode;
    const bool valid =
        optional_number("shadowResolution", out.shadow_resolution) &&
        optional_number("shadowPriority", out.shadow_priority) &&
        optional_number("shadowStrength", out.shadow_strength) && optional_number("shadowBias", out.shadow_bias) &&
        optional_number("shadowNormalBias", out.shadow_normal_bias) && optional_number("shadowFilter", shadow_filter) &&
        optional_bool("contactShadows", out.contact_shadows) &&
        optional_number("contactShadowLength", out.contact_shadow_length) &&
        optional_number("shadowCacheMode", cache_mode) && optional_number("cascadeCount", out.cascade_count) &&
        optional_number("shadowDistance", out.shadow_distance) &&
        optional_number("cascadeSplitLambda", out.cascade_split_lambda) &&
        optional_number("cascadeBlendFraction", out.cascade_blend_fraction) &&
        optional_bool("stableCascades", out.stable_cascades);
    out.shadow_filter = static_cast<std::uint8_t>(shadow_filter);
    out.shadow_cache_mode = static_cast<std::uint8_t>(cache_mode);
    return valid;
}

std::string vec3_json(const host_vec3& value)
{
    std::ostringstream stream;
    stream << '[' << value.x << ',' << value.y << ',' << value.z << ']';
    return stream.str();
}

std::string quat_json(const host_quat& value)
{
    std::ostringstream stream;
    stream << '[' << value.x << ',' << value.y << ',' << value.z << ',' << value.w << ']';
    return stream.str();
}

std::string vec4_json(const host_vec4& value)
{
    std::ostringstream stream;
    stream << '[' << value.x << ',' << value.y << ',' << value.z << ',' << value.w << ']';
    return stream.str();
}

std::string environment_json(const host_environment_visibility& value)
{
    return std::string("{\"sky\":") + bool_json(value.sky) + ",\"fog\":" + bool_json(value.fog) +
           ",\"terrain\":" + bool_json(value.terrain) + ",\"water\":" + bool_json(value.water) +
           ",\"vegetation\":" + bool_json(value.vegetation) + ",\"decals\":" + bool_json(value.decals) + '}';
}

template <class Enum>
bool parse_enum(std::string_view payload, std::string_view key, const std::pair<std::string_view, Enum>* values,
                std::size_t count, Enum& out);

bool parse_environment(std::string_view payload, host_environment_visibility& out)
{
    std::string_view object;
    if (!object_value(payload, "environment", object)) return true;
    bool_value(object, "sky", out.sky);
    bool_value(object, "fog", out.fog);
    bool_value(object, "terrain", out.terrain);
    bool_value(object, "water", out.water);
    bool_value(object, "vegetation", out.vegetation);
    bool_value(object, "decals", out.decals);
    return true;
}

std::string cloud_layer_json(const host_cloud_layer& value)
{
    std::ostringstream stream;
    stream << "{\"enabled\":" << bool_json(value.enabled) << ",\"coverage\":" << value.coverage
           << ",\"density\":" << value.density << ",\"altitude\":" << value.altitude
           << ",\"thickness\":" << value.thickness << ",\"scale\":" << value.scale << ",\"detail\":" << value.detail
           << ",\"softness\":" << value.softness << ",\"windX\":" << value.wind_x << ",\"windY\":" << value.wind_y
           << ",\"windSpeed\":" << value.wind_speed << ",\"lightingStrength\":" << value.lighting_strength
           << ",\"silverLining\":" << value.silver_lining << '}';
    return stream.str();
}

void parse_cloud_layer(std::string_view json, host_cloud_layer& value)
{
    bool_value(json, "enabled", value.enabled);
    number_value(json, "coverage", value.coverage);
    number_value(json, "density", value.density);
    number_value(json, "altitude", value.altitude);
    number_value(json, "thickness", value.thickness);
    number_value(json, "scale", value.scale);
    number_value(json, "detail", value.detail);
    number_value(json, "softness", value.softness);
    number_value(json, "windX", value.wind_x);
    number_value(json, "windY", value.wind_y);
    number_value(json, "windSpeed", value.wind_speed);
    number_value(json, "lightingStrength", value.lighting_strength);
    number_value(json, "silverLining", value.silver_lining);
}

bool parse_world_environment(std::string_view payload, host_world_environment_snapshot& value,
                             bool require_entity = true)
{
    std::string_view json;
    if (!object_value(payload, "environment", json)) json = payload;
    entity_field_value(json, "entity", value.entity);
    bool_value(json, "enabled", value.enabled);
    bool_value(json, "skyVisible", value.sky_visible);
    bool_value(json, "affectLighting", value.affect_lighting);
    static constexpr std::pair<std::string_view, host_sky_source> sky_sources[]{
        {"physicalAtmosphere", host_sky_source::physical_atmosphere},
        {"hdri", host_sky_source::hdri},
        {"solidColor", host_sky_source::solid_color}};
    parse_enum(json, "skySource", sky_sources, std::size(sky_sources), value.sky_source);
    array3_value(json, "solidColor", value.solid_color);
    string_value(json, "hdriPath", value.hdri_path);
    number_value(json, "hdriRotationDegrees", value.hdri_rotation_degrees);
    number_value(json, "radianceIntensity", value.radiance_intensity);
    number_value(json, "planetRadius", value.planet_radius);
    number_value(json, "atmosphereRadius", value.atmosphere_radius);
    number_value(json, "rayleighStrength", value.rayleigh_strength);
    number_value(json, "mieStrength", value.mie_strength);
    number_value(json, "ozoneStrength", value.ozone_strength);
    array3_value(json, "atmosphereTint", value.atmosphere_tint);
    array3_value(json, "groundAlbedo", value.ground_albedo);
    number_value(json, "mieAnisotropy", value.mie_anisotropy);
    number_value(json, "rayleighScaleHeight", value.rayleigh_scale_height);
    number_value(json, "mieScaleHeight", value.mie_scale_height);
    number_value(json, "multiScatteringFactor", value.multi_scattering_factor);
    number_value(json, "exposure", value.exposure);
    number_value(json, "sunDiskSize", value.sun_disk_size);
    number_value(json, "sunDiskIntensity", value.sun_disk_intensity);
    static constexpr std::pair<std::string_view, host_sun_position_mode> sun_modes[]{
        {"manualLight", host_sun_position_mode::manual_light}, {"geographic", host_sun_position_mode::geographic}};
    static constexpr std::pair<std::string_view, host_celestial_time_mode> time_modes[]{
        {"fixed", host_celestial_time_mode::fixed},
        {"simulated", host_celestial_time_mode::simulated},
        {"systemClock", host_celestial_time_mode::system_clock}};
    parse_enum(json, "sunMode", sun_modes, std::size(sun_modes), value.sun_mode);
    parse_enum(json, "timeMode", time_modes, std::size(time_modes), value.time_mode);
    number_value(json, "latitudeDegrees", value.latitude_degrees);
    number_value(json, "longitudeDegrees", value.longitude_degrees);
    number_value(json, "northOffsetDegrees", value.north_offset_degrees);
    number_value(json, "year", value.year);
    number_value(json, "month", value.month);
    number_value(json, "day", value.day);
    number_value(json, "localTimeHours", value.local_time_hours);
    number_value(json, "utcOffsetHours", value.utc_offset_hours);
    bool_value(json, "playing", value.playing);
    bool_value(json, "loopDay", value.loop_day);
    number_value(json, "timeScale", value.time_scale);
    bool_value(json, "automaticSunLight", value.automatic_sun_light);
    number_value(json, "sunIntensityMultiplier", value.sun_intensity_multiplier);
    number_value(json, "sunTemperatureMultiplier", value.sun_temperature_multiplier);
    bool_value(json, "moonEnabled", value.moon_enabled);
    bool_value(json, "automaticMoonPhase", value.automatic_moon_phase);
    number_value(json, "moonPhase", value.moon_phase);
    number_value(json, "moonIntensity", value.moon_intensity);
    number_value(json, "moonAngularRadiusDegrees", value.moon_angular_radius_degrees);
    bool_value(json, "starsEnabled", value.stars_enabled);
    number_value(json, "starDensity", value.star_density);
    number_value(json, "starIntensity", value.star_intensity);
    number_value(json, "starTwinkle", value.star_twinkle);
    bool_value(json, "cloudsEnabled", value.clouds_enabled);
    bool_value(json, "cloudShadows", value.cloud_shadows);
    std::string_view layer;
    if (object_value(json, "cumulus", layer)) parse_cloud_layer(layer, value.cumulus);
    if (object_value(json, "cirrus", layer)) parse_cloud_layer(layer, value.cirrus);
    bool_value(json, "fogEnabled", value.fog_enabled);
    array3_value(json, "fogColor", value.fog_color);
    number_value(json, "fogDensity", value.fog_density);
    number_value(json, "fogHeightFalloff", value.fog_height_falloff);
    number_value(json, "fogStartDistance", value.fog_start_distance);
    number_value(json, "fogMaxOpacity", value.fog_max_opacity);
    number_value(json, "fogSunScattering", value.fog_sun_scattering);
    bool_value(json, "lightingEnabled", value.lighting_enabled);
    static constexpr std::pair<std::string_view, host_environment_lighting_source> lighting_sources[]{
        {"followSky", host_environment_lighting_source::follow_sky},
        {"hdri", host_environment_lighting_source::hdri},
        {"constantColor", host_environment_lighting_source::constant_color}};
    parse_enum(json, "lightingSource", lighting_sources, std::size(lighting_sources), value.lighting_source);
    array3_value(json, "lightingColor", value.lighting_color);
    number_value(json, "diffuseIntensity", value.diffuse_intensity);
    number_value(json, "specularIntensity", value.specular_intensity);
    bool_value(json, "indirectLightingEnabled", value.indirect_lighting_enabled);
    static constexpr std::pair<std::string_view, host_indirect_lighting_method> indirect_methods[]{
        {"autoSelect", host_indirect_lighting_method::auto_select},
        {"bakedProbe", host_indirect_lighting_method::baked_probe},
        {"screenSpace", host_indirect_lighting_method::screen_space},
        {"software", host_indirect_lighting_method::software},
        {"hybridHardware", host_indirect_lighting_method::hybrid_hardware}};
    parse_enum(json, "indirectLightingMethod", indirect_methods, std::size(indirect_methods),
               value.indirect_lighting_method);
    number_value(json, "indirectDiffuseIntensity", value.indirect_diffuse_intensity);
    number_value(json, "reflectionIntensity", value.reflection_intensity);
    number_value(json, "emissiveContribution", value.emissive_contribution);
    number_value(json, "maximumTraceDistance", value.maximum_trace_distance);
    number_value(json, "surfaceCacheDetail", value.surface_cache_detail);
    bool_value(json, "allowHardwareRayTracing", value.allow_hardware_ray_tracing);
    return !require_entity || value.entity.valid();
}

template <class Enum>
bool parse_enum(std::string_view payload, std::string_view key, const std::pair<std::string_view, Enum>* values,
                std::size_t count, Enum& out)
{
    std::string text;
    return string_value(payload, key, text) && enum_from_string(std::string_view(text), values, count, out);
}

} // namespace

const char* to_string(host_event_type value) noexcept
{
    switch (value)
    {
        case host_event_type::host_started:
            return "host.started";
        case host_event_type::host_shutdown:
            return "host.shutdown";
        case host_event_type::project_opened:
            return "project.opened";
        case host_event_type::project_closed:
            return "project.closed";
        case host_event_type::project_module_reloaded:
            return "project.moduleReloaded";
        case host_event_type::scene_changed:
            return "scene.changed";
        case host_event_type::entity_created:
            return "entity.created";
        case host_event_type::entity_deleted:
            return "entity.deleted";
        case host_event_type::entity_selected:
            return "entity.selected";
        case host_event_type::component_changed:
            return "component.changed";
        case host_event_type::command_failed:
            return "command.failed";
        case host_event_type::viewport_error:
            return "viewport.error";
        case host_event_type::profiler_snapshot:
            return "profiler.snapshot";
        case host_event_type::terrain_tool_changed:
            return "terrain.toolChanged";
        case host_event_type::terrain_stroke_committed:
            return "terrain.strokeCommitted";
        case host_event_type::runtime_state_changed:
            return "runtime.stateChanged";
        case host_event_type::runtime_tick_completed:
            return "runtime.tickCompleted";
        case host_event_type::runtime_fault:
            return "runtime.fault";
        case host_event_type::asset_changed:
            return "asset.changed";
    }
    return "unknown";
}

const char* to_string(host_runtime_state value) noexcept
{
    switch (value)
    {
        case host_runtime_state::stopped:
            return "stopped";
        case host_runtime_state::running:
            return "running";
        case host_runtime_state::paused:
            return "paused";
        case host_runtime_state::faulted:
            return "faulted";
    }
    return "stopped";
}

const char* to_string(host_entity_kind value) noexcept
{
    switch (value)
    {
        case host_entity_kind::camera:
            return "camera";
        case host_entity_kind::light:
            return "light";
        case host_entity_kind::environment:
            return "environment";
        case host_entity_kind::mesh:
            return "mesh";
        case host_entity_kind::primitive:
            return "primitive";
        case host_entity_kind::imported:
            return "imported";
        case host_entity_kind::unknown:
            return "unknown";
    }
    return "unknown";
}

const char* to_string(host_component_kind value) noexcept
{
    switch (value)
    {
        case host_component_kind::transform:
            return "transform";
        case host_component_kind::camera:
            return "camera";
        case host_component_kind::mesh_renderer:
            return "meshRenderer";
        case host_component_kind::directional_light:
            return "directionalLight";
        case host_component_kind::point_light:
            return "pointLight";
        case host_component_kind::spot_light:
            return "spotLight";
        case host_component_kind::area_light:
            return "areaLight";
        case host_component_kind::world_environment:
            return "worldEnvironment";
        case host_component_kind::sky_atmosphere:
            return "skyAtmosphere";
        case host_component_kind::celestial_sky:
            return "celestialSky";
        case host_component_kind::cloud_layers:
            return "cloudLayers";
        case host_component_kind::environment_lighting:
            return "environmentLighting";
        case host_component_kind::height_fog:
            return "heightFog";
        case host_component_kind::terrain:
            return "terrain";
        case host_component_kind::water:
            return "water";
        case host_component_kind::vegetation:
            return "vegetation";
        case host_component_kind::decal:
            return "decal";
        case host_component_kind::prefab_instance:
            return "prefabInstance";
    }
    return "unknown";
}

const char* to_string(host_create_entity_kind value) noexcept
{
    switch (value)
    {
        case host_create_entity_kind::empty:
            return "empty";
        case host_create_entity_kind::plane:
            return "plane";
        case host_create_entity_kind::cube:
            return "cube";
        case host_create_entity_kind::sphere:
            return "sphere";
        case host_create_entity_kind::cylinder:
            return "cylinder";
        case host_create_entity_kind::cone:
            return "cone";
        case host_create_entity_kind::capsule:
            return "capsule";
        case host_create_entity_kind::world_environment:
            return "worldEnvironment";
        case host_create_entity_kind::terrain:
            return "terrain";
        case host_create_entity_kind::water:
            return "water";
        case host_create_entity_kind::grass_patch:
            return "grassPatch";
        case host_create_entity_kind::decal:
            return "decal";
    }
    return "cube";
}

const char* to_string(host_camera_projection value) noexcept
{
    return value == host_camera_projection::orthographic ? "orthographic" : "perspective";
}

const char* to_string(host_mobility value) noexcept
{
    switch (value)
    {
        case host_mobility::static_object:
            return "static";
        case host_mobility::stationary:
            return "stationary";
        case host_mobility::movable:
            return "movable";
    }
    return "movable";
}

const char* to_string(host_render_mode value) noexcept
{
    return value == host_render_mode::wireframe ? "wireframe" : "shaded";
}

const char* to_string(host_visualization_mode value) noexcept
{
    switch (value)
    {
        case host_visualization_mode::standard:
            return "standard";
        case host_visualization_mode::albedo:
            return "albedo";
        case host_visualization_mode::opacity:
            return "opacity";
        case host_visualization_mode::world_normal:
            return "worldNormal";
        case host_visualization_mode::specularity:
            return "specularity";
        case host_visualization_mode::gloss:
            return "gloss";
        case host_visualization_mode::metalness:
            return "metalness";
        case host_visualization_mode::ao:
            return "ao";
        case host_visualization_mode::emission:
            return "emission";
        case host_visualization_mode::lighting:
            return "lighting";
        case host_visualization_mode::uv0:
            return "uv0";
        case host_visualization_mode::cascade_debug:
            return "cascadeDebug";
        case host_visualization_mode::shadow_mask:
            return "shadowMask";
        case host_visualization_mode::light_complexity:
            return "lightComplexity";
        case host_visualization_mode::cluster_debug:
            return "clusterDebug";
        case host_visualization_mode::virtual_hierarchy_level:
            return "virtualHierarchyLevel";
        case host_visualization_mode::virtual_geometric_error:
            return "virtualGeometricError";
        case host_visualization_mode::virtual_page_residency:
            return "virtualPageResidency";
        case host_visualization_mode::virtual_overdraw:
            return "virtualOverdraw";
        case host_visualization_mode::virtual_triangles_per_pixel:
            return "virtualTrianglesPerPixel";
        case host_visualization_mode::surface_cards:
            return "surfaceCards";
        case host_visualization_mode::surface_card_residency:
            return "surfaceCardResidency";
        case host_visualization_mode::surface_material_cache:
            return "surfaceMaterialCache";
        case host_visualization_mode::surface_radiance_cache:
            return "surfaceRadianceCache";
        case host_visualization_mode::mesh_distance_fields:
            return "meshDistanceFields";
        case host_visualization_mode::global_distance_field:
            return "globalDistanceField";
        case host_visualization_mode::radiance_probes:
            return "radianceProbes";
        case host_visualization_mode::lighting_trace_source:
            return "lightingTraceSource";
        case host_visualization_mode::lighting_hit_distance:
            return "lightingHitDistance";
        case host_visualization_mode::lighting_temporal_confidence:
            return "lightingTemporalConfidence";
        case host_visualization_mode::indirect_diffuse:
            return "indirectDiffuse";
        case host_visualization_mode::reflections:
            return "reflections";
        case host_visualization_mode::denoiser_variance:
            return "denoiserVariance";
    }
    return "standard";
}

const char* to_string(host_overlay_mode value) noexcept
{
    switch (value)
    {
        case host_overlay_mode::none:
            return "none";
        case host_overlay_mode::selected_wireframe:
            return "selectedWireframe";
        case host_overlay_mode::all_wireframe:
            return "allWireframe";
    }
    return "selectedWireframe";
}

const char* to_string(host_sky_source value) noexcept
{
    switch (value)
    {
        case host_sky_source::physical_atmosphere:
            return "physicalAtmosphere";
        case host_sky_source::hdri:
            return "hdri";
        case host_sky_source::solid_color:
            return "solidColor";
    }
    return "physicalAtmosphere";
}

const char* to_string(host_sun_position_mode value) noexcept
{
    return value == host_sun_position_mode::geographic ? "geographic" : "manualLight";
}

const char* to_string(host_celestial_time_mode value) noexcept
{
    switch (value)
    {
        case host_celestial_time_mode::fixed:
            return "fixed";
        case host_celestial_time_mode::simulated:
            return "simulated";
        case host_celestial_time_mode::system_clock:
            return "systemClock";
    }
    return "fixed";
}

const char* to_string(host_environment_lighting_source value) noexcept
{
    switch (value)
    {
        case host_environment_lighting_source::follow_sky:
            return "followSky";
        case host_environment_lighting_source::hdri:
            return "hdri";
        case host_environment_lighting_source::constant_color:
            return "constantColor";
    }
    return "followSky";
}

const char* to_string(host_indirect_lighting_method value) noexcept
{
    switch (value)
    {
        case host_indirect_lighting_method::auto_select:
            return "autoSelect";
        case host_indirect_lighting_method::baked_probe:
            return "bakedProbe";
        case host_indirect_lighting_method::screen_space:
            return "screenSpace";
        case host_indirect_lighting_method::software:
            return "software";
        case host_indirect_lighting_method::hybrid_hardware:
            return "hybridHardware";
    }
    return "autoSelect";
}

const char* to_string(host_world_environment_preset value) noexcept
{
    switch (value)
    {
        case host_world_environment_preset::clear_day:
            return "clearDay";
        case host_world_environment_preset::alpine_late_morning:
            return "alpineLateMorning";
        case host_world_environment_preset::golden_hour:
            return "goldenHour";
        case host_world_environment_preset::overcast:
            return "overcast";
        case host_world_environment_preset::night:
            return "night";
        case host_world_environment_preset::indoor_neutral:
            return "indoorNeutral";
    }
    return "alpineLateMorning";
}

std::string command_type(const host_command_payload& payload)
{
    return std::visit(
        [](const auto& value) -> std::string
        {
            using type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<type, host_open_project_command>)
                return "project.open";
            else if constexpr (std::is_same_v<type, host_close_project_command>)
                return "project.close";
            else if constexpr (std::is_same_v<type, host_reload_project_module_command>)
                return "project.reloadModule";
            else if constexpr (std::is_same_v<type, host_open_scene_command>)
                return "scene.open";
            else if constexpr (std::is_same_v<type, host_new_scene_command>)
                return "scene.new";
            else if constexpr (std::is_same_v<type, host_save_scene_command>)
                return "scene.save";
            else if constexpr (std::is_same_v<type, host_save_scene_as_command>)
                return "scene.saveAs";
            else if constexpr (std::is_same_v<type, host_autosave_scene_command>)
                return "scene.autosave";
            else if constexpr (std::is_same_v<type, host_open_recovery_scene_command>)
                return "scene.openRecovery";
            else if constexpr (std::is_same_v<type, host_asset_reimport_command>)
                return "asset.reimport";
            else if constexpr (std::is_same_v<type, host_asset_cancel_import_command>)
                return "asset.cancelImport";
            else if constexpr (std::is_same_v<type, host_asset_move_command>)
                return "asset.move";
            else if constexpr (std::is_same_v<type, host_asset_rename_command>)
                return "asset.rename";
            else if constexpr (std::is_same_v<type, host_create_entity_command>)
                return "entity.create";
            else if constexpr (std::is_same_v<type, host_delete_entity_command>)
                return "entity.delete";
            else if constexpr (std::is_same_v<type, host_duplicate_entity_command>)
                return "entity.duplicate";
            else if constexpr (std::is_same_v<type, host_create_prefab_command>)
                return "prefab.create";
            else if constexpr (std::is_same_v<type, host_instantiate_prefab_command>)
                return "prefab.instantiate";
            else if constexpr (std::is_same_v<type, host_apply_prefab_command>)
                return "prefab.apply";
            else if constexpr (std::is_same_v<type, host_revert_prefab_command>)
                return "prefab.revert";
            else if constexpr (std::is_same_v<type, host_unpack_prefab_command>)
                return "prefab.unpack";
            else if constexpr (std::is_same_v<type, host_revert_prefab_override_command>)
                return "prefab.revertOverride";
            else if constexpr (std::is_same_v<type, host_reparent_entity_command>)
                return "entity.reparent";
            else if constexpr (std::is_same_v<type, host_reorder_entity_command>)
                return "entity.reorder";
            else if constexpr (std::is_same_v<type, host_rename_entity_command>)
                return "entity.rename";
            else if constexpr (std::is_same_v<type, host_select_entity_command>)
                return "entity.select";
            else if constexpr (std::is_same_v<type, host_clear_selection_command>)
                return "entity.clearSelection";
            else if constexpr (std::is_same_v<type, host_set_active_command>)
                return "entity.setActive";
            else if constexpr (std::is_same_v<type, host_set_tag_command>)
                return "entity.setTag";
            else if constexpr (std::is_same_v<type, host_set_transform_command>)
                return "entity.setTransform";
            else if constexpr (std::is_same_v<type, host_set_render_layer_command>)
                return "entity.setRenderLayer";
            else if constexpr (std::is_same_v<type, host_set_mobility_command>)
                return "entity.setMobility";
            else if constexpr (std::is_same_v<type, host_set_camera_command>)
                return "entity.setCamera";
            else if constexpr (std::is_same_v<type, host_set_light_command>)
                return "entity.setLight";
            else if constexpr (std::is_same_v<type, host_set_mesh_renderer_command>)
                return "entity.setMeshRenderer";
            else if constexpr (std::is_same_v<type, host_set_terrain_command>)
                return "terrain.update";
            else if constexpr (std::is_same_v<type, host_set_terrain_brush_command>)
                return "terrain.setBrush";
            else if constexpr (std::is_same_v<type, host_set_terrain_layer_command>)
                return "terrain.assignLayer";
            else if constexpr (std::is_same_v<type, host_terrain_stroke_command>)
                return "terrain.stroke";
            else if constexpr (std::is_same_v<type, host_terrain_hover_command>)
                return "terrain.hover";
            else if constexpr (std::is_same_v<type, host_set_entity_material_command>)
                return "entity.setMaterial";
            else if constexpr (std::is_same_v<type, host_component_operation_command>)
            {
                switch (value.operation)
                {
                    case host_component_operation::add:
                        return "component.add";
                    case host_component_operation::remove:
                        return "component.remove";
                    case host_component_operation::reset:
                        return "component.reset";
                }
                return "component.reset";
            }
            else if constexpr (std::is_same_v<type, host_patch_project_component_command>)
                return "component.patchField";
            else if constexpr (std::is_same_v<type, host_set_world_environment_command>)
                return "environment.update";
            else if constexpr (std::is_same_v<type, host_apply_world_environment_preset_command>)
                return "environment.applyPreset";
            else if constexpr (std::is_same_v<type, host_set_environment_hdri_command>)
                return "environment.setHdri";
            else if constexpr (std::is_same_v<type, host_set_camera_projection_command>)
                return "camera.setProjection";
            else if constexpr (std::is_same_v<type, host_viewport_attach_command>)
                return "viewport.attach";
            else if constexpr (std::is_same_v<type, host_viewport_resize_command>)
                return "viewport.resize";
            else if constexpr (std::is_same_v<type, host_viewport_detach_command>)
                return "viewport.detach";
            else if constexpr (std::is_same_v<type, host_viewport_set_camera_mode_command>)
                return "viewport.setCameraMode";
            else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>)
                return "viewport.setRenderOptions";
            else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)
                return "viewport.cameraInput";
            else if constexpr (std::is_same_v<type, host_viewport_set_pose_command>)
                return "viewport.setPose";
            else if constexpr (std::is_same_v<type, host_history_undo_command>)
                return "history.undo";
            else if constexpr (std::is_same_v<type, host_history_redo_command>)
                return "history.redo";
            else if constexpr (std::is_same_v<type, host_history_begin_transaction_command>)
                return "history.beginTransaction";
            else if constexpr (std::is_same_v<type, host_history_commit_transaction_command>)
                return "history.commitTransaction";
            else if constexpr (std::is_same_v<type, host_history_cancel_transaction_command>)
                return "history.cancelTransaction";
            else if constexpr (std::is_same_v<type, host_runtime_resume_command>)
                return "runtime.resume";
            else if constexpr (std::is_same_v<type, host_runtime_pause_command>)
                return "runtime.pause";
            else if constexpr (std::is_same_v<type, host_runtime_stop_command>)
                return "runtime.stop";
            else if constexpr (std::is_same_v<type, host_runtime_step_command>)
                return "runtime.step";
            else if constexpr (std::is_same_v<type, host_runtime_set_time_scale_command>)
                return "runtime.setTimeScale";
            else if constexpr (std::is_same_v<type, host_runtime_capture_snapshot_command>)
                return "runtime.captureSnapshot";
            else if constexpr (std::is_same_v<type, host_runtime_restore_snapshot_command>)
                return "runtime.restoreSnapshot";
            else if constexpr (std::is_same_v<type, host_viewport_set_tool_command>)
                return "viewport.setTool";
            else if constexpr (std::is_same_v<type, host_viewport_pick_command>)
                return "viewport.pick";
            else if constexpr (std::is_same_v<type, host_viewport_capture_command>)
                return "viewport.capture";
            else
                return "unknown";
        },
        payload);
}

std::string query_type(const host_query_payload& payload)
{
    return std::visit(
        [](const auto& value) -> std::string
        {
            using type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<type, host_scene_hierarchy_query>)
                return "scene.hierarchy";
            else if constexpr (std::is_same_v<type, host_selected_entity_query>)
                return "entity.selected";
            else if constexpr (std::is_same_v<type, host_scene_entities_query>)
                return "gateway.sceneEntities";
            else if constexpr (std::is_same_v<type, host_entity_by_guid_query>)
                return "gateway.entity";
            else if constexpr (std::is_same_v<type, host_scene_spatial_query>)
                return "gateway.spatialQuery";
            else if constexpr (std::is_same_v<type, host_component_schema_query>)
                return "gateway.componentSchemas";
            else if constexpr (std::is_same_v<type, host_workspace_documents_query>)
                return "workspace.documents";
            else if constexpr (std::is_same_v<type, host_gateway_diagnostics_query>)
                return "gateway.diagnostics";
            else if constexpr (std::is_same_v<type, host_viewport_capture_query>)
                return "viewport.captureResult";
            else if constexpr (std::is_same_v<type, host_project_assets_query>)
                return "project.assets";
            else if constexpr (std::is_same_v<type, host_asset_thumbnail_query>)
                return "asset.thumbnail";
            else if constexpr (std::is_same_v<type, host_viewport_state_query>)
                return "viewport.state";
            else if constexpr (std::is_same_v<type, host_world_environment_query>)
                return "environment.state";
            else if constexpr (std::is_same_v<type, host_history_state_query>)
                return "history.state";
            else if constexpr (std::is_same_v<type, host_runtime_state_query>)
                return "runtime.state";
            else if constexpr (std::is_same_v<type, host_terrain_tool_state_query>)
                return "terrain.toolState";
            else
                return "unknown";
        },
        payload);
}

std::string to_json(const host_runtime_snapshot& snapshot)
{
    std::ostringstream stream;
    stream << "{\"state\":" << quote(to_string(snapshot.state)) << ",\"tickId\":" << snapshot.tick_id
           << ",\"revision\":" << snapshot.revision << ",\"discardedTicks\":" << snapshot.discarded_ticks
           << ",\"timeScale\":" << snapshot.time_scale << ",\"interpolationAlpha\":" << snapshot.interpolation_alpha
           << ",\"worldCount\":" << snapshot.world_count << '}';
    return stream.str();
}

std::string to_json(const host_terrain_tool_snapshot& snapshot)
{
    const char* tool = snapshot.tool == host_terrain_brush_tool::smooth    ? "smooth"
                       : snapshot.tool == host_terrain_brush_tool::flatten ? "flatten"
                       : snapshot.tool == host_terrain_brush_tool::paint   ? "paint"
                                                                           : "sculpt";
    return "{\"entity\":" + to_json(snapshot.entity) + ",\"active\":" + bool_json(snapshot.active) +
           ",\"hoverVisible\":" + bool_json(snapshot.hover_visible) + ",\"tool\":" + quote(tool) +
           ",\"radius\":" + std::to_string(snapshot.radius) + ",\"strength\":" + std::to_string(snapshot.strength) +
           ",\"falloff\":" + std::to_string(snapshot.falloff) +
           ",\"activeLayer\":" + std::to_string(snapshot.active_layer) + '}';
}

std::string to_json(const host_entity_id& entity)
{
    return "{\"index\":" + std::to_string(entity.index) + ",\"generation\":" + std::to_string(entity.generation) + '}';
}

std::string to_json_string(std::string_view value)
{
    return quote(value);
}

std::string to_json(const host_transform& transform)
{
    return "{\"position\":" + vec3_json(transform.position) + ",\"rotation\":" + quat_json(transform.rotation) +
           ",\"scale\":" + vec3_json(transform.scale) + '}';
}

std::string to_json(const host_camera_snapshot& camera)
{
    std::ostringstream stream;
    stream << "{\"projection\":" << quote(to_string(camera.projection)) << ",\"fovYDegrees\":" << camera.fov_y_degrees
           << ",\"orthographicHeight\":" << camera.orthographic_height << ",\"nearPlane\":" << camera.near_plane
           << ",\"farPlane\":" << camera.far_plane << ",\"active\":" << bool_json(camera.active)
           << ",\"clearColor\":" << vec4_json(camera.clear_color)
           << ",\"exposureMode\":" << quote(camera.exposure_mode == host_exposure_mode::manual ? "manual" : "automatic")
           << ",\"exposureMetering\":"
           << quote(camera.exposure_metering == host_exposure_metering_mode::center_weighted ? "centerWeighted"
                                                                                             : "average")
           << ",\"manualEV100\":" << camera.manual_ev100 << ",\"exposureCompensation\":" << camera.exposure_compensation
           << ",\"minimumEV100\":" << camera.minimum_ev100 << ",\"maximumEV100\":" << camera.maximum_ev100
           << ",\"brightenSpeed\":" << camera.brighten_speed << ",\"darkenSpeed\":" << camera.darken_speed << '}';
    return stream.str();
}

std::string to_json(const host_light_snapshot& light)
{
    const char* kind = light.kind == host_light_kind::directional ? "directional"
                       : light.kind == host_light_kind::point     ? "point"
                       : light.kind == host_light_kind::spot      ? "spot"
                       : light.kind == host_light_kind::disk      ? "disk"
                                                                  : "rectangle";
    const char* unit = light.unit == host_light_unit::lumen     ? "lumens"
                       : light.unit == host_light_unit::candela ? "candela"
                       : light.unit == host_light_unit::lux     ? "lux"
                       : light.unit == host_light_unit::nit     ? "nits"
                                                                : "unitless";
    std::ostringstream stream;
    stream << "{\"kind\":" << quote(kind) << ",\"unit\":" << quote(unit) << ",\"color\":" << vec3_json(light.color)
           << ",\"intensity\":" << light.intensity << ",\"range\":" << light.range
           << ",\"innerAngleDegrees\":" << light.inner_angle_degrees
           << ",\"outerAngleDegrees\":" << light.outer_angle_degrees << ",\"width\":" << light.width
           << ",\"height\":" << light.height << ",\"twoSided\":" << bool_json(light.two_sided)
           << ",\"enabled\":" << bool_json(light.enabled) << ",\"castsShadows\":" << bool_json(light.casts_shadows)
           << ",\"shadowResolution\":" << light.shadow_resolution << ",\"shadowPriority\":" << light.shadow_priority
           << ",\"shadowStrength\":" << light.shadow_strength << ",\"shadowBias\":" << light.shadow_bias
           << ",\"shadowNormalBias\":" << light.shadow_normal_bias
           << ",\"shadowFilter\":" << static_cast<unsigned>(light.shadow_filter)
           << ",\"contactShadows\":" << bool_json(light.contact_shadows)
           << ",\"contactShadowLength\":" << light.contact_shadow_length
           << ",\"shadowCacheMode\":" << static_cast<unsigned>(light.shadow_cache_mode)
           << ",\"cascadeCount\":" << light.cascade_count << ",\"shadowDistance\":" << light.shadow_distance
           << ",\"cascadeSplitLambda\":" << light.cascade_split_lambda
           << ",\"cascadeBlendFraction\":" << light.cascade_blend_fraction
           << ",\"stableCascades\":" << bool_json(light.stable_cascades)
           << ",\"useColorTemperature\":" << bool_json(light.use_color_temperature)
           << ",\"temperatureKelvin\":" << light.temperature_kelvin << '}';
    return stream.str();
}

std::string to_json(const host_mesh_renderer_snapshot& mesh_renderer)
{
    const auto representation = mesh_renderer.representation == 1u   ? "conventional"
                                : mesh_renderer.representation == 2u ? "virtualized"
                                                                     : "auto";
    return std::string("{\"representation\":") + quote(representation) +
           ",\"visible\":" + bool_json(mesh_renderer.visible) +
           ",\"castsShadows\":" + bool_json(mesh_renderer.casts_shadows) +
           ",\"receivesShadows\":" + bool_json(mesh_renderer.receives_shadows) +
           ",\"shadowLodBias\":" + std::to_string(mesh_renderer.shadow_lod_bias) +
           ",\"maximumShadowDistance\":" + std::to_string(mesh_renderer.maximum_shadow_distance) +
           ",\"baseColorTint\":" + vec4_json(mesh_renderer.base_color_tint) +
           ",\"hasMaterial\":" + bool_json(mesh_renderer.has_material) +
           ",\"assetBackedMaterial\":" + bool_json(mesh_renderer.asset_backed_material) +
           ",\"materialName\":" + quote(mesh_renderer.material_name) +
           ",\"materialPath\":" + quote(mesh_renderer.material_path) + '}';
}

std::string to_json(const host_terrain_snapshot& terrain)
{
    const char* tool = terrain.brush_tool == host_terrain_brush_tool::smooth    ? "smooth"
                       : terrain.brush_tool == host_terrain_brush_tool::flatten ? "flatten"
                       : terrain.brush_tool == host_terrain_brush_tool::paint   ? "paint"
                                                                                : "sculpt";
    std::ostringstream stream;
    stream << "{\"enabled\":" << bool_json(terrain.enabled) << ",\"size\":" << terrain.size
           << ",\"resolution\":" << terrain.resolution << ",\"chunkQuads\":" << terrain.chunk_quads
           << ",\"patchQuads\":" << terrain.patch_quads << ",\"maximumHierarchyDepth\":"
           << terrain.maximum_hierarchy_depth << ",\"geometricErrorMultiplier\":"
           << terrain.geometric_error_multiplier
           << ",\"receiveShadows\":" << bool_json(terrain.receive_shadows)
           << ",\"castShadows\":" << bool_json(terrain.cast_shadows) << ",\"shadowLodBias\":" << terrain.shadow_lod_bias
           << ",\"maximumShadowDistance\":" << terrain.maximum_shadow_distance
           << ",\"contentRevision\":" << terrain.content_revision << ",\"brushTool\":" << quote(tool)
           << ",\"brushRadius\":" << terrain.brush_radius << ",\"brushStrength\":" << terrain.brush_strength
           << ",\"brushFalloff\":" << terrain.brush_falloff << ",\"activeLayer\":" << terrain.active_layer
           << ",\"layers\":[";
    for (std::size_t index = 0; index < terrain.layer_names.size(); ++index)
    {
        if (index != 0) stream << ',';
        stream << "{\"name\":" << quote(terrain.layer_names[index])
               << ",\"baseColorPath\":" << quote(terrain.layer_base_color_paths[index]) << '}';
    }
    stream << "]}";
    return stream.str();
}

std::string to_json(const host_world_environment_snapshot& value)
{
    std::ostringstream stream;
    stream << "{\"entity\":" << to_json(value.entity) << ",\"enabled\":" << bool_json(value.enabled)
           << ",\"skyVisible\":" << bool_json(value.sky_visible)
           << ",\"affectLighting\":" << bool_json(value.affect_lighting)
           << ",\"skySource\":" << quote(to_string(value.sky_source))
           << ",\"solidColor\":" << vec3_json(value.solid_color) << ",\"hdriPath\":" << quote(value.hdri_path)
           << ",\"hdriRotationDegrees\":" << value.hdri_rotation_degrees
           << ",\"radianceIntensity\":" << value.radiance_intensity << ",\"planetRadius\":" << value.planet_radius
           << ",\"atmosphereRadius\":" << value.atmosphere_radius << ",\"rayleighStrength\":" << value.rayleigh_strength
           << ",\"mieStrength\":" << value.mie_strength << ",\"ozoneStrength\":" << value.ozone_strength
           << ",\"atmosphereTint\":" << vec3_json(value.atmosphere_tint)
           << ",\"groundAlbedo\":" << vec3_json(value.ground_albedo) << ",\"mieAnisotropy\":" << value.mie_anisotropy
           << ",\"rayleighScaleHeight\":" << value.rayleigh_scale_height
           << ",\"mieScaleHeight\":" << value.mie_scale_height
           << ",\"multiScatteringFactor\":" << value.multi_scattering_factor << ",\"exposure\":" << value.exposure
           << ",\"sunDiskSize\":" << value.sun_disk_size << ",\"sunDiskIntensity\":" << value.sun_disk_intensity
           << ",\"sunMode\":" << quote(to_string(value.sun_mode))
           << ",\"timeMode\":" << quote(to_string(value.time_mode)) << ",\"latitudeDegrees\":" << value.latitude_degrees
           << ",\"longitudeDegrees\":" << value.longitude_degrees
           << ",\"northOffsetDegrees\":" << value.north_offset_degrees << ",\"year\":" << value.year
           << ",\"month\":" << value.month << ",\"day\":" << value.day
           << ",\"localTimeHours\":" << value.local_time_hours << ",\"utcOffsetHours\":" << value.utc_offset_hours
           << ",\"playing\":" << bool_json(value.playing) << ",\"loopDay\":" << bool_json(value.loop_day)
           << ",\"timeScale\":" << value.time_scale << ",\"automaticSunLight\":" << bool_json(value.automatic_sun_light)
           << ",\"sunIntensityMultiplier\":" << value.sun_intensity_multiplier
           << ",\"sunTemperatureMultiplier\":" << value.sun_temperature_multiplier
           << ",\"moonEnabled\":" << bool_json(value.moon_enabled)
           << ",\"automaticMoonPhase\":" << bool_json(value.automatic_moon_phase)
           << ",\"moonPhase\":" << value.moon_phase << ",\"moonIntensity\":" << value.moon_intensity
           << ",\"moonAngularRadiusDegrees\":" << value.moon_angular_radius_degrees
           << ",\"starsEnabled\":" << bool_json(value.stars_enabled) << ",\"starDensity\":" << value.star_density
           << ",\"starIntensity\":" << value.star_intensity << ",\"starTwinkle\":" << value.star_twinkle
           << ",\"cloudsEnabled\":" << bool_json(value.clouds_enabled)
           << ",\"cloudShadows\":" << bool_json(value.cloud_shadows)
           << ",\"cumulus\":" << cloud_layer_json(value.cumulus) << ",\"cirrus\":" << cloud_layer_json(value.cirrus)
           << ",\"fogEnabled\":" << bool_json(value.fog_enabled) << ",\"fogColor\":" << vec3_json(value.fog_color)
           << ",\"fogDensity\":" << value.fog_density << ",\"fogHeightFalloff\":" << value.fog_height_falloff
           << ",\"fogStartDistance\":" << value.fog_start_distance << ",\"fogMaxOpacity\":" << value.fog_max_opacity
           << ",\"fogSunScattering\":" << value.fog_sun_scattering
           << ",\"lightingEnabled\":" << bool_json(value.lighting_enabled)
           << ",\"lightingSource\":" << quote(to_string(value.lighting_source))
           << ",\"lightingColor\":" << vec3_json(value.lighting_color)
           << ",\"diffuseIntensity\":" << value.diffuse_intensity
           << ",\"specularIntensity\":" << value.specular_intensity
           << ",\"indirectLightingEnabled\":" << bool_json(value.indirect_lighting_enabled)
           << ",\"indirectLightingMethod\":" << quote(to_string(value.indirect_lighting_method))
           << ",\"indirectDiffuseIntensity\":" << value.indirect_diffuse_intensity
           << ",\"reflectionIntensity\":" << value.reflection_intensity
           << ",\"emissiveContribution\":" << value.emissive_contribution
           << ",\"maximumTraceDistance\":" << value.maximum_trace_distance
           << ",\"surfaceCacheDetail\":" << value.surface_cache_detail
           << ",\"allowHardwareRayTracing\":" << bool_json(value.allow_hardware_ray_tracing) << '}';
    return stream.str();
}

std::string to_json(const host_command_envelope& envelope)
{
    const std::string type = envelope.command_type.empty() ? command_type(envelope.payload) : envelope.command_type;
    std::string payload_json = std::visit(
        [](const auto& payload) -> std::string
        {
            using type = std::decay_t<decltype(payload)>;
            if constexpr (std::is_same_v<type, host_open_project_command>)
                return "{\"name\":" + quote(payload.name) + ",\"root\":" + quote(payload.root.generic_string()) +
                       ",\"descriptorPath\":" + quote(payload.descriptor_path.generic_string()) +
                       ",\"contentRoots\":" + path_array_json(payload.content_roots) +
                       ",\"cacheRoot\":" + quote(payload.cache_root.generic_string()) +
                       ",\"defaultScene\":" + quote(payload.default_scene.generic_string()) +
                       ",\"projectGuid\":" + quote(payload.project_guid) +
                       ",\"engineVersion\":" + quote(payload.engine_version) +
                       ",\"editorModuleId\":" + quote(payload.editor_module_id) +
                       ",\"editorModulePath\":" + quote(payload.editor_module_path.generic_string()) +
                       ",\"readOnly\":" + bool_json(payload.read_only) + '}';
            else if constexpr (std::is_same_v<type, host_reload_project_module_command>)
                return "{\"path\":" + quote(payload.path.generic_string()) +
                       ",\"engineVersion\":" + quote(payload.engine_version) +
                       ",\"projectGuid\":" + quote(payload.project_guid) + ",\"moduleId\":" + quote(payload.module_id) +
                       '}';
            else if constexpr (std::is_same_v<type, host_open_scene_command>)
                return "{\"path\":" + quote(payload.path.generic_string()) +
                       ",\"append\":" + bool_json(payload.append) + '}';
            else if constexpr (std::is_same_v<type, host_new_scene_command>)
                return "{\"name\":" + quote(payload.name) + '}';
            else if constexpr (std::is_same_v<type, host_save_scene_as_command>)
                return "{\"path\":" + quote(payload.path.generic_string()) + '}';
            else if constexpr (std::is_same_v<type, host_autosave_scene_command>)
                return "{\"path\":" + quote(payload.path.generic_string()) + '}';
            else if constexpr (std::is_same_v<type, host_open_recovery_scene_command>)
                return "{\"path\":" + quote(payload.path.generic_string()) +
                       ",\"originalPath\":" + quote(payload.original_path.generic_string()) + '}';
            else if constexpr (std::is_same_v<type, host_asset_reimport_command> ||
                               std::is_same_v<type, host_asset_cancel_import_command>)
                return "{\"guid\":" + quote(payload.guid) + '}';
            else if constexpr (std::is_same_v<type, host_asset_move_command>)
                return "{\"guid\":" + quote(payload.guid) + ",\"path\":" + quote(payload.path.generic_string()) + '}';
            else if constexpr (std::is_same_v<type, host_asset_rename_command>)
                return "{\"guid\":" + quote(payload.guid) + ",\"name\":" + quote(payload.name) + '}';
            else if constexpr (std::is_same_v<type, host_create_entity_command>)
                return "{\"kind\":" + quote(to_string(payload.kind)) + ",\"parent\":" + to_json(payload.parent) + '}';
            else if constexpr (std::is_same_v<type, host_delete_entity_command> ||
                               std::is_same_v<type, host_duplicate_entity_command>)
                return "{\"entity\":" + to_json(payload.entity) + '}';
            else if constexpr (std::is_same_v<type, host_select_entity_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"additive\":" + bool_json(payload.additive) +
                       ",\"toggle\":" + bool_json(payload.toggle) + '}';
            else if constexpr (std::is_same_v<type, host_create_prefab_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"path\":" + quote(payload.path.generic_string()) +
                       '}';
            else if constexpr (std::is_same_v<type, host_instantiate_prefab_command>)
                return "{\"path\":" + quote(payload.path.generic_string()) + ",\"parent\":" + to_json(payload.parent) +
                       '}';
            else if constexpr (std::is_same_v<type, host_apply_prefab_command> ||
                               std::is_same_v<type, host_revert_prefab_command> ||
                               std::is_same_v<type, host_unpack_prefab_command>)
                return "{\"entity\":" + to_json(payload.entity) + '}';
            else if constexpr (std::is_same_v<type, host_revert_prefab_override_command>)
                return "{\"entity\":" + to_json(payload.entity) +
                       ",\"sourceEntity\":" + quote(payload.source_entity) +
                       ",\"componentId\":" + quote(payload.component_id) +
                       ",\"fieldId\":" + std::to_string(payload.field_id) +
                       ",\"kind\":" + quote(payload.kind) + '}';
            else if constexpr (std::is_same_v<type, host_reparent_entity_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"parent\":" + to_json(payload.parent) +
                       ",\"beforeSibling\":" + to_json(payload.before_sibling) +
                       ",\"preserveWorld\":" + bool_json(payload.preserve_world) + '}';
            else if constexpr (std::is_same_v<type, host_reorder_entity_command>)
                return "{\"entity\":" + to_json(payload.entity) +
                       ",\"beforeSibling\":" + to_json(payload.before_sibling) + '}';
            else if constexpr (std::is_same_v<type, host_rename_entity_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"name\":" + quote(payload.name) + '}';
            else if constexpr (std::is_same_v<type, host_set_active_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"active\":" + bool_json(payload.active) + '}';
            else if constexpr (std::is_same_v<type, host_set_tag_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"tag\":" + quote(payload.tag) + '}';
            else if constexpr (std::is_same_v<type, host_set_transform_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"transform\":" + to_json(payload.transform) + '}';
            else if constexpr (std::is_same_v<type, host_set_render_layer_command>)
                return "{\"entity\":" + to_json(payload.entity) +
                       ",\"renderLayerMask\":" + std::to_string(payload.render_layer_mask) + '}';
            else if constexpr (std::is_same_v<type, host_set_mobility_command>)
                return "{\"entity\":" + to_json(payload.entity) +
                       ",\"mobility\":" + quote(to_string(payload.mobility)) + '}';
            else if constexpr (std::is_same_v<type, host_set_camera_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"camera\":" + to_json(payload.camera) + '}';
            else if constexpr (std::is_same_v<type, host_set_light_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"light\":" + to_json(payload.light) + '}';
            else if constexpr (std::is_same_v<type, host_set_mesh_renderer_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"visible\":" + bool_json(payload.visible) +
                       ",\"castsShadows\":" + bool_json(payload.casts_shadows) +
                       ",\"receivesShadows\":" + bool_json(payload.receives_shadows) +
                       ",\"shadowLodBias\":" + std::to_string(payload.shadow_lod_bias) +
                       ",\"maximumShadowDistance\":" + std::to_string(payload.maximum_shadow_distance) +
                       ",\"baseColorTint\":" + vec4_json(payload.base_color_tint) + '}';
            else if constexpr (std::is_same_v<type, host_set_terrain_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"enabled\":" + bool_json(payload.enabled) +
                       ",\"receiveShadows\":" + bool_json(payload.receive_shadows) +
                       ",\"castShadows\":" + bool_json(payload.cast_shadows) +
                       ",\"shadowLodBias\":" + std::to_string(payload.shadow_lod_bias) +
                       ",\"maximumShadowDistance\":" + std::to_string(payload.maximum_shadow_distance) + '}';
            else if constexpr (std::is_same_v<type, host_set_terrain_brush_command>)
            {
                const char* tool = payload.tool == host_terrain_brush_tool::smooth    ? "smooth"
                                   : payload.tool == host_terrain_brush_tool::flatten ? "flatten"
                                   : payload.tool == host_terrain_brush_tool::paint   ? "paint"
                                                                                      : "sculpt";
                return "{\"entity\":" + to_json(payload.entity) + ",\"tool\":" + quote(tool) +
                       ",\"radius\":" + std::to_string(payload.radius) +
                       ",\"strength\":" + std::to_string(payload.strength) +
                       ",\"falloff\":" + std::to_string(payload.falloff) +
                       ",\"activeLayer\":" + std::to_string(payload.active_layer) + '}';
            }
            else if constexpr (std::is_same_v<type, host_set_terrain_layer_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"layer\":" + std::to_string(payload.layer) +
                       ",\"path\":" + quote(payload.path.generic_string()) + '}';
            else if constexpr (std::is_same_v<type, host_terrain_stroke_command>)
            {
                const char* phase = payload.phase == host_edit_phase::update   ? "update"
                                    : payload.phase == host_edit_phase::commit ? "commit"
                                    : payload.phase == host_edit_phase::cancel ? "cancel"
                                                                               : "begin";
                return "{\"entity\":" + to_json(payload.entity) + ",\"x\":" + std::to_string(payload.x) +
                       ",\"y\":" + std::to_string(payload.y) + ",\"phase\":" + quote(phase) +
                       ",\"invert\":" + bool_json(payload.invert) + '}';
            }
            else if constexpr (std::is_same_v<type, host_terrain_hover_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"x\":" + std::to_string(payload.x) +
                       ",\"y\":" + std::to_string(payload.y) + ",\"clear\":" + bool_json(payload.clear) + '}';
            else if constexpr (std::is_same_v<type, host_set_entity_material_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"path\":" + quote(payload.path.generic_string()) +
                       '}';
            else if constexpr (std::is_same_v<type, host_component_operation_command>)
                return "{\"component\":" + quote(payload.component) + '}';
            else if constexpr (std::is_same_v<type, host_patch_project_component_command>)
                return "{\"component\":" + quote(payload.component) + ",\"field\":" + quote(payload.field) +
                       ",\"value\":" + payload.value_json + '}';
            else if constexpr (std::is_same_v<type, host_set_world_environment_command>)
                return "{\"environment\":" + to_json(payload.environment) + '}';
            else if constexpr (std::is_same_v<type, host_apply_world_environment_preset_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"preset\":" + quote(to_string(payload.preset)) +
                       '}';
            else if constexpr (std::is_same_v<type, host_set_environment_hdri_command>)
                return "{\"entity\":" + to_json(payload.entity) + ",\"path\":" + quote(payload.path.generic_string()) +
                       '}';
            else if constexpr (std::is_same_v<type, host_set_camera_projection_command>)
                return "{\"projection\":" + quote(to_string(payload.projection)) + '}';
            else if constexpr (std::is_same_v<type, host_viewport_attach_command>)
                return "{\"viewportId\":" + quote(payload.viewport_id) +
                       ",\"nativeHandle\":" + std::to_string(payload.native_handle) +
                       ",\"x\":" + std::to_string(payload.x) + ",\"y\":" + std::to_string(payload.y) +
                       ",\"width\":" + std::to_string(payload.width) + ",\"height\":" + std::to_string(payload.height) +
                       '}';
            else if constexpr (std::is_same_v<type, host_viewport_resize_command>)
                return "{\"viewportId\":" + quote(payload.viewport_id) +
                       ",\"x\":" + std::to_string(payload.x) + ",\"y\":" + std::to_string(payload.y) +
                       ",\"width\":" + std::to_string(payload.width) + ",\"height\":" + std::to_string(payload.height) +
                       '}';
            else if constexpr (std::is_same_v<type, host_viewport_detach_command>)
                return "{\"viewportId\":" + quote(payload.viewport_id) + '}';
            else if constexpr (std::is_same_v<type, host_viewport_set_camera_mode_command>)
                return "{\"viewportId\":" + quote(payload.viewport_id) +
                       ",\"projection\":" + quote(to_string(payload.projection)) + '}';
            else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>)
                return "{\"viewportId\":" + quote(payload.viewport_id) +
                       ",\"renderMode\":" + quote(to_string(payload.render_mode)) +
                       ",\"visualization\":" + quote(to_string(payload.visualization)) +
                       ",\"overlay\":" + quote(to_string(payload.overlay)) +
                       ",\"shadows\":" + bool_json(payload.shadows) + ",\"grid\":" + bool_json(payload.grid) +
                       ",\"realtime\":" + bool_json(payload.realtime) +
                       ",\"cameraSpeed\":" + std::to_string(payload.camera_speed) +
                       ",\"environment\":" + environment_json(payload.environment) + '}';
            else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)
                return "{\"viewportId\":" + quote(payload.viewport_id) +
                       ",\"orbitX\":" + std::to_string(payload.orbit_x) +
                       ",\"orbitY\":" + std::to_string(payload.orbit_y) +
                       ",\"lookX\":" + std::to_string(payload.look_x) + ",\"lookY\":" + std::to_string(payload.look_y) +
                       ",\"panX\":" + std::to_string(payload.pan_x) + ",\"panY\":" + std::to_string(payload.pan_y) +
                       ",\"forward\":" + std::to_string(payload.forward) + ",\"zoom\":" + std::to_string(payload.zoom) +
                       ",\"focusSelected\":" + bool_json(payload.focus_selected) + '}';
            else if constexpr (std::is_same_v<type, host_viewport_set_pose_command>)
                return "{\"viewportId\":" + quote(payload.viewport_id) +
                       ",\"position\":" + vec3_json(payload.position) + ",\"target\":" + vec3_json(payload.target) +
                       '}';
            else if constexpr (std::is_same_v<type, host_runtime_step_command>)
                return "{\"ticks\":" + std::to_string(payload.ticks) + '}';
            else if constexpr (std::is_same_v<type, host_runtime_set_time_scale_command>)
                return "{\"value\":" + std::to_string(payload.value) + '}';
            else if constexpr (std::is_same_v<type, host_runtime_capture_snapshot_command>)
                return "{\"label\":" + quote(payload.label) + '}';
            else if constexpr (std::is_same_v<type, host_runtime_restore_snapshot_command>)
                return "{\"snapshotId\":" + std::to_string(payload.snapshot_id) + '}';
            else if constexpr (std::is_same_v<type, host_history_begin_transaction_command>)
                return "{\"id\":" + std::to_string(payload.id) + ",\"label\":" + quote(payload.label) + '}';
            else if constexpr (std::is_same_v<type, host_history_commit_transaction_command> ||
                               std::is_same_v<type, host_history_cancel_transaction_command>)
                return "{\"id\":" + std::to_string(payload.id) + '}';
            else if constexpr (std::is_same_v<type, host_viewport_set_tool_command>)
                return "{\"tool\":" +
                       quote(payload.tool == host_viewport_tool::translate ? "translate"
                             : payload.tool == host_viewport_tool::rotate  ? "rotate"
                             : payload.tool == host_viewport_tool::scale   ? "scale"
                             : payload.tool == host_viewport_tool::terrain ? "terrain"
                                                                           : "select") +
                       ",\"coordinateSpace\":" +
                       quote(payload.coordinate_space == host_coordinate_space::local ? "local" : "world") +
                       ",\"snapping\":" + bool_json(payload.snapping) +
                       ",\"translationSnap\":" + std::to_string(payload.translation_snap) +
                       ",\"rotationSnapDegrees\":" + std::to_string(payload.rotation_snap_degrees) +
                       ",\"scaleSnap\":" + std::to_string(payload.scale_snap) + '}';
            else if constexpr (std::is_same_v<type, host_viewport_pick_command>)
                return "{\"viewportId\":" + quote(payload.viewport_id) + ",\"x\":" +
                       std::to_string(payload.x) + ",\"y\":" + std::to_string(payload.y) + '}';
            else if constexpr (std::is_same_v<type, host_viewport_capture_command>)
                return "{\"captureId\":" + std::to_string(payload.capture_id) +
                       ",\"color\":" + bool_json(payload.color) + ",\"depth\":" + bool_json(payload.depth) +
                       ",\"objectId\":" + bool_json(payload.object_id) + ",\"normals\":" + bool_json(payload.normals) +
                       ",\"sceneColor\":" + bool_json(payload.scene_color) +
                       ",\"baseColor\":" + bool_json(payload.base_color) +
                       ",\"materialProperties\":" + bool_json(payload.material_properties) +
                       ",\"emissive\":" + bool_json(payload.emissive) +
                       ",\"indirectDiffuse\":" + bool_json(payload.indirect_diffuse) +
                       ",\"reflections\":" + bool_json(payload.reflections) +
                       ",\"traceSource\":" + bool_json(payload.trace_source) +
                       ",\"distanceField\":" + bool_json(payload.distance_field) +
                       ",\"temporalConfidence\":" + bool_json(payload.temporal_confidence) + '}';
            else
                return "{}";
        },
        envelope.payload);

    std::string edit_json;
    if (envelope.edit)
    {
        const char* phase = envelope.edit->phase == host_edit_phase::begin    ? "begin"
                            : envelope.edit->phase == host_edit_phase::update ? "update"
                            : envelope.edit->phase == host_edit_phase::commit ? "commit"
                            : envelope.edit->phase == host_edit_phase::cancel ? "cancel"
                                                                              : "none";
        edit_json = ",\"edit\":{\"id\":" + std::to_string(envelope.edit->id) + ",\"phase\":" + quote(phase) +
                    ",\"label\":" + quote(envelope.edit->label) + '}';
    }
    const std::string revision_json =
        envelope.expected_scene_revision
            ? ",\"expectedSceneRevision\":" + std::to_string(*envelope.expected_scene_revision)
            : std::string{};
    return "{\"kind\":\"command\",\"requestId\":" + std::to_string(envelope.request_id) + ",\"type\":" + quote(type) +
           ",\"payload\":" + payload_json + edit_json + revision_json + '}';
}

std::string to_json(const host_query_envelope& envelope)
{
    const std::string type = envelope.query_type.empty() ? query_type(envelope.payload) : envelope.query_type;
    const std::string payload = std::visit(
        [](const auto& value) -> std::string
        {
            using query = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<query, host_world_environment_query>)
                return "{\"entity\":" + to_json(value.entity) + '}';
            else if constexpr (std::is_same_v<query, host_asset_thumbnail_query>)
                return "{\"path\":" + quote(value.path) + ",\"maxSize\":" + std::to_string(value.max_size) + '}';
            else if constexpr (std::is_same_v<query, host_scene_entities_query>)
                return "{\"search\":" + quote(value.search) + ",\"offset\":" + std::to_string(value.offset) +
                       ",\"limit\":" + std::to_string(value.limit) + '}';
            else if constexpr (std::is_same_v<query, host_entity_by_guid_query>)
                return "{\"guid\":" + quote(value.guid) + '}';
            else if constexpr (std::is_same_v<query, host_scene_spatial_query>)
                return "{\"kind\":" +
                       quote(value.kind == host_spatial_query_kind::raycast  ? "raycast"
                             : value.kind == host_spatial_query_kind::bounds ? "bounds"
                                                                             : "nearby") +
                       ",\"origin\":" + vec3_json(value.origin) + ",\"direction\":" + vec3_json(value.direction) +
                       ",\"center\":" + vec3_json(value.center) + ",\"extent\":" + vec3_json(value.extent) +
                       ",\"radius\":" + std::to_string(value.radius) + ",\"limit\":" + std::to_string(value.limit) +
                       '}';
            else if constexpr (std::is_same_v<query, host_viewport_capture_query>)
                return "{\"captureId\":" + std::to_string(value.capture_id) + '}';
            else
                return "{}";
        },
        envelope.payload);
    return "{\"kind\":\"query\",\"requestId\":" + std::to_string(envelope.request_id) + ",\"type\":" + quote(type) +
           ",\"payload\":" + payload + '}';
}

std::string to_json(const host_response& response)
{
    const std::string payload = response.payload_json.empty() ? "{}" : response.payload_json;
    return "{\"kind\":\"response\",\"requestId\":" + std::to_string(response.request_id) +
           ",\"succeeded\":" + bool_json(response.succeeded) + ",\"error\":" + quote(response.error) +
           ",\"sceneRevision\":" + std::to_string(response.scene_revision) +
           ",\"worldEpoch\":" + std::to_string(response.world_epoch) +
           ",\"frameRevision\":" + std::to_string(response.frame_revision) + ",\"payload\":" + payload + '}';
}

std::string to_json(const host_event& event)
{
    const std::string payload = event.payload_json.empty() ? "{}" : event.payload_json;
    return "{\"kind\":\"event\",\"sequence\":" + std::to_string(event.sequence) +
           ",\"type\":" + quote(to_string(event.event_type)) + ",\"entity\":" + to_json(event.entity) +
           ",\"message\":" + quote(event.message) + ",\"payload\":" + payload + '}';
}

std::string to_json(const host_profiler_snapshot& snapshot)
{
    std::string json = "{\"timestampNanoseconds\":" + std::to_string(snapshot.timestamp_nanoseconds) +
                       ",\"memory\":{\"bytes\":" + std::to_string(snapshot.memory_bytes) +
                       ",\"softLimit\":" + std::to_string(snapshot.memory_soft_limit) +
                       ",\"hardLimit\":" + std::to_string(snapshot.memory_hard_limit) +
                       ",\"pressureEvents\":" + std::to_string(snapshot.memory_pressure_events) + ",\"domains\":[";
    for (std::size_t index = 0; index < snapshot.memory_domains.size(); ++index)
    {
        if (index != 0) json += ',';
        const auto& domain = snapshot.memory_domains[index];
        json += "{\"domain\":" + quote(domain.domain) + ",\"bytes\":" + std::to_string(domain.bytes_outstanding) +
                ",\"peakBytes\":" + std::to_string(domain.peak_bytes) +
                ",\"softLimit\":" + std::to_string(domain.soft_limit) +
                ",\"hardLimit\":" + std::to_string(domain.hard_limit) + ",\"pressure\":" + bool_json(domain.pressure) +
                '}';
    }
    json += "],\"groups\":[";
    for (std::size_t index = 0; index < snapshot.allocation_groups.size(); ++index)
    {
        if (index != 0) json += ',';
        const auto& group = snapshot.allocation_groups[index];
        json += "{\"domain\":" + quote(group.domain) + ",\"tag\":" + quote(group.tag) +
                ",\"worldId\":" + std::to_string(group.world_id) + ",\"threadId\":" + std::to_string(group.thread_id) +
                ",\"stackId\":" + std::to_string(group.stack_id) +
                ",\"allocationCount\":" + std::to_string(group.allocation_count) +
                ",\"bytes\":" + std::to_string(group.bytes_outstanding) + '}';
    }
    json += "]},\"scheduler\":{\"submitted\":" + std::to_string(snapshot.jobs_submitted) +
            ",\"completed\":" + std::to_string(snapshot.jobs_completed) +
            ",\"stolen\":" + std::to_string(snapshot.jobs_stolen) +
            ",\"cancelled\":" + std::to_string(snapshot.jobs_cancelled) +
            ",\"failed\":" + std::to_string(snapshot.jobs_failed) +
            ",\"queued\":" + std::to_string(snapshot.jobs_queued) +
            ",\"droppedEvents\":" + std::to_string(snapshot.dropped_profile_events) + ",\"jobs\":[";
    for (std::size_t index = 0; index < snapshot.jobs.size(); ++index)
    {
        if (index != 0) json += ',';
        const auto& job = snapshot.jobs[index];
        json += "{\"sequence\":" + std::to_string(job.sequence) + ",\"name\":" + quote(job.name) +
                ",\"priority\":" + quote(job.priority) + ",\"affinity\":" + quote(job.affinity) +
                ",\"status\":" + quote(job.status) + ",\"threadId\":" + std::to_string(job.thread_id) +
                ",\"queuedNanoseconds\":" + std::to_string(job.queued_nanoseconds) +
                ",\"startedNanoseconds\":" + std::to_string(job.started_nanoseconds) +
                ",\"completedNanoseconds\":" + std::to_string(job.completed_nanoseconds) + '}';
    }
    json += "]}}";
    return json;
}

std::string to_json(const host_scene_snapshot& snapshot)
{
    std::string json = "{\"sceneGuid\":" + quote(snapshot.scene_guid) + ",\"sceneName\":" + quote(snapshot.scene_name) +
                       ",\"activeScenePath\":" + quote(snapshot.active_scene_path) +
                       ",\"sceneRevision\":" + std::to_string(snapshot.scene_revision) +
                       ",\"worldEpoch\":" + std::to_string(snapshot.world_epoch) +
                       ",\"frameRevision\":" + std::to_string(snapshot.frame_revision) +
                       ",\"totalEntityCount\":" + std::to_string(snapshot.total_entity_count) +
                       ",\"offset\":" + std::to_string(snapshot.offset) +
                       ",\"hasMore\":" + bool_json(snapshot.has_more) + ",\"dirty\":" + bool_json(snapshot.dirty) +
                       ",\"canUndo\":" + bool_json(snapshot.can_undo) + ",\"canRedo\":" + bool_json(snapshot.can_redo) +
                       ",\"undoLabel\":" + quote(snapshot.undo_label) + ",\"redoLabel\":" + quote(snapshot.redo_label) +
                       ",\"entities\":[";
    for (std::size_t index = 0; index < snapshot.entities.size(); ++index)
    {
        const auto& entity = snapshot.entities[index];
        if (index != 0) json += ',';
        json += "{\"entity\":" + to_json(entity.entity) + ",\"guid\":" + quote(entity.guid) +
                ",\"parentGuid\":" + quote(entity.parent_guid) +
                ",\"siblingOrder\":" + std::to_string(entity.sibling_order) + ",\"name\":" + quote(entity.name) +
                ",\"kind\":" + quote(to_string(entity.kind)) + ",\"documentGuid\":" +
                quote(entity.document_guid) + ",\"editorFolder\":" + quote(entity.editor_folder) +
                ",\"collection\":" + quote(entity.collection) + ",\"layer\":" + quote(entity.layer) +
                ",\"active\":" + bool_json(entity.active) + ",\"locked\":" + bool_json(entity.locked) +
                ",\"visible\":" + bool_json(entity.visible) + ",\"pickable\":" + bool_json(entity.pickable) +
                ",\"prefabOverrideCount\":" + std::to_string(entity.prefab_override_count) +
                ",\"selected\":" + bool_json(entity.selected) + '}';
    }
    json += "]}";
    return json;
}

std::string to_json(const host_selected_entity_snapshot& snapshot)
{
    std::string json = "{\"entity\":" + to_json(snapshot.entity) +
                       ",\"selectionCount\":" + std::to_string(snapshot.selection_count) + ",\"selectedGuids\":[";
    for (std::size_t index = 0; index < snapshot.selected_guids.size(); ++index)
    {
        if (index != 0) json += ',';
        json += quote(snapshot.selected_guids[index]);
    }
    json += "],\"guid\":" + quote(snapshot.guid) + ",\"name\":" + quote(snapshot.name) +
            ",\"tag\":" + quote(snapshot.tag) + ",\"active\":" + bool_json(snapshot.active) +
            ",\"renderLayerMask\":" + std::to_string(snapshot.render_layer_mask) +
            ",\"mobility\":" + quote(to_string(snapshot.mobility)) + ",\"transform\":";
    json += snapshot.transform ? to_json(*snapshot.transform) : "null";
    json += ",\"bounds\":";
    if (snapshot.bounds)
        json += "{\"minimum\":" + vec3_json(snapshot.bounds->minimum) +
                ",\"maximum\":" + vec3_json(snapshot.bounds->maximum) + '}';
    else
        json += "null";
    json += ",\"camera\":";
    json += snapshot.camera ? to_json(*snapshot.camera) : "null";
    json += ",\"light\":";
    json += snapshot.light ? to_json(*snapshot.light) : "null";
    json += ",\"meshRenderer\":";
    json += snapshot.mesh_renderer ? to_json(*snapshot.mesh_renderer) : "null";
    json += ",\"terrain\":";
    json += snapshot.terrain ? to_json(*snapshot.terrain) : "null";
    json += ",\"prefab\":";
    if (snapshot.prefab)
    {
        json += "{\"prefabGuid\":" + quote(snapshot.prefab->prefab_guid) +
                ",\"prefabPath\":" + quote(snapshot.prefab->prefab_path) +
                ",\"overrideCount\":" + std::to_string(snapshot.prefab->override_count) +
                ",\"sourceMissing\":" + bool_json(snapshot.prefab->source_missing) + ",\"overrides\":[";
        for (std::size_t index = 0; index < snapshot.prefab->overrides.size(); ++index)
        {
            if (index != 0) json += ',';
            const auto& value = snapshot.prefab->overrides[index];
            json += "{\"sourceEntity\":" + quote(value.source_entity) +
                    ",\"componentId\":" + quote(value.component_id) +
                    ",\"fieldId\":" + std::to_string(value.field_id) +
                    ",\"kind\":" + quote(value.kind) + '}';
        }
        json += "]}";
    }
    else
        json += "null";
    json += ",\"components\":[";
    for (std::size_t index = 0; index < snapshot.components.size(); ++index)
    {
        const auto& component = snapshot.components[index];
        if (index != 0) json += ',';
        json += "{\"kind\":" + quote(to_string(component.kind)) + ",\"typeId\":" + quote(component.type_id) +
                ",\"label\":" + quote(component.label) + ",\"revision\":" + std::to_string(component.revision) +
                ",\"dirtyFields\":" + std::to_string(component.dirty_fields) +
                ",\"editable\":" + bool_json(component.editable) + '}';
    }
    json += "],\"projectComponents\":[";
    for (std::size_t index = 0; index < snapshot.project_components.size(); ++index)
    {
        if (index) json += ',';
        const auto& component = snapshot.project_components[index];
        json += "{\"typeId\":" + quote(component.type_id) + ",\"canonicalName\":" + quote(component.canonical_name) +
                ",\"displayName\":" + quote(component.display_name) +
                ",\"schemaVersion\":" + std::to_string(component.schema_version) +
                ",\"values\":" + component.values_json + '}';
    }
    json += "]}";
    return json;
}

std::string to_json(const host_project_assets_snapshot& snapshot)
{
    std::string json = "{\"projectName\":" + quote(snapshot.project_name) +
                       ",\"projectRoot\":" + quote(snapshot.project_root.generic_string()) +
                       ",\"assetRoot\":" + quote(snapshot.asset_root.generic_string()) +
                       ",\"defaultMeshPath\":" + quote(snapshot.default_mesh_path) +
                       ",\"defaultMeshLoaded\":" + bool_json(snapshot.default_mesh_loaded) +
                       ",\"defaultMeshMessage\":" + quote(snapshot.default_mesh_message) +
                       ",\"cacheRoot\":" + quote(snapshot.cache_root.generic_string()) +
                       ",\"cacheLocalBytes\":" + std::to_string(snapshot.cache_local_bytes) +
                       ",\"cacheLocalHits\":" + std::to_string(snapshot.cache_local_hits) +
                       ",\"cacheLocalMisses\":" + std::to_string(snapshot.cache_local_misses) +
                       ",\"cacheSharedHits\":" + std::to_string(snapshot.cache_shared_hits) +
                       ",\"cacheSharedMisses\":" + std::to_string(snapshot.cache_shared_misses) +
                       ",\"cacheCorruptEntries\":" + std::to_string(snapshot.cache_corrupt_entries) +
                       ",\"cacheEvictions\":" + std::to_string(snapshot.cache_evictions) +
                       ",\"cacheHitRate\":" + std::to_string(snapshot.cache_hit_rate) + ",\"assets\":[";
    for (std::size_t index = 0; index < snapshot.assets.size(); ++index)
    {
        const auto& asset = snapshot.assets[index];
        if (index != 0) json += ',';
        json += "{\"guid\":" + quote(asset.guid) + ",\"path\":" + quote(asset.path) + ",\"kind\":" + quote(asset.kind) +
                ",\"typeId\":" + quote(asset.type_id) + ",\"importerId\":" + quote(asset.importer_id) +
                ",\"state\":" + quote(asset.state) + ",\"residency\":" + quote(asset.residency) +
                ",\"generation\":" + std::to_string(asset.generation) +
                ",\"strongReferences\":" + std::to_string(asset.strong_references) +
                ",\"pins\":" + std::to_string(asset.pins) + ",\"diagnostic\":" + quote(asset.diagnostic) +
                ",\"dependencies\":[";
        for (std::size_t dependency = 0; dependency < asset.dependencies.size(); ++dependency)
        {
            if (dependency != 0) json += ',';
            json += quote(asset.dependencies[dependency]);
        }
        json += "],\"reverseDependencies\":[";
        for (std::size_t dependency = 0; dependency < asset.reverse_dependencies.size(); ++dependency)
        {
            if (dependency != 0) json += ',';
            json += quote(asset.reverse_dependencies[dependency]);
        }
        json += "],\"imported\":" + bool_json(asset.imported) +
                ",\"importRunning\":" + bool_json(asset.import_running) + '}';
    }
    json += "]}";
    return json;
}

std::string to_json(const host_asset_thumbnail_snapshot& snapshot)
{
    return "{\"path\":" + quote(snapshot.path) + ",\"width\":" + std::to_string(snapshot.width) +
           ",\"height\":" + std::to_string(snapshot.height) + ",\"dataUrl\":" + quote(snapshot.data_url) + '}';
}

bool from_json(std::string_view json, host_command_envelope& envelope, std::string& error)
{
    std::string type;
    if (!number_value(json, "requestId", envelope.request_id) || !string_value(json, "type", type))
    {
        error = "Host command envelope requires requestId and type";
        return false;
    }

    std::string_view payload;
    if (!object_value(json, "payload", payload)) payload = "{}";

    if (type == "project.open")
    {
        host_open_project_command command;
        string_value(payload, "name", command.name);
        std::string root;
        string_value(payload, "root", root);
        command.root = root;
        std::string descriptor_path;
        std::string cache_root;
        std::string default_scene;
        string_value(payload, "descriptorPath", descriptor_path);
        string_value(payload, "cacheRoot", cache_root);
        string_value(payload, "defaultScene", default_scene);
        std::string editor_module_path;
        string_value(payload, "projectGuid", command.project_guid);
        string_value(payload, "engineVersion", command.engine_version);
        string_value(payload, "editorModuleId", command.editor_module_id);
        string_value(payload, "editorModulePath", editor_module_path);
        command.descriptor_path = descriptor_path;
        command.cache_root = cache_root;
        command.default_scene = default_scene;
        command.editor_module_path = editor_module_path;
        string_array_value(payload, "contentRoots", command.content_roots);
        bool_value(payload, "readOnly", command.read_only);
        envelope.payload = std::move(command);
    }
    else if (type == "project.close")
    {
        envelope.payload = host_close_project_command{};
    }
    else if (type == "project.reloadModule")
    {
        host_reload_project_module_command command;
        std::string path;
        if (!string_value(payload, "path", path) || path.empty() ||
            !string_value(payload, "engineVersion", command.engine_version) || command.engine_version.empty() ||
            !string_value(payload, "projectGuid", command.project_guid) || command.project_guid.empty() ||
            !string_value(payload, "moduleId", command.module_id) || command.module_id.empty())
        {
            error = "Project module reload requires path and module identity";
            return false;
        }
        command.path = std::move(path);
        envelope.payload = std::move(command);
    }
    else if (type == "scene.open")
    {
        host_open_scene_command command;
        std::string scene_path;
        if (!string_value(payload, "path", scene_path) || scene_path.empty())
        {
            error = "Scene open command requires path";
            return false;
        }
        command.path = scene_path;
        bool_value(payload, "append", command.append);
        envelope.payload = std::move(command);
    }
    else if (type == "scene.new")
    {
        host_new_scene_command command;
        string_value(payload, "name", command.name);
        envelope.payload = std::move(command);
    }
    else if (type == "scene.save")
        envelope.payload = host_save_scene_command{};
    else if (type == "scene.saveAs")
    {
        host_save_scene_as_command command;
        std::string path;
        if (!string_value(payload, "path", path) || path.empty())
        {
            error = "Scene save-as command requires path";
            return false;
        }
        command.path = std::move(path);
        envelope.payload = std::move(command);
    }
    else if (type == "scene.autosave")
    {
        host_autosave_scene_command command;
        std::string path;
        if (!string_value(payload, "path", path) || path.empty())
        {
            error = "Scene autosave command requires path";
            return false;
        }
        command.path = std::move(path);
        envelope.payload = std::move(command);
    }
    else if (type == "scene.openRecovery")
    {
        host_open_recovery_scene_command command;
        std::string recovery_path;
        std::string original_path;
        if (!string_value(payload, "path", recovery_path) || recovery_path.empty())
        {
            error = "Scene recovery command requires path";
            return false;
        }
        string_value(payload, "originalPath", original_path);
        command.path = std::move(recovery_path);
        command.original_path = std::move(original_path);
        envelope.payload = std::move(command);
    }
    else if (type == "asset.reimport" || type == "asset.cancelImport")
    {
        std::string guid;
        if (!string_value(payload, "guid", guid) || guid.empty())
        {
            error = "Asset command requires a GUID";
            return false;
        }
        if (type == "asset.reimport")
            envelope.payload = host_asset_reimport_command{.guid = std::move(guid)};
        else
            envelope.payload = host_asset_cancel_import_command{.guid = std::move(guid)};
    }
    else if (type == "asset.move")
    {
        host_asset_move_command command;
        std::string path;
        if (!string_value(payload, "guid", command.guid) || !string_value(payload, "path", path) || path.empty())
        {
            error = "Asset move requires a GUID and destination path";
            return false;
        }
        command.path = std::move(path);
        envelope.payload = std::move(command);
    }
    else if (type == "asset.rename")
    {
        host_asset_rename_command command;
        if (!string_value(payload, "guid", command.guid) || !string_value(payload, "name", command.name) ||
            command.name.empty())
        {
            error = "Asset rename requires a GUID and filename";
            return false;
        }
        envelope.payload = std::move(command);
    }
    else if (type == "entity.create")
    {
        static constexpr std::pair<std::string_view, host_create_entity_kind> values[]{
            {"empty", host_create_entity_kind::empty},
            {"plane", host_create_entity_kind::plane},
            {"cube", host_create_entity_kind::cube},
            {"sphere", host_create_entity_kind::sphere},
            {"cylinder", host_create_entity_kind::cylinder},
            {"cone", host_create_entity_kind::cone},
            {"capsule", host_create_entity_kind::capsule},
            {"worldEnvironment", host_create_entity_kind::world_environment},
            {"terrain", host_create_entity_kind::terrain},
            {"water", host_create_entity_kind::water},
            {"grassPatch", host_create_entity_kind::grass_patch},
            {"decal", host_create_entity_kind::decal}};
        host_create_entity_command command;
        parse_enum(payload, "kind", values, std::size(values), command.kind);
        entity_field_value(payload, "parent", command.parent);
        envelope.payload = command;
    }
    else if (type == "entity.delete" || type == "entity.select")
    {
        host_entity_id entity;
        if (!entity_field_value(payload, "entity", entity))
        {
            error = "Entity command requires entity";
            return false;
        }
        if (type == "entity.delete")
            envelope.payload = host_delete_entity_command{.entity = entity};
        else
        {
            host_select_entity_command command{.entity = entity};
            bool_value(payload, "additive", command.additive);
            bool_value(payload, "toggle", command.toggle);
            envelope.payload = command;
        }
    }
    else if (type == "entity.duplicate")
    {
        host_duplicate_entity_command command;
        if (!entity_field_value(payload, "entity", command.entity))
        {
            error = "Duplicate command requires entity";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "prefab.create")
    {
        host_create_prefab_command command;
        std::string path;
        if (!entity_field_value(payload, "entity", command.entity) || !string_value(payload, "path", path))
        {
            error = "Prefab create requires entity and path";
            return false;
        }
        command.path = std::filesystem::path(path);
        envelope.payload = std::move(command);
    }
    else if (type == "prefab.instantiate")
    {
        host_instantiate_prefab_command command;
        std::string path;
        if (!string_value(payload, "path", path))
        {
            error = "Prefab instantiate requires path";
            return false;
        }
        command.path = std::filesystem::path(path);
        entity_field_value(payload, "parent", command.parent);
        envelope.payload = std::move(command);
    }
    else if (type == "prefab.apply" || type == "prefab.revert" || type == "prefab.unpack")
    {
        host_entity_id entity;
        if (!entity_field_value(payload, "entity", entity))
        {
            error = "Prefab command requires entity";
            return false;
        }
        if (type == "prefab.apply")
            envelope.payload = host_apply_prefab_command{entity};
        else if (type == "prefab.revert")
            envelope.payload = host_revert_prefab_command{entity};
        else
            envelope.payload = host_unpack_prefab_command{entity};
    }
    else if (type == "entity.reparent")
    {
        host_reparent_entity_command command;
        if (!entity_field_value(payload, "entity", command.entity))
        {
            error = "Reparent command requires entity";
            return false;
        }
        entity_field_value(payload, "parent", command.parent);
        entity_field_value(payload, "beforeSibling", command.before_sibling);
        bool_value(payload, "preserveWorld", command.preserve_world);
        envelope.payload = command;
    }
    else if (type == "entity.reorder")
    {
        host_reorder_entity_command command;
        if (!entity_field_value(payload, "entity", command.entity))
        {
            error = "Reorder command requires entity";
            return false;
        }
        entity_field_value(payload, "beforeSibling", command.before_sibling);
        envelope.payload = command;
    }
    else if (type == "entity.rename")
    {
        host_rename_entity_command command;
        if (!entity_field_value(payload, "entity", command.entity))
        {
            error = "Rename command requires entity";
            return false;
        }
        string_value(payload, "name", command.name);
        envelope.payload = std::move(command);
    }
    else if (type == "entity.clearSelection")
    {
        envelope.payload = host_clear_selection_command{};
    }
    else if (type == "entity.setActive")
    {
        host_set_active_command command;
        if (!entity_field_value(payload, "entity", command.entity))
        {
            error = "Active command requires entity";
            return false;
        }
        bool_value(payload, "active", command.active);
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = command;
    }
    else if (type == "entity.setTag")
    {
        host_set_tag_command command;
        if (!entity_field_value(payload, "entity", command.entity))
        {
            error = "Tag command requires entity";
            return false;
        }
        string_value(payload, "tag", command.tag);
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = std::move(command);
    }
    else if (type == "entity.setTransform")
    {
        host_set_transform_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !transform_value(payload, "transform", command.transform))
        {
            error = "Transform command requires entity and transform";
            return false;
        }
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = command;
    }
    else if (type == "entity.setRenderLayer")
    {
        host_set_render_layer_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !number_value(payload, "renderLayerMask", command.render_layer_mask))
        {
            error = "Render layer command requires entity and renderLayerMask";
            return false;
        }
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = command;
    }
    else if (type == "entity.setMobility")
    {
        host_set_mobility_command command;
        std::string mobility;
        if (!entity_field_value(payload, "entity", command.entity) || !string_value(payload, "mobility", mobility) ||
            (mobility != "static" && mobility != "stationary" && mobility != "movable"))
        {
            error = "Mobility command requires entity and static, stationary, or movable mobility";
            return false;
        }
        command.mobility = mobility == "static"       ? host_mobility::static_object
                           : mobility == "stationary" ? host_mobility::stationary
                                                      : host_mobility::movable;
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = command;
    }
    else if (type == "entity.setCamera")
    {
        host_set_camera_command command;
        if (!entity_field_value(payload, "entity", command.entity) || !camera_value(payload, "camera", command.camera))
        {
            error = "Camera command requires entity and a typed camera snapshot";
            return false;
        }
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = command;
    }
    else if (type == "entity.setLight")
    {
        host_set_light_command command;
        if (!entity_field_value(payload, "entity", command.entity) || !light_value(payload, "light", command.light))
        {
            error = "Light command requires entity and a typed light snapshot";
            return false;
        }
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = command;
    }
    else if (type == "entity.setMeshRenderer")
    {
        host_set_mesh_renderer_command command;
        std::string representation{"auto"};
        if (!entity_field_value(payload, "entity", command.entity) ||
            !bool_value(payload, "visible", command.visible) ||
            !bool_value(payload, "castsShadows", command.casts_shadows) ||
            !bool_value(payload, "receivesShadows", command.receives_shadows) ||
            !number_value(payload, "shadowLodBias", command.shadow_lod_bias) ||
            !number_value(payload, "maximumShadowDistance", command.maximum_shadow_distance) ||
            !array4_value(payload, "baseColorTint", command.base_color_tint))
        {
            error = "Mesh renderer command requires entity, visible, and baseColorTint";
            return false;
        }
        string_value(payload, "representation", representation);
        if (representation == "auto")
            command.representation = 0;
        else if (representation == "conventional")
            command.representation = 1;
        else if (representation == "virtualized")
            command.representation = 2;
        else
        {
            error = "Mesh renderer command has an invalid geometry representation";
            return false;
        }
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = command;
    }
    else if (type == "terrain.update")
    {
        host_set_terrain_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !bool_value(payload, "enabled", command.enabled) ||
            !bool_value(payload, "receiveShadows", command.receive_shadows) ||
            !bool_value(payload, "castShadows", command.cast_shadows) ||
            !number_value(payload, "patchQuads", command.patch_quads) ||
            !number_value(payload, "maximumHierarchyDepth", command.maximum_hierarchy_depth) ||
            !number_value(payload, "geometricErrorMultiplier", command.geometric_error_multiplier) ||
            !number_value(payload, "shadowLodBias", command.shadow_lod_bias) ||
            !number_value(payload, "maximumShadowDistance", command.maximum_shadow_distance))
        {
            error = "Terrain update requires entity, enabled, and receiveShadows";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "terrain.setBrush")
    {
        static constexpr std::pair<std::string_view, host_terrain_brush_tool> tools[]{
            {"sculpt", host_terrain_brush_tool::sculpt},
            {"smooth", host_terrain_brush_tool::smooth},
            {"flatten", host_terrain_brush_tool::flatten},
            {"paint", host_terrain_brush_tool::paint}};
        host_set_terrain_brush_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !number_value(payload, "radius", command.radius) || !number_value(payload, "strength", command.strength) ||
            !number_value(payload, "falloff", command.falloff) ||
            !number_value(payload, "activeLayer", command.active_layer))
        {
            error = "Terrain brush requires entity and complete brush settings";
            return false;
        }
        parse_enum(payload, "tool", tools, std::size(tools), command.tool);
        envelope.payload = command;
    }
    else if (type == "terrain.stroke")
    {
        host_terrain_stroke_command command;
        std::string phase;
        if (!entity_field_value(payload, "entity", command.entity) || !number_value(payload, "x", command.x) ||
            !number_value(payload, "y", command.y) || !string_value(payload, "phase", phase))
        {
            error = "Terrain stroke requires entity, viewport coordinates, and phase";
            return false;
        }
        command.phase = phase == "update"   ? host_edit_phase::update
                        : phase == "commit" ? host_edit_phase::commit
                        : phase == "cancel" ? host_edit_phase::cancel
                                            : host_edit_phase::begin;
        bool_value(payload, "invert", command.invert);
        envelope.payload = command;
    }
    else if (type == "terrain.hover")
    {
        host_terrain_hover_command command;
        bool_value(payload, "clear", command.clear);
        if (!entity_field_value(payload, "entity", command.entity) ||
            (!command.clear && (!number_value(payload, "x", command.x) || !number_value(payload, "y", command.y))))
        {
            error = "Terrain hover requires entity and viewport coordinates";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "terrain.assignLayer")
    {
        host_set_terrain_layer_command command;
        std::string path;
        if (!entity_field_value(payload, "entity", command.entity) || !number_value(payload, "layer", command.layer) ||
            !string_value(payload, "path", path))
        {
            error = "Terrain layer assignment requires entity, layer, and path";
            return false;
        }
        command.path = std::move(path);
        envelope.payload = std::move(command);
    }
    else if (type == "entity.setMaterial")
    {
        host_set_entity_material_command command;
        std::string path;
        if (!entity_field_value(payload, "entity", command.entity) || !string_value(payload, "path", path) ||
            path.empty())
        {
            error = "Material assignment requires entity and material path";
            return false;
        }
        command.path = std::move(path);
        bool_value(payload, "applyToSelection", command.apply_to_selection);
        envelope.payload = std::move(command);
    }
    else if (type == "component.add" || type == "component.remove" || type == "component.reset")
    {
        host_component_operation_command command;
        if (!string_value(payload, "component", command.component) || command.component.empty())
        {
            error = "Component operation requires a component name";
            return false;
        }
        command.operation = type == "component.add"      ? host_component_operation::add
                            : type == "component.remove" ? host_component_operation::remove
                                                         : host_component_operation::reset;
        envelope.payload = std::move(command);
    }
    else if (type == "component.patchField")
    {
        host_patch_project_component_command command;
        const auto document = nlohmann::json::parse(payload, nullptr, false);
        if (!document.is_object() || !document.contains("value") ||
            !string_value(payload, "component", command.component) || command.component.empty() ||
            !string_value(payload, "field", command.field) || command.field.empty())
        {
            error = "Project component patch requires component, field, and value";
            return false;
        }
        command.value_json = document.at("value").dump();
        envelope.payload = std::move(command);
    }
    else if (type == "environment.update")
    {
        host_set_world_environment_command command;
        if (!parse_world_environment(payload, command.environment))
        {
            error = "Environment update requires a typed environment snapshot";
            return false;
        }
        envelope.payload = std::move(command);
    }
    else if (type == "environment.applyPreset")
    {
        static constexpr std::pair<std::string_view, host_world_environment_preset> presets[]{
            {"clearDay", host_world_environment_preset::clear_day},
            {"alpineLateMorning", host_world_environment_preset::alpine_late_morning},
            {"goldenHour", host_world_environment_preset::golden_hour},
            {"overcast", host_world_environment_preset::overcast},
            {"night", host_world_environment_preset::night},
            {"indoorNeutral", host_world_environment_preset::indoor_neutral}};
        host_apply_world_environment_preset_command command;
        if (!entity_field_value(payload, "entity", command.entity))
        {
            error = "Environment preset requires entity";
            return false;
        }
        parse_enum(payload, "preset", presets, std::size(presets), command.preset);
        envelope.payload = command;
    }
    else if (type == "environment.setHdri")
    {
        host_set_environment_hdri_command command;
        std::string path;
        if (!entity_field_value(payload, "entity", command.entity) || !string_value(payload, "path", path))
        {
            error = "Environment HDRI assignment requires entity and path";
            return false;
        }
        command.path = path;
        envelope.payload = std::move(command);
    }
    else if (type == "camera.setProjection" || type == "viewport.setCameraMode")
    {
        static constexpr std::pair<std::string_view, host_camera_projection> values[]{
            {"perspective", host_camera_projection::perspective},
            {"orthographic", host_camera_projection::orthographic}};
        host_camera_projection projection{host_camera_projection::perspective};
        parse_enum(payload, "projection", values, std::size(values), projection);
        if (type == "camera.setProjection")
            envelope.payload = host_set_camera_projection_command{.projection = projection};
        else
        {
            host_viewport_set_camera_mode_command command{.projection = projection};
            string_value(payload, "viewportId", command.viewport_id);
            envelope.payload = std::move(command);
        }
    }
    else if (type == "viewport.attach")
    {
        host_viewport_attach_command command;
        string_value(payload, "viewportId", command.viewport_id);
        number_value(payload, "nativeHandle", command.native_handle);
        number_value(payload, "x", command.x);
        number_value(payload, "y", command.y);
        number_value(payload, "width", command.width);
        number_value(payload, "height", command.height);
        envelope.payload = command;
    }
    else if (type == "viewport.resize")
    {
        host_viewport_resize_command command;
        string_value(payload, "viewportId", command.viewport_id);
        number_value(payload, "x", command.x);
        number_value(payload, "y", command.y);
        number_value(payload, "width", command.width);
        number_value(payload, "height", command.height);
        envelope.payload = command;
    }
    else if (type == "prefab.revertOverride")
    {
        host_revert_prefab_override_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !string_value(payload, "sourceEntity", command.source_entity) ||
            !string_value(payload, "componentId", command.component_id))
        {
            error = "Prefab override revert requires entity, source entity, and component ID";
            return false;
        }
        number_value(payload, "fieldId", command.field_id);
        string_value(payload, "kind", command.kind);
        envelope.payload = std::move(command);
    }
    else if (type == "viewport.detach")
    {
        host_viewport_detach_command command;
        string_value(payload, "viewportId", command.viewport_id);
        envelope.payload = std::move(command);
    }
    else if (type == "viewport.setRenderOptions")
    {
        static constexpr std::pair<std::string_view, host_render_mode> render_modes[]{
            {"shaded", host_render_mode::shaded}, {"wireframe", host_render_mode::wireframe}};
        static constexpr std::pair<std::string_view, host_visualization_mode> visualizations[]{
            {"standard", host_visualization_mode::standard},
            {"albedo", host_visualization_mode::albedo},
            {"opacity", host_visualization_mode::opacity},
            {"worldNormal", host_visualization_mode::world_normal},
            {"specularity", host_visualization_mode::specularity},
            {"gloss", host_visualization_mode::gloss},
            {"metalness", host_visualization_mode::metalness},
            {"ao", host_visualization_mode::ao},
            {"emission", host_visualization_mode::emission},
            {"lighting", host_visualization_mode::lighting},
            {"uv0", host_visualization_mode::uv0},
            {"cascadeDebug", host_visualization_mode::cascade_debug},
            {"shadowMask", host_visualization_mode::shadow_mask},
            {"lightComplexity", host_visualization_mode::light_complexity},
            {"clusterDebug", host_visualization_mode::cluster_debug},
            {"virtualHierarchyLevel", host_visualization_mode::virtual_hierarchy_level},
            {"virtualGeometricError", host_visualization_mode::virtual_geometric_error},
            {"virtualPageResidency", host_visualization_mode::virtual_page_residency},
            {"virtualOverdraw", host_visualization_mode::virtual_overdraw},
            {"virtualTrianglesPerPixel", host_visualization_mode::virtual_triangles_per_pixel},
            {"surfaceCards", host_visualization_mode::surface_cards},
            {"surfaceCardResidency", host_visualization_mode::surface_card_residency},
            {"surfaceMaterialCache", host_visualization_mode::surface_material_cache},
            {"surfaceRadianceCache", host_visualization_mode::surface_radiance_cache},
            {"meshDistanceFields", host_visualization_mode::mesh_distance_fields},
            {"globalDistanceField", host_visualization_mode::global_distance_field},
            {"radianceProbes", host_visualization_mode::radiance_probes},
            {"lightingTraceSource", host_visualization_mode::lighting_trace_source},
            {"lightingHitDistance", host_visualization_mode::lighting_hit_distance},
            {"lightingTemporalConfidence", host_visualization_mode::lighting_temporal_confidence},
            {"indirectDiffuse", host_visualization_mode::indirect_diffuse},
            {"reflections", host_visualization_mode::reflections},
            {"denoiserVariance", host_visualization_mode::denoiser_variance}};
        static constexpr std::pair<std::string_view, host_overlay_mode> overlays[]{
            {"none", host_overlay_mode::none},
            {"selectedWireframe", host_overlay_mode::selected_wireframe},
            {"allWireframe", host_overlay_mode::all_wireframe}};
        host_viewport_set_render_options_command command;
        string_value(payload, "viewportId", command.viewport_id);
        parse_enum(payload, "renderMode", render_modes, std::size(render_modes), command.render_mode);
        parse_enum(payload, "visualization", visualizations, std::size(visualizations), command.visualization);
        parse_enum(payload, "overlay", overlays, std::size(overlays), command.overlay);
        bool_value(payload, "shadows", command.shadows);
        bool_value(payload, "grid", command.grid);
        bool_value(payload, "realtime", command.realtime);
        number_value(payload, "cameraSpeed", command.camera_speed);
        command.camera_speed = std::clamp(command.camera_speed, 0.25f, 16.0f);
        parse_environment(payload, command.environment);
        envelope.payload = command;
    }
    else if (type == "viewport.cameraInput")
    {
        host_viewport_camera_input_command command;
        string_value(payload, "viewportId", command.viewport_id);
        number_value(payload, "orbitX", command.orbit_x);
        number_value(payload, "orbitY", command.orbit_y);
        number_value(payload, "lookX", command.look_x);
        number_value(payload, "lookY", command.look_y);
        number_value(payload, "panX", command.pan_x);
        number_value(payload, "panY", command.pan_y);
        number_value(payload, "forward", command.forward);
        number_value(payload, "zoom", command.zoom);
        bool_value(payload, "focusSelected", command.focus_selected);
        envelope.payload = command;
    }
    else if (type == "viewport.setPose")
    {
        host_viewport_set_pose_command command;
        string_value(payload, "viewportId", command.viewport_id);
        if (!array3_value(payload, "position", command.position) || !array3_value(payload, "target", command.target))
        {
            error = "Viewport pose requires position and target vectors";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "history.undo")
        envelope.payload = host_history_undo_command{};
    else if (type == "history.redo")
        envelope.payload = host_history_redo_command{};
    else if (type == "history.beginTransaction")
    {
        host_history_begin_transaction_command command;
        if (!number_value(payload, "id", command.id) || command.id == 0)
        {
            error = "History begin transaction requires a non-zero id";
            return false;
        }
        string_value(payload, "label", command.label);
        envelope.payload = std::move(command);
    }
    else if (type == "history.commitTransaction" || type == "history.cancelTransaction")
    {
        std::uint64_t id{};
        if (!number_value(payload, "id", id) || id == 0)
        {
            error = "History transaction command requires a non-zero id";
            return false;
        }
        if (type == "history.commitTransaction")
            envelope.payload = host_history_commit_transaction_command{id};
        else
            envelope.payload = host_history_cancel_transaction_command{id};
    }
    else if (type == "runtime.resume")
        envelope.payload = host_runtime_resume_command{};
    else if (type == "runtime.pause")
        envelope.payload = host_runtime_pause_command{};
    else if (type == "runtime.stop")
        envelope.payload = host_runtime_stop_command{};
    else if (type == "runtime.step")
    {
        host_runtime_step_command command;
        number_value(payload, "ticks", command.ticks);
        command.ticks = std::clamp(command.ticks, 1u, 1024u);
        envelope.payload = command;
    }
    else if (type == "runtime.setTimeScale")
    {
        host_runtime_set_time_scale_command command;
        if (!number_value(payload, "value", command.value))
        {
            error = "Runtime time-scale command requires value";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "runtime.captureSnapshot")
    {
        host_runtime_capture_snapshot_command command;
        string_value(payload, "label", command.label);
        envelope.payload = std::move(command);
    }
    else if (type == "runtime.restoreSnapshot")
    {
        host_runtime_restore_snapshot_command command;
        if (!number_value(payload, "snapshotId", command.snapshot_id) || command.snapshot_id == 0)
        {
            error = "Runtime snapshot restore requires snapshotId";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "viewport.setTool")
    {
        static constexpr std::pair<std::string_view, host_viewport_tool> tools[]{
            {"select", host_viewport_tool::select},
            {"translate", host_viewport_tool::translate},
            {"rotate", host_viewport_tool::rotate},
            {"scale", host_viewport_tool::scale},
            {"terrain", host_viewport_tool::terrain}};
        static constexpr std::pair<std::string_view, host_coordinate_space> spaces[]{
            {"world", host_coordinate_space::world}, {"local", host_coordinate_space::local}};
        host_viewport_set_tool_command command;
        parse_enum(payload, "tool", tools, std::size(tools), command.tool);
        parse_enum(payload, "coordinateSpace", spaces, std::size(spaces), command.coordinate_space);
        bool_value(payload, "snapping", command.snapping);
        number_value(payload, "translationSnap", command.translation_snap);
        number_value(payload, "rotationSnapDegrees", command.rotation_snap_degrees);
        number_value(payload, "scaleSnap", command.scale_snap);
        envelope.payload = command;
    }
    else if (type == "viewport.pick")
    {
        host_viewport_pick_command command;
        string_value(payload, "viewportId", command.viewport_id);
        number_value(payload, "x", command.x);
        number_value(payload, "y", command.y);
        envelope.payload = command;
    }
    else if (type == "viewport.capture")
    {
        host_viewport_capture_command command;
        number_value(payload, "captureId", command.capture_id);
        bool_value(payload, "color", command.color);
        bool_value(payload, "depth", command.depth);
        bool_value(payload, "objectId", command.object_id);
        bool_value(payload, "normals", command.normals);
        bool_value(payload, "sceneColor", command.scene_color);
        bool_value(payload, "baseColor", command.base_color);
        bool_value(payload, "materialProperties", command.material_properties);
        bool_value(payload, "emissive", command.emissive);
        bool_value(payload, "indirectDiffuse", command.indirect_diffuse);
        bool_value(payload, "reflections", command.reflections);
        bool_value(payload, "traceSource", command.trace_source);
        bool_value(payload, "distanceField", command.distance_field);
        bool_value(payload, "temporalConfidence", command.temporal_confidence);
        envelope.payload = command;
    }
    else
    {
        error = "Unsupported host command type: " + type;
        return false;
    }

    std::string_view edit;
    if (object_value(json, "edit", edit))
    {
        host_edit_transaction transaction;
        number_value(edit, "id", transaction.id);
        string_value(edit, "label", transaction.label);
        std::string phase;
        string_value(edit, "phase", phase);
        transaction.phase = phase == "begin"    ? host_edit_phase::begin
                            : phase == "update" ? host_edit_phase::update
                            : phase == "commit" ? host_edit_phase::commit
                            : phase == "cancel" ? host_edit_phase::cancel
                                                : host_edit_phase::none;
        if (transaction.id != 0 && transaction.phase != host_edit_phase::none) envelope.edit = std::move(transaction);
    }
    std::uint64_t expected_revision{};
    if (number_value(json, "expectedSceneRevision", expected_revision))
        envelope.expected_scene_revision = expected_revision;
    envelope.command_type = std::move(type);
    return true;
}

bool from_json(std::string_view json, host_query_envelope& envelope, std::string& error)
{
    std::string type;
    if (!number_value(json, "requestId", envelope.request_id) || !string_value(json, "type", type))
    {
        error = "Host query envelope requires requestId and type";
        return false;
    }

    if (type == "scene.hierarchy")
        envelope.payload = host_scene_hierarchy_query{};
    else if (type == "entity.selected")
        envelope.payload = host_selected_entity_query{};
    else if (type == "gateway.sceneEntities")
    {
        host_scene_entities_query query;
        string_value(json, "search", query.search);
        number_value(json, "offset", query.offset);
        number_value(json, "limit", query.limit);
        query.limit = std::clamp<std::size_t>(query.limit, 1u, 200u);
        envelope.payload = std::move(query);
    }
    else if (type == "gateway.entity")
    {
        host_entity_by_guid_query query;
        if (!string_value(json, "guid", query.guid) || query.guid.empty())
        {
            error = "Gateway entity query requires a valid GUID";
            return false;
        }
        envelope.payload = std::move(query);
    }
    else if (type == "gateway.spatialQuery")
    {
        host_scene_spatial_query query;
        std::string kind;
        string_value(json, "kind", kind);
        query.kind = kind == "raycast"   ? host_spatial_query_kind::raycast
                     : kind == "bounds"  ? host_spatial_query_kind::bounds
                     : kind == "frustum" ? host_spatial_query_kind::frustum
                                         : host_spatial_query_kind::nearby;
        array3_value(json, "origin", query.origin);
        array3_value(json, "direction", query.direction);
        array3_value(json, "center", query.center);
        array3_value(json, "extent", query.extent);
        number_value(json, "radius", query.radius);
        number_value(json, "limit", query.limit);
        query.limit = std::clamp<std::size_t>(query.limit, 1u, 500u);
        envelope.payload = query;
    }
    else if (type == "gateway.componentSchemas")
        envelope.payload = host_component_schema_query{};
    else if (type == "workspace.documents")
        envelope.payload = host_workspace_documents_query{};
    else if (type == "gateway.diagnostics")
        envelope.payload = host_gateway_diagnostics_query{};
    else if (type == "viewport.captureResult")
    {
        host_viewport_capture_query capture;
        if (!number_value(json, "captureId", capture.capture_id) || capture.capture_id == 0)
        {
            error = "viewport.captureResult requires a non-zero captureId";
            return false;
        }
        envelope.payload = capture;
    }
    else if (type == "project.assets")
        envelope.payload = host_project_assets_query{};
    else if (type == "asset.thumbnail")
    {
        host_asset_thumbnail_query thumbnail;
        if (!string_value(json, "path", thumbnail.path) || thumbnail.path.empty())
        {
            error = "Asset thumbnail query requires a path";
            return false;
        }
        number_value(json, "maxSize", thumbnail.max_size);
        thumbnail.max_size = std::clamp(thumbnail.max_size, 32u, 256u);
        envelope.payload = std::move(thumbnail);
    }
    else if (type == "viewport.state")
    {
        host_viewport_state_query query;
        string_value(json, "viewportId", query.viewport_id);
        envelope.payload = std::move(query);
    }
    else if (type == "environment.state")
    {
        host_entity_id entity;
        if (!entity_field_value(json, "entity", entity))
        {
            error = "Environment query requires entity";
            return false;
        }
        envelope.payload = host_world_environment_query{.entity = entity};
    }
    else if (type == "history.state")
        envelope.payload = host_history_state_query{};
    else if (type == "runtime.state")
        envelope.payload = host_runtime_state_query{};
    else if (type == "terrain.toolState")
        envelope.payload = host_terrain_tool_state_query{};
    else
    {
        error = "Unsupported host query type: " + type;
        return false;
    }
    envelope.query_type = std::move(type);
    return true;
}

bool from_json(std::string_view json, host_world_environment_snapshot& environment, std::string& error)
{
    if (!parse_world_environment(json, environment, false))
    {
        error = "invalid world environment snapshot";
        return false;
    }
    error.clear();
    return true;
}

} // namespace arc::editor
