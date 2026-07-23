#include <arc/editor/host_protocol.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <iterator>
#include <sstream>
#include <type_traits>

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

template <class Enum>
bool enum_from_string(std::string_view text, const std::pair<std::string_view, Enum>* values, std::size_t count, Enum& out)
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
    if (key_pos == std::string_view::npos)
        return std::string_view::npos;
    const auto colon = json.find(':', key_pos + needle.size());
    if (colon == std::string_view::npos)
        return std::string_view::npos;
    return colon + 1;
}

std::size_t skip_ws(std::string_view json, std::size_t pos)
{
    while (pos < json.size())
    {
        const char ch = json[pos];
        if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t')
            break;
        ++pos;
    }
    return pos;
}

bool string_value(std::string_view json, std::string_view key, std::string& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '"')
        return false;
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

template <class Number>
bool number_value(std::string_view json, std::string_view key, Number& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size())
        return false;

    const auto start = pos;
    while (pos < json.size())
    {
        const char ch = json[pos];
        if ((ch < '0' || ch > '9') && ch != '-' && ch != '+' && ch != '.' && ch != 'e' && ch != 'E')
            break;
        ++pos;
    }
    const auto token = json.substr(start, pos - start);
    if constexpr (std::is_floating_point_v<Number>)
    {
        std::istringstream stream{ std::string(token) };
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
    if (pos == std::string_view::npos || pos >= json.size())
        return false;
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
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '{')
        return false;

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
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '[')
        return false;
    ++pos;

    std::array<float, 3> values{};
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        pos = skip_ws(json, pos);
        const auto start = pos;
        while (pos < json.size() && json[pos] != ',' && json[pos] != ']')
            ++pos;
        std::istringstream stream{ std::string(json.substr(start, pos - start)) };
        stream >> values[index];
        if (stream.fail())
            return false;
        pos = skip_ws(json, pos);
        if (index + 1 < values.size())
        {
            if (pos >= json.size() || json[pos] != ',')
                return false;
            ++pos;
        }
    }
    out = { values[0], values[1], values[2] };
    return true;
}

bool quat_value(std::string_view json, std::string_view key, host_quat& out)
{
    auto pos = skip_ws(json, find_key(json, key));
    if (pos == std::string_view::npos || pos >= json.size() || json[pos] != '[')
        return false;
    ++pos;

    std::array<float, 4> values{};
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        pos = skip_ws(json, pos);
        const auto start = pos;
        while (pos < json.size() && json[pos] != ',' && json[pos] != ']')
            ++pos;
        std::istringstream stream{ std::string(json.substr(start, pos - start)) };
        stream >> values[index];
        if (stream.fail())
            return false;
        pos = skip_ws(json, pos);
        if (index + 1 < values.size())
        {
            if (pos >= json.size() || json[pos] != ',')
                return false;
            ++pos;
        }
    }
    out = { values[0], values[1], values[2], values[3] };
    return true;
}

bool array4_value(std::string_view json, std::string_view key, host_vec4& out)
{
    host_quat value;
    if (!quat_value(json, key, value))
        return false;
    out = { value.x, value.y, value.z, value.w };
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
    if (!object_value(json, key, object))
        return false;
    return array3_value(object, "position", out.position) &&
        quat_value(object, "rotation", out.rotation) &&
        array3_value(object, "scale", out.scale);
}

bool camera_value(std::string_view json, std::string_view key, host_camera_snapshot& out)
{
    std::string_view object;
    if (!object_value(json, key, object))
        return false;
    std::string projection;
    if (!string_value(object, "projection", projection) ||
        (projection != "perspective" && projection != "orthographic"))
        return false;
    out.projection = projection == "orthographic"
        ? host_camera_projection::orthographic
        : host_camera_projection::perspective;
    return
        number_value(object, "fovYDegrees", out.fov_y_degrees) &&
        number_value(object, "orthographicHeight", out.orthographic_height) &&
        number_value(object, "nearPlane", out.near_plane) &&
        number_value(object, "farPlane", out.far_plane) &&
        bool_value(object, "active", out.active) &&
        array4_value(object, "clearColor", out.clear_color);
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
    return std::string("{\"sky\":") + bool_json(value.sky) +
        ",\"fog\":" + bool_json(value.fog) +
        ",\"terrain\":" + bool_json(value.terrain) +
        ",\"water\":" + bool_json(value.water) +
        ",\"vegetation\":" + bool_json(value.vegetation) +
        ",\"decals\":" + bool_json(value.decals) + '}';
}

template <class Enum>
bool parse_enum(std::string_view payload, std::string_view key, const std::pair<std::string_view, Enum>* values, std::size_t count, Enum& out);

bool parse_environment(std::string_view payload, host_environment_visibility& out)
{
    std::string_view object;
    if (!object_value(payload, "environment", object))
        return true;
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
    stream << "{\"enabled\":" << bool_json(value.enabled)
        << ",\"coverage\":" << value.coverage
        << ",\"density\":" << value.density
        << ",\"altitude\":" << value.altitude
        << ",\"thickness\":" << value.thickness
        << ",\"scale\":" << value.scale
        << ",\"detail\":" << value.detail
        << ",\"softness\":" << value.softness
        << ",\"windX\":" << value.wind_x
        << ",\"windY\":" << value.wind_y
        << ",\"windSpeed\":" << value.wind_speed
        << ",\"lightingStrength\":" << value.lighting_strength
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

bool parse_world_environment(std::string_view payload, host_world_environment_snapshot& value, bool require_entity = true)
{
    std::string_view json;
    if (!object_value(payload, "environment", json))
        json = payload;
    entity_field_value(json, "entity", value.entity);
    bool_value(json, "enabled", value.enabled);
    bool_value(json, "skyVisible", value.sky_visible);
    bool_value(json, "affectLighting", value.affect_lighting);
    static constexpr std::pair<std::string_view, host_sky_source> sky_sources[]{
        { "physicalAtmosphere", host_sky_source::physical_atmosphere },
        { "hdri", host_sky_source::hdri },
        { "solidColor", host_sky_source::solid_color }
    };
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
        { "manualLight", host_sun_position_mode::manual_light },
        { "geographic", host_sun_position_mode::geographic }
    };
    static constexpr std::pair<std::string_view, host_celestial_time_mode> time_modes[]{
        { "fixed", host_celestial_time_mode::fixed },
        { "simulated", host_celestial_time_mode::simulated },
        { "systemClock", host_celestial_time_mode::system_clock }
    };
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
        { "followSky", host_environment_lighting_source::follow_sky },
        { "hdri", host_environment_lighting_source::hdri },
        { "constantColor", host_environment_lighting_source::constant_color }
    };
    parse_enum(json, "lightingSource", lighting_sources, std::size(lighting_sources), value.lighting_source);
    array3_value(json, "lightingColor", value.lighting_color);
    number_value(json, "diffuseIntensity", value.diffuse_intensity);
    number_value(json, "specularIntensity", value.specular_intensity);
    return !require_entity || value.entity.valid();
}

template <class Enum>
bool parse_enum(std::string_view payload, std::string_view key, const std::pair<std::string_view, Enum>* values, std::size_t count, Enum& out)
{
    std::string text;
    return string_value(payload, key, text) && enum_from_string(std::string_view(text), values, count, out);
}

} // namespace

const char* to_string(host_event_type value) noexcept
{
    switch (value)
    {
    case host_event_type::host_started: return "host.started";
    case host_event_type::host_shutdown: return "host.shutdown";
    case host_event_type::project_opened: return "project.opened";
    case host_event_type::project_closed: return "project.closed";
    case host_event_type::scene_changed: return "scene.changed";
    case host_event_type::entity_created: return "entity.created";
    case host_event_type::entity_deleted: return "entity.deleted";
    case host_event_type::entity_selected: return "entity.selected";
    case host_event_type::component_changed: return "component.changed";
    case host_event_type::command_failed: return "command.failed";
    case host_event_type::viewport_error: return "viewport.error";
    case host_event_type::profiler_snapshot: return "profiler.snapshot";
    }
    return "unknown";
}

const char* to_string(host_entity_kind value) noexcept
{
    switch (value)
    {
    case host_entity_kind::camera: return "camera";
    case host_entity_kind::light: return "light";
    case host_entity_kind::environment: return "environment";
    case host_entity_kind::mesh: return "mesh";
    case host_entity_kind::primitive: return "primitive";
    case host_entity_kind::imported: return "imported";
    case host_entity_kind::unknown: return "unknown";
    }
    return "unknown";
}

const char* to_string(host_component_kind value) noexcept
{
    switch (value)
    {
    case host_component_kind::transform: return "transform";
    case host_component_kind::camera: return "camera";
    case host_component_kind::mesh_renderer: return "meshRenderer";
    case host_component_kind::directional_light: return "directionalLight";
    case host_component_kind::point_light: return "pointLight";
    case host_component_kind::spot_light: return "spotLight";
    case host_component_kind::world_environment: return "worldEnvironment";
    case host_component_kind::sky_atmosphere: return "skyAtmosphere";
    case host_component_kind::celestial_sky: return "celestialSky";
    case host_component_kind::cloud_layers: return "cloudLayers";
    case host_component_kind::environment_lighting: return "environmentLighting";
    case host_component_kind::height_fog: return "heightFog";
    case host_component_kind::terrain: return "terrain";
    case host_component_kind::water: return "water";
    case host_component_kind::vegetation: return "vegetation";
    case host_component_kind::decal: return "decal";
    }
    return "unknown";
}

const char* to_string(host_create_entity_kind value) noexcept
{
    switch (value)
    {
    case host_create_entity_kind::empty: return "empty";
    case host_create_entity_kind::plane: return "plane";
    case host_create_entity_kind::cube: return "cube";
    case host_create_entity_kind::sphere: return "sphere";
    case host_create_entity_kind::cylinder: return "cylinder";
    case host_create_entity_kind::world_environment: return "worldEnvironment";
    case host_create_entity_kind::terrain: return "terrain";
    case host_create_entity_kind::water: return "water";
    case host_create_entity_kind::grass_patch: return "grassPatch";
    case host_create_entity_kind::decal: return "decal";
    }
    return "cube";
}

const char* to_string(host_camera_projection value) noexcept
{
    return value == host_camera_projection::orthographic ? "orthographic" : "perspective";
}

const char* to_string(host_render_mode value) noexcept
{
    return value == host_render_mode::wireframe ? "wireframe" : "shaded";
}

const char* to_string(host_visualization_mode value) noexcept
{
    switch (value)
    {
    case host_visualization_mode::standard: return "standard";
    case host_visualization_mode::albedo: return "albedo";
    case host_visualization_mode::opacity: return "opacity";
    case host_visualization_mode::world_normal: return "worldNormal";
    case host_visualization_mode::specularity: return "specularity";
    case host_visualization_mode::gloss: return "gloss";
    case host_visualization_mode::metalness: return "metalness";
    case host_visualization_mode::ao: return "ao";
    case host_visualization_mode::emission: return "emission";
    case host_visualization_mode::lighting: return "lighting";
    case host_visualization_mode::uv0: return "uv0";
    case host_visualization_mode::cascade_debug: return "cascadeDebug";
    case host_visualization_mode::shadow_mask: return "shadowMask";
    case host_visualization_mode::light_complexity: return "lightComplexity";
    case host_visualization_mode::cluster_debug: return "clusterDebug";
    }
    return "standard";
}

const char* to_string(host_overlay_mode value) noexcept
{
    switch (value)
    {
    case host_overlay_mode::none: return "none";
    case host_overlay_mode::selected_wireframe: return "selectedWireframe";
    case host_overlay_mode::all_wireframe: return "allWireframe";
    }
    return "selectedWireframe";
}

const char* to_string(host_sky_source value) noexcept
{
    switch (value)
    {
    case host_sky_source::physical_atmosphere: return "physicalAtmosphere";
    case host_sky_source::hdri: return "hdri";
    case host_sky_source::solid_color: return "solidColor";
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
    case host_celestial_time_mode::fixed: return "fixed";
    case host_celestial_time_mode::simulated: return "simulated";
    case host_celestial_time_mode::system_clock: return "systemClock";
    }
    return "fixed";
}

const char* to_string(host_environment_lighting_source value) noexcept
{
    switch (value)
    {
    case host_environment_lighting_source::follow_sky: return "followSky";
    case host_environment_lighting_source::hdri: return "hdri";
    case host_environment_lighting_source::constant_color: return "constantColor";
    }
    return "followSky";
}

const char* to_string(host_world_environment_preset value) noexcept
{
    switch (value)
    {
    case host_world_environment_preset::clear_day: return "clearDay";
    case host_world_environment_preset::alpine_late_morning: return "alpineLateMorning";
    case host_world_environment_preset::golden_hour: return "goldenHour";
    case host_world_environment_preset::overcast: return "overcast";
    case host_world_environment_preset::night: return "night";
    case host_world_environment_preset::indoor_neutral: return "indoorNeutral";
    }
    return "alpineLateMorning";
}

std::string command_type(const host_command_payload& payload)
{
    return std::visit([](const auto& value) -> std::string {
        using type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<type, host_open_project_command>) return "project.open";
        else if constexpr (std::is_same_v<type, host_close_project_command>) return "project.close";
        else if constexpr (std::is_same_v<type, host_open_scene_command>) return "scene.open";
        else if constexpr (std::is_same_v<type, host_new_scene_command>) return "scene.new";
        else if constexpr (std::is_same_v<type, host_save_scene_command>) return "scene.save";
        else if constexpr (std::is_same_v<type, host_save_scene_as_command>) return "scene.saveAs";
        else if constexpr (std::is_same_v<type, host_create_entity_command>) return "entity.create";
        else if constexpr (std::is_same_v<type, host_delete_entity_command>) return "entity.delete";
        else if constexpr (std::is_same_v<type, host_duplicate_entity_command>) return "entity.duplicate";
        else if constexpr (std::is_same_v<type, host_reparent_entity_command>) return "entity.reparent";
        else if constexpr (std::is_same_v<type, host_reorder_entity_command>) return "entity.reorder";
        else if constexpr (std::is_same_v<type, host_rename_entity_command>) return "entity.rename";
        else if constexpr (std::is_same_v<type, host_select_entity_command>) return "entity.select";
        else if constexpr (std::is_same_v<type, host_clear_selection_command>) return "entity.clearSelection";
        else if constexpr (std::is_same_v<type, host_set_active_command>) return "entity.setActive";
        else if constexpr (std::is_same_v<type, host_set_tag_command>) return "entity.setTag";
        else if constexpr (std::is_same_v<type, host_set_transform_command>) return "entity.setTransform";
        else if constexpr (std::is_same_v<type, host_set_render_layer_command>) return "entity.setRenderLayer";
        else if constexpr (std::is_same_v<type, host_set_camera_command>) return "entity.setCamera";
        else if constexpr (std::is_same_v<type, host_set_mesh_renderer_command>) return "entity.setMeshRenderer";
        else if constexpr (std::is_same_v<type, host_set_terrain_command>) return "terrain.update";
        else if constexpr (std::is_same_v<type, host_set_terrain_brush_command>) return "terrain.setBrush";
        else if constexpr (std::is_same_v<type, host_set_terrain_layer_command>) return "terrain.assignLayer";
        else if constexpr (std::is_same_v<type, host_terrain_stroke_command>) return "terrain.stroke";
        else if constexpr (std::is_same_v<type, host_set_entity_material_command>) return "entity.setMaterial";
        else if constexpr (std::is_same_v<type, host_set_world_environment_command>) return "environment.update";
        else if constexpr (std::is_same_v<type, host_apply_world_environment_preset_command>) return "environment.applyPreset";
        else if constexpr (std::is_same_v<type, host_set_environment_hdri_command>) return "environment.setHdri";
        else if constexpr (std::is_same_v<type, host_set_camera_projection_command>) return "camera.setProjection";
        else if constexpr (std::is_same_v<type, host_viewport_attach_command>) return "viewport.attach";
        else if constexpr (std::is_same_v<type, host_viewport_resize_command>) return "viewport.resize";
        else if constexpr (std::is_same_v<type, host_viewport_set_camera_mode_command>) return "viewport.setCameraMode";
        else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>) return "viewport.setRenderOptions";
        else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>) return "viewport.cameraInput";
        else if constexpr (std::is_same_v<type, host_history_undo_command>) return "history.undo";
        else if constexpr (std::is_same_v<type, host_history_redo_command>) return "history.redo";
        else if constexpr (std::is_same_v<type, host_viewport_set_tool_command>) return "viewport.setTool";
        else if constexpr (std::is_same_v<type, host_viewport_pick_command>) return "viewport.pick";
        else return "unknown";
    }, payload);
}

std::string query_type(const host_query_payload& payload)
{
    return std::visit([](const auto& value) -> std::string {
        using type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<type, host_scene_hierarchy_query>) return "scene.hierarchy";
        else if constexpr (std::is_same_v<type, host_selected_entity_query>) return "entity.selected";
        else if constexpr (std::is_same_v<type, host_project_assets_query>) return "project.assets";
        else if constexpr (std::is_same_v<type, host_asset_thumbnail_query>) return "asset.thumbnail";
        else if constexpr (std::is_same_v<type, host_viewport_state_query>) return "viewport.state";
        else if constexpr (std::is_same_v<type, host_world_environment_query>) return "environment.state";
        else if constexpr (std::is_same_v<type, host_history_state_query>) return "history.state";
        else return "unknown";
    }, payload);
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
    return "{\"position\":" + vec3_json(transform.position) +
        ",\"rotation\":" + quat_json(transform.rotation) +
        ",\"scale\":" + vec3_json(transform.scale) + '}';
}

std::string to_json(const host_camera_snapshot& camera)
{
    std::ostringstream stream;
    stream << "{\"projection\":" << quote(to_string(camera.projection))
        << ",\"fovYDegrees\":" << camera.fov_y_degrees
        << ",\"orthographicHeight\":" << camera.orthographic_height
        << ",\"nearPlane\":" << camera.near_plane
        << ",\"farPlane\":" << camera.far_plane
        << ",\"active\":" << bool_json(camera.active)
        << ",\"clearColor\":" << vec4_json(camera.clear_color) << '}';
    return stream.str();
}

std::string to_json(const host_mesh_renderer_snapshot& mesh_renderer)
{
    return std::string("{\"visible\":") + bool_json(mesh_renderer.visible) +
        ",\"baseColorTint\":" + vec4_json(mesh_renderer.base_color_tint) +
        ",\"hasMaterial\":" + bool_json(mesh_renderer.has_material) +
        ",\"assetBackedMaterial\":" + bool_json(mesh_renderer.asset_backed_material) +
        ",\"materialName\":" + quote(mesh_renderer.material_name) +
        ",\"materialPath\":" + quote(mesh_renderer.material_path) + '}';
}

std::string to_json(const host_terrain_snapshot& terrain)
{
    const char* tool = terrain.brush_tool == host_terrain_brush_tool::smooth ? "smooth" :
        terrain.brush_tool == host_terrain_brush_tool::flatten ? "flatten" :
        terrain.brush_tool == host_terrain_brush_tool::paint ? "paint" : "sculpt";
    std::ostringstream stream;
    stream << "{\"enabled\":" << bool_json(terrain.enabled)
        << ",\"size\":" << terrain.size
        << ",\"resolution\":" << terrain.resolution
        << ",\"chunkQuads\":" << terrain.chunk_quads
        << ",\"receiveShadows\":" << bool_json(terrain.receive_shadows)
        << ",\"contentRevision\":" << terrain.content_revision
        << ",\"brushTool\":" << quote(tool)
        << ",\"brushRadius\":" << terrain.brush_radius
        << ",\"brushStrength\":" << terrain.brush_strength
        << ",\"brushFalloff\":" << terrain.brush_falloff
        << ",\"activeLayer\":" << terrain.active_layer
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
    stream << "{\"entity\":" << to_json(value.entity)
        << ",\"enabled\":" << bool_json(value.enabled)
        << ",\"skyVisible\":" << bool_json(value.sky_visible)
        << ",\"affectLighting\":" << bool_json(value.affect_lighting)
        << ",\"skySource\":" << quote(to_string(value.sky_source))
        << ",\"solidColor\":" << vec3_json(value.solid_color)
        << ",\"hdriPath\":" << quote(value.hdri_path)
        << ",\"hdriRotationDegrees\":" << value.hdri_rotation_degrees
        << ",\"radianceIntensity\":" << value.radiance_intensity
        << ",\"planetRadius\":" << value.planet_radius
        << ",\"atmosphereRadius\":" << value.atmosphere_radius
        << ",\"rayleighStrength\":" << value.rayleigh_strength
        << ",\"mieStrength\":" << value.mie_strength
        << ",\"ozoneStrength\":" << value.ozone_strength
        << ",\"atmosphereTint\":" << vec3_json(value.atmosphere_tint)
        << ",\"groundAlbedo\":" << vec3_json(value.ground_albedo)
        << ",\"mieAnisotropy\":" << value.mie_anisotropy
        << ",\"rayleighScaleHeight\":" << value.rayleigh_scale_height
        << ",\"mieScaleHeight\":" << value.mie_scale_height
        << ",\"multiScatteringFactor\":" << value.multi_scattering_factor
        << ",\"exposure\":" << value.exposure
        << ",\"sunDiskSize\":" << value.sun_disk_size
        << ",\"sunDiskIntensity\":" << value.sun_disk_intensity
        << ",\"sunMode\":" << quote(to_string(value.sun_mode))
        << ",\"timeMode\":" << quote(to_string(value.time_mode))
        << ",\"latitudeDegrees\":" << value.latitude_degrees
        << ",\"longitudeDegrees\":" << value.longitude_degrees
        << ",\"northOffsetDegrees\":" << value.north_offset_degrees
        << ",\"year\":" << value.year << ",\"month\":" << value.month << ",\"day\":" << value.day
        << ",\"localTimeHours\":" << value.local_time_hours
        << ",\"utcOffsetHours\":" << value.utc_offset_hours
        << ",\"playing\":" << bool_json(value.playing)
        << ",\"loopDay\":" << bool_json(value.loop_day)
        << ",\"timeScale\":" << value.time_scale
        << ",\"automaticSunLight\":" << bool_json(value.automatic_sun_light)
        << ",\"sunIntensityMultiplier\":" << value.sun_intensity_multiplier
        << ",\"sunTemperatureMultiplier\":" << value.sun_temperature_multiplier
        << ",\"moonEnabled\":" << bool_json(value.moon_enabled)
        << ",\"automaticMoonPhase\":" << bool_json(value.automatic_moon_phase)
        << ",\"moonPhase\":" << value.moon_phase
        << ",\"moonIntensity\":" << value.moon_intensity
        << ",\"moonAngularRadiusDegrees\":" << value.moon_angular_radius_degrees
        << ",\"starsEnabled\":" << bool_json(value.stars_enabled)
        << ",\"starDensity\":" << value.star_density
        << ",\"starIntensity\":" << value.star_intensity
        << ",\"starTwinkle\":" << value.star_twinkle
        << ",\"cloudsEnabled\":" << bool_json(value.clouds_enabled)
        << ",\"cloudShadows\":" << bool_json(value.cloud_shadows)
        << ",\"cumulus\":" << cloud_layer_json(value.cumulus)
        << ",\"cirrus\":" << cloud_layer_json(value.cirrus)
        << ",\"fogEnabled\":" << bool_json(value.fog_enabled)
        << ",\"fogColor\":" << vec3_json(value.fog_color)
        << ",\"fogDensity\":" << value.fog_density
        << ",\"fogHeightFalloff\":" << value.fog_height_falloff
        << ",\"fogStartDistance\":" << value.fog_start_distance
        << ",\"fogMaxOpacity\":" << value.fog_max_opacity
        << ",\"fogSunScattering\":" << value.fog_sun_scattering
        << ",\"lightingEnabled\":" << bool_json(value.lighting_enabled)
        << ",\"lightingSource\":" << quote(to_string(value.lighting_source))
        << ",\"lightingColor\":" << vec3_json(value.lighting_color)
        << ",\"diffuseIntensity\":" << value.diffuse_intensity
        << ",\"specularIntensity\":" << value.specular_intensity << '}';
    return stream.str();
}

std::string to_json(const host_command_envelope& envelope)
{
    const std::string type = envelope.command_type.empty() ? command_type(envelope.payload) : envelope.command_type;
    std::string payload_json = std::visit([](const auto& payload) -> std::string {
        using type = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<type, host_open_project_command>)
            return "{\"name\":" + quote(payload.name) + ",\"root\":" + quote(payload.root.generic_string()) + '}';
        else if constexpr (std::is_same_v<type, host_open_scene_command>)
            return "{\"path\":" + quote(payload.path.generic_string()) + ",\"append\":" + bool_json(payload.append) + '}';
        else if constexpr (std::is_same_v<type, host_new_scene_command>)
            return "{\"name\":" + quote(payload.name) + '}';
        else if constexpr (std::is_same_v<type, host_save_scene_as_command>)
            return "{\"path\":" + quote(payload.path.generic_string()) + '}';
        else if constexpr (std::is_same_v<type, host_create_entity_command>)
            return "{\"kind\":" + quote(to_string(payload.kind)) + ",\"parent\":" + to_json(payload.parent) + '}';
        else if constexpr (std::is_same_v<type, host_delete_entity_command> || std::is_same_v<type, host_select_entity_command> ||
            std::is_same_v<type, host_duplicate_entity_command>)
            return "{\"entity\":" + to_json(payload.entity) + '}';
        else if constexpr (std::is_same_v<type, host_reparent_entity_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"parent\":" + to_json(payload.parent) +
                ",\"beforeSibling\":" + to_json(payload.before_sibling) + ",\"preserveWorld\":" + bool_json(payload.preserve_world) + '}';
        else if constexpr (std::is_same_v<type, host_reorder_entity_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"beforeSibling\":" + to_json(payload.before_sibling) + '}';
        else if constexpr (std::is_same_v<type, host_rename_entity_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"name\":" + quote(payload.name) + '}';
        else if constexpr (std::is_same_v<type, host_set_active_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"active\":" + bool_json(payload.active) + '}';
        else if constexpr (std::is_same_v<type, host_set_tag_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"tag\":" + quote(payload.tag) + '}';
        else if constexpr (std::is_same_v<type, host_set_transform_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"transform\":" + to_json(payload.transform) + '}';
        else if constexpr (std::is_same_v<type, host_set_render_layer_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"renderLayerMask\":" +
                std::to_string(payload.render_layer_mask) + '}';
        else if constexpr (std::is_same_v<type, host_set_camera_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"camera\":" + to_json(payload.camera) + '}';
        else if constexpr (std::is_same_v<type, host_set_mesh_renderer_command>)
            return "{\"entity\":" + to_json(payload.entity) +
                ",\"visible\":" + bool_json(payload.visible) +
                ",\"baseColorTint\":" + vec4_json(payload.base_color_tint) + '}';
        else if constexpr (std::is_same_v<type, host_set_terrain_command>)
            return "{\"entity\":" + to_json(payload.entity) +
                ",\"enabled\":" + bool_json(payload.enabled) +
                ",\"receiveShadows\":" + bool_json(payload.receive_shadows) + '}';
        else if constexpr (std::is_same_v<type, host_set_terrain_brush_command>)
        {
            const char* tool = payload.tool == host_terrain_brush_tool::smooth ? "smooth" :
                payload.tool == host_terrain_brush_tool::flatten ? "flatten" :
                payload.tool == host_terrain_brush_tool::paint ? "paint" : "sculpt";
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
            const char* phase = payload.phase == host_edit_phase::update ? "update" :
                payload.phase == host_edit_phase::commit ? "commit" :
                payload.phase == host_edit_phase::cancel ? "cancel" : "begin";
            return "{\"entity\":" + to_json(payload.entity) +
                ",\"x\":" + std::to_string(payload.x) + ",\"y\":" + std::to_string(payload.y) +
                ",\"phase\":" + quote(phase) + ",\"invert\":" + bool_json(payload.invert) + '}';
        }
        else if constexpr (std::is_same_v<type, host_set_entity_material_command>)
            return "{\"entity\":" + to_json(payload.entity) +
                ",\"path\":" + quote(payload.path.generic_string()) + '}';
        else if constexpr (std::is_same_v<type, host_set_world_environment_command>)
            return "{\"environment\":" + to_json(payload.environment) + '}';
        else if constexpr (std::is_same_v<type, host_apply_world_environment_preset_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"preset\":" + quote(to_string(payload.preset)) + '}';
        else if constexpr (std::is_same_v<type, host_set_environment_hdri_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"path\":" + quote(payload.path.generic_string()) + '}';
        else if constexpr (std::is_same_v<type, host_set_camera_projection_command>)
            return "{\"projection\":" + quote(to_string(payload.projection)) + '}';
        else if constexpr (std::is_same_v<type, host_viewport_attach_command>)
            return "{\"nativeHandle\":" + std::to_string(payload.native_handle) +
                ",\"x\":" + std::to_string(payload.x) +
                ",\"y\":" + std::to_string(payload.y) +
                ",\"width\":" + std::to_string(payload.width) +
                ",\"height\":" + std::to_string(payload.height) + '}';
        else if constexpr (std::is_same_v<type, host_viewport_resize_command>)
            return "{\"x\":" + std::to_string(payload.x) +
                ",\"y\":" + std::to_string(payload.y) +
                ",\"width\":" + std::to_string(payload.width) +
                ",\"height\":" + std::to_string(payload.height) + '}';
        else if constexpr (std::is_same_v<type, host_viewport_set_camera_mode_command>)
            return "{\"projection\":" + quote(to_string(payload.projection)) + '}';
        else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>)
            return "{\"renderMode\":" + quote(to_string(payload.render_mode)) +
                ",\"visualization\":" + quote(to_string(payload.visualization)) +
                ",\"overlay\":" + quote(to_string(payload.overlay)) +
                ",\"shadows\":" + bool_json(payload.shadows) +
                ",\"environment\":" + environment_json(payload.environment) + '}';
        else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)
            return "{\"orbitX\":" + std::to_string(payload.orbit_x) +
                ",\"orbitY\":" + std::to_string(payload.orbit_y) +
                ",\"panX\":" + std::to_string(payload.pan_x) +
                ",\"panY\":" + std::to_string(payload.pan_y) +
                ",\"forward\":" + std::to_string(payload.forward) +
                ",\"zoom\":" + std::to_string(payload.zoom) +
                ",\"focusSelected\":" + bool_json(payload.focus_selected) + '}';
        else if constexpr (std::is_same_v<type, host_viewport_set_tool_command>)
            return "{\"tool\":" + quote(payload.tool == host_viewport_tool::translate ? "translate" :
                    payload.tool == host_viewport_tool::rotate ? "rotate" : payload.tool == host_viewport_tool::scale ? "scale" :
                    payload.tool == host_viewport_tool::terrain ? "terrain" : "select") +
                ",\"coordinateSpace\":" + quote(payload.coordinate_space == host_coordinate_space::local ? "local" : "world") +
                ",\"snapping\":" + bool_json(payload.snapping) + ",\"translationSnap\":" + std::to_string(payload.translation_snap) +
                ",\"rotationSnapDegrees\":" + std::to_string(payload.rotation_snap_degrees) +
                ",\"scaleSnap\":" + std::to_string(payload.scale_snap) + '}';
        else if constexpr (std::is_same_v<type, host_viewport_pick_command>)
            return "{\"x\":" + std::to_string(payload.x) + ",\"y\":" + std::to_string(payload.y) + '}';
        else
            return "{}";
    }, envelope.payload);

    std::string edit_json;
    if (envelope.edit)
    {
        const char* phase = envelope.edit->phase == host_edit_phase::begin ? "begin" :
            envelope.edit->phase == host_edit_phase::update ? "update" :
            envelope.edit->phase == host_edit_phase::commit ? "commit" :
            envelope.edit->phase == host_edit_phase::cancel ? "cancel" : "none";
        edit_json = ",\"edit\":{\"id\":" + std::to_string(envelope.edit->id) +
            ",\"phase\":" + quote(phase) + ",\"label\":" + quote(envelope.edit->label) + '}';
    }
    return "{\"kind\":\"command\",\"requestId\":" + std::to_string(envelope.request_id) +
        ",\"type\":" + quote(type) + ",\"payload\":" + payload_json + edit_json + '}';
}

std::string to_json(const host_query_envelope& envelope)
{
    const std::string type = envelope.query_type.empty() ? query_type(envelope.payload) : envelope.query_type;
    const std::string payload = std::visit([](const auto& value) -> std::string {
        using query = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<query, host_world_environment_query>)
            return "{\"entity\":" + to_json(value.entity) + '}';
        else if constexpr (std::is_same_v<query, host_asset_thumbnail_query>)
            return "{\"path\":" + quote(value.path) + ",\"maxSize\":" + std::to_string(value.max_size) + '}';
        return "{}";
    }, envelope.payload);
    return "{\"kind\":\"query\",\"requestId\":" + std::to_string(envelope.request_id) +
        ",\"type\":" + quote(type) + ",\"payload\":" + payload + '}';
}

std::string to_json(const host_response& response)
{
    const std::string payload = response.payload_json.empty() ? "{}" : response.payload_json;
    return "{\"kind\":\"response\",\"requestId\":" + std::to_string(response.request_id) +
        ",\"succeeded\":" + bool_json(response.succeeded) +
        ",\"error\":" + quote(response.error) +
        ",\"payload\":" + payload + '}';
}

std::string to_json(const host_event& event)
{
    const std::string payload = event.payload_json.empty() ? "{}" : event.payload_json;
    return "{\"kind\":\"event\",\"sequence\":" + std::to_string(event.sequence) +
        ",\"type\":" + quote(to_string(event.event_type)) +
        ",\"entity\":" + to_json(event.entity) +
        ",\"message\":" + quote(event.message) +
        ",\"payload\":" + payload + '}';
}

std::string to_json(const host_profiler_snapshot& snapshot)
{
    std::string json =
        "{\"timestampNanoseconds\":" + std::to_string(snapshot.timestamp_nanoseconds) +
        ",\"memory\":{\"bytes\":" + std::to_string(snapshot.memory_bytes) +
        ",\"softLimit\":" + std::to_string(snapshot.memory_soft_limit) +
        ",\"hardLimit\":" + std::to_string(snapshot.memory_hard_limit) +
        ",\"pressureEvents\":" + std::to_string(snapshot.memory_pressure_events) +
        ",\"domains\":[";
    for (std::size_t index = 0; index < snapshot.memory_domains.size(); ++index)
    {
        if (index != 0)
            json += ',';
        const auto& domain = snapshot.memory_domains[index];
        json += "{\"domain\":" + quote(domain.domain) +
            ",\"bytes\":" + std::to_string(domain.bytes_outstanding) +
            ",\"peakBytes\":" + std::to_string(domain.peak_bytes) +
            ",\"softLimit\":" + std::to_string(domain.soft_limit) +
            ",\"hardLimit\":" + std::to_string(domain.hard_limit) +
            ",\"pressure\":" + bool_json(domain.pressure) + '}';
    }
    json += "],\"groups\":[";
    for (std::size_t index = 0; index < snapshot.allocation_groups.size(); ++index)
    {
        if (index != 0)
            json += ',';
        const auto& group = snapshot.allocation_groups[index];
        json += "{\"domain\":" + quote(group.domain) +
            ",\"tag\":" + quote(group.tag) +
            ",\"worldId\":" + std::to_string(group.world_id) +
            ",\"threadId\":" + std::to_string(group.thread_id) +
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
        ",\"droppedEvents\":" + std::to_string(snapshot.dropped_profile_events) +
        ",\"jobs\":[";
    for (std::size_t index = 0; index < snapshot.jobs.size(); ++index)
    {
        if (index != 0)
            json += ',';
        const auto& job = snapshot.jobs[index];
        json += "{\"sequence\":" + std::to_string(job.sequence) +
            ",\"name\":" + quote(job.name) +
            ",\"priority\":" + quote(job.priority) +
            ",\"affinity\":" + quote(job.affinity) +
            ",\"status\":" + quote(job.status) +
            ",\"threadId\":" + std::to_string(job.thread_id) +
            ",\"queuedNanoseconds\":" + std::to_string(job.queued_nanoseconds) +
            ",\"startedNanoseconds\":" + std::to_string(job.started_nanoseconds) +
            ",\"completedNanoseconds\":" + std::to_string(job.completed_nanoseconds) + '}';
    }
    json += "]}}";
    return json;
}

std::string to_json(const host_scene_snapshot& snapshot)
{
    std::string json = "{\"sceneGuid\":" + quote(snapshot.scene_guid) +
        ",\"sceneName\":" + quote(snapshot.scene_name) +
        ",\"activeScenePath\":" + quote(snapshot.active_scene_path) +
        ",\"dirty\":" + bool_json(snapshot.dirty) +
        ",\"canUndo\":" + bool_json(snapshot.can_undo) +
        ",\"canRedo\":" + bool_json(snapshot.can_redo) +
        ",\"undoLabel\":" + quote(snapshot.undo_label) +
        ",\"redoLabel\":" + quote(snapshot.redo_label) +
        ",\"entities\":[";
    for (std::size_t index = 0; index < snapshot.entities.size(); ++index)
    {
        const auto& entity = snapshot.entities[index];
        if (index != 0)
            json += ',';
        json += "{\"entity\":" + to_json(entity.entity) +
            ",\"guid\":" + quote(entity.guid) +
            ",\"parentGuid\":" + quote(entity.parent_guid) +
            ",\"siblingOrder\":" + std::to_string(entity.sibling_order) +
            ",\"name\":" + quote(entity.name) +
            ",\"kind\":" + quote(to_string(entity.kind)) +
            ",\"active\":" + bool_json(entity.active) +
            ",\"selected\":" + bool_json(entity.selected) + '}';
    }
    json += "]}";
    return json;
}

std::string to_json(const host_selected_entity_snapshot& snapshot)
{
    std::string json = "{\"entity\":" + to_json(snapshot.entity) +
        ",\"name\":" + quote(snapshot.name) +
        ",\"tag\":" + quote(snapshot.tag) +
        ",\"active\":" + bool_json(snapshot.active) +
        ",\"renderLayerMask\":" + std::to_string(snapshot.render_layer_mask) +
        ",\"transform\":";
    json += snapshot.transform ? to_json(*snapshot.transform) : "null";
    json += ",\"camera\":";
    json += snapshot.camera ? to_json(*snapshot.camera) : "null";
    json += ",\"meshRenderer\":";
    json += snapshot.mesh_renderer ? to_json(*snapshot.mesh_renderer) : "null";
    json += ",\"terrain\":";
    json += snapshot.terrain ? to_json(*snapshot.terrain) : "null";
    json += ",\"components\":[";
    for (std::size_t index = 0; index < snapshot.components.size(); ++index)
    {
        const auto& component = snapshot.components[index];
        if (index != 0)
            json += ',';
        json += "{\"kind\":" + quote(to_string(component.kind)) +
            ",\"label\":" + quote(component.label) +
            ",\"editable\":" + bool_json(component.editable) + '}';
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
        ",\"assets\":[";
    for (std::size_t index = 0; index < snapshot.assets.size(); ++index)
    {
        const auto& asset = snapshot.assets[index];
        if (index != 0)
            json += ',';
        json += "{\"path\":" + quote(asset.path) +
            ",\"kind\":" + quote(asset.kind) +
            ",\"imported\":" + bool_json(asset.imported) +
            ",\"importRunning\":" + bool_json(asset.import_running) + '}';
    }
    json += "]}";
    return json;
}

std::string to_json(const host_asset_thumbnail_snapshot& snapshot)
{
    return "{\"path\":" + quote(snapshot.path) +
        ",\"width\":" + std::to_string(snapshot.width) +
        ",\"height\":" + std::to_string(snapshot.height) +
        ",\"dataUrl\":" + quote(snapshot.data_url) + '}';
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
    if (!object_value(json, "payload", payload))
        payload = "{}";

    if (type == "project.open")
    {
        host_open_project_command command;
        string_value(payload, "name", command.name);
        std::string root;
        string_value(payload, "root", root);
        command.root = root;
        envelope.payload = std::move(command);
    }
    else if (type == "project.close")
    {
        envelope.payload = host_close_project_command{};
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
    else if (type == "entity.create")
    {
        static constexpr std::pair<std::string_view, host_create_entity_kind> values[]{
            { "empty", host_create_entity_kind::empty },
            { "plane", host_create_entity_kind::plane },
            { "cube", host_create_entity_kind::cube },
            { "sphere", host_create_entity_kind::sphere },
            { "cylinder", host_create_entity_kind::cylinder },
            { "worldEnvironment", host_create_entity_kind::world_environment },
            { "terrain", host_create_entity_kind::terrain },
            { "water", host_create_entity_kind::water },
            { "grassPatch", host_create_entity_kind::grass_patch },
            { "decal", host_create_entity_kind::decal }
        };
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
            envelope.payload = host_delete_entity_command{ .entity = entity };
        else
            envelope.payload = host_select_entity_command{ .entity = entity };
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
        envelope.payload = command;
    }
    else if (type == "entity.setCamera")
    {
        host_set_camera_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !camera_value(payload, "camera", command.camera))
        {
            error = "Camera command requires entity and a typed camera snapshot";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "entity.setMeshRenderer")
    {
        host_set_mesh_renderer_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !bool_value(payload, "visible", command.visible) ||
            !array4_value(payload, "baseColorTint", command.base_color_tint))
        {
            error = "Mesh renderer command requires entity, visible, and baseColorTint";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "terrain.update")
    {
        host_set_terrain_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !bool_value(payload, "enabled", command.enabled) ||
            !bool_value(payload, "receiveShadows", command.receive_shadows))
        {
            error = "Terrain update requires entity, enabled, and receiveShadows";
            return false;
        }
        envelope.payload = command;
    }
    else if (type == "terrain.setBrush")
    {
        static constexpr std::pair<std::string_view, host_terrain_brush_tool> tools[]{
            { "sculpt", host_terrain_brush_tool::sculpt }, { "smooth", host_terrain_brush_tool::smooth },
            { "flatten", host_terrain_brush_tool::flatten }, { "paint", host_terrain_brush_tool::paint } };
        host_set_terrain_brush_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !number_value(payload, "radius", command.radius) ||
            !number_value(payload, "strength", command.strength) ||
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
        if (!entity_field_value(payload, "entity", command.entity) ||
            !number_value(payload, "x", command.x) || !number_value(payload, "y", command.y) ||
            !string_value(payload, "phase", phase))
        {
            error = "Terrain stroke requires entity, viewport coordinates, and phase";
            return false;
        }
        command.phase = phase == "update" ? host_edit_phase::update : phase == "commit" ? host_edit_phase::commit :
            phase == "cancel" ? host_edit_phase::cancel : host_edit_phase::begin;
        bool_value(payload, "invert", command.invert);
        envelope.payload = command;
    }
    else if (type == "terrain.assignLayer")
    {
        host_set_terrain_layer_command command;
        std::string path;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !number_value(payload, "layer", command.layer) || !string_value(payload, "path", path))
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
        if (!entity_field_value(payload, "entity", command.entity) ||
            !string_value(payload, "path", path) || path.empty())
        {
            error = "Material assignment requires entity and material path";
            return false;
        }
        command.path = std::move(path);
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
            { "clearDay", host_world_environment_preset::clear_day },
            { "alpineLateMorning", host_world_environment_preset::alpine_late_morning },
            { "goldenHour", host_world_environment_preset::golden_hour },
            { "overcast", host_world_environment_preset::overcast },
            { "night", host_world_environment_preset::night },
            { "indoorNeutral", host_world_environment_preset::indoor_neutral }
        };
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
            { "perspective", host_camera_projection::perspective },
            { "orthographic", host_camera_projection::orthographic }
        };
        host_camera_projection projection{ host_camera_projection::perspective };
        parse_enum(payload, "projection", values, std::size(values), projection);
        if (type == "camera.setProjection")
            envelope.payload = host_set_camera_projection_command{ .projection = projection };
        else
            envelope.payload = host_viewport_set_camera_mode_command{ .projection = projection };
    }
    else if (type == "viewport.attach")
    {
        host_viewport_attach_command command;
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
        number_value(payload, "x", command.x);
        number_value(payload, "y", command.y);
        number_value(payload, "width", command.width);
        number_value(payload, "height", command.height);
        envelope.payload = command;
    }
    else if (type == "viewport.setRenderOptions")
    {
        static constexpr std::pair<std::string_view, host_render_mode> render_modes[]{
            { "shaded", host_render_mode::shaded },
            { "wireframe", host_render_mode::wireframe }
        };
        static constexpr std::pair<std::string_view, host_visualization_mode> visualizations[]{
            { "standard", host_visualization_mode::standard },
            { "albedo", host_visualization_mode::albedo },
            { "opacity", host_visualization_mode::opacity },
            { "worldNormal", host_visualization_mode::world_normal },
            { "specularity", host_visualization_mode::specularity },
            { "gloss", host_visualization_mode::gloss },
            { "metalness", host_visualization_mode::metalness },
            { "ao", host_visualization_mode::ao },
            { "emission", host_visualization_mode::emission },
            { "lighting", host_visualization_mode::lighting },
            { "uv0", host_visualization_mode::uv0 },
            { "cascadeDebug", host_visualization_mode::cascade_debug },
            { "shadowMask", host_visualization_mode::shadow_mask },
            { "lightComplexity", host_visualization_mode::light_complexity },
            { "clusterDebug", host_visualization_mode::cluster_debug }
        };
        static constexpr std::pair<std::string_view, host_overlay_mode> overlays[]{
            { "none", host_overlay_mode::none },
            { "selectedWireframe", host_overlay_mode::selected_wireframe },
            { "allWireframe", host_overlay_mode::all_wireframe }
        };
        host_viewport_set_render_options_command command;
        parse_enum(payload, "renderMode", render_modes, std::size(render_modes), command.render_mode);
        parse_enum(payload, "visualization", visualizations, std::size(visualizations), command.visualization);
        parse_enum(payload, "overlay", overlays, std::size(overlays), command.overlay);
        bool_value(payload, "shadows", command.shadows);
        parse_environment(payload, command.environment);
        envelope.payload = command;
    }
    else if (type == "viewport.cameraInput")
    {
        host_viewport_camera_input_command command;
        number_value(payload, "orbitX", command.orbit_x);
        number_value(payload, "orbitY", command.orbit_y);
        number_value(payload, "panX", command.pan_x);
        number_value(payload, "panY", command.pan_y);
        number_value(payload, "forward", command.forward);
        number_value(payload, "zoom", command.zoom);
        bool_value(payload, "focusSelected", command.focus_selected);
        envelope.payload = command;
    }
    else if (type == "history.undo")
        envelope.payload = host_history_undo_command{};
    else if (type == "history.redo")
        envelope.payload = host_history_redo_command{};
    else if (type == "viewport.setTool")
    {
        static constexpr std::pair<std::string_view, host_viewport_tool> tools[]{
            { "select", host_viewport_tool::select }, { "translate", host_viewport_tool::translate },
            { "rotate", host_viewport_tool::rotate }, { "scale", host_viewport_tool::scale },
            { "terrain", host_viewport_tool::terrain } };
        static constexpr std::pair<std::string_view, host_coordinate_space> spaces[]{
            { "world", host_coordinate_space::world }, { "local", host_coordinate_space::local } };
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
        number_value(payload, "x", command.x);
        number_value(payload, "y", command.y);
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
        transaction.phase = phase == "begin" ? host_edit_phase::begin : phase == "update" ? host_edit_phase::update :
            phase == "commit" ? host_edit_phase::commit : phase == "cancel" ? host_edit_phase::cancel : host_edit_phase::none;
        if (transaction.id != 0 && transaction.phase != host_edit_phase::none)
            envelope.edit = std::move(transaction);
    }
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
        envelope.payload = host_viewport_state_query{};
    else if (type == "environment.state")
    {
        host_entity_id entity;
        if (!entity_field_value(json, "entity", entity))
        {
            error = "Environment query requires entity";
            return false;
        }
        envelope.payload = host_world_environment_query{ .entity = entity };
    }
    else if (type == "history.state")
        envelope.payload = host_history_state_query{};
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
