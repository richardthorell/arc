#include <arc/editor/host_protocol.h>

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

std::string environment_json(const host_environment_visibility& value)
{
    return std::string("{\"sky\":") + bool_json(value.sky) +
        ",\"fog\":" + bool_json(value.fog) +
        ",\"terrain\":" + bool_json(value.terrain) +
        ",\"water\":" + bool_json(value.water) +
        ",\"vegetation\":" + bool_json(value.vegetation) +
        ",\"decals\":" + bool_json(value.decals) + '}';
}

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
    case host_component_kind::mesh_renderer: return "meshRenderer";
    case host_component_kind::directional_light: return "directionalLight";
    case host_component_kind::point_light: return "pointLight";
    case host_component_kind::spot_light: return "spotLight";
    case host_component_kind::sky_atmosphere: return "skyAtmosphere";
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

std::string command_type(const host_command_payload& payload)
{
    return std::visit([](const auto& value) -> std::string {
        using type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<type, host_open_project_command>) return "project.open";
        else if constexpr (std::is_same_v<type, host_close_project_command>) return "project.close";
        else if constexpr (std::is_same_v<type, host_open_scene_command>) return "scene.open";
        else if constexpr (std::is_same_v<type, host_create_entity_command>) return "entity.create";
        else if constexpr (std::is_same_v<type, host_delete_entity_command>) return "entity.delete";
        else if constexpr (std::is_same_v<type, host_rename_entity_command>) return "entity.rename";
        else if constexpr (std::is_same_v<type, host_select_entity_command>) return "entity.select";
        else if constexpr (std::is_same_v<type, host_clear_selection_command>) return "entity.clearSelection";
        else if constexpr (std::is_same_v<type, host_set_active_command>) return "entity.setActive";
        else if constexpr (std::is_same_v<type, host_set_tag_command>) return "entity.setTag";
        else if constexpr (std::is_same_v<type, host_set_transform_command>) return "entity.setTransform";
        else if constexpr (std::is_same_v<type, host_set_camera_projection_command>) return "camera.setProjection";
        else if constexpr (std::is_same_v<type, host_viewport_attach_command>) return "viewport.attach";
        else if constexpr (std::is_same_v<type, host_viewport_resize_command>) return "viewport.resize";
        else if constexpr (std::is_same_v<type, host_viewport_set_camera_mode_command>) return "viewport.setCameraMode";
        else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>) return "viewport.setRenderOptions";
        else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>) return "viewport.cameraInput";
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
        else if constexpr (std::is_same_v<type, host_viewport_state_query>) return "viewport.state";
        else return "unknown";
    }, payload);
}

std::string to_json(const host_entity_id& entity)
{
    return "{\"index\":" + std::to_string(entity.index) + ",\"generation\":" + std::to_string(entity.generation) + '}';
}

std::string to_json(const host_transform& transform)
{
    return "{\"position\":" + vec3_json(transform.position) +
        ",\"rotation\":" + quat_json(transform.rotation) +
        ",\"scale\":" + vec3_json(transform.scale) + '}';
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
        else if constexpr (std::is_same_v<type, host_create_entity_command>)
            return "{\"kind\":" + quote(to_string(payload.kind)) + '}';
        else if constexpr (std::is_same_v<type, host_delete_entity_command> || std::is_same_v<type, host_select_entity_command>)
            return "{\"entity\":" + to_json(payload.entity) + '}';
        else if constexpr (std::is_same_v<type, host_rename_entity_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"name\":" + quote(payload.name) + '}';
        else if constexpr (std::is_same_v<type, host_set_active_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"active\":" + bool_json(payload.active) + '}';
        else if constexpr (std::is_same_v<type, host_set_tag_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"tag\":" + quote(payload.tag) + '}';
        else if constexpr (std::is_same_v<type, host_set_transform_command>)
            return "{\"entity\":" + to_json(payload.entity) + ",\"transform\":" + to_json(payload.transform) + '}';
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
        else
            return "{}";
    }, envelope.payload);

    return "{\"kind\":\"command\",\"requestId\":" + std::to_string(envelope.request_id) +
        ",\"type\":" + quote(type) + ",\"payload\":" + payload_json + '}';
}

std::string to_json(const host_query_envelope& envelope)
{
    const std::string type = envelope.query_type.empty() ? query_type(envelope.payload) : envelope.query_type;
    return "{\"kind\":\"query\",\"requestId\":" + std::to_string(envelope.request_id) +
        ",\"type\":" + quote(type) + ",\"payload\":{}}";
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

std::string to_json(const host_scene_snapshot& snapshot)
{
    std::string json = "{\"entities\":[";
    for (std::size_t index = 0; index < snapshot.entities.size(); ++index)
    {
        const auto& entity = snapshot.entities[index];
        if (index != 0)
            json += ',';
        json += "{\"entity\":" + to_json(entity.entity) +
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
        ",\"transform\":";
    json += snapshot.transform ? to_json(*snapshot.transform) : "null";
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
    else if (type == "entity.create")
    {
        static constexpr std::pair<std::string_view, host_create_entity_kind> values[]{
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
    else
    {
        error = "Unsupported host command type: " + type;
        return false;
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
    else if (type == "viewport.state")
        envelope.payload = host_viewport_state_query{};
    else
    {
        error = "Unsupported host query type: " + type;
        return false;
    }
    envelope.query_type = std::move(type);
    return true;
}

} // namespace arc::editor
