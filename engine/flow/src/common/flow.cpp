#include <arc/flow/flow.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace arc::flow
{
namespace
{

using json = nlohmann::json;

enum class node_kind : std::uint8_t
{
    begin_play,
    end_play,
    tick,
    fixed_tick,
    input_action,
    branch,
    self_entity,
    create_entity,
    destroy_entity,
    has_core_component,
    remove_core_component,
    entity_alive,
    get_name,
    set_name,
    get_tag,
    set_tag,
    get_active,
    set_active,
    get_transform,
    set_transform,
    bool_literal,
    string_literal,
    vector3_literal,
    vector4_literal,
};

enum class pin_kind : std::uint8_t
{
    execution,
    value,
};

enum class connection_kind : std::uint8_t
{
    execution,
    value,
};

struct pin_info
{
    pin_kind kind{pin_kind::execution};
    std::optional<value_type> type;
};

struct source_node
{
    std::string id;
    node_kind kind{node_kind::begin_play};
    json values{json::object()};
};

struct source_pin_ref
{
    std::string node_id;
    std::string pin;
};

struct source_connection
{
    std::string id;
    connection_kind kind{connection_kind::execution};
    source_pin_ref from;
    source_pin_ref to;
};

struct source_graph
{
    std::vector<variable> variables;
    std::vector<source_node> nodes;
    std::vector<source_connection> connections;
};

void add_diagnostic(std::vector<diagnostic>& diagnostics, diagnostic_severity severity, std::string code,
                    std::string message, std::string node_id = {}, std::string pin_id = {},
                    std::string connection_id = {})
{
    diagnostics.push_back({.severity = severity,
                           .code = std::move(code),
                           .message = std::move(message),
                           .node_id = std::move(node_id),
                           .pin_id = std::move(pin_id),
                           .connection_id = std::move(connection_id)});
}

bool has_errors(const std::vector<diagnostic>& diagnostics)
{
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                       [](const diagnostic& item) { return item.severity == diagnostic_severity::error; });
}

std::optional<node_kind> parse_node_kind(std::string_view value)
{
    if (value == "beginPlay") return node_kind::begin_play;
    if (value == "endPlay") return node_kind::end_play;
    if (value == "tick") return node_kind::tick;
    if (value == "fixedTick") return node_kind::fixed_tick;
    if (value == "inputAction") return node_kind::input_action;
    if (value == "branch") return node_kind::branch;
    if (value == "selfEntity") return node_kind::self_entity;
    if (value == "createEntity") return node_kind::create_entity;
    if (value == "destroyEntity") return node_kind::destroy_entity;
    if (value == "hasCoreComponent") return node_kind::has_core_component;
    if (value == "removeCoreComponent") return node_kind::remove_core_component;
    if (value == "isEntityAlive") return node_kind::entity_alive;
    if (value == "getName") return node_kind::get_name;
    if (value == "setName") return node_kind::set_name;
    if (value == "getTag") return node_kind::get_tag;
    if (value == "setTag") return node_kind::set_tag;
    if (value == "getActive") return node_kind::get_active;
    if (value == "setActive") return node_kind::set_active;
    if (value == "getTransform") return node_kind::get_transform;
    if (value == "setTransform") return node_kind::set_transform;
    if (value == "boolLiteral") return node_kind::bool_literal;
    if (value == "stringLiteral") return node_kind::string_literal;
    if (value == "vector3Literal") return node_kind::vector3_literal;
    if (value == "vector4Literal") return node_kind::vector4_literal;
    return std::nullopt;
}

std::optional<value_type> parse_value_type(std::string_view value)
{
    if (value == "bool") return value_type::boolean;
    if (value == "int") return value_type::integer;
    if (value == "float") return value_type::float32;
    if (value == "vec2") return value_type::vector2;
    if (value == "vec3") return value_type::vector3;
    if (value == "vec4") return value_type::vector4;
    if (value == "string") return value_type::string;
    if (value == "name") return value_type::name;
    if (value == "entity") return value_type::entity;
    if (value == "component") return value_type::component;
    return std::nullopt;
}

flow_value default_value(value_type type)
{
    switch (type)
    {
        case value_type::boolean:
            return false;
        case value_type::integer:
            return std::int64_t{0};
        case value_type::float32:
            return 0.0;
        case value_type::vector2:
            return std::array<double, 2>{0.0, 0.0};
        case value_type::vector3:
            return std::array<double, 3>{0.0, 0.0, 0.0};
        case value_type::vector4:
            return std::array<double, 4>{0.0, 0.0, 0.0, 0.0};
        case value_type::string:
        case value_type::name:
            return std::string{};
        case value_type::entity:
        case value_type::component:
            return std::monostate{};
    }
    return std::monostate{};
}

template <std::size_t N> std::optional<flow_value> parse_vector_value(const json& value)
{
    if (!value.is_array() || value.size() != N) return std::nullopt;

    std::array<double, N> result{};
    for (std::size_t index = 0; index < N; ++index)
    {
        if (!value[index].is_number()) return std::nullopt;
        result[index] = value[index].get<double>();
    }
    return flow_value{result};
}

std::optional<flow_value> parse_default_value(value_type type, const json& value)
{
    switch (type)
    {
        case value_type::boolean:
            if (value.is_boolean()) return flow_value{value.get<bool>()};
            break;
        case value_type::integer:
            if (value.is_number_integer()) return flow_value{value.get<std::int64_t>()};
            break;
        case value_type::float32:
            if (value.is_number()) return flow_value{value.get<double>()};
            break;
        case value_type::vector2:
            return parse_vector_value<2>(value);
        case value_type::vector3:
            return parse_vector_value<3>(value);
        case value_type::vector4:
            return parse_vector_value<4>(value);
        case value_type::string:
        case value_type::name:
            if (value.is_string()) return flow_value{value.get<std::string>()};
            break;
        case value_type::entity:
        case value_type::component:
            if (value.is_null()) return flow_value{std::monostate{}};
            break;
    }
    return std::nullopt;
}

std::optional<world_core_component> parse_core_component(std::string_view value)
{
    if (value == "name") return world_core_component::name;
    if (value == "transform") return world_core_component::transform;
    if (value == "tag") return world_core_component::tag;
    if (value == "active") return world_core_component::active;
    return std::nullopt;
}

std::optional<value_type> literal_type(node_kind kind)
{
    switch (kind)
    {
        case node_kind::bool_literal:
            return value_type::boolean;
        case node_kind::string_literal:
            return value_type::string;
        case node_kind::vector3_literal:
            return value_type::vector3;
        case node_kind::vector4_literal:
            return value_type::vector4;
        default:
            return std::nullopt;
    }
}

bool is_event_node(node_kind kind)
{
    return kind == node_kind::begin_play || kind == node_kind::end_play || kind == node_kind::tick ||
           kind == node_kind::fixed_tick || kind == node_kind::input_action;
}

bool is_executable_node(node_kind kind)
{
    switch (kind)
    {
        case node_kind::branch:
        case node_kind::create_entity:
        case node_kind::destroy_entity:
        case node_kind::has_core_component:
        case node_kind::remove_core_component:
        case node_kind::entity_alive:
        case node_kind::get_name:
        case node_kind::set_name:
        case node_kind::get_tag:
        case node_kind::set_tag:
        case node_kind::get_active:
        case node_kind::set_active:
        case node_kind::get_transform:
        case node_kind::set_transform:
            return true;
        default:
            return false;
    }
}

std::optional<pin_info> output_pin(node_kind kind, std::string_view pin)
{
    switch (kind)
    {
        case node_kind::begin_play:
        case node_kind::end_play:
            if (pin == "exec") return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::tick:
        case node_kind::fixed_tick:
            if (pin == "exec") return pin_info{.kind = pin_kind::execution};
            if (pin == "deltaSeconds") return pin_info{.kind = pin_kind::value, .type = value_type::float32};
            break;
        case node_kind::input_action:
            if (pin == "triggered" || pin == "completed") return pin_info{.kind = pin_kind::execution};
            if (pin == "value") return pin_info{.kind = pin_kind::value, .type = value_type::float32};
            break;
        case node_kind::branch:
            if (pin == "true" || pin == "false") return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::self_entity:
            if (pin == "entity") return pin_info{.kind = pin_kind::value, .type = value_type::entity};
            break;
        case node_kind::create_entity:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            if (pin == "entity") return pin_info{.kind = pin_kind::value, .type = value_type::entity};
            break;
        case node_kind::destroy_entity:
        case node_kind::remove_core_component:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::has_core_component:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            if (pin == "has") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
            break;
        case node_kind::entity_alive:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            if (pin == "alive") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
            break;
        case node_kind::get_name:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            if (pin == "name") return pin_info{.kind = pin_kind::value, .type = value_type::string};
            break;
        case node_kind::get_tag:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            if (pin == "tag") return pin_info{.kind = pin_kind::value, .type = value_type::string};
            break;
        case node_kind::get_active:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            if (pin == "active") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
            break;
        case node_kind::get_transform:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            if (pin == "position") return pin_info{.kind = pin_kind::value, .type = value_type::vector3};
            if (pin == "rotation") return pin_info{.kind = pin_kind::value, .type = value_type::vector4};
            if (pin == "scale") return pin_info{.kind = pin_kind::value, .type = value_type::vector3};
            break;
        case node_kind::set_name:
        case node_kind::set_tag:
        case node_kind::set_active:
        case node_kind::set_transform:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::bool_literal:
            if (pin == "value") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
            break;
        case node_kind::string_literal:
            if (pin == "value") return pin_info{.kind = pin_kind::value, .type = value_type::string};
            break;
        case node_kind::vector3_literal:
            if (pin == "value") return pin_info{.kind = pin_kind::value, .type = value_type::vector3};
            break;
        case node_kind::vector4_literal:
            if (pin == "value") return pin_info{.kind = pin_kind::value, .type = value_type::vector4};
            break;
    }
    return std::nullopt;
}

std::optional<pin_info> input_pin(node_kind kind, std::string_view pin)
{
    if (kind == node_kind::branch)
    {
        if (pin == "exec") return pin_info{.kind = pin_kind::execution};
        if (pin == "condition") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
        return std::nullopt;
    }

    if (!is_executable_node(kind)) return std::nullopt;
    if (pin == "exec") return pin_info{.kind = pin_kind::execution};
    if (kind != node_kind::create_entity && pin == "entity")
        return pin_info{.kind = pin_kind::value, .type = value_type::entity};

    switch (kind)
    {
        case node_kind::set_name:
            if (pin == "name") return pin_info{.kind = pin_kind::value, .type = value_type::string};
            break;
        case node_kind::set_tag:
            if (pin == "tag") return pin_info{.kind = pin_kind::value, .type = value_type::string};
            break;
        case node_kind::set_active:
            if (pin == "active") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
            break;
        case node_kind::set_transform:
            if (pin == "position") return pin_info{.kind = pin_kind::value, .type = value_type::vector3};
            if (pin == "rotation") return pin_info{.kind = pin_kind::value, .type = value_type::vector4};
            if (pin == "scale") return pin_info{.kind = pin_kind::value, .type = value_type::vector3};
            break;
        default:
            break;
    }
    return std::nullopt;
}

std::string pin_key(std::string_view node_id, std::string_view pin)
{
    std::string result;
    result.reserve(node_id.size() + pin.size() + 1);
    result.append(node_id);
    result.push_back('\x1f');
    result.append(pin);
    return result;
}

std::optional<source_graph> parse_source(std::string_view source, std::vector<diagnostic>& diagnostics)
{
    json root;
    try
    {
        root = json::parse(source.begin(), source.end());
    }
    catch (const std::exception& exception)
    {
        add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_PARSE_ERROR",
                       std::string{"Unable to parse Flow asset JSON: "} + exception.what());
        return std::nullopt;
    }

    if (!root.is_object() || root.value("version", 0) != 1 || root.value("assetType", std::string{}) != "flow")
    {
        add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_ASSET_SCHEMA",
                       "Flow compiler expects a version-1 asset with assetType 'flow'.");
        return std::nullopt;
    }

    const auto graph_iterator = root.find("graph");
    if (graph_iterator == root.end() || !graph_iterator->is_object())
    {
        add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_GRAPH_MISSING",
                       "Flow asset does not contain a graph object.");
        return std::nullopt;
    }

    const json& graph_json = *graph_iterator;
    if (graph_json.value("version", 0) != 1)
    {
        add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_GRAPH_VERSION",
                       "Unsupported Flow graph version. Expected version 1.");
        return std::nullopt;
    }

    source_graph graph;

    const auto variables_iterator = graph_json.find("variables");
    if (variables_iterator == graph_json.end() || !variables_iterator->is_array())
    {
        add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VARIABLES_SCHEMA",
                       "Flow graph variables must be an array.");
        return std::nullopt;
    }

    for (const json& variable_json : *variables_iterator)
    {
        if (!variable_json.is_object())
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VARIABLE_SCHEMA",
                           "Flow variable entries must be objects.");
            continue;
        }

        const std::string id = variable_json.value("id", std::string{});
        const std::string name = variable_json.value("name", std::string{});
        const std::string type_name = variable_json.value("type", std::string{});
        const auto type = parse_value_type(type_name);
        if (!type)
        {
            add_diagnostic(diagnostics, diagnostic_severity::error,
                           type_name == "any" ? "FLOW_UNRESOLVED_ANY" : "FLOW_VARIABLE_TYPE",
                           type_name == "any" ? "Flow variables cannot use the unresolved 'any' type."
                                              : "Flow variable has an unsupported type.",
                           id);
            continue;
        }

        const auto default_iterator = variable_json.find("defaultValue");
        if (id.empty() || name.empty() || default_iterator == variable_json.end() ||
            !variable_json.contains("exposed") || !variable_json["exposed"].is_boolean())
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VARIABLE_SCHEMA",
                           "Flow variable requires id, name, defaultValue, and exposed fields.", id);
            continue;
        }

        const auto parsed_default = parse_default_value(*type, *default_iterator);
        if (!parsed_default)
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VARIABLE_DEFAULT",
                           "Flow variable defaultValue does not match its declared type.", id);
            continue;
        }

        graph.variables.push_back({.id = id,
                                   .name = name,
                                   .type = *type,
                                   .default_value = *parsed_default,
                                   .exposed = variable_json["exposed"].get<bool>()});
    }

    const auto nodes_iterator = graph_json.find("nodes");
    if (nodes_iterator == graph_json.end() || !nodes_iterator->is_array())
    {
        add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_NODES_SCHEMA",
                       "Flow graph nodes must be an array.");
        return std::nullopt;
    }

    for (const json& node_json : *nodes_iterator)
    {
        if (!node_json.is_object())
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_NODE_SCHEMA",
                           "Flow node entries must be objects.");
            continue;
        }

        const std::string id = node_json.value("id", std::string{});
        const std::string type_name = node_json.value("type", std::string{});
        const auto kind = parse_node_kind(type_name);
        if (id.empty() || !kind)
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_NODE_TYPE",
                           "Flow node has an empty id or unsupported node type.", id);
            continue;
        }

        const auto values_iterator = node_json.find("values");
        if (values_iterator == node_json.end() || !values_iterator->is_object())
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_NODE_VALUES",
                           "Flow node values must be an object.", id);
            continue;
        }

        graph.nodes.push_back({.id = id, .kind = *kind, .values = *values_iterator});
    }

    const auto connections_iterator = graph_json.find("connections");
    if (connections_iterator == graph_json.end() || !connections_iterator->is_array())
    {
        add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_CONNECTIONS_SCHEMA",
                       "Flow graph connections must be an array.");
        return std::nullopt;
    }

    for (const json& connection_json : *connections_iterator)
    {
        if (!connection_json.is_object())
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_CONNECTION_SCHEMA",
                           "Flow connection entries must be objects.");
            continue;
        }

        const std::string id = connection_json.value("id", std::string{});
        const std::string kind_name = connection_json.value("kind", std::string{});
        const auto from_iterator = connection_json.find("from");
        const auto to_iterator = connection_json.find("to");
        if (id.empty() || (kind_name != "execution" && kind_name != "value") ||
            from_iterator == connection_json.end() || to_iterator == connection_json.end() ||
            !from_iterator->is_object() || !to_iterator->is_object())
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_CONNECTION_SCHEMA",
                           "Flow connection requires id, kind, from, and to fields.", {}, {}, id);
            continue;
        }

        source_connection connection;
        connection.id = id;
        connection.kind = kind_name == "execution" ? connection_kind::execution : connection_kind::value;
        connection.from.node_id = from_iterator->value("nodeId", std::string{});
        connection.from.pin = from_iterator->value("pin", std::string{});
        connection.to.node_id = to_iterator->value("nodeId", std::string{});
        connection.to.pin = to_iterator->value("pin", std::string{});
        if (connection.from.node_id.empty() || connection.from.pin.empty() || connection.to.node_id.empty() ||
            connection.to.pin.empty())
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_CONNECTION_ENDPOINT",
                           "Flow connection endpoints require nodeId and pin.", {}, {}, id);
            continue;
        }
        graph.connections.push_back(std::move(connection));
    }

    return graph;
}

struct validation_state
{
    std::unordered_map<std::string, const source_node*> nodes;
    std::unordered_map<std::string, const source_connection*> incoming;
    std::unordered_map<std::string, const source_connection*> execution_outgoing;
    std::unordered_map<std::string, std::vector<std::string>> execution_adjacency;
    std::unordered_set<std::string> reachable;
};

validation_state validate_graph(const source_graph& graph, std::vector<diagnostic>& diagnostics)
{
    validation_state state;

    std::unordered_set<std::string> variable_ids;
    for (const variable& item : graph.variables)
    {
        if (!variable_ids.insert(item.id).second)
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_DUPLICATE_VARIABLE_ID",
                           "Flow variable ids must be unique.", item.id);
    }

    for (const source_node& node : graph.nodes)
    {
        if (!state.nodes.emplace(node.id, &node).second)
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_DUPLICATE_NODE_ID",
                           "Flow node ids must be unique.", node.id);

        if (node.kind == node_kind::input_action)
        {
            const auto action = node.values.find("action");
            if (action == node.values.end() || !action->is_string() || action->get<std::string>().empty())
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_INPUT_ACTION_NAME",
                               "Input Action nodes require a non-empty action name.", node.id);
        }

        if (node.kind == node_kind::has_core_component || node.kind == node_kind::remove_core_component)
        {
            const auto component = node.values.find("component");
            if (component == node.values.end() || !component->is_string() ||
                !parse_core_component(component->get<std::string>()))
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_CORE_COMPONENT",
                               "Core-component nodes require one of name, transform, tag, or active.", node.id,
                               "component");
        }

        if (const auto type = literal_type(node.kind))
        {
            const auto value = node.values.find("value");
            if (value == node.values.end() || !parse_default_value(*type, *value))
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_LITERAL_VALUE",
                               "Flow literal value does not match its node type.", node.id, "value");
        }
    }

    std::unordered_set<std::string> connection_ids;
    for (const source_connection& connection : graph.connections)
    {
        if (!connection_ids.insert(connection.id).second)
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_DUPLICATE_CONNECTION_ID",
                           "Flow connection ids must be unique.", {}, {}, connection.id);

        const auto from_node_iterator = state.nodes.find(connection.from.node_id);
        const auto to_node_iterator = state.nodes.find(connection.to.node_id);
        if (from_node_iterator == state.nodes.end() || to_node_iterator == state.nodes.end())
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_CONNECTION_NODE",
                           "Flow connection references a node that does not exist.", {}, {}, connection.id);
            continue;
        }
        if (connection.from.node_id == connection.to.node_id)
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_SELF_CONNECTION",
                           "Flow nodes cannot connect to themselves.", connection.from.node_id, connection.from.pin,
                           connection.id);
            continue;
        }

        const auto from_pin = output_pin(from_node_iterator->second->kind, connection.from.pin);
        const auto to_pin = input_pin(to_node_iterator->second->kind, connection.to.pin);
        if (!from_pin || !to_pin)
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_PIN_NOT_FOUND",
                           "Flow connection references an unknown output or input pin.",
                           !from_pin ? connection.from.node_id : connection.to.node_id,
                           !from_pin ? connection.from.pin : connection.to.pin, connection.id);
            continue;
        }

        const pin_kind expected_kind =
            connection.kind == connection_kind::execution ? pin_kind::execution : pin_kind::value;
        if (from_pin->kind != expected_kind || to_pin->kind != expected_kind)
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_CONNECTION_KIND",
                           "Flow connection kind does not match its endpoint pins.", connection.to.node_id,
                           connection.to.pin, connection.id);
            continue;
        }
        if (expected_kind == pin_kind::value && from_pin->type != to_pin->type)
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VALUE_TYPE_MISMATCH",
                           "Flow value connection types are incompatible.", connection.to.node_id, connection.to.pin,
                           connection.id);
            continue;
        }

        const std::string incoming_key = pin_key(connection.to.node_id, connection.to.pin);
        if (!state.incoming.emplace(incoming_key, &connection).second)
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_MULTIPLE_INPUTS",
                           "A Flow input pin can have only one incoming connection.", connection.to.node_id,
                           connection.to.pin, connection.id);

        if (connection.kind == connection_kind::execution)
        {
            const std::string outgoing_key = pin_key(connection.from.node_id, connection.from.pin);
            if (!state.execution_outgoing.emplace(outgoing_key, &connection).second)
                add_diagnostic(
                    diagnostics, diagnostic_severity::error, "FLOW_EXECUTION_FANOUT",
                    "Execution outputs must have a single target; use an explicit Sequence node for fan-out.",
                    connection.from.node_id, connection.from.pin, connection.id);
            state.execution_adjacency[connection.from.node_id].push_back(connection.to.node_id);
        }
    }

    const auto require_input = [&](const source_node& node, std::string_view pin)
    {
        if (state.incoming.find(pin_key(node.id, pin)) == state.incoming.end())
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_REQUIRED_INPUT",
                           "Gameplay node requires this value input to be connected.", node.id, std::string{pin});
    };

    for (const source_node& node : graph.nodes)
    {
        switch (node.kind)
        {
            case node_kind::destroy_entity:
            case node_kind::has_core_component:
            case node_kind::remove_core_component:
            case node_kind::entity_alive:
            case node_kind::get_name:
            case node_kind::get_tag:
            case node_kind::get_active:
            case node_kind::get_transform:
                require_input(node, "entity");
                break;
            case node_kind::set_name:
                require_input(node, "entity");
                require_input(node, "name");
                break;
            case node_kind::set_tag:
                require_input(node, "entity");
                require_input(node, "tag");
                break;
            case node_kind::set_active:
                require_input(node, "entity");
                require_input(node, "active");
                break;
            case node_kind::set_transform:
                require_input(node, "entity");
                require_input(node, "position");
                require_input(node, "rotation");
                require_input(node, "scale");
                break;
            default:
                break;
        }
    }

    enum class visit_state : std::uint8_t
    {
        visiting,
        visited,
    };
    std::unordered_map<std::string, visit_state> visits;
    std::function<bool(const std::string&)> visit = [&](const std::string& node_id)
    {
        const auto existing = visits.find(node_id);
        if (existing != visits.end())
        {
            if (existing->second == visit_state::visiting)
            {
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_EXECUTION_CYCLE",
                               "Execution cycles are not allowed unless represented by an explicit loop node.",
                               node_id);
                return false;
            }
            return true;
        }

        visits[node_id] = visit_state::visiting;
        const auto adjacency = state.execution_adjacency.find(node_id);
        if (adjacency != state.execution_adjacency.end())
            for (const std::string& target : adjacency->second)
                if (!visit(target)) return false;
        visits[node_id] = visit_state::visited;
        return true;
    };

    for (const source_node& node : graph.nodes)
    {
        if (visits.find(node.id) == visits.end() && !visit(node.id)) break;
    }

    std::queue<std::string> pending;
    for (const source_node& node : graph.nodes)
    {
        if (!is_event_node(node.kind)) continue;
        state.reachable.insert(node.id);
        pending.push(node.id);
    }
    while (!pending.empty())
    {
        const std::string current = pending.front();
        pending.pop();
        const auto adjacency = state.execution_adjacency.find(current);
        if (adjacency == state.execution_adjacency.end()) continue;
        for (const std::string& target : adjacency->second)
            if (state.reachable.insert(target).second) pending.push(target);
    }
    for (const source_node& node : graph.nodes)
    {
        if (is_executable_node(node.kind) && state.reachable.find(node.id) == state.reachable.end())
            add_diagnostic(diagnostics, diagnostic_severity::warning, "FLOW_UNREACHABLE_NODE",
                           "Flow node is not reachable from an event entry point.", node.id);
    }

    return state;
}

std::vector<const source_node*> sorted_nodes(const source_graph& graph)
{
    std::vector<const source_node*> result;
    result.reserve(graph.nodes.size());
    for (const source_node& node : graph.nodes)
        result.push_back(&node);
    std::sort(result.begin(), result.end(),
              [](const source_node* left, const source_node* right) { return left->id < right->id; });
    return result;
}

ir_opcode opcode_for(node_kind kind)
{
    switch (kind)
    {
        case node_kind::branch:
            return ir_opcode::branch;
        case node_kind::create_entity:
            return ir_opcode::world_create_entity;
        case node_kind::destroy_entity:
            return ir_opcode::world_destroy_entity;
        case node_kind::has_core_component:
            return ir_opcode::world_has_core_component;
        case node_kind::remove_core_component:
            return ir_opcode::world_remove_core_component;
        case node_kind::entity_alive:
            return ir_opcode::world_entity_alive;
        case node_kind::get_name:
            return ir_opcode::world_get_name;
        case node_kind::set_name:
            return ir_opcode::world_set_name;
        case node_kind::get_tag:
            return ir_opcode::world_get_tag;
        case node_kind::set_tag:
            return ir_opcode::world_set_tag;
        case node_kind::get_active:
            return ir_opcode::world_get_active;
        case node_kind::set_active:
            return ir_opcode::world_set_active;
        case node_kind::get_transform:
            return ir_opcode::world_get_transform;
        case node_kind::set_transform:
            return ir_opcode::world_set_transform;
        default:
            return ir_opcode::branch;
    }
}

ir_program build_ir(const source_graph& graph, const validation_state& validation)
{
    ir_program program;
    program.variables = graph.variables;
    std::sort(program.variables.begin(), program.variables.end(),
              [](const variable& left, const variable& right) { return left.id < right.id; });

    const auto nodes = sorted_nodes(graph);
    std::unordered_map<std::string, std::uint32_t> value_slots;
    auto allocate_slot = [&](const source_node& node, std::string_view pin, value_type type, flow_value initial)
    {
        const std::string key = pin_key(node.id, pin);
        const auto existing = value_slots.find(key);
        if (existing != value_slots.end()) return existing->second;
        const auto index = static_cast<std::uint32_t>(program.value_slots.size());
        program.value_slots.push_back({.index = index,
                                       .type = type,
                                       .initial_value = std::move(initial),
                                       .source_node_id = node.id,
                                       .source_pin_id = std::string{pin}});
        value_slots.emplace(key, index);
        return index;
    };

    for (const source_node* node : nodes)
    {
        switch (node->kind)
        {
            case node_kind::tick:
            case node_kind::fixed_tick:
                allocate_slot(*node, "deltaSeconds", value_type::float32, default_value(value_type::float32));
                break;
            case node_kind::input_action:
                allocate_slot(*node, "value", value_type::float32, default_value(value_type::float32));
                break;
            case node_kind::self_entity:
            case node_kind::create_entity:
                allocate_slot(*node, "entity", value_type::entity, default_value(value_type::entity));
                break;
            case node_kind::has_core_component:
                allocate_slot(*node, "has", value_type::boolean, false);
                break;
            case node_kind::entity_alive:
                allocate_slot(*node, "alive", value_type::boolean, false);
                break;
            case node_kind::get_name:
                allocate_slot(*node, "name", value_type::string, std::string{});
                break;
            case node_kind::get_tag:
                allocate_slot(*node, "tag", value_type::string, std::string{});
                break;
            case node_kind::get_active:
                allocate_slot(*node, "active", value_type::boolean, false);
                break;
            case node_kind::get_transform:
                allocate_slot(*node, "position", value_type::vector3, default_value(value_type::vector3));
                allocate_slot(*node, "rotation", value_type::vector4,
                              flow_value{std::array<double, 4>{0.0, 0.0, 0.0, 1.0}});
                allocate_slot(*node, "scale", value_type::vector3, flow_value{std::array<double, 3>{1.0, 1.0, 1.0}});
                break;
            case node_kind::bool_literal:
            case node_kind::string_literal:
            case node_kind::vector3_literal:
            case node_kind::vector4_literal:
            {
                const value_type type = *literal_type(node->kind);
                allocate_slot(*node, "value", type, *parse_default_value(type, node->values.at("value")));
                break;
            }
            default:
                break;
        }
    }

    const auto input_slot = [&](const source_node& node, std::string_view pin)
    {
        const source_connection* connection = validation.incoming.at(pin_key(node.id, pin));
        return value_slots.at(pin_key(connection->from.node_id, connection->from.pin));
    };

    std::unordered_map<std::string, std::uint32_t> instruction_indices;
    for (const source_node* node : nodes)
    {
        if (!is_executable_node(node->kind)) continue;
        const auto index = static_cast<std::uint32_t>(program.instructions.size());
        instruction_indices.emplace(node->id, index);
        program.instructions.push_back({.opcode = opcode_for(node->kind), .node_id = node->id});
    }

    const auto execution_target = [&](const source_node& node, std::string_view pin)
    {
        const auto connection = validation.execution_outgoing.find(pin_key(node.id, pin));
        if (connection == validation.execution_outgoing.end()) return invalid_instruction;
        const auto target = instruction_indices.find(connection->second->to.node_id);
        return target == instruction_indices.end() ? invalid_instruction : target->second;
    };

    for (const source_node* node : nodes)
    {
        if (!is_executable_node(node->kind)) continue;
        ir_instruction& instruction = program.instructions[instruction_indices.at(node->id)];
        switch (node->kind)
        {
            case node_kind::branch:
            {
                const auto condition_connection = validation.incoming.find(pin_key(node->id, "condition"));
                if (condition_connection == validation.incoming.end())
                    instruction.condition_slot = allocate_slot(*node, "condition", value_type::boolean, false);
                else
                    instruction.condition_slot = value_slots.at(
                        pin_key(condition_connection->second->from.node_id, condition_connection->second->from.pin));
                instruction.true_instruction = execution_target(*node, "true");
                instruction.false_instruction = execution_target(*node, "false");
                break;
            }
            case node_kind::create_entity:
                instruction.operand0 = value_slots.at(pin_key(node->id, "entity"));
                instruction.operand1 = execution_target(*node, "then");
                break;
            case node_kind::destroy_entity:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = execution_target(*node, "then");
                break;
            case node_kind::has_core_component:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = static_cast<std::uint32_t>(
                    *parse_core_component(node->values.at("component").get<std::string>()));
                instruction.operand2 = value_slots.at(pin_key(node->id, "has"));
                instruction.operand3 = execution_target(*node, "then");
                break;
            case node_kind::remove_core_component:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = static_cast<std::uint32_t>(
                    *parse_core_component(node->values.at("component").get<std::string>()));
                instruction.operand2 = execution_target(*node, "then");
                break;
            case node_kind::entity_alive:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = value_slots.at(pin_key(node->id, "alive"));
                instruction.operand2 = execution_target(*node, "then");
                break;
            case node_kind::get_name:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = value_slots.at(pin_key(node->id, "name"));
                instruction.operand2 = execution_target(*node, "then");
                break;
            case node_kind::set_name:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = input_slot(*node, "name");
                instruction.operand2 = execution_target(*node, "then");
                break;
            case node_kind::get_tag:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = value_slots.at(pin_key(node->id, "tag"));
                instruction.operand2 = execution_target(*node, "then");
                break;
            case node_kind::set_tag:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = input_slot(*node, "tag");
                instruction.operand2 = execution_target(*node, "then");
                break;
            case node_kind::get_active:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = value_slots.at(pin_key(node->id, "active"));
                instruction.operand2 = execution_target(*node, "then");
                break;
            case node_kind::set_active:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = input_slot(*node, "active");
                instruction.operand2 = execution_target(*node, "then");
                break;
            case node_kind::get_transform:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = value_slots.at(pin_key(node->id, "position"));
                instruction.operand2 = value_slots.at(pin_key(node->id, "rotation"));
                instruction.operand3 = value_slots.at(pin_key(node->id, "scale"));
                instruction.operand4 = execution_target(*node, "then");
                break;
            case node_kind::set_transform:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 = input_slot(*node, "position");
                instruction.operand2 = input_slot(*node, "rotation");
                instruction.operand3 = input_slot(*node, "scale");
                instruction.operand4 = execution_target(*node, "then");
                break;
            default:
                break;
        }
    }

    const auto add_binding =
        [&](ir_entry_point& entry, const source_node& node, std::string_view pin, entry_value_kind source)
    { entry.value_bindings.push_back({.source = source, .slot = value_slots.at(pin_key(node.id, pin))}); };

    for (const source_node* node : nodes)
    {
        switch (node->kind)
        {
            case node_kind::begin_play:
                program.entry_points.push_back({.kind = entry_point_kind::begin_play,
                                                .node_id = node->id,
                                                .instruction = execution_target(*node, "exec")});
                break;
            case node_kind::end_play:
                program.entry_points.push_back({.kind = entry_point_kind::end_play,
                                                .node_id = node->id,
                                                .instruction = execution_target(*node, "exec")});
                break;
            case node_kind::tick:
            {
                ir_entry_point entry{.kind = entry_point_kind::tick,
                                     .node_id = node->id,
                                     .instruction = execution_target(*node, "exec")};
                add_binding(entry, *node, "deltaSeconds", entry_value_kind::delta_seconds);
                program.entry_points.push_back(std::move(entry));
                break;
            }
            case node_kind::fixed_tick:
            {
                ir_entry_point entry{.kind = entry_point_kind::fixed_tick,
                                     .node_id = node->id,
                                     .instruction = execution_target(*node, "exec")};
                add_binding(entry, *node, "deltaSeconds", entry_value_kind::delta_seconds);
                program.entry_points.push_back(std::move(entry));
                break;
            }
            case node_kind::input_action:
            {
                const std::string action = node->values.at("action").get<std::string>();
                ir_entry_point triggered{.kind = entry_point_kind::input_action_triggered,
                                         .node_id = node->id,
                                         .action = action,
                                         .instruction = execution_target(*node, "triggered")};
                add_binding(triggered, *node, "value", entry_value_kind::input_action_value);
                program.entry_points.push_back(std::move(triggered));

                ir_entry_point completed{.kind = entry_point_kind::input_action_completed,
                                         .node_id = node->id,
                                         .action = action,
                                         .instruction = execution_target(*node, "completed")};
                add_binding(completed, *node, "value", entry_value_kind::input_action_value);
                program.entry_points.push_back(std::move(completed));
                break;
            }
            default:
                break;
        }
    }

    const auto reachable_from = [&](std::string_view start, std::string_view target)
    {
        if (start == target) return true;
        std::queue<std::string> pending;
        std::unordered_set<std::string> seen;
        pending.push(std::string{start});
        seen.insert(std::string{start});
        while (!pending.empty())
        {
            const std::string current = pending.front();
            pending.pop();
            const auto adjacency = validation.execution_adjacency.find(current);
            if (adjacency == validation.execution_adjacency.end()) continue;
            for (const std::string& next : adjacency->second)
            {
                if (next == target) return true;
                if (seen.insert(next).second) pending.push(next);
            }
        }
        return false;
    };

    std::vector<const source_node*> self_nodes;
    for (const source_node* node : nodes)
        if (node->kind == node_kind::self_entity) self_nodes.push_back(node);

    for (ir_entry_point& entry : program.entry_points)
    {
        std::uint32_t target = entry.instruction;
        for (auto iterator = self_nodes.rbegin(); iterator != self_nodes.rend(); ++iterator)
        {
            const source_node& self = **iterator;
            bool needed = false;
            for (const source_connection& connection : graph.connections)
            {
                if (connection.kind != connection_kind::value || connection.from.node_id != self.id) continue;
                if (reachable_from(entry.node_id, connection.to.node_id))
                {
                    needed = true;
                    break;
                }
            }
            if (!needed) continue;

            const auto index = static_cast<std::uint32_t>(program.instructions.size());
            program.instructions.push_back({.opcode = ir_opcode::self_entity,
                                            .node_id = self.id,
                                            .operand0 = value_slots.at(pin_key(self.id, "entity")),
                                            .operand1 = target});
            target = index;
        }
        entry.instruction = target;
    }

    return program;
}

bytecode_opcode lower_opcode(ir_opcode opcode)
{
    switch (opcode)
    {
        case ir_opcode::branch:
            return bytecode_opcode::branch;
        case ir_opcode::self_entity:
            return bytecode_opcode::self_entity;
        case ir_opcode::world_create_entity:
            return bytecode_opcode::world_create_entity;
        case ir_opcode::world_destroy_entity:
            return bytecode_opcode::world_destroy_entity;
        case ir_opcode::world_has_core_component:
            return bytecode_opcode::world_has_core_component;
        case ir_opcode::world_remove_core_component:
            return bytecode_opcode::world_remove_core_component;
        case ir_opcode::world_entity_alive:
            return bytecode_opcode::world_entity_alive;
        case ir_opcode::world_get_name:
            return bytecode_opcode::world_get_name;
        case ir_opcode::world_set_name:
            return bytecode_opcode::world_set_name;
        case ir_opcode::world_get_transform:
            return bytecode_opcode::world_get_transform;
        case ir_opcode::world_set_transform:
            return bytecode_opcode::world_set_transform;
        case ir_opcode::world_get_tag:
            return bytecode_opcode::world_get_tag;
        case ir_opcode::world_set_tag:
            return bytecode_opcode::world_set_tag;
        case ir_opcode::world_get_active:
            return bytecode_opcode::world_get_active;
        case ir_opcode::world_set_active:
            return bytecode_opcode::world_set_active;
    }
    return bytecode_opcode::branch;
}

bytecode_program lower_bytecode(const ir_program& ir)
{
    bytecode_program bytecode;
    bytecode.variables = ir.variables;
    bytecode.value_slots.reserve(ir.value_slots.size());
    for (const ir_value_slot& slot : ir.value_slots)
        bytecode.value_slots.push_back({.type = slot.type, .initial_value = slot.initial_value});

    bytecode.entry_points.reserve(ir.entry_points.size());
    for (const ir_entry_point& entry : ir.entry_points)
        bytecode.entry_points.push_back({.kind = entry.kind,
                                         .action = entry.action,
                                         .instruction = entry.instruction,
                                         .value_bindings = entry.value_bindings});

    bytecode.instructions.reserve(ir.instructions.size());
    bytecode.instruction_nodes.reserve(ir.instructions.size());
    for (const ir_instruction& instruction : ir.instructions)
    {
        if (instruction.opcode == ir_opcode::branch)
        {
            bytecode.instructions.push_back({.opcode = bytecode_opcode::branch,
                                             .operand0 = instruction.condition_slot,
                                             .operand1 = instruction.true_instruction,
                                             .operand2 = instruction.false_instruction});
        }
        else
        {
            bytecode.instructions.push_back({.opcode = lower_opcode(instruction.opcode),
                                             .operand0 = instruction.operand0,
                                             .operand1 = instruction.operand1,
                                             .operand2 = instruction.operand2,
                                             .operand3 = instruction.operand3,
                                             .operand4 = instruction.operand4});
        }
        bytecode.instruction_nodes.push_back(instruction.node_id);
    }
    return bytecode;
}

} // namespace

compile_result compile_asset(std::string_view source)
{
    compile_result result;
    const auto graph = parse_source(source, result.diagnostics);
    if (!graph) return result;

    const validation_state validation = validate_graph(*graph, result.diagnostics);
    if (has_errors(result.diagnostics)) return result;

    result.ir = build_ir(*graph, validation);
    result.bytecode = lower_bytecode(*result.ir);
    result.succeeded = true;
    return result;
}

} // namespace arc::flow
