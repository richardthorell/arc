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
    sequence,
    switch_integer,
    do_once,
    gate,
    for_loop,
    while_loop,
    delay,
    retriggerable_delay,
    timer,
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
    int_literal,
    float_literal,
    vector2_literal,
    string_literal,
    vector3_literal,
    vector4_literal,
    get_variable,
    set_variable,
    add,
    subtract,
    multiply,
    divide,
    compare,
    boolean_and,
    boolean_or,
    boolean_not,
    vector_dot,
    vector_length,
    vector_normalize,
    vector_scale,
    select,
    convert_number,
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
    if (value == "sequence") return node_kind::sequence;
    if (value == "switchInt") return node_kind::switch_integer;
    if (value == "doOnce") return node_kind::do_once;
    if (value == "gate") return node_kind::gate;
    if (value == "forLoop") return node_kind::for_loop;
    if (value == "whileLoop") return node_kind::while_loop;
    if (value == "delay") return node_kind::delay;
    if (value == "retriggerableDelay") return node_kind::retriggerable_delay;
    if (value == "timer") return node_kind::timer;
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
    if (value == "intLiteral") return node_kind::int_literal;
    if (value == "floatLiteral") return node_kind::float_literal;
    if (value == "vector2Literal") return node_kind::vector2_literal;
    if (value == "stringLiteral") return node_kind::string_literal;
    if (value == "vector3Literal") return node_kind::vector3_literal;
    if (value == "vector4Literal") return node_kind::vector4_literal;
    if (value == "getVariable") return node_kind::get_variable;
    if (value == "setVariable") return node_kind::set_variable;
    if (value == "add") return node_kind::add;
    if (value == "subtract") return node_kind::subtract;
    if (value == "multiply") return node_kind::multiply;
    if (value == "divide") return node_kind::divide;
    if (value == "compare") return node_kind::compare;
    if (value == "boolAnd") return node_kind::boolean_and;
    if (value == "boolOr") return node_kind::boolean_or;
    if (value == "boolNot") return node_kind::boolean_not;
    if (value == "vectorDot") return node_kind::vector_dot;
    if (value == "vectorLength") return node_kind::vector_length;
    if (value == "vectorNormalize") return node_kind::vector_normalize;
    if (value == "vectorScale") return node_kind::vector_scale;
    if (value == "select") return node_kind::select;
    if (value == "convertNumber") return node_kind::convert_number;
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

bool is_numeric_type(value_type type)
{
    return type == value_type::integer || type == value_type::float32;
}

bool is_arithmetic_type(value_type type)
{
    return is_numeric_type(type) || type == value_type::vector2 || type == value_type::vector3 ||
           type == value_type::vector4;
}

bool is_vector_type(value_type type)
{
    return type == value_type::vector2 || type == value_type::vector3 || type == value_type::vector4;
}

bool is_select_type(value_type type)
{
    return type != value_type::component;
}

std::optional<value_type> configured_type(const source_node& node)
{
    const auto iterator = node.values.find("valueType");
    if (iterator == node.values.end() || !iterator->is_string()) return std::nullopt;
    return parse_value_type(iterator->get<std::string>());
}

const variable* variable_for(const source_graph& graph, const source_node& node)
{
    const auto iterator = node.values.find("variableId");
    if (iterator == node.values.end() || !iterator->is_string()) return nullptr;
    const std::string id = iterator->get<std::string>();
    const auto found = std::find_if(graph.variables.begin(), graph.variables.end(),
                                    [&id](const variable& item) { return item.id == id; });
    return found == graph.variables.end() ? nullptr : &*found;
}

std::optional<value_type> literal_type(node_kind kind)
{
    switch (kind)
    {
        case node_kind::bool_literal:
            return value_type::boolean;
        case node_kind::int_literal:
            return value_type::integer;
        case node_kind::float_literal:
            return value_type::float32;
        case node_kind::vector2_literal:
            return value_type::vector2;
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

std::optional<value_type> node_data_type(const source_graph& graph, const source_node& node)
{
    if (const auto type = literal_type(node.kind)) return type;
    if (node.kind == node_kind::get_variable || node.kind == node_kind::set_variable)
    {
        const variable* item = variable_for(graph, node);
        return item ? std::optional<value_type>{item->type} : std::nullopt;
    }
    if (node.kind == node_kind::convert_number)
    {
        const auto conversion = node.values.find("conversion");
        if (conversion == node.values.end() || !conversion->is_string()) return std::nullopt;
        if (conversion->get<std::string>() == "intToFloat") return value_type::float32;
        if (conversion->get<std::string>() == "floatToInt") return value_type::integer;
        return std::nullopt;
    }
    return configured_type(node);
}

std::optional<value_type> convert_input_type(const source_node& node)
{
    const auto conversion = node.values.find("conversion");
    if (conversion == node.values.end() || !conversion->is_string()) return std::nullopt;
    if (conversion->get<std::string>() == "intToFloat") return value_type::integer;
    if (conversion->get<std::string>() == "floatToInt") return value_type::float32;
    return std::nullopt;
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
        case node_kind::sequence:
        case node_kind::switch_integer:
        case node_kind::do_once:
        case node_kind::gate:
        case node_kind::for_loop:
        case node_kind::while_loop:
        case node_kind::delay:
        case node_kind::retriggerable_delay:
        case node_kind::timer:
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
        case node_kind::set_variable:
            return true;
        default:
            return false;
    }
}

bool is_latent_execution_edge(node_kind kind, std::string_view pin)
{
    if ((kind == node_kind::delay || kind == node_kind::retriggerable_delay) && pin == "completed") return true;
    return kind == node_kind::timer && (pin == "tick" || pin == "completed");
}

bool is_computed_value_node(node_kind kind)
{
    switch (kind)
    {
        case node_kind::get_variable:
        case node_kind::add:
        case node_kind::subtract:
        case node_kind::multiply:
        case node_kind::divide:
        case node_kind::compare:
        case node_kind::boolean_and:
        case node_kind::boolean_or:
        case node_kind::boolean_not:
        case node_kind::vector_dot:
        case node_kind::vector_length:
        case node_kind::vector_normalize:
        case node_kind::vector_scale:
        case node_kind::select:
        case node_kind::convert_number:
            return true;
        default:
            return false;
    }
}

std::optional<pin_info> output_pin(const source_graph& graph, const source_node& node, std::string_view pin)
{
    switch (node.kind)
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
        case node_kind::sequence:
            if (pin == "then0" || pin == "then1" || pin == "then2" || pin == "then3")
                return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::switch_integer:
            if (pin == "case0" || pin == "case1" || pin == "case2" || pin == "case3" || pin == "default")
                return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::do_once:
            if (pin == "then") return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::gate:
            if (pin == "exit") return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::for_loop:
            if (pin == "loopBody" || pin == "completed") return pin_info{.kind = pin_kind::execution};
            if (pin == "index") return pin_info{.kind = pin_kind::value, .type = value_type::integer};
            break;
        case node_kind::while_loop:
            if (pin == "loopBody" || pin == "completed") return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::delay:
        case node_kind::retriggerable_delay:
            if (pin == "completed") return pin_info{.kind = pin_kind::execution};
            break;
        case node_kind::timer:
            if (pin == "started" || pin == "tick" || pin == "completed" || pin == "stopped")
                return pin_info{.kind = pin_kind::execution};
            if (pin == "active") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
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
        case node_kind::set_variable:
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
        case node_kind::int_literal:
        case node_kind::float_literal:
        case node_kind::vector2_literal:
        case node_kind::string_literal:
        case node_kind::vector3_literal:
        case node_kind::vector4_literal:
            if (pin == "value") return pin_info{.kind = pin_kind::value, .type = literal_type(node.kind)};
            break;
        case node_kind::get_variable:
        case node_kind::add:
        case node_kind::subtract:
        case node_kind::multiply:
        case node_kind::divide:
        case node_kind::vector_normalize:
        case node_kind::vector_scale:
        case node_kind::select:
        case node_kind::convert_number:
            if (pin == "value") return pin_info{.kind = pin_kind::value, .type = node_data_type(graph, node)};
            break;
        case node_kind::compare:
        case node_kind::boolean_and:
        case node_kind::boolean_or:
        case node_kind::boolean_not:
            if (pin == "result") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
            break;
        case node_kind::vector_dot:
        case node_kind::vector_length:
            if (pin == "value") return pin_info{.kind = pin_kind::value, .type = value_type::float32};
            break;
    }
    return std::nullopt;
}

std::optional<pin_info> input_pin(const source_graph& graph, const source_node& node, std::string_view pin)
{
    if (node.kind == node_kind::sequence)
    {
        if (pin == "exec") return pin_info{.kind = pin_kind::execution};
        return std::nullopt;
    }
    if (node.kind == node_kind::switch_integer)
    {
        if (pin == "exec") return pin_info{.kind = pin_kind::execution};
        if (pin == "selection") return pin_info{.kind = pin_kind::value, .type = value_type::integer};
        return std::nullopt;
    }
    if (node.kind == node_kind::do_once)
    {
        if (pin == "exec" || pin == "reset") return pin_info{.kind = pin_kind::execution};
        return std::nullopt;
    }
    if (node.kind == node_kind::gate)
    {
        if (pin == "enter" || pin == "open" || pin == "close" || pin == "toggle")
            return pin_info{.kind = pin_kind::execution};
        return std::nullopt;
    }
    if (node.kind == node_kind::for_loop)
    {
        if (pin == "exec") return pin_info{.kind = pin_kind::execution};
        if (pin == "first" || pin == "last") return pin_info{.kind = pin_kind::value, .type = value_type::integer};
        return std::nullopt;
    }
    if (node.kind == node_kind::while_loop)
    {
        if (pin == "exec") return pin_info{.kind = pin_kind::execution};
        if (pin == "condition") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
        return std::nullopt;
    }
    if (node.kind == node_kind::delay || node.kind == node_kind::retriggerable_delay)
    {
        if (pin == "exec") return pin_info{.kind = pin_kind::execution};
        if (pin == "duration") return pin_info{.kind = pin_kind::value, .type = value_type::float32};
        return std::nullopt;
    }
    if (node.kind == node_kind::timer)
    {
        if (pin == "start" || pin == "stop") return pin_info{.kind = pin_kind::execution};
        if (pin == "interval") return pin_info{.kind = pin_kind::value, .type = value_type::float32};
        if (pin == "looping") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
        return std::nullopt;
    }

    if (node.kind == node_kind::branch)
    {
        if (pin == "exec") return pin_info{.kind = pin_kind::execution};
        if (pin == "condition") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
        return std::nullopt;
    }

    if (node.kind == node_kind::set_variable)
    {
        if (pin == "exec") return pin_info{.kind = pin_kind::execution};
        if (pin == "value") return pin_info{.kind = pin_kind::value, .type = node_data_type(graph, node)};
        return std::nullopt;
    }

    if (node.kind == node_kind::add || node.kind == node_kind::subtract || node.kind == node_kind::multiply ||
        node.kind == node_kind::divide || node.kind == node_kind::compare || node.kind == node_kind::vector_dot)
    {
        if (pin == "a" || pin == "b") return pin_info{.kind = pin_kind::value, .type = configured_type(node)};
        return std::nullopt;
    }
    if (node.kind == node_kind::boolean_and || node.kind == node_kind::boolean_or)
    {
        if (pin == "a" || pin == "b") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
        return std::nullopt;
    }
    if (node.kind == node_kind::boolean_not)
    {
        if (pin == "value") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
        return std::nullopt;
    }
    if (node.kind == node_kind::vector_length || node.kind == node_kind::vector_normalize)
    {
        if (pin == "value") return pin_info{.kind = pin_kind::value, .type = configured_type(node)};
        return std::nullopt;
    }
    if (node.kind == node_kind::vector_scale)
    {
        if (pin == "vector") return pin_info{.kind = pin_kind::value, .type = configured_type(node)};
        if (pin == "scale") return pin_info{.kind = pin_kind::value, .type = value_type::float32};
        return std::nullopt;
    }
    if (node.kind == node_kind::select)
    {
        if (pin == "condition") return pin_info{.kind = pin_kind::value, .type = value_type::boolean};
        if (pin == "trueValue" || pin == "falseValue")
            return pin_info{.kind = pin_kind::value, .type = configured_type(node)};
        return std::nullopt;
    }
    if (node.kind == node_kind::convert_number)
    {
        if (pin == "value") return pin_info{.kind = pin_kind::value, .type = convert_input_type(node)};
        return std::nullopt;
    }

    if (!is_executable_node(node.kind)) return std::nullopt;
    if (pin == "exec") return pin_info{.kind = pin_kind::execution};
    if (node.kind != node_kind::create_entity && pin == "entity")
        return pin_info{.kind = pin_kind::value, .type = value_type::entity};

    switch (node.kind)
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
    std::unordered_map<std::string, std::vector<std::string>> execution_cycle_adjacency;
    std::unordered_map<std::string, std::vector<std::string>> value_adjacency;
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

        if (node.kind == node_kind::get_variable || node.kind == node_kind::set_variable)
        {
            if (!variable_for(graph, node))
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VARIABLE_REFERENCE",
                               "Variable node references a variable that does not exist.", node.id, "variableId");
        }

        if (node.kind == node_kind::add || node.kind == node_kind::subtract || node.kind == node_kind::multiply ||
            node.kind == node_kind::divide)
        {
            const auto type = configured_type(node);
            if (!type || !is_arithmetic_type(*type))
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VALUE_TYPE",
                               "Arithmetic nodes require int, float, vec2, vec3, or vec4 valueType.", node.id,
                               "valueType");
        }
        if (node.kind == node_kind::compare)
        {
            const auto type = configured_type(node);
            if (!type || !is_numeric_type(*type))
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VALUE_TYPE",
                               "Compare nodes require int or float valueType.", node.id, "valueType");
            const auto operation = node.values.find("operator");
            if (operation == node.values.end() || !operation->is_string() ||
                (operation->get<std::string>() != "equal" && operation->get<std::string>() != "notEqual" &&
                 operation->get<std::string>() != "less" && operation->get<std::string>() != "lessEqual" &&
                 operation->get<std::string>() != "greater" && operation->get<std::string>() != "greaterEqual"))
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_COMPARE_OPERATOR",
                               "Compare node has an unsupported operator.", node.id, "operator");
        }
        if (node.kind == node_kind::vector_dot || node.kind == node_kind::vector_length ||
            node.kind == node_kind::vector_normalize || node.kind == node_kind::vector_scale)
        {
            const auto type = configured_type(node);
            if (!type || !is_vector_type(*type))
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VALUE_TYPE",
                               "Vector math nodes require vec2, vec3, or vec4 valueType.", node.id, "valueType");
        }
        if (node.kind == node_kind::select)
        {
            const auto type = configured_type(node);
            if (!type || !is_select_type(*type))
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VALUE_TYPE",
                               "Select requires a concrete selectable valueType.", node.id, "valueType");
        }
        if (node.kind == node_kind::convert_number && (!convert_input_type(node) || !node_data_type(graph, node)))
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_CONVERSION",
                           "Convert Number requires intToFloat or floatToInt conversion.", node.id, "conversion");

        if (node.kind == node_kind::switch_integer)
        {
            const auto cases = node.values.find("cases");
            bool valid = cases != node.values.end() && cases->is_array() && cases->size() == 4;
            std::unordered_set<std::int64_t> unique;
            if (valid)
                for (const json& item : *cases)
                    if (!item.is_number_integer() || !unique.insert(item.get<std::int64_t>()).second)
                    {
                        valid = false;
                        break;
                    }
            if (!valid)
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_SWITCH_CASES",
                               "Switch Integer requires four unique integer case values.", node.id, "cases");
        }
        if (node.kind == node_kind::gate)
        {
            const auto start_closed = node.values.find("startClosed");
            if (start_closed == node.values.end() || !start_closed->is_boolean())
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_GATE_STATE",
                               "Gate requires a boolean startClosed value.", node.id, "startClosed");
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

        const auto from_pin = output_pin(graph, *from_node_iterator->second, connection.from.pin);
        const auto to_pin = input_pin(graph, *to_node_iterator->second, connection.to.pin);
        if (!from_pin || !to_pin || !from_pin->type.has_value() == (from_pin && from_pin->kind == pin_kind::value) ||
            !to_pin->type.has_value() == (to_pin && to_pin->kind == pin_kind::value))
        {
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_PIN_NOT_FOUND",
                           "Flow connection references an unknown or unresolved output/input pin.",
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
            if (!is_latent_execution_edge(from_node_iterator->second->kind, connection.from.pin))
                state.execution_cycle_adjacency[connection.from.node_id].push_back(connection.to.node_id);
        }
        else
        {
            state.value_adjacency[connection.from.node_id].push_back(connection.to.node_id);
        }
    }

    const auto require_input = [&](const source_node& node, std::string_view pin)
    {
        if (state.incoming.find(pin_key(node.id, pin)) == state.incoming.end())
            add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_REQUIRED_INPUT",
                           "Flow node requires this value input to be connected.", node.id, std::string{pin});
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
            case node_kind::set_variable:
            case node_kind::boolean_not:
            case node_kind::vector_length:
            case node_kind::vector_normalize:
            case node_kind::convert_number:
                require_input(node, "value");
                break;
            case node_kind::switch_integer:
                require_input(node, "selection");
                break;
            case node_kind::for_loop:
                require_input(node, "first");
                require_input(node, "last");
                break;
            case node_kind::while_loop:
                require_input(node, "condition");
                break;
            case node_kind::delay:
            case node_kind::retriggerable_delay:
                require_input(node, "duration");
                break;
            case node_kind::timer:
                require_input(node, "interval");
                require_input(node, "looping");
                break;
            case node_kind::add:
            case node_kind::subtract:
            case node_kind::multiply:
            case node_kind::divide:
            case node_kind::compare:
            case node_kind::boolean_and:
            case node_kind::boolean_or:
            case node_kind::vector_dot:
                require_input(node, "a");
                require_input(node, "b");
                break;
            case node_kind::vector_scale:
                require_input(node, "vector");
                require_input(node, "scale");
                break;
            case node_kind::select:
                require_input(node, "condition");
                require_input(node, "trueValue");
                require_input(node, "falseValue");
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
    std::function<bool(const std::string&)> visit_execution = [&](const std::string& node_id)
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
        const auto adjacency = state.execution_cycle_adjacency.find(node_id);
        if (adjacency != state.execution_cycle_adjacency.end())
            for (const std::string& target : adjacency->second)
                if (!visit_execution(target)) return false;
        visits[node_id] = visit_state::visited;
        return true;
    };

    for (const source_node& node : graph.nodes)
    {
        if (visits.find(node.id) == visits.end() && !visit_execution(node.id)) break;
    }

    visits.clear();
    std::function<bool(const std::string&)> visit_value = [&](const std::string& node_id)
    {
        const auto existing = visits.find(node_id);
        if (existing != visits.end())
        {
            if (existing->second == visit_state::visiting)
            {
                add_diagnostic(diagnostics, diagnostic_severity::error, "FLOW_VALUE_CYCLE",
                               "Value dependency cycles are not allowed.", node_id);
                return false;
            }
            return true;
        }
        visits[node_id] = visit_state::visiting;
        const auto adjacency = state.value_adjacency.find(node_id);
        if (adjacency != state.value_adjacency.end())
            for (const std::string& target : adjacency->second)
                if (!visit_value(target)) return false;
        visits[node_id] = visit_state::visited;
        return true;
    };
    for (const source_node& node : graph.nodes)
    {
        if (visits.find(node.id) == visits.end() && !visit_value(node.id)) break;
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

ir_opcode executable_opcode_for(node_kind kind)
{
    switch (kind)
    {
        case node_kind::branch:
            return ir_opcode::branch;
        case node_kind::sequence:
            return ir_opcode::sequence;
        case node_kind::switch_integer:
            return ir_opcode::switch_integer;
        case node_kind::do_once:
            return ir_opcode::do_once;
        case node_kind::gate:
            return ir_opcode::gate_enter;
        case node_kind::for_loop:
            return ir_opcode::for_loop;
        case node_kind::while_loop:
            return ir_opcode::while_loop;
        case node_kind::delay:
            return ir_opcode::delay;
        case node_kind::retriggerable_delay:
            return ir_opcode::retriggerable_delay;
        case node_kind::timer:
            return ir_opcode::timer_start;
        case node_kind::set_variable:
            return ir_opcode::store_variable;
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

ir_opcode compare_opcode(const source_node& node)
{
    const std::string operation = node.values.at("operator").get<std::string>();
    if (operation == "equal") return ir_opcode::compare_equal;
    if (operation == "notEqual") return ir_opcode::compare_not_equal;
    if (operation == "less") return ir_opcode::compare_less;
    if (operation == "lessEqual") return ir_opcode::compare_less_equal;
    if (operation == "greater") return ir_opcode::compare_greater;
    return ir_opcode::compare_greater_equal;
}

std::vector<std::string_view> data_inputs(node_kind kind)
{
    switch (kind)
    {
        case node_kind::add:
        case node_kind::subtract:
        case node_kind::multiply:
        case node_kind::divide:
        case node_kind::compare:
        case node_kind::boolean_and:
        case node_kind::boolean_or:
        case node_kind::vector_dot:
            return {"a", "b"};
        case node_kind::boolean_not:
        case node_kind::vector_length:
        case node_kind::vector_normalize:
        case node_kind::convert_number:
            return {"value"};
        case node_kind::vector_scale:
            return {"vector", "scale"};
        case node_kind::select:
            return {"condition", "trueValue", "falseValue"};
        default:
            return {};
    }
}

ir_program build_ir(const source_graph& graph, const validation_state& validation)
{
    ir_program program;
    program.variables = graph.variables;
    std::sort(program.variables.begin(), program.variables.end(),
              [](const variable& left, const variable& right) { return left.id < right.id; });

    std::unordered_map<std::string, std::uint32_t> variable_indices;
    for (std::uint32_t index = 0; index < program.variables.size(); ++index)
        variable_indices.emplace(program.variables[index].id, index);

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
            case node_kind::int_literal:
            case node_kind::float_literal:
            case node_kind::vector2_literal:
            case node_kind::string_literal:
            case node_kind::vector3_literal:
            case node_kind::vector4_literal:
            {
                const value_type type = *literal_type(node->kind);
                allocate_slot(*node, "value", type, *parse_default_value(type, node->values.at("value")));
                break;
            }
            case node_kind::get_variable:
            case node_kind::add:
            case node_kind::subtract:
            case node_kind::multiply:
            case node_kind::divide:
            case node_kind::vector_normalize:
            case node_kind::vector_scale:
            case node_kind::select:
            case node_kind::convert_number:
            {
                const value_type type = *node_data_type(graph, *node);
                allocate_slot(*node, "value", type, default_value(type));
                break;
            }
            case node_kind::compare:
            case node_kind::boolean_and:
            case node_kind::boolean_or:
            case node_kind::boolean_not:
                allocate_slot(*node, "result", value_type::boolean, false);
                break;
            case node_kind::vector_dot:
            case node_kind::vector_length:
                allocate_slot(*node, "value", value_type::float32, 0.0);
                break;
            case node_kind::do_once:
                allocate_slot(*node, "$fired", value_type::boolean, false);
                break;
            case node_kind::gate:
                allocate_slot(*node, "$open", value_type::boolean, !node->values.at("startClosed").get<bool>());
                break;
            case node_kind::for_loop:
                allocate_slot(*node, "index", value_type::integer, std::int64_t{0});
                break;
            case node_kind::delay:
            case node_kind::retriggerable_delay:
                allocate_slot(*node, "$active", value_type::boolean, false);
                allocate_slot(*node, "$remaining", value_type::float32, 0.0);
                break;
            case node_kind::timer:
                allocate_slot(*node, "active", value_type::boolean, false);
                allocate_slot(*node, "$remaining", value_type::float32, 0.0);
                allocate_slot(*node, "$period", value_type::float32, 0.0);
                allocate_slot(*node, "$looping", value_type::boolean, false);
                allocate_slot(*node, "$generation", value_type::integer, std::int64_t{0});
                break;
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
    std::unordered_map<std::string, std::uint32_t> entry_instruction_indices;
    const auto add_entry_instruction = [&](const source_node& node, std::string_view pin, ir_opcode opcode)
    {
        const auto index = static_cast<std::uint32_t>(program.instructions.size());
        program.instructions.push_back({.opcode = opcode, .node_id = node.id});
        entry_instruction_indices.emplace(pin_key(node.id, pin), index);
        return index;
    };
    for (const source_node* node : nodes)
    {
        if (!is_executable_node(node->kind)) continue;
        if (node->kind == node_kind::do_once)
        {
            const auto index = add_entry_instruction(*node, "exec", ir_opcode::do_once);
            instruction_indices.emplace(node->id, index);
            add_entry_instruction(*node, "reset", ir_opcode::do_once_reset);
        }
        else if (node->kind == node_kind::gate)
        {
            const auto index = add_entry_instruction(*node, "enter", ir_opcode::gate_enter);
            instruction_indices.emplace(node->id, index);
            add_entry_instruction(*node, "open", ir_opcode::gate_open);
            add_entry_instruction(*node, "close", ir_opcode::gate_close);
            add_entry_instruction(*node, "toggle", ir_opcode::gate_toggle);
        }
        else if (node->kind == node_kind::timer)
        {
            const auto index = add_entry_instruction(*node, "start", ir_opcode::timer_start);
            instruction_indices.emplace(node->id, index);
            add_entry_instruction(*node, "stop", ir_opcode::timer_stop);
        }
        else
        {
            const auto index = add_entry_instruction(*node, "exec", executable_opcode_for(node->kind));
            instruction_indices.emplace(node->id, index);
        }
    }

    const auto node_by_id = [&](std::string_view id) -> const source_node&
    { return *validation.nodes.at(std::string{id}); };

    const auto make_data_instruction = [&](const source_node& node, std::uint32_t next)
    {
        ir_instruction instruction;
        instruction.node_id = node.id;
        switch (node.kind)
        {
            case node_kind::get_variable:
                instruction.opcode = ir_opcode::load_variable;
                instruction.operand0 = variable_indices.at(variable_for(graph, node)->id);
                instruction.operand1 = value_slots.at(pin_key(node.id, "value"));
                instruction.operand2 = next;
                break;
            case node_kind::add:
            case node_kind::subtract:
            case node_kind::multiply:
            case node_kind::divide:
                instruction.opcode = node.kind == node_kind::add        ? ir_opcode::add
                                     : node.kind == node_kind::subtract ? ir_opcode::subtract
                                     : node.kind == node_kind::multiply ? ir_opcode::multiply
                                                                        : ir_opcode::divide;
                instruction.operand0 = input_slot(node, "a");
                instruction.operand1 = input_slot(node, "b");
                instruction.operand2 = value_slots.at(pin_key(node.id, "value"));
                instruction.operand3 = next;
                break;
            case node_kind::compare:
                instruction.opcode = compare_opcode(node);
                instruction.operand0 = input_slot(node, "a");
                instruction.operand1 = input_slot(node, "b");
                instruction.operand2 = value_slots.at(pin_key(node.id, "result"));
                instruction.operand3 = next;
                break;
            case node_kind::boolean_and:
            case node_kind::boolean_or:
                instruction.opcode =
                    node.kind == node_kind::boolean_and ? ir_opcode::boolean_and : ir_opcode::boolean_or;
                instruction.operand0 = input_slot(node, "a");
                instruction.operand1 = input_slot(node, "b");
                instruction.operand2 = value_slots.at(pin_key(node.id, "result"));
                instruction.operand3 = next;
                break;
            case node_kind::boolean_not:
                instruction.opcode = ir_opcode::boolean_not;
                instruction.operand0 = input_slot(node, "value");
                instruction.operand1 = value_slots.at(pin_key(node.id, "result"));
                instruction.operand2 = next;
                break;
            case node_kind::vector_dot:
                instruction.opcode = ir_opcode::vector_dot;
                instruction.operand0 = input_slot(node, "a");
                instruction.operand1 = input_slot(node, "b");
                instruction.operand2 = value_slots.at(pin_key(node.id, "value"));
                instruction.operand3 = next;
                break;
            case node_kind::vector_length:
                instruction.opcode = ir_opcode::vector_length;
                instruction.operand0 = input_slot(node, "value");
                instruction.operand1 = value_slots.at(pin_key(node.id, "value"));
                instruction.operand2 = next;
                break;
            case node_kind::vector_normalize:
                instruction.opcode = ir_opcode::vector_normalize;
                instruction.operand0 = input_slot(node, "value");
                instruction.operand1 = value_slots.at(pin_key(node.id, "value"));
                instruction.operand2 = next;
                break;
            case node_kind::vector_scale:
                instruction.opcode = ir_opcode::vector_scale;
                instruction.operand0 = input_slot(node, "vector");
                instruction.operand1 = input_slot(node, "scale");
                instruction.operand2 = value_slots.at(pin_key(node.id, "value"));
                instruction.operand3 = next;
                break;
            case node_kind::select:
                instruction.opcode = ir_opcode::select;
                instruction.operand0 = input_slot(node, "condition");
                instruction.operand1 = input_slot(node, "trueValue");
                instruction.operand2 = input_slot(node, "falseValue");
                instruction.operand3 = value_slots.at(pin_key(node.id, "value"));
                instruction.operand4 = next;
                break;
            case node_kind::convert_number:
                instruction.opcode = node.values.at("conversion").get<std::string>() == "intToFloat"
                                         ? ir_opcode::convert_int_to_float
                                         : ir_opcode::convert_float_to_int;
                instruction.operand0 = input_slot(node, "value");
                instruction.operand1 = value_slots.at(pin_key(node.id, "value"));
                instruction.operand2 = next;
                break;
            default:
                break;
        }
        return instruction;
    };

    const auto collect_data_order = [&](const source_node& executable, std::optional<std::string_view> only_pin)
    {
        std::vector<const source_node*> order;
        std::unordered_set<std::string> seen;
        std::function<void(const source_node&)> collect = [&](const source_node& node)
        {
            if (!is_computed_value_node(node.kind) || !seen.insert(node.id).second) return;
            for (const std::string_view pin : data_inputs(node.kind))
            {
                const auto incoming = validation.incoming.find(pin_key(node.id, pin));
                if (incoming == validation.incoming.end()) continue;
                collect(node_by_id(incoming->second->from.node_id));
            }
            order.push_back(&node);
        };

        if (only_pin)
        {
            const auto incoming = validation.incoming.find(pin_key(executable.id, *only_pin));
            if (incoming != validation.incoming.end()) collect(node_by_id(incoming->second->from.node_id));
        }
        else
        {
            for (const source_connection& connection : graph.connections)
            {
                if (connection.kind != connection_kind::value || connection.to.node_id != executable.id) continue;
                collect(node_by_id(connection.from.node_id));
            }
        }
        return order;
    };

    const auto execution_inputs = [](node_kind kind)
    {
        if (kind == node_kind::do_once) return std::vector<std::string_view>{"exec", "reset"};
        if (kind == node_kind::gate) return std::vector<std::string_view>{"enter", "open", "close", "toggle"};
        if (kind == node_kind::timer) return std::vector<std::string_view>{"start", "stop"};
        return std::vector<std::string_view>{"exec"};
    };

    std::unordered_map<std::string, std::uint32_t> execution_entries;
    for (const source_node* executable : nodes)
    {
        if (!is_executable_node(executable->kind)) continue;
        for (const std::string_view input : execution_inputs(executable->kind))
        {
            const std::vector<const source_node*> order = executable->kind == node_kind::timer && input == "stop"
                                                              ? std::vector<const source_node*>{}
                                                              : collect_data_order(*executable, std::nullopt);
            std::uint32_t target = entry_instruction_indices.at(pin_key(executable->id, input));
            for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator)
            {
                const auto index = static_cast<std::uint32_t>(program.instructions.size());
                program.instructions.push_back(make_data_instruction(**iterator, target));
                target = index;
            }
            execution_entries.emplace(pin_key(executable->id, input), target);
        }
    }

    const auto execution_target = [&](const source_node& node, std::string_view pin)
    {
        const auto connection = validation.execution_outgoing.find(pin_key(node.id, pin));
        if (connection == validation.execution_outgoing.end()) return invalid_instruction;
        const auto target = execution_entries.find(pin_key(connection->second->to.node_id, connection->second->to.pin));
        return target == execution_entries.end() ? invalid_instruction : target->second;
    };

    std::unordered_map<std::string, std::uint32_t> while_condition_entries;
    for (const source_node* node : nodes)
    {
        if (node->kind != node_kind::while_loop) continue;
        const std::vector<const source_node*> order = collect_data_order(*node, std::string_view{"condition"});
        std::uint32_t target = invalid_instruction;
        for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator)
        {
            const auto index = static_cast<std::uint32_t>(program.instructions.size());
            program.instructions.push_back(make_data_instruction(**iterator, target));
            target = index;
        }
        while_condition_entries.emplace(node->id, target);
    }

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
            case node_kind::sequence:
                instruction.operand0 = execution_target(*node, "then0");
                instruction.operand1 = execution_target(*node, "then1");
                instruction.operand2 = execution_target(*node, "then2");
                instruction.operand3 = execution_target(*node, "then3");
                break;
            case node_kind::switch_integer:
            {
                switch_int_table table;
                const json& cases = node->values.at("cases");
                for (std::size_t case_index = 0; case_index < table.values.size(); ++case_index)
                {
                    table.values[case_index] = cases[case_index].get<std::int64_t>();
                    const std::string pin = std::string{"case"} + std::to_string(case_index);
                    table.instructions[case_index] = execution_target(*node, pin);
                }
                table.instructions[4] = execution_target(*node, "default");
                instruction.operand0 = input_slot(*node, "selection");
                instruction.operand1 = static_cast<std::uint32_t>(program.switch_int_tables.size());
                program.switch_int_tables.push_back(table);
                break;
            }
            case node_kind::do_once:
            {
                const std::uint32_t state = value_slots.at(pin_key(node->id, "$fired"));
                instruction.operand0 = state;
                instruction.operand1 = execution_target(*node, "then");
                ir_instruction& reset = program.instructions[entry_instruction_indices.at(pin_key(node->id, "reset"))];
                reset.operand0 = state;
                break;
            }
            case node_kind::gate:
            {
                const std::uint32_t state = value_slots.at(pin_key(node->id, "$open"));
                instruction.operand0 = state;
                instruction.operand1 = execution_target(*node, "exit");
                program.instructions[entry_instruction_indices.at(pin_key(node->id, "open"))].operand0 = state;
                program.instructions[entry_instruction_indices.at(pin_key(node->id, "close"))].operand0 = state;
                program.instructions[entry_instruction_indices.at(pin_key(node->id, "toggle"))].operand0 = state;
                break;
            }
            case node_kind::for_loop:
                instruction.operand0 = input_slot(*node, "first");
                instruction.operand1 = input_slot(*node, "last");
                instruction.operand2 = value_slots.at(pin_key(node->id, "index"));
                instruction.operand3 = execution_target(*node, "loopBody");
                instruction.operand4 = execution_target(*node, "completed");
                break;
            case node_kind::while_loop:
                instruction.operand0 = input_slot(*node, "condition");
                instruction.operand1 = execution_target(*node, "loopBody");
                instruction.operand2 = execution_target(*node, "completed");
                instruction.operand3 = while_condition_entries.at(node->id);
                break;
            case node_kind::delay:
            case node_kind::retriggerable_delay:
            {
                const latent_action_kind kind = node->kind == node_kind::delay
                                                    ? latent_action_kind::delay
                                                    : latent_action_kind::retriggerable_delay;
                const std::uint32_t latent_index = static_cast<std::uint32_t>(program.latent_actions.size());
                program.latent_actions.push_back({.kind = kind,
                                                  .active_slot = value_slots.at(pin_key(node->id, "$active")),
                                                  .remaining_slot = value_slots.at(pin_key(node->id, "$remaining")),
                                                  .completed_instruction = execution_target(*node, "completed")});
                instruction.operand0 = input_slot(*node, "duration");
                instruction.operand1 = latent_index;
                break;
            }
            case node_kind::timer:
            {
                const std::uint32_t latent_index = static_cast<std::uint32_t>(program.latent_actions.size());
                program.latent_actions.push_back({.kind = latent_action_kind::timer,
                                                  .active_slot = value_slots.at(pin_key(node->id, "active")),
                                                  .remaining_slot = value_slots.at(pin_key(node->id, "$remaining")),
                                                  .period_slot = value_slots.at(pin_key(node->id, "$period")),
                                                  .looping_slot = value_slots.at(pin_key(node->id, "$looping")),
                                                  .generation_slot = value_slots.at(pin_key(node->id, "$generation")),
                                                  .tick_instruction = execution_target(*node, "tick"),
                                                  .completed_instruction = execution_target(*node, "completed")});
                instruction.operand0 = input_slot(*node, "interval");
                instruction.operand1 = input_slot(*node, "looping");
                instruction.operand2 = latent_index;
                instruction.operand3 = execution_target(*node, "started");
                ir_instruction& stop = program.instructions[entry_instruction_indices.at(pin_key(node->id, "stop"))];
                stop.operand0 = latent_index;
                stop.operand1 = execution_target(*node, "stopped");
                break;
            }
            case node_kind::set_variable:
                instruction.operand0 = variable_indices.at(variable_for(graph, *node)->id);
                instruction.operand1 = input_slot(*node, "value");
                instruction.operand2 = execution_target(*node, "then");
                break;
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
                instruction.operand1 =
                    static_cast<std::uint32_t>(*parse_core_component(node->values.at("component").get<std::string>()));
                instruction.operand2 = value_slots.at(pin_key(node->id, "has"));
                instruction.operand3 = execution_target(*node, "then");
                break;
            case node_kind::remove_core_component:
                instruction.operand0 = input_slot(*node, "entity");
                instruction.operand1 =
                    static_cast<std::uint32_t>(*parse_core_component(node->values.at("component").get<std::string>()));
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

    const auto value_reaches = [&](std::string_view start, std::string_view target)
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
            const auto adjacency = validation.value_adjacency.find(current);
            if (adjacency == validation.value_adjacency.end()) continue;
            for (const std::string& next : adjacency->second)
            {
                if (next == target) return true;
                if (seen.insert(next).second) pending.push(next);
            }
        }
        return false;
    };

    std::vector<const source_node*> self_nodes;
    std::vector<const source_node*> executable_nodes;
    for (const source_node* node : nodes)
    {
        if (node->kind == node_kind::self_entity) self_nodes.push_back(node);
        if (is_executable_node(node->kind)) executable_nodes.push_back(node);
    }

    for (ir_entry_point& entry : program.entry_points)
    {
        std::uint32_t target = entry.instruction;
        for (auto iterator = self_nodes.rbegin(); iterator != self_nodes.rend(); ++iterator)
        {
            const source_node& self = **iterator;
            const bool needed =
                std::any_of(executable_nodes.begin(), executable_nodes.end(), [&](const source_node* node)
                            { return reachable_from(entry.node_id, node->id) && value_reaches(self.id, node->id); });
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
        case ir_opcode::sequence:
            return bytecode_opcode::sequence;
        case ir_opcode::switch_integer:
            return bytecode_opcode::switch_integer;
        case ir_opcode::do_once:
            return bytecode_opcode::do_once;
        case ir_opcode::do_once_reset:
            return bytecode_opcode::do_once_reset;
        case ir_opcode::gate_enter:
            return bytecode_opcode::gate_enter;
        case ir_opcode::gate_open:
            return bytecode_opcode::gate_open;
        case ir_opcode::gate_close:
            return bytecode_opcode::gate_close;
        case ir_opcode::gate_toggle:
            return bytecode_opcode::gate_toggle;
        case ir_opcode::for_loop:
            return bytecode_opcode::for_loop;
        case ir_opcode::while_loop:
            return bytecode_opcode::while_loop;
        case ir_opcode::delay:
            return bytecode_opcode::delay;
        case ir_opcode::retriggerable_delay:
            return bytecode_opcode::retriggerable_delay;
        case ir_opcode::timer_start:
            return bytecode_opcode::timer_start;
        case ir_opcode::timer_stop:
            return bytecode_opcode::timer_stop;
        case ir_opcode::load_variable:
            return bytecode_opcode::load_variable;
        case ir_opcode::store_variable:
            return bytecode_opcode::store_variable;
        case ir_opcode::add:
            return bytecode_opcode::add;
        case ir_opcode::subtract:
            return bytecode_opcode::subtract;
        case ir_opcode::multiply:
            return bytecode_opcode::multiply;
        case ir_opcode::divide:
            return bytecode_opcode::divide;
        case ir_opcode::compare_equal:
            return bytecode_opcode::compare_equal;
        case ir_opcode::compare_not_equal:
            return bytecode_opcode::compare_not_equal;
        case ir_opcode::compare_less:
            return bytecode_opcode::compare_less;
        case ir_opcode::compare_less_equal:
            return bytecode_opcode::compare_less_equal;
        case ir_opcode::compare_greater:
            return bytecode_opcode::compare_greater;
        case ir_opcode::compare_greater_equal:
            return bytecode_opcode::compare_greater_equal;
        case ir_opcode::boolean_and:
            return bytecode_opcode::boolean_and;
        case ir_opcode::boolean_or:
            return bytecode_opcode::boolean_or;
        case ir_opcode::boolean_not:
            return bytecode_opcode::boolean_not;
        case ir_opcode::vector_dot:
            return bytecode_opcode::vector_dot;
        case ir_opcode::vector_length:
            return bytecode_opcode::vector_length;
        case ir_opcode::vector_normalize:
            return bytecode_opcode::vector_normalize;
        case ir_opcode::vector_scale:
            return bytecode_opcode::vector_scale;
        case ir_opcode::select:
            return bytecode_opcode::select;
        case ir_opcode::convert_int_to_float:
            return bytecode_opcode::convert_int_to_float;
        case ir_opcode::convert_float_to_int:
            return bytecode_opcode::convert_float_to_int;
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
    bytecode.switch_int_tables = ir.switch_int_tables;
    bytecode.latent_actions = ir.latent_actions;
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
