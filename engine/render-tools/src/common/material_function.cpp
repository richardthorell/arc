#include <arc/render_tools/material_graph.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::render::tools
{
namespace
{
using json = nlohmann::json;

struct parsed_material_function
{
    std::string path;
    std::string name;
    std::string description;
    std::vector<material_function_pin> inputs;
    std::vector<material_function_pin> outputs;
    bool shader_backed{};
    json graph;
    std::string shader_source;
    std::string entry_point{"arc_material_function"};
};

using function_parse_result = core::result<parsed_material_function, shader_compile_error>;
using graph_expand_result = core::result<json, shader_compile_error>;

shader_compile_error validation_error(std::string message)
{
    return {.code = shader_compile_error_code::validation_failed, .message = std::move(message)};
}

std::string normalize_path(std::string_view path)
{
    std::string result(path);
    std::replace(result.begin(), result.end(), '\\', '/');
    while (result.starts_with("./"))
        result.erase(0, 2);
    return result;
}

std::optional<shader_parameter_type> function_pin_type(std::string_view type) noexcept
{
    if (type == "float" || type == "float1") return shader_parameter_type::float32;
    if (type == "float2" || type == "vec2") return shader_parameter_type::float2;
    if (type == "float3" || type == "vec3") return shader_parameter_type::float3;
    if (type == "float4" || type == "vec4") return shader_parameter_type::float4;
    return std::nullopt;
}

std::uint8_t component_count(shader_parameter_type type) noexcept
{
    switch (type)
    {
        case shader_parameter_type::float2:
            return 2;
        case shader_parameter_type::float3:
            return 3;
        case shader_parameter_type::float4:
            return 4;
        default:
            return 1;
    }
}

std::optional<material_ir_literal> function_default(const json& value, shader_parameter_type type)
{
    material_ir_literal result;
    result.components = component_count(type);
    if (result.components == 1)
    {
        if (!value.is_number()) return std::nullopt;
        result.values[0] = value.get<float>();
        return result;
    }
    if (!value.is_array() || value.size() != result.components) return std::nullopt;
    for (std::uint8_t index = 0; index < result.components; ++index)
    {
        if (!value[index].is_number()) return std::nullopt;
        result.values[index] = value[index].get<float>();
    }
    return result;
}

function_parse_result parse_function(std::string_view source, std::string_view source_path)
{
    const auto document = json::parse(source, nullptr, false);
    const std::string path = normalize_path(source_path);
    const std::string label = path.empty() ? std::string{"material function"} : path;
    if (document.is_discarded() || !document.is_object() || document.value("kind", "") != "materialFunction" ||
        document.value("version", 0) != static_cast<int>(material_function_version))
        return function_parse_result::failure(validation_error(label + " is not a valid Material Function v1 document"));

    parsed_material_function function;
    function.path = path;
    function.name = document.value("name", "");
    function.description = document.value("description", "");
    if (function.name.empty()) return function_parse_result::failure(validation_error(label + " has no function name"));

    if (!document.contains("inputs") || !document["inputs"].is_array() || !document.contains("outputs") ||
        !document["outputs"].is_array())
        return function_parse_result::failure(validation_error(label + " must declare input and output arrays"));

    const auto parse_pins = [&](const json& authored, bool allow_defaults,
                                std::vector<material_function_pin>& output) -> std::optional<std::string>
    {
        std::set<std::string> ids;
        for (const auto& pin : authored)
        {
            if (!pin.is_object()) return "contains a non-object function pin";
            const auto id = pin.value("id", "");
            const auto name = pin.value("name", id);
            const auto type_name = pin.value("type", "");
            const auto type = function_pin_type(type_name);
            if (id.empty() || name.empty() || !type) return "contains a function pin with an invalid id, name, or type";
            if (!ids.insert(id).second) return "contains duplicate function pin id '" + id + "'";

            material_function_pin parsed{.id = id, .name = name, .type = *type};
            if (pin.contains("default"))
            {
                if (!allow_defaults) return "declares a default value on output pin '" + id + "'";
                const auto value = function_default(pin["default"], *type);
                if (!value) return "declares an invalid default value for input pin '" + id + "'";
                parsed.has_default = true;
                parsed.default_value = *value;
            }
            output.push_back(std::move(parsed));
        }
        return std::nullopt;
    };

    if (const auto error = parse_pins(document["inputs"], true, function.inputs))
        return function_parse_result::failure(validation_error(label + ' ' + *error));
    if (const auto error = parse_pins(document["outputs"], false, function.outputs))
        return function_parse_result::failure(validation_error(label + ' ' + *error));
    if (function.outputs.empty())
        return function_parse_result::failure(validation_error(label + " must declare at least one output"));

    const bool has_graph = document.contains("graph") && document["graph"].is_object();
    const bool has_shader = document.contains("shader") && document["shader"].is_object();
    if (has_graph == has_shader)
        return function_parse_result::failure(
            validation_error(label + " must provide exactly one implementation: graph or shader"));

    if (has_shader)
    {
        function.shader_backed = true;
        const auto& shader = document["shader"];
        function.shader_source = shader.value("source", "");
        function.entry_point = shader.value("entryPoint", "arc_material_function");
        if (function.shader_source.empty() || function.entry_point.empty())
            return function_parse_result::failure(
                validation_error(label + " shader implementation requires source and entryPoint"));
        return function_parse_result::success(std::move(function));
    }

    function.graph = document["graph"];
    if (function.graph.value("version", 0) != 1 || !function.graph.contains("nodes") ||
        !function.graph["nodes"].is_array() || !function.graph.contains("connections") ||
        !function.graph["connections"].is_array())
        return function_parse_result::failure(validation_error(label + " contains malformed graph JSON"));

    std::set<std::string> input_ids;
    for (const auto& input : function.inputs)
        input_ids.insert(input.id);
    std::set<std::string> output_ids;
    for (const auto& output : function.outputs)
        output_ids.insert(output.id);

    std::size_t output_nodes{};
    for (const auto& node : function.graph["nodes"])
    {
        if (!node.is_object()) return function_parse_result::failure(validation_error(label + " contains an invalid node"));
        const auto type = node.value("type", "");
        if (type == "output")
            return function_parse_result::failure(
                validation_error(label + " uses a Material Output node; functions require Function Output"));
        if (type == "functionOutput") ++output_nodes;
        if (type == "functionInput")
        {
            const auto values = node.value("values", json::object());
            const auto input = values.value("input", "");
            if (!input_ids.contains(input))
                return function_parse_result::failure(
                    validation_error(label + " references unknown function input '" + input + "'"));
        }
        const auto parameter = node.value("parameter", json::object());
        if (parameter.value("exposed", false))
            return function_parse_result::failure(
                validation_error(label + " contains an exposed parameter; expose it as a function input instead"));
    }
    if (output_nodes != 1)
        return function_parse_result::failure(validation_error(label + " must contain exactly one Function Output node"));

    for (const auto& connection : function.graph["connections"])
    {
        if (!connection.is_object() || !connection.contains("from") || !connection["from"].is_object() ||
            !connection.contains("to") || !connection["to"].is_object())
            return function_parse_result::failure(validation_error(label + " contains an invalid graph connection"));
        const auto target = connection["to"].value("nodeId", "");
        const auto target_pin = connection["to"].value("pin", "");
        const auto target_node = std::ranges::find_if(function.graph["nodes"], [&](const json& node)
                                                      { return node.value("id", "") == target; });
        if (target_node != function.graph["nodes"].end() && target_node->value("type", "") == "functionOutput" &&
            !output_ids.contains(target_pin))
            return function_parse_result::failure(
                validation_error(label + " writes unknown function output '" + target_pin + "'"));
    }

    return function_parse_result::success(std::move(function));
}

const parsed_material_function* find_function(const std::map<std::string, parsed_material_function>& functions,
                                              std::string_view authored_path)
{
    const auto path = normalize_path(authored_path);
    if (const auto found = functions.find(path); found != functions.end()) return &found->second;

    const parsed_material_function* match{};
    for (const auto& [candidate_path, function] : functions)
    {
        if (candidate_path.size() < path.size() || !candidate_path.ends_with(path)) continue;
        if (candidate_path.size() != path.size() && candidate_path[candidate_path.size() - path.size() - 1] != '/')
            continue;
        if (match) return nullptr;
        match = &function;
    }
    return match;
}

json default_node(std::string id, const material_function_pin& pin)
{
    std::string type = "constant";
    if (pin.type == shader_parameter_type::float2)
        type = "vector2";
    else if (pin.type == shader_parameter_type::float3)
        type = "vector3";
    else if (pin.type == shader_parameter_type::float4)
        type = "vector4";

    json value;
    if (pin.default_value.components <= 1)
        value = pin.default_value.values[0];
    else
    {
        value = json::array();
        for (std::uint8_t index = 0; index < pin.default_value.components; ++index)
            value.push_back(pin.default_value.values[index]);
    }
    return json{{"id", std::move(id)}, {"type", std::move(type)}, {"values", json{{"value", std::move(value)}}}};
}

json pin_json(const material_function_pin& pin)
{
    std::string type = "float";
    if (pin.type == shader_parameter_type::float2)
        type = "float2";
    else if (pin.type == shader_parameter_type::float3)
        type = "float3";
    else if (pin.type == shader_parameter_type::float4)
        type = "float4";

    json result{{"id", pin.id}, {"name", pin.name}, {"type", std::move(type)}, {"hasDefault", pin.has_default}};
    if (pin.has_default)
    {
        if (pin.default_value.components <= 1)
            result["default"] = pin.default_value.values[0];
        else
        {
            result["default"] = json::array();
            for (std::uint8_t index = 0; index < pin.default_value.components; ++index)
                result["default"].push_back(pin.default_value.values[index]);
        }
    }
    return result;
}

struct endpoint
{
    std::string node;
    std::string pin;
};

using endpoint_result = core::result<endpoint, shader_compile_error>;

graph_expand_result expand_graph(json graph, const std::map<std::string, parsed_material_function>& functions,
                                 std::vector<std::string>& stack);

graph_expand_result inline_graph_function(json graph, const json& call, const parsed_material_function& function,
                                          const std::map<std::string, parsed_material_function>& functions,
                                          std::vector<std::string>& stack)
{
    const auto call_id = call.value("id", "");
    if (std::ranges::find(stack, function.path) != stack.end())
        return graph_expand_result::failure(
            validation_error("recursive Material Function dependency detected at '" + function.path + "'"));

    stack.push_back(function.path);
    auto expanded_function = expand_graph(function.graph, functions, stack);
    stack.pop_back();
    if (!expanded_function) return graph_expand_result::failure(expanded_function.error());
    const auto function_graph = std::move(expanded_function).value();

    std::map<std::string, material_function_pin> input_signature;
    for (const auto& pin : function.inputs)
        input_signature.emplace(pin.id, pin);
    std::set<std::string> output_signature;
    for (const auto& pin : function.outputs)
        output_signature.insert(pin.id);

    std::map<std::string, endpoint> external_inputs;
    std::vector<json> external_outputs;
    json retained_connections = json::array();
    for (const auto& connection : graph["connections"])
    {
        const auto from_node = connection["from"].value("nodeId", "");
        const auto to_node = connection["to"].value("nodeId", "");
        if (to_node == call_id)
        {
            const auto pin = connection["to"].value("pin", "");
            if (!input_signature.contains(pin) || external_inputs.contains(pin))
                return graph_expand_result::failure(
                    validation_error("Material Function call '" + call_id + "' has an invalid or duplicate input '" + pin + "'"));
            external_inputs.emplace(pin, endpoint{connection["from"].value("nodeId", ""),
                                                  connection["from"].value("pin", "")});
        }
        else if (from_node == call_id)
        {
            const auto pin = connection["from"].value("pin", "");
            if (!output_signature.contains(pin))
                return graph_expand_result::failure(
                    validation_error("Material Function call '" + call_id + "' references unknown output '" + pin + "'"));
            external_outputs.push_back(connection);
        }
        else
        {
            retained_connections.push_back(connection);
        }
    }

    std::map<std::string, std::string> function_inputs;
    std::string function_output;
    for (const auto& node : function_graph["nodes"])
    {
        const auto id = node.value("id", "");
        const auto type = node.value("type", "");
        if (type == "functionInput")
            function_inputs.emplace(id, node.value("values", json::object()).value("input", ""));
        else if (type == "functionOutput")
            function_output = id;
    }

    json added_nodes = json::array();
    for (const auto& node : function_graph["nodes"])
    {
        const auto type = node.value("type", "");
        if (type == "functionInput" || type == "functionOutput") continue;
        auto clone = node;
        clone["id"] = call_id + "::" + node.value("id", "");
        added_nodes.push_back(std::move(clone));
    }

    std::map<std::string, endpoint> default_sources;
    const auto input_source = [&](std::string_view boundary_node) -> endpoint_result
    {
        const auto boundary = function_inputs.find(std::string(boundary_node));
        if (boundary == function_inputs.end())
            return endpoint_result::failure(validation_error("Material Function contains an invalid Function Input node"));
        if (const auto external = external_inputs.find(boundary->second); external != external_inputs.end())
            return endpoint_result::success(external->second);
        if (const auto existing = default_sources.find(boundary->second); existing != default_sources.end())
            return endpoint_result::success(existing->second);
        const auto signature = input_signature.find(boundary->second);
        if (signature == input_signature.end() || !signature->second.has_default)
            return endpoint_result::failure(validation_error("Material Function call '" + call_id +
                                                              "' is missing required input '" + boundary->second + "'"));
        const auto id = call_id + "::default::" + boundary->second;
        added_nodes.push_back(default_node(id, signature->second));
        const endpoint value{id, "value"};
        default_sources.emplace(boundary->second, value);
        return endpoint_result::success(value);
    };

    const auto source_endpoint = [&](const json& from) -> endpoint_result
    {
        const auto node = from.value("nodeId", "");
        if (function_inputs.contains(node)) return input_source(node);
        if (node == function_output)
            return endpoint_result::failure(validation_error("Function Output cannot be used as a graph source"));
        return endpoint_result::success({call_id + "::" + node, from.value("pin", "")});
    };

    std::map<std::string, endpoint> function_outputs;
    for (const auto& connection : function_graph["connections"])
    {
        const auto target_node = connection["to"].value("nodeId", "");
        if (function_inputs.contains(target_node))
            return graph_expand_result::failure(validation_error("Function Input cannot be used as a graph target"));
        auto source = source_endpoint(connection["from"]);
        if (!source) return graph_expand_result::failure(source.error());

        if (target_node == function_output)
        {
            const auto pin = connection["to"].value("pin", "");
            if (!output_signature.contains(pin) || !function_outputs.emplace(pin, source.value()).second)
                return graph_expand_result::failure(
                    validation_error("Material Function has an invalid or multiply-connected output '" + pin + "'"));
            continue;
        }

        retained_connections.push_back(
            json{{"from", json{{"nodeId", source.value().node}, {"pin", source.value().pin}}},
                 {"to", json{{"nodeId", call_id + "::" + target_node}, {"pin", connection["to"].value("pin", "")}}}});
    }

    for (const auto& connection : external_outputs)
    {
        const auto pin = connection["from"].value("pin", "");
        const auto output = function_outputs.find(pin);
        if (output == function_outputs.end())
            return graph_expand_result::failure(validation_error("Material Function call '" + call_id +
                                                                  "' uses unconnected output '" + pin + "'"));
        auto rewritten = connection;
        rewritten["from"]["nodeId"] = output->second.node;
        rewritten["from"]["pin"] = output->second.pin;
        retained_connections.push_back(std::move(rewritten));
    }

    json nodes = json::array();
    for (const auto& node : graph["nodes"])
        if (node.value("id", "") != call_id) nodes.push_back(node);
    for (auto& node : added_nodes)
        nodes.push_back(std::move(node));
    graph["nodes"] = std::move(nodes);
    graph["connections"] = std::move(retained_connections);
    return graph_expand_result::success(std::move(graph));
}

graph_expand_result expand_graph(json graph, const std::map<std::string, parsed_material_function>& functions,
                                 std::vector<std::string>& stack)
{
    if (!graph.is_object() || graph.value("version", 0) != 1 || !graph.contains("nodes") || !graph["nodes"].is_array() ||
        !graph.contains("connections") || !graph["connections"].is_array())
        return graph_expand_result::failure(validation_error("material graph JSON is malformed"));

    for (;;)
    {
        const auto call = std::ranges::find_if(graph["nodes"], [](const json& node)
                                               { return node.is_object() && node.value("type", "") == "functionCall"; });
        if (call == graph["nodes"].end()) break;

        const auto id = call->value("id", "");
        const auto values = call->value("values", json::object());
        const auto path = values.value("path", "");
        if (id.empty() || path.empty())
            return graph_expand_result::failure(validation_error("Material Function call has no stable id or path"));
        const auto* function = find_function(functions, path);
        if (!function)
            return graph_expand_result::failure(validation_error("Material Function '" + path + "' is missing or ambiguous"));

        if (!function->shader_backed)
        {
            auto inlined = inline_graph_function(std::move(graph), *call, *function, functions, stack);
            if (!inlined) return inlined;
            graph = std::move(inlined).value();
            continue;
        }

        auto replacement = *call;
        replacement["type"] = "shaderFunctionCall";
        replacement["values"] = json{{"path", function->path.empty() ? normalize_path(path) : function->path},
                                      {"entryPoint", function->entry_point},
                                      {"source", function->shader_source},
                                      {"inputs", json::array()},
                                      {"outputs", json::array()}};
        for (const auto& pin : function->inputs)
            replacement["values"]["inputs"].push_back(pin_json(pin));
        for (const auto& pin : function->outputs)
            replacement["values"]["outputs"].push_back(pin_json(pin));
        *call = std::move(replacement);
    }
    return graph_expand_result::success(std::move(graph));
}

} // namespace

bool is_material_function_json(std::string_view source) noexcept
{
    const auto document = json::parse(source, nullptr, false);
    return !document.is_discarded() && document.is_object() && document.value("kind", "") == "materialFunction";
}

material_function_validation_result validate_material_function_json(std::string_view function_json,
                                                                    std::string_view source_path)
{
    auto parsed = parse_function(function_json, source_path);
    if (!parsed) return material_function_validation_result::failure(parsed.error());
    auto pins = parsed.value().inputs;
    pins.insert(pins.end(), parsed.value().outputs.begin(), parsed.value().outputs.end());
    return material_function_validation_result::success(std::move(pins));
}

material_graph_compile_result compile_material_graph_json(std::string_view graph_json,
                                                          std::span<const material_function_source> function_sources)
{
    std::map<std::string, parsed_material_function> functions;
    for (const auto& source : function_sources)
    {
        auto parsed = parse_function(source.source, source.path);
        if (!parsed) return material_graph_compile_result::failure(parsed.error());
        auto function = std::move(parsed).value();
        if (function.path.empty())
            return material_graph_compile_result::failure(validation_error("Material Function source path cannot be empty"));
        if (!functions.emplace(function.path, std::move(function)).second)
            return material_graph_compile_result::failure(
                validation_error("duplicate Material Function source path: " + normalize_path(source.path)));
    }

    auto document = json::parse(graph_json, nullptr, false);
    if (document.is_discarded())
        return material_graph_compile_result::failure(
            {.code = shader_compile_error_code::invalid_request, .message = "material graph JSON is malformed"});

    std::vector<std::string> stack;
    auto expanded = expand_graph(std::move(document), functions, stack);
    if (!expanded) return material_graph_compile_result::failure(expanded.error());
    return compile_material_graph_json(expanded.value().dump());
}

} // namespace arc::render::tools
