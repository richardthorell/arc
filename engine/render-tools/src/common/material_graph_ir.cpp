#include <arc/render_tools/material_graph.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>

namespace arc::render::tools
{
namespace
{
using json = nlohmann::json;

std::string concatenate(std::initializer_list<std::string_view> parts)
{
    std::size_t size{};
    for (const auto part : parts)
        size += part.size();
    std::string result;
    result.reserve(size);
    for (const auto part : parts)
        result.append(part);
    return result;
}

std::optional<material_ir_node_kind> node_kind(std::string_view type) noexcept
{
    if (type == "constant") return material_ir_node_kind::constant;
    if (type == "vector2") return material_ir_node_kind::vector2;
    if (type == "vector3") return material_ir_node_kind::vector3;
    if (type == "vector4") return material_ir_node_kind::vector4;
    if (type == "texCoord") return material_ir_node_kind::tex_coord;
    if (type == "time") return material_ir_node_kind::time;
    if (type == "textureSample") return material_ir_node_kind::texture_sample;
    if (type == "normalMap") return material_ir_node_kind::normal_map;
    if (type == "saturate") return material_ir_node_kind::saturate;
    if (type == "clamp") return material_ir_node_kind::clamp;
    if (type == "lerp") return material_ir_node_kind::lerp;
    if (type == "add") return material_ir_node_kind::add;
    if (type == "subtract") return material_ir_node_kind::subtract;
    if (type == "multiply") return material_ir_node_kind::multiply;
    if (type == "divide") return material_ir_node_kind::divide;
    if (type == "shaderFunctionCall") return material_ir_node_kind::function_call;
    if (type == "output") return material_ir_node_kind::output;
    return std::nullopt;
}

std::optional<shader_parameter_type> function_pin_type(std::string_view type) noexcept
{
    if (type == "float" || type == "float1") return shader_parameter_type::float32;
    if (type == "float2" || type == "vec2") return shader_parameter_type::float2;
    if (type == "float3" || type == "vec3") return shader_parameter_type::float3;
    if (type == "float4" || type == "vec4") return shader_parameter_type::float4;
    return std::nullopt;
}

std::uint8_t function_pin_components(shader_parameter_type type) noexcept
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

bool parse_function_default(const json& value, shader_parameter_type type, material_ir_literal& output)
{
    output.components = function_pin_components(type);
    if (output.components == 1)
    {
        if (!value.is_number()) return false;
        output.values[0] = value.get<float>();
        return true;
    }
    if (!value.is_array() || value.size() != output.components) return false;
    for (std::uint8_t index = 0; index < output.components; ++index)
    {
        if (!value[index].is_number()) return false;
        output.values[index] = value[index].get<float>();
    }
    return true;
}

bool parse_function_pins(const json& authored, bool allow_defaults, std::vector<material_function_pin>& output)
{
    if (!authored.is_array()) return false;
    std::set<std::string> ids;
    for (const auto& pin : authored)
    {
        if (!pin.is_object()) return false;
        const auto id = pin.value("id", "");
        const auto name = pin.value("name", id);
        const auto type = function_pin_type(pin.value("type", ""));
        if (id.empty() || name.empty() || !type || !ids.insert(id).second) return false;

        material_function_pin parsed{.id = id, .name = name, .type = *type};
        parsed.has_default = pin.value("hasDefault", false);
        if (parsed.has_default)
        {
            if (!allow_defaults || !pin.contains("default") ||
                !parse_function_default(pin["default"], *type, parsed.default_value))
                return false;
        }
        output.push_back(std::move(parsed));
    }
    return true;
}

bool function_has_pin(const std::vector<material_function_pin>& pins, std::string_view id)
{
    return std::ranges::any_of(pins, [id](const material_function_pin& pin) { return pin.id == id; });
}

float number(const json& value, float fallback)
{
    return value.is_number() ? value.get<float>() : fallback;
}

material_ir_literal literal(const json& values, material_ir_node_kind kind)
{
    material_ir_literal result;
    const auto value = values.value("value", json{});
    if (kind == material_ir_node_kind::constant)
    {
        result.values[0] = number(value, 0.5f);
        result.components = 1;
        return result;
    }

    std::size_t components{};
    float fallback{};
    if (kind == material_ir_node_kind::vector2)
        components = 2;
    else if (kind == material_ir_node_kind::vector3)
        components = 3;
    else if (kind == material_ir_node_kind::vector4)
    {
        components = 4;
        fallback = 1.0f;
    }
    if (components == 0) return result;

    for (std::size_t index = 0; index < components; ++index)
        result.values[index] = number(value.is_array() && index < value.size() ? value[index] : json{}, fallback);
    result.components = static_cast<std::uint8_t>(components);
    return result;
}

shader_parameter_type parameter_type(material_ir_node_kind kind) noexcept
{
    switch (kind)
    {
        case material_ir_node_kind::vector2:
            return shader_parameter_type::float2;
        case material_ir_node_kind::vector3:
            return shader_parameter_type::float3;
        case material_ir_node_kind::vector4:
            return shader_parameter_type::float4;
        case material_ir_node_kind::texture_sample:
            return shader_parameter_type::texture_2d;
        default:
            return shader_parameter_type::float32;
    }
}

std::uint32_t parameter_size(shader_parameter_type type) noexcept
{
    switch (type)
    {
        case shader_parameter_type::float2:
            return 8;
        case shader_parameter_type::float3:
            return 12;
        case shader_parameter_type::float4:
            return 16;
        default:
            return 4;
    }
}

std::vector<std::byte> parameter_default(const material_ir_node& node, shader_parameter_type type)
{
    if (type == shader_parameter_type::texture_2d || node.literal.components == 0) return {};
    const auto count = static_cast<std::size_t>(node.literal.components);
    const auto bytes = std::as_bytes(std::span<const float>(node.literal.values.data(), count));
    return {bytes.begin(), bytes.end()};
}

struct input_connection
{
    std::string source_node;
    std::string source_pin;
};

using input_map = std::map<std::pair<std::string, std::string>, input_connection>;

bool has_cycle(const std::map<std::string, std::vector<std::string>>& adjacency,
               const std::vector<material_ir_node>& nodes)
{
    enum class visit_state : std::uint8_t
    {
        unseen,
        visiting,
        complete
    };

    std::map<std::string, visit_state> states;
    const auto visit = [&](const auto& self, const std::string& id) -> bool
    {
        auto& state = states[id];
        if (state == visit_state::visiting) return true;
        if (state == visit_state::complete) return false;
        state = visit_state::visiting;
        if (const auto found = adjacency.find(id); found != adjacency.end())
            for (const auto& target : found->second)
                if (self(self, target)) return true;
        state = visit_state::complete;
        return false;
    };

    for (const auto& node : nodes)
        if (visit(visit, node.id)) return true;
    return false;
}

std::set<std::string> reachable_nodes(const std::string& output_node, const input_map& inputs)
{
    std::set<std::string> reachable;
    const auto visit = [&](const auto& self, const std::string& id) -> void
    {
        if (!reachable.insert(id).second) return;
        for (const auto& [target, source] : inputs)
            if (target.first == id) self(self, source.source_node);
    };
    visit(visit, output_node);
    return reachable;
}

constexpr std::array<std::pair<material_surface_output, std::string_view>, 24> surface_outputs{{
    {material_surface_output::base_color, "baseColor"},
    {material_surface_output::metallic, "metallic"},
    {material_surface_output::roughness, "roughness"},
    {material_surface_output::normal, "normal"},
    {material_surface_output::clear_coat_normal, "clearCoatNormal"},
    {material_surface_output::tangent, "tangent"},
    {material_surface_output::ambient_occlusion, "ao"},
    {material_surface_output::emissive, "emissive"},
    {material_surface_output::opacity, "opacity"},
    {material_surface_output::alpha_cutoff, "alphaClip"},
    {material_surface_output::index_of_refraction, "indexOfRefraction"},
    {material_surface_output::clear_coat, "clearCoat"},
    {material_surface_output::clear_coat_roughness, "clearCoatRoughness"},
    {material_surface_output::sheen, "sheen"},
    {material_surface_output::sheen_color, "sheenColor"},
    {material_surface_output::sheen_roughness, "sheenRoughness"},
    {material_surface_output::anisotropy, "anisotropy"},
    {material_surface_output::anisotropy_rotation, "anisotropyRotation"},
    {material_surface_output::transmission, "transmission"},
    {material_surface_output::thickness, "thickness"},
    {material_surface_output::attenuation_color, "attenuationColor"},
    {material_surface_output::attenuation_distance, "attenuationDistance"},
    {material_surface_output::subsurface_color, "subsurfaceColor"},
    {material_surface_output::subsurface, "subsurface"},
}};

} // namespace

material_graph_compile_result compile_material_graph_json(std::string_view graph_json)
{
    const auto document = json::parse(graph_json, nullptr, false);
    if (document.is_discarded() || !document.is_object() || document.value("version", 0) != 1 ||
        !document.contains("nodes") || !document["nodes"].is_array() || !document.contains("connections") ||
        !document["connections"].is_array())
        return material_graph_compile_result::failure(
            {.code = shader_compile_error_code::invalid_request, .message = "material graph JSON is malformed"});

    std::map<std::string, material_ir_node> normalized_nodes;
    std::string output_node;
    for (const auto& authored_node : document["nodes"])
    {
        if (!authored_node.is_object())
            return material_graph_compile_result::failure({.code = shader_compile_error_code::validation_failed,
                                                           .message = "material graph contains an invalid node"});

        const auto id = authored_node.value("id", "");
        const auto type = authored_node.value("type", "");
        const auto kind = node_kind(type);
        if (id.empty() || type.empty() || normalized_nodes.contains(id))
            return material_graph_compile_result::failure(
                {.code = shader_compile_error_code::validation_failed,
                 .message = "material graph contains a missing or duplicate stable node ID"});
        if (!kind)
            return material_graph_compile_result::failure({.code = shader_compile_error_code::validation_failed,
                                                           .message = "unsupported material graph node type: " + type});
        if (*kind == material_ir_node_kind::output)
        {
            if (!output_node.empty())
                return material_graph_compile_result::failure(
                    {.code = shader_compile_error_code::validation_failed,
                     .message = "material graph contains multiple output nodes"});
            output_node = id;
        }

        const auto values = authored_node.value("values", json::object());
        const auto parameter = authored_node.value("parameter", json::object());
        material_ir_node node{.id = id,
                              .kind = *kind,
                              .literal = literal(values, *kind),
                              .strength = number(values.value("strength", json{}), 1.0f),
                              .minimum = number(values.value("min", json{}), 0.0f),
                              .maximum = number(values.value("max", json{}), 1.0f),
                              .exposed_parameter = parameter.value("exposed", false),
                              .parameter_id = make_shader_parameter_id(id),
                              .parameter_name = parameter.value("name", id)};

        if (*kind == material_ir_node_kind::function_call)
        {
            node.function_path = values.value("path", "");
            node.function_entry_point = values.value("entryPoint", "");
            node.function_source = values.value("source", "");
            if (node.exposed_parameter || node.function_path.empty() || node.function_entry_point.empty() ||
                node.function_source.empty() || !values.contains("inputs") || !values.contains("outputs") ||
                !parse_function_pins(values["inputs"], true, node.function_inputs) ||
                !parse_function_pins(values["outputs"], false, node.function_outputs) || node.function_outputs.empty())
                return material_graph_compile_result::failure(
                    {.code = shader_compile_error_code::validation_failed,
                     .message = "material graph contains an invalid shader-backed Material Function call: " + id});
        }
        normalized_nodes.emplace(id, std::move(node));
    }

    if (output_node.empty())
        return material_graph_compile_result::failure(
            {.code = shader_compile_error_code::validation_failed, .message = "material graph has no output node"});

    input_map inputs;
    std::vector<material_ir_connection> connections;
    std::map<std::string, std::vector<std::string>> adjacency;
    for (const auto& edge : document["connections"])
    {
        if (!edge.is_object() || !edge.contains("from") || !edge["from"].is_object() || !edge.contains("to") ||
            !edge["to"].is_object())
            return material_graph_compile_result::failure({.code = shader_compile_error_code::validation_failed,
                                                           .message = "material graph contains an invalid connection"});

        const auto source = edge["from"].value("nodeId", "");
        const auto source_pin = edge["from"].value("pin", "");
        const auto target = edge["to"].value("nodeId", "");
        const auto target_pin = edge["to"].value("pin", "");
        if (!normalized_nodes.contains(source) || !normalized_nodes.contains(target) || source_pin.empty() ||
            target_pin.empty() ||
            !inputs.emplace(std::pair{target, target_pin}, input_connection{source, source_pin}).second)
            return material_graph_compile_result::failure(
                {.code = shader_compile_error_code::validation_failed,
                 .message = "material graph contains an invalid or multiply-connected input"});

        const auto& source_node = normalized_nodes.at(source);
        if (source_node.kind == material_ir_node_kind::function_call &&
            !function_has_pin(source_node.function_outputs, source_pin))
            return material_graph_compile_result::failure(
                {.code = shader_compile_error_code::validation_failed,
                 .message =
                     concatenate({"Material Function call '", source, "' has no output pin '", source_pin, "'"})});
        const auto& target_node = normalized_nodes.at(target);
        if (target_node.kind == material_ir_node_kind::function_call &&
            !function_has_pin(target_node.function_inputs, target_pin))
            return material_graph_compile_result::failure(
                {.code = shader_compile_error_code::validation_failed,
                 .message =
                     concatenate({"Material Function call '", target, "' has no input pin '", target_pin, "'"})});

        connections.push_back(
            {.source_node = source, .source_pin = source_pin, .target_node = target, .target_pin = target_pin});
        adjacency[source].push_back(target);
    }

    material_graph_compilation compilation;
    compilation.ir.output_node_id = output_node;
    compilation.ir.nodes.reserve(normalized_nodes.size());
    for (auto& [id, node] : normalized_nodes)
    {
        static_cast<void>(id);
        compilation.ir.nodes.push_back(std::move(node));
    }
    std::ranges::sort(connections, {}, [](const material_ir_connection& connection) {
        return std::tuple{connection.target_node, connection.target_pin, connection.source_node, connection.source_pin};
    });
    compilation.ir.connections = std::move(connections);

    if (has_cycle(adjacency, compilation.ir.nodes))
        return material_graph_compile_result::failure(
            {.code = shader_compile_error_code::validation_failed, .message = "material graph contains a cycle"});

    const auto reachable = reachable_nodes(output_node, inputs);
    std::set<std::uint64_t> parameter_ids;
    for (const auto& node : compilation.ir.nodes)
    {
        if (!reachable.contains(node.id) || node.kind == material_ir_node_kind::output) continue;

        if (node.kind == material_ir_node_kind::function_call)
        {
            for (const auto& pin : node.function_inputs)
                if (!inputs.contains({node.id, pin.id}) && !pin.has_default)
                    return material_graph_compile_result::failure(
                        {.code = shader_compile_error_code::validation_failed,
                         .message =
                             "Material Function call '" + node.id + "' is missing required input '" + pin.id + "'"});
        }

        if (node.exposed_parameter)
        {
            const auto id = node.parameter_id.representation();
            if (!parameter_ids.insert(id).second)
                return material_graph_compile_result::failure(
                    {.code = shader_compile_error_code::validation_failed,
                     .message = "material graph contains colliding stable parameter IDs"});
            const auto type = parameter_type(node.kind);
            compilation.descriptor.parameters.push_back({.id = node.parameter_id,
                                                         .name = node.parameter_name,
                                                         .type = type,
                                                         .size = parameter_size(type),
                                                         .default_value = parameter_default(node, type)});
        }

        switch (node.kind)
        {
            case material_ir_node_kind::time:
                compilation.descriptor.requirements.uses_time = true;
                break;
            case material_ir_node_kind::tex_coord:
                compilation.descriptor.requirements.uses_uv0 = true;
                break;
            case material_ir_node_kind::texture_sample:
            {
                compilation.descriptor.requirements.uses_texture_sampling = true;
                if (!inputs.contains({node.id, "uv"})) compilation.descriptor.requirements.uses_uv0 = true;
                const auto slot = static_cast<std::uint32_t>(compilation.descriptor.textures.size());
                compilation.descriptor.textures.push_back(
                    {.node_id = node.id,
                     .slot = slot,
                     .parameter_id = node.exposed_parameter ? node.parameter_id : shader_parameter_id{},
                     .parameter_name = node.exposed_parameter ? node.parameter_name : std::string{}});
                break;
            }
            case material_ir_node_kind::normal_map:
                compilation.descriptor.requirements.uses_normal_mapping = true;
                break;
            default:
                break;
        }
    }

    std::ranges::sort(compilation.descriptor.parameters, {},
                      [](const shader_parameter_descriptor& parameter) { return parameter.id.representation(); });

    compilation.descriptor.outputs.reserve(surface_outputs.size());
    for (const auto& [semantic, pin] : surface_outputs)
    {
        material_surface_output_binding binding{.output = semantic};
        if (const auto found = inputs.find({output_node, std::string(pin)}); found != inputs.end())
        {
            binding.connected = true;
            binding.source_node = found->second.source_node;
            binding.source_pin = found->second.source_pin;
        }
        compilation.descriptor.outputs.push_back(std::move(binding));
    }

    return material_graph_compile_result::success(std::move(compilation));
}

} // namespace arc::render::tools
