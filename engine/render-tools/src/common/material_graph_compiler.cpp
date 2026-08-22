#include <arc/render_tools/render_tools.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace arc::render::tools
{
namespace
{
using json = nlohmann::json;

std::string number(const json& value, float fallback = 0.0f)
{
    const float resolved = value.is_number() ? value.get<float>() : fallback;
    std::ostringstream output;
    output << std::setprecision(9) << resolved;
    if (output.str().find_first_of(".eE") == std::string::npos) output << ".0";
    return output.str();
}

std::string vector_value(const json& value, std::size_t width, float fallback = 0.0f)
{
    std::ostringstream output;
    output << "float" << width << '(';
    for (std::size_t index = 0; index < width; ++index)
    {
        if (index != 0) output << ',';
        output << number(value.is_array() && index < value.size() ? value[index] : json{}, fallback);
    }
    output << ')';
    return output.str();
}

std::string sanitize(std::string_view text)
{
    std::string result;
    result.reserve(text.size() + 2);
    for (const char character : text)
        result.push_back(std::isalnum(static_cast<unsigned char>(character)) ? character : '_');
    if (result.empty() || std::isdigit(static_cast<unsigned char>(result.front()))) result.insert(result.begin(), '_');
    return result;
}

shader_parameter_type parameter_type(std::string_view node_type)
{
    if (node_type == "vector2") return shader_parameter_type::float2;
    if (node_type == "vector3") return shader_parameter_type::float3;
    if (node_type == "vector4") return shader_parameter_type::float4;
    if (node_type == "textureSample") return shader_parameter_type::texture_2d;
    return shader_parameter_type::float32;
}

std::uint32_t parameter_size(shader_parameter_type type)
{
    switch (type)
    {
        case shader_parameter_type::float2: return 8;
        case shader_parameter_type::float3: return 12;
        case shader_parameter_type::float4: return 16;
        case shader_parameter_type::texture_2d: return 4;
        default: return 4;
    }
}

std::string_view slang_parameter_type(shader_parameter_type type)
{
    switch (type)
    {
        case shader_parameter_type::float2: return "float2";
        case shader_parameter_type::float3: return "float3";
        case shader_parameter_type::float4: return "float4";
        default: return "float";
    }
}
} // namespace

material_graph_lowering_result lower_material_graph_json(std::string_view graph_json)
{
    const auto document = json::parse(graph_json, nullptr, false);
    if (document.is_discarded() || !document.is_object() || document.value("version", 0) != 1 ||
        !document.contains("nodes") || !document["nodes"].is_array() || !document.contains("connections") ||
        !document["connections"].is_array())
        return material_graph_lowering_result::failure(
            {.code = shader_compile_error_code::invalid_request, .message = "material graph JSON is malformed"});

    std::map<std::string, json> nodes;
    for (const auto& node : document["nodes"])
    {
        const auto id = node.value("id", "");
        const auto type = node.value("type", "");
        if (id.empty() || type.empty() || nodes.contains(id))
            return material_graph_lowering_result::failure(
                {.code = shader_compile_error_code::validation_failed,
                 .message = "material graph contains a missing or duplicate stable node ID"});
        nodes.emplace(id, node);
    }

    struct connection
    {
        std::string source_node;
        std::string source_pin;
    };
    std::map<std::pair<std::string, std::string>, connection> inputs;
    for (const auto& edge : document["connections"])
    {
        if (!edge.contains("from") || !edge.contains("to")) continue;
        const auto source = edge["from"].value("nodeId", "");
        const auto source_pin = edge["from"].value("pin", "");
        const auto target = edge["to"].value("nodeId", "");
        const auto target_pin = edge["to"].value("pin", "");
        if (!nodes.contains(source) || !nodes.contains(target) || source_pin.empty() || target_pin.empty() ||
            !inputs.emplace(std::pair{target, target_pin}, connection{source, source_pin}).second)
            return material_graph_lowering_result::failure(
                {.code = shader_compile_error_code::validation_failed,
                 .message = "material graph contains an invalid or multiply-connected input"});
    }

    const auto output = std::ranges::find_if(nodes, [](const auto& entry)
                                             { return entry.second.value("type", "") == "output"; });
    if (output == nodes.end())
        return material_graph_lowering_result::failure(
            {.code = shader_compile_error_code::validation_failed, .message = "material graph has no output node"});

    material_graph_lowering lowered;
    std::ostringstream body;
    std::set<std::pair<std::string, std::string>> visiting;
    std::map<std::pair<std::string, std::string>, std::string> expressions;
    std::map<std::string, std::string> parameter_fields;
    std::set<std::string> reflected_parameters;
    std::vector<std::string> generated_nodes;
    std::uint32_t texture_index{};

    const auto emit = [&](const auto& self, const std::string& node_id, const std::string& pin) -> std::string
    {
        const auto key = std::pair{node_id, pin};
        if (const auto found = expressions.find(key); found != expressions.end()) return found->second;
        if (!visiting.insert(key).second) throw std::runtime_error("material graph contains a cycle");
        const auto& node = nodes.at(node_id);
        const auto type = node.value("type", "");
        const auto values = node.value("values", json::object());
        const auto input = [&](std::string_view name, std::string fallback)
        {
            const auto found = inputs.find({node_id, std::string(name)});
            return found == inputs.end() ? fallback : self(self, found->second.source_node, found->second.source_pin);
        };

        std::string expression;
        if (type == "constant") expression = number(values.value("value", json{}), 0.5f);
        else if (type == "vector2") expression = vector_value(values.value("value", json::array()), 2);
        else if (type == "vector3") expression = vector_value(values.value("value", json::array()), 3);
        else if (type == "vector4") expression = vector_value(values.value("value", json::array()), 4, 1.0f);
        else if (type == "texCoord") expression = "input.uv";
        else if (type == "time") expression = "arcFrame.timeSeconds";
        else if (type == "textureSample")
        {
            const auto sample = "arcMaterialTextures[" + std::to_string(texture_index++) +
                                "].Sample(arcMaterialSampler," + input("uv", "input.uv") + ')';
            expression = pin == "rgb" ? sample + ".rgb" : pin == "rgba" ? sample : sample + '.' + pin;
        }
        else if (type == "normalMap")
            expression = "normalize(lerp(float3(0.0,0.0,1.0)," + input("texture", "float3(0.5,0.5,1.0)") +
                         "*2.0-1.0," + number(values.value("strength", json{}), 1.0f) + "))";
        else if (type == "saturate") expression = "saturate(" + input("value", "0.0") + ')';
        else if (type == "clamp")
            expression = "clamp(" + input("value", "0.0") + ',' + input("min", number(values.value("min", json{}))) +
                         ',' + input("max", number(values.value("max", json{}), 1.0f)) + ')';
        else if (type == "lerp")
            expression = "lerp(" + input("a", "0.0") + ',' + input("b", "0.0") + ',' + input("t", "0.5") + ')';
        else if (type == "add" || type == "subtract" || type == "multiply" || type == "divide")
        {
            const char operation = type == "add" ? '+' : type == "subtract" ? '-' : type == "multiply" ? '*' : '/';
            expression = '(' + input("a", "0.0") + operation + input("b", type == "divide" ? "1.0" : "0.0") + ')';
        }
        else
            throw std::runtime_error("unsupported material graph node type: " + type);

        if (node.contains("parameter") && node["parameter"].value("exposed", false))
        {
            const auto stable_name = node["parameter"].value("name", node_id);
            const auto reflected_type = parameter_type(type);
            if (reflected_parameters.insert(node_id).second)
                lowered.parameters.push_back({.id = make_shader_parameter_id(node_id),
                                              .name = stable_name,
                                              .type = reflected_type,
                                              .size = parameter_size(reflected_type)});
            if (reflected_type != shader_parameter_type::texture_2d)
            {
                const auto field = "arc_param_" + std::to_string(make_shader_parameter_id(node_id).representation());
                parameter_fields.emplace(field, std::string(slang_parameter_type(reflected_type)));
                expression = "arcMaterialParameters." + field;
            }
        }
        const auto variable = "arc_node_" + sanitize(node_id) + '_' + sanitize(pin);
        body << "    auto " << variable << " = " << expression << ";\n";
        generated_nodes.push_back(node_id);
        visiting.erase(key);
        expressions.emplace(key, variable);
        return variable;
    };

    try
    {
        const auto output_value = [&](std::string_view pin, std::string fallback)
        {
            const auto found = inputs.find({output->first, std::string(pin)});
            return found == inputs.end() ? fallback : emit(emit, found->second.source_node, found->second.source_pin);
        };
        const auto base_color = output_value("baseColor", "float3(0.8,0.8,0.8)");
        const auto metallic = output_value("metallic", "0.0");
        const auto roughness = output_value("roughness", "0.6");
        const auto normal = output_value("normal", "input.normalWS");
        const auto ao = output_value("ao", "1.0");
        const auto emissive = output_value("emissive", "float3(0.0)");
        const auto opacity = output_value("opacity", "1.0");
        const auto alpha_clip = output_value("alphaClip", "0.5");

        std::ostringstream source;
        source << "// ARC generated material graph v1; shader-library v1\n"
                  "struct ArcFrame { float timeSeconds; };\n"
                  "struct ArcSurfaceInput { float2 uv:TEXCOORD0; float3 normalWS:TEXCOORD1; };\n"
                  "struct ArcSurfaceData { float3 baseColor; float metallic; float roughness; float3 normalWS; "
                  "float ao; float3 emissive; float opacity; float alphaClip; };\n"
                  "struct ArcMaterialParameters\n{\n";
        for (const auto& [field, type] : parameter_fields)
            source << "    " << type << ' ' << field << ";\n";
        source << "};\nParameterBlock<ArcMaterialParameters> arcMaterialParameters;\n"
                  "ParameterBlock<ArcFrame> arcFrame;\n"
                  "Texture2D<float4> arcMaterialTextures[];\nSamplerState arcMaterialSampler;\n"
                  "ArcSurfaceData arc_evaluate_surface(ArcSurfaceInput input)\n{\n"
               << body.str() << "    ArcSurfaceData surface;\n"
               << "    surface.baseColor=" << base_color << "; surface.metallic=" << metallic
               << "; surface.roughness=" << roughness << ";\n"
               << "    surface.normalWS=" << normal << "; surface.ao=" << ao << "; surface.emissive=" << emissive
               << ";\n    surface.opacity=" << opacity << "; surface.alphaClip=" << alpha_clip
               << "; return surface;\n}\n"
                  "[shader(\"fragment\")] float4 main(ArcSurfaceInput input):SV_Target\n{ ArcSurfaceData s="
                  "arc_evaluate_surface(input); return float4(s.baseColor+s.emissive,s.opacity); }\n";
        lowered.source = std::move(source).str();

        // Preamble: four fixed lines, parameter struct open/fields/close,
        // two parameter blocks, texture, sampler, function declaration/open.
        const auto body_first_line = static_cast<std::uint32_t>(14 + parameter_fields.size());
        for (std::size_t index = 0; index < generated_nodes.size(); ++index)
            lowered.generated_line_nodes.emplace(body_first_line + static_cast<std::uint32_t>(index),
                                                  generated_nodes[index]);
    }
    catch (const std::exception& error)
    {
        return material_graph_lowering_result::failure(
            {.code = shader_compile_error_code::validation_failed, .message = error.what()});
    }

    std::ranges::sort(lowered.parameters, {}, [](const shader_parameter_descriptor& value)
                      { return value.id.representation(); });
    return material_graph_lowering_result::success(std::move(lowered));
}

} // namespace arc::render::tools
