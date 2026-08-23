#include <arc/render_tools/material_graph.h>

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace arc::render::tools
{
namespace
{

std::string number(float value)
{
    std::ostringstream output;
    output << std::setprecision(9) << value;
    const auto result = output.str();
    return result.find_first_of(".eE") == std::string::npos ? result + ".0" : result;
}

std::string literal_value(const material_ir_literal& literal)
{
    if (literal.components <= 1) return number(literal.values[0]);

    std::ostringstream output;
    output << "float" << static_cast<unsigned int>(literal.components) << '(';
    for (std::uint8_t index = 0; index < literal.components; ++index)
    {
        if (index != 0) output << ',';
        output << number(literal.values[index]);
    }
    output << ')';
    return output.str();
}

std::string literal_value(const material_ir_node& node)
{
    return literal_value(node.literal);
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

std::string escaped_path(std::string_view text)
{
    std::string result;
    result.reserve(text.size());
    for (const char character : text)
    {
        if (character == '\\' || character == '"') result.push_back('\\');
        result.push_back(character == '\\' ? '/' : character);
    }
    return result;
}

std::string parameter_field(shader_parameter_id id)
{
    return "arc_param_" + std::to_string(id.representation());
}

std::string_view slang_parameter_type(shader_parameter_type type)
{
    switch (type)
    {
        case shader_parameter_type::float2:
            return "float2";
        case shader_parameter_type::float3:
            return "float3";
        case shader_parameter_type::float4:
            return "float4";
        default:
            return "float";
    }
}

std::string function_namespace(const material_ir_node& node)
{
    return "arc_function_" + std::to_string(make_shader_parameter_id(node.function_path).representation());
}

std::string function_output_variable(const material_ir_node& node, std::string_view pin)
{
    return "arc_node_" + sanitize(node.id) + '_' + sanitize(pin);
}

std::string function_prototype(const material_ir_node& node)
{
    std::ostringstream declaration;
    declaration << "void " << node.function_entry_point << '(';
    bool first = true;
    for (const auto& pin : node.function_inputs)
    {
        if (!first) declaration << ',';
        first = false;
        declaration << slang_parameter_type(pin.type) << " arc_input_" << sanitize(pin.id);
    }
    for (const auto& pin : node.function_outputs)
    {
        if (!first) declaration << ',';
        first = false;
        declaration << "out " << slang_parameter_type(pin.type) << " arc_output_" << sanitize(pin.id);
    }
    declaration << ");";
    return std::move(declaration).str();
}

enum class material_expression_type : std::uint8_t
{
    scalar = 1,
    vector2 = 2,
    vector3 = 3,
    vector4 = 4
};

material_expression_type expression_type(shader_parameter_type type)
{
    switch (type)
    {
        case shader_parameter_type::float2:
            return material_expression_type::vector2;
        case shader_parameter_type::float3:
            return material_expression_type::vector3;
        case shader_parameter_type::float4:
            return material_expression_type::vector4;
        default:
            return material_expression_type::scalar;
    }
}

material_expression_type expression_type(std::uint8_t components)
{
    switch (components)
    {
        case 2:
            return material_expression_type::vector2;
        case 3:
            return material_expression_type::vector3;
        case 4:
            return material_expression_type::vector4;
        default:
            return material_expression_type::scalar;
    }
}

std::string_view slang_expression_type(material_expression_type type)
{
    switch (type)
    {
        case material_expression_type::vector2:
            return "float2";
        case material_expression_type::vector3:
            return "float3";
        case material_expression_type::vector4:
            return "float4";
        default:
            return "float";
    }
}

material_expression_type widest(material_expression_type lhs, material_expression_type rhs)
{
    return static_cast<material_expression_type>(
        std::max(static_cast<std::uint8_t>(lhs), static_cast<std::uint8_t>(rhs)));
}

struct generated_statement
{
    std::string text;
    std::string node_id;
};

struct source_builder
{
    material_shader_source& generated;
    std::uint32_t line{1};

    void append(std::string_view text, std::string_view node_id = {})
    {
        if (!node_id.empty()) generated.generated_line_nodes.emplace(line, std::string(node_id));
        generated.source.append(text);
        generated.source.push_back('\n');
        ++line;
    }

    void append_lines(std::string_view text)
    {
        std::size_t begin{};
        while (begin < text.size())
        {
            const auto end = text.find('\n', begin);
            if (end == std::string_view::npos)
            {
                append(text.substr(begin));
                return;
            }
            append(text.substr(begin, end - begin));
            begin = end + 1;
        }
    }
};

class material_expression_generator
{
public:
    explicit material_expression_generator(const material_graph_compilation& compilation) : compilation_(compilation)
    {
        for (const auto& node : compilation_.ir.nodes)
        {
            nodes_.emplace(node.id, &node);
            if (node.kind != material_ir_node_kind::function_call) continue;
            const auto [found, inserted] = shader_functions_.try_emplace(node.function_path, &node);
            if (!inserted)
            {
                const auto& existing = *found->second;
                if (existing.function_entry_point != node.function_entry_point ||
                    existing.function_source != node.function_source ||
                    existing.function_inputs != node.function_inputs ||
                    existing.function_outputs != node.function_outputs)
                    throw std::runtime_error("Material Function path resolves to inconsistent shader definitions: " +
                                             node.function_path);
            }
        }
        for (const auto& connection : compilation_.ir.connections)
            inputs_.emplace(std::pair{connection.target_node, connection.target_pin}, &connection);
        for (const auto& texture : compilation_.descriptor.textures)
            texture_slots_.emplace(texture.node_id, texture.slot);
        for (const auto& parameter : compilation_.descriptor.parameters)
            parameter_types_.emplace(parameter.id.representation(), parameter.type);
    }

    std::string output(material_surface_output output, std::string fallback)
    {
        const auto binding =
            std::ranges::find_if(compilation_.descriptor.outputs, [output](const material_surface_output_binding& value)
                                 { return value.output == output; });
        if (binding == compilation_.descriptor.outputs.end() || !binding->connected) return fallback;
        return emit(binding->source_node, binding->source_pin);
    }

    const std::vector<generated_statement>& statements() const noexcept
    {
        return statements_;
    }

    const std::map<std::string, const material_ir_node*>& shader_functions() const noexcept
    {
        return shader_functions_;
    }

private:
    std::string input(const std::string& node_id, std::string_view pin, std::string fallback)
    {
        const auto found = inputs_.find({node_id, std::string(pin)});
        if (found == inputs_.end()) return fallback;
        return emit(found->second->source_node, found->second->source_pin);
    }

    material_expression_type input_type(const std::string& node_id, std::string_view pin,
                                        material_expression_type fallback)
    {
        const auto found = inputs_.find({node_id, std::string(pin)});
        if (found == inputs_.end()) return fallback;
        return infer_type(found->second->source_node, found->second->source_pin);
    }

    material_expression_type infer_type(const std::string& node_id, const std::string& pin)
    {
        const auto key = std::pair{node_id, pin};
        if (const auto found = expression_types_.find(key); found != expression_types_.end()) return found->second;
        if (!type_visiting_.insert(key).second)
            throw std::runtime_error("material IR contains a type cycle during code generation");

        const auto found_node = nodes_.find(node_id);
        if (found_node == nodes_.end()) throw std::runtime_error("material IR references a missing node: " + node_id);
        const auto& node = *found_node->second;

        material_expression_type type = material_expression_type::scalar;
        if (node.exposed_parameter && node.kind != material_ir_node_kind::texture_sample)
        {
            const auto parameter = parameter_types_.find(node.parameter_id.representation());
            if (parameter == parameter_types_.end())
                throw std::runtime_error("material IR exposed parameter is missing descriptor metadata: " + node.id);
            type = expression_type(parameter->second);
        }
        else
        {
            switch (node.kind)
            {
                case material_ir_node_kind::constant:
                case material_ir_node_kind::vector2:
                case material_ir_node_kind::vector3:
                case material_ir_node_kind::vector4:
                    type = expression_type(node.literal.components);
                    break;
                case material_ir_node_kind::tex_coord:
                    type = material_expression_type::vector2;
                    break;
                case material_ir_node_kind::time:
                    type = material_expression_type::scalar;
                    break;
                case material_ir_node_kind::texture_sample:
                    type = pin == "rgba"  ? material_expression_type::vector4
                           : pin == "rgb" ? material_expression_type::vector3
                                          : material_expression_type::scalar;
                    break;
                case material_ir_node_kind::normal_map:
                    type = material_expression_type::vector3;
                    break;
                case material_ir_node_kind::saturate:
                case material_ir_node_kind::clamp:
                    type = input_type(node.id, "value", material_expression_type::scalar);
                    break;
                case material_ir_node_kind::lerp:
                    type = widest(input_type(node.id, "a", material_expression_type::scalar),
                                  input_type(node.id, "b", material_expression_type::scalar));
                    break;
                case material_ir_node_kind::add:
                case material_ir_node_kind::subtract:
                case material_ir_node_kind::multiply:
                case material_ir_node_kind::divide:
                    type = widest(input_type(node.id, "a", material_expression_type::scalar),
                                  input_type(node.id, "b", material_expression_type::scalar));
                    break;
                case material_ir_node_kind::function_call:
                {
                    const auto output =
                        std::ranges::find_if(node.function_outputs, [&pin](const material_function_pin& candidate)
                                             { return candidate.id == pin; });
                    if (output == node.function_outputs.end())
                        throw std::runtime_error("Material Function call references an unknown output: " + node.id +
                                                 "." + pin);
                    type = expression_type(output->type);
                    break;
                }
                case material_ir_node_kind::output:
                    throw std::runtime_error("material output nodes cannot be emitted as expressions");
            }
        }

        type_visiting_.erase(key);
        expression_types_.emplace(key, type);
        return type;
    }

    std::string emit_function_call(const material_ir_node& node, const std::string& requested_pin)
    {
        if (emitted_function_calls_.insert(node.id).second)
        {
            for (const auto& output : node.function_outputs)
            {
                const auto variable = function_output_variable(node, output.id);
                statements_.push_back(
                    {.text = "    " + std::string(slang_parameter_type(output.type)) + ' ' + variable + ';',
                     .node_id = node.id});
                expressions_.emplace(std::pair{node.id, output.id}, variable);
                expression_types_.emplace(std::pair{node.id, output.id}, expression_type(output.type));
            }

            std::ostringstream call;
            call << "    " << function_namespace(node) << "::" << node.function_entry_point << '(';
            bool first = true;
            for (const auto& function_input : node.function_inputs)
            {
                if (!first) call << ',';
                first = false;
                const auto connected = inputs_.find({node.id, function_input.id});
                if (connected != inputs_.end())
                    call << emit(connected->second->source_node, connected->second->source_pin);
                else if (function_input.has_default)
                    call << literal_value(function_input.default_value);
                else
                    throw std::runtime_error("Material Function call is missing required input: " + node.id + "." +
                                             function_input.id);
            }
            for (const auto& function_output : node.function_outputs)
            {
                if (!first) call << ',';
                first = false;
                call << function_output_variable(node, function_output.id);
            }
            call << ");";
            statements_.push_back({.text = std::move(call).str(), .node_id = node.id});
        }

        const auto found = expressions_.find({node.id, requested_pin});
        if (found == expressions_.end())
            throw std::runtime_error("Material Function call references an unknown output: " + node.id + "." +
                                     requested_pin);
        return found->second;
    }

    std::string emit(const std::string& node_id, const std::string& pin)
    {
        const auto key = std::pair{node_id, pin};
        if (const auto found = expressions_.find(key); found != expressions_.end()) return found->second;
        if (!visiting_.insert(key).second)
            throw std::runtime_error("material IR contains a cycle during code generation");

        const auto found_node = nodes_.find(node_id);
        if (found_node == nodes_.end()) throw std::runtime_error("material IR references a missing node: " + node_id);
        const auto& node = *found_node->second;
        if (node.kind == material_ir_node_kind::function_call)
        {
            const auto variable = emit_function_call(node, pin);
            visiting_.erase(key);
            return variable;
        }

        const auto type = infer_type(node_id, pin);
        std::string expression;
        switch (node.kind)
        {
            case material_ir_node_kind::constant:
            case material_ir_node_kind::vector2:
            case material_ir_node_kind::vector3:
            case material_ir_node_kind::vector4:
                expression = literal_value(node);
                break;
            case material_ir_node_kind::tex_coord:
                expression = "input.uv0";
                break;
            case material_ir_node_kind::time:
                expression = "arcFrame.timeSeconds";
                break;
            case material_ir_node_kind::texture_sample:
            {
                const auto slot = texture_slots_.find(node.id);
                if (slot == texture_slots_.end())
                    throw std::runtime_error("material IR texture node has no descriptor slot: " + node.id);
                const auto sample = "arcMaterialTextures[" + std::to_string(slot->second) +
                                    "].Sample(arcMaterialSampler," + input(node.id, "uv", "input.uv0") + ')';
                expression = pin == "rgb" ? sample + ".rgb" : pin == "rgba" ? sample : sample + '.' + pin;
                break;
            }
            case material_ir_node_kind::normal_map:
                expression = "normalize(lerp(float3(0.0,0.0,1.0)," + input(node.id, "texture", "float3(0.5,0.5,1.0)") +
                             "*2.0-1.0," + number(node.strength) + "))";
                break;
            case material_ir_node_kind::saturate:
                expression = "saturate(" + input(node.id, "value", "0.0") + ')';
                break;
            case material_ir_node_kind::clamp:
                expression = "clamp(" + input(node.id, "value", "0.0") + ',' +
                             input(node.id, "min", number(node.minimum)) + ',' +
                             input(node.id, "max", number(node.maximum)) + ')';
                break;
            case material_ir_node_kind::lerp:
                expression = "lerp(" + input(node.id, "a", "0.0") + ',' + input(node.id, "b", "0.0") + ',' +
                             input(node.id, "t", "0.5") + ')';
                break;
            case material_ir_node_kind::add:
                expression = '(' + input(node.id, "a", "0.0") + '+' + input(node.id, "b", "0.0") + ')';
                break;
            case material_ir_node_kind::subtract:
                expression = '(' + input(node.id, "a", "0.0") + '-' + input(node.id, "b", "0.0") + ')';
                break;
            case material_ir_node_kind::multiply:
                expression = '(' + input(node.id, "a", "0.0") + '*' + input(node.id, "b", "0.0") + ')';
                break;
            case material_ir_node_kind::divide:
                expression = '(' + input(node.id, "a", "0.0") + '/' + input(node.id, "b", "1.0") + ')';
                break;
            case material_ir_node_kind::function_call:
                break;
            case material_ir_node_kind::output:
                throw std::runtime_error("material output nodes cannot be emitted as expressions");
        }

        if (node.exposed_parameter && node.kind != material_ir_node_kind::texture_sample)
            expression = "arcMaterialParameters." + parameter_field(node.parameter_id);

        const auto variable = "arc_node_" + sanitize(node.id) + '_' + sanitize(pin);
        statements_.push_back(
            {.text = "    " + std::string(slang_expression_type(type)) + ' ' + variable + " = " + expression + ';',
             .node_id = node.id});
        visiting_.erase(key);
        expressions_.emplace(key, variable);
        return variable;
    }

    const material_graph_compilation& compilation_;
    std::map<std::string, const material_ir_node*> nodes_;
    std::map<std::pair<std::string, std::string>, const material_ir_connection*> inputs_;
    std::map<std::string, std::uint32_t> texture_slots_;
    std::map<std::uint64_t, shader_parameter_type> parameter_types_;
    std::map<std::pair<std::string, std::string>, std::string> expressions_;
    std::map<std::pair<std::string, std::string>, material_expression_type> expression_types_;
    std::map<std::string, const material_ir_node*> shader_functions_;
    std::set<std::pair<std::string, std::string>> visiting_;
    std::set<std::pair<std::string, std::string>> type_visiting_;
    std::set<std::string> emitted_function_calls_;
    std::vector<generated_statement> statements_;
};

void append_material_abi(source_builder& source)
{
    source.append("static const uint ARC_MATERIAL_ABI_VERSION = 1;");
    source.append("struct ArcSurfaceInput");
    source.append("{");
    source.append("    float3 positionWS;");
    source.append("    float3 normalWS;");
    source.append("    float4 tangentWS;");
    source.append("    float2 uv0;");
    source.append("    float2 uv1;");
    source.append("    float4 vertexColor;");
    source.append("    float3 viewWS;");
    source.append("};");
    source.append("struct ArcSurfaceData");
    source.append("{");
    source.append("    float3 baseColor;");
    source.append("    float metallic;");
    source.append("    float roughness;");
    source.append("    float3 normalWS;");
    source.append("    float3 clearCoatNormalWS;");
    source.append("    float3 tangentWS;");
    source.append("    float ambientOcclusion;");
    source.append("    float3 emissiveRadiance;");
    source.append("    float opacity;");
    source.append("    float alphaCutoff;");
    source.append("    float indexOfRefraction;");
    source.append("    float clearCoat;");
    source.append("    float clearCoatRoughness;");
    source.append("    float sheen;");
    source.append("    float3 sheenColor;");
    source.append("    float sheenRoughness;");
    source.append("    float anisotropy;");
    source.append("    float anisotropyRotation;");
    source.append("    float transmission;");
    source.append("    float thickness;");
    source.append("    float3 attenuationColor;");
    source.append("    float attenuationDistance;");
    source.append("    float3 subsurfaceColor;");
    source.append("    float subsurface;");
    source.append("};");
    source.append("ArcSurfaceData arcDefaultSurface(float3 normalWS)");
    source.append("{");
    source.append("    ArcSurfaceData surface;");
    source.append("    surface.baseColor = float3(0.8);");
    source.append("    surface.metallic = 0.0;");
    source.append("    surface.roughness = 0.6;");
    source.append("    surface.normalWS = normalize(normalWS);");
    source.append("    surface.clearCoatNormalWS = surface.normalWS;");
    source.append("    surface.tangentWS = float3(1.0, 0.0, 0.0);");
    source.append("    surface.ambientOcclusion = 1.0;");
    source.append("    surface.emissiveRadiance = float3(0.0);");
    source.append("    surface.opacity = 1.0;");
    source.append("    surface.alphaCutoff = 0.5;");
    source.append("    surface.indexOfRefraction = 1.5;");
    source.append("    surface.clearCoat = 0.0;");
    source.append("    surface.clearCoatRoughness = 0.1;");
    source.append("    surface.sheen = 0.0;");
    source.append("    surface.sheenColor = float3(0.0);");
    source.append("    surface.sheenRoughness = 0.5;");
    source.append("    surface.anisotropy = 0.0;");
    source.append("    surface.anisotropyRotation = 0.0;");
    source.append("    surface.transmission = 0.0;");
    source.append("    surface.thickness = 0.0;");
    source.append("    surface.attenuationColor = float3(1.0);");
    source.append("    surface.attenuationDistance = 1.0;");
    source.append("    surface.subsurfaceColor = float3(1.0, 0.35, 0.2);");
    source.append("    surface.subsurface = 0.0;");
    source.append("    return surface;");
    source.append("}");
}

void append_function_declarations(source_builder& source,
                                  const std::map<std::string, const material_ir_node*>& functions)
{
    for (const auto& [path, function] : functions)
    {
        static_cast<void>(path);
        source.append("namespace " + function_namespace(*function));
        source.append("{");
        source.append("    " + function_prototype(*function));
        source.append("}");
    }
}

void append_function_definitions(source_builder& source,
                                 const std::map<std::string, const material_ir_node*>& functions)
{
    for (const auto& [path, function] : functions)
    {
        source.append("namespace " + function_namespace(*function));
        source.append("{");
        source.append("#line 1 \"" + escaped_path(path) + "\"");
        source.append_lines(function->function_source);
        source.append("#line 1 \"arc-generated-material.slang\"");
        source.append("}");
    }
}

void append_compiler_input(source_builder& source)
{
    source.append("struct ArcCompilerInput");
    source.append("{");
    source.append("    float3 positionWS : TEXCOORD0;");
    source.append("    float3 normalWS : TEXCOORD1;");
    source.append("    float4 tangentWS : TEXCOORD2;");
    source.append("    float2 uv0 : TEXCOORD3;");
    source.append("    float2 uv1 : TEXCOORD4;");
    source.append("    float4 vertexColor : COLOR0;");
    source.append("    float3 viewWS : TEXCOORD5;");
    source.append("};");
    source.append("ArcSurfaceInput arcMakeSurfaceInput(ArcCompilerInput compilerInput)");
    source.append("{");
    source.append("    ArcSurfaceInput input;");
    source.append("    input.positionWS = compilerInput.positionWS;");
    source.append("    input.normalWS = compilerInput.normalWS;");
    source.append("    input.tangentWS = compilerInput.tangentWS;");
    source.append("    input.uv0 = compilerInput.uv0;");
    source.append("    input.uv1 = compilerInput.uv1;");
    source.append("    input.vertexColor = compilerInput.vertexColor;");
    source.append("    input.viewWS = compilerInput.viewWS;");
    source.append("    return input;");
    source.append("}");
}

} // namespace

material_shader_codegen_result generate_material_slang(const material_graph_compilation& compilation)
{
    if (compilation.ir.version != material_ir_version || compilation.descriptor.material_abi != material_abi_version)
        return material_shader_codegen_result::failure(
            {.code = shader_compile_error_code::validation_failed,
             .message = "material code generation requires the current Material IR and ABI versions"});

    material_shader_source generated;
    generated.parameters = compilation.descriptor.parameters;
    generated.diagnostics = compilation.diagnostics;

    try
    {
        material_expression_generator expressions(compilation);
        const auto base_color = expressions.output(material_surface_output::base_color, {});
        const auto metallic = expressions.output(material_surface_output::metallic, {});
        const auto roughness = expressions.output(material_surface_output::roughness, {});
        const auto normal = expressions.output(material_surface_output::normal, {});
        const auto clear_coat_normal = expressions.output(material_surface_output::clear_coat_normal, {});
        const auto tangent = expressions.output(material_surface_output::tangent, {});
        const auto ambient_occlusion = expressions.output(material_surface_output::ambient_occlusion, {});
        const auto emissive = expressions.output(material_surface_output::emissive, {});
        const auto opacity = expressions.output(material_surface_output::opacity, {});
        const auto alpha_cutoff = expressions.output(material_surface_output::alpha_cutoff, {});
        const auto index_of_refraction = expressions.output(material_surface_output::index_of_refraction, {});
        const auto clear_coat = expressions.output(material_surface_output::clear_coat, {});
        const auto clear_coat_roughness = expressions.output(material_surface_output::clear_coat_roughness, {});
        const auto sheen = expressions.output(material_surface_output::sheen, {});
        const auto sheen_color = expressions.output(material_surface_output::sheen_color, {});
        const auto sheen_roughness = expressions.output(material_surface_output::sheen_roughness, {});
        const auto anisotropy = expressions.output(material_surface_output::anisotropy, {});
        const auto anisotropy_rotation = expressions.output(material_surface_output::anisotropy_rotation, {});
        const auto transmission = expressions.output(material_surface_output::transmission, {});
        const auto thickness = expressions.output(material_surface_output::thickness, {});
        const auto attenuation_color = expressions.output(material_surface_output::attenuation_color, {});
        const auto attenuation_distance = expressions.output(material_surface_output::attenuation_distance, {});
        const auto subsurface_color = expressions.output(material_surface_output::subsurface_color, {});
        const auto subsurface = expressions.output(material_surface_output::subsurface, {});

        source_builder source{generated};
        source.append("// ARC generated Material IR v1; Material ABI v1; codegen v3.");
        append_material_abi(source);

        const auto has_material_parameters =
            std::ranges::any_of(generated.parameters, [](const shader_parameter_descriptor& parameter)
                                { return parameter.type != shader_parameter_type::texture_2d; });
        if (has_material_parameters)
        {
            source.append("struct ArcMaterialParameters");
            source.append("{");
            for (const auto& parameter : generated.parameters)
                if (parameter.type != shader_parameter_type::texture_2d)
                    source.append("    " + std::string(slang_parameter_type(parameter.type)) + ' ' +
                                  parameter_field(parameter.id) + ';');
            source.append("};");
            source.append("ParameterBlock<ArcMaterialParameters> arcMaterialParameters;");
        }
        if (compilation.descriptor.requirements.uses_time)
        {
            source.append("struct ArcFrame");
            source.append("{");
            source.append("    float timeSeconds;");
            source.append("};");
            source.append("ParameterBlock<ArcFrame> arcFrame;");
        }
        if (compilation.descriptor.requirements.uses_texture_sampling)
        {
            source.append("Texture2D<float4> arcMaterialTextures[];");
            source.append("SamplerState arcMaterialSampler;");
        }

        append_function_declarations(source, expressions.shader_functions());
        source.append("ArcSurfaceData arc_evaluate_material(ArcSurfaceInput input)");
        source.append("{");
        source.append("    ArcSurfaceData surface = arcDefaultSurface(input.normalWS);");
        for (const auto& statement : expressions.statements())
            source.append(statement.text, statement.node_id);
        if (!base_color.empty()) source.append("    surface.baseColor = " + base_color + ';');
        if (!metallic.empty()) source.append("    surface.metallic = " + metallic + ';');
        if (!roughness.empty()) source.append("    surface.roughness = " + roughness + ';');
        if (!normal.empty()) source.append("    surface.normalWS = " + normal + ';');
        if (!clear_coat_normal.empty()) source.append("    surface.clearCoatNormalWS = " + clear_coat_normal + ';');
        if (!tangent.empty()) source.append("    surface.tangentWS = " + tangent + ';');
        if (!ambient_occlusion.empty()) source.append("    surface.ambientOcclusion = " + ambient_occlusion + ';');
        if (!emissive.empty()) source.append("    surface.emissiveRadiance = " + emissive + ';');
        if (!opacity.empty()) source.append("    surface.opacity = " + opacity + ';');
        if (!alpha_cutoff.empty()) source.append("    surface.alphaCutoff = " + alpha_cutoff + ';');
        if (!index_of_refraction.empty()) source.append("    surface.indexOfRefraction = " + index_of_refraction + ';');
        if (!clear_coat.empty()) source.append("    surface.clearCoat = " + clear_coat + ';');
        if (!clear_coat_roughness.empty())
            source.append("    surface.clearCoatRoughness = " + clear_coat_roughness + ';');
        if (!sheen.empty()) source.append("    surface.sheen = " + sheen + ';');
        if (!sheen_color.empty()) source.append("    surface.sheenColor = " + sheen_color + ';');
        if (!sheen_roughness.empty()) source.append("    surface.sheenRoughness = " + sheen_roughness + ';');
        if (!anisotropy.empty()) source.append("    surface.anisotropy = " + anisotropy + ';');
        if (!anisotropy_rotation.empty())
            source.append("    surface.anisotropyRotation = " + anisotropy_rotation + ';');
        if (!transmission.empty()) source.append("    surface.transmission = " + transmission + ';');
        if (!thickness.empty()) source.append("    surface.thickness = " + thickness + ';');
        if (!attenuation_color.empty()) source.append("    surface.attenuationColor = " + attenuation_color + ';');
        if (!attenuation_distance.empty())
            source.append("    surface.attenuationDistance = " + attenuation_distance + ';');
        if (!subsurface_color.empty()) source.append("    surface.subsurfaceColor = " + subsurface_color + ';');
        if (!subsurface.empty()) source.append("    surface.subsurface = " + subsurface + ';');
        source.append("    return surface;");
        source.append("}");

        append_function_definitions(source, expressions.shader_functions());
        append_compiler_input(source);
        source.append("[shader(\"fragment\")] float4 main(ArcCompilerInput compilerInput) : SV_Target");
        source.append("{");
        source.append("    ArcSurfaceData surface = arc_evaluate_material(arcMakeSurfaceInput(compilerInput));");
        source.append("    return float4(surface.baseColor + surface.emissiveRadiance, surface.opacity);");
        source.append("}");
    }
    catch (const std::exception& error)
    {
        return material_shader_codegen_result::failure(
            {.code = shader_compile_error_code::validation_failed, .message = error.what()});
    }

    return material_shader_codegen_result::success(std::move(generated));
}

} // namespace arc::render::tools
