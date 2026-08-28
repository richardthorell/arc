#include <arc/editor/material_preview_realizer.h>

#include <arc/render_tools/render_tools.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace arc::editor
{
namespace
{

using render::tools::material_graph_compilation;
using render::tools::material_ir_node;
using render::tools::material_ir_node_kind;
using render::tools::material_surface_output;

struct static_material_value
{
    std::array<float, 4> values{};
    std::uint8_t components{};
};

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

static_material_value literal_value(const render::tools::material_ir_literal& literal)
{
    return {.values = literal.values, .components = literal.components};
}

static_material_value scalar_value(float value)
{
    return {.values = {value, 0.0f, 0.0f, 0.0f}, .components = 1};
}

std::optional<static_material_value> promote(static_material_value value, std::uint8_t components)
{
    if (value.components == components) return value;
    if (value.components != 1 || components == 0 || components > 4) return std::nullopt;
    for (std::uint8_t index = 1; index < components; ++index)
        value.values[index] = value.values[0];
    value.components = components;
    return value;
}

std::optional<static_material_value> combine(const static_material_value& lhs, const static_material_value& rhs,
                                             char operation)
{
    const auto components = std::max(lhs.components, rhs.components);
    auto left = promote(lhs, components);
    auto right = promote(rhs, components);
    if (!left || !right) return std::nullopt;

    static_material_value result;
    result.components = components;
    for (std::uint8_t index = 0; index < components; ++index)
    {
        switch (operation)
        {
            case '+':
                result.values[index] = left->values[index] + right->values[index];
                break;
            case '-':
                result.values[index] = left->values[index] - right->values[index];
                break;
            case '*':
                result.values[index] = left->values[index] * right->values[index];
                break;
            case '/':
                if (std::abs(right->values[index]) <= 1.0e-8f) return std::nullopt;
                result.values[index] = left->values[index] / right->values[index];
                break;
            default:
                return std::nullopt;
        }
    }
    return result;
}

class static_material_evaluator
{
public:
    explicit static_material_evaluator(const material_graph_compilation& compilation) : compilation_(compilation)
    {
        for (const auto& node : compilation_.ir.nodes)
            nodes_.emplace(node.id, &node);
        for (const auto& connection : compilation_.ir.connections)
            inputs_.emplace(std::pair{connection.target_node, connection.target_pin},
                            std::pair{connection.source_node, connection.source_pin});
    }

    std::optional<static_material_value> output(material_surface_output output)
    {
        const auto binding = std::ranges::find(compilation_.descriptor.outputs, output,
                                               &render::tools::material_surface_output_binding::output);
        if (binding == compilation_.descriptor.outputs.end() || !binding->connected) return std::nullopt;
        return evaluate(binding->source_node, binding->source_pin);
    }

    bool connected(material_surface_output output) const
    {
        const auto binding = std::ranges::find(compilation_.descriptor.outputs, output,
                                               &render::tools::material_surface_output_binding::output);
        return binding != compilation_.descriptor.outputs.end() && binding->connected;
    }

private:
    std::optional<static_material_value> input(const std::string& node_id, std::string_view pin,
                                               static_material_value fallback)
    {
        const auto found = inputs_.find({node_id, std::string(pin)});
        return found == inputs_.end() ? std::optional<static_material_value>{fallback}
                                      : evaluate(found->second.first, found->second.second);
    }

    std::optional<static_material_value> evaluate(const std::string& node_id, const std::string& pin)
    {
        const auto key = std::pair{node_id, pin};
        if (const auto cached = cache_.find(key); cached != cache_.end()) return cached->second;
        if (!active_.insert(key).second) return std::nullopt;

        const auto found = nodes_.find(node_id);
        if (found == nodes_.end())
        {
            active_.erase(key);
            return std::nullopt;
        }

        const auto& node = *found->second;
        std::optional<static_material_value> result;
        switch (node.kind)
        {
            case material_ir_node_kind::constant:
            case material_ir_node_kind::vector2:
            case material_ir_node_kind::vector3:
            case material_ir_node_kind::vector4:
                result = literal_value(node.literal);
                break;
            case material_ir_node_kind::saturate:
            {
                result = input(node.id, "value", scalar_value(0.0f));
                if (result)
                    for (std::uint8_t index = 0; index < result->components; ++index)
                        result->values[index] = std::clamp(result->values[index], 0.0f, 1.0f);
                break;
            }
            case material_ir_node_kind::clamp:
            {
                auto value = input(node.id, "value", scalar_value(0.0f));
                auto minimum = input(node.id, "min", scalar_value(node.minimum));
                auto maximum = input(node.id, "max", scalar_value(node.maximum));
                if (value && minimum && maximum)
                {
                    const auto components = value->components;
                    minimum = promote(*minimum, components);
                    maximum = promote(*maximum, components);
                    if (minimum && maximum)
                    {
                        result = *value;
                        for (std::uint8_t index = 0; index < components; ++index)
                            result->values[index] =
                                std::clamp(result->values[index], minimum->values[index], maximum->values[index]);
                    }
                }
                break;
            }
            case material_ir_node_kind::lerp:
            {
                auto lhs = input(node.id, "a", scalar_value(0.0f));
                auto rhs = input(node.id, "b", scalar_value(0.0f));
                auto amount = input(node.id, "t", scalar_value(0.5f));
                if (lhs && rhs && amount)
                {
                    const auto components = std::max(lhs->components, rhs->components);
                    lhs = promote(*lhs, components);
                    rhs = promote(*rhs, components);
                    amount = promote(*amount, components);
                    if (lhs && rhs && amount)
                    {
                        static_material_value value;
                        value.components = components;
                        for (std::uint8_t index = 0; index < components; ++index)
                            value.values[index] = lhs->values[index] * (1.0f - amount->values[index]) +
                                                  rhs->values[index] * amount->values[index];
                        result = value;
                    }
                }
                break;
            }
            case material_ir_node_kind::add:
            case material_ir_node_kind::subtract:
            case material_ir_node_kind::multiply:
            case material_ir_node_kind::divide:
            {
                auto lhs = input(node.id, "a", scalar_value(0.0f));
                auto rhs = input(node.id, "b", scalar_value(node.kind == material_ir_node_kind::divide ? 1.0f : 0.0f));
                if (lhs && rhs)
                {
                    const char operation = node.kind == material_ir_node_kind::add        ? '+'
                                           : node.kind == material_ir_node_kind::subtract ? '-'
                                           : node.kind == material_ir_node_kind::multiply ? '*'
                                                                                          : '/';
                    result = combine(*lhs, *rhs, operation);
                }
                break;
            }
            case material_ir_node_kind::normal_map:
            {
                auto texture = input(node.id, "texture", static_material_value{{0.5f, 0.5f, 1.0f, 0.0f}, 3});
                if (texture)
                {
                    texture = promote(*texture, 3);
                    if (texture)
                    {
                        const std::array<float, 3> flat{0.0f, 0.0f, 1.0f};
                        std::array<float, 3> normal{};
                        for (std::size_t index = 0; index < normal.size(); ++index)
                        {
                            const float sampled = texture->values[index] * 2.0f - 1.0f;
                            normal[index] = flat[index] * (1.0f - node.strength) + sampled * node.strength;
                        }
                        const float length = std::sqrt(
                            std::max(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2], 1.0e-12f));
                        result = static_material_value{
                            {normal[0] / length, normal[1] / length, normal[2] / length, 0.0f}, 3};
                    }
                }
                break;
            }
            case material_ir_node_kind::math:
            case material_ir_node_kind::tex_coord:
            case material_ir_node_kind::time:
            case material_ir_node_kind::texture_sample:
            case material_ir_node_kind::function_call:
            case material_ir_node_kind::output:
                break;
        }

        active_.erase(key);
        cache_.emplace(key, result);
        return result;
    }

    const material_graph_compilation& compilation_;
    std::map<std::string, const material_ir_node*> nodes_;
    std::map<std::pair<std::string, std::string>, std::pair<std::string, std::string>> inputs_;
    std::map<std::pair<std::string, std::string>, std::optional<static_material_value>> cache_;
    std::set<std::pair<std::string, std::string>> active_;
};

const char* output_name(material_surface_output output) noexcept
{
    switch (output)
    {
        case material_surface_output::base_color:
            return "Base Color";
        case material_surface_output::metallic:
            return "Metallic";
        case material_surface_output::roughness:
            return "Roughness";
        case material_surface_output::normal:
            return "Normal";
        case material_surface_output::clear_coat_normal:
            return "Clear Coat Normal";
        case material_surface_output::tangent:
            return "Tangent";
        case material_surface_output::ambient_occlusion:
            return "Ambient Occlusion";
        case material_surface_output::emissive:
            return "Emissive";
        case material_surface_output::opacity:
            return "Opacity";
        case material_surface_output::alpha_cutoff:
            return "Alpha Cutoff";
        case material_surface_output::index_of_refraction:
            return "Index of Refraction";
        case material_surface_output::clear_coat:
            return "Clear Coat";
        case material_surface_output::clear_coat_roughness:
            return "Clear Coat Roughness";
        case material_surface_output::sheen:
            return "Sheen";
        case material_surface_output::sheen_color:
            return "Sheen Color";
        case material_surface_output::sheen_roughness:
            return "Sheen Roughness";
        case material_surface_output::anisotropy:
            return "Anisotropy";
        case material_surface_output::anisotropy_rotation:
            return "Anisotropy Rotation";
        case material_surface_output::transmission:
            return "Transmission";
        case material_surface_output::thickness:
            return "Thickness";
        case material_surface_output::attenuation_color:
            return "Attenuation Color";
        case material_surface_output::attenuation_distance:
            return "Attenuation Distance";
        case material_surface_output::subsurface_color:
            return "Subsurface Color";
        case material_surface_output::subsurface:
            return "Subsurface";
    }
    return "Material Output";
}

void append_dynamic_diagnostic(material_preview_descriptor_result& result, const static_material_evaluator& evaluator,
                               material_surface_output output)
{
    if (evaluator.connected(output))
        result.diagnostics.push_back(std::string(output_name(output)) +
                                     " is dynamic; native preview is using the Material ABI default until compiled "
                                     "runtime pass binding is available");
}

std::optional<float> scalar(const std::optional<static_material_value>& value)
{
    if (!value || value->components != 1 || !std::isfinite(value->values[0])) return std::nullopt;
    return value->values[0];
}

std::optional<std::array<float, 3>> vector3(const std::optional<static_material_value>& value)
{
    if (!value) return std::nullopt;
    auto promoted = promote(*value, 3);
    if (!promoted) return std::nullopt;
    for (std::uint8_t index = 0; index < 3; ++index)
        if (!std::isfinite(promoted->values[index])) return std::nullopt;
    return std::array<float, 3>{promoted->values[0], promoted->values[1], promoted->values[2]};
}

render::material_descriptor material_abi_preview_defaults(std::string_view name,
                                                          const render::tools::material_authoring_document& authored)
{
    render::material_descriptor material;
    material.name = name.empty() ? "Material Preview" : std::string{name};
    material.domain = authored.domain;
    material.shading_model = authored.shading_model;
    material.alpha_mode = authored.alpha_mode;
    material.double_sided = authored.double_sided;
    material.base_color = {0.8f, 0.8f, 0.8f, 1.0f};
    material.metallic = 0.0f;
    material.roughness = 0.6f;
    material.alpha_cutoff = 0.5f;
    material.emissive_factor = {0.0f, 0.0f, 0.0f};
    material.emissive_strength = 1.0f;
    material.index_of_refraction = 1.5f;
    material.clear_coat_factor = 0.0f;
    material.clear_coat_roughness = 0.1f;
    material.sheen_factor = 0.0f;
    material.sheen_color = {0.0f, 0.0f, 0.0f};
    material.anisotropy_factor = 0.0f;
    material.anisotropy_rotation = 0.0f;
    material.transmission_factor = 0.0f;
    material.thickness_factor = 0.0f;
    material.attenuation_color = {1.0f, 1.0f, 1.0f};
    material.attenuation_distance = 1.0f;
    material.subsurface_color = {1.0f, 0.35f, 0.2f};
    material.subsurface_factor = 0.0f;
    return material;
}

std::shared_ptr<render::material_runtime_program>
compile_preview_runtime_program(const material_graph_compilation& compilation,
                                const render::material_descriptor& material, std::vector<std::string>& diagnostics)
{
    auto evaluator = render::tools::make_graph_material_evaluator(compilation);
    if (!evaluator)
    {
        diagnostics.push_back("Compiled Material ABI preview evaluator generation failed: " +
                              evaluator.error().message);
        return {};
    }

    auto generated =
        render::tools::generate_material_pass_slang(evaluator.value(), material, render::material_pass::gbuffer);
    if (!generated)
    {
        diagnostics.push_back("Compiled Material ABI G-buffer generation failed: " + generated.error().message);
        return {};
    }

    render::tools::slang_shader_compiler compiler;
    if (!compiler.available())
    {
        diagnostics.push_back(
            "Pinned Slang compiler is unavailable; native preview is using static Material ABI defaults");
        return {};
    }

    render::shader_compile_request request{.source_path = material.name + ".preview.gbuffer.generated.slang",
                                           .source_override = generated.value().source,
                                           .entry_point = generated.value().entry_point,
                                           .profile = "spirv_1_5",
                                           .library_version = "arc-material-preview/1",
                                           .domain = render::shader_domain::surface,
                                           .stage = render::shader_stage::fragment,
                                           .target = render::shader_target::spirv,
                                           .optimization = render::shader_optimization::development,
                                           .required_passes = {render::material_pass::gbuffer},
                                           .generated_line_nodes = generated.value().generated_line_nodes,
                                           .generate_debug_information = true};
    auto compiled = compiler.compile(request);
    if (!compiled)
    {
        diagnostics.push_back("Compiled Material ABI preview shader compilation failed: " + compiled.error().message);
        return {};
    }

    auto program = std::make_shared<render::material_runtime_program>();
    program->uses_time = compilation.descriptor.requirements.uses_time;
    program->uses_texture_sampling = compilation.descriptor.requirements.uses_texture_sampling;
    for (const auto& texture : compilation.descriptor.textures)
        program->texture_bindings.push_back({.slot = texture.slot, .parameter_id = texture.parameter_id});

    for (const auto& authored_parameter : evaluator.value().parameters)
    {
        if (authored_parameter.type == render::shader_parameter_type::texture_2d ||
            authored_parameter.type == render::shader_parameter_type::texture_cube ||
            authored_parameter.type == render::shader_parameter_type::sampler)
            continue;

        const auto field_name = "arc_param_" + std::to_string(authored_parameter.id.representation());
        const auto reflected = std::ranges::find(compiled.value().reflection.parameters, field_name,
                                                 &render::shader_parameter_descriptor::name);
        if (reflected == compiled.value().reflection.parameters.end())
        {
            diagnostics.push_back("Compiled Material ABI reflection is missing parameter field '" + field_name + "'");
            return {};
        }

        auto parameter = authored_parameter;
        parameter.offset = reflected->offset;
        parameter.size = reflected->size;
        program->parameter_block_size = std::max(program->parameter_block_size, parameter.offset + parameter.size);
        program->parameters.push_back(std::move(parameter));
    }

    program->parameter_defaults.assign(program->parameter_block_size, std::byte{});
    for (const auto& parameter : program->parameters)
    {
        const auto authored =
            std::ranges::find(evaluator.value().parameters, parameter.id, &render::shader_parameter_descriptor::id);
        if (authored == evaluator.value().parameters.end() || authored->default_value.empty() ||
            parameter.offset >= program->parameter_defaults.size())
            continue;
        const auto available = program->parameter_defaults.size() - parameter.offset;
        const auto bytes = std::min<std::size_t>({authored->default_value.size(), parameter.size, available});
        std::memcpy(program->parameter_defaults.data() + parameter.offset, authored->default_value.data(), bytes);
    }

    program->passes.push_back({.pass = render::material_pass::gbuffer,
                               .permutation = generated.value().permutation,
                               .compiled = std::move(compiled).value()});
    return program;
}

} // namespace

material_preview_descriptor_result realize_material_preview_descriptor(std::string_view source, std::string_view name)
{
    material_preview_descriptor_result result;
    auto authored = render::tools::parse_material_authoring_json(source);
    if (!authored)
    {
        result.message = authored.error().message;
        return result;
    }

    result.material = material_abi_preview_defaults(name, authored.value());
    if (authored.value().graph_json.empty())
    {
        result.succeeded = true;
        result.message = "Custom Material Shader preview is using Material ABI defaults until compiled runtime pass "
                         "binding is available";
        result.diagnostics.push_back(result.message);
        result.material.render_path = render::resolve_material_render_path(result.material);
        return result;
    }

    auto compiled = render::tools::compile_material_graph_json(authored.value().graph_json);
    if (!compiled)
    {
        result.message = compiled.error().message;
        return result;
    }

    result.texture_sources.resize(compiled.value().descriptor.textures.size());
    const auto graph_document = nlohmann::json::parse(authored.value().graph_json, nullptr, false);
    if (!graph_document.is_discarded() && graph_document.contains("nodes") && graph_document["nodes"].is_array())
    {
        for (const auto& texture : compiled.value().descriptor.textures)
        {
            if (texture.slot >= result.texture_sources.size()) continue;
            const auto node =
                std::ranges::find_if(graph_document["nodes"], [&](const auto& candidate)
                                     { return candidate.is_object() && candidate.value("id", "") == texture.node_id; });
            if (node == graph_document["nodes"].end()) continue;
            const auto values = node->value("values", nlohmann::json::object());
            result.texture_sources[texture.slot] = values.value("texture", "");
        }
    }

    static_material_evaluator evaluator(compiled.value());
    const auto apply_scalar = [&](material_surface_output output, float& target)
    {
        const auto evaluated = scalar(evaluator.output(output));
        if (evaluated)
            target = *evaluated;
        else
            append_dynamic_diagnostic(result, evaluator, output);
    };
    const auto apply_color = [&](material_surface_output output, math::vector3f& target)
    {
        const auto evaluated = vector3(evaluator.output(output));
        if (evaluated)
            target = {(*evaluated)[0], (*evaluated)[1], (*evaluated)[2]};
        else
            append_dynamic_diagnostic(result, evaluator, output);
    };

    if (const auto base_color = vector3(evaluator.output(material_surface_output::base_color)))
    {
        result.material.base_color[0] = (*base_color)[0];
        result.material.base_color[1] = (*base_color)[1];
        result.material.base_color[2] = (*base_color)[2];
    }
    else
        append_dynamic_diagnostic(result, evaluator, material_surface_output::base_color);

    apply_scalar(material_surface_output::metallic, result.material.metallic);
    apply_scalar(material_surface_output::roughness, result.material.roughness);
    apply_color(material_surface_output::emissive, result.material.emissive_factor);
    if (const auto opacity = scalar(evaluator.output(material_surface_output::opacity)))
        result.material.base_color[3] = *opacity;
    else
        append_dynamic_diagnostic(result, evaluator, material_surface_output::opacity);
    apply_scalar(material_surface_output::alpha_cutoff, result.material.alpha_cutoff);
    apply_scalar(material_surface_output::index_of_refraction, result.material.index_of_refraction);
    apply_scalar(material_surface_output::clear_coat, result.material.clear_coat_factor);
    apply_scalar(material_surface_output::clear_coat_roughness, result.material.clear_coat_roughness);
    apply_scalar(material_surface_output::sheen, result.material.sheen_factor);
    apply_color(material_surface_output::sheen_color, result.material.sheen_color);
    apply_scalar(material_surface_output::anisotropy, result.material.anisotropy_factor);
    apply_scalar(material_surface_output::anisotropy_rotation, result.material.anisotropy_rotation);
    apply_scalar(material_surface_output::transmission, result.material.transmission_factor);
    apply_scalar(material_surface_output::thickness, result.material.thickness_factor);
    apply_color(material_surface_output::attenuation_color, result.material.attenuation_color);
    apply_scalar(material_surface_output::attenuation_distance, result.material.attenuation_distance);
    apply_color(material_surface_output::subsurface_color, result.material.subsurface_color);
    apply_scalar(material_surface_output::subsurface, result.material.subsurface_factor);

    // The descriptor-backed preview shader has no constant slots for these Material ABI outputs. Preserve the ABI
    // default rather than silently pretending the preview can represent a dynamic normal/AO/tangent expression.
    append_dynamic_diagnostic(result, evaluator, material_surface_output::normal);
    append_dynamic_diagnostic(result, evaluator, material_surface_output::clear_coat_normal);
    append_dynamic_diagnostic(result, evaluator, material_surface_output::tangent);
    append_dynamic_diagnostic(result, evaluator, material_surface_output::ambient_occlusion);
    append_dynamic_diagnostic(result, evaluator, material_surface_output::sheen_roughness);

    result.material.render_path = render::resolve_material_render_path(result.material);
    result.material.runtime_program =
        compile_preview_runtime_program(compiled.value(), result.material, result.diagnostics);
    if (result.material.runtime_program)
    {
        std::erase_if(
            result.diagnostics, [](const std::string& diagnostic)
            { return diagnostic.find("until compiled runtime pass binding is available") != std::string::npos; });
        result.diagnostics.push_back("Compiled Material ABI G-buffer preview pass is active");
    }

    result.succeeded = true;
    result.message =
        result.material.runtime_program ? "Material preview realized through compiled Material ABI G-buffer pass"
        : result.diagnostics.empty()    ? "Material preview realized from native Material IR"
                                     : "Material preview realized from native Material IR with dynamic-output defaults";
    return result;
}

material_preview_descriptor_result load_material_preview_descriptor(const std::filesystem::path& source_path)
{
    const auto source = read_text_file(source_path);
    if (source.empty())
        return {.message = "Material preview source could not be read: " + source_path.generic_string()};
    return realize_material_preview_descriptor(source, source_path.stem().string());
}

} // namespace arc::editor
