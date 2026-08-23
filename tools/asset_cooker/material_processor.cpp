#include "material_processor.h"

#include <arc/render_tools/render_tools.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::tools
{
namespace
{
using json = nlohmann::json;

constexpr std::array material_passes{render::material_pass::depth,    render::material_pass::shadow,
                                     render::material_pass::gbuffer,  render::material_pass::forward,
                                     render::material_pass::motion,   render::material_pass::object_id,
                                     render::material_pass::selection};

std::string_view pass_name(render::material_pass pass) noexcept
{
    switch (pass)
    {
        case render::material_pass::depth:
            return "depth";
        case render::material_pass::shadow:
            return "shadow";
        case render::material_pass::gbuffer:
            return "gbuffer";
        case render::material_pass::forward:
            return "forward";
        case render::material_pass::motion:
            return "motion";
        case render::material_pass::object_id:
            return "object-id";
        case render::material_pass::selection:
            return "selection";
        case render::material_pass::ray_hit:
            return "ray-hit";
    }
    return "unknown";
}

std::string compile_error_message(const render::shader_compile_error& error)
{
    std::string message = error.message;
    for (const auto& diagnostic : error.diagnostics)
        message += "\n" + diagnostic.location.path + ":" + std::to_string(diagnostic.location.line) + ":" +
                   std::to_string(diagnostic.location.column) + ": " + diagnostic.message;
    return message;
}

std::string normalized_path(const std::filesystem::path& path)
{
    auto text = path.lexically_normal().generic_string();
    while (text.starts_with("./"))
        text.erase(0, 2);
    return text;
}

bool path_matches(std::filesystem::path dependency, std::string_view authored)
{
    auto dependency_text = normalized_path(dependency);
    auto authored_text = normalized_path(std::filesystem::path(authored));
    if (authored_text.starts_with("assets/")) authored_text.erase(0, 7);
    if (dependency_text == authored_text) return true;
    return dependency_text.size() > authored_text.size() && dependency_text.ends_with(authored_text) &&
           dependency_text[dependency_text.size() - authored_text.size() - 1] == '/';
}

core::result<std::string, std::string> read_text_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return core::result<std::string, std::string>::failure("source could not be opened: " + path.generic_string());
    std::ostringstream source;
    source << input.rdbuf();
    if (!input.eof() && input.fail())
        return core::result<std::string, std::string>::failure("source could not be read: " + path.generic_string());
    return core::result<std::string, std::string>::success(std::move(source).str());
}

const assets::asset_snapshot* find_material_shader_dependency(const assets::asset_cook_context& context,
                                                              std::string_view shader_path)
{
    const assets::asset_snapshot* fallback{};
    for (const auto& dependency : context.dependencies)
    {
        if (dependency.type != assets::asset_types::shader) continue;
        if (!fallback) fallback = &dependency;
        if (path_matches(dependency.source_path, shader_path)) return &dependency;
    }
    return fallback;
}

core::result<std::string, std::string> read_material_shader(const assets::asset_cook_context& context,
                                                            std::string_view shader_path)
{
    const auto* dependency = find_material_shader_dependency(context, shader_path);
    if (!dependency)
        return core::result<std::string, std::string>::failure("Material Shader '" + std::string(shader_path) +
                                                               "' is not a registered shader dependency");
    return read_text_file(dependency->source_path);
}

std::vector<std::string> nested_function_paths(std::string_view source)
{
    const auto document = json::parse(source, nullptr, false);
    if (document.is_discarded()) return {};

    std::vector<std::string> paths;
    const auto visit = [&](const auto& self, const json& value) -> void
    {
        if (value.is_object())
        {
            if (value.value("type", "") == "functionCall")
            {
                const auto values = value.value("values", json::object());
                const auto path = values.value("path", "");
                if (!path.empty()) paths.push_back(path);
            }
            for (const auto& [key, child] : value.items())
            {
                static_cast<void>(key);
                self(self, child);
            }
        }
        else if (value.is_array())
        {
            for (const auto& child : value)
                self(self, child);
        }
    };
    visit(visit, document);
    std::ranges::sort(paths);
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    return paths;
}

std::filesystem::path assets_root_for(const std::filesystem::path& source_path)
{
    auto current = source_path.parent_path();
    while (!current.empty())
    {
        if (current.filename() == "assets") return current;
        const auto parent = current.parent_path();
        if (parent == current) break;
        current = parent;
    }
    return {};
}

std::filesystem::path resolve_function_path(const std::filesystem::path& owner, std::string_view authored)
{
    const std::filesystem::path authored_path(authored);
    if (authored_path.is_absolute()) return authored_path.lexically_normal();
    const auto text = normalized_path(authored_path);
    if (text.starts_with("assets/"))
    {
        const auto root = assets_root_for(owner);
        if (!root.empty()) return (root.parent_path() / authored_path).lexically_normal();
    }
    return (owner.parent_path() / authored_path).lexically_normal();
}

using function_source_result = core::result<std::vector<render::tools::material_function_source>, std::string>;

function_source_result material_function_sources(const assets::asset_cook_context& context)
{
    std::map<std::string, std::pair<std::filesystem::path, std::string>> pending;
    for (const auto& dependency : context.dependencies)
    {
        if (dependency.type != assets::asset_types::material) continue;
        auto source = read_text_file(dependency.source_path);
        if (!source) return function_source_result::failure(source.error());
        if (!render::tools::is_material_function_json(source.value())) continue;
        pending.emplace(normalized_path(dependency.source_path),
                        std::pair{dependency.source_path, std::move(source).value()});
    }

    std::map<std::string, render::tools::material_function_source> functions;
    while (!pending.empty())
    {
        auto entry = pending.extract(pending.begin());
        auto path = std::move(entry.mapped().first);
        auto source = std::move(entry.mapped().second);
        const auto key = normalized_path(path);
        if (functions.contains(key)) continue;
        functions.emplace(key, render::tools::material_function_source{.path = key, .source = source});

        for (const auto& nested : nested_function_paths(source))
        {
            const auto nested_path = resolve_function_path(path, nested);
            const auto nested_key = normalized_path(nested_path);
            if (functions.contains(nested_key) || pending.contains(nested_key)) continue;
            auto nested_source = read_text_file(nested_path);
            if (!nested_source)
            {
                std::string message = "Material Function '";
                message.append(nested);
                message.append("' referenced by '");
                message.append(key);
                message.append("' could not be loaded: ");
                message.append(nested_source.error());
                return function_source_result::failure(std::move(message));
            }
            if (!render::tools::is_material_function_json(nested_source.value()))
                return function_source_result::failure("Material Function dependency is not a function document: " +
                                                       nested_key);
            pending.emplace(nested_key, std::pair{nested_path, std::move(nested_source).value()});
        }
    }

    std::vector<render::tools::material_function_source> result;
    result.reserve(functions.size());
    for (auto& [path, source] : functions)
    {
        static_cast<void>(path);
        result.push_back(std::move(source));
    }
    return function_source_result::success(std::move(result));
}

bool same_parameter_layout(const std::vector<render::shader_parameter_descriptor>& lhs,
                           const std::vector<render::shader_parameter_descriptor>& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t index = 0; index < lhs.size(); ++index)
        if (lhs[index].id != rhs[index].id || lhs[index].type != rhs[index].type ||
            lhs[index].offset != rhs[index].offset || lhs[index].size != rhs[index].size)
            return false;
    return true;
}

float json_float(const json& object, std::string_view key, float fallback)
{
    const auto found = object.find(std::string(key));
    return found != object.end() && found->is_number() ? found->get<float>() : fallback;
}

math::vector3f json_vec3(const json& object, std::string_view key, const math::vector3f& fallback)
{
    const auto found = object.find(std::string(key));
    if (found == object.end() || !found->is_object()) return fallback;
    return {json_float(*found, "r", fallback[0]), json_float(*found, "g", fallback[1]),
            json_float(*found, "b", fallback[2])};
}

std::string slang_number(float value)
{
    std::ostringstream output;
    output << std::setprecision(9) << value;
    auto text = std::move(output).str();
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}

std::string slang_vec3(const math::vector3f& value)
{
    return "float3(" + slang_number(value[0]) + ',' + slang_number(value[1]) + ',' + slang_number(value[2]) + ')';
}

render::material_descriptor authored_pass_material(const render::tools::material_authoring_document& authored,
                                                   const json& document)
{
    render::material_descriptor material{.domain = authored.domain,
                                         .shading_model = authored.shading_model,
                                         .alpha_mode = authored.alpha_mode,
                                         .double_sided = authored.double_sided};
    const auto advanced = document.value("advanced", json::object());
    if (!advanced.is_object()) return material;
    material.clear_coat_factor = json_float(advanced, "clearCoat", 0.0f);
    material.clear_coat_roughness = json_float(advanced, "clearCoatRoughness", 0.0f);
    material.sheen_factor = json_float(advanced, "sheen", 0.0f);
    material.transmission_factor = json_float(advanced, "transmission", 0.0f);
    material.index_of_refraction = json_float(advanced, "indexOfRefraction", 1.5f);
    material.thickness_factor = json_float(advanced, "thickness", 0.0f);
    material.attenuation_color = json_vec3(advanced, "attenuationColor", math::vector3f::one);
    material.attenuation_distance = json_float(advanced, "attenuationDistance", 1.0f);
    material.subsurface_factor = json_float(advanced, "subsurface", 0.0f);
    material.subsurface_color = json_vec3(advanced, "subsurfaceColor", {1.0f, 0.35f, 0.2f});
    material.anisotropy_factor = json_float(advanced, "anisotropy", 0.0f);
    material.anisotropy_rotation = json_float(advanced, "anisotropyRotation", 0.0f);
    material.parallax_height_scale = json_float(advanced, "parallaxHeightScale", 0.0f);
    if (material.parallax_height_scale != 0.0f) material.displacement_mode = render::material_displacement_mode::parallax;
    return material;
}

bool apply_authored_advanced_properties(render::tools::material_evaluator_source& evaluator, const json& document)
{
    const auto advanced = document.value("advanced", json::object());
    if (!advanced.is_object()) return true;

    const auto evaluator_begin = evaluator.source.find("ArcSurfaceData arc_evaluate_material");
    if (evaluator_begin == std::string::npos) return false;
    const auto return_position = evaluator.source.find("    return surface;\n", evaluator_begin);
    if (return_position == std::string::npos) return false;

    std::string assignments;
    const auto append_scalar = [&](std::string_view field, std::string_view key, float fallback)
    {
        assignments += "    surface.";
        assignments += field;
        assignments += " = ";
        assignments += slang_number(json_float(advanced, key, fallback));
        assignments += ";\n";
    };
    const auto append_vec3 = [&](std::string_view field, std::string_view key, const math::vector3f& fallback)
    {
        assignments += "    surface.";
        assignments += field;
        assignments += " = ";
        assignments += slang_vec3(json_vec3(advanced, key, fallback));
        assignments += ";\n";
    };

    append_scalar("clearCoat", "clearCoat", 0.0f);
    append_scalar("clearCoatRoughness", "clearCoatRoughness", 0.1f);
    append_scalar("sheen", "sheen", 0.0f);
    append_vec3("sheenColor", "sheenColor", {});
    append_scalar("sheenRoughness", "sheenRoughness", 0.5f);
    append_scalar("anisotropy", "anisotropy", 0.0f);
    append_scalar("anisotropyRotation", "anisotropyRotation", 0.0f);
    append_scalar("transmission", "transmission", 0.0f);
    append_scalar("indexOfRefraction", "indexOfRefraction", 1.5f);
    append_scalar("thickness", "thickness", 0.0f);
    append_vec3("attenuationColor", "attenuationColor", math::vector3f::one);
    append_scalar("attenuationDistance", "attenuationDistance", 1.0f);
    append_vec3("subsurfaceColor", "subsurfaceColor", {1.0f, 0.35f, 0.2f});
    append_scalar("subsurface", "subsurface", 0.0f);
    evaluator.source.insert(return_position, assignments);
    return true;
}

class material_processor final : public assets::asset_cook_processor
{
public:
    material_processor()
    {
        descriptor_.id = assets::cook_processor_ids::material;
        descriptor_.name = "ARC Material";
        descriptor_.schema = assets::artifact_schemas::material;
        descriptor_.version = 8;
        descriptor_.schema_version = render::tools::material_package_version;
        descriptor_.input_types = {assets::asset_types::material};
    }

    const assets::asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }

    std::string toolchain_fingerprint() const override
    {
        return "arc.material-cooker/8;arc-material-package/3;arc-material-authoring/4;arc-material-ir/1;"
               "arc-material-codegen/2;arc-material-function/1;arc-material-pass-contract/1;"
               "arc-material-pass-codegen/2;arc-custom-material-shader/1;arc-authored-advanced/1;" +
               std::string(compiler_.fingerprint());
    }

    assets::asset_cook_result cook(const assets::asset_cook_context& context) override
    {
        const std::string source(reinterpret_cast<const char*>(context.source.bytes.data()),
                                 context.source.bytes.size());

        if (render::tools::is_material_function_json(source))
        {
            auto validated =
                render::tools::validate_material_function_json(source, context.source.source_path.generic_string());
            if (!validated)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = validated.error().message}};
            return {.artifacts = {{.name = context.source.source_path.stem().string(),
                                   .extension = ".arcmatfnc",
                                   .schema = descriptor_.schema,
                                   .schema_version = render::tools::material_function_version,
                                   .bytes = context.source.bytes}},
                    .diagnostics = {{.severity = assets::asset_diagnostic_severity::information,
                                     .guid = context.asset.guid,
                                     .category = "material.function",
                                     .message = "Validated Material/Shader Function v" +
                                                std::to_string(render::tools::material_function_version)}}};
        }

        auto authored = render::tools::parse_material_authoring_json(source);
        if (!authored)
            return {.error = {.code = assets::asset_error_code::import_failed,
                              .guid = context.asset.guid,
                              .path = context.source.source_path,
                              .message = authored.error().message}};

        const auto document = json::parse(authored.value().canonical_json, nullptr, false);
        if (!document.is_object())
            return {.error = {.code = assets::asset_error_code::import_failed,
                              .guid = context.asset.guid,
                              .path = context.source.source_path,
                              .message = "Canonical material document is invalid"}};

        std::vector<assets::cooked_artifact> artifacts;
        std::vector<assets::asset_diagnostic> diagnostics;
        render::material_compiled_program program;
        std::vector<render::shader_parameter_descriptor> parameters;
        render::tools::material_evaluator_source evaluator;
        bool handwritten{};
        std::filesystem::path custom_shader_path;

        if (!authored.value().graph_json.empty())
        {
            auto functions = material_function_sources(context);
            if (!functions)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = functions.error()}};
            auto compiled_graph =
                render::tools::compile_material_graph_json(authored.value().graph_json, functions.value());
            if (!compiled_graph)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = compiled_graph.error().message}};

            auto generated_evaluator = render::tools::make_graph_material_evaluator(compiled_graph.value());
            if (!generated_evaluator)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = generated_evaluator.error().message}};
            evaluator = std::move(generated_evaluator).value();
            if (!apply_authored_advanced_properties(evaluator, document))
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = "Generated Material ABI evaluator could not receive authored advanced properties"}};
            parameters = evaluator.parameters;
            std::uint32_t parameter_block_size{};
            for (auto& parameter : parameters)
            {
                parameter.offset = parameter_block_size;
                parameter_block_size += (parameter.size + 15u) & ~15u;
            }
            evaluator.parameters = parameters;
        }
        else
        {
            auto custom_source = read_material_shader(context, authored.value().shader_path);
            if (!custom_source)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = custom_source.error()}};
            const auto* dependency = find_material_shader_dependency(context, authored.value().shader_path);
            custom_shader_path =
                dependency ? dependency->source_path : std::filesystem::path(authored.value().shader_path);
            auto custom_evaluator = render::tools::make_custom_material_evaluator(custom_source.value(),
                                                                                  custom_shader_path.generic_string());
            if (!custom_evaluator)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = custom_shader_path,
                                  .message = custom_evaluator.error().message}};
            evaluator = std::move(custom_evaluator).value();
            handwritten = true;
            diagnostics.push_back({.severity = assets::asset_diagnostic_severity::information,
                                   .guid = context.asset.guid,
                                   .category = "material.shader",
                                   .message = "Compiled handwritten Material Shader '" + authored.value().shader_path +
                                              "' through Material ABI v" +
                                              std::to_string(render::material_abi_version)});
        }

        const auto pass_material = authored_pass_material(authored.value(), document);
        program.package = {.high = context.asset.guid.high, .low = context.asset.guid.low};

        for (const auto pass : material_passes)
        {
            if (!render::material_supports_pass(pass_material, pass)) continue;

            auto generated = render::tools::generate_material_pass_slang(evaluator, pass_material, pass);
            if (!generated)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = generated.error().message}};

            const std::string pass_label{pass_name(pass)};
            render::shader_compile_request request{
                .source_path = handwritten
                                   ? custom_shader_path.generic_string()
                                   : context.source.source_path.string() + "." + pass_label + ".generated.slang",
                .source_override = generated.value().source,
                .entry_point = generated.value().entry_point,
                .profile = "spirv_1_5",
                .library_version = handwritten ? "arc-custom-material/1" : "arc-material-pass/2",
                .domain = render::shader_domain::surface,
                .stage = render::shader_stage::fragment,
                .target = render::shader_target::spirv,
                .optimization = context.target.configuration == assets::cook_configuration::shipping
                                    ? render::shader_optimization::performance
                                    : render::shader_optimization::development,
                .required_passes = {pass},
                .generated_line_nodes = generated.value().generated_line_nodes,
                .generate_debug_information = context.target.configuration != assets::cook_configuration::shipping};
            if (handwritten && !custom_shader_path.empty()) request.include_directories.push_back(custom_shader_path.parent_path());

            auto compiled = cache_.compile_or_get(compiler_, request);
            if (!compiled)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = handwritten ? custom_shader_path : context.source.source_path,
                                  .message = compile_error_message(compiled.error())}};

            if (handwritten)
            {
                if (parameters.empty())
                    parameters = compiled.value().reflection.parameters;
                else if (!same_parameter_layout(parameters, compiled.value().reflection.parameters))
                    return {.error = {.code = assets::asset_error_code::import_failed,
                                      .guid = context.asset.guid,
                                      .path = custom_shader_path,
                                      .message = "Material Shader parameter layout differs between render passes"}};
            }
            else
            {
                for (const auto& [line, node] : request.generated_line_nodes)
                    compiled.value().source_map.push_back(
                        {.generated_line = line,
                         .source = {.path = context.source.source_path.generic_string(),
                                    .line = line,
                                    .graph_node_id = node}});
                std::ranges::sort(compiled.value().source_map, {}, &render::shader_source_map_entry::generated_line);
                compiled.value().reflection.parameters = parameters;
                std::uint32_t parameter_block_size{};
                for (const auto& parameter : parameters)
                    parameter_block_size = std::max(parameter_block_size, parameter.offset + parameter.size);
                compiled.value().reflection.parameter_block_size = parameter_block_size;
            }

            const auto entry_point = render::make_shader_entry_point_id(request.entry_point, render::shader_stage::fragment);
            program.passes.push_back({.pass = pass,
                                      .permutation = generated.value().permutation,
                                      .entry_point = entry_point,
                                      .build_hash = compiled.value().build_hash});

            render::shader_package package{.id = program.package,
                                           .generation = {std::max<std::uint64_t>(context.asset.generation, 1)},
                                           .target = render::shader_target::spirv,
                                           .permutation = generated.value().permutation,
                                           .compiled = std::move(compiled).value()};
            auto bytes = render::serialize_shader_package(package);
            if (!bytes)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = bytes.error().message}};
            artifacts.push_back({.name = context.source.source_path.stem().string() + "." + pass_label,
                                 .extension = ".arcshader",
                                 .schema = assets::artifact_schemas::shader,
                                 .schema_version = render::shader_package::current_version,
                                 .bytes = std::move(bytes).value()});
        }

        auto material_bytes = render::tools::serialize_material_package_v3({.compiled = std::move(program),
                                                                            .parameters = std::move(parameters),
                                                                            .canonical_document_json = authored.value().canonical_json});
        artifacts.push_back({.name = context.source.source_path.stem().string(),
                             .extension = ".arcmatc",
                             .schema = descriptor_.schema,
                             .schema_version = descriptor_.schema_version,
                             .bytes = std::move(material_bytes)});
        return {.artifacts = std::move(artifacts), .diagnostics = std::move(diagnostics)};
    }

private:
    assets::asset_cook_processor_descriptor descriptor_;
    render::tools::slang_shader_compiler compiler_;
    render::shader_library_cache cache_;
};

} // namespace

std::unique_ptr<assets::asset_cook_processor> make_material_processor()
{
    return std::make_unique<material_processor>();
}

} // namespace arc::tools
