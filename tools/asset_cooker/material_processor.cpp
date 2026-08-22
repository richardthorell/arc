#include "material_processor.h"

#include <arc/render_tools/render_tools.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace arc::tools
{
namespace
{

class material_processor final : public assets::asset_cook_processor
{
public:
    material_processor()
    {
        descriptor_.id = assets::cook_processor_ids::material;
        descriptor_.name = "ARC Material";
        descriptor_.schema = assets::artifact_schemas::material;
        descriptor_.version = 3;
        descriptor_.schema_version = render::tools::material_package_version;
        descriptor_.input_types = {assets::asset_types::material};
    }

    const assets::asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }

    std::string toolchain_fingerprint() const override
    {
        return "arc.material-cooker/3;arc-material-package/2;arc-material-authoring/4;arc-material-ir/1;"
               "arc-material-graph/1;" +
               std::string(compiler_.fingerprint());
    }

    assets::asset_cook_result cook(const assets::asset_cook_context& context) override
    {
        const std::string source(reinterpret_cast<const char*>(context.source.bytes.data()),
                                 context.source.bytes.size());
        auto authored = render::tools::parse_material_authoring_json(source);
        if (!authored)
            return {.error = {.code = assets::asset_error_code::import_failed,
                              .guid = context.asset.guid,
                              .path = context.source.source_path,
                              .message = authored.error().message}};

        std::vector<assets::cooked_artifact> artifacts;
        std::vector<assets::asset_diagnostic> diagnostics;
        if (authored.value().migrated)
            diagnostics.push_back(
                {.severity = assets::asset_diagnostic_severity::information,
                 .guid = context.asset.guid,
                 .category = "material.schema",
                 .message = "Migrated authored material schema v" + std::to_string(authored.value().source_version) +
                            " to v" + std::to_string(render::tools::material_authoring_version) + " during cook"});

        render::shader_package_id package_id{};
        render::shader_permutation_id permutation{};
        std::vector<render::shader_parameter_descriptor> parameters;
        if (!authored.value().graph_json.empty())
        {
            auto compiled_graph = render::tools::compile_material_graph_json(authored.value().graph_json);
            if (!compiled_graph)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = compiled_graph.error().message}};

            auto lowered = render::tools::lower_material_graph_json(authored.value().graph_json);
            if (!lowered)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = lowered.error().message}};

            render::shader_compile_request request{
                .source_path = context.source.source_path.string() + ".generated.slang",
                .source_override = lowered.value().source,
                .entry_point = "main",
                .profile = "spirv_1_5",
                .domain = render::shader_domain::surface,
                .stage = render::shader_stage::fragment,
                .target = render::shader_target::spirv,
                .optimization = context.target.configuration == assets::cook_configuration::shipping
                                    ? render::shader_optimization::performance
                                    : render::shader_optimization::development,
                .required_passes = {render::material_pass::depth, render::material_pass::shadow,
                                    render::material_pass::gbuffer, render::material_pass::forward,
                                    render::material_pass::motion, render::material_pass::object_id,
                                    render::material_pass::selection},
                .generated_line_nodes = lowered.value().generated_line_nodes,
                .generate_debug_information = context.target.configuration != assets::cook_configuration::shipping};
            auto compiled = cache_.compile_or_get(compiler_, request);
            if (!compiled)
            {
                std::string message = compiled.error().message;
                for (const auto& diagnostic : compiled.error().diagnostics)
                    message += "\n" + diagnostic.location.path + ":" + std::to_string(diagnostic.location.line) + ":" +
                               std::to_string(diagnostic.location.column) + ": " + diagnostic.message;
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = std::move(message)}};
            }

            parameters = compiled_graph.value().descriptor.parameters;
            std::uint32_t offset{};
            for (auto& parameter : parameters)
            {
                parameter.offset = offset;
                offset += (parameter.size + 15u) & ~15u;
            }
            for (const auto& [line, node] : request.generated_line_nodes)
                compiled.value().source_map.push_back({.generated_line = line,
                                                       .source = {.path = context.source.source_path.generic_string(),
                                                                  .line = line,
                                                                  .graph_node_id = node}});
            std::ranges::sort(compiled.value().source_map, {}, &render::shader_source_map_entry::generated_line);
            compiled.value().reflection.parameters = parameters;
            compiled.value().reflection.parameter_block_size = offset;

            package_id = {.high = context.asset.guid.high, .low = context.asset.guid.low};
            permutation = {1};
            render::shader_package package{.id = package_id,
                                           .generation = {std::max<std::uint64_t>(context.asset.generation, 1)},
                                           .target = render::shader_target::spirv,
                                           .permutation = permutation,
                                           .compiled = std::move(compiled).value()};
            auto bytes = render::serialize_shader_package(package);
            if (!bytes)
                return {.error = {.code = assets::asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = bytes.error().message}};
            artifacts.push_back({.name = context.source.source_path.stem().string(),
                                 .extension = ".arcshader",
                                 .schema = assets::artifact_schemas::shader,
                                 .schema_version = render::shader_package::current_version,
                                 .bytes = std::move(bytes).value()});
        }

        auto material_bytes =
            render::tools::serialize_material_package_v2({.shader_package = package_id,
                                                          .permutation = permutation,
                                                          .parameters = parameters,
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
