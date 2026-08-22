#include <arc/assets/cook.h>
#include <arc/framework/service.h>
#include <arc/io/io.h>
#include <arc/memory/memory.h>
#include <arc/persistence/persistence.h>
#include <arc/project/project.h>
#include <arc/render/mesh.h>
#include <arc/render/lighting_scene.h>
#include <arc/render/texture.h>
#include <arc/render/virtual_mesh.h>
#include <arc/scene/persistence.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <type_traits>

namespace
{

using namespace arc;
using namespace arc::assets;

std::optional<std::string> environment_value(const char* name)
{
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return std::nullopt;

    std::string result(value);
    std::free(value);
    return result;
#else
    // Cooker configuration is read during single-threaded process startup.
    if (const char* value = std::getenv(name)) // NOLINT(concurrency-mt-unsafe)
        return std::string(value);
    return std::nullopt;
#endif
}

template <class T> void append_value(std::vector<std::byte>& output, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(T));
}

void append_string(std::vector<std::byte>& output, std::string_view value)
{
    append_value(output, static_cast<std::uint64_t>(value.size()));
    output.insert(output.end(), reinterpret_cast<const std::byte*>(value.data()),
                  reinterpret_cast<const std::byte*>(value.data() + value.size()));
}

void append_bytes(std::vector<std::byte>& output, std::span<const std::byte> value)
{
    append_value(output, static_cast<std::uint64_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
}

void append_vector3(std::vector<std::byte>& output, const math::vector3f& value)
{
    append_value(output, value[0]);
    append_value(output, value[1]);
    append_value(output, value[2]);
}

void append_virtual_cluster(std::vector<std::byte>& output, const render::virtual_mesh_cluster& cluster)
{
    append_value(output, cluster.first_index);
    append_value(output, cluster.index_count);
    append_value(output, cluster.first_triangle);
    append_value(output, cluster.triangle_count);
    append_value(output, cluster.first_vertex);
    append_value(output, cluster.vertex_count);
    append_value(output, static_cast<std::uint64_t>(cluster.material_index));
    append_vector3(output, cluster.bounds_min);
    append_vector3(output, cluster.bounds_max);
    append_vector3(output, cluster.sphere_center);
    append_value(output, cluster.sphere_radius);
    append_vector3(output, cluster.cone_axis);
    append_value(output, cluster.cone_cutoff);
    append_value(output, cluster.geometric_error);
    append_value(output, cluster.hierarchy_node);
    append_value(output, cluster.page_index);
    append_value(output, cluster.hierarchy_level);
    append_value(output, cluster.flags);
}

void append_virtual_node(std::vector<std::byte>& output, const render::virtual_mesh_lod_node& node)
{
    append_value(output, node.first_cluster);
    append_value(output, node.cluster_count);
    append_value(output, node.first_child);
    append_value(output, node.child_count);
    append_value(output, node.parent);
    append_value(output, node.page_index);
    append_value(output, node.error);
    append_vector3(output, node.bounds_min);
    append_vector3(output, node.bounds_max);
    append_vector3(output, node.sphere_center);
    append_value(output, node.sphere_radius);
    append_vector3(output, node.cone_axis);
    append_value(output, node.cone_cutoff);
    append_value(output, node.level);
    append_value(output, node.flags);
}

class document_processor final : public asset_cook_processor
{
public:
    document_processor(asset_type_id type, cook_processor_id id, artifact_schema_id schema, std::string name,
                       std::string extension, std::optional<persistence::document_kind> document_kind = std::nullopt)
        : extension_(std::move(extension)), document_kind_(document_kind)
    {
        descriptor_.id = id;
        descriptor_.name = std::move(name);
        descriptor_.schema = schema;
        descriptor_.input_types.push_back(type);
        if (document_kind_)
        {
            descriptor_.version = 2;
            descriptor_.schema_version = 2;
        }
    }

    const asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }
    std::string toolchain_fingerprint() const override
    {
        return "arc.document-cooker/2";
    }

    asset_cook_result cook(const asset_cook_context& context) override
    {
        const auto document = nlohmann::json::parse(
            reinterpret_cast<const char*>(context.source.bytes.data()),
            reinterpret_cast<const char*>(context.source.bytes.data() + context.source.bytes.size()), nullptr, false);
        if (document.is_discarded())
            return {.error = {.code = asset_error_code::import_failed,
                              .guid = context.asset.guid,
                              .path = context.source.source_path,
                              .message = "Authored JSON document is invalid"}};
        if (document_kind_)
        {
            persistence::component_persistence_registry components;
            persistence::schema_migration_registry migrations;
            auto migration_status = scene::register_persistence_migrations(migrations);
            if (!scene::register_persistence_components(components) || !migration_status)
                return {.error = {.code = asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = "Persistence registry initialization failed: " +
                                             (migration_status ? std::string("component registration failed")
                                                               : migration_status.error().message)}};
            const std::string_view source(reinterpret_cast<const char*>(context.source.bytes.data()),
                                          context.source.bytes.size());
            auto archive = persistence::read_reflected_json(source, components, &migrations);
            if (!archive.succeeded() || archive.document.kind != *document_kind_)
                return {.error = {.code = asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = archive.error.empty() ? "Document kind does not match its asset type"
                                                                   : archive.error}};
            for (const auto& dependency : archive.document.dependencies)
            {
                if (!dependency.required) continue;
                const auto found = std::find_if(context.dependencies.begin(), context.dependencies.end(),
                                                [&](const asset_snapshot& candidate)
                                                { return candidate.guid == dependency.reference.guid; });
                if (!dependency.reference.guid.valid() || found == context.dependencies.end() ||
                    (dependency.reference.expected_type.valid() && found->type != dependency.reference.expected_type))
                {
                    return {.error = {.code = asset_error_code::dependency_failed,
                                      .guid = context.asset.guid,
                                      .path = context.source.source_path,
                                      .message = "Required document dependency is unresolved or has the wrong type: " +
                                                 dependency.reference.path_hint}};
                }
            }
            auto binary = persistence::write_tagged_binary(archive.document, canonical_cook_target(context.target));
            if (!binary)
                return {.error = {.code = asset_error_code::import_failed,
                                  .guid = context.asset.guid,
                                  .path = context.source.source_path,
                                  .message = binary.error().message}};
            return {.artifacts = {{.name = context.source.source_path.stem().string(),
                                   .extension = extension_,
                                   .schema = descriptor_.schema,
                                   .schema_version = descriptor_.schema_version,
                                   .bytes = std::move(binary).value()}}};
        }
        const auto canonical = document.dump();
        std::vector<std::byte> bytes;
        append_string(bytes, "ARC_DOCUMENT_1");
        append_string(bytes, canonical);
        return {.artifacts = {{.name = context.source.source_path.stem().string(),
                               .extension = extension_,
                               .schema = descriptor_.schema,
                               .schema_version = descriptor_.schema_version,
                               .bytes = std::move(bytes)}}};
    }

private:
    asset_cook_processor_descriptor descriptor_;
    std::string extension_;
    std::optional<persistence::document_kind> document_kind_;
};

class source_processor final : public asset_cook_processor
{
public:
    source_processor()
    {
        descriptor_.id = cook_processor_ids::source;
        descriptor_.name = "ARC Binary";
        descriptor_.schema = artifact_schemas::source;
        descriptor_.input_types = {asset_types::binary_blob};
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }
    std::string toolchain_fingerprint() const override
    {
        return "arc.source-cooker/1";
    }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        return {.artifacts = {{.name = context.source.source_path.stem().string(),
                               .extension = ".arcbin",
                               .schema = descriptor_.schema,
                               .schema_version = descriptor_.schema_version,
                               .bytes = context.source.bytes}}};
    }

private:
    asset_cook_processor_descriptor descriptor_;
};

class texture_processor final : public asset_cook_processor
{
public:
    texture_processor()
    {
        descriptor_.id = cook_processor_ids::texture;
        descriptor_.name = "ARC Texture";
        descriptor_.schema = artifact_schemas::texture;
        descriptor_.input_types = {asset_types::texture_2d, asset_types::environment};
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }
    std::string toolchain_fingerprint() const override
    {
        return "arc.texture-cooker/1;stb;basis-contract-1";
    }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        auto loaded = render::load_texture_asset_bytes(context.source.bytes, context.source.source_path);
        if (!loaded.succeeded())
            return {.error = {.code = asset_error_code::import_failed,
                              .guid = context.asset.guid,
                              .path = context.source.source_path,
                              .message = loaded.message.empty() ? "Texture decode failed" : loaded.message}};
        auto& texture = loaded.texture;
        std::vector<std::byte> bytes;
        append_string(bytes, "ARC_TEXTURE_1");
        append_value(bytes, texture.width);
        append_value(bytes, texture.height);
        append_value(bytes, texture.depth);
        append_value(bytes, texture.array_layers);
        append_value(bytes, texture.mip_levels);
        append_value(bytes, texture.dimension);
        append_value(bytes, texture.format);
        append_value(bytes, texture.color_space);
        append_value(bytes, texture.semantic);
        append_value(bytes, static_cast<std::uint32_t>(texture.mips.size()));
        for (const auto& mip : texture.mips)
        {
            append_value(bytes, mip.width);
            append_value(bytes, mip.height);
            append_value(bytes, static_cast<std::uint64_t>(mip.offset));
            append_value(bytes, static_cast<std::uint64_t>(mip.size));
        }
        const auto& payload = texture.has_encoded_mips() ? texture.encoded : texture.pixels;
        append_bytes(bytes, payload);
        std::vector<asset_diagnostic> diagnostics;
        if (context.target.textures == cook_texture_family::bc && !texture.compressed)
            diagnostics.push_back(
                {.severity = asset_diagnostic_severity::information,
                 .guid = context.asset.guid,
                 .category = "cook.texture",
                 .message =
                     "Texture mip chain is ready for the registered BC encoder; stored uncompressed by this build"});
        return {.artifacts = {{.name = context.source.source_path.stem().string(),
                               .extension = ".arctex",
                               .schema = descriptor_.schema,
                               .schema_version = descriptor_.schema_version,
                               .gpu_compressed = texture.compressed,
                               .bytes = std::move(bytes)}},
                .diagnostics = std::move(diagnostics)};
    }

private:
    asset_cook_processor_descriptor descriptor_;
};

class mesh_processor final : public asset_cook_processor
{
public:
    mesh_processor()
    {
        descriptor_.id = cook_processor_ids::mesh;
        descriptor_.name = "ARC Mesh";
        descriptor_.schema = artifact_schemas::mesh;
        descriptor_.version = 3;
        descriptor_.schema_version = 3;
        descriptor_.input_types = {asset_types::imported_scene, asset_types::static_mesh};
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }
    std::string toolchain_fingerprint() const override
    {
        return "arc.mesh-cooker/3;meshoptimizer-1.2;arc-virtual-geometry-2;arc-lighting-geometry-1";
    }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        const auto loaded = render::load_scene_asset(context.source.source_path);
        if (!loaded.succeeded())
            return {.error = {.code = asset_error_code::import_failed,
                              .guid = context.asset.guid,
                              .path = context.source.source_path,
                              .message = loaded.message.empty() ? "Mesh import failed" : loaded.message}};
        std::vector<std::byte> conventional_bytes;
        std::vector<std::byte> virtual_bytes;
        std::vector<std::byte> card_bytes;
        std::vector<std::byte> distance_field_bytes;
        append_string(conventional_bytes, "ARC_MESH_2");
        append_string(virtual_bytes, "ARC_VIRTUAL_GEOMETRY_2");
        append_string(card_bytes, "ARC_SURFACE_CARDS_1");
        append_string(distance_field_bytes, "ARC_MESH_DISTANCE_FIELD_1");
        append_value(conventional_bytes, static_cast<std::uint32_t>(loaded.meshes.size()));
        append_value(virtual_bytes, static_cast<std::uint32_t>(loaded.meshes.size()));
        append_value(card_bytes, static_cast<std::uint32_t>(loaded.meshes.size()));
        append_value(distance_field_bytes, static_cast<std::uint32_t>(loaded.meshes.size()));
        for (const auto& mesh : loaded.meshes)
        {
            const auto geometry = render::build_virtual_mesh(mesh);
            const auto lighting = render::build_lighting_geometry(mesh);

            append_string(conventional_bytes, mesh.name);
            append_value(conventional_bytes, static_cast<std::uint64_t>(mesh.material_index));
            append_value(conventional_bytes, static_cast<std::uint32_t>(geometry.conventional_lods.size()));
            for (const auto& lod : geometry.conventional_lods)
            {
                append_value(conventional_bytes, lod.ratio);
                append_value(conventional_bytes, lod.geometric_error);
                append_bytes(conventional_bytes, std::as_bytes(std::span(lod.vertices)));
                append_bytes(conventional_bytes, std::as_bytes(std::span(lod.indices)));
            }

            append_string(virtual_bytes, mesh.name);
            append_value(virtual_bytes, static_cast<std::uint64_t>(mesh.material_index));
            append_value(virtual_bytes, static_cast<std::uint32_t>(geometry.clusters.size()));
            for (const auto& cluster : geometry.clusters)
                append_virtual_cluster(virtual_bytes, cluster);
            append_value(virtual_bytes, static_cast<std::uint32_t>(geometry.lod_nodes.size()));
            for (const auto& node : geometry.lod_nodes)
                append_virtual_node(virtual_bytes, node);
            append_bytes(virtual_bytes, std::as_bytes(std::span(geometry.hierarchy_children)));
            append_bytes(virtual_bytes, std::as_bytes(std::span(geometry.root_nodes)));
            append_value(virtual_bytes, static_cast<std::uint32_t>(geometry.pages.size()));
            for (const auto& page : geometry.pages)
            {
                append_value(virtual_bytes, page.first_cluster);
                append_value(virtual_bytes, page.cluster_count);
                append_value(virtual_bytes, page.uncompressed_offset);
                append_value(virtual_bytes, page.uncompressed_size);
                append_value(virtual_bytes, page.compressed_offset);
                append_value(virtual_bytes, page.compressed_size);
                append_value(virtual_bytes, page.content_hash);
                append_value(virtual_bytes, static_cast<std::uint8_t>(page.root));
            }
            append_bytes(virtual_bytes, geometry.page_payload);

            append_string(card_bytes, mesh.name);
            append_value(card_bytes, static_cast<std::uint64_t>(mesh.material_index));
            append_value(card_bytes, lighting.geometry.generation);
            append_value(card_bytes, static_cast<std::uint32_t>(lighting.geometry.cards.size()));
            for (const auto& card : lighting.geometry.cards)
            {
                append_vector3(card_bytes, card.center);
                append_vector3(card_bytes, card.normal);
                append_vector3(card_bytes, card.tangent);
                append_value(card_bytes, card.extent[0]);
                append_value(card_bytes, card.extent[1]);
                append_value(card_bytes, card.depth_extent);
                append_value(card_bytes, card.texel_density);
                append_value(card_bytes, card.geometric_error);
                append_value(card_bytes, card.material_section);
                append_value(card_bytes, card.fallback_card);
            }

            const auto& field = lighting.geometry.distance_field;
            append_string(distance_field_bytes, mesh.name);
            append_vector3(distance_field_bytes, field.bounds.min.as_vector());
            append_vector3(distance_field_bytes, field.bounds.max.as_vector());
            for (const auto dimension : field.dimensions)
                append_value(distance_field_bytes, dimension);
            append_vector3(distance_field_bytes, field.voxel_size);
            append_value(distance_field_bytes, field.distance_scale);
            append_value(distance_field_bytes, field.mode);
            append_value(distance_field_bytes, field.content_hash);
            append_value(distance_field_bytes, static_cast<std::uint32_t>(field.bricks.size()));
            for (const auto& brick : field.bricks)
            {
                for (const auto coordinate : brick.coordinate)
                    append_value(distance_field_bytes, coordinate);
                append_value(distance_field_bytes, brick.page_index);
                append_value(distance_field_bytes, brick.page_offset);
                append_value(distance_field_bytes, brick.byte_size);
                append_value(distance_field_bytes, brick.minimum_distance);
                append_value(distance_field_bytes, brick.maximum_distance);
            }
            append_bytes(distance_field_bytes, std::as_bytes(std::span(field.page_offsets)));
            append_bytes(distance_field_bytes, field.pages);
        }
        const auto name = context.source.source_path.stem().string();
        return {.artifacts = {{.name = name,
                               .extension = ".arcmesh",
                               .schema = artifact_schemas::mesh,
                               .schema_version = 2,
                               .bytes = std::move(conventional_bytes)},
                              {.name = name,
                               .extension = ".arcvg",
                               .schema = artifact_schemas::virtual_geometry,
                               .schema_version = 2,
                               .bytes = std::move(virtual_bytes)},
                              {.name = name,
                               .extension = ".arccards",
                               .schema = artifact_schemas::surface_cards,
                               .schema_version = 1,
                               .bytes = std::move(card_bytes)},
                              {.name = name,
                               .extension = ".arcsdf",
                               .schema = artifact_schemas::mesh_distance_field,
                               .schema_version = 1,
                               .bytes = std::move(distance_field_bytes)}}};
    }

private:
    asset_cook_processor_descriptor descriptor_;
};

class shader_processor final : public asset_cook_processor
{
public:
    shader_processor()
    {
        descriptor_.id = cook_processor_ids::shader;
        descriptor_.name = "ARC Shader";
        descriptor_.schema = artifact_schemas::shader;
        descriptor_.input_types = {asset_types::shader};
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }
    std::string toolchain_fingerprint() const override
    {
        const auto compiler = environment_value("ARC_SHADER_COMPILER_FINGERPRINT");
        return compiler.value_or("arc.glsl-source-package/1");
    }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        std::vector<std::byte> bytes;
        append_string(bytes, "ARC_SHADER_1");
        append_string(bytes, context.source.source_path.extension().string());
        append_string(bytes, canonical_cook_target(context.target));
        append_bytes(bytes, context.source.bytes);
        return {
            .artifacts = {{.name = context.source.source_path.filename().string(),
                           .extension = ".arcshader",
                           .schema = descriptor_.schema,
                           .schema_version = descriptor_.schema_version,
                           .bytes = std::move(bytes)}},
            .diagnostics = {
                {.severity = asset_diagnostic_severity::information,
                 .guid = context.asset.guid,
                 .category = "cook.shader",
                 .message = "Shader source and target metadata packaged; binary compiler adapter was not configured"}}};
    }

private:
    asset_cook_processor_descriptor descriptor_;
};

class unsupported_processor final : public asset_cook_processor
{
public:
    unsupported_processor(asset_type_id type, cook_processor_id id, std::string name)
    {
        descriptor_.id = id;
        descriptor_.name = std::move(name);
        descriptor_.schema = artifact_schemas::source;
        descriptor_.input_types = {type};
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }
    std::string toolchain_fingerprint() const override
    {
        return "unsupported/1";
    }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        return {.error = {.code = asset_error_code::import_failed,
                          .guid = context.asset.guid,
                          .path = context.source.source_path,
                          .message = descriptor_.name +
                                     " cooking is not implemented because its runtime module does not exist"}};
    }

private:
    asset_cook_processor_descriptor descriptor_;
};

struct command_line
{
    std::string command;
    std::filesystem::path project{std::filesystem::current_path()};
    std::filesystem::path output;
    std::filesystem::path manifest;
    std::string profile{"windows-x64-vulkan"};
    std::vector<std::string> roots;
    bool fail_on_warning{};
    bool require_shared{};
    bool json{};
};

void print_usage()
{
    std::cout << "arc-cook <cook|package|verify|clean|cache> [options]\n"
                 "  --project <path> --root <guid-or-path> --profile <name> --output <path>\n"
                 "  --manifest <path> --fail-on-warning --require-shared-cache --json\n"
                 "  cache subcommands: stats, verify, prune\n";
}

std::optional<command_line> parse_command_line(int argc, char** argv)
{
    if (argc < 2) return std::nullopt;
    command_line result;
    result.command = argv[1];
    for (int index = 2; index < argc; ++index)
    {
        const std::string_view argument = argv[index];
        const auto value = [&]() -> const char* { return index + 1 < argc ? argv[++index] : nullptr; };
        if (argument == "--project")
        {
            if (const auto* v = value())
                result.project = v;
            else
                return std::nullopt;
        }
        else if (argument == "--output")
        {
            if (const auto* v = value())
                result.output = v;
            else
                return std::nullopt;
        }
        else if (argument == "--manifest")
        {
            if (const auto* v = value())
                result.manifest = v;
            else
                return std::nullopt;
        }
        else if (argument == "--profile")
        {
            if (const auto* v = value())
                result.profile = v;
            else
                return std::nullopt;
        }
        else if (argument == "--root")
        {
            if (const auto* v = value())
                result.roots.emplace_back(v);
            else
                return std::nullopt;
        }
        else if (argument == "--fail-on-warning")
            result.fail_on_warning = true;
        else if (argument == "--require-shared-cache")
            result.require_shared = true;
        else if (argument == "--json")
            result.json = true;
        else if (result.command == "cache" && result.profile == "windows-x64-vulkan")
            result.profile = std::string(argument);
        else
            return std::nullopt;
    }
    result.project = std::filesystem::absolute(result.project).lexically_normal();
    return result;
}

cook_target target_for(const project::cook_profile_descriptor& profile)
{
    cook_target result;
    result.name = profile.id;
    result.platform = profile.platform == "linux"   ? cook_platform::linux_os
                      : profile.platform == "macos" ? cook_platform::macos
                                                    : cook_platform::windows;
    result.architecture = profile.architecture == "arm64" ? cook_architecture::arm64 : cook_architecture::x86_64;
    result.renderer = profile.renderer == "none"         ? cook_renderer::none
                      : profile.renderer == "direct3d12" ? cook_renderer::direct3d12
                      : profile.renderer == "metal"      ? cook_renderer::metal
                                                         : cook_renderer::vulkan;
    result.textures = profile.texture_family == "astc"       ? cook_texture_family::astc
                      : profile.texture_family == "etc2"     ? cook_texture_family::etc2
                      : profile.texture_family == "portable" ? cook_texture_family::portable
                                                             : cook_texture_family::bc;
    result.configuration =
        profile.configuration == "Shipping" ? cook_configuration::shipping : cook_configuration::development;
    const auto separator = profile.api.find('.');
    if (separator != std::string::npos)
    {
        result.api_major = static_cast<std::uint32_t>(std::stoul(profile.api.substr(0, separator)));
        result.api_minor = static_cast<std::uint32_t>(std::stoul(profile.api.substr(separator + 1)));
    }
    else if (profile.api.empty())
    {
        result.api_major = 0;
        result.api_minor = 0;
    }
    return result;
}

void register_processors(asset_cooker& cooker)
{
    cooker.register_processor(std::make_unique<source_processor>());
    cooker.register_processor(std::make_unique<mesh_processor>());
    cooker.register_processor(std::make_unique<texture_processor>());
    cooker.register_processor(std::make_unique<shader_processor>());
    cooker.register_processor(std::make_unique<document_processor>(
        asset_types::material, cook_processor_ids::material, artifact_schemas::material, "ARC Material", ".arcmatc"));
    cooker.register_processor(std::make_unique<document_processor>(asset_types::scene, cook_processor_ids::scene,
                                                                   artifact_schemas::scene, "ARC Scene", ".arcscenec",
                                                                   persistence::document_kind::scene));
    cooker.register_processor(std::make_unique<document_processor>(asset_types::prefab, cook_processor_ids::scene,
                                                                   artifact_schemas::scene, "ARC Prefab", ".arcprefabc",
                                                                   persistence::document_kind::prefab));
    cooker.register_processor(std::make_unique<unsupported_processor>(
        asset_types::animation_clip, cook_processor_ids::animation, "Animation compression"));
    cooker.register_processor(
        std::make_unique<unsupported_processor>(asset_types::collision, cook_processor_ids::collision, "Collision"));
    cooker.register_processor(
        std::make_unique<unsupported_processor>(asset_types::navigation, cook_processor_ids::navigation, "Navigation"));
    cooker.register_processor(
        std::make_unique<unsupported_processor>(asset_types::audio_clip, cook_processor_ids::audio, "Audio encoding"));
}

std::filesystem::path find_project_descriptor(const std::filesystem::path& candidate)
{
    if (std::filesystem::is_regular_file(candidate) && candidate.extension() == ".arcproject") return candidate;
    if (!std::filesystem::is_directory(candidate)) return {};
    std::vector<std::filesystem::path> descriptors;
    for (const auto& entry : std::filesystem::directory_iterator(candidate))
        if (entry.is_regular_file() && entry.path().extension() == ".arcproject") descriptors.push_back(entry.path());
    return descriptors.size() == 1 ? descriptors.front() : std::filesystem::path{};
}

std::vector<asset_guid> resolve_roots(asset_manager& assets, const command_line& command,
                                      const project::project_descriptor& descriptor)
{
    std::vector<std::string> authored = command.roots;
    if (authored.empty())
    {
        if (descriptor.default_scene)
            authored.push_back(descriptor.default_scene->guid.empty() ? descriptor.default_scene->path_hint
                                                                      : descriptor.default_scene->guid);
        for (const auto& scene : descriptor.startup_scenes)
            authored.push_back(scene.guid.empty() ? scene.path_hint : scene.guid);
    }
    std::vector<asset_guid> result;
    for (const auto& root : authored)
    {
        if (const auto guid = parse_asset_guid(root))
            result.push_back(*guid);
        else if (const auto asset = assets.find(normalize_asset_path(root)))
            result.push_back(asset->guid);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h" ||
                      std::string_view(argv[1]) == "help"))
    {
        print_usage();
        return 0;
    }
    const auto parsed = parse_command_line(argc, argv);
    if (!parsed)
    {
        print_usage();
        return 2;
    }
    auto command = *parsed;
    const auto descriptor_path = find_project_descriptor(command.project);
    if (descriptor_path.empty())
    {
        std::cerr << "--project must identify a directory containing exactly one .arcproject descriptor\n";
        return 2;
    }
    const auto descriptor = project::load_descriptor(descriptor_path);
    if (!descriptor)
    {
        std::cerr << descriptor.error().message << '\n';
        return 1;
    }
    const auto project_context = project::resolve_context(descriptor_path, descriptor.value());
    if (!project_context)
    {
        std::cerr << project_context.error().message << '\n';
        return 1;
    }
    command.project = project_context.value().root;
    const auto selected_profile =
        std::find_if(descriptor.value().cook_profiles.begin(), descriptor.value().cook_profiles.end(),
                     [&](const auto& profile) { return profile.id == command.profile; });
    if ((command.command == "cook" || (command.command == "package" && command.manifest.empty())) &&
        selected_profile == descriptor.value().cook_profiles.end())
    {
        std::cerr << "Cook profile '" << command.profile << "' is not declared by the project\n";
        return 2;
    }
    if (command.output.empty()) command.output = project_context.value().build_root / "Cooked" / command.profile;
    const auto cache_root = project_context.value().asset_cache_root;

    if (command.command == "clean")
    {
        std::error_code error;
        std::filesystem::remove_all(cache_root / "cas", error);
        std::filesystem::remove_all(cache_root / "actions", error);
        return error ? 1 : 0;
    }

    std::shared_ptr<shared_cache_backend> shared;
    if (const auto shared_path = environment_value("ARC_SHARED_CACHE_PATH"))
        shared = std::make_shared<filesystem_shared_cache>(*shared_path,
                                                           environment_value("ARC_SHARED_CACHE_READ_ONLY").has_value());
    derived_data_cache cache({.root = cache_root,
                              .access = cache_access::read_write,
                              .shared = std::move(shared),
                              .require_shared = command.require_shared});

    if (command.command == "cache")
    {
        if (command.profile == "verify")
        {
            std::vector<std::string> diagnostics;
            const auto valid = cache.verify(&diagnostics);
            for (const auto& value : diagnostics)
                std::cerr << value << '\n';
            std::cout << "verified " << valid << " blobs\n";
            return diagnostics.empty() ? 0 : 1;
        }
        if (command.profile == "prune")
        {
            std::cout << "removed " << cache.prune(true) << " bytes\n";
            return 0;
        }
        const auto stats = cache.statistics();
        if (command.json)
        {
            std::cout << nlohmann::json{{"localBytes", stats.local_bytes},
                                        {"localHits", stats.local_hits},
                                        {"localMisses", stats.local_misses},
                                        {"sharedHits", stats.shared_hits},
                                        {"sharedMisses", stats.shared_misses},
                                        {"hitRate", stats.hit_rate()},
                                        {"evictions", stats.evictions},
                                        {"corruptEntries", stats.corrupt_entries},
                                        {"avoidedProcessorRuns", stats.avoided_processor_runs}}
                             .dump()
                      << '\n';
        }
        else
        {
            std::cout << "localBytes=" << stats.local_bytes << " hits=" << stats.local_hits
                      << " misses=" << stats.local_misses << " hitRate=" << stats.hit_rate()
                      << " evictions=" << stats.evictions << " corrupt=" << stats.corrupt_entries << '\n';
        }
        return 0;
    }

    if (command.command == "verify")
    {
        const auto manifest_path =
            command.manifest.empty() ? command.output / (command.profile + ".arccookmanifest") : command.manifest;
        auto loaded_manifest = load_cook_manifest(manifest_path);
        if (!loaded_manifest)
        {
            std::cerr << loaded_manifest.error().message << '\n';
            return 1;
        }
        cook_manifest manifest = std::move(loaded_manifest).value();
        asset_package_mount mount;
        auto mounted = mount.mount(manifest_path);
        if (mounted)
        {
            for (const auto& artifact : mount.manifest().artifacts)
            {
                auto read = mount.read(artifact.asset, artifact.schema);
                if (!read)
                {
                    std::cerr << read.error().message << '\n';
                    return 1;
                }
            }
            std::cout << "verified package " << manifest_path.generic_string() << '\n';
            return 0;
        }
        auto verified = verify_cook_manifest(manifest, cache);
        if (!verified)
        {
            std::cerr << verified.error().message << '\n';
            return 1;
        }
        std::cout << "verified manifest " << manifest_path.generic_string() << '\n';
        return 0;
    }

    if (command.command == "package" && !command.manifest.empty() && std::filesystem::is_regular_file(command.manifest))
    {
        auto manifest = load_cook_manifest(command.manifest);
        if (!manifest)
        {
            std::cerr << manifest.error().message << '\n';
            return 1;
        }
        auto package = build_asset_packages(manifest.value(), cache, command.output);
        if (!package.succeeded())
        {
            std::cerr << package.error << '\n';
            return 1;
        }
        if (command.json)
            std::cout << nlohmann::json{{"event", "package.complete"},
                                        {"artifacts", manifest.value().artifacts.size()},
                                        {"chunks", package.chunks.size()}}
                             .dump()
                      << '\n';
        else
            std::cout << "packaged " << manifest.value().artifacts.size() << " artifacts\n";
        return 0;
    }

    memory::memory_system memory;
    jobs::job_system jobs({.memory = &memory});
    io::async_file_service files(jobs);
    const auto& asset_roots = project_context.value().asset_roots;
    asset_manager assets(
        {.project_root = command.project,
         .asset_root = asset_roots.front(),
         .additional_source_roots = std::vector<std::filesystem::path>(asset_roots.begin() + 1, asset_roots.end()),
         .cache_root = cache_root,
         .target_profile = command.profile,
         .enable_source_monitor = false},
        jobs, files, memory);
    framework::runtime_service_registry services;
    framework::runtime_service_context context(services);
    assets.on_start(context);

    asset_cooker cooker(assets, cache);
    register_processors(cooker);
    const auto roots = resolve_roots(assets, command, descriptor.value());
    if (roots.empty())
    {
        std::cerr << "No cook roots resolved. Add a default/startup scene to .arcproject or pass --root.\n";
        assets.on_shutdown(context);
        return 1;
    }
    const auto cooked = cooker.cook({.roots = roots,
                                     .target = target_for(*selected_profile),
                                     .output = command.output,
                                     .fail_on_warning = command.fail_on_warning});
    if (!cooked.succeeded())
    {
        std::cerr << cooked.error.message << '\n';
        assets.on_shutdown(context);
        return 1;
    }

    std::string error;
    std::filesystem::create_directories(command.output);
    const auto manifest_path = command.output / (command.profile + ".arccookmanifest");
    if (command.command == "package")
    {
        auto package = build_asset_packages(cooked.manifest, cache, command.output);
        if (!package.succeeded())
        {
            std::cerr << package.error << '\n';
            assets.on_shutdown(context);
            return 1;
        }
        if (command.json)
            std::cout << nlohmann::json{{"event", "package.complete"},
                                        {"cooked", cooked.cooked},
                                        {"cacheHits", cooked.cache_hits},
                                        {"artifacts", cooked.manifest.artifacts.size()},
                                        {"chunks", package.chunks.size()},
                                        {"sourceBytes", package.source_bytes},
                                        {"storedBytes", package.stored_bytes},
                                        {"manifest", package.manifest_path.generic_string()}}
                             .dump()
                      << '\n';
        else
            std::cout << "packaged " << cooked.manifest.artifacts.size() << " artifacts into " << package.chunks.size()
                      << " chunks\n";
    }
    else if (command.command == "cook")
    {
        auto saved = save_cook_manifest(manifest_path, cooked.manifest);
        if (!saved)
        {
            std::cerr << saved.error().message << '\n';
            assets.on_shutdown(context);
            return 1;
        }
        if (command.json)
            std::cout << nlohmann::json{{"event", "cook.complete"},
                                        {"cooked", cooked.cooked},
                                        {"cacheHits", cooked.cache_hits},
                                        {"artifacts", cooked.manifest.artifacts.size()},
                                        {"manifest", manifest_path.generic_string()}}
                             .dump()
                      << '\n';
        else
            std::cout << "cooked=" << cooked.cooked << " cacheHits=" << cooked.cache_hits
                      << " artifacts=" << cooked.manifest.artifacts.size() << '\n';
    }
    else
    {
        print_usage();
        assets.on_shutdown(context);
        return 2;
    }
    assets.on_shutdown(context);
    return 0;
}
