#include <arc/assets/cook.h>
#include <arc/framework/service.h>
#include <arc/io/io.h>
#include <arc/memory/memory.h>
#include <arc/persistence/persistence.h>
#include <arc/render/mesh.h>
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

template <class T>
void append_value(std::vector<std::byte>& output, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(T));
}

void append_string(std::vector<std::byte>& output, std::string_view value)
{
    append_value(output, static_cast<std::uint64_t>(value.size()));
    output.insert(output.end(),
        reinterpret_cast<const std::byte*>(value.data()),
        reinterpret_cast<const std::byte*>(value.data() + value.size()));
}

void append_bytes(std::vector<std::byte>& output, std::span<const std::byte> value)
{
    append_value(output, static_cast<std::uint64_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
}

class document_processor final : public asset_cook_processor
{
public:
    document_processor(asset_type_id type, cook_processor_id id, artifact_schema_id schema,
        std::string name, std::string extension,
        std::optional<persistence::document_kind> document_kind = std::nullopt)
        : extension_(std::move(extension))
        , document_kind_(document_kind)
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

    const asset_cook_processor_descriptor& descriptor() const noexcept override { return descriptor_; }
    std::string toolchain_fingerprint() const override { return "arc.document-cooker/2"; }

    asset_cook_result cook(const asset_cook_context& context) override
    {
        const auto document = nlohmann::json::parse(
            reinterpret_cast<const char*>(context.source.bytes.data()),
            reinterpret_cast<const char*>(context.source.bytes.data() + context.source.bytes.size()),
            nullptr, false);
        if (document.is_discarded())
            return { .error = { .code = asset_error_code::import_failed,
                .guid = context.asset.guid, .path = context.source.source_path,
                .message = "Authored JSON document is invalid" } };
        if (document_kind_)
        {
            persistence::component_persistence_registry components;
            persistence::schema_migration_registry migrations;
            std::string error;
            if (!scene::register_persistence_components(components) ||
                !scene::register_persistence_migrations(migrations, error))
                return { .error = { .code = asset_error_code::import_failed,
                    .guid = context.asset.guid, .path = context.source.source_path,
                    .message = "Persistence registry initialization failed: " + error } };
            const std::string_view source(
                reinterpret_cast<const char*>(context.source.bytes.data()),
                context.source.bytes.size());
            auto archive = persistence::read_reflected_json(
                source, components, &migrations);
            if (!archive.succeeded() || archive.document.kind != *document_kind_)
                return { .error = { .code = asset_error_code::import_failed,
                    .guid = context.asset.guid, .path = context.source.source_path,
                    .message = archive.error.empty()
                        ? "Document kind does not match its asset type" : archive.error } };
            for (const auto& dependency : archive.document.dependencies)
            {
                if (!dependency.required)
                    continue;
                const auto found = std::find_if(
                    context.dependencies.begin(), context.dependencies.end(),
                    [&](const asset_snapshot& candidate) {
                        return candidate.guid == dependency.reference.guid;
                    });
                if (!dependency.reference.guid.valid() ||
                    found == context.dependencies.end() ||
                    (dependency.reference.expected_type.valid() &&
                        found->type != dependency.reference.expected_type))
                {
                    return { .error = { .code = asset_error_code::dependency_failed,
                        .guid = context.asset.guid, .path = context.source.source_path,
                        .message = "Required document dependency is unresolved or has the wrong type: " +
                            dependency.reference.path_hint } };
                }
            }
            auto bytes = persistence::write_tagged_binary(
                archive.document, canonical_cook_target(context.target), error);
            if (!error.empty())
                return { .error = { .code = asset_error_code::import_failed,
                    .guid = context.asset.guid, .path = context.source.source_path,
                    .message = std::move(error) } };
            return { .artifacts = {{
                .name = context.source.source_path.stem().string(),
                .extension = extension_,
                .schema = descriptor_.schema,
                .schema_version = descriptor_.schema_version,
                .bytes = std::move(bytes)
            }} };
        }
        const auto canonical = document.dump();
        std::vector<std::byte> bytes;
        append_string(bytes, "ARC_DOCUMENT_1");
        append_string(bytes, canonical);
        return { .artifacts = {{
            .name = context.source.source_path.stem().string(),
            .extension = extension_,
            .schema = descriptor_.schema,
            .schema_version = descriptor_.schema_version,
            .bytes = std::move(bytes)
        }} };
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
        descriptor_.input_types = { asset_types::binary_blob };
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override { return descriptor_; }
    std::string toolchain_fingerprint() const override { return "arc.source-cooker/1"; }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        return { .artifacts = {{
            .name = context.source.source_path.stem().string(),
            .extension = ".arcbin",
            .schema = descriptor_.schema,
            .schema_version = descriptor_.schema_version,
            .bytes = context.source.bytes
        }} };
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
        descriptor_.input_types = { asset_types::texture_2d, asset_types::environment };
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override { return descriptor_; }
    std::string toolchain_fingerprint() const override { return "arc.texture-cooker/1;stb;basis-contract-1"; }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        auto loaded = render::load_texture_asset_bytes(context.source.bytes, context.source.source_path);
        if (!loaded.succeeded())
            return { .error = { .code = asset_error_code::import_failed,
                .guid = context.asset.guid, .path = context.source.source_path,
                .message = loaded.message.empty() ? "Texture decode failed" : loaded.message } };
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
            diagnostics.push_back({ .severity = asset_diagnostic_severity::information,
                .guid = context.asset.guid, .category = "cook.texture",
                .message = "Texture mip chain is ready for the registered BC encoder; stored uncompressed by this build" });
        return {
            .artifacts = {{
                .name = context.source.source_path.stem().string(),
                .extension = ".arctex",
                .schema = descriptor_.schema,
                .schema_version = descriptor_.schema_version,
                .gpu_compressed = texture.compressed,
                .bytes = std::move(bytes)
            }},
            .diagnostics = std::move(diagnostics)
        };
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
        descriptor_.input_types = { asset_types::imported_scene, asset_types::static_mesh };
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override { return descriptor_; }
    std::string toolchain_fingerprint() const override { return "arc.mesh-cooker/1;meshoptimizer-contract-1"; }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        const auto loaded = render::load_scene_asset(context.source.source_path);
        if (!loaded.succeeded())
            return { .error = { .code = asset_error_code::import_failed,
                .guid = context.asset.guid, .path = context.source.source_path,
                .message = loaded.message.empty() ? "Mesh import failed" : loaded.message } };
        std::vector<std::byte> bytes;
        append_string(bytes, "ARC_MESH_1");
        append_value(bytes, static_cast<std::uint32_t>(loaded.meshes.size()));
        for (const auto& mesh : loaded.meshes)
        {
            const auto virtual_mesh = render::build_virtual_mesh(mesh, { .max_triangles_per_cluster = 124 });
            append_string(bytes, mesh.name);
            append_value(bytes, static_cast<std::uint64_t>(mesh.vertices.size()));
            append_bytes(bytes, std::as_bytes(std::span(mesh.vertices)));
            append_value(bytes, static_cast<std::uint64_t>(mesh.indices.size()));
            append_bytes(bytes, std::as_bytes(std::span(mesh.indices)));
            append_value(bytes, static_cast<std::uint32_t>(virtual_mesh.clusters.size()));
            append_bytes(bytes, std::as_bytes(std::span(virtual_mesh.clusters)));
            constexpr std::array<float, 4> lod_ratios{ 1.0f, 0.5f, 0.25f, 0.125f };
            append_value(bytes, static_cast<std::uint32_t>(lod_ratios.size()));
            for (const auto ratio : lod_ratios)
            {
                const auto triangle_count = mesh.indices.size() / 3;
                const auto lod_triangles = std::max<std::size_t>(1,
                    static_cast<std::size_t>(triangle_count * ratio));
                append_value(bytes, ratio);
                append_value(bytes, static_cast<std::uint32_t>(
                    std::min(lod_triangles, triangle_count)));
            }
        }
        return { .artifacts = {{
            .name = context.source.source_path.stem().string(),
            .extension = ".arcmesh",
            .schema = descriptor_.schema,
            .schema_version = descriptor_.schema_version,
            .bytes = std::move(bytes)
        }} };
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
        descriptor_.input_types = { asset_types::shader };
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override { return descriptor_; }
    std::string toolchain_fingerprint() const override
    {
        const char* compiler = std::getenv("ARC_SHADER_COMPILER_FINGERPRINT");
        return compiler ? compiler : "arc.glsl-source-package/1";
    }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        std::vector<std::byte> bytes;
        append_string(bytes, "ARC_SHADER_1");
        append_string(bytes, context.source.source_path.extension().string());
        append_string(bytes, canonical_cook_target(context.target));
        append_bytes(bytes, context.source.bytes);
        return { .artifacts = {{
            .name = context.source.source_path.filename().string(),
            .extension = ".arcshader",
            .schema = descriptor_.schema,
            .schema_version = descriptor_.schema_version,
            .bytes = std::move(bytes)
        }}, .diagnostics = {{
            .severity = asset_diagnostic_severity::information,
            .guid = context.asset.guid,
            .category = "cook.shader",
            .message = "Shader source and target metadata packaged; binary compiler adapter was not configured"
        }} };
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
        descriptor_.input_types = { type };
    }
    const asset_cook_processor_descriptor& descriptor() const noexcept override { return descriptor_; }
    std::string toolchain_fingerprint() const override { return "unsupported/1"; }
    asset_cook_result cook(const asset_cook_context& context) override
    {
        return { .error = { .code = asset_error_code::import_failed,
            .guid = context.asset.guid, .path = context.source.source_path,
            .message = descriptor_.name + " cooking is not implemented because its runtime module does not exist" } };
    }
private:
    asset_cook_processor_descriptor descriptor_;
};

struct command_line
{
    std::string command;
    std::filesystem::path project{ std::filesystem::current_path() };
    std::filesystem::path output;
    std::filesystem::path manifest;
    std::string profile{ "windows-x64-vulkan" };
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
    if (argc < 2)
        return std::nullopt;
    command_line result;
    result.command = argv[1];
    for (int index = 2; index < argc; ++index)
    {
        const std::string_view argument = argv[index];
        const auto value = [&]() -> const char* {
            return index + 1 < argc ? argv[++index] : nullptr;
        };
        if (argument == "--project") { if (const auto* v = value()) result.project = v; else return std::nullopt; }
        else if (argument == "--output") { if (const auto* v = value()) result.output = v; else return std::nullopt; }
        else if (argument == "--manifest") { if (const auto* v = value()) result.manifest = v; else return std::nullopt; }
        else if (argument == "--profile") { if (const auto* v = value()) result.profile = v; else return std::nullopt; }
        else if (argument == "--root") { if (const auto* v = value()) result.roots.emplace_back(v); else return std::nullopt; }
        else if (argument == "--fail-on-warning") result.fail_on_warning = true;
        else if (argument == "--require-shared-cache") result.require_shared = true;
        else if (argument == "--json") result.json = true;
        else if (result.command == "cache" && result.profile == "windows-x64-vulkan") result.profile = std::string(argument);
        else return std::nullopt;
    }
    result.project = std::filesystem::absolute(result.project).lexically_normal();
    if (result.output.empty())
        result.output = result.project / "out" / "cooked" / result.profile;
    return result;
}

cook_target target_for(std::string_view profile)
{
    return profile == "linux-x64-vulkan"
        ? linux_vulkan_cook_target() : windows_vulkan_cook_target();
}

void register_processors(asset_cooker& cooker)
{
    cooker.register_processor(std::make_unique<source_processor>());
    cooker.register_processor(std::make_unique<mesh_processor>());
    cooker.register_processor(std::make_unique<texture_processor>());
    cooker.register_processor(std::make_unique<shader_processor>());
    cooker.register_processor(std::make_unique<document_processor>(
        asset_types::material, cook_processor_ids::material, artifact_schemas::material,
        "ARC Material", ".arcmatc"));
    cooker.register_processor(std::make_unique<document_processor>(
        asset_types::scene, cook_processor_ids::scene, artifact_schemas::scene,
        "ARC Scene", ".arcscenec", persistence::document_kind::scene));
    cooker.register_processor(std::make_unique<document_processor>(
        asset_types::prefab, cook_processor_ids::scene, artifact_schemas::scene,
        "ARC Prefab", ".arcprefabc", persistence::document_kind::prefab));
    cooker.register_processor(std::make_unique<unsupported_processor>(
        asset_types::animation_clip, cook_processor_ids::animation, "Animation compression"));
    cooker.register_processor(std::make_unique<unsupported_processor>(
        asset_types::collision, cook_processor_ids::collision, "Collision"));
    cooker.register_processor(std::make_unique<unsupported_processor>(
        asset_types::navigation, cook_processor_ids::navigation, "Navigation"));
    cooker.register_processor(std::make_unique<unsupported_processor>(
        asset_types::audio_clip, cook_processor_ids::audio, "Audio encoding"));
}

std::vector<asset_guid> resolve_roots(asset_manager& assets, const command_line& command)
{
    std::vector<std::string> authored = command.roots;
    const auto config_path = command.project / "arc.cook.json";
    if (authored.empty())
    {
        std::ifstream stream(config_path);
        const auto config = stream ? nlohmann::json::parse(stream, nullptr, false) : nlohmann::json{};
        if (config.is_object() && config.contains("roots") && config["roots"].is_array())
            for (const auto& root : config["roots"])
                if (root.is_string()) authored.push_back(root.get<std::string>());
    }
    std::vector<asset_guid> result;
    for (const auto& root : authored)
    {
        if (const auto guid = parse_asset_guid(root))
            result.push_back(*guid);
        else if (const auto asset = assets.find(normalize_asset_path(root)))
            result.push_back(asset->guid);
        else if (const auto asset = assets.find(normalize_asset_path(std::filesystem::path("assets") / root)))
            result.push_back(asset->guid);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

}

int main(int argc, char** argv)
{
    if (argc == 2 && (std::string_view(argv[1]) == "--help" ||
        std::string_view(argv[1]) == "-h" || std::string_view(argv[1]) == "help"))
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
    const auto& command = *parsed;
    const auto cache_root = command.project / ".arc" / "cache";

    if (command.command == "clean")
    {
        std::error_code error;
        std::filesystem::remove_all(cache_root / "cas", error);
        std::filesystem::remove_all(cache_root / "actions", error);
        return error ? 1 : 0;
    }

    std::shared_ptr<shared_cache_backend> shared;
    if (const char* shared_path = std::getenv("ARC_SHARED_CACHE_PATH"))
        shared = std::make_shared<filesystem_shared_cache>(shared_path,
            std::getenv("ARC_SHARED_CACHE_READ_ONLY") != nullptr);
    derived_data_cache cache({
        .root = cache_root,
        .access = cache_access::read_write,
        .shared = std::move(shared),
        .require_shared = command.require_shared
    });

    if (command.command == "cache")
    {
        if (command.profile == "verify")
        {
            std::vector<std::string> diagnostics;
            const auto valid = cache.verify(&diagnostics);
            for (const auto& value : diagnostics) std::cerr << value << '\n';
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
            std::cout << nlohmann::json{
                { "localBytes", stats.local_bytes },
                { "localHits", stats.local_hits },
                { "localMisses", stats.local_misses },
                { "sharedHits", stats.shared_hits },
                { "sharedMisses", stats.shared_misses },
                { "hitRate", stats.hit_rate() },
                { "evictions", stats.evictions },
                { "corruptEntries", stats.corrupt_entries },
                { "avoidedProcessorRuns", stats.avoided_processor_runs }
            }.dump() << '\n';
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
        cook_manifest manifest;
        std::string error;
        const auto manifest_path = command.manifest.empty()
            ? command.output / (command.profile + ".arccookmanifest") : command.manifest;
        if (!load_cook_manifest(manifest_path, manifest, error))
        {
            std::cerr << error << '\n';
            return 1;
        }
        asset_package_mount mount;
        if (mount.mount(manifest_path, error))
        {
            for (const auto& artifact : mount.manifest().artifacts)
                if (!mount.read(artifact.asset, artifact.schema, error))
                {
                    std::cerr << error << '\n';
                    return 1;
                }
            std::cout << "verified package " << manifest_path.generic_string() << '\n';
            return 0;
        }
        std::vector<std::string> diagnostics;
        if (!verify_cook_manifest(manifest, cache, diagnostics))
        {
            for (const auto& value : diagnostics) std::cerr << value << '\n';
            return 1;
        }
        std::cout << "verified manifest " << manifest_path.generic_string() << '\n';
        return 0;
    }

    memory_system memory;
    job_system jobs({ .memory = &memory });
    io::async_file_service files(jobs);
    asset_manager assets({
        .project_root = command.project,
        .asset_root = command.project / "assets",
        .cache_root = cache_root,
        .target_profile = command.profile,
        .enable_source_monitor = false
    }, jobs, files, memory);
    runtime_service_registry services;
    runtime_service_context context(services);
    assets.on_start(context);

    asset_cooker cooker(assets, cache);
    register_processors(cooker);
    const auto roots = resolve_roots(assets, command);
    if (roots.empty())
    {
        std::cerr << "No cook roots resolved. Add roots to arc.cook.json or pass --root.\n";
        assets.on_shutdown(context);
        return 1;
    }
    const auto cooked = cooker.cook({
        .roots = roots,
        .target = target_for(command.profile),
        .output = command.output,
        .fail_on_warning = command.fail_on_warning
    });
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
            std::cout << nlohmann::json{
                { "event", "package.complete" },
                { "cooked", cooked.cooked },
                { "cacheHits", cooked.cache_hits },
                { "artifacts", cooked.manifest.artifacts.size() },
                { "chunks", package.chunks.size() },
                { "sourceBytes", package.source_bytes },
                { "storedBytes", package.stored_bytes },
                { "manifest", package.manifest_path.generic_string() }
            }.dump() << '\n';
        else
            std::cout << "packaged " << cooked.manifest.artifacts.size() << " artifacts into "
                << package.chunks.size() << " chunks\n";
    }
    else if (command.command == "cook")
    {
        if (!save_cook_manifest(manifest_path, cooked.manifest, error))
        {
            std::cerr << error << '\n';
            assets.on_shutdown(context);
            return 1;
        }
        if (command.json)
            std::cout << nlohmann::json{
                { "event", "cook.complete" },
                { "cooked", cooked.cooked },
                { "cacheHits", cooked.cache_hits },
                { "artifacts", cooked.manifest.artifacts.size() },
                { "manifest", manifest_path.generic_string() }
            }.dump() << '\n';
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
