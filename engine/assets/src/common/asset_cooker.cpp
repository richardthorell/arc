#include <arc/assets/cook.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace arc::assets
{
namespace
{

using json = nlohmann::json;

std::string platform_name(cook_platform value)
{
    switch (value)
    {
        case cook_platform::windows:
            return "windows";
        case cook_platform::linux_os:
            return "linux";
        case cook_platform::macos:
            return "macos";
    }
    return "windows";
}

std::string architecture_name(cook_architecture value)
{
    return value == cook_architecture::arm64 ? "arm64" : "x86_64";
}

std::string renderer_name(cook_renderer value)
{
    switch (value)
    {
        case cook_renderer::none:
            return "none";
        case cook_renderer::vulkan:
            return "vulkan";
        case cook_renderer::direct3d12:
            return "direct3d12";
        case cook_renderer::metal:
            return "metal";
    }
    return "vulkan";
}

std::string texture_family_name(cook_texture_family value)
{
    switch (value)
    {
        case cook_texture_family::bc:
            return "bc";
        case cook_texture_family::astc:
            return "astc";
        case cook_texture_family::etc2:
            return "etc2";
        case cook_texture_family::portable:
            return "portable";
    }
    return "bc";
}

json target_json(const cook_target& target)
{
    return {{"name", target.name},
            {"platform", platform_name(target.platform)},
            {"architecture", architecture_name(target.architecture)},
            {"renderer", renderer_name(target.renderer)},
            {"textures", texture_family_name(target.textures)},
            {"configuration", target.configuration == cook_configuration::shipping ? "shipping" : "development"},
            {"apiMajor", target.api_major},
            {"apiMinor", target.api_minor},
            {"littleEndian", target.little_endian},
            {"features", target.features}};
}

bool parse_target(const json& value, cook_target& target)
{
    if (!value.is_object()) return false;
    target.name = value.value("name", "windows-x64-vulkan");
    const auto platform = value.value("platform", "windows");
    target.platform = platform == "linux"   ? cook_platform::linux_os
                      : platform == "macos" ? cook_platform::macos
                                            : cook_platform::windows;
    target.architecture =
        value.value("architecture", "x86_64") == "arm64" ? cook_architecture::arm64 : cook_architecture::x86_64;
    const auto renderer = value.value("renderer", "vulkan");
    target.renderer = renderer == "none"       ? cook_renderer::none
                      : renderer == "direct3d12" ? cook_renderer::direct3d12
                      : renderer == "metal"    ? cook_renderer::metal
                                               : cook_renderer::vulkan;
    const auto textures = value.value("textures", "bc");
    target.textures = textures == "astc"       ? cook_texture_family::astc
                      : textures == "etc2"     ? cook_texture_family::etc2
                      : textures == "portable" ? cook_texture_family::portable
                                               : cook_texture_family::bc;
    target.configuration = value.value("configuration", "shipping") == "development" ? cook_configuration::development
                                                                                     : cook_configuration::shipping;
    target.api_major = value.value("apiMajor", 1u);
    target.api_minor = value.value("apiMinor", 2u);
    target.little_endian = value.value("littleEndian", true);
    target.features = value.value("features", std::vector<std::string>{});
    return !target.name.empty();
}

json artifact_metadata(const std::vector<cooked_artifact>& artifacts)
{
    json result = json::array();
    for (const auto& artifact : artifacts)
        result.push_back({{"name", artifact.name},
                          {"extension", artifact.extension},
                          {"schema", to_string(artifact.schema)},
                          {"schemaVersion", artifact.schema_version},
                          {"hash", to_string(artifact.hash)},
                          {"size", artifact.size},
                          {"gpuCompressed", artifact.gpu_compressed}});
    return result;
}

struct cached_artifact_metadata
{
    std::string name;
    std::string extension;
    artifact_schema_id schema{};
    std::uint32_t schema_version{};
    asset_hash hash{};
    std::uint64_t size{};
    bool gpu_compressed{};
};

std::optional<cached_artifact_metadata> parse_cached_artifact(const json& value)
{
    if (!value.is_object()) return std::nullopt;
    const auto schema_text = value.value("schema", "");
    if (schema_text.size() != 32) return std::nullopt;
    const auto parse_half = [](std::string_view text) -> std::optional<std::uint64_t>
    {
        std::uint64_t result{};
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result, 16);
        return error == std::errc{} && end == text.data() + text.size() ? std::optional(result) : std::nullopt;
    };
    const auto high = parse_half(std::string_view(schema_text).substr(0, 16));
    const auto low = parse_half(std::string_view(schema_text).substr(16, 16));
    const auto hash = parse_asset_hash(value.value("hash", ""));
    if (!high || !low || !hash) return std::nullopt;
    return cached_artifact_metadata{.name = value.value("name", ""),
                                    .extension = value.value("extension", ""),
                                    .schema = {*high, *low},
                                    .schema_version = value.value("schemaVersion", 0u),
                                    .hash = *hash,
                                    .size = value.value("size", 0ull),
                                    .gpu_compressed = value.value("gpuCompressed", false)};
}

} // namespace

struct asset_cooker::implementation
{
    asset_manager* assets{};
    derived_data_cache* cache{};
    std::vector<std::unique_ptr<asset_cook_processor>> processors;

    asset_cook_processor* processor_for(asset_type_id type) const
    {
        for (const auto& processor : processors)
            if (std::find(processor->descriptor().input_types.begin(), processor->descriptor().input_types.end(),
                          type) != processor->descriptor().input_types.end())
                return processor.get();
        return nullptr;
    }
};

asset_cooker::asset_cooker(asset_manager& assets, derived_data_cache& cache)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->assets = &assets;
    implementation_->cache = &cache;
}

asset_cooker::~asset_cooker() = default;

bool asset_cooker::register_processor(std::unique_ptr<asset_cook_processor> processor)
{
    if (!processor || !processor->descriptor().id.valid() || !processor->descriptor().schema.valid() ||
        processor->descriptor().input_types.empty())
        return false;
    for (const auto& type : processor->descriptor().input_types)
        if (implementation_->processor_for(type)) return false;
    implementation_->processors.push_back(std::move(processor));
    return true;
}

cook_result asset_cooker::cook(const cook_request& request)
{
    cook_result result;
    result.manifest.target = request.target;
    result.manifest.roots = request.roots;
    if (request.roots.empty())
    {
        result.error = {.code = asset_error_code::invalid_request, .message = "Cook request has no root assets"};
        return result;
    }

    std::vector<asset_guid> order;
    std::unordered_set<asset_guid, asset_guid_hash> visiting;
    std::unordered_set<asset_guid, asset_guid_hash> visited;
    std::function<bool(asset_guid)> visit = [&](asset_guid guid)
    {
        if (request.cancellation.stop_requested())
        {
            result.error = {.code = asset_error_code::cancelled, .guid = guid, .message = "Asset cook was cancelled"};
            return false;
        }
        if (visited.contains(guid)) return true;
        if (!visiting.insert(guid).second)
        {
            result.error = {.code = asset_error_code::dependency_cycle,
                            .guid = guid,
                            .message = "Asset cook dependency graph contains a cycle"};
            return false;
        }
        auto asset = implementation_->assets->find(guid);
        if (!asset)
        {
            result.error = {
                .code = asset_error_code::not_found, .guid = guid, .message = "Asset cook dependency is missing"};
            return false;
        }
        if (asset->state != asset_state::ready)
        {
            const auto imported = implementation_->assets
                                      ->load<source_asset_data>(
                                          {.reference = {guid, asset->type, normalize_asset_path(asset->source_path)},
                                           .priority = asset_streaming_priority::high,
                                           .residency = asset_residency::cpu,
                                           .cancellation = request.cancellation,
                                           .allow_fallback = false})
                                      .get();
            if (!imported)
            {
                result.error = imported.error;
                return false;
            }
            asset = implementation_->assets->find(guid);
            if (!asset)
            {
                result.error = {.code = asset_error_code::not_found,
                                .guid = guid,
                                .message = "Asset disappeared while preparing its cook dependencies"};
                return false;
            }
        }
        auto dependencies = asset->dependencies;
        std::sort(dependencies.begin(), dependencies.end());
        for (const auto dependency : dependencies)
            if (!visit(dependency)) return false;
        visiting.erase(guid);
        visited.insert(guid);
        order.push_back(guid);
        return true;
    };
    auto roots = request.roots;
    std::sort(roots.begin(), roots.end());
    for (const auto root : roots)
        if (!visit(root)) return result;
    result.manifest.dependency_closure = order;

    std::vector<asset_hash> build_hashes;
    for (const auto guid : order)
    {
        const auto snapshot = implementation_->assets->find(guid);
        auto* processor = snapshot ? implementation_->processor_for(snapshot->type) : nullptr;
        if (!snapshot || !processor)
        {
            result.error = {.code = asset_error_code::importer_missing,
                            .guid = guid,
                            .message = "No cook processor is registered for asset type " +
                                       (snapshot ? to_string(snapshot->type) : std::string("unknown"))};
            return result;
        }

        std::vector<asset_hash> dependency_hashes;
        std::vector<asset_snapshot> dependencies;
        for (const auto dependency_guid : snapshot->dependencies)
            if (const auto dependency = implementation_->assets->find(dependency_guid))
            {
                dependency_hashes.push_back(dependency->dependency_hash);
                dependencies.push_back(*dependency);
            }
        std::sort(dependency_hashes.begin(), dependency_hashes.end());
        const auto& descriptor = processor->descriptor();
        const auto key = make_asset_build_key({.source_hash = snapshot->source_hash,
                                               .dependency_hashes = dependency_hashes,
                                               .importer = snapshot->importer,
                                               .importer_version = snapshot->importer_version,
                                               .processor = descriptor.id,
                                               .processor_version = descriptor.version,
                                               .schema = descriptor.schema,
                                               .schema_version = descriptor.schema_version,
                                               .canonical_settings = "{}",
                                               .toolchain_fingerprint = processor->toolchain_fingerprint(),
                                               .target = request.target});
        build_hashes.push_back(key);

        cache_error cache_error_value;
        bool used_cache{};
        if (const auto action = implementation_->cache->get_action(key, cache_error_value))
        {
            const auto metadata = json::parse(action->metadata, nullptr, false);
            if (metadata.is_array() && metadata.size() == action->artifacts.size())
            {
                std::vector<cook_manifest_artifact> cached;
                bool complete = true;
                for (std::size_t index = 0; index < action->artifacts.size(); ++index)
                {
                    const auto artifact = parse_cached_artifact(metadata[index]);
                    cache_error blob_error;
                    const auto blob = implementation_->cache->get_blob(action->artifacts[index], blob_error);
                    if (!artifact || !blob || blob->bytes.size() != artifact->size)
                    {
                        complete = false;
                        break;
                    }
                    cached.push_back({.asset = guid,
                                      .type = snapshot->type,
                                      .name = artifact->name,
                                      .schema = artifact->schema,
                                      .schema_version = artifact->schema_version,
                                      .hash = artifact->hash,
                                      .size = artifact->size,
                                      .chunk = snapshot->type == asset_types::shader ? "boot" : "startup"});
                }
                if (complete)
                {
                    result.manifest.artifacts.insert(result.manifest.artifacts.end(), cached.begin(), cached.end());
                    ++result.cache_hits;
                    implementation_->cache->note_avoided_processor_run();
                    used_cache = true;
                }
            }
        }
        if (used_cache) continue;

        const auto loaded = implementation_->assets
                                ->load<source_asset_data>(
                                    {.reference = {guid, snapshot->type, normalize_asset_path(snapshot->source_path)},
                                     .priority = asset_streaming_priority::high,
                                     .residency = asset_residency::cpu,
                                     .cancellation = request.cancellation,
                                     .allow_fallback = false})
                                .get();
        if (!loaded)
        {
            result.error = loaded.error;
            return result;
        }
        asset_cook_context context{.asset = *snapshot,
                                   .source = *loaded.asset,
                                   .target = request.target,
                                   .dependencies = std::move(dependencies),
                                   .cancellation = request.cancellation};
        auto cooked = processor->cook(context);
        result.diagnostics.insert(result.diagnostics.end(), cooked.diagnostics.begin(), cooked.diagnostics.end());
        if (!cooked.succeeded())
        {
            result.error = cooked.error ? cooked.error
                                        : asset_error{.code = asset_error_code::import_failed,
                                                      .guid = guid,
                                                      .message = "Cook processor returned no artifacts"};
            return result;
        }
        if (request.fail_on_warning &&
            std::any_of(cooked.diagnostics.begin(), cooked.diagnostics.end(), [](const auto& diagnostic)
                        { return diagnostic.severity != asset_diagnostic_severity::information; }))
        {
            result.error = {.code = asset_error_code::import_failed,
                            .guid = guid,
                            .message = "Cook failed because fail-on-warning is enabled"};
            return result;
        }

        cache_action action{.key = key};
        for (auto& artifact : cooked.artifacts)
        {
            artifact.hash = hash_bytes(artifact.bytes);
            artifact.size = artifact.bytes.size();
            cache_error blob_error;
            if (!implementation_->cache->put_blob(artifact.hash, artifact.bytes, blob_error))
            {
                result.error = {.code = asset_error_code::io_failed,
                                .guid = guid,
                                .message = "Could not publish cooked artifact: " + blob_error.message};
                return result;
            }
            action.artifacts.push_back(artifact.hash);
            result.manifest.artifacts.push_back({.asset = guid,
                                                 .type = snapshot->type,
                                                 .name = artifact.name,
                                                 .schema = artifact.schema,
                                                 .schema_version = artifact.schema_version,
                                                 .hash = artifact.hash,
                                                 .size = artifact.size,
                                                 .chunk = snapshot->type == asset_types::shader ? "boot" : "startup"});
        }
        action.metadata = artifact_metadata(cooked.artifacts).dump();
        cache_error action_error;
        if (!implementation_->cache->put_action(action, action_error))
        {
            result.error = {.code = asset_error_code::io_failed,
                            .guid = guid,
                            .message = "Could not publish cook action: " + action_error.message};
            return result;
        }
        ++result.cooked;
    }
    std::sort(result.manifest.artifacts.begin(), result.manifest.artifacts.end(),
              [](const auto& lhs, const auto& rhs)
              {
                  if (lhs.asset != rhs.asset) return lhs.asset < rhs.asset;
                  if (lhs.schema != rhs.schema) return lhs.schema < rhs.schema;
                  return lhs.name < rhs.name;
              });
    result.manifest.build_id = to_string(combine_hashes(build_hashes));
    return result;
}

asset_status save_cook_manifest(const std::filesystem::path& path, const cook_manifest& manifest)
{
    const auto failure = [&](std::string message)
    {
        return asset_status::failure(
            {.code = asset_error_code::io_failed, .path = path, .message = std::move(message)});
    };
    json artifacts = json::array();
    for (const auto& artifact : manifest.artifacts)
        artifacts.push_back({{"asset", to_string(artifact.asset)},
                             {"type", to_string(artifact.type)},
                             {"name", artifact.name},
                             {"schema", to_string(artifact.schema)},
                             {"schemaVersion", artifact.schema_version},
                             {"hash", to_string(artifact.hash)},
                             {"size", artifact.size},
                             {"chunk", artifact.chunk},
                             {"offset", artifact.offset},
                             {"storedSize", artifact.stored_size},
                             {"compressed", artifact.compressed}});
    json roots = json::array();
    for (const auto guid : manifest.roots)
        roots.push_back(to_string(guid));
    json closure = json::array();
    for (const auto guid : manifest.dependency_closure)
        closure.push_back(to_string(guid));
    const json document{{"format", "arc.cook-manifest"},    {"version", manifest.version},
                        {"buildId", manifest.build_id},     {"target", target_json(manifest.target)},
                        {"roots", std::move(roots)},        {"dependencyClosure", std::move(closure)},
                        {"artifacts", std::move(artifacts)}};
    std::error_code filesystem_error;
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream)
        {
            return failure("Could not create cook manifest");
        }
        stream << document.dump(2) << '\n';
        if (!stream)
        {
            return failure("Could not write cook manifest");
        }
    }
    std::filesystem::remove(path, filesystem_error);
    filesystem_error.clear();
    std::filesystem::rename(temporary, path, filesystem_error);
    if (filesystem_error)
    {
        return failure("Could not publish cook manifest: " + filesystem_error.message());
    }
    return asset_status::success();
}

cook_manifest_result load_cook_manifest(const std::filesystem::path& path)
{
    const auto failure = [&](std::string message)
    {
        return cook_manifest_result::failure(
            {.code = asset_error_code::invalid_metadata, .path = path, .message = std::move(message)});
    };
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return failure("Could not open cook manifest");
    cook_manifest manifest;
    const auto document = json::parse(stream, nullptr, false);
    if (!document.is_object() || document.value("format", "") != "arc.cook-manifest" ||
        document.value("version", 0u) != cook_manifest::current_version ||
        !parse_target(document.value("target", json{}), manifest.target))
    {
        return failure("Cook manifest is invalid or unsupported");
    }
    manifest = {};
    manifest.version = document.value("version", 0u);
    manifest.build_id = document.value("buildId", "");
    parse_target(document["target"], manifest.target);
    const auto parse_guids = [&](std::string_view field, std::vector<asset_guid>& output)
    {
        const std::string key(field);
        if (!document.contains(key) || !document[key].is_array()) return false;
        for (const auto& value : document[key])
        {
            const auto guid = value.is_string() ? parse_asset_guid(value.get<std::string>()) : std::nullopt;
            if (!guid) return false;
            output.push_back(*guid);
        }
        return true;
    };
    if (!parse_guids("roots", manifest.roots) || !parse_guids("dependencyClosure", manifest.dependency_closure) ||
        !document.contains("artifacts") || !document["artifacts"].is_array())
    {
        return failure("Cook manifest contains invalid asset identities");
    }
    for (const auto& value : document["artifacts"])
    {
        const auto asset = parse_asset_guid(value.value("asset", ""));
        const auto type = parse_asset_type_id(value.value("type", ""));
        const auto hash = parse_asset_hash(value.value("hash", ""));
        const auto schema_value = parse_cached_artifact(json{{"name", value.value("name", "")},
                                                             {"extension", ""},
                                                             {"schema", value.value("schema", "")},
                                                             {"schemaVersion", value.value("schemaVersion", 0u)},
                                                             {"hash", value.value("hash", "")},
                                                             {"size", value.value("size", 0ull)}});
        if (!asset || !type || !hash || !schema_value)
        {
            return failure("Cook manifest contains an invalid artifact");
        }
        manifest.artifacts.push_back({.asset = *asset,
                                      .type = *type,
                                      .name = value.value("name", ""),
                                      .schema = schema_value->schema,
                                      .schema_version = value.value("schemaVersion", 0u),
                                      .hash = *hash,
                                      .size = value.value("size", 0ull),
                                      .chunk = value.value("chunk", "startup"),
                                      .offset = value.value("offset", 0ull),
                                      .stored_size = value.value("storedSize", 0ull),
                                      .compressed = value.value("compressed", false)});
    }
    return cook_manifest_result::success(std::move(manifest));
}

asset_status verify_cook_manifest(const cook_manifest& manifest, derived_data_cache& cache)
{
    for (const auto& artifact : manifest.artifacts)
    {
        cache_error error;
        const auto blob = cache.get_blob(artifact.hash, error);
        if (!blob || blob->bytes.size() != artifact.size)
        {
            return asset_status::failure({.code = asset_error_code::not_found,
                                          .guid = artifact.asset,
                                          .message = "Missing or invalid artifact " + to_string(artifact.hash)});
        }
    }
    return asset_status::success();
}

} // namespace arc::assets
