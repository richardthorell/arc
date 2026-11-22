#include <arc/assets/remote_content_generation.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <exception>
#include <optional>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace arc::assets
{
namespace
{

asset_error invalid(std::string message)
{
    return {.code = asset_error_code::invalid_metadata, .message = std::move(message)};
}

bool safe_filename(std::string_view name, std::string_view extension)
{
    if (name.empty() || name.size() > 255 || !name.ends_with(extension) ||
        !std::isalnum(static_cast<unsigned char>(name.front())))
        return false;
    return std::all_of(name.begin(), name.end(),
                       [](unsigned char c) { return std::isalnum(c) || c == '.' || c == '-' || c == '_'; });
}

std::string fold_name(std::string_view name)
{
    std::string folded(name);
    std::transform(folded.begin(), folded.end(), folded.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return folded;
}

bool matching_target(const cook_target& target, const remote_content_release& release)
{
    const auto platform = [value = target.platform]() -> std::string_view
    {
        switch (value)
        {
            case cook_platform::windows:
                return "windows";
            case cook_platform::linux_os:
                return "linux";
            case cook_platform::macos:
                return "macos";
            case cook_platform::android:
                return "android";
        }
        return {};
    }();
    const auto renderer = [value = target.renderer]() -> std::string_view
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
        return {};
    }();
    const auto textures = [value = target.textures]() -> std::string_view
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
        return {};
    }();
    return target.name == release.target.cook_profile && platform == release.target.platform &&
           (target.architecture == cook_architecture::arm64 ? "arm64" : "x86_64") == release.target.architecture &&
           renderer == release.target.renderer && textures == release.target.texture_family;
}

std::uint64_t artifact_generation(const cook_manifest_artifact& artifact, std::uint64_t release_generation) noexcept
{
    std::uint64_t prefix{};
    for (std::size_t index = 0; index < 8; ++index)
        prefix |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(artifact.hash.bytes[index]))
                  << (8u * index);
    const auto generation = prefix ^ release_generation ^ artifact.offset ^ artifact.stored_size;
    return generation == 0 ? 1 : generation;
}

io::file_result<io::file_buffer> read_failure(io::file_error_code code, std::string key, std::string message)
{
    return io::file_result<io::file_buffer>::failure(
        {.code = code, .message = std::move(message), .logical_path = std::move(key)});
}

} // namespace

struct remote_content_artifact_provider::implementation
{
    struct record
    {
        cook_manifest_artifact artifact;
        remote_content_object chunk;
        std::uint64_t generation{};
    };
    derived_data_cache* cache{};
    io::async_file_service* files{};
    std::optional<lazy_remote_content_options> lazy;
    remote_generation_fetch_executor executor;
    std::uint64_t generation{};
    cook_manifest manifest;
    std::unordered_map<std::string, record> records;
};

remote_content_artifact_provider::remote_content_artifact_provider(std::shared_ptr<implementation> state)
    : implementation_(std::move(state))
{
}

remote_content_artifact_provider::~remote_content_artifact_provider() = default;

remote_content_artifact_provider::create_result
remote_content_artifact_provider::create(const remote_content_release& release, derived_data_cache& cache,
                                         io::async_file_service& files)
{
    return create_impl(release, cache, files, std::nullopt, {});
}

remote_content_artifact_provider::create_result remote_content_artifact_provider::create_lazy(
    const remote_content_release& release, derived_data_cache& cache, io::async_file_service& files,
    lazy_remote_content_options options, remote_generation_fetch_executor executor)
{
    return create_impl(release, cache, files, std::move(options), std::move(executor));
}

remote_content_artifact_provider::create_result remote_content_artifact_provider::create_impl(
    const remote_content_release& release, derived_data_cache& cache, io::async_file_service& files,
    std::optional<lazy_remote_content_options> lazy, remote_generation_fetch_executor executor)
{
    const auto reject = [](std::string message) -> create_result
    { return create_result::failure(invalid(std::move(message))); };

    if (release.generation == 0 || release.base_build_id.empty() ||
        !safe_filename(release.cook_manifest_name, ".arccookmanifest"))
        return reject("Remote generation identity or cook manifest filename is invalid");

    cache_error cache_error_value;
    auto manifest_bytes = load_verified_remote_object(cache, release.cook_manifest, cache_error_value);
    if (!manifest_bytes)
        return reject("Remote generation cook manifest is missing or corrupt: " + cache_error_value.message);
    if (manifest_bytes->size() > 8u * 1024u * 1024u)
        return reject("Remote generation cook manifest exceeds metadata limits");

    cook_manifest manifest;
    try
    {
        auto parsed = parse_cook_manifest(*manifest_bytes, release.cook_manifest_name);
        if (!parsed) return create_result::failure(std::move(parsed).error());
        manifest = std::move(parsed).value();
    }
    catch (const std::exception&)
    {
        return reject("Remote cook manifest contains invalid field types");
    }

    // C9.1 still validates the packaged base runtime build independently.
    const auto& cook_build_id = release.cook_build_id.empty() ? release.base_build_id : release.cook_build_id;
    if (manifest.build_id != cook_build_id || !matching_target(manifest.target, release) ||
        release.cook_manifest_name != manifest.target.name + ".arccookmanifest")
        return reject("Remote cook manifest build, target or filename does not match release");

    std::unordered_map<std::string, const remote_content_chunk*> chunks;
    std::unordered_set<std::string> folded_names;
    std::unordered_set<std::string> object_hashes{release.cook_manifest.sha256};
    chunks.reserve(release.chunks.size());
    // A lazy release publishes metadata before content bytes arrive, but
    // never unsafe addresses, oversized objects or unbounded work.
    if (lazy)
    {
        if (lazy->maximum_chunks == 0 || release.chunks.size() >= lazy->maximum_chunks ||
            lazy->maximum_generation_bytes == 0 || release.cook_manifest.size > 8u * 1024u * 1024u ||
            release.cook_manifest.size > lazy->http.maximum_object_bytes ||
            release.cook_manifest.size > lazy->maximum_generation_bytes)
            return reject("Lazy remote generation exceeds configured manifest, object or chunk limits");
        if (!lazy->offline_only)
        {
            const http::content_http_request probe{
                .origin = lazy->origin, .object_path = release.cook_manifest.path,
                .object_size = release.cook_manifest.size};
            if (!http::validate_content_http_request(probe, lazy->http).succeeded())
                return reject("Lazy remote generation origin or manifest request is invalid");
        }
    }
    std::uint64_t declared_bytes = release.cook_manifest.size;
    for (const auto& entry : release.chunks)
    {
        if (!safe_filename(entry.name, ".arcpak") || !folded_names.insert(fold_name(entry.name)).second ||
            !object_hashes.insert(entry.object.sha256).second)
            return reject("Remote release has an invalid or duplicate package chunk");
        if (lazy)
        {
            const auto hash = parse_asset_hash(entry.object.sha256);
            if (!hash || to_string(*hash) != entry.object.sha256 ||
                entry.object.path != "objects/sha256/" + entry.object.sha256 ||
                entry.object.size < 8 || entry.object.size > lazy->http.maximum_object_bytes ||
                entry.object.size > lazy->maximum_generation_bytes - declared_bytes)
                return reject("Lazy remote release declares an invalid or oversized package chunk");
            declared_bytes += entry.object.size;
            if (!lazy->offline_only)
            {
                const http::content_http_request probe{
                    .origin = lazy->origin, .object_path = entry.object.path, .object_size = entry.object.size};
                if (!http::validate_content_http_request(probe, lazy->http).succeeded())
                    return reject("Lazy remote release chunk origin or object path is invalid");
            }
        }
        chunks.emplace(entry.name, &entry);
    }

    std::unordered_map<std::string, std::vector<const cook_manifest_artifact*>> by_chunk;
    std::unordered_set<std::string> artifact_keys;
    for (const auto& artifact : manifest.artifacts)
    {
        const cooked_artifact_address address{
            .asset = artifact.asset, .schema = artifact.schema, .name = artifact.name};
        auto logical = cooked_artifact_virtual_path(address);
        if (!logical || artifact.schema_version == 0 || artifact.size == 0 || artifact.stored_size != artifact.size ||
            artifact.compressed || !artifact_keys.insert(std::string(logical.value().relative_path())).second)
            return reject("Remote cook manifest has a duplicate or unsupported artifact record");
        if (!chunks.contains(artifact.chunk))
            return reject("Remote cook manifest references an undeclared package chunk: " + artifact.chunk);
        by_chunk[artifact.chunk].push_back(&artifact);
    }
    if (by_chunk.size() != chunks.size()) return reject("Remote release has missing or unreferenced package chunks");

    auto state = std::make_shared<implementation>();
    state->cache = &cache;
    state->files = &files;
    state->lazy = std::move(lazy);
    state->executor = std::move(executor);
    state->generation = release.generation;

    // Validate every immutable chunk and every addressed artifact *before*
    // publishing any provider. All checks use the existing cooker offsets.
    for (const auto& entry : release.chunks)
    {
        // Strict mode checks cached bytes now; lazy mode checks declared
        // lengths and validates immutable chunk/artifact hashes on first read.
        std::optional<io::file_buffer> bytes;
        if (!state->lazy)
        {
            bytes = load_verified_remote_object(cache, entry.object, cache_error_value);
            if (!bytes)
                return reject("Remote package chunk is missing or corrupt (" + entry.name +
                              "): " + cache_error_value.message);
            constexpr char magic[] = "ARCPAK1";
            if (bytes->size() < 8 || std::memcmp(bytes->data(), magic, 7) != 0 || (*bytes)[7] != std::byte{0})
                return reject("Remote package chunk has an invalid ARC package header: " + entry.name);
        }
        const auto chunk_size = bytes ? bytes->size() : entry.object.size;

        const auto group = by_chunk.find(entry.name);
        if (group == by_chunk.end())
            return reject("Remote package chunk has no cook manifest references: " + entry.name);

        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        ranges.reserve(group->second.size());
        for (const auto* artifact : group->second)
        {
            if (artifact->offset < 8 || artifact->offset > chunk_size ||
                artifact->stored_size > chunk_size - artifact->offset)
                return reject("Remote cook artifact range exceeds package chunk: " + entry.name);
            if (bytes)
            {
                const auto begin = static_cast<std::size_t>(artifact->offset);
                const auto size = static_cast<std::size_t>(artifact->stored_size);
                if (hash_bytes(std::span<const std::byte>(bytes->data() + begin, size)) != artifact->hash)
                    return reject("Remote cook artifact bytes do not match manifest hash: " + entry.name);
            }
            ranges.emplace_back(artifact->offset, artifact->offset + artifact->stored_size);

            auto logical = cooked_artifact_virtual_path(
                {.asset = artifact->asset, .schema = artifact->schema, .name = artifact->name});
            const auto key = std::string(logical.value().relative_path());
            state->records.emplace(
                key, implementation::record{.artifact = *artifact,
                                            .chunk = entry.object,
                                            .generation = artifact_generation(*artifact, release.generation)});
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t index = 1; index < ranges.size(); ++index)
            if (ranges[index].first < ranges[index - 1].second)
                return reject("Remote cook manifest contains overlapping artifact ranges: " + entry.name);
    }

    state->manifest = std::move(manifest);
    return create_result::success(
        std::shared_ptr<remote_content_artifact_provider>(new remote_content_artifact_provider(std::move(state))));
}

io::provider_capabilities remote_content_artifact_provider::capabilities() const noexcept
{
    return io::provider_capability::full_read | io::provider_capability::range_read |
           io::provider_capability::metadata | io::provider_capability::enumeration;
}

std::uint64_t remote_content_artifact_provider::provider_generation() const noexcept
{
    return implementation_->generation;
}

io::provider_lookup_result remote_content_artifact_provider::resolve(std::string_view relative_path)
{
    const auto found = implementation_->records.find(std::string(relative_path));
    if (found == implementation_->records.end()) return {};
    return {.status = io::provider_lookup_status::found,
            .file = {.key = found->first,
                     .size = found->second.artifact.stored_size,
                     .content_generation = found->second.generation}};
}

jobs::job_future<io::file_result<io::file_buffer>>
remote_content_artifact_provider::read_range(const io::provider_file& file, std::uint64_t offset, std::size_t bytes,
                                             jobs::cancellation_token cancellation)
{
    return implementation_->files->scheduler().submit_future(
        {.name = "assets.remote_content.read",
         .priority = jobs::job_priority::normal,
         // Coordinate the IO-worker HTTP future from a general worker.
         .affinity = implementation_->lazy ? jobs::job_affinity::any_worker : jobs::job_affinity::io_thread},
        [state = implementation_, file, offset, bytes, cancellation]() -> io::file_result<io::file_buffer>
        {
            if (cancellation.stop_requested())
                return read_failure(io::file_error_code::cancelled, file.key, "Remote cooked artifact read cancelled");
            const auto found = state->records.find(file.key);
            if (found == state->records.end() || found->second.generation != file.content_generation ||
                found->second.artifact.stored_size != file.size)
                return read_failure(io::file_error_code::stale_handle, file.key,
                                    "Remote cooked artifact handle is stale");
            const auto& artifact = found->second.artifact;
            if (offset > artifact.stored_size || bytes > artifact.stored_size - offset)
                return read_failure(io::file_error_code::invalid_request, file.key,
                                    "Remote cooked artifact range exceeds declared size");

            cache_error error;
            auto chunk = load_verified_remote_object(*state->cache, found->second.chunk, error);
            if (!chunk && state->lazy)
            {
                if (state->lazy->offline_only)
                    return read_failure(io::file_error_code::not_found, file.key,
                                        "Missing immutable package chunk in offline cache");
                const http::content_http_request request{
                    .origin = state->lazy->origin, .object_path = found->second.chunk.path,
                    .object_size = found->second.chunk.size};
                const auto valid = http::validate_content_http_request(request, state->lazy->http);
                if (!valid.succeeded())
                    return read_failure(io::file_error_code::invalid_request, file.key, valid.message);
                auto future = state->executor
                                  ? state->executor(state->files->scheduler(), request, state->lazy->http, cancellation)
                                  : http::fetch_content_http(state->files->scheduler(), request, state->lazy->http,
                                                             cancellation);
                if (!future.valid())
                    return read_failure(io::file_error_code::provider_unavailable, file.key,
                                        "Remote HTTP fetch executor unavailable");
                const auto waited = future.handle().wait_result();
                if (!waited.succeeded())
                    return read_failure(cancellation.stop_requested() ? io::file_error_code::cancelled
                                                                      : io::file_error_code::read_failed,
                                        file.key, "Remote immutable chunk HTTP fetch was cancelled or failed");
                http::content_http_fetch_result response;
                try
                {
                    response = future.get();
                }
                catch (const std::exception&)
                {
                    return read_failure(io::file_error_code::read_failed, file.key,
                                        "Remote immutable chunk HTTP worker failed");
                }
                if (!response.succeeded())
                    return read_failure(response.validation.code == http::content_http_error_code::cancelled
                                            ? io::file_error_code::cancelled
                                            : io::file_error_code::read_failed,
                                        file.key, "Remote immutable chunk fetch: " + response.validation.message);
                if (cancellation.stop_requested())
                    return read_failure(io::file_error_code::cancelled, file.key,
                                        "Remote cooked artifact read cancelled");
                const auto ingested = ingest_verified_remote_object(*state->cache, found->second.chunk, request,
                                                                    *response.response, state->lazy->http);
                if (!ingested.succeeded())
                    return read_failure(io::file_error_code::corrupt_content, file.key,
                                        "Remote immutable chunk failed CAS verification: " + ingested.message);
                chunk = load_verified_remote_object(*state->cache, found->second.chunk, error);
            }
            if (!chunk)
                return read_failure(io::file_error_code::read_failed, file.key,
                                    "Remote cooked package chunk failed CAS verification: " + error.message);
            if (cancellation.stop_requested())
                return read_failure(io::file_error_code::cancelled, file.key, "Remote cooked artifact read cancelled");
            if (state->lazy)
            {
                constexpr char magic[] = "ARCPAK1";
                if (chunk->size() < 8 || std::memcmp(chunk->data(), magic, 7) != 0 ||
                    (*chunk)[7] != std::byte{0} || artifact.offset > chunk->size() ||
                    artifact.stored_size > chunk->size() - artifact.offset)
                    return read_failure(io::file_error_code::corrupt_content, file.key,
                                        "Remote lazy chunk header or artifact range is corrupt");
                const auto begin = static_cast<std::size_t>(artifact.offset);
                if (hash_bytes(std::span<const std::byte>(chunk->data() + begin,
                                                          static_cast<std::size_t>(artifact.stored_size))) != artifact.hash)
                    return read_failure(io::file_error_code::corrupt_content, file.key,
                                        "Remote lazy cooked artifact hash differs from verified cook manifest");
            }
            const auto start = static_cast<std::size_t>(artifact.offset + offset);
            if (start > chunk->size() || bytes > chunk->size() - start)
                return read_failure(io::file_error_code::read_failed, file.key,
                                    "Remote cooked package chunk was truncated after validation");
            return io::file_result<io::file_buffer>::success(
                io::file_buffer(chunk->begin() + static_cast<std::ptrdiff_t>(start),
                                chunk->begin() + static_cast<std::ptrdiff_t>(start + bytes)));
        });
}

io::file_result<std::vector<io::provider_directory_entry>>
remote_content_artifact_provider::enumerate(std::string_view relative_prefix)
{
    std::vector<io::provider_directory_entry> entries;
    for (const auto& [key, record] : implementation_->records)
    {
        if (!relative_prefix.empty() && !key.starts_with(relative_prefix)) continue;
        entries.push_back(
            {.relative_path = key, .size = record.artifact.stored_size, .content_generation = record.generation});
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.relative_path < rhs.relative_path; });
    return io::file_result<std::vector<io::provider_directory_entry>>::success(std::move(entries));
}

const cook_manifest& remote_content_artifact_provider::manifest() const noexcept
{
    return implementation_->manifest;
}

} // namespace arc::assets
