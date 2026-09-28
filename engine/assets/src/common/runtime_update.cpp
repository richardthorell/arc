#include <arc/assets/runtime_content.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace arc::assets
{
namespace
{

using json = nlohmann::json;

asset_status failed(asset_error_code code, std::string message, const std::filesystem::path& path = {})
{
    return asset_status::failure({.code = code, .path = path, .message = std::move(message)});
}

std::filesystem::path blob_path(const std::filesystem::path& root, content_hash hash)
{
    const auto text = to_string(hash);
    return root / "blobs" / "sha256" / text.substr(0, 2) / text;
}

std::optional<artifact_schema_id> parse_schema(std::string_view value)
{
    if (value.size() != 32) return std::nullopt;
    artifact_schema_id result;
    const auto high = std::from_chars(value.data(), value.data() + 16, result.high, 16);
    const auto low = std::from_chars(value.data() + 16, value.data() + 32, result.low, 16);
    if (high.ec != std::errc{} || high.ptr != value.data() + 16 || low.ec != std::errc{} ||
        low.ptr != value.data() + 32 || !result.valid())
        return std::nullopt;
    return result;
}

json manifest_json(const runtime_update_manifest& manifest)
{
    json artifacts = json::array();
    for (const auto& artifact : manifest.artifacts)
        artifacts.push_back({{"asset", to_string(artifact.address.asset)},
                             {"schema", to_string(artifact.address.schema)},
                             {"name", artifact.address.name},
                             {"type", to_string(artifact.type)},
                             {"schemaVersion", artifact.schema_version},
                             {"hash", to_string(artifact.hash)},
                             {"size", artifact.size},
                             {"tombstone", artifact.tombstone}});
    return {{"format", "arc.runtime-update"},
            {"version", manifest.format_version},
            {"targetProfile", manifest.target_profile},
            {"baseBuildId", manifest.base_build_id},
            {"updateSequence", manifest.update_sequence},
            {"artifacts", std::move(artifacts)}};
}

core::result<runtime_update_manifest, asset_error> parse_manifest(const json& document,
                                                                  const std::filesystem::path& source)
{
    const auto reject = [&](std::string message)
    {
        return core::result<runtime_update_manifest, asset_error>::failure(
            {.code = asset_error_code::invalid_metadata, .path = source, .message = std::move(message)});
    };
    if (!document.is_object() || document.value("format", "") != "arc.runtime-update" ||
        document.value("version", 0u) != runtime_update_manifest::current_format_version ||
        !document.contains("artifacts") || !document["artifacts"].is_array())
        return reject("Runtime update manifest is invalid or unsupported");
    runtime_update_manifest result;
    result.format_version = document.value("version", 0u);
    result.target_profile = document.value("targetProfile", "");
    result.base_build_id = document.value("baseBuildId", "");
    result.update_sequence = document.value("updateSequence", 0ull);
    for (const auto& value : document["artifacts"])
    {
        const auto asset = parse_asset_guid(value.value("asset", ""));
        const auto schema = parse_schema(value.value("schema", ""));
        const auto type = parse_asset_type_id(value.value("type", ""));
        const auto hash = parse_asset_hash(value.value("hash", ""));
        const auto name = value.value("name", "");
        if (!asset || !schema || !type || !hash || name.empty())
            return reject("Runtime update manifest contains an invalid artifact identity");
        result.artifacts.push_back({.address = {.asset = *asset, .schema = *schema, .name = name},
                                    .type = *type,
                                    .schema_version = value.value("schemaVersion", 0u),
                                    .hash = *hash,
                                    .size = value.value("size", 0ull),
                                    .tombstone = value.value("tombstone", false)});
    }
    return core::result<runtime_update_manifest, asset_error>::success(std::move(result));
}

std::uint64_t content_generation(const runtime_update_artifact& artifact) noexcept
{
    if (artifact.tombstone) return 1;
    std::uint64_t result{};
    for (std::size_t index = 0; index < sizeof(result); ++index)
        result |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(artifact.hash.bytes[index]))
                  << (index * 8u);
    return result == 0 ? 1 : result;
}

io::file_result<io::file_buffer> read_blob_range(const std::filesystem::path& path, std::uint64_t offset,
                                                 std::size_t bytes, const content_hash& expected_hash,
                                                 std::uint64_t expected_size,
                                                 const jobs::cancellation_token& cancellation)
{
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size != expected_size)
        return io::file_result<io::file_buffer>::failure(
            {.code = io::file_error_code::corrupt_content,
             .path = path,
             .message = "CAS blob size does not match its active manifest"});
    if (offset > size || bytes > size - offset)
        return io::file_result<io::file_buffer>::failure(
            {.code = io::file_error_code::invalid_range, .path = path, .message = "CAS read range is invalid"});
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return io::file_result<io::file_buffer>::failure(
            {.code = io::file_error_code::provider_unavailable, .path = path, .message = "CAS blob is unavailable"});
    stream.seekg(static_cast<std::streamoff>(offset));
    io::file_buffer result(bytes);
    constexpr std::size_t chunk_size = 1024u * 1024u;
    std::size_t completed{};
    while (completed < result.size())
    {
        if (cancellation.stop_requested())
            return io::file_result<io::file_buffer>::failure(
                {.code = io::file_error_code::cancelled, .path = path, .message = "CAS read was cancelled"});
        const auto count = std::min(chunk_size, result.size() - completed);
        stream.read(reinterpret_cast<char*>(result.data() + completed), static_cast<std::streamsize>(count));
        if (stream.gcount() != static_cast<std::streamsize>(count))
            return io::file_result<io::file_buffer>::failure({.code = io::file_error_code::corrupt_content,
                                                              .path = path,
                                                              .message = "CAS blob ended before the requested range"});
        completed += count;
    }
    if (offset == 0 && bytes == expected_size && hash_bytes(result) != expected_hash)
        return io::file_result<io::file_buffer>::failure({.code = io::file_error_code::corrupt_content,
                                                          .path = path,
                                                          .message = "CAS blob failed SHA-256 verification"});
    return io::file_result<io::file_buffer>::success(std::move(result));
}

} // namespace

struct cas_overlay_provider::implementation
{
    struct record
    {
        runtime_update_artifact artifact;
        std::filesystem::path blob;
        std::uint64_t generation{};
    };

    io::async_file_service* files{};
    std::filesystem::path root;
    mutable std::shared_mutex mutex;
    std::unordered_map<std::string, record> records;
    std::vector<io::provider_change> changes;
    runtime_update_manifest active;
    std::uint64_t generation{1};
};

cas_overlay_provider::cas_overlay_provider(io::async_file_service& files, std::filesystem::path cache_root)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->files = &files;
    implementation_->root = std::filesystem::absolute(std::move(cache_root)).lexically_normal();
}

cas_overlay_provider::~cas_overlay_provider() = default;

asset_status cas_overlay_provider::load(std::string_view expected_target_profile,
                                        std::string_view expected_base_build_id)
{
    const auto path = implementation_->root / "overlay-manifest.json";
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        std::error_code error;
        if (!std::filesystem::exists(path, error)) return asset_status::success();
        return failed(asset_error_code::io_failed, "Could not open persisted overlay manifest", path);
    }
    auto parsed = parse_manifest(json::parse(input, nullptr, false), path);
    if (!parsed) return asset_status::failure(std::move(parsed).error());
    if (parsed.value().target_profile != expected_target_profile ||
        parsed.value().base_build_id != expected_base_build_id)
        return failed(asset_error_code::invalid_metadata, "Persisted overlay targets a different profile or base build",
                      path);
    for (const auto& artifact : parsed.value().artifacts)
    {
        if (artifact.tombstone) continue;
        const auto blob = blob_path(implementation_->root, artifact.hash);
        std::error_code error;
        if (std::filesystem::file_size(blob, error) != artifact.size || error)
            return failed(asset_error_code::invalid_metadata, "Persisted overlay references an incomplete blob", blob);
        auto hashed = hash_file(blob);
        if (!hashed || hashed.value() != artifact.hash)
            return failed(asset_error_code::invalid_metadata, "Persisted overlay references a corrupt blob", blob);
    }
    return activate(parsed.value());
}

asset_status cas_overlay_provider::activate(const runtime_update_manifest& manifest)
{
    std::unordered_map<std::string, implementation::record> staged;
    for (const auto& artifact : manifest.artifacts)
    {
        auto logical = cooked_artifact_virtual_path(artifact.address);
        if (!logical)
            return failed(asset_error_code::invalid_metadata, "Runtime update contains an invalid artifact path");
        const auto key = std::string(logical.value().relative_path());
        if (staged.contains(key))
            return failed(asset_error_code::invalid_metadata, "Runtime update contains duplicate artifact addresses");
        staged.emplace(key, implementation::record{.artifact = artifact,
                                                   .blob = artifact.tombstone
                                                               ? std::filesystem::path{}
                                                               : blob_path(implementation_->root, artifact.hash),
                                                   .generation = content_generation(artifact)});
    }

    std::unique_lock lock(implementation_->mutex);
    std::vector<io::provider_change> changes;
    for (const auto& [path, record] : staged)
    {
        const auto previous = implementation_->records.find(path);
        if (record.artifact.tombstone)
            changes.push_back({.kind = io::provider_change_kind::removed,
                               .relative_path = path,
                               .content_generation = record.generation});
        else if (previous == implementation_->records.end() || previous->second.artifact.tombstone)
            changes.push_back({.kind = io::provider_change_kind::added,
                               .relative_path = path,
                               .content_generation = record.generation});
        else if (previous->second.generation != record.generation)
            changes.push_back({.kind = io::provider_change_kind::modified,
                               .relative_path = path,
                               .content_generation = record.generation});
    }
    for (const auto& [path, record] : implementation_->records)
    {
        static_cast<void>(record);
        if (!staged.contains(path))
            changes.push_back({.kind = io::provider_change_kind::removed, .relative_path = path});
    }
    std::sort(changes.begin(), changes.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.relative_path < rhs.relative_path; });
    implementation_->records = std::move(staged);
    implementation_->active = manifest;
    ++implementation_->generation;
    implementation_->changes.insert(implementation_->changes.end(), std::make_move_iterator(changes.begin()),
                                    std::make_move_iterator(changes.end()));
    return asset_status::success();
}

io::provider_capabilities cas_overlay_provider::capabilities() const noexcept
{
    return io::provider_capability::full_read | io::provider_capability::range_read |
           io::provider_capability::metadata | io::provider_capability::enumeration |
           io::provider_capability::change_polling;
}

std::uint64_t cas_overlay_provider::provider_generation() const noexcept
{
    std::shared_lock lock(implementation_->mutex);
    return implementation_->generation;
}

io::provider_lookup_result cas_overlay_provider::resolve(std::string_view relative_path)
{
    std::shared_lock lock(implementation_->mutex);
    const auto found = implementation_->records.find(std::string(relative_path));
    if (found == implementation_->records.end()) return {};
    if (found->second.artifact.tombstone) return {.status = io::provider_lookup_status::tombstone};
    return {.status = io::provider_lookup_status::found,
            .file = {.key = found->first,
                     .size = found->second.artifact.size,
                     .content_generation = found->second.generation}};
}

jobs::job_future<io::file_result<io::file_buffer>>
cas_overlay_provider::read_range(const io::provider_file& file, std::uint64_t offset, std::size_t bytes,
                                 jobs::cancellation_token cancellation)
{
    return implementation_->files->scheduler().submit_future(
        {.name = "assets.cas_overlay.read",
         .priority = jobs::job_priority::normal,
         .affinity = jobs::job_affinity::io_thread,
         .cancellation = cancellation,
         .dependency_policy = jobs::job_dependency_policy::cancel_on_failure},
        [state = implementation_.get(), file, offset, bytes, cancellation]
        {
            runtime_update_artifact artifact;
            std::filesystem::path blob;
            {
                std::shared_lock lock(state->mutex);
                const auto found = state->records.find(file.key);
                if (found == state->records.end() || found->second.artifact.tombstone ||
                    found->second.generation != file.content_generation)
                    return io::file_result<io::file_buffer>::failure({.code = io::file_error_code::stale_handle,
                                                                      .message = "CAS overlay handle was superseded",
                                                                      .logical_path = file.key});
                artifact = found->second.artifact;
                blob = found->second.blob;
            }
            auto result = read_blob_range(blob, offset, bytes, artifact.hash, artifact.size, cancellation);
            if (!result) return result;
            {
                std::shared_lock lock(state->mutex);
                const auto found = state->records.find(file.key);
                if (found == state->records.end() || found->second.generation != file.content_generation)
                    return io::file_result<io::file_buffer>::failure(
                        {.code = io::file_error_code::stale_handle,
                         .message = "CAS overlay changed while a read was in flight",
                         .logical_path = file.key});
            }
            return result;
        });
}

io::file_result<std::vector<io::provider_directory_entry>>
cas_overlay_provider::enumerate(std::string_view relative_prefix)
{
    std::shared_lock lock(implementation_->mutex);
    std::vector<io::provider_directory_entry> result;
    for (const auto& [path, record] : implementation_->records)
    {
        if (!relative_prefix.empty() && !path.starts_with(relative_prefix)) continue;
        result.push_back({.relative_path = path,
                          .size = record.artifact.size,
                          .content_generation = record.generation,
                          .tombstone = record.artifact.tombstone});
    }
    std::sort(result.begin(), result.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.relative_path < rhs.relative_path; });
    return io::file_result<std::vector<io::provider_directory_entry>>::success(std::move(result));
}

std::vector<io::provider_change> cas_overlay_provider::poll_changes()
{
    std::unique_lock lock(implementation_->mutex);
    auto result = std::move(implementation_->changes);
    implementation_->changes.clear();
    return result;
}

const std::filesystem::path& cas_overlay_provider::cache_root() const noexcept
{
    return implementation_->root;
}

std::uint64_t cas_overlay_provider::active_sequence() const noexcept
{
    std::shared_lock lock(implementation_->mutex);
    return implementation_->active.update_sequence;
}

struct runtime_update_receiver::implementation
{
    std::shared_ptr<cas_overlay_provider> overlay;
    runtime_update_receiver_config config;
    std::optional<runtime_update_manifest> pending;
    std::filesystem::path staging_root;
    std::optional<content_hash> blob_hash;
    std::uint64_t blob_expected_size{};
    std::uint64_t blob_written{};
    std::ofstream blob_stream;
    std::unordered_set<std::string> verified;
    bool verification_complete{};
    runtime_update_telemetry telemetry;
};

runtime_update_receiver::runtime_update_receiver(std::shared_ptr<cas_overlay_provider> overlay,
                                                 runtime_update_receiver_config config)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->overlay = std::move(overlay);
    implementation_->config = std::move(config);
    implementation_->telemetry.active_sequence = implementation_->overlay->active_sequence();
    std::error_code error;
    std::filesystem::remove_all(implementation_->config.cache_root / "staging", error);
}

runtime_update_receiver::~runtime_update_receiver()
{
    abort();
}

asset_status runtime_update_receiver::begin(runtime_update_manifest manifest)
{
    abort();
    const auto reject = [&](std::string message)
    {
        ++implementation_->telemetry.rejected_updates;
        return failed(asset_error_code::invalid_request, std::move(message));
    };
    if (manifest.format_version != runtime_update_manifest::current_format_version)
        return reject("Runtime update format is unsupported");
    if (manifest.target_profile != implementation_->config.target_profile ||
        manifest.base_build_id != implementation_->config.base_build_id)
        return reject("Runtime update targets a different profile or base build");
    if (manifest.update_sequence <= implementation_->overlay->active_sequence())
        return reject("Runtime update sequence is stale");
    std::unordered_set<std::string> addresses;
    for (const auto& artifact : manifest.artifacts)
    {
        auto path = cooked_artifact_virtual_path(artifact.address);
        if (!path || !addresses.insert(std::string(path.value().string())).second)
            return reject("Runtime update contains an invalid or duplicate artifact address");
        if (!artifact.tombstone && (artifact.hash.empty() || artifact.size == 0))
            return reject("Runtime update contains an incomplete artifact record");
    }
    implementation_->staging_root =
        implementation_->config.cache_root / "staging" / std::to_string(manifest.update_sequence);
    std::error_code error;
    std::filesystem::create_directories(implementation_->staging_root, error);
    if (error) return reject("Runtime update staging directory could not be created");
    implementation_->pending = std::move(manifest);
    implementation_->verification_complete = false;
    return asset_status::success();
}

asset_status runtime_update_receiver::begin_blob(content_hash hash, std::uint64_t size)
{
    if (!implementation_->pending || implementation_->blob_stream.is_open())
        return failed(asset_error_code::invalid_request, "No update is ready for a new blob");
    const auto expected = std::find_if(
        implementation_->pending->artifacts.begin(), implementation_->pending->artifacts.end(),
        [&](const auto& artifact) { return !artifact.tombstone && artifact.hash == hash && artifact.size == size; });
    if (expected == implementation_->pending->artifacts.end())
        return failed(asset_error_code::invalid_request, "Blob is not declared by the pending update");
    const auto text = to_string(hash);
    const auto destination = blob_path(implementation_->config.cache_root, hash);
    std::error_code error;
    if (std::filesystem::is_regular_file(destination, error) && !error)
    {
        auto hashed = hash_file(destination);
        if (hashed && hashed.value() == hash && std::filesystem::file_size(destination, error) == size && !error)
        {
            implementation_->verified.insert(text);
            implementation_->blob_hash = hash;
            implementation_->blob_expected_size = size;
            implementation_->blob_written = size;
            ++implementation_->telemetry.reused_blobs;
            return asset_status::success();
        }
    }
    const auto temporary = implementation_->staging_root / (text + ".part");
    implementation_->blob_stream.open(temporary, std::ios::binary | std::ios::trunc);
    if (!implementation_->blob_stream)
        return failed(asset_error_code::io_failed, "Could not create staged update blob", temporary);
    implementation_->blob_hash = hash;
    implementation_->blob_expected_size = size;
    implementation_->blob_written = 0;
    return asset_status::success();
}

asset_status runtime_update_receiver::stage_blob(std::span<const std::byte> bytes)
{
    if (!implementation_->blob_hash) return failed(asset_error_code::invalid_request, "No update blob is active");
    if (!implementation_->blob_stream.is_open())
        return bytes.empty() ? asset_status::success()
                             : failed(asset_error_code::invalid_request, "Existing CAS blob requires no payload");
    if (bytes.size() > implementation_->blob_expected_size - implementation_->blob_written)
        return failed(asset_error_code::invalid_request, "Staged blob exceeds its declared size");
    implementation_->blob_stream.write(reinterpret_cast<const char*>(bytes.data()),
                                       static_cast<std::streamsize>(bytes.size()));
    if (!implementation_->blob_stream) return failed(asset_error_code::io_failed, "Could not write staged update blob");
    implementation_->blob_written += bytes.size();
    implementation_->telemetry.staged_bytes += bytes.size();
    return asset_status::success();
}

asset_status runtime_update_receiver::finish_blob()
{
    if (!implementation_->blob_hash) return failed(asset_error_code::invalid_request, "No update blob is active");
    const auto hash = *implementation_->blob_hash;
    const auto text = to_string(hash);
    if (implementation_->blob_stream.is_open())
    {
        implementation_->blob_stream.flush();
        implementation_->blob_stream.close();
        const auto temporary = implementation_->staging_root / (text + ".part");
        if (implementation_->blob_written != implementation_->blob_expected_size)
            return failed(asset_error_code::invalid_metadata, "Staged blob is truncated", temporary);
        auto hashed = hash_file(temporary);
        if (!hashed || hashed.value() != hash)
            return failed(asset_error_code::invalid_metadata, "Staged blob failed SHA-256 verification", temporary);
        const auto destination = blob_path(implementation_->config.cache_root, hash);
        std::error_code error;
        std::filesystem::create_directories(destination.parent_path(), error);
        if (error) return failed(asset_error_code::io_failed, "Could not create CAS blob directory", destination);
        if (std::filesystem::exists(destination, error))
            std::filesystem::remove(temporary, error);
        else
            std::filesystem::rename(temporary, destination, error);
        if (error) return failed(asset_error_code::io_failed, "Could not publish CAS blob", destination);
        implementation_->verified.insert(text);
        ++implementation_->telemetry.staged_blobs;
    }
    implementation_->blob_hash.reset();
    implementation_->blob_expected_size = 0;
    implementation_->blob_written = 0;
    return asset_status::success();
}

asset_status runtime_update_receiver::finish_verify()
{
    if (!implementation_->pending || implementation_->blob_stream.is_open())
        return failed(asset_error_code::invalid_request, "Runtime update is not ready for verification");
    for (const auto& artifact : implementation_->pending->artifacts)
    {
        if (artifact.tombstone) continue;
        const auto destination = blob_path(implementation_->config.cache_root, artifact.hash);
        std::error_code error;
        if (std::filesystem::file_size(destination, error) != artifact.size || error)
            return failed(asset_error_code::invalid_metadata, "Runtime update is missing a complete blob", destination);
        auto hashed = hash_file(destination);
        if (!hashed || hashed.value() != artifact.hash)
            return failed(asset_error_code::invalid_metadata, "Runtime update contains a corrupt blob", destination);
        implementation_->verified.insert(to_string(artifact.hash));
    }
    implementation_->verification_complete = true;
    return asset_status::success();
}

asset_status runtime_update_receiver::commit()
{
    if (!implementation_->pending || !implementation_->verification_complete)
        return failed(asset_error_code::invalid_request, "Runtime update must be verified before commit");
    const auto text = manifest_json(*implementation_->pending).dump(2) + '\n';
    const auto byte_view = std::as_bytes(std::span(text.data(), text.size()));
    const auto manifest_path = implementation_->config.cache_root / "overlay-manifest.json";
    auto written = implementation_->overlay->implementation_->files->write_atomic(manifest_path, byte_view).get();
    if (!written)
    {
        ++implementation_->telemetry.rejected_updates;
        return failed(asset_error_code::io_failed, "Could not atomically publish runtime update manifest",
                      manifest_path);
    }
    auto activated = implementation_->overlay->activate(*implementation_->pending);
    if (!activated)
    {
        ++implementation_->telemetry.rejected_updates;
        return activated;
    }
    ++implementation_->telemetry.accepted_updates;
    implementation_->telemetry.active_sequence = implementation_->pending->update_sequence;
    implementation_->telemetry.active_artifacts = implementation_->pending->artifacts.size();
    std::error_code error;
    std::filesystem::remove_all(implementation_->staging_root, error);
    implementation_->pending.reset();
    implementation_->verification_complete = false;
    return asset_status::success();
}

void runtime_update_receiver::abort() noexcept
{
    if (!implementation_) return;
    if (implementation_->blob_stream.is_open()) implementation_->blob_stream.close();
    std::error_code error;
    if (!implementation_->staging_root.empty()) std::filesystem::remove_all(implementation_->staging_root, error);
    implementation_->pending.reset();
    implementation_->staging_root.clear();
    implementation_->blob_hash.reset();
    implementation_->blob_expected_size = 0;
    implementation_->blob_written = 0;
    implementation_->verified.clear();
    implementation_->verification_complete = false;
}

asset_status runtime_update_receiver::ingest(const runtime_update_manifest& manifest,
                                             std::span<const std::pair<content_hash, io::file_buffer>> blobs)
{
    auto started = begin(manifest);
    if (!started) return started;
    for (const auto& [hash, bytes] : blobs)
    {
        auto opened = begin_blob(hash, bytes.size());
        if (!opened)
        {
            ++implementation_->telemetry.rejected_updates;
            abort();
            return opened;
        }
        auto staged = stage_blob(bytes);
        if (!staged)
        {
            ++implementation_->telemetry.rejected_updates;
            abort();
            return staged;
        }
        auto finished = finish_blob();
        if (!finished)
        {
            ++implementation_->telemetry.rejected_updates;
            abort();
            return finished;
        }
    }
    auto verified = finish_verify();
    if (!verified)
    {
        ++implementation_->telemetry.rejected_updates;
        abort();
        return verified;
    }
    return commit();
}

runtime_update_telemetry runtime_update_receiver::telemetry() const noexcept
{
    return implementation_->telemetry;
}

} // namespace arc::assets
