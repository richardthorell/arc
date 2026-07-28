#include <arc/assets/cook.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_set>

namespace arc::assets
{
namespace
{

using json = nlohmann::json;

std::string id_string(std::uint64_t high, std::uint64_t low)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << high << std::setw(16) << low;
    return stream.str();
}

void append(std::vector<std::byte>& output, std::string_view value)
{
    const auto size = static_cast<std::uint64_t>(value.size());
    for (std::size_t index = 0; index < sizeof(size); ++index)
        output.push_back(static_cast<std::byte>((size >> (index * 8u)) & 0xffu));
    output.insert(output.end(),
        reinterpret_cast<const std::byte*>(value.data()),
        reinterpret_cast<const std::byte*>(value.data() + value.size()));
}

void append(std::vector<std::byte>& output, const asset_hash& value)
{
    output.insert(output.end(), value.bytes.begin(), value.bytes.end());
}

std::filesystem::path blob_path(const std::filesystem::path& root, asset_hash hash)
{
    const auto text = to_string(hash);
    return root / "cas" / "sha256" / text.substr(0, 2) / text;
}

std::filesystem::path action_path(const std::filesystem::path& root, asset_hash key)
{
    const auto text = to_string(key);
    return root / "actions" / text.substr(0, 2) / (text + ".json");
}

bool write_atomic(
    const std::filesystem::path& destination,
    std::span<const std::byte> bytes,
    std::string& error)
{
    std::error_code filesystem_error;
    std::filesystem::create_directories(destination.parent_path(), filesystem_error);
    if (filesystem_error)
    {
        error = "Could not create cache directory: " + filesystem_error.message();
        return false;
    }
    if (std::filesystem::exists(destination, filesystem_error) && !filesystem_error)
        return true;

    const auto temporary = destination.string() + ".tmp-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream)
        {
            error = "Could not create temporary cache file";
            return false;
        }
        stream.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        stream.flush();
        if (!stream)
        {
            error = "Could not flush temporary cache file";
            std::filesystem::remove(temporary, filesystem_error);
            return false;
        }
    }
    std::filesystem::rename(temporary, destination, filesystem_error);
    if (!filesystem_error)
        return true;
    filesystem_error.clear();
    if (std::filesystem::exists(destination, filesystem_error) && !filesystem_error)
    {
        std::filesystem::remove(temporary, filesystem_error);
        return true;
    }
    std::filesystem::remove(temporary, filesystem_error);
    error = "Could not atomically publish cache file";
    return false;
}

std::optional<std::vector<std::byte>> read_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return std::nullopt;
    const auto end = stream.tellg();
    if (end < 0)
        return std::nullopt;
    std::vector<std::byte> result(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!result.empty())
        stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
    if (!stream)
        return std::nullopt;
    return result;
}

json action_json(const cache_action& action)
{
    json artifacts = json::array();
    for (const auto& hash : action.artifacts)
        artifacts.push_back(to_string(hash));
    return {
        { "format", "arc.ddc-action" },
        { "version", 1 },
        { "key", to_string(action.key) },
        { "artifacts", std::move(artifacts) },
        { "metadata", action.metadata }
    };
}

std::optional<cache_action> parse_action(const std::filesystem::path& path, asset_hash expected)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return std::nullopt;
    const auto document = json::parse(stream, nullptr, false);
    if (!document.is_object() || document.value("format", "") != "arc.ddc-action" ||
        document.value("version", 0) != 1)
        return std::nullopt;
    const auto key = parse_asset_hash(document.value("key", ""));
    if (!key || *key != expected || !document.contains("artifacts") ||
        !document["artifacts"].is_array())
        return std::nullopt;
    cache_action result;
    result.key = *key;
    result.metadata = document.value("metadata", "");
    for (const auto& value : document["artifacts"])
    {
        if (!value.is_string())
            return std::nullopt;
        const auto hash = parse_asset_hash(value.get<std::string>());
        if (!hash)
            return std::nullopt;
        result.artifacts.push_back(*hash);
    }
    return result;
}

std::optional<cache_action> parse_action(std::span<const std::byte> bytes, asset_hash expected)
{
    const auto document = json::parse(
        reinterpret_cast<const char*>(bytes.data()),
        reinterpret_cast<const char*>(bytes.data() + bytes.size()),
        nullptr,
        false);
    if (!document.is_object() || document.value("format", "") != "arc.ddc-action" ||
        document.value("version", 0) != 1)
        return std::nullopt;
    const auto key = parse_asset_hash(document.value("key", ""));
    if (!key || *key != expected || !document.contains("artifacts") ||
        !document["artifacts"].is_array())
        return std::nullopt;
    cache_action result;
    result.key = *key;
    result.metadata = document.value("metadata", "");
    for (const auto& value : document["artifacts"])
    {
        if (!value.is_string())
            return std::nullopt;
        const auto hash = parse_asset_hash(value.get<std::string>());
        if (!hash)
            return std::nullopt;
        result.artifacts.push_back(*hash);
    }
    return result;
}

}

std::string to_string(cook_processor_id value) { return id_string(value.high, value.low); }
std::string to_string(artifact_schema_id value) { return id_string(value.high, value.low); }

cook_target windows_vulkan_cook_target() { return {}; }

cook_target linux_vulkan_cook_target()
{
    auto result = windows_vulkan_cook_target();
    result.name = "linux-x64-vulkan";
    result.platform = cook_platform::linux;
    return result;
}

std::string canonical_cook_target(const cook_target& target)
{
    std::vector<std::string> features = target.features;
    std::sort(features.begin(), features.end());
    std::ostringstream stream;
    stream << target.name << '|'
        << static_cast<unsigned>(target.platform) << '|'
        << static_cast<unsigned>(target.architecture) << '|'
        << static_cast<unsigned>(target.renderer) << '|'
        << static_cast<unsigned>(target.textures) << '|'
        << static_cast<unsigned>(target.configuration) << '|'
        << target.api_major << '.' << target.api_minor << '|'
        << (target.little_endian ? "le" : "be");
    for (const auto& feature : features)
        stream << '|' << feature;
    return stream.str();
}

asset_build_key make_asset_build_key(const asset_build_key_desc& description)
{
    std::vector<std::byte> bytes;
    append(bytes, "arc.asset-build-key.v1");
    append(bytes, description.source_hash);
    for (const auto& hash : description.dependency_hashes)
        append(bytes, hash);
    append(bytes, to_string(description.importer));
    append(bytes, std::to_string(description.importer_version));
    append(bytes, to_string(description.processor));
    append(bytes, std::to_string(description.processor_version));
    append(bytes, to_string(description.schema));
    append(bytes, std::to_string(description.schema_version));
    append(bytes, description.canonical_settings);
    append(bytes, description.toolchain_fingerprint);
    for (const auto& hash : description.shader_include_hashes)
        append(bytes, hash);
    append(bytes, description.shader_compiler_fingerprint);
    append(bytes, description.shader_entry_point);
    auto defines = description.shader_defines;
    std::sort(defines.begin(), defines.end());
    for (const auto& define : defines)
        append(bytes, define);
    append(bytes, canonical_cook_target(description.target));
    return hash_bytes(bytes);
}

double cache_statistics::hit_rate() const noexcept
{
    const auto hits = local_hits + shared_hits;
    const auto total = hits + local_misses + shared_misses;
    return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
}

struct filesystem_shared_cache::implementation
{
    std::filesystem::path root;
    bool read_only{};
    std::mutex mutex;
};

filesystem_shared_cache::filesystem_shared_cache(std::filesystem::path root, bool read_only)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->root = std::move(root);
    implementation_->read_only = read_only;
}

filesystem_shared_cache::~filesystem_shared_cache() = default;

std::optional<std::vector<std::byte>> filesystem_shared_cache::get_blob(
    content_hash hash, cache_error& error)
{
    std::lock_guard lock(implementation_->mutex);
    auto bytes = read_file(blob_path(implementation_->root, hash));
    if (bytes && hash_bytes(*bytes) != hash)
    {
        error.message = "Shared cache blob failed content verification";
        return std::nullopt;
    }
    return bytes;
}

bool filesystem_shared_cache::put_blob(
    content_hash hash, std::span<const std::byte> bytes, cache_error& error)
{
    if (implementation_->read_only)
    {
        error.message = "Shared cache is read-only";
        return false;
    }
    if (hash_bytes(bytes) != hash)
    {
        error.message = "Shared cache publication hash does not match the payload";
        return false;
    }
    std::lock_guard lock(implementation_->mutex);
    if (auto existing = read_file(blob_path(implementation_->root, hash)))
    {
        if (hash_bytes(*existing) == hash)
            return true;
        error.message = "Shared cache already contains corrupt data for an immutable blob";
        return false;
    }
    return write_atomic(blob_path(implementation_->root, hash), bytes, error.message);
}

std::optional<cache_action> filesystem_shared_cache::get_action(
    asset_build_key key, cache_error& error)
{
    std::lock_guard lock(implementation_->mutex);
    auto result = parse_action(action_path(implementation_->root, key), key);
    if (!result && std::filesystem::exists(action_path(implementation_->root, key)))
        error.message = "Shared cache action is invalid";
    return result;
}

bool filesystem_shared_cache::put_action(const cache_action& action, cache_error& error)
{
    if (implementation_->read_only)
    {
        error.message = "Shared cache is read-only";
        return false;
    }
    const auto serialized = action_json(action).dump();
    std::lock_guard lock(implementation_->mutex);
    if (auto existing = read_file(action_path(implementation_->root, action.key)))
    {
        const std::string existing_text(
            reinterpret_cast<const char*>(existing->data()), existing->size());
        if (existing_text == serialized)
            return true;
        error.message = "Shared cache action key already maps to a different immutable result";
        return false;
    }
    return write_atomic(action_path(implementation_->root, action.key),
        std::as_bytes(std::span(serialized.data(), serialized.size())), error.message);
}

struct http_shared_cache::implementation
{
    http_shared_cache_config config;

    http_cache_request request(http_cache_method method, std::string path) const
    {
        while (!path.empty() && path.front() == '/')
            path.erase(path.begin());
        std::string endpoint = config.endpoint;
        while (!endpoint.empty() && endpoint.back() == '/')
            endpoint.pop_back();
        http_cache_request result{ .method = method, .url = endpoint + '/' + path };
        if (!config.bearer_token.empty())
            result.headers.emplace_back("authorization", "Bearer " + config.bearer_token);
        return result;
    }
};

http_shared_cache::http_shared_cache(http_shared_cache_config config)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->config = std::move(config);
}

http_shared_cache::~http_shared_cache() = default;

std::optional<std::vector<std::byte>> http_shared_cache::get_blob(
    content_hash hash, cache_error& error)
{
    if (!implementation_->config.transport)
    {
        error.message = "HTTP shared cache has no transport";
        return std::nullopt;
    }
    const auto hash_text = to_string(hash);
    const auto response = implementation_->config.transport(implementation_->request(
        http_cache_method::get, "v1/blobs/sha256/" + hash_text));
    if (response.status == 404)
        return std::nullopt;
    if (response.status != 200)
    {
        error.message = !response.error.empty() ? response.error :
            "HTTP shared cache blob request failed with status " + std::to_string(response.status);
        return std::nullopt;
    }
    const auto header = [&](std::string_view name) -> std::string {
        for (const auto& [key, value] : response.headers)
            if (key == name) return value;
        return {};
    };
    auto etag = header("etag");
    if (etag.size() >= 2 && etag.front() == '"' && etag.back() == '"')
        etag = etag.substr(1, etag.size() - 2);
    if (etag != hash_text || hash_bytes(response.body) != hash)
    {
        error.message = "HTTP shared cache blob failed ETag or SHA-256 verification";
        return std::nullopt;
    }
    return response.body;
}

bool http_shared_cache::put_blob(
    content_hash hash, std::span<const std::byte> bytes, cache_error& error)
{
    if (implementation_->config.read_only)
    {
        error.message = "HTTP shared cache is read-only";
        return false;
    }
    if (!implementation_->config.transport || hash_bytes(bytes) != hash)
    {
        error.message = implementation_->config.transport
            ? "HTTP shared cache publication hash does not match the payload"
            : "HTTP shared cache has no transport";
        return false;
    }
    auto request = implementation_->request(
        http_cache_method::put, "v1/blobs/sha256/" + to_string(hash));
    request.headers.emplace_back("if-none-match", "*");
    request.headers.emplace_back("content-length", std::to_string(bytes.size()));
    request.body.assign(bytes.begin(), bytes.end());
    const auto response = implementation_->config.transport(request);
    if (response.status == 200 || response.status == 201 || response.status == 204)
        return true;
    if (response.status == 409)
    {
        cache_error verify_error;
        const auto existing = get_blob(hash, verify_error);
        if (existing && *existing == std::vector<std::byte>(bytes.begin(), bytes.end()))
            return true;
        error.message = verify_error ? verify_error.message :
            "HTTP shared cache rejected an immutable blob with different content";
        return false;
    }
    error.message = !response.error.empty() ? response.error :
        "HTTP shared cache blob publication failed with status " + std::to_string(response.status);
    return false;
}

std::optional<cache_action> http_shared_cache::get_action(
    asset_build_key key, cache_error& error)
{
    if (!implementation_->config.transport)
    {
        error.message = "HTTP shared cache has no transport";
        return std::nullopt;
    }
    const auto response = implementation_->config.transport(implementation_->request(
        http_cache_method::get, "v1/actions/" + to_string(key)));
    if (response.status == 404)
        return std::nullopt;
    if (response.status != 200)
    {
        error.message = !response.error.empty() ? response.error :
            "HTTP shared cache action request failed with status " + std::to_string(response.status);
        return std::nullopt;
    }
    auto result = parse_action(response.body, key);
    if (!result)
        error.message = "HTTP shared cache returned an invalid action";
    return result;
}

bool http_shared_cache::put_action(const cache_action& action, cache_error& error)
{
    if (implementation_->config.read_only)
    {
        error.message = "HTTP shared cache is read-only";
        return false;
    }
    if (!implementation_->config.transport)
    {
        error.message = "HTTP shared cache has no transport";
        return false;
    }
    const auto serialized = action_json(action).dump();
    auto request = implementation_->request(
        http_cache_method::put, "v1/actions/" + to_string(action.key));
    request.headers.emplace_back("if-none-match", "*");
    request.headers.emplace_back("content-type", "application/json");
    request.body.assign(
        reinterpret_cast<const std::byte*>(serialized.data()),
        reinterpret_cast<const std::byte*>(serialized.data() + serialized.size()));
    const auto response = implementation_->config.transport(request);
    if (response.status == 200 || response.status == 201 || response.status == 204)
        return true;
    if (response.status == 409)
    {
        cache_error verify_error;
        const auto existing = get_action(action.key, verify_error);
        if (existing && existing->artifacts == action.artifacts &&
            existing->metadata == action.metadata)
            return true;
        error.message = verify_error ? verify_error.message :
            "HTTP shared cache rejected an immutable action with a different result";
        return false;
    }
    error.message = !response.error.empty() ? response.error :
        "HTTP shared cache action publication failed with status " + std::to_string(response.status);
    return false;
}

struct derived_data_cache::implementation
{
    derived_data_cache_config config;
    mutable std::mutex mutex;
    cache_statistics statistics;
    std::unordered_set<std::string> pins;

    void load_statistics()
    {
        std::ifstream stream(config.root / "statistics.json", std::ios::binary);
        const auto document = stream ? json::parse(stream, nullptr, false) : json{};
        if (!document.is_object() || document.value("format", "") != "arc.ddc-statistics" ||
            document.value("version", 0) != 1)
            return;
        statistics.local_hits = document.value("localHits", 0ull);
        statistics.local_misses = document.value("localMisses", 0ull);
        statistics.shared_hits = document.value("sharedHits", 0ull);
        statistics.shared_misses = document.value("sharedMisses", 0ull);
        statistics.bytes_read = document.value("bytesRead", 0ull);
        statistics.bytes_written = document.value("bytesWritten", 0ull);
        statistics.bytes_downloaded = document.value("bytesDownloaded", 0ull);
        statistics.bytes_uploaded = document.value("bytesUploaded", 0ull);
        statistics.corrupt_entries = document.value("corruptEntries", 0ull);
        statistics.evictions = document.value("evictions", 0ull);
        statistics.avoided_processor_runs = document.value("avoidedProcessorRuns", 0ull);
    }

    void persist_statistics() const
    {
        if (config.access == cache_access::read_only)
            return;
        const auto serialized = json{
            { "format", "arc.ddc-statistics" },
            { "version", 1 },
            { "localHits", statistics.local_hits },
            { "localMisses", statistics.local_misses },
            { "sharedHits", statistics.shared_hits },
            { "sharedMisses", statistics.shared_misses },
            { "bytesRead", statistics.bytes_read },
            { "bytesWritten", statistics.bytes_written },
            { "bytesDownloaded", statistics.bytes_downloaded },
            { "bytesUploaded", statistics.bytes_uploaded },
            { "corruptEntries", statistics.corrupt_entries },
            { "evictions", statistics.evictions },
            { "avoidedProcessorRuns", statistics.avoided_processor_runs }
        }.dump();
        std::ofstream stream(config.root / "statistics.json", std::ios::binary | std::ios::trunc);
        if (stream)
            stream.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
    }

    void refresh_size()
    {
        std::uint64_t size{};
        std::error_code error;
        const auto root = config.root / "cas";
        if (std::filesystem::exists(root, error))
            for (std::filesystem::recursive_directory_iterator it(root, error), end; it != end && !error; it.increment(error))
                if (it->is_regular_file(error))
                    size += it->file_size(error);
        statistics.local_bytes = size;
    }
};

derived_data_cache::derived_data_cache(derived_data_cache_config config)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->config = std::move(config);
    if (implementation_->config.root.empty())
        implementation_->config.root = std::filesystem::current_path() / ".arc" / "cache";
    std::error_code error;
    std::filesystem::create_directories(implementation_->config.root / "cas" / "sha256", error);
    std::filesystem::create_directories(implementation_->config.root / "actions", error);
    implementation_->load_statistics();
    implementation_->refresh_size();
}

derived_data_cache::~derived_data_cache()
{
    if (implementation_)
    {
        std::lock_guard lock(implementation_->mutex);
        implementation_->persist_statistics();
    }
}
derived_data_cache::derived_data_cache(derived_data_cache&&) noexcept = default;
derived_data_cache& derived_data_cache::operator=(derived_data_cache&&) noexcept = default;

std::optional<cache_blob> derived_data_cache::get_blob(content_hash hash, cache_error& error)
{
    std::lock_guard lock(implementation_->mutex);
    const auto path = blob_path(implementation_->config.root, hash);
    if (auto bytes = read_file(path))
    {
        if (hash_bytes(*bytes) == hash)
        {
            ++implementation_->statistics.local_hits;
            implementation_->statistics.bytes_read += bytes->size();
            std::error_code touch_error;
            std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), touch_error);
            return cache_blob{ hash, std::move(*bytes), cache_layer::local };
        }
        ++implementation_->statistics.corrupt_entries;
        std::error_code quarantine_error;
        std::filesystem::rename(path, path.string() + ".corrupt", quarantine_error);
    }
    ++implementation_->statistics.local_misses;
    if (implementation_->config.access != cache_access::offline && implementation_->config.shared)
    {
        cache_error shared_error;
        if (auto bytes = implementation_->config.shared->get_blob(hash, shared_error))
        {
            ++implementation_->statistics.shared_hits;
            implementation_->statistics.bytes_downloaded += bytes->size();
            std::string write_error;
            write_atomic(path, *bytes, write_error);
            implementation_->statistics.local_bytes += bytes->size();
            return cache_blob{ hash, std::move(*bytes), cache_layer::shared };
        }
        ++implementation_->statistics.shared_misses;
        if (implementation_->config.require_shared)
            error.message = shared_error ? std::move(shared_error.message) :
                "Required shared cache blob was not found";
    }
    return std::nullopt;
}

std::optional<cache_action> derived_data_cache::get_action(asset_build_key key, cache_error& error)
{
    std::lock_guard lock(implementation_->mutex);
    const auto path = action_path(implementation_->config.root, key);
    if (auto result = parse_action(path, key))
    {
        ++implementation_->statistics.local_hits;
        return result;
    }
    ++implementation_->statistics.local_misses;
    if (implementation_->config.access != cache_access::offline && implementation_->config.shared)
    {
        cache_error shared_error;
        if (auto result = implementation_->config.shared->get_action(key, shared_error))
        {
            ++implementation_->statistics.shared_hits;
            const auto serialized = action_json(*result).dump();
            std::string ignored;
            write_atomic(path, std::as_bytes(std::span(serialized.data(), serialized.size())), ignored);
            return result;
        }
        ++implementation_->statistics.shared_misses;
        if (implementation_->config.require_shared)
            error.message = shared_error ? std::move(shared_error.message) :
                "Required shared cache action was not found";
    }
    return std::nullopt;
}

bool derived_data_cache::put_blob(
    content_hash hash, std::span<const std::byte> bytes, cache_error& error)
{
    if (hash_bytes(bytes) != hash)
    {
        error.message = "Blob content does not match its content hash";
        return false;
    }
    if (implementation_->config.access == cache_access::read_only)
    {
        error.message = "Local cache is read-only";
        return false;
    }
    std::lock_guard lock(implementation_->mutex);
    const auto destination = blob_path(implementation_->config.root, hash);
    const bool existed = std::filesystem::exists(destination);
    if (existed)
    {
        const auto existing = read_file(destination);
        if (!existing || hash_bytes(*existing) != hash)
        {
            error.message = "Local cache already contains corrupt data for an immutable blob";
            return false;
        }
    }
    if (!write_atomic(destination, bytes, error.message))
        return false;
    if (!existed)
    {
        implementation_->statistics.bytes_written += bytes.size();
        implementation_->statistics.local_bytes += bytes.size();
    }
    if (implementation_->config.access != cache_access::offline && implementation_->config.shared)
    {
        cache_error shared_error;
        if (implementation_->config.shared->put_blob(hash, bytes, shared_error))
            implementation_->statistics.bytes_uploaded += bytes.size();
        else if (implementation_->config.require_shared)
        {
            error = std::move(shared_error);
            return false;
        }
    }
    return true;
}

bool derived_data_cache::put_action(const cache_action& action, cache_error& error)
{
    if (implementation_->config.access == cache_access::read_only)
    {
        error.message = "Local cache is read-only";
        return false;
    }
    const auto serialized = action_json(action).dump();
    std::lock_guard lock(implementation_->mutex);
    const auto destination = action_path(implementation_->config.root, action.key);
    if (auto existing = read_file(destination))
    {
        const std::string existing_text(
            reinterpret_cast<const char*>(existing->data()), existing->size());
        if (existing_text != serialized)
        {
            error.message = "Local cache action key already maps to a different immutable result";
            return false;
        }
    }
    if (!write_atomic(destination,
            std::as_bytes(std::span(serialized.data(), serialized.size())), error.message))
        return false;
    if (implementation_->config.access != cache_access::offline && implementation_->config.shared)
    {
        cache_error ignored;
        if (!implementation_->config.shared->put_action(action, ignored) &&
            implementation_->config.require_shared)
        {
            error = std::move(ignored);
            return false;
        }
    }
    return true;
}

bool derived_data_cache::pin(content_hash hash)
{
    std::lock_guard lock(implementation_->mutex);
    return implementation_->pins.insert(to_string(hash)).second;
}

bool derived_data_cache::unpin(content_hash hash)
{
    std::lock_guard lock(implementation_->mutex);
    return implementation_->pins.erase(to_string(hash)) != 0;
}

std::size_t derived_data_cache::verify(std::vector<std::string>* diagnostics)
{
    std::lock_guard lock(implementation_->mutex);
    std::size_t valid{};
    std::error_code error;
    const auto root = implementation_->config.root / "cas" / "sha256";
    if (!std::filesystem::exists(root, error))
        return 0;
    for (std::filesystem::recursive_directory_iterator it(root, error), end; it != end && !error; it.increment(error))
    {
        if (!it->is_regular_file(error) || it->path().extension() == ".corrupt")
            continue;
        const auto expected = parse_asset_hash(it->path().filename().string());
        std::string hash_error;
        if (expected && hash_file(it->path(), &hash_error) == *expected)
            ++valid;
        else
        {
            ++implementation_->statistics.corrupt_entries;
            if (diagnostics)
                diagnostics->push_back("Corrupt cache blob: " + it->path().generic_string());
        }
    }
    return valid;
}

std::uint64_t derived_data_cache::prune(bool force)
{
    std::lock_guard lock(implementation_->mutex);
    implementation_->refresh_size();
    const auto& policy = implementation_->config.cleanup;
    if (!force && implementation_->statistics.local_bytes <=
        static_cast<std::uint64_t>(policy.maximum_bytes * policy.prune_threshold))
        return 0;
    const auto target = force ? 0ull :
        static_cast<std::uint64_t>(policy.maximum_bytes * policy.prune_target);
    struct candidate { std::filesystem::path path; std::uint64_t size{}; std::filesystem::file_time_type time{}; };
    std::vector<candidate> candidates;
    std::error_code error;
    const auto root = implementation_->config.root / "cas" / "sha256";
    if (std::filesystem::exists(root, error))
        for (std::filesystem::recursive_directory_iterator it(root, error), end; it != end && !error; it.increment(error))
            if (it->is_regular_file(error) &&
                !implementation_->pins.contains(it->path().filename().string()))
                candidates.push_back({ it->path(), it->file_size(error), it->last_write_time(error) });
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.time < rhs.time;
    });
    std::uint64_t removed{};
    for (const auto& candidate : candidates)
    {
        if (implementation_->statistics.local_bytes <= target)
            break;
        std::filesystem::remove(candidate.path, error);
        if (!error)
        {
            removed += candidate.size;
            implementation_->statistics.local_bytes -= candidate.size;
            ++implementation_->statistics.evictions;
        }
        error.clear();
    }
    const auto now = std::filesystem::file_time_type::clock::now();
    const auto remove_expired = [&](const std::filesystem::path& root, auto lifetime, bool temporary_only) {
        if (!std::filesystem::exists(root, error))
            return;
        for (std::filesystem::recursive_directory_iterator it(root, error), end;
            it != end && !error; it.increment(error))
        {
            if (!it->is_regular_file(error))
                continue;
            if (temporary_only && it->path().filename().string().find(".tmp-") == std::string::npos)
                continue;
            if (now - it->last_write_time(error) > lifetime)
                std::filesystem::remove(it->path(), error);
            error.clear();
        }
    };
    remove_expired(implementation_->config.root,
        implementation_->config.cleanup.temporary_lifetime, true);
    remove_expired(implementation_->config.root / "actions",
        implementation_->config.cleanup.action_lifetime, false);
    return removed;
}

void derived_data_cache::note_avoided_processor_run()
{
    std::lock_guard lock(implementation_->mutex);
    ++implementation_->statistics.avoided_processor_runs;
}

cache_statistics derived_data_cache::statistics() const
{
    std::lock_guard lock(implementation_->mutex);
    return implementation_->statistics;
}

const derived_data_cache_config& derived_data_cache::config() const noexcept
{
    return implementation_->config;
}

} // namespace arc::assets
