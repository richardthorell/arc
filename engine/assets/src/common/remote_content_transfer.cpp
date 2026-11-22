#include <arc/assets/remote_content_transfer.h>
#include <arc/assets/remote_content_download.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace arc::assets
{
namespace
{

bool safe_filename(std::string_view name, std::string_view suffix)
{
    if (name.empty() || name.size() > 255 || !name.ends_with(suffix) ||
        !std::isalnum(static_cast<unsigned char>(name.front())))
        return false;
    return std::all_of(name.begin(), name.end(),
                       [](unsigned char c) { return std::isalnum(c) || c == '-' || c == '_' || c == '.'; });
}

std::string folded(std::string_view text)
{
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

bool valid_object(const remote_content_object& object, const http::content_http_limits& limits)
{
    if (object.size == 0 || object.size > limits.maximum_object_bytes ||
        object.path != "objects/sha256/" + object.sha256)
        return false;
    const auto digest = parse_asset_hash(object.sha256);
    return digest && to_string(*digest) == object.sha256;
}

} // namespace

jobs::job_future<remote_generation_transfer_result>
acquire_remote_generation(jobs::job_system& jobs, io::async_file_service& files, derived_data_cache& cache,
                          remote_content_release release, remote_generation_transfer_options options,
                          jobs::cancellation_token cancellation, remote_generation_fetch_executor executor)
{
    // Do not hand cancellation to the coordinator's job descriptor: callers
    // always receive a structured cancelled result, including pre-cancellation.
    return jobs.submit_future(
        {.name = "assets.remote_content.acquire",
         .priority = jobs::job_priority::normal,
         .affinity = jobs::job_affinity::any_worker},
        [&jobs, &files, &cache, release = std::move(release), options = std::move(options), cancellation,
         executor = std::move(executor)]() mutable -> remote_generation_transfer_result
        {
            remote_generation_transfer_result result;
            const auto reject = [&](std::string path, std::string message)
            {
                result.failed_object = std::move(path);
                result.error = std::move(message);
                return result;
            };
            if (cancellation.stop_requested())
            {
                result.transport_error = http::content_http_error_code::cancelled;
                return reject({}, "Remote generation acquisition cancelled");
            }
            if (release.generation == 0 || release.base_build_id.empty() ||
                !safe_filename(release.cook_manifest_name, ".arccookmanifest"))
                return reject({}, "Remote release has invalid generation identity or manifest filename");
            if (options.maximum_objects == 0 || release.chunks.size() >= options.maximum_objects ||
                options.maximum_generation_bytes == 0)
                return reject({}, "Remote release exceeds generation object or byte limits");

            std::unordered_set<std::string> hashes;
            std::unordered_set<std::string> names;
            std::vector<const remote_content_object*> objects;
            objects.reserve(release.chunks.size() + 1);
            std::uint64_t total_bytes{};
            const auto add_object = [&](const remote_content_object& object)
            {
                if (!valid_object(object, options.http) || !hashes.insert(object.sha256).second ||
                    object.size > options.maximum_generation_bytes - total_bytes)
                    return false;
                total_bytes += object.size;
                objects.push_back(&object);
                return true;
            };

            // Process the manifest first. The release descriptor is the only
            // source of object identities, never an HTTP-supplied path/URL.
            if (!add_object(release.cook_manifest))
                return reject(release.cook_manifest.path, "Remote cook manifest object is invalid or oversized");
            for (const auto& chunk : release.chunks)
            {
                if (!safe_filename(chunk.name, ".arcpak") || !names.insert(folded(chunk.name)).second ||
                    !add_object(chunk.object))
                    return reject(chunk.object.path, "Remote generation has a duplicate or invalid chunk object");
            }

            const auto downloads = options.downloads
                                       ? options.downloads
                                       : std::make_shared<remote_object_download_coordinator>(jobs, cache);
            for (const auto* object : objects)
            {
                if (cancellation.stop_requested())
                {
                    result.transport_error = http::content_http_error_code::cancelled;
                    return reject(object->path, "Remote generation acquisition cancelled");
                }

                if (options.policy != remote_generation_transfer_policy::refresh_all)
                {
                    cache_error error;
                    if (load_verified_remote_object(cache, *object, error))
                    {
                        ++result.reused_objects;
                        continue;
                    }
                }
                if (options.policy == remote_generation_transfer_policy::offline_only)
                    return reject(object->path, "Required remote generation object is unavailable offline");

                const auto transfer = downloads->ensure(*object, options.origin, options.http, cancellation,
                                                        executor,
                                                        options.policy == remote_generation_transfer_policy::refresh_all);
                if (!transfer.succeeded())
                {
                    result.transport_error = transfer.transport_error;
                    result.cache_error = transfer.cache_error;
                    return reject(object->path, transfer.error);
                }
                if (transfer.reused) ++result.reused_objects;
                if (transfer.downloaded)
                {
                    ++result.downloaded_objects;
                    result.downloaded_bytes += object->size;
                }
            }

            if (cancellation.stop_requested())
            {
                result.transport_error = http::content_http_error_code::cancelled;
                return reject({}, "Remote generation acquisition cancelled");
            }

            // Do not claim a completed download until the *entire* cook
            // manifest, chunk table and artifact hash closure pass C9.2c2.
            auto verified = remote_content_artifact_provider::create(release, cache, files);
            if (!verified) return reject({}, "Remote generation validation failed: " + verified.error().message);
            return result;
        });
}

} // namespace arc::assets
