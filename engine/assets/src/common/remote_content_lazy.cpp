#include <arc/assets/remote_content_lazy.h>
#include <arc/assets/remote_content_download.h>

#include <exception>
#include <string>
#include <utility>

namespace arc::assets
{

jobs::job_future<remote_lazy_generation_result>
prepare_lazy_remote_generation(jobs::job_system& jobs, io::async_file_service& files, derived_data_cache& cache,
                               remote_content_release release, lazy_remote_content_options options,
                               jobs::cancellation_token cancellation, remote_generation_fetch_executor executor)
{
    // Deliberately omit coordinator job cancellation. Always return a typed
    // result, including when the caller cancelled before scheduling.
    return jobs.submit_future(
        {.name = "assets.remote_content.lazy_prepare",
         .priority = jobs::job_priority::normal,
         .affinity = jobs::job_affinity::any_worker},
        [&jobs, &files, &cache, release = std::move(release), options = std::move(options), cancellation,
         executor = std::move(executor)]() mutable -> remote_lazy_generation_result
        {
            remote_lazy_generation_result result;
            const auto reject = [&](std::string message)
            {
                result.error = std::move(message);
                return result;
            };
            if (cancellation.stop_requested())
            {
                result.transport_error = http::content_http_error_code::cancelled;
                return reject("Lazy remote generation preparation cancelled");
            }
            if (release.cook_manifest.size == 0 || release.cook_manifest.size > 8u * 1024u * 1024u ||
                release.cook_manifest.size > options.http.maximum_object_bytes)
                return reject("Remote cook manifest exceeds metadata or HTTP object limits");

            // Reuse a caller-owned coordinator across generations or create
            // one for this prepared lazy provider and its future chunk reads.
            if (!options.downloads)
                options.downloads = std::make_shared<remote_object_download_coordinator>(jobs, cache);
            cache_error error;
            auto manifest = load_verified_remote_object(cache, release.cook_manifest, error);
            if (manifest)
                result.manifest_reused = true;
            else if (options.offline_only)
                return reject("Remote cook manifest unavailable in offline immutable cache");
            else
            {
                const auto downloaded = options.downloads->ensure(release.cook_manifest, options.origin, options.http,
                                                                  cancellation, executor);
                if (!downloaded.succeeded())
                {
                    result.transport_error = downloaded.transport_error;
                    result.cache_error = downloaded.cache_error;
                    return reject("Remote immutable cook manifest acquisition failed: " + downloaded.error);
                }
                result.manifest_reused = downloaded.reused;
                result.downloaded_bytes = downloaded.downloaded ? release.cook_manifest.size : 0;
            }

            if (cancellation.stop_requested())
            {
                result.transport_error = http::content_http_error_code::cancelled;
                return reject("Lazy remote generation preparation cancelled");
            }

            // Metadata comes from the verified immutable cook manifest, not
            // from an untrusted URL or directly from the release descriptor.
            auto provider = remote_content_artifact_provider::create_lazy(release, cache, files, options, executor);
            if (!provider) return reject("Remote lazy manifest layout validation failed: " + provider.error().message);
            result.provider = std::move(provider).value();
            return result;
        });
}

} // namespace arc::assets
