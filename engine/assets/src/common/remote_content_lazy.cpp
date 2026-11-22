#include <arc/assets/remote_content_lazy.h>

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

            cache_error error;
            auto manifest = load_verified_remote_object(cache, release.cook_manifest, error);
            if (manifest)
                result.manifest_reused = true;
            else if (options.offline_only)
                return reject("Remote cook manifest unavailable in offline immutable cache");
            else
            {
                http::content_http_request request{.origin = options.origin,
                                                   .object_path = release.cook_manifest.path,
                                                   .object_size = release.cook_manifest.size};
                const auto valid = http::validate_content_http_request(request, options.http);
                if (!valid.succeeded())
                {
                    result.transport_error = valid.code;
                    return reject(valid.message);
                }
                auto fetch = executor ? executor(jobs, request, options.http, cancellation)
                                      : http::fetch_content_http(jobs, request, options.http, cancellation);
                if (!fetch.valid())
                {
                    result.transport_error = http::content_http_error_code::transport_unavailable;
                    return reject("Remote manifest fetch executor unavailable");
                }
                if (!fetch.handle().wait_result().succeeded())
                {
                    result.transport_error = cancellation.stop_requested()
                                                 ? http::content_http_error_code::cancelled
                                                 : http::content_http_error_code::network_failure;
                    return reject("Remote manifest fetch worker failed or was cancelled");
                }
                http::content_http_fetch_result response;
                try
                {
                    response = fetch.get();
                }
                catch (const std::exception&)
                {
                    result.transport_error = http::content_http_error_code::network_failure;
                    return reject("Remote manifest fetch worker failed");
                }
                if (!response.succeeded())
                {
                    result.transport_error = response.validation.code;
                    return reject(response.validation.message);
                }
                if (cancellation.stop_requested())
                {
                    result.transport_error = http::content_http_error_code::cancelled;
                    return reject("Remote generation preparation cancelled before CAS ingestion");
                }
                auto ingested = ingest_verified_remote_object(cache, release.cook_manifest, request, *response.response,
                                                              options.http);
                if (!ingested.succeeded())
                {
                    result.cache_error = ingested.code;
                    return reject("Remote manifest SHA-256/CAS verification failed: " + ingested.message);
                }
                result.downloaded_bytes = ingested.reused ? 0 : release.cook_manifest.size;
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
