#pragma once

/** @file
 * @brief C9.2f1 prepare one metadata-only, demand-loaded remote cook generation.
 *
 * The producer/publisher has already supplied a C9.1-validated release.
 * Only the immutable cook manifest is needed to construct the lazy provider.
 * Chunk bodies are downloaded and verified on the first artifact read.
 */

#include <arc/assets/remote_content_generation.h>

#include <cstdint>
#include <memory>
#include <string>

namespace arc::assets
{

struct [[nodiscard]] remote_lazy_generation_result
{
    std::shared_ptr<remote_content_artifact_provider> provider;
    bool manifest_reused{};
    std::uint64_t downloaded_bytes{};
    http::content_http_error_code transport_error{http::content_http_error_code::none};
    remote_object_cache_error cache_error{remote_object_cache_error::none};
    std::string error;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return provider != nullptr;
    }
};

/** Runs on a general worker. Fetches only the SHA-verified immutable cook
 * manifest over an IO-worker HTTP future; validates the complete chunk
 * declaration graph and returns an unmounted lazy provider.
 *
 * Call catalog.mount_remote_provider(result.provider) on a safe caller thread
 * to publish the metadata. This call never changes an active catalog.
 * Cache and scheduler must outlive the returned provider and future.
 */
[[nodiscard]] jobs::job_future<remote_lazy_generation_result>
prepare_lazy_remote_generation(jobs::job_system& jobs, io::async_file_service& files, derived_data_cache& cache,
                               remote_content_release release, lazy_remote_content_options options,
                               jobs::cancellation_token cancellation = {}, remote_generation_fetch_executor executor = {});

} // namespace arc::assets
