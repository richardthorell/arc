#pragma once

/** @file
 * @brief C9.2d1 cache-first acquisition of a *validated* remote cook generation.
 *
 * Accepts a parsed C9.1 release descriptor, fetches immutable manifest/chunk
 * objects via the C9.2 HTTP client, and verifies the entire generation using
 * C9.2c2 before reporting success. Never mounts a partially fetched release.
 */

#include <arc/assets/remote_content_generation.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace arc::assets
{

enum class remote_generation_transfer_policy : std::uint8_t
{
    cache_first,
    offline_only,
    refresh_all
};

struct remote_generation_transfer_options
{
    std::string origin;
    io::content_http_limits http{};
    remote_generation_transfer_policy policy{remote_generation_transfer_policy::cache_first};
    std::uint64_t maximum_generation_bytes{16ull * 1024ull * 1024ull * 1024ull};
    std::size_t maximum_objects{100000};
};

struct [[nodiscard]] remote_generation_transfer_result
{
    std::size_t reused_objects{};
    std::size_t downloaded_objects{};
    std::uint64_t downloaded_bytes{};
    io::content_http_error_code transport_error{io::content_http_error_code::none};
    remote_object_cache_error cache_error{remote_object_cache_error::none};
    std::string failed_object;
    std::string error;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return error.empty();
    }
};

/** Injection seam for deterministic native tests and future platform backends.
 * The same HTTP response validator and SHA-verified CAS ingestion are applied
 * regardless of which executor supplies bytes. Production callers omit this
 * argument and use fetch_content_http.
 */
using remote_generation_fetch_executor =
    std::function<jobs::job_future<io::content_http_fetch_result>(
        jobs::job_system&, io::content_http_request, io::content_http_limits, jobs::cancellation_token)>;

/** Runs the transfer coordinator on a general worker and each HTTP GET on an
 * IO worker. Fetches serially to keep at most one object body in memory; later
 * work may add bounded parallelism and request coalescing.
 *
 * The caller is responsible for validating the C9.1 release descriptor and
 * performing catalog.mount_remote_generation(...) only *after* this future
 * succeeds. This never modifies an active catalog. jobs, files and cache must
 * outlive the future. No render/main-thread blocking is required.
 */
[[nodiscard]] jobs::job_future<remote_generation_transfer_result>
acquire_remote_generation(jobs::job_system& jobs, io::async_file_service& files, derived_data_cache& cache,
                          remote_content_release release, remote_generation_transfer_options options,
                          jobs::cancellation_token cancellation = {}, remote_generation_fetch_executor executor = {});

} // namespace arc::assets
