#pragma once

/** @file
 * @brief C9.2f2 bounded and coalesced immutable cooked-object transfers.
 *
 * One coordinator may be shared by multiple remote generations that use the
 * same job scheduler and persistent derived-data cache. Calls must originate
 * on general/background workers, not on the IO/render/main thread.
 */

#include <arc/assets/remote_content_generation.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace arc::assets
{

struct remote_download_limits
{
    /** Maximum simultaneous in-flight HTTP GETs (not queued callers). */
    std::size_t maximum_concurrent_requests{4};
    /** Bound the sum of declared response bytes across in-flight HTTP GETs. */
    std::uint64_t maximum_inflight_bytes{512ull * 1024ull * 1024ull};
    /** Total attempts including the first request. */
    std::size_t maximum_attempts{3};
    /** Bounded exponential retry delay; no retry for hash/metadata failures. */
    std::chrono::milliseconds initial_backoff{50};
    std::chrono::milliseconds maximum_backoff{750};
};

struct [[nodiscard]] remote_download_result
{
    bool reused{};
    bool downloaded{};
    bool coalesced{};
    std::size_t attempts{};
    http::content_http_error_code transport_error{http::content_http_error_code::none};
    remote_object_cache_error cache_error{remote_object_cache_error::none};
    std::string error;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return error.empty();
    }
};

/** Cumulative, lock-consistent transfer diagnostics without origins, paths,
 * tokens, object hashes or session credentials. All counts are coordinator-
 * local. Calls include direct cache hits, followers and rejected requests;
 * network_attempts counts only real backend invocations by leaders.
 */
struct remote_download_metrics
{
    std::uint64_t calls{};
    std::uint64_t cache_hits{};
    std::uint64_t coalesced_calls{};
    std::uint64_t network_attempts{};
    std::uint64_t retry_attempts{};
    std::uint64_t completed_downloads{};
    std::uint64_t downloaded_bytes{};
    std::uint64_t failed_calls{};
    std::uint64_t cancelled_calls{};
    std::uint64_t integrity_failures{};
    std::uint64_t elapsed_milliseconds{};
    std::size_t active_requests{};
    std::size_t peak_active_requests{};
    std::size_t queued_requests{};
    std::size_t peak_queued_requests{};
    std::uint64_t active_declared_bytes{};
    std::uint64_t peak_declared_bytes{};
};

/** Coalesces identical SHA-256 object fetches while serializing their CAS
 * publication. A follower never reads a partially published object. It can
 * independently cancel its wait without cancelling the leader.
 *
 * Only transport timeout/network_failure/rate_limited are retried. HTTP 4xx,
 * bad framing, invalid TLS/origin, mismatched digest and cache failures never
 * retry. Last-good cooked catalog mounts are untouched.
 *
 * Must outlive all calls to ensure(). The caller owns the jobs and CAS.
 */
class remote_object_download_coordinator
{
public:
    remote_object_download_coordinator(jobs::job_system& jobs, derived_data_cache& cache,
                                       remote_download_limits limits = {});
    ~remote_object_download_coordinator();

    remote_object_download_coordinator(const remote_object_download_coordinator&) = delete;
    remote_object_download_coordinator& operator=(const remote_object_download_coordinator&) = delete;

    /** Blocking worker-side coordination; HTTP itself runs on an IO worker.
     * If force_refresh is true, do not accept a cache hit before the GET.
     * Injected fetchers are expected to return ARC IO-worker job futures.
     */
    [[nodiscard]] remote_download_result ensure(const remote_content_object& object, std::string origin,
                                                http::content_http_limits http_limits,
                                                jobs::cancellation_token cancellation = {},
                                                remote_generation_fetch_executor executor = {},
                                                bool force_refresh = false);

    /** Snapshot consistent under concurrent downloads, usable by diagnostics
     * and tests without exposing sensitive remote endpoint data.
     */
    [[nodiscard]] remote_download_metrics metrics() const;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace arc::assets
