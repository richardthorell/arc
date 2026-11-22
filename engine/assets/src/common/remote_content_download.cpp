#include <arc/assets/remote_content_download.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace arc::assets
{
namespace
{

remote_download_result failed(std::string message, http::content_http_error_code code = {})
{
    return {.transport_error = code, .error = std::move(message)};
}

bool retryable(http::content_http_error_code code) noexcept
{
    return code == http::content_http_error_code::timeout || code == http::content_http_error_code::network_failure ||
           code == http::content_http_error_code::rate_limited;
}

} // namespace

struct remote_object_download_coordinator::implementation
{
    struct inflight
    {
        bool done{};
        remote_download_result outcome;
    };

    jobs::job_system* jobs{};
    derived_data_cache* cache{};
    remote_download_limits limits;
    std::mutex mutex;
    std::condition_variable changed;
    std::unordered_map<std::string, std::shared_ptr<inflight>> requests;
    std::size_t active_requests{};
    std::uint64_t active_bytes{};

    [[nodiscard]] bool acquire_slot(std::uint64_t size, const jobs::cancellation_token& cancellation)
    {
        std::unique_lock lock(mutex);
        while (active_requests >= limits.maximum_concurrent_requests ||
               size > limits.maximum_inflight_bytes - active_bytes)
        {
            if (cancellation.stop_requested()) return false;
            changed.wait_for(lock, std::chrono::milliseconds(25));
        }
        if (cancellation.stop_requested()) return false;
        ++active_requests;
        active_bytes += size;
        return true;
    }

    void release_slot(std::uint64_t size)
    {
        {
            std::lock_guard lock(mutex);
            --active_requests;
            active_bytes -= size;
        }
        changed.notify_all();
    }

    [[nodiscard]] bool backoff(std::chrono::milliseconds duration, const jobs::cancellation_token& cancellation)
    {
        const auto end = std::chrono::steady_clock::now() + duration;
        std::unique_lock lock(mutex);
        while (!cancellation.stop_requested())
        {
            const auto now = std::chrono::steady_clock::now();
            if (now >= end) return true;
            changed.wait_until(lock, std::min(end, now + std::chrono::milliseconds(25)));
        }
        return false;
    }

    [[nodiscard]] remote_download_result download(const remote_content_object& object,
                                                  const http::content_http_request& request,
                                                  http::content_http_limits http_limits,
                                                  const jobs::cancellation_token& cancellation,
                                                  const remote_generation_fetch_executor& executor)
    {
        remote_download_result result;
        auto delay = limits.initial_backoff;
        for (std::size_t attempt = 0; attempt < limits.maximum_attempts; ++attempt)
        {
            if (cancellation.stop_requested())
                return failed("Remote immutable object transfer cancelled", http::content_http_error_code::cancelled);
            if (!acquire_slot(object.size, cancellation))
                return failed("Remote immutable object transfer cancelled in queue",
                              http::content_http_error_code::cancelled);

            http::content_http_fetch_result fetched;
            http::content_http_error_code failure_code = http::content_http_error_code::none;
            std::string failure_message;
            ++result.attempts;
            try
            {
                auto future = executor ? executor(*jobs, request, http_limits, cancellation)
                                       : http::fetch_content_http(*jobs, request, http_limits, cancellation);
                if (!future.valid())
                {
                    failure_code = http::content_http_error_code::transport_unavailable;
                    failure_message = "Remote HTTP executor returned no job";
                }
                else if (!future.handle().wait_result().succeeded())
                {
                    failure_code = cancellation.stop_requested() ? http::content_http_error_code::cancelled
                                                                 : http::content_http_error_code::network_failure;
                    failure_message = "Remote HTTP fetch job failed or was cancelled";
                }
                else
                {
                    fetched = future.get();
                    if (!fetched.succeeded())
                    {
                        failure_code = fetched.validation.code;
                        failure_message = fetched.validation.message;
                    }
                }
            }
            catch (const std::exception&)
            {
                failure_code = cancellation.stop_requested() ? http::content_http_error_code::cancelled
                                                             : http::content_http_error_code::network_failure;
                failure_message = "Remote HTTP fetch executor threw an exception";
            }
            release_slot(object.size);

            if (cancellation.stop_requested())
                return failed("Remote immutable object transfer cancelled", http::content_http_error_code::cancelled);
            if (failure_code == http::content_http_error_code::none)
            {
                const auto ingested =
                    ingest_verified_remote_object(*cache, object, request, *fetched.response, http_limits);
                if (!ingested.succeeded())
                {
                    result.cache_error = ingested.code;
                    result.error = "Remote immutable object integrity/CAS failure: " + ingested.message;
                    return result;
                }
                // HTTP was transferred even if CAS found a matching blob
                // during the fetch. Report network bytes, not CAS writes.
                result.downloaded = true;
                result.reused = false;
                return result;
            }

            result.transport_error = failure_code;
            result.error = failure_message.empty() ? "Remote HTTP transfer failed" : std::move(failure_message);
            if (!retryable(failure_code) || attempt + 1 == limits.maximum_attempts) return result;
            if (!backoff(delay, cancellation))
                return failed("Remote immutable object transfer cancelled during retry",
                              http::content_http_error_code::cancelled);
            delay = std::min(delay * 2, limits.maximum_backoff);
        }
        return failed("Remote immutable object transfer exhausted retries",
                      http::content_http_error_code::network_failure);
    }
};

remote_object_download_coordinator::remote_object_download_coordinator(jobs::job_system& jobs,
                                                                       derived_data_cache& cache,
                                                                       remote_download_limits limits)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->jobs = &jobs;
    implementation_->cache = &cache;
    // Never permit an unbounded queue to turn an unlimited number of large
    // objects into concurrent in-memory buffers.
    limits.maximum_concurrent_requests = std::clamp<std::size_t>(limits.maximum_concurrent_requests, 1, 64);
    limits.maximum_attempts = std::clamp<std::size_t>(limits.maximum_attempts, 1, 8);
    limits.initial_backoff =
        std::clamp(limits.initial_backoff, std::chrono::milliseconds{0}, std::chrono::milliseconds{5000});
    limits.maximum_backoff =
        std::clamp(limits.maximum_backoff, limits.initial_backoff, std::chrono::milliseconds{10000});
    implementation_->limits = limits;
}

remote_object_download_coordinator::~remote_object_download_coordinator() = default;

remote_download_result
remote_object_download_coordinator::ensure(const remote_content_object& object, std::string origin,
                                           http::content_http_limits http_limits, jobs::cancellation_token cancellation,
                                           remote_generation_fetch_executor executor, bool force_refresh)
{
    if (cancellation.stop_requested())
        return failed("Remote immutable object transfer cancelled", http::content_http_error_code::cancelled);
    const http::content_http_request request{
        .origin = std::move(origin), .object_path = object.path, .object_size = object.size};
    const auto validation = http::validate_content_http_request(request, http_limits);
    if (!validation.succeeded()) return failed(validation.message, validation.code);
    const auto digest = parse_asset_hash(object.sha256);
    if (!digest || to_string(*digest) != object.sha256 || object.path != "objects/sha256/" + object.sha256)
        return failed("Remote immutable object has invalid SHA-256 descriptor",
                      http::content_http_error_code::invalid_request);

    if (!force_refresh)
    {
        cache_error error;
        if (load_verified_remote_object(*implementation_->cache, object, error)) return {.reused = true};
    }

    // A verified local cache hit never consumes a network-transfer budget.
    if (implementation_->limits.maximum_inflight_bytes == 0 ||
        object.size > implementation_->limits.maximum_inflight_bytes)
        return failed("Remote immutable object exceeds in-flight byte budget",
                      http::content_http_error_code::oversized_response);

    std::shared_ptr<implementation::inflight> flight;
    bool leader = false;
    {
        std::lock_guard lock(implementation_->mutex);
        const auto [it, inserted] =
            implementation_->requests.try_emplace(object.sha256, std::make_shared<implementation::inflight>());
        leader = inserted;
        flight = it->second;
    }
    if (!leader)
    {
        std::unique_lock lock(implementation_->mutex);
        while (!flight->done)
        {
            if (cancellation.stop_requested())
                return failed("Remote immutable object transfer wait cancelled",
                              http::content_http_error_code::cancelled);
            implementation_->changed.wait_for(lock, std::chrono::milliseconds(25));
        }
        auto outcome = flight->outcome;
        lock.unlock();
        outcome.coalesced = true;
        outcome.downloaded = false;
        if (!outcome.succeeded()) return outcome;
        cache_error error;
        if (!load_verified_remote_object(*implementation_->cache, object, error))
        {
            outcome.cache_error = remote_object_cache_error::cache_failure;
            outcome.error = "Coalesced remote object missing after successful CAS publication";
            return outcome;
        }
        outcome.reused = true;
        return outcome;
    }

    remote_download_result result;
    try
    {
        result = implementation_->download(object, request, http_limits, cancellation, executor);
    }
    catch (const std::exception&)
    {
        result = failed("Remote immutable object transfer failed unexpectedly",
                        http::content_http_error_code::network_failure);
    }

    {
        std::lock_guard lock(implementation_->mutex);
        flight->outcome = result;
        flight->done = true;
        implementation_->requests.erase(object.sha256);
    }
    implementation_->changed.notify_all();
    return result;
}

} // namespace arc::assets
