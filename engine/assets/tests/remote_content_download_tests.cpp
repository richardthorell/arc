#include <arc/assets/remote_content_download.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{

arc::io::file_buffer bytes(std::string_view value)
{
    const auto span = std::as_bytes(std::span(value.data(), value.size()));
    return {span.begin(), span.end()};
}

arc::assets::remote_content_object object_for(std::string_view value)
{
    const auto body = bytes(value);
    const auto digest = arc::assets::to_string(arc::assets::hash_bytes(body));
    return {.path = "objects/sha256/" + digest, .sha256 = digest, .size = body.size()};
}

arc::http::content_http_fetch_result full_response(const arc::io::file_buffer& body)
{
    arc::http::content_http_response response{
        .status = 200, .content_length = std::to_string(body.size()), .body = body};
    return {.response = std::move(response), .validation = {}};
}

arc::http::content_http_fetch_result http_failure(arc::http::content_http_error_code code)
{
    return {.response = std::nullopt, .validation = {.code = code, .message = "Injected HTTP failure"}};
}

struct fixture
{
    std::filesystem::path root = std::filesystem::temp_directory_path() /
                                 ("arc-c9-transfers-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    arc::jobs::job_system jobs{
        {.worker_count = 4, .run_inline = false, .io_worker_count = 4, .enable_render_thread = false}};
    arc::assets::derived_data_cache cache{{.root = root / "cache"}};

    ~fixture()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

constexpr std::string_view origin = "https://content.example.org";

} // namespace

TEST_CASE("C9.2f2 coalesces concurrent immutable CAS misses to one HTTP request")
{
    fixture f;
    const auto body = bytes("deduplicated remote cooked object");
    const auto object = object_for("deduplicated remote cooked object");
    arc::assets::remote_download_limits policy;
    policy.maximum_attempts = 1;
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache, policy);
    std::atomic<int> requests{};

    arc::assets::remote_generation_fetch_executor executor =
        [&requests, body](arc::jobs::job_system& jobs, arc::http::content_http_request request,
                          arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.download.coalesce", .affinity = arc::jobs::job_affinity::io_thread},
            [&requests, body, request = std::move(request)]() -> arc::http::content_http_fetch_result
            {
                ++requests;
                std::this_thread::sleep_for(std::chrono::milliseconds(125));
                if (request.object_size != body.size())
                    return http_failure(arc::http::content_http_error_code::invalid_request);
                return full_response(body);
            });
    };

    std::promise<void> start;
    auto go = start.get_future().share();
    auto a = std::async(std::launch::async,
                        [&]
                        {
                            go.wait();
                            return downloads.ensure(object, std::string(origin), {}, {}, executor);
                        });
    auto b = std::async(std::launch::async,
                        [&]
                        {
                            go.wait();
                            return downloads.ensure(object, std::string(origin), {}, {}, executor);
                        });
    start.set_value();
    auto first = a.get();
    auto second = b.get();
    REQUIRE(first.succeeded());
    REQUIRE(second.succeeded());
    CHECK(requests.load() == 1);
    const auto coalesced = downloads.metrics();
    CHECK(coalesced.calls == 2);
    CHECK(coalesced.coalesced_calls == 1);
    CHECK(coalesced.network_attempts == 1);
    CHECK(coalesced.completed_downloads == 1);
    CHECK(coalesced.downloaded_bytes == object.size);
    CHECK(coalesced.peak_active_requests == 1);
    CHECK(coalesced.active_requests == 0);
    CHECK(coalesced.active_declared_bytes == 0);
    CHECK(first.downloaded != second.downloaded);
    CHECK(first.coalesced != second.coalesced);
    CHECK(first.reused != second.reused);
    arc::assets::cache_error error;
    auto cached = arc::assets::load_verified_remote_object(f.cache, object, error);
    REQUIRE(cached);
    CHECK(*cached == body);

    auto again = downloads.ensure(object, std::string(origin), {}, {}, executor);
    REQUIRE(again.succeeded());
    CHECK(again.reused);
    CHECK(requests.load() == 1);
    const auto reuse = downloads.metrics();
    CHECK(reuse.calls == 3);
    CHECK(reuse.cache_hits == 1);
    CHECK(reuse.network_attempts == 1);
}

TEST_CASE("C9.2f2 retries only transient transport failure and reuses verified CAS")
{
    fixture f;
    const auto body = bytes("retriable content transfer");
    const auto object = object_for("retriable content transfer");
    arc::assets::remote_download_limits policy;
    policy.maximum_attempts = 3;
    policy.initial_backoff = std::chrono::milliseconds{0};
    policy.maximum_backoff = std::chrono::milliseconds{0};
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache, policy);
    std::atomic<int> requests{};

    arc::assets::remote_generation_fetch_executor executor =
        [&requests, body](arc::jobs::job_system& jobs, arc::http::content_http_request, arc::http::content_http_limits,
                          arc::jobs::cancellation_token)
    {
        return jobs.submit_future({.name = "assets.download.transient", .affinity = arc::jobs::job_affinity::io_thread},
                                  [&requests, body]() -> arc::http::content_http_fetch_result
                                  {
                                      if (++requests <= 2)
                                          return http_failure(arc::http::content_http_error_code::rate_limited);
                                      return full_response(body);
                                  });
    };

    auto result = downloads.ensure(object, std::string(origin), {}, {}, executor);
    REQUIRE(result.succeeded());
    CHECK(result.downloaded);
    CHECK(result.attempts == 3);
    CHECK(requests.load() == 3);
    auto cached = downloads.ensure(object, std::string(origin), {}, {}, executor);
    CHECK(cached.succeeded());
    CHECK(cached.reused);
    CHECK(cached.attempts == 0);
    CHECK(requests.load() == 3);
    const auto stats = downloads.metrics();
    CHECK(stats.calls == 2);
    CHECK(stats.network_attempts == 3);
    CHECK(stats.retry_attempts == 2);
    CHECK(stats.cache_hits == 1);
    CHECK(stats.completed_downloads == 1);
    CHECK(stats.failed_calls == 0);
}

TEST_CASE("C9.2f2 never retries corrupt content, unauthorized responses or malformed object URLs")
{
    fixture f;
    const auto body = bytes("verified immutable payload");
    const auto object = object_for("verified immutable payload");
    arc::assets::remote_download_limits policy;
    policy.maximum_attempts = 5;
    policy.initial_backoff = std::chrono::milliseconds{0};
    policy.maximum_backoff = std::chrono::milliseconds{0};
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache, policy);
    std::atomic<int> requests{};

    arc::assets::remote_generation_fetch_executor corrupt =
        [&requests, body](arc::jobs::job_system& jobs, arc::http::content_http_request, arc::http::content_http_limits,
                          arc::jobs::cancellation_token)
    {
        return jobs.submit_future({.name = "assets.download.corrupt", .affinity = arc::jobs::job_affinity::io_thread},
                                  [&requests, body]() mutable -> arc::http::content_http_fetch_result
                                  {
                                      ++requests;
                                      auto invalid = body;
                                      invalid.back() ^= std::byte{0x01};
                                      return full_response(invalid);
                                  });
    };
    auto rejected = downloads.ensure(object, std::string(origin), {}, {}, corrupt);
    CHECK_FALSE(rejected.succeeded());
    CHECK(rejected.cache_error == arc::assets::remote_object_cache_error::hash_mismatch);
    CHECK(rejected.attempts == 1);
    CHECK(requests.load() == 1);

    arc::assets::remote_generation_fetch_executor unauthorized =
        [&requests](arc::jobs::job_system& jobs, arc::http::content_http_request, arc::http::content_http_limits,
                    arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.download.unauthorized", .affinity = arc::jobs::job_affinity::io_thread},
            [&requests]() -> arc::http::content_http_fetch_result
            {
                ++requests;
                return http_failure(arc::http::content_http_error_code::unauthorized);
            });
    };
    auto denied = downloads.ensure(object, std::string(origin), {}, {}, unauthorized);
    CHECK_FALSE(denied.succeeded());
    CHECK(denied.transport_error == arc::http::content_http_error_code::unauthorized);
    CHECK(denied.attempts == 1);
    CHECK(requests.load() == 2);

    auto invalid = object;
    invalid.path = "../source";
    auto bad = downloads.ensure(invalid, std::string(origin), {}, {}, unauthorized);
    CHECK_FALSE(bad.succeeded());
    CHECK(bad.transport_error == arc::http::content_http_error_code::invalid_request);
    CHECK(requests.load() == 2);
    arc::assets::cache_error error;
    CHECK_FALSE(arc::assets::load_verified_remote_object(f.cache, object, error));
    const auto stats = downloads.metrics();
    CHECK(stats.network_attempts == 2);
    CHECK(stats.integrity_failures == 1);
    CHECK(stats.failed_calls == 3);
}

TEST_CASE("C9.2f2 bounds concurrent GET count and aggregate declared in-flight bytes")
{
    fixture f;
    arc::assets::remote_download_limits policy;
    policy.maximum_concurrent_requests = 4;
    policy.maximum_inflight_bytes = 18; // Each object is 12 bytes: at most one at a time.
    policy.maximum_attempts = 1;
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache, policy);
    const std::vector<std::string> payloads{"content-A1234", "content-B1234", "content-C1234"};
    std::atomic<int> active{};
    std::atomic<int> peak{};
    std::atomic<int> requests{};

    arc::assets::remote_generation_fetch_executor executor =
        [&active, &peak, &requests, &payloads](arc::jobs::job_system& jobs, arc::http::content_http_request request,
                                               arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future({.name = "assets.download.bound", .affinity = arc::jobs::job_affinity::io_thread},
                                  [&active, &peak, &requests, &payloads,
                                   request = std::move(request)]() -> arc::http::content_http_fetch_result
                                  {
                                      const int now = ++active;
                                      peak.store(std::max(peak.load(), now));
                                      ++requests;
                                      std::this_thread::sleep_for(std::chrono::milliseconds(50));
                                      --active;
                                      for (const auto& payload : payloads)
                                          if (object_for(payload).path == request.object_path)
                                              return full_response(bytes(payload));
                                      return http_failure(arc::http::content_http_error_code::not_found);
                                  });
    };

    std::vector<std::future<arc::assets::remote_download_result>> running;
    for (const auto& payload : payloads)
    {
        const auto object = object_for(payload);
        running.push_back(std::async(std::launch::async, [&, object]
                                     { return downloads.ensure(object, std::string(origin), {}, {}, executor); }));
    }
    for (auto& task : running)
        REQUIRE(task.get().succeeded());
    CHECK(requests.load() == 3);
    CHECK(peak.load() == 1); // Byte budget prevents a second 12-byte body.
    const auto stats = downloads.metrics();
    CHECK(stats.calls == 3);
    CHECK(stats.network_attempts == 3);
    CHECK(stats.peak_active_requests == 1);
    CHECK(stats.peak_declared_bytes == payloads.front().size());
    CHECK(stats.active_declared_bytes == 0);
}

TEST_CASE("C9.2f2 follower cancellation never cancels an unrelated in-flight leader")
{
    fixture f;
    const auto body = bytes("cancellable waiter");
    const auto object = object_for("cancellable waiter");
    arc::assets::remote_download_limits policy;
    policy.maximum_attempts = 1;
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache, policy);
    std::promise<void> started;
    auto entered = started.get_future();
    std::promise<void> release_leader;
    const auto can_finish = release_leader.get_future().share();
    std::atomic<int> requests{};

    arc::assets::remote_generation_fetch_executor executor =
        [&requests, &started, can_finish, body](arc::jobs::job_system& jobs, arc::http::content_http_request,
                                                arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.download.cancel_waiter", .affinity = arc::jobs::job_affinity::io_thread},
            [&requests, &started, can_finish, body]() -> arc::http::content_http_fetch_result
            {
                ++requests;
                started.set_value();
                can_finish.wait();
                return full_response(body);
            });
    };

    auto leader =
        std::async(std::launch::async, [&] { return downloads.ensure(object, std::string(origin), {}, {}, executor); });
    entered.wait();
    arc::jobs::cancellation_source cancel_waiter;
    auto follower =
        std::async(std::launch::async,
                   [&] { return downloads.ensure(object, std::string(origin), {}, cancel_waiter.token(), executor); });
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    CHECK(cancel_waiter.request_cancel());
    const auto cancelled_status = follower.wait_for(std::chrono::seconds(2));
    release_leader.set_value(); // Always unblock the IO job before checking cancellation.
    REQUIRE(cancelled_status == std::future_status::ready);
    const auto cancelled = follower.get();
    CHECK_FALSE(cancelled.succeeded());
    CHECK(cancelled.transport_error == arc::http::content_http_error_code::cancelled);
    const auto completed = leader.get();
    REQUIRE(completed.succeeded());
    CHECK(requests.load() == 1);
    const auto stats = downloads.metrics();
    CHECK(stats.calls == 2);
    CHECK(stats.network_attempts == 1);
    CHECK(stats.coalesced_calls == 1);
    CHECK(stats.cancelled_calls == 1);
    CHECK(stats.completed_downloads == 1);
}

TEST_CASE("C9.2h rejects truncated HTTP objects without retry, then clean restart heals a miss")
{
    fixture f;
    const auto payload = bytes("verified restartable immutable content");
    const auto object = object_for("verified restartable immutable content");
    arc::assets::remote_download_limits policy;
    policy.maximum_attempts = 3;
    policy.initial_backoff = std::chrono::milliseconds{0};
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache, policy);
    std::atomic<int> calls{};
    arc::assets::remote_generation_fetch_executor executor =
        [&calls, payload](arc::jobs::job_system& jobs, arc::http::content_http_request,
                          arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.download.truncated_restart", .affinity = arc::jobs::job_affinity::io_thread},
            [&calls, payload]() -> arc::http::content_http_fetch_result
            {
                auto response = full_response(payload);
                if (++calls == 1) response.response->body.pop_back();
                return response;
            });
    };

    const auto bad = downloads.ensure(object, std::string(origin), {}, {}, executor);
    CHECK_FALSE(bad.succeeded());
    CHECK(bad.cache_error == arc::assets::remote_object_cache_error::incompatible_response);
    CHECK(bad.attempts == 1);
    arc::assets::cache_error error;
    CHECK_FALSE(arc::assets::load_verified_remote_object(f.cache, object, error));

    const auto repaired = downloads.ensure(object, std::string(origin), {}, {}, executor);
    REQUIRE(repaired.succeeded());
    CHECK(repaired.downloaded);
    CHECK(calls.load() == 2);
    auto cached = arc::assets::load_verified_remote_object(f.cache, object, error);
    REQUIRE(cached);
    CHECK(*cached == payload);

    const auto stats = downloads.metrics();
    CHECK(stats.calls == 2);
    CHECK(stats.network_attempts == 2);
    CHECK(stats.retry_attempts == 0); // Corrupt framing is not a retryable timeout.
    CHECK(stats.integrity_failures == 1);
    CHECK(stats.failed_calls == 1);
    CHECK(stats.completed_downloads == 1);
    CHECK(stats.downloaded_bytes == object.size);
}

TEST_CASE("C9.2h forced diagnostic refresh does not discard last verified CAS when HTTP returns 404")
{
    fixture f;
    const auto payload = bytes("existing trusted cached asset");
    const auto object = object_for("existing trusted cached asset");
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache);
    std::atomic<int> requests{};
    arc::assets::remote_generation_fetch_executor fetch =
        [&requests, payload](arc::jobs::job_system& jobs, arc::http::content_http_request,
                             arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.download.refresh", .affinity = arc::jobs::job_affinity::io_thread},
            [&requests, payload]() -> arc::http::content_http_fetch_result
            {
                const auto index = ++requests;
                if (index > 1)
                    return http_failure(arc::http::content_http_error_code::not_found);
                return full_response(payload);
            });
    };

    REQUIRE(downloads.ensure(object, std::string(origin), {}, {}, fetch).succeeded());
    const auto refresh = downloads.ensure(object, std::string(origin), {}, {}, fetch, true);
    CHECK_FALSE(refresh.succeeded());
    CHECK(refresh.transport_error == arc::http::content_http_error_code::not_found);
    CHECK(refresh.attempts == 1);
    CHECK(requests.load() == 2);
    arc::assets::cache_error error;
    const auto still_cached = arc::assets::load_verified_remote_object(f.cache, object, error);
    REQUIRE(still_cached);
    CHECK(*still_cached == payload);

    const auto offline = downloads.ensure(object, std::string(origin), {}, {}, fetch);
    CHECK(offline.succeeded());
    CHECK(offline.reused);
    CHECK(requests.load() == 2);
    const auto stats = downloads.metrics();
    CHECK(stats.network_attempts == 2);
    CHECK(stats.cache_hits == 1);
    CHECK(stats.failed_calls == 1);
    CHECK(stats.completed_downloads == 1);
}

TEST_CASE("C9.2h cancelling a distinct queued object keeps the running fetch and cache intact")
{
    fixture f;
    const auto first = object_for("first-queued-content");
    const auto second = object_for("second-queued-content");
    arc::assets::remote_download_limits policy;
    policy.maximum_concurrent_requests = 1;
    policy.maximum_inflight_bytes = 64;
    policy.maximum_attempts = 1;
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache, policy);

    std::promise<void> running;
    auto entered = running.get_future();
    std::promise<void> release;
    const auto release_signal = release.get_future().share();
    std::atomic<int> requests{};
    arc::assets::remote_generation_fetch_executor fetch =
        [&running, release_signal, &requests](arc::jobs::job_system& jobs, arc::http::content_http_request request,
                                              arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.download.queued_cancel", .affinity = arc::jobs::job_affinity::io_thread},
            [&running, release_signal, &requests, request]() -> arc::http::content_http_fetch_result
            {
                ++requests;
                running.set_value();
                release_signal.wait();
                if (request.object_path == object_for("first-queued-content").path)
                    return full_response(bytes("first-queued-content"));
                return full_response(bytes("second-queued-content"));
            });
    };

    auto leader = std::async(std::launch::async, [&]
                             { return downloads.ensure(first, std::string(origin), {}, {}, fetch); });
    entered.wait();
    arc::jobs::cancellation_source cancel;
    auto queued = std::async(std::launch::async, [&]
                             { return downloads.ensure(second, std::string(origin), {}, cancel.token(), fetch); });
    // Allow the queued thread to enter its budget wait without relying on a
    // fragile scheduler ordering; the first request remains held explicitly.
    bool observed_queue = false;
    for (int n = 0; n < 100; ++n)
    {
        if (downloads.metrics().queued_requests > 0)
        {
            observed_queue = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(cancel.request_cancel());
    const auto stopped = queued.wait_for(std::chrono::seconds(2));
    release.set_value(); // Unblock before REQUIRE to avoid leaking a waiting job.
    REQUIRE(stopped == std::future_status::ready);
    CHECK(observed_queue);
    const auto cancelled = queued.get();
    CHECK_FALSE(cancelled.succeeded());
    CHECK(cancelled.transport_error == arc::http::content_http_error_code::cancelled);
    CHECK(leader.get().succeeded());
    CHECK(requests.load() == 1);

    const auto stats = downloads.metrics();
    CHECK(stats.peak_active_requests == 1);
    CHECK(stats.peak_queued_requests >= 1);
    CHECK(stats.active_requests == 0);
    CHECK(stats.queued_requests == 0);
    CHECK(stats.network_attempts == 1);
    CHECK(stats.cancelled_calls == 1);
}

TEST_CASE("C9.2h metrics reject oversized objects without leaking a sensitive origin")
{
    fixture f;
    arc::assets::remote_download_limits policy;
    policy.maximum_inflight_bytes = 16;
    arc::assets::remote_object_download_coordinator downloads(f.jobs, f.cache, policy);
    const auto object = object_for("this immutable object is too large for the budget");
    const auto denied = downloads.ensure(object, std::string(origin), {});
    CHECK_FALSE(denied.succeeded());
    CHECK(denied.transport_error == arc::http::content_http_error_code::oversized_response);
    const auto stats = downloads.metrics();
    CHECK(stats.calls == 1);
    CHECK(stats.failed_calls == 1);
    CHECK(stats.network_attempts == 0);
    CHECK(stats.active_requests == 0);
    CHECK(stats.active_declared_bytes == 0);
}
