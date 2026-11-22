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
}
