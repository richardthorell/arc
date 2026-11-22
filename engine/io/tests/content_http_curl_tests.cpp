#include <arc/io/content_http.h>

#if defined(__linux__) && defined(ARC_CONTENT_HTTP_CURL) && !defined(__ANDROID__)

#include <catch2/catch_test_macros.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace
{

struct loopback_server
{
    int listener{-1};
    std::uint16_t port{};
    std::atomic_bool saw_path{};
    std::atomic_bool saw_range{};
    std::atomic_bool saw_no_cache{};
    std::jthread worker;

    loopback_server(std::string expected_path, std::string response)
    {
        listener = socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(listener >= 0);
        const int reuse = 1;
        REQUIRE(setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == 0);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        REQUIRE(bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
        socklen_t length = sizeof(address);
        REQUIRE(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) == 0);
        port = ntohs(address.sin_port);
        REQUIRE(listen(listener, 1) == 0);

        worker = std::jthread(
            [this, path = std::move(expected_path), reply = std::move(response)]
            {
                const int client = accept(listener, nullptr, nullptr);
                if (client < 0) return;
                timeval timeout{.tv_sec = 5, .tv_usec = 0};
                static_cast<void>(setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)));
                std::string received;
                char buffer[4096];
                while (received.find("\r\n\r\n") == std::string::npos && received.size() < 16384)
                {
                    const auto count = recv(client, buffer, sizeof(buffer), 0);
                    if (count <= 0) break;
                    received.append(buffer, static_cast<std::size_t>(count));
                }
                saw_path.store(received.find("GET " + path + " HTTP/") != std::string::npos);
                saw_range.store(received.find("Range: bytes=2-4") != std::string::npos);
                saw_no_cache.store(received.find("Cache-Control: no-cache, no-store") != std::string::npos);
                for (std::size_t offset = 0; offset < reply.size();)
                {
                    const auto count = send(client, reply.data() + offset, reply.size() - offset, MSG_NOSIGNAL);
                    if (count <= 0) break;
                    offset += static_cast<std::size_t>(count);
                }
                shutdown(client, SHUT_RDWR);
                close(client);
            });
    }

    ~loopback_server()
    {
        if (listener >= 0)
        {
            shutdown(listener, SHUT_RDWR);
            close(listener);
        }
        if (worker.joinable()) worker.join();
    }

    [[nodiscard]] std::string origin() const
    {
        return "http://127.0.0.1:" + std::to_string(port);
    }
};

std::string response(unsigned int status, std::string_view reason, std::string body, std::string headers = {})
{
    return "HTTP/1.1 " + std::to_string(status) + " " + std::string(reason) + "\r\n" +
           "Content-Length: " + std::to_string(body.size()) + "\r\n" + std::move(headers) +
           "Connection: close\r\n\r\n" + body;
}

arc::io::content_http_request immutable_request(const loopback_server& server)
{
    return {.origin = server.origin(), .object_path = "objects/sha256/" + std::string(64, 'a'), .object_size = 6};
}

arc::io::content_http_limits development_limits()
{
    arc::io::content_http_limits limits;
    limits.allow_loopback_http = true;
    limits.timeout = std::chrono::seconds(5);
    return limits;
}

} // namespace

TEST_CASE("C9.2e1 Linux curl fetches and frames a complete immutable object over loopback HTTP")
{
    loopback_server server("/objects/sha256/" + std::string(64, 'a'),
                           response(200, "OK", "abcdef", "ETag: \"digest\"\r\n"));
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    auto request = immutable_request(server);
    request.if_match_etag = "\"digest\"";
    const auto fetched = arc::io::fetch_content_http(jobs, request, development_limits()).get();
    REQUIRE(fetched.succeeded());
    CHECK(fetched.response->status == 200);
    CHECK(fetched.response->body.size() == 6);
    CHECK(fetched.response->etag == "\"digest\"");
    CHECK(server.saw_path.load());
}

TEST_CASE("C9.2e1 Linux curl sends bounded range GET and validates Content-Range")
{
    loopback_server server("/objects/sha256/" + std::string(64, 'a'),
                           response(206, "Partial Content", "cde", "Content-Range: bytes 2-4/6\r\n"));
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    auto request = immutable_request(server);
    request.range_start = 2;
    request.range_length = 3;
    const auto fetched = arc::io::fetch_content_http(jobs, request, development_limits()).get();
    REQUIRE(fetched.succeeded());
    CHECK(fetched.response->body.size() == 3);
    CHECK(server.saw_path.load());
    CHECK(server.saw_range.load());
}

TEST_CASE("C9.2e1 Linux curl retrieves mutable current release with no-cache and bounded bytes")
{
    const std::string guid = "11111111-2222-4333-8444-555555555555";
    loopback_server server("/releases/" + guid + "/windows-x64-vulkan/current.json",
                           response(200, "OK", "{\"generation\":6}"));
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    arc::io::content_http_release_request request{
        .origin = server.origin(), .project_guid = guid, .target_profile = "windows-x64-vulkan"};
    const auto fetched = arc::io::fetch_content_http_release(jobs, request, development_limits()).get();
    REQUIRE(fetched.succeeded());
    CHECK(fetched.response->status == 200);
    CHECK(server.saw_path.load());
    CHECK(server.saw_no_cache.load());
}

TEST_CASE("C9.2e1 Linux curl refuses redirects and oversized objects before trusting content")
{
    SECTION("Untrusted redirect")
    {
        loopback_server server("/objects/sha256/" + std::string(64, 'a'),
                               response(302, "Found", "", "Location: https://evil.invalid/payload\r\n"));
        arc::jobs::job_system jobs(
            {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
        auto fetched = arc::io::fetch_content_http(jobs, immutable_request(server), development_limits()).get();
        CHECK_FALSE(fetched.succeeded());
        CHECK(fetched.validation.code == arc::io::content_http_error_code::rejected_redirect);
    }
    SECTION("Excess body bytes")
    {
        loopback_server server("/objects/sha256/" + std::string(64, 'a'), response(200, "OK", "abcdef extra bytes"));
        arc::jobs::job_system jobs(
            {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
        auto fetched = arc::io::fetch_content_http(jobs, immutable_request(server), development_limits()).get();
        CHECK_FALSE(fetched.succeeded());
        CHECK(fetched.validation.code == arc::io::content_http_error_code::oversized_response);
    }
}

TEST_CASE("C9.2e1 Linux curl reports missing objects and pre-cancelled requests")
{
    SECTION("404")
    {
        loopback_server server("/objects/sha256/" + std::string(64, 'a'), response(404, "Not Found", ""));
        arc::jobs::job_system jobs(
            {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
        auto fetched = arc::io::fetch_content_http(jobs, immutable_request(server), development_limits()).get();
        CHECK_FALSE(fetched.succeeded());
        CHECK(fetched.validation.code == arc::io::content_http_error_code::not_found);
    }
    SECTION("Pre-cancelled")
    {
        arc::jobs::job_system jobs(
            {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
        arc::jobs::cancellation_source cancellation;
        REQUIRE(cancellation.request_cancel());
        arc::io::content_http_request request{.origin = "https://content.example.org",
                                              .object_path = "objects/sha256/" + std::string(64, 'a'),
                                              .object_size = 6};
        const auto fetched =
            arc::io::fetch_content_http(jobs, request, development_limits(), cancellation.token()).get();
        CHECK_FALSE(fetched.succeeded());
        CHECK(fetched.validation.code == arc::io::content_http_error_code::cancelled);
    }
}

#endif
