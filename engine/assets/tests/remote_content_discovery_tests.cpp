#include <arc/assets/remote_content_discovery.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>

namespace
{

arc::assets::remote_content_expectations expectations()
{
    return {.project_guid = "11111111-2222-4333-8444-555555555555",
            .engine_version = "0.1.0",
            .base_build_id = "build-abc",
            .target = {.platform = "windows",
                       .architecture = "x86_64",
                       .renderer = "vulkan",
                       .cook_profile = "windows-x64-vulkan",
                       .texture_family = "bc"}};
}

std::string descriptor_fixture()
{
    const auto path = std::filesystem::path(__FILE__).parent_path() / "fixtures" / "remote-content-v1.json";
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

arc::assets::remote_release_fetch_executor stub_fetch(std::string json, unsigned int status = 200,
                                                      std::string redirect = {}, std::string content_length = {})
{
    return [json = std::move(json), status, redirect = std::move(redirect), content_length = std::move(content_length)](
               arc::jobs::job_system& jobs, arc::http::content_http_release_request request,
               arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.remote_content.discovery_test", .affinity = arc::jobs::job_affinity::io_thread},
            [json, status, redirect, content_length, request = std::move(request)]()
            {
                const auto byte_view = std::as_bytes(std::span(json.data(), json.size()));
                arc::http::content_http_response response{.status = status,
                                                          .content_length = content_length,
                                                          .redirect_location = redirect,
                                                          .body =
                                                              arc::io::file_buffer(byte_view.begin(), byte_view.end())};
                return arc::http::content_http_fetch_result{.response = std::move(response), .validation = {}};
            });
    };
}

} // namespace

TEST_CASE("C9.2d2 fetches current release and validates C9.1 identity before acquisition")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    const auto result = arc::assets::discover_remote_release(jobs, "https://content.example.org", expectations(), {},
                                                             {}, stub_fetch(descriptor_fixture()))
                            .get();
    REQUIRE(result.succeeded());
    REQUIRE(result.release);
    CHECK(result.release->generation == 5);
    CHECK(result.release->base_build_id == "build-abc");
    CHECK(result.release->chunks.size() == 1);
    CHECK(result.release->cook_manifest_name == "windows-x64-vulkan.arccookmanifest");
}

TEST_CASE("C9.2d2 refuses stale generation, wrong target and incomplete release")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    auto expected = expectations();
    expected.minimum_generation = 5;
    CHECK_FALSE(arc::assets::discover_remote_release(jobs, "https://content.example.org", expected, {}, {},
                                                     stub_fetch(descriptor_fixture()))
                    .get()
                    .succeeded());

    expected = expectations();
    expected.target.platform = "android";
    CHECK_FALSE(arc::assets::discover_remote_release(jobs, "https://content.example.org", expected, {}, {},
                                                     stub_fetch(descriptor_fixture()))
                    .get()
                    .succeeded());

    auto incomplete = descriptor_fixture();
    const auto position = incomplete.find("\"complete\": true");
    REQUIRE(position != std::string::npos);
    incomplete.replace(position, std::string("\"complete\": true").size(), "\"complete\": false");
    CHECK_FALSE(arc::assets::discover_remote_release(jobs, "https://content.example.org", expectations(), {}, {},
                                                     stub_fetch(incomplete))
                    .get()
                    .succeeded());
}

TEST_CASE("C9.2d2 cannot bypass transport validation using an injected executor")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    auto descriptor = descriptor_fixture();
    const auto redirected =
        arc::assets::discover_remote_release(jobs, "https://content.example.org", expectations(), {}, {},
                                             stub_fetch(descriptor, 302, "https://evil.example.org/current.json"))
            .get();
    CHECK_FALSE(redirected.succeeded());
    CHECK(redirected.transport_error == arc::http::content_http_error_code::rejected_redirect);

    const auto wrong_length = arc::assets::discover_remote_release(jobs, "https://content.example.org", expectations(),
                                                                   {}, {}, stub_fetch(descriptor, 200, {}, "1"))
                                  .get();
    CHECK_FALSE(wrong_length.succeeded());
    CHECK(wrong_length.transport_error == arc::http::content_http_error_code::inconsistent_length);

    const auto missing = arc::assets::discover_remote_release(jobs, "https://content.example.org", expectations(), {},
                                                              {}, stub_fetch(descriptor, 404))
                             .get();
    CHECK_FALSE(missing.succeeded());
    CHECK(missing.transport_error == arc::http::content_http_error_code::not_found);
}

TEST_CASE("C9.2d2 rejects invalid origin and pre-cancelled requests before fetching")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    auto fetcher = stub_fetch(descriptor_fixture());
    const auto invalid =
        arc::assets::discover_remote_release(jobs, "file:///tmp/current.json", expectations(), {}, {}, fetcher).get();
    CHECK_FALSE(invalid.succeeded());
    CHECK(invalid.transport_error == arc::http::content_http_error_code::invalid_origin);

    arc::jobs::cancellation_source source;
    REQUIRE(source.request_cancel());
    const auto cancelled = arc::assets::discover_remote_release(jobs, "https://content.example.org", expectations(), {},
                                                                source.token(), fetcher)
                               .get();
    CHECK_FALSE(cancelled.succeeded());
    CHECK(cancelled.transport_error == arc::http::content_http_error_code::cancelled);
}
