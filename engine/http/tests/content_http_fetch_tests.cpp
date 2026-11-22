#include <arc/http/content_http.h>

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("C9.2b HTTP fetch validates request before network access")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    arc::http::content_http_request request{
        .origin = "file:///etc/passwd", .object_path = "objects/sha256/" + std::string(64, 'a'), .object_size = 4};
    auto result = arc::http::fetch_content_http(jobs, request).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::http::content_http_error_code::invalid_origin);

    request.origin = "https://content.example.org";
    request.object_path = "../source";
    result = arc::http::fetch_content_http(jobs, request).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::http::content_http_error_code::invalid_request);

    request.object_path = "objects/sha256/" + std::string(64, 'a');
    request.range_start = 3;
    request.range_length = 2;
    result = arc::http::fetch_content_http(jobs, request).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::http::content_http_error_code::invalid_range);
}

#if !defined(_WIN32) && !defined(ARC_CONTENT_HTTP_CURL)
TEST_CASE("C9.2b unsupported platforms fail explicitly without network access")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    arc::http::content_http_request request{.origin = "https://content.example.org",
                                          .object_path = "objects/sha256/" + std::string(64, 'a'),
                                          .object_size = 4};
    const auto result = arc::http::fetch_content_http(jobs, std::move(request)).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::http::content_http_error_code::transport_unavailable);
}
#endif

TEST_CASE("C9.2d2 release lookup preflight restricts origin, project and target path")
{
    using namespace arc::io;
    content_http_release_request request{.origin = "https://content.example.org",
                                         .project_guid = "11111111-2222-4333-8444-555555555555",
                                         .target_profile = "windows-x64-vulkan"};
    CHECK(validate_content_http_release_request(request, {}).succeeded());
    request.origin = "file:///etc/passwd";
    CHECK(validate_content_http_release_request(request, {}).code == content_http_error_code::invalid_origin);
    request.origin = "http://192.168.1.10:9000";
    content_http_limits dev;
    dev.allow_loopback_http = true;
    CHECK(validate_content_http_release_request(request, dev).code == content_http_error_code::invalid_origin);
    request.origin = "http://127.0.0.1:9000";
    CHECK(validate_content_http_release_request(request, dev).succeeded());
    CHECK_FALSE(validate_content_http_release_request(request, {}).succeeded());
    request.origin = "https://content.example.org";
    request.target_profile = "../windows";
    CHECK(validate_content_http_release_request(request, {}).code == content_http_error_code::invalid_request);
    request.target_profile = "windows%2Fmalicious";
    CHECK(validate_content_http_release_request(request, {}).code == content_http_error_code::invalid_request);
    request.target_profile = "windows-x64-vulkan";
    request.project_guid = "11111111-2222-4333-8444-55555555555x";
    CHECK(validate_content_http_release_request(request, {}).code == content_http_error_code::invalid_request);
}

TEST_CASE("C9.2d2 release response has strict status, redirect and eight-MiB size bounds")
{
    using namespace arc::io;
    content_http_release_request request{.origin = "https://content.example.org",
                                         .project_guid = "11111111-2222-4333-8444-555555555555",
                                         .target_profile = "windows-x64-vulkan"};
    content_http_response response{.status = 200, .content_length = "2"};
    response.body = {std::byte{0x7B}, std::byte{0x7D}};
    CHECK(validate_content_http_release_response(request, response, {}).succeeded());

    response.content_length = "999";
    CHECK(validate_content_http_release_response(request, response, {}).code ==
          content_http_error_code::inconsistent_length);
    response.content_length.clear(); // chunked transfer is valid if bounded
    CHECK(validate_content_http_release_response(request, response, {}).succeeded());
    response.redirect_location = "https://other.example.org/current.json";
    CHECK(validate_content_http_release_response(request, response, {}).code ==
          content_http_error_code::rejected_redirect);
    response.redirect_location.clear();
    response.status = 404;
    CHECK(validate_content_http_release_response(request, response, {}).code == content_http_error_code::not_found);
    response.status = 200;
    response.content_range = "bytes 0-1/2";
    CHECK(validate_content_http_release_response(request, response, {}).code == content_http_error_code::invalid_range);
    response.content_range.clear();
    response.body.resize(8u * 1024u * 1024u + 1u);
    CHECK(validate_content_http_release_response(request, response, {}).code ==
          content_http_error_code::oversized_response);
}

#if !defined(_WIN32) && !defined(ARC_CONTENT_HTTP_CURL)
TEST_CASE("C9.2d2 release discovery reports unsupported TLS backend rather than inventing content")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    arc::http::content_http_release_request request{.origin = "https://content.example.org",
                                                  .project_guid = "11111111-2222-4333-8444-555555555555",
                                                  .target_profile = "windows-x64-vulkan"};
    const auto result = arc::http::fetch_content_http_release(jobs, request).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::http::content_http_error_code::transport_unavailable);
}
#endif
