#include <arc/io/content_http.h>

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("C9.2b HTTP fetch validates request before network access")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    arc::io::content_http_request request{
        .origin = "file:///etc/passwd", .object_path = "objects/sha256/" + std::string(64, 'a'),
        .object_size = 4};
    auto result = arc::io::fetch_content_http(jobs, request).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::io::content_http_error_code::invalid_origin);

    request.origin = "https://content.example.org";
    request.object_path = "../source";
    result = arc::io::fetch_content_http(jobs, request).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::io::content_http_error_code::invalid_request);

    request.object_path = "objects/sha256/" + std::string(64, 'a');
    request.range_start = 3;
    request.range_length = 2;
    result = arc::io::fetch_content_http(jobs, request).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::io::content_http_error_code::invalid_range);
}

#if !defined(_WIN32)
TEST_CASE("C9.2b unsupported platforms fail explicitly without network access")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    arc::io::content_http_request request{
        .origin = "https://content.example.org", .object_path = "objects/sha256/" + std::string(64, 'a'),
        .object_size = 4};
    const auto result = arc::io::fetch_content_http(jobs, std::move(request)).get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.validation.code == arc::io::content_http_error_code::transport_unavailable);
}
#endif
