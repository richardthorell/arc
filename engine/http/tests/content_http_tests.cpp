#include <arc/http/content_http.h>

#include <catch2/catch_test_macros.hpp>

#include <string>

namespace
{

arc::http::content_http_request request()
{
    return {.origin = "https://content.example.org",
            .object_path = "objects/sha256/" + std::string(64, 'a'),
            .object_size = 8};
}

arc::http::content_http_response response()
{
    arc::http::content_http_response result;
    result.status = 200;
    result.content_length = "8";
    result.etag = "\"v1\"";
    result.body.resize(8);
    return result;
}

} // namespace

TEST_CASE("C9.2a immutable HTTP object validates strict successful full and ranged responses")
{
    const auto full = request();
    CHECK(arc::http::validate_content_http_request(full, {}).succeeded());
    CHECK(arc::http::validate_content_http_response(full, response(), {}).succeeded());
    auto partial = full;
    partial.range_start = 2;
    partial.range_length = 3;
    auto received = response();
    received.status = 206;
    received.content_length = "3";
    received.content_range = "bytes 2-4/8";
    received.body.resize(3);
    CHECK(arc::http::validate_content_http_response(partial, received, {}).succeeded());
}

TEST_CASE("C9.2a rejects unsafe origins and malformed immutable object addresses")
{
    auto req = request();
    for (const auto* origin :
         {"http://content.example.org", "file:///tmp", "https://user@content.example.org",
          "https://content.example.org/path", "https://content.example.org#x", "https://content.example.org:0",
          "https://content.example.org:99999", "http://192.168.1.2", "https://[::2]"})
    {
        req.origin = origin;
        CHECK_FALSE(arc::http::validate_content_http_request(req, {}).succeeded());
    }
    req.origin = "http://127.0.0.1:8123";
    CHECK_FALSE(arc::http::validate_content_http_request(req, {}).succeeded());
    CHECK(arc::http::validate_content_http_request(req, {.allow_loopback_http = true}).succeeded());
    req = request();
    req.object_path = "../secrets";
    CHECK_FALSE(arc::http::validate_content_http_request(req, {}).succeeded());
}

TEST_CASE("C9.2a rejects mismatched range boundaries, ignored ranges and oversized replies")
{
    auto req = request();
    req.range_start = 3;
    req.range_length = 2;
    auto received = response();
    CHECK_FALSE(arc::http::validate_content_http_response(req, received, {}).succeeded());
    received.status = 206;
    received.content_length = "2";
    received.body.resize(2);
    received.content_range = "bytes 3-4/9";
    CHECK_FALSE(arc::http::validate_content_http_response(req, received, {}).succeeded());
    received.content_range = "bytes 3-4/8";
    CHECK(arc::http::validate_content_http_response(req, received, {}).succeeded());
    received.content_length = "3";
    CHECK_FALSE(arc::http::validate_content_http_response(req, received, {}).succeeded());

    req = request();
    CHECK_FALSE(arc::http::validate_content_http_request(req, {.maximum_object_bytes = 4}).succeeded());
    req.range_start = 7;
    req.range_length = 2;
    CHECK_FALSE(arc::http::validate_content_http_request(req, {}).succeeded());
}

TEST_CASE("C9.2a rejects redirects and untrusted or changed ETags")
{
    auto req = request();
    auto received = response();
    received.status = 302;
    received.redirect_location = "https://elsewhere.invalid/object";
    CHECK_FALSE(arc::http::validate_content_http_response(req, received, {}).succeeded());
    received = response();
    req.if_match_etag = "\"expected\"";
    CHECK_FALSE(arc::http::validate_content_http_response(req, received, {}).succeeded());
    received.etag = "W/\"expected\"";
    CHECK_FALSE(arc::http::validate_content_http_response(req, received, {}).succeeded());
    received.etag = "\"expected\"";
    CHECK(arc::http::validate_content_http_response(req, received, {}).succeeded());
}

TEST_CASE("L1.3 Development Bearer pairing is restricted to explicit loopback content fetches")
{
    const auto token = std::string(64, 'f');
    arc::http::content_http_limits limits;
    limits.allow_loopback_http = true;
    limits.development_bearer_token = token;
    auto object = request();
    object.origin = "http://127.0.0.1:8123";
    CHECK(arc::http::validate_content_http_request(object, limits).succeeded());
    const arc::http::content_http_release_request descriptor{
        .origin = object.origin,
        .project_guid = "11111111-1111-1111-1111-111111111111",
        .target_profile = "windows-x64-vulkan"};
    CHECK(arc::http::validate_content_http_release_request(descriptor, limits).succeeded());

    object.origin = "https://content.example.org";
    CHECK_FALSE(arc::http::validate_content_http_request(object, limits).succeeded());
    object.origin = "http://localhost:8123";
    CHECK_FALSE(arc::http::validate_content_http_request(object, limits).succeeded());
    object.origin = "http://127.0.0.1:8123/path";
    CHECK_FALSE(arc::http::validate_content_http_request(object, limits).succeeded());
    object.origin = "http://127.0.0.1:8123";
    limits.allow_loopback_http = false;
    CHECK_FALSE(arc::http::validate_content_http_request(object, limits).succeeded());
    limits.allow_loopback_http = true;
    limits.development_bearer_token = "not-a-valid-token";
    CHECK_FALSE(arc::http::validate_content_http_request(object, limits).succeeded());
    const auto invalid_descriptor = arc::http::content_http_release_request{
        .origin = object.origin,
        .project_guid = descriptor.project_guid,
        .target_profile = descriptor.target_profile};
    CHECK_FALSE(arc::http::validate_content_http_release_request(invalid_descriptor, limits).succeeded());
}
