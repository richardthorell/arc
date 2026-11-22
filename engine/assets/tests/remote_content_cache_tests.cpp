#include <arc/assets/remote_content_cache.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace
{

arc::io::file_buffer bytes(std::string_view text)
{
    const auto view = std::as_bytes(std::span(text.data(), text.size()));
    return {view.begin(), view.end()};
}

struct fixture
{
    arc::io::file_buffer body = bytes("immutable cooked bytes");
    arc::assets::content_hash hash = arc::assets::hash_bytes(body);
    arc::assets::remote_content_object object{.path = "objects/sha256/" + arc::assets::to_string(hash),
                                              .sha256 = arc::assets::to_string(hash),
                                              .size = body.size()};
    arc::http::content_http_request request{
        .origin = "https://content.example.org", .object_path = object.path, .object_size = object.size};
    arc::http::content_http_response response{
        .status = 200, .content_length = std::to_string(body.size()), .body = body};
    std::filesystem::path root = std::filesystem::temp_directory_path() /
                                 ("arc-remote-content-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    arc::assets::derived_data_cache cache{arc::assets::derived_data_cache_config{.root = root}};

    ~fixture()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

} // namespace

TEST_CASE("C9.2c1 verifies and persists a cooked object, then reuses its cache entry")
{
    fixture f;
    auto first = arc::assets::ingest_verified_remote_object(f.cache, f.object, f.request, f.response, {});
    REQUIRE(first.succeeded());
    CHECK_FALSE(first.reused);
    arc::assets::cache_error error;
    auto loaded = arc::assets::load_verified_remote_object(f.cache, f.object, error);
    REQUIRE(loaded);
    CHECK(*loaded == f.body);
    auto repeated = arc::assets::ingest_verified_remote_object(f.cache, f.object, f.request, f.response, {});
    REQUIRE(repeated.succeeded());
    CHECK(repeated.reused);
}

TEST_CASE("C9.2c1 rejects corrupt bytes without exposing an active cached object")
{
    fixture f;
    f.response.body[0] = std::byte{0x00};
    auto result = arc::assets::ingest_verified_remote_object(f.cache, f.object, f.request, f.response, {});
    CHECK(result.code == arc::assets::remote_object_cache_error::hash_mismatch);
    arc::assets::cache_error error;
    CHECK_FALSE(arc::assets::load_verified_remote_object(f.cache, f.object, error));
}

TEST_CASE("C9.2c1 rejects mismatched object references, partial fetches and truncated payloads")
{
    fixture f;
    f.request.object_size += 1;
    CHECK(arc::assets::ingest_verified_remote_object(f.cache, f.object, f.request, f.response, {}).code ==
          arc::assets::remote_object_cache_error::invalid_reference);
    f.request.object_size = f.object.size;
    f.request.range_start = 0;
    f.request.range_length = 2;
    CHECK(arc::assets::ingest_verified_remote_object(f.cache, f.object, f.request, f.response, {}).code ==
          arc::assets::remote_object_cache_error::invalid_reference);
    f.request.range_start.reset();
    f.request.range_length.reset();
    f.response.body.pop_back();
    CHECK(arc::assets::ingest_verified_remote_object(f.cache, f.object, f.request, f.response, {}).code ==
          arc::assets::remote_object_cache_error::incompatible_response);
}

TEST_CASE("C9.2c1 rejects dangerous object paths and metadata mismatches")
{
    fixture f;
    f.object.path = "../source";
    CHECK(arc::assets::ingest_verified_remote_object(f.cache, f.object, f.request, f.response, {}).code ==
          arc::assets::remote_object_cache_error::invalid_reference);
    f.object.path = f.request.object_path;
    f.object.sha256 = std::string(64, 'f');
    CHECK(arc::assets::ingest_verified_remote_object(f.cache, f.object, f.request, f.response, {}).code ==
          arc::assets::remote_object_cache_error::invalid_reference);
}
