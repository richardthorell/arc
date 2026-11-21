#include <arc/assets/remote_content.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

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

std::string fixture()
{
    const auto path = std::filesystem::path(__FILE__).parent_path() / "fixtures" / "remote-content-v1.json";
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void replace_once(std::string& text, const std::string& from, const std::string& to)
{
    const auto index = text.find(from);
    REQUIRE(index != std::string::npos);
    text.replace(index, from.size(), to);
}

} // namespace

TEST_CASE("C9.1 native remote descriptor accepts shared publisher fixture")
{
    const auto result = arc::assets::parse_remote_content_release(fixture(), expectations());
    REQUIRE(result.succeeded());
    REQUIRE(result.release);
    CHECK(result.release->generation == 5);
    CHECK(result.release->chunks.size() == 1);
    CHECK(result.release->chunks.front().name == "startup-hash.arcpak");
    CHECK(result.release->cook_manifest.size == 123);
    CHECK(result.release->cook_build_id == "build-abc"); // legacy descriptor
}

TEST_CASE("C9.1 optional cookBuildId is distinct from packaged baseBuildId")
{
    auto changed = fixture();
    replace_once(changed, "\"baseBuildId\": \"build-abc\",",
                 "\"baseBuildId\": \"build-abc\", \"cookBuildId\": \"unsaved-cook-123\",");
    const auto result = arc::assets::parse_remote_content_release(changed, expectations());
    REQUIRE(result.succeeded());
    REQUIRE(result.release);
    CHECK(result.release->base_build_id == "build-abc");
    CHECK(result.release->cook_build_id == "unsaved-cook-123");

    replace_once(changed, "unsaved-cook-123", "../unsafe");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());
    changed = fixture();
    replace_once(changed, "\"baseBuildId\": \"build-abc\",",
                 "\"baseBuildId\": \"build-abc\", \"cookBuildId\": 12,");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());
}

TEST_CASE("C9.1 native descriptor rejects mismatched runtime identity and stale generations")
{
    const auto valid = fixture();
    auto expected = expectations();
    expected.target.texture_family = "astc";
    CHECK_FALSE(arc::assets::parse_remote_content_release(valid, expected).succeeded());
    expected = expectations();
    expected.minimum_generation = 5;
    CHECK_FALSE(arc::assets::parse_remote_content_release(valid, expected).succeeded());

    auto changed = valid;
    replace_once(changed, "build-abc", "wrong-build");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());

    changed = valid;
    replace_once(changed, "\"formatVersion\": 1", "\"formatVersion\": 2");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());
}

TEST_CASE("C9.1 native descriptor rejects malformed, unsafe and partial object references")
{
    const auto valid = fixture();
    auto changed = valid;
    replace_once(changed, "startup-hash.arcpak", "../escaped.arcpak");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());

    changed = valid;
    replace_once(changed, "objects/sha256/", "https://evil.invalid/");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());

    changed = valid;
    replace_once(changed, "\"complete\": true", "\"complete\": false");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());

    changed = valid;
    replace_once(changed, "\"size\": 456", "\"size\": true");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());

    changed = valid;
    replace_once(changed, "\"generation\": 5", "\"generation\": -1");
    CHECK_FALSE(arc::assets::parse_remote_content_release(changed, expectations()).succeeded());
}

TEST_CASE("C9.1 native dev-feed recovery follows contract")
{
    using arc::assets::remote_content_feed_action;
    using arc::assets::remote_content_reconnect;
    CHECK(remote_content_reconnect(8, 7, 10) == remote_content_feed_action::replay);
    CHECK(remote_content_reconnect(5, 8, 10) == remote_content_feed_action::full_snapshot);
    CHECK(remote_content_reconnect(10, 8, 10) == remote_content_feed_action::up_to_date);
    CHECK(remote_content_reconnect(11, 8, 10) == remote_content_feed_action::reject_regression);
    CHECK(remote_content_reconnect(2, 8, 5) == remote_content_feed_action::invalid);
}
