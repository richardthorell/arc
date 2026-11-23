#include <arc/assets/remote_content.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
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

std::string fixture()
{
    const auto path = std::filesystem::path(__FILE__).parent_path() / "fixtures" / "remote-content-v1.json";
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string feed_fixture(std::string_view filename = "remote-content-feed-v1.json")
{
    const auto file = std::filesystem::path(__FILE__).parent_path() / "fixtures" / filename;
    std::ifstream stream(file, std::ios::binary);
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
    replace_once(changed, "\"baseBuildId\": \"build-abc\",", "\"baseBuildId\": \"build-abc\", \"cookBuildId\": 12,");
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

TEST_CASE("C9.1c native feed decodes shared immutable-release replay fixture")
{
    using arc::assets::remote_content_feed_action;
    const auto parsed = arc::assets::parse_remote_content_feed(feed_fixture(), expectations(), 4);
    REQUIRE(parsed.succeeded());
    REQUIRE(parsed.feed);
    CHECK(parsed.feed->action == remote_content_feed_action::replay);
    CHECK(parsed.feed->oldest_available == 4);
    CHECK(parsed.feed->newest_available == 6);
    CHECK(parsed.feed->current_generation == 10);
    REQUIRE(parsed.feed->events.size() == 2);
    CHECK(parsed.feed->events[0].sequence == 5);
    CHECK(parsed.feed->events[0].generation == 9);
    CHECK(parsed.feed->events[1].sequence == 6);
    CHECK(parsed.feed->events[1].generation == 10);
    CHECK(parsed.feed->current_release.sha256 == parsed.feed->events.back().release.sha256);
    REQUIRE(parsed.feed->events[1].changed_guids.size() == 1);
    REQUIRE(parsed.feed->events[1].removed_guids.size() == 1);
    CHECK(parsed.feed->events[1].changed_guids[0] != parsed.feed->events[1].removed_guids[0]);
}

TEST_CASE("C9.1c native feed validates bounded full-snapshot and up-to-date responses")
{
    using arc::assets::remote_content_feed_action;
    auto reset = feed_fixture("remote-content-feed-snapshot-v1.json");
    const auto parsed = arc::assets::parse_remote_content_feed(reset, expectations(), 1);
    REQUIRE(parsed.succeeded());
    REQUIRE(parsed.feed);
    CHECK(parsed.feed->action == remote_content_feed_action::full_snapshot);
    CHECK(parsed.feed->events.empty());
    CHECK(parsed.feed->current_release.size == 702);

    replace_once(reset, "\"afterSequence\": 1", "\"afterSequence\": 6");
    replace_once(reset, "\"mode\": \"full-snapshot\"", "\"mode\": \"up-to-date\"");
    const auto up_to_date = arc::assets::parse_remote_content_feed(reset, expectations(), 6);
    REQUIRE(up_to_date.succeeded());
    CHECK(up_to_date.feed->action == remote_content_feed_action::up_to_date);
    CHECK(up_to_date.feed->events.empty());

    CHECK_FALSE(arc::assets::parse_remote_content_feed(reset, expectations(), 7).succeeded());
    CHECK_FALSE(arc::assets::parse_remote_content_feed(feed_fixture(), expectations(), 1).succeeded());
}

TEST_CASE("C9.1c native feed rejects missing replay entries and unsafe event references")
{
    const auto valid = feed_fixture();
    auto invalid = valid;
    replace_once(invalid, "\"sequence\": 5", "\"sequence\": 6");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = valid;
    replace_once(invalid, "\"generation\": 9", "\"generation\": 10");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = valid;
    replace_once(invalid, "\"currentGeneration\": 10", "\"currentGeneration\": 11");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = valid;
    replace_once(invalid, "\"afterSequence\": 4", "\"afterSequence\": 3");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = valid;
    replace_once(invalid, "\"mode\": \"replay\"", "\"mode\": \"full-snapshot\"");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = valid;
    replace_once(invalid, "objects/sha256/", "../unsafe-object/");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = valid;
    replace_once(invalid, "44444444-5555-4666-8777-888888888888", "../invalid-guid");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = valid;
    replace_once(invalid, "\"formatVersion\": 1", "\"formatVersion\": 2");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    auto incompatible = expectations();
    incompatible.target.texture_family = "astc";
    CHECK_FALSE(arc::assets::parse_remote_content_feed(valid, incompatible, 4).succeeded());
}

TEST_CASE("C9.1c native feed rejects cross-generation confusion and duplicate GUID mutations")
{
    auto invalid = feed_fixture();
    replace_once(invalid, "\"sha256\": \"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\"",
                         "\"sha256\": \"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\"");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = feed_fixture();
    replace_once(invalid, "\"removed\": [\n        \"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee\"",
                          "\"removed\": [\n        \"44444444-5555-4666-8777-888888888888\"");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());

    invalid = feed_fixture();
    replace_once(invalid, "\"engineVersion\": \"0.1.0\"", "\"engineVersion\": \"0.2.0\"");
    CHECK_FALSE(arc::assets::parse_remote_content_feed(invalid, expectations(), 4).succeeded());
}

TEST_CASE("C9.1c bounded replay falls back to current immutable snapshot")
{
    using arc::assets::remote_content_feed_action;
    using arc::assets::remote_content_reconnect;
    CHECK(remote_content_reconnect(1, 1, 257) == remote_content_feed_action::replay);
    CHECK(remote_content_reconnect(1, 1, 258) == remote_content_feed_action::full_snapshot);
    CHECK(remote_content_reconnect(4, 4, 5, 0) == remote_content_feed_action::invalid);
    CHECK(remote_content_reconnect(4, 4, 5, 1) == remote_content_feed_action::replay);
    CHECK(remote_content_reconnect(2, 4, 5) == remote_content_feed_action::full_snapshot);
    CHECK(remote_content_reconnect(6, 4, 5) == remote_content_feed_action::reject_regression);
}

TEST_CASE("C9.1c Android variant reuses the exact feed format and strict target checks")
{
    auto android = feed_fixture();
    replace_once(android, "\"platform\": \"windows\"", "\"platform\": \"android\"");
    replace_once(android, "\"architecture\": \"x86_64\"", "\"architecture\": \"arm64\"");
    replace_once(android, "\"cookProfile\": \"windows-x64-vulkan\"",
                          "\"cookProfile\": \"android-arm64-vulkan\"");
    replace_once(android, "\"textureFamily\": \"bc\"", "\"textureFamily\": \"astc\"");
    auto expected = expectations();
    expected.target = {.platform = "android",
                       .architecture = "arm64",
                       .renderer = "vulkan",
                       .cook_profile = "android-arm64-vulkan",
                       .texture_family = "astc"};
    CHECK(arc::assets::parse_remote_content_feed(android, expected, 4).succeeded());
    CHECK_FALSE(arc::assets::parse_remote_content_feed(android, expectations(), 4).succeeded());
}
