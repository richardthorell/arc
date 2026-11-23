#include <arc/project/runtime_content_source.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace project = arc::project;

project::distribution_manifest development_manifest()
{
    project::distribution_manifest manifest;
    manifest.project_guid = "11111111-1111-1111-1111-111111111111";
    manifest.target.id = "windows-x64-vulkan";
    manifest.target.platform = "windows";
    manifest.target.configuration = "Debug";
    manifest.target.cook_profile = "windows-x64-vulkan";
    return manifest;
}

project::runtime_content_options_result parse(std::initializer_list<std::string_view> args)
{
    const std::vector<std::string_view> values(args);
    return project::parse_runtime_content_arguments(values);
}

project::runtime_content_selection_result select(const project::runtime_content_launch& config,
                                                 std::string_view configuration = "Debug")
{
    auto manifest = development_manifest();
    manifest.target.configuration = configuration;
    return project::select_runtime_content(manifest, config, std::filesystem::temp_directory_path() / "arc-c9-3-test");
}

} // namespace

TEST_CASE("C9.3 content launch parser supports Windows split/equal forms and unrelated host arguments")
{
    auto options = parse({"--ci-smoke", "--content-server", "http://127.0.0.1:8123", "--content-mode=hybrid"});
    REQUIRE(options);
    CHECK(options.value().server == "http://127.0.0.1:8123");
    CHECK(options.value().mode == project::runtime_content_mode::hybrid);

    options = parse({"--content-server=https://example.org", "--content-mode", "remote"});
    REQUIRE(options);
    CHECK(options.value().mode == project::runtime_content_mode::remote);

    options = parse({"--ci-smoke"});
    REQUIRE(options);
    CHECK_FALSE(options.value().server.has_value());
    CHECK_FALSE(options.value().mode.has_value());
}

TEST_CASE("C9.3 content launch parser rejects malformed, duplicate and unknown content flags")
{
    CHECK_FALSE(parse({"--content-server"}));
    CHECK_FALSE(parse({"--content-server="}));
    CHECK_FALSE(parse({"--content-mode", ""}));
    CHECK_FALSE(parse({"--content-mode", "unknown"}));
    CHECK_FALSE(parse({"--content-server", "https://example.org", "--content-server=https://other.org"}));
    CHECK_FALSE(parse({"--content-mode=remote", "--content-mode=hybrid"}));
    CHECK_FALSE(parse({"--content-servr=https://example.org"}));

    auto result = parse({"--content-mode", "invalid"});
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "content.mode");
}

TEST_CASE("C9.3 Android Intent metadata uses the same typed settings and errors")
{
    auto good = project::parse_android_content_metadata("http://127.0.0.1:8020", "remote");
    REQUIRE(good);
    CHECK(good.value().server == "http://127.0.0.1:8020");
    CHECK(good.value().mode == project::runtime_content_mode::remote);
    CHECK_FALSE(project::parse_android_content_metadata("", "remote"));
    CHECK_FALSE(project::parse_android_content_metadata(std::nullopt, "Remote"));
}

TEST_CASE("C9.3 selection defaults to packaged and isolates the writable cache root")
{
    auto selected = select({});
    REQUIRE(selected);
    CHECK(selected.value().mode == project::runtime_content_mode::packaged);
    CHECK(selected.value().origin == project::runtime_content_origin::distribution);
    CHECK(selected.value().server_origin.empty());
    CHECK(selected.value().project_guid == "11111111-1111-1111-1111-111111111111");
    CHECK(selected.value().target_id == "windows-x64-vulkan");
    CHECK(selected.value().cook_profile == "windows-x64-vulkan");
    CHECK(selected.value().cache_root.filename() == "RemoteContent");
    CHECK(selected.value().cache_root.is_absolute());
}

TEST_CASE("C9.3 trusted Development launch overrides project settings without changing D0 identity")
{
    project::runtime_content_launch config;
    config.project_development.server = "https://project.example.org";
    config.project_development.mode = project::runtime_content_mode::hybrid;
    auto selected = select(config);
    REQUIRE(selected);
    CHECK(selected.value().server_origin == "https://project.example.org");
    CHECK(selected.value().mode == project::runtime_content_mode::hybrid);
    CHECK(selected.value().origin == project::runtime_content_origin::project_development);

    config.trusted_launch.server = "http://127.0.0.1:8123";
    selected = select(config);
    REQUIRE(selected);
    CHECK(selected.value().mode == project::runtime_content_mode::remote);
    CHECK(selected.value().server_origin == "http://127.0.0.1:8123");
    CHECK(selected.value().origin == project::runtime_content_origin::trusted_launch);

    config.trusted_launch.mode = project::runtime_content_mode::hybrid;
    selected = select(config);
    REQUIRE(selected);
    CHECK(selected.value().mode == project::runtime_content_mode::hybrid);

    config.trusted_launch.mode = project::runtime_content_mode::packaged;
    selected = select(config);
    REQUIRE_FALSE(selected);
    CHECK(selected.error().field == "content.server");

    config.trusted_launch.server.reset();
    selected = select(config);
    REQUIRE(selected);
    CHECK(selected.value().mode == project::runtime_content_mode::packaged);
    CHECK(selected.value().server_origin.empty());
}

TEST_CASE("C9.3 Development endpoint rejects credentials, arbitrary paths and plaintext LAN")
{
    project::runtime_content_launch config;
    config.trusted_launch.mode = project::runtime_content_mode::remote;
    for (const std::string_view origin :
         {"http://example.org:8123", "https://user:password@example.org", "https://example.org/path",
          "https://example.org?token=abc", "https://example.org/#fragment", "ftp://example.org", "http://127.0.0.1:0",
          "https://example.org:abc"})
    {
        config.trusted_launch.server = std::string(origin);
        auto result = select(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().field == "content.server");
        CHECK(result.error().message.find(std::string(origin)) == std::string::npos);
    }
    config.trusted_launch.server.reset();
    auto missing = select(config);
    REQUIRE_FALSE(missing);
    CHECK(missing.error().field == "content.server");
}

TEST_CASE("C9.3 Shipping rejects any Development override, including packaged mode")
{
    project::runtime_content_launch config;
    CHECK(select(config, "Shipping"));
    config.trusted_launch.mode = project::runtime_content_mode::packaged;
    auto result = select(config, "Shipping");
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "content");

    config.trusted_launch.mode.reset();
    config.project_development.server = "https://example.org";
    result = select(config, "Shipping");
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "content");
}
