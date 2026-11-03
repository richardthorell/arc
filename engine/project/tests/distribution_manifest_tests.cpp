#include <arc/project/distribution_manifest.h>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <string>

namespace
{
const std::string digest(64, 'a');
arc::project::distribution_manifest client()
{
    arc::project::distribution_manifest m;
    m.project_guid = "11111111-1111-1111-1111-111111111111";
    m.engine_version = "0.1.0";
    m.sdk_version = "0.1.0";
    m.target = {.id = "windows-x64-vulkan",
                .platform = "windows",
                .architecture = "x86_64",
                .renderer = "vulkan",
                .configuration = "Shipping",
                .cook_profile = "windows-x64-vulkan"};
    m.role = "client";
    m.finalizer = "windows-directory";
    m.entrypoint = {.kind = "executable", .value = "bin/Game.exe"};
    m.startup_scenes = {"22222222-2222-2222-2222-222222222222"};
    m.runtime_config = "config/runtime.json";
    m.cook_manifests = {"content/game.arccookmanifest"};
    m.files = {{.path = "content/game.arccookmanifest", .kind = "content", .size = 50, .sha256 = digest},
               {.path = "bin/Game.exe", .kind = "executable", .size = 100, .sha256 = digest},
               {.path = "config/runtime.json", .kind = "configuration", .size = 20, .sha256 = digest}};
    return m;
}
arc::project::distribution_manifest android()
{
    auto m = client();
    m.target = {.id = "android-arm64-vulkan",
                .platform = "android",
                .architecture = "arm64",
                .renderer = "vulkan",
                .configuration = "Shipping",
                .cook_profile = "android-arm64-vulkan"};
    m.finalizer = "android-gradle";
    m.entrypoint = {.kind = "android-activity", .value = "com.example.game.MainActivity"};
    m.files.erase(
        std::remove_if(m.files.begin(), m.files.end(), [](const auto& file) { return file.kind == "executable"; }),
        m.files.end());
    return m;
}
void rejected(const arc::project::distribution_manifest& m, arc::project::project_error_code code)
{
    const auto output = arc::project::serialize_distribution_manifest(m);
    REQUIRE_FALSE(output);
    CHECK(output.error().code == code);
}
} // namespace

TEST_CASE("distribution v1 windows client round trips canonically")
{
    const auto written = arc::project::serialize_distribution_manifest(client());
    REQUIRE(written);
    const auto parsed = arc::project::parse_distribution_manifest(written.value());
    REQUIRE(parsed);
    const auto rewritten = arc::project::serialize_distribution_manifest(parsed.value());
    REQUIRE(rewritten);
    CHECK(written.value() == rewritten.value());
    CHECK(parsed.value().role == "client");
    CHECK(parsed.value().target.cook_profile == "windows-x64-vulkan");
}
TEST_CASE("distribution v1 Android client uses same schema")
{
    const auto written = arc::project::serialize_distribution_manifest(android());
    REQUIRE(written);
    const auto parsed = arc::project::parse_distribution_manifest(written.value());
    REQUIRE(parsed);
    CHECK(parsed.value().target.platform == "android");
    CHECK(parsed.value().entrypoint.kind == "android-activity");
}
TEST_CASE("distribution v1 supports headless dedicated servers")
{
    auto m = client();
    m.role = "server";
    m.target.renderer = "none";
    m.target.id = "windows-x64-headless";
    m.target.cook_profile = m.target.id;
    m.entrypoint.value = "bin/Server.exe";
    m.startup_scenes.clear();
    m.files[1].path = "bin/Server.exe";
    const auto output = arc::project::serialize_distribution_manifest(m);
    REQUIRE(output);
    CHECK(arc::project::parse_distribution_manifest(output.value()));
}
TEST_CASE("distribution file and manifest reference ordering is canonical")
{
    auto a = client();
    auto b = a;
    std::reverse(b.files.begin(), b.files.end());
    CHECK(arc::project::serialize_distribution_manifest(a).value() ==
          arc::project::serialize_distribution_manifest(b).value());
}
TEST_CASE("distribution rejects unsafe paths and case collisions")
{
    for (const auto* path : {"../Game.exe", "/Game.exe", "C:/Game.exe", "bin\\Game.exe", "bin//Game.exe",
                             "bin/./Game.exe", "bin/../Game.exe", "bin/Game.exe/"})
    {
        auto m = client();
        m.files[1].path = path;
        rejected(m, arc::project::project_error_code::unsafe_path);
    }
    auto duplicate = client();
    duplicate.files.push_back({.path = "BIN/game.EXE", .kind = "library", .size = 10, .sha256 = digest});
    rejected(duplicate, arc::project::project_error_code::invalid_descriptor);
    auto traversal = client();
    traversal.cook_manifests = {"../game.arccookmanifest"};
    rejected(traversal, arc::project::project_error_code::unsafe_path);
}
TEST_CASE("distribution rejects future schema versions without implicit migration")
{
    const auto written = arc::project::serialize_distribution_manifest(client());
    REQUIRE(written);
    auto source = written.value();
    const auto key = source.find("\"formatVersion\": 1");
    REQUIRE(key != std::string::npos);
    source.replace(key, std::string("\"formatVersion\": 1").size(), "\"formatVersion\": 2");
    const auto parsed = arc::project::parse_distribution_manifest(source);
    REQUIRE_FALSE(parsed);
    CHECK(parsed.error().code == arc::project::project_error_code::unsupported_version);
    auto future = client();
    future.format_version = 2;
    rejected(future, arc::project::project_error_code::unsupported_version);
}
TEST_CASE("distribution validates target role renderer and configuration")
{
    auto headless_client = client();
    headless_client.target.renderer = "none";
    rejected(headless_client, arc::project::project_error_code::invalid_descriptor);
    auto graphical_server = client();
    graphical_server.role = "server";
    rejected(graphical_server, arc::project::project_error_code::invalid_descriptor);
    auto mismatched_profile = client();
    mismatched_profile.target.cook_profile.clear();
    rejected(mismatched_profile, arc::project::project_error_code::invalid_descriptor);
    auto invalid_renderer = android();
    invalid_renderer.target.renderer = "direct3d12";
    rejected(invalid_renderer, arc::project::project_error_code::invalid_descriptor);
    auto invalid_configuration = client();
    invalid_configuration.target.configuration = "Test";
    rejected(invalid_configuration, arc::project::project_error_code::invalid_descriptor);
    auto wrong_entrypoint = client();
    wrong_entrypoint.entrypoint.kind = "android-activity";
    wrong_entrypoint.entrypoint.value = "com.example.Game";
    rejected(wrong_entrypoint, arc::project::project_error_code::invalid_descriptor);
}
TEST_CASE("distribution requires staged executable configuration and cook manifest")
{
    auto missing_exe = client();
    missing_exe.files.erase(missing_exe.files.begin() + 1);
    rejected(missing_exe, arc::project::project_error_code::invalid_descriptor);
    auto missing_config = client();
    missing_config.files.pop_back();
    rejected(missing_config, arc::project::project_error_code::invalid_descriptor);
    auto missing_content = client();
    missing_content.cook_manifests.clear();
    rejected(missing_content, arc::project::project_error_code::invalid_descriptor);
    auto incomplete_content = client();
    incomplete_content.files.erase(incomplete_content.files.begin());
    rejected(incomplete_content, arc::project::project_error_code::invalid_descriptor);
    auto malformed_hash = client();
    malformed_hash.files.front().sha256 = "abc";
    rejected(malformed_hash, arc::project::project_error_code::invalid_descriptor);
}
TEST_CASE("distribution parser rejects malformed or incomplete documents")
{
    const auto invalid = arc::project::parse_distribution_manifest("{");
    REQUIRE_FALSE(invalid);
    CHECK(invalid.error().code == arc::project::project_error_code::invalid_json);
    const auto missing =
        arc::project::parse_distribution_manifest(R"({"format":"arc.distribution","formatVersion":1})");
    REQUIRE_FALSE(missing);
}
