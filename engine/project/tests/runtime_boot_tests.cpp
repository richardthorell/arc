#include <arc/project/runtime_boot.h>

#include <arc/assets/assets.h>
#include <arc/io/virtual_file_system.h>
#include <arc/jobs/jobs.h>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
namespace project = arc::project;
namespace fs = std::filesystem;

constexpr std::string_view executable_bytes = "arc game executable";
constexpr std::string_view config_bytes = R"({"quality":"high"})";
constexpr std::string_view cook_bytes = R"({"format":"arc-cook"})";

struct temporary_directory
{
    fs::path root;

    temporary_directory()
    {
        static std::atomic<std::uint64_t> unique{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root = fs::temp_directory_path() /
               ("arc-d1-2-" + std::to_string(stamp) + "-" + std::to_string(unique.fetch_add(1)));
        fs::create_directories(root);
    }

    ~temporary_directory()
    {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }

    temporary_directory(const temporary_directory&) = delete;
    temporary_directory& operator=(const temporary_directory&) = delete;
};

void write(const fs::path& path, std::string_view contents)
{
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    REQUIRE(output);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    REQUIRE(output.good());
}

std::string hash(std::string_view contents)
{
    const auto bytes = std::as_bytes(std::span(contents.data(), contents.size()));
    return arc::assets::to_string(arc::assets::hash_bytes(bytes));
}

project::distribution_file staged(std::string path, std::string kind, std::string_view contents)
{
    return {.path = std::move(path),
            .kind = std::move(kind),
            .size = contents.size(),
            .sha256 = hash(contents)};
}

project::distribution_manifest manifest(bool server = false, bool android = false)
{
    project::distribution_manifest result;
    result.project_guid = "11111111-1111-1111-1111-111111111111";
    result.engine_version = "0.1.0";
    result.sdk_version = "0.1.0";
    result.target = {.id = android ? "android-arm64-vulkan" : (server ? "windows-x64-headless" : "windows-x64-vulkan"),
                     .platform = android ? "android" : "windows",
                     .architecture = android ? "arm64" : "x86_64",
                     .renderer = server ? "none" : "vulkan",
                     .configuration = "Shipping",
                     .cook_profile = android ? "android-arm64-vulkan" : (server ? "windows-server" : "windows-client")};
    result.role = server ? "server" : "client";
    result.finalizer = android ? "android-gradle" : "windows-directory";
    result.entrypoint = android ? project::distribution_entrypoint{.kind = "android-activity",
                                                                    .value = "com.example.arc.GameActivity"}
                                : project::distribution_entrypoint{.kind = "executable",
                                                                    .value = server ? "bin/Server.exe" : "bin/Game.exe"};
    if (!server) result.startup_scenes = {"22222222-2222-2222-2222-222222222222"};
    result.runtime_config = "Config/runtime.json";
    result.cook_manifests = {"Content/startup.arccookmanifest"};
    result.files = {staged("Config/runtime.json", "configuration", config_bytes),
                    staged("Content/startup.arccookmanifest", "content", cook_bytes)};
    if (!android) result.files.push_back(staged(result.entrypoint.value, "executable", executable_bytes));
    return result;
}

void stage_native(const fs::path& root, const project::distribution_manifest& staged_manifest)
{
    for (const auto& file : staged_manifest.files)
    {
        const auto contents = file.kind == "configuration" ? config_bytes
                              : file.kind == "executable"  ? executable_bytes
                                                            : cook_bytes;
        write(root / file.path, contents);
    }
    const auto text = project::serialize_distribution_manifest(staged_manifest);
    REQUIRE(text);
    write(root / project::runtime_distribution_manifest_name, text.value());
}

project::runtime_boot_request native_request(const fs::path& root, bool server = false,
                                              project::runtime_distribution_location_kind kind =
                                                  project::runtime_distribution_location_kind::executable)
{
    auto m = manifest(server);
    project::runtime_boot_request request;
    request.role = server ? project::runtime_launch_role::server : project::runtime_launch_role::client;
    request.host.distribution.kind = kind;
    request.host.distribution.native_path = kind == project::runtime_distribution_location_kind::directory
                                                ? root
                                                : root / m.entrypoint.value;
    request.host.writable_storage_root = root / "UserData";
    request.host.capabilities.family = arc::framework::platform_family::windows;
    request.host.capabilities.window_system = !server;
    request.expected = {.engine_version = "0.1.0",
                        .sdk_version = "0.1.0",
                        .architecture = "x86_64",
                        .renderer = server ? "none" : "vulkan",
                        .configuration = "Shipping",
                        .project_guid = m.project_guid,
                        .target_id = m.target.id,
                        .cook_profile = m.target.cook_profile};
    request.startup_bundles = {"Engine", "Boot", "Startup/Common"};
    return request;
}

void expect_failure(project::runtime_boot_request request, project::project_error_code code, std::string_view field)
{
    const auto result = project::discover_runtime_boot(std::move(request));
    REQUIRE_FALSE(result);
    CHECK(result.error().code == code);
    CHECK(result.error().field == field);
    CHECK_FALSE(result.error().message.empty());
}

arc::io::file_buffer bytes(std::string_view content)
{
    const auto view = std::as_bytes(std::span(content.data(), content.size()));
    return {view.begin(), view.end()};
}

struct mounted_fixture
{
    arc::jobs::job_system jobs{ {.worker_count = 1,
                                 .run_inline = false,
                                 .io_worker_count = 1,
                                 .enable_render_thread = false} };
    arc::io::virtual_file_system vfs{jobs};
    std::shared_ptr<arc::io::memory_file_provider> provider{std::make_shared<arc::io::memory_file_provider>(jobs)};
    std::string base = "games/demo/";

    mounted_fixture()
    {
        REQUIRE(vfs.mount({.root = arc::io::virtual_path::parse("package://app").value(), .provider = provider}));
    }

    void stage(const project::distribution_manifest& m, std::string_view config = config_bytes)
    {
        const auto serialized = project::serialize_distribution_manifest(m);
        REQUIRE(serialized);
        const std::vector<arc::io::memory_provider_update> updates = {
            {.relative_path = base + "arc-distribution.json", .bytes = bytes(serialized.value())},
            {.relative_path = base + m.runtime_config, .bytes = bytes(config)},
            {.relative_path = base + m.cook_manifests.front(), .bytes = bytes(cook_bytes)}};
        REQUIRE(provider->publish(updates));
    }

    project::runtime_boot_request request() const
    {
        project::runtime_boot_request r;
        r.host.distribution.kind = project::runtime_distribution_location_kind::mounted_vfs;
        r.host.distribution.files = const_cast<arc::io::virtual_file_system*>(&vfs);
        r.host.distribution.virtual_root = arc::io::virtual_path::parse("package://app/games/demo").value();
        r.host.writable_storage_root = fs::temp_directory_path() / "arc-android-user";
        r.host.capabilities.family = arc::framework::platform_family::android;
        r.host.capabilities.native_package_assets = true;
        r.expected = {.engine_version = "0.1.0",
                      .sdk_version = "0.1.0",
                      .architecture = "arm64",
                      .renderer = "vulkan",
                      .configuration = "Shipping",
                      .project_guid = "11111111-1111-1111-1111-111111111111",
                      .target_id = "android-arm64-vulkan",
                      .cook_profile = "android-arm64-vulkan"};
        return r;
    }
};

} // namespace

TEST_CASE("D1.2 discovers packaged client relative to executable and survives relocation")
{
    temporary_directory temp;
    const auto first = temp.root / "original";
    stage_native(first, manifest());

    const auto initial = project::discover_runtime_boot(native_request(first));
    REQUIRE(initial);
    CHECK(initial.value().native_distribution_root == fs::canonical(first));
    CHECK(initial.value().launch.manifest.startup_scenes.front() == "22222222-2222-2222-2222-222222222222");
    CHECK(initial.value().launch.manifest.runtime_config == "Config/runtime.json");
    CHECK(initial.value().launch.startup_bundles.size() == 3);

    const auto relocated = temp.root / "relocated";
    fs::rename(first, relocated);
    const auto reopened = project::discover_runtime_boot(native_request(relocated));
    REQUIRE(reopened);
    CHECK(reopened.value().native_distribution_root == fs::canonical(relocated));
    CHECK(reopened.value().launch.host.distribution.native_path == relocated / "bin/Game.exe");
}

TEST_CASE("D1.2 accepts explicit native root and headless server without any scene")
{
    temporary_directory temp;
    const auto root = temp.root / "server";
    stage_native(root, manifest(true));
    auto request = native_request(root, true, project::runtime_distribution_location_kind::directory);
    const auto booted = project::discover_runtime_boot(std::move(request));
    REQUIRE(booted);
    CHECK(booted.value().launch.role == project::runtime_launch_role::server);
    CHECK(booted.value().launch.manifest.startup_scenes.empty());
    CHECK(booted.value().native_distribution_root == fs::canonical(root));
}

TEST_CASE("D1.2 rejects missing, malformed or unsupported distribution manifests")
{
    temporary_directory temp;
    const auto root = temp.root / "client";
    stage_native(root, manifest());
    fs::remove(root / "arc-distribution.json");
    expect_failure(native_request(root), project::project_error_code::not_found, "manifest");

    write(root / "arc-distribution.json", "{");
    expect_failure(native_request(root), project::project_error_code::invalid_json, "manifest");

    auto json = project::serialize_distribution_manifest(manifest());
    REQUIRE(json);
    const auto version = json.value().find("\"formatVersion\": 1");
    REQUIRE(version != std::string::npos);
    json.value().replace(version, std::string("\"formatVersion\": 1").size(), "\"formatVersion\": 8");
    write(root / "arc-distribution.json", json.value());
    expect_failure(native_request(root), project::project_error_code::unsupported_version, "formatVersion");
}

TEST_CASE("D1.2 rejects wrong executable entry and does not walk to unrelated manifest")
{
    temporary_directory temp;
    const auto root = temp.root / "release";
    auto m = manifest();
    m.entrypoint.value = "bin/Other.exe";
    m.files.back().path = m.entrypoint.value;
    stage_native(root, m);
    write(root / "bin/Game.exe", executable_bytes);
    expect_failure(native_request(root), project::project_error_code::invalid_descriptor, "entrypoint.value");

    // A nearer manifest is authoritative even when a different one exists above it.
    const auto nested = root / "bin";
    write(nested / "arc-distribution.json", "{");
    expect_failure(native_request(root), project::project_error_code::invalid_json, "manifest");
}

TEST_CASE("D1.2 catches missing and corrupt required staged native files")
{
    temporary_directory temp;
    const auto root = temp.root / "client";
    stage_native(root, manifest());
    fs::remove(root / "Config/runtime.json");
    expect_failure(native_request(root), project::project_error_code::not_found, "files.path");

    write(root / "Config/runtime.json", "x");
    expect_failure(native_request(root), project::project_error_code::corrupt_content, "files.size");

    write(root / "Config/runtime.json", std::string(config_bytes.size(), 'x'));
    expect_failure(native_request(root), project::project_error_code::corrupt_content, "files.sha256");

    write(root / "Config/runtime.json", config_bytes);
    write(root / "Content/startup.arccookmanifest", std::string(cook_bytes.size(), 'x'));
    expect_failure(native_request(root), project::project_error_code::corrupt_content, "files.sha256");
}

TEST_CASE("D1.2 default boot checks required hashes and full mode checks executable SHA too")
{
    temporary_directory temp;
    const auto root = temp.root / "client";
    stage_native(root, manifest());
    write(root / "bin/Game.exe", std::string(executable_bytes.size(), 'x'));
    REQUIRE(project::discover_runtime_boot(native_request(root)));
    auto full = native_request(root);
    full.integrity = project::runtime_boot_integrity::full;
    expect_failure(std::move(full), project::project_error_code::corrupt_content, "files.sha256");
}

TEST_CASE("D1.2 checks engine SDK target profile role and native-path expectations")
{
    temporary_directory temp;
    const auto root = temp.root / "client";
    stage_native(root, manifest());

    auto bad = native_request(root);
    bad.expected.engine_version = "9.0.0";
    expect_failure(bad, project::project_error_code::incompatible_engine, "engineVersion");

    bad = native_request(root);
    bad.expected.sdk_version = "9.0.0";
    expect_failure(bad, project::project_error_code::incompatible_engine, "sdkVersion");

    bad = native_request(root);
    bad.expected.configuration = "Debug";
    expect_failure(bad, project::project_error_code::incompatible_engine, "target.configuration");

    bad = native_request(root);
    bad.expected.architecture = "arm64";
    expect_failure(bad, project::project_error_code::incompatible_engine, "target.architecture");

    bad = native_request(root);
    bad.expected.cook_profile = "wrong-cook";
    expect_failure(bad, project::project_error_code::incompatible_engine, "target.cookProfile");

    bad = native_request(root);
    bad.role = project::runtime_launch_role::server;
    expect_failure(bad, project::project_error_code::invalid_descriptor, "role");

    bad = native_request(root);
    bad.host.distribution.native_path = "relative/bin/Game.exe";
    expect_failure(bad, project::project_error_code::unsafe_path, "host.distribution");

    bad = native_request(root);
    bad.expected.renderer.clear();
    expect_failure(bad, project::project_error_code::invalid_descriptor, "expected");
}

TEST_CASE("D1.2 rejects staged native symlinks that escape the package root")
{
#ifndef _WIN32
    temporary_directory temp;
    const auto root = temp.root / "client";
    stage_native(root, manifest());
    const auto outside = temp.root / "outside.json";
    write(outside, config_bytes);
    fs::remove(root / "Config/runtime.json");
    std::error_code error;
    fs::create_symlink(outside, root / "Config/runtime.json", error);
    REQUIRE_FALSE(error);
    expect_failure(native_request(root), project::project_error_code::unsafe_path, "files.path");
#endif
}

TEST_CASE("D1.2 discovers Android manifest through a nested mounted virtual root")
{
    mounted_fixture fixture;
    fixture.stage(manifest(false, true));
    const auto result = project::discover_runtime_boot(fixture.request());
    REQUIRE(result);
    CHECK(result.value().native_distribution_root.empty());
    CHECK(result.value().launch.manifest.target.platform == "android");
    CHECK(result.value().launch.manifest.entrypoint.kind == "android-activity");
    CHECK(result.value().launch.host.distribution.virtual_root.string() == "package://app/games/demo");
}

TEST_CASE("D1.2 fails closed on missing or modified Android package inputs")
{
    mounted_fixture fixture;
    fixture.stage(manifest(false, true));

    auto missing = fixture.request();
    missing.host.distribution.virtual_root = arc::io::virtual_path::parse("package://app/other").value();
    expect_failure(missing, project::project_error_code::not_found, "manifest");

    fixture.stage(manifest(false, true), std::string(config_bytes.size(), 'z'));
    expect_failure(fixture.request(), project::project_error_code::corrupt_content, "files.sha256");

    fixture.stage(manifest(false, true));
    auto full = fixture.request();
    full.integrity = project::runtime_boot_integrity::full;
    REQUIRE(project::discover_runtime_boot(std::move(full)));
}
