#include <arc/project/runtime_launch.h>

#include <arc/jobs/jobs.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace project = arc::project;
namespace framework = arc::framework;

const std::string digest(64, 'a');

project::distribution_manifest client_manifest()
{
    project::distribution_manifest manifest;
    manifest.project_guid = "11111111-1111-1111-1111-111111111111";
    manifest.engine_version = "0.1.0";
    manifest.sdk_version = "0.1.0";
    manifest.target = {.id = "windows-x64-vulkan",
                       .platform = "windows",
                       .architecture = "x86_64",
                       .renderer = "vulkan",
                       .configuration = "Shipping",
                       .cook_profile = "windows-x64-vulkan"};
    manifest.role = "client";
    manifest.finalizer = "windows-directory";
    manifest.entrypoint = {.kind = "executable", .value = "bin/Game.exe"};
    manifest.startup_scenes = {"22222222-2222-2222-2222-222222222222", "33333333-3333-3333-3333-333333333333"};
    manifest.runtime_config = "config/runtime.json";
    manifest.cook_manifests = {"content/game.arccookmanifest"};
    manifest.files = {{.path = "bin/Game.exe", .kind = "executable", .size = 100, .sha256 = digest},
                      {.path = "config/runtime.json", .kind = "configuration", .size = 20, .sha256 = digest},
                      {.path = "content/game.arccookmanifest", .kind = "content", .size = 50, .sha256 = digest}};
    return manifest;
}

project::runtime_launch_descriptor client()
{
    project::runtime_launch_descriptor descriptor;
    descriptor.manifest = client_manifest();
    descriptor.host.distribution.kind = project::runtime_distribution_location_kind::executable;
    descriptor.host.distribution.native_path = std::filesystem::temp_directory_path() / "arc-game" / "bin" / "Game.exe";
    descriptor.host.writable_storage_root = std::filesystem::temp_directory_path() / "arc-game-user";
    descriptor.host.capabilities.family = framework::platform_family::windows;
    descriptor.host.capabilities.window_system = true;
    descriptor.host.device_profile = "desktop-high";
    descriptor.startup_bundles = {"Engine", "Boot", "Startup/Common"};
    return descriptor;
}

project::runtime_launch_descriptor server()
{
    auto descriptor = client();
    descriptor.manifest.role = "server";
    descriptor.manifest.target.renderer = "none";
    descriptor.manifest.target.id = "windows-x64-headless";
    descriptor.manifest.target.cook_profile = descriptor.manifest.target.id;
    descriptor.manifest.entrypoint.value = "bin/Server.exe";
    descriptor.manifest.files.front().path = "bin/Server.exe";
    descriptor.manifest.startup_scenes.clear();
    descriptor.role = project::runtime_launch_role::server;
    descriptor.host.capabilities.window_system = false;
    return descriptor;
}

void reject(const project::runtime_launch_descriptor& descriptor, project::project_error_code code,
            std::string_view field)
{
    const auto prepared = project::prepare_runtime_launch(descriptor);
    REQUIRE_FALSE(prepared);
    CHECK(prepared.error().code == code);
    CHECK(prepared.error().field == field);
}
} // namespace

TEST_CASE("D1 launch preserves D0 identities and ordered startup scenes and bundles")
{
    auto descriptor = client();
    const auto prepared = project::prepare_runtime_launch(std::move(descriptor));
    REQUIRE(prepared);
    CHECK(prepared.value().manifest.project_guid == "11111111-1111-1111-1111-111111111111");
    CHECK(prepared.value().manifest.engine_version == "0.1.0");
    CHECK(prepared.value().manifest.target.cook_profile == "windows-x64-vulkan");
    CHECK(prepared.value().manifest.runtime_config == "config/runtime.json");
    REQUIRE(prepared.value().manifest.startup_scenes.size() == 2);
    CHECK(prepared.value().manifest.startup_scenes[0] == "22222222-2222-2222-2222-222222222222");
    CHECK(prepared.value().manifest.startup_scenes[1] == "33333333-3333-3333-3333-333333333333");
    const std::vector<std::string> expected_bundles = {"Engine", "Boot", "Startup/Common"};
    CHECK(prepared.value().startup_bundles == expected_bundles);
    CHECK(prepared.value().host.device_profile == "desktop-high");
}

TEST_CASE("D1 server is headless and may have no startup scene")
{
    auto descriptor = server();
    const auto prepared = project::prepare_runtime_launch(std::move(descriptor));
    REQUIRE(prepared);
    CHECK(prepared.value().role == project::runtime_launch_role::server);
    CHECK(prepared.value().manifest.target.renderer == "none");
    CHECK(prepared.value().manifest.startup_scenes.empty());
    CHECK_FALSE(prepared.value().host.capabilities.window_system);
}

TEST_CASE("D1 rejects host/manifest role mismatches and an empty client entry scene")
{
    auto descriptor = client();
    descriptor.role = project::runtime_launch_role::server;
    reject(descriptor, project::project_error_code::invalid_descriptor, "role");
    descriptor = client();
    descriptor.manifest.startup_scenes.clear();
    reject(descriptor, project::project_error_code::invalid_scene, "startupScenes");
    descriptor = client();
    descriptor.role = static_cast<project::runtime_launch_role>(99);
    reject(descriptor, project::project_error_code::invalid_descriptor, "role");
}

TEST_CASE("D1 rejects incompatible host target and malformed D0 manifest")
{
    auto descriptor = client();
    descriptor.host.capabilities.family = framework::platform_family::linux_os;
    reject(descriptor, project::project_error_code::unsupported_platform, "host.capabilities.family");

    descriptor = client();
    descriptor.manifest.format_version = 999;
    reject(descriptor, project::project_error_code::unsupported_version, "formatVersion");
}

TEST_CASE("D1 rejects relative, ambiguous and noncanonical native distribution paths")
{
    auto descriptor = client();
    descriptor.host.distribution.native_path = "Build/Packages/Game";
    reject(descriptor, project::project_error_code::unsafe_path, "host.distribution");
    descriptor = client();
    descriptor.host.distribution.native_path /= "..";
    reject(descriptor, project::project_error_code::unsafe_path, "host.distribution");
    descriptor = client();
    descriptor.host.distribution.virtual_root = arc::io::virtual_path::parse("package://game").value();
    reject(descriptor, project::project_error_code::unsafe_path, "host.distribution");
    descriptor = client();
    descriptor.host.writable_storage_root = "Saved";
    reject(descriptor, project::project_error_code::unsafe_path, "host.writableStorageRoot");
}

TEST_CASE("D1 permits an explicit relocated distribution directory without filesystem access")
{
    auto descriptor = client();
    descriptor.host.distribution.kind = project::runtime_distribution_location_kind::directory;
    descriptor.host.distribution.native_path = std::filesystem::temp_directory_path() / "staged game";
    descriptor.host.device_profile.clear();
    const auto prepared = project::prepare_runtime_launch(std::move(descriptor));
    REQUIRE(prepared);
    CHECK(prepared.value().host.distribution.kind == project::runtime_distribution_location_kind::directory);
}

TEST_CASE("D1 accepts Android package assets through the same mounted VFS contract")
{
    arc::jobs::job_system jobs;
    arc::io::virtual_file_system files(jobs);
    auto descriptor = client();
    descriptor.manifest.target.platform = "android";
    descriptor.manifest.target.architecture = "arm64";
    descriptor.manifest.target.id = "android-arm64-vulkan";
    descriptor.manifest.target.cook_profile = "android-arm64-vulkan";
    descriptor.manifest.finalizer = "android-gradle";
    descriptor.manifest.entrypoint = {.kind = "android-activity", .value = "com.example.arc.GameActivity"};
    descriptor.manifest.files.erase(descriptor.manifest.files.begin());
    descriptor.host.capabilities.family = framework::platform_family::android;
    descriptor.host.capabilities.window_system = false; // Surface can arrive after process startup.
    descriptor.host.capabilities.native_package_assets = true;
    descriptor.host.distribution.kind = project::runtime_distribution_location_kind::mounted_vfs;
    descriptor.host.distribution.native_path.clear();
    descriptor.host.distribution.files = &files;
    descriptor.host.distribution.virtual_root = arc::io::virtual_path::parse("package://app").value();

    const auto prepared = project::prepare_runtime_launch(std::move(descriptor));
    REQUIRE(prepared);
    CHECK(prepared.value().host.distribution.files == &files);
    CHECK(prepared.value().manifest.target.platform == "android");
}

TEST_CASE("D1 mounted assets require a live VFS and a virtual root")
{
    arc::jobs::job_system jobs;
    arc::io::virtual_file_system files(jobs);
    auto descriptor = client();
    descriptor.host.distribution.kind = project::runtime_distribution_location_kind::mounted_vfs;
    descriptor.host.distribution.native_path.clear();
    descriptor.host.distribution.virtual_root = arc::io::virtual_path::parse("package://app").value();
    reject(descriptor, project::project_error_code::invalid_descriptor, "host.distribution");
    descriptor.host.distribution.files = &files;
    descriptor.host.distribution.virtual_root = {};
    reject(descriptor, project::project_error_code::invalid_descriptor, "host.distribution");
}

TEST_CASE("D1 rejects duplicate bundle activation roots and invalid device identifiers")
{
    auto descriptor = client();
    descriptor.startup_bundles.push_back("Boot");
    reject(descriptor, project::project_error_code::invalid_descriptor, "startupBundles");
    descriptor = client();
    descriptor.startup_bundles[1].clear();
    reject(descriptor, project::project_error_code::invalid_descriptor, "startupBundles");
    descriptor = client();
    descriptor.host.device_profile = "bad\\profile";
    reject(descriptor, project::project_error_code::invalid_descriptor, "host.deviceProfile");
}
