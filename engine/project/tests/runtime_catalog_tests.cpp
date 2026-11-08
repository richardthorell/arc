#include <arc/project/runtime_catalog.h>

#include <arc/assets/assets.h>
#include <arc/assets/cook.h>
#include <arc/assets/runtime_content.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
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
namespace assets = arc::assets;
namespace io = arc::io;
namespace fs = std::filesystem;

io::file_buffer bytes(std::string_view text)
{
    const auto content = std::as_bytes(std::span(text.data(), text.size()));
    return {content.begin(), content.end()};
}

void write(const fs::path& path, std::string_view data)
{
    fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    stream.write(data.data(), static_cast<std::streamsize>(data.size()));
    REQUIRE(stream.good());
}

std::string read_text(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

io::virtual_path virtual_path(std::string_view text)
{
    auto parsed = io::virtual_path::parse(text);
    REQUIRE(parsed.succeeded());
    return std::move(parsed).value();
}

struct cooked_distribution_fixture
{
    fs::path root = fs::temp_directory_path() / ("arc-d1-3-" + assets::to_string(assets::generate_asset_guid()));
    assets::asset_guid scene = assets::generate_asset_guid();
    assets::asset_guid other = assets::generate_asset_guid();
    io::file_buffer payload = bytes("cooked scene payload");
    std::vector<std::string> bundle_ids{"Boot", "Engine", "Startup/Common"};
    std::string cook_manifest_relative;
    fs::path chunk;
    arc::jobs::job_system jobs{
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false}};
    io::async_file_service native_files{jobs};
    io::virtual_file_system files{jobs};

    ~cooked_distribution_fixture()
    {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }

    void stage(bool server = false, bool missing_scene = false, bool wrong_type = false,
               bool wrong_renderer = false)
    {
        assets::derived_data_cache cache({.root = root / "cache"});
        assets::cache_error error;
        const auto hash = assets::hash_bytes(payload);
        REQUIRE(cache.put_blob(hash, payload, error));

        assets::cook_manifest cooked;
        cooked.target = assets::windows_vulkan_cook_target();
        if (server) cooked.target.renderer = assets::cook_renderer::none;
        if (wrong_renderer) cooked.target.renderer = assets::cook_renderer::metal;
        cooked.build_id = "test-build";
        cooked.bundle_ids = bundle_ids;
        cooked.roots = {missing_scene ? other : scene};
        cooked.dependency_closure = cooked.roots;
        cooked.artifacts.push_back({.asset = missing_scene ? other : scene,
                                    .type = wrong_type ? assets::asset_types::binary_blob : assets::asset_types::scene,
                                    .name = "startup/scene",
                                    .schema = assets::artifact_schemas::scene,
                                    .schema_version = 1,
                                    .hash = hash,
                                    .size = payload.size(),
                                    .chunk = "startup"});
        const auto package = assets::build_asset_packages(std::move(cooked), cache, root / "Content");
        REQUIRE(package.succeeded());
        REQUIRE_FALSE(package.chunks.empty());
        chunk = package.chunks.front();
        cook_manifest_relative = fs::relative(package.manifest_path, root).generic_string();

        write(root / (server ? "bin/Server.exe" : "bin/Game.exe"), "arc binary");
        write(root / "Config/runtime.json", R"({"scene":"startup"})");

        project::distribution_manifest distribution;
        distribution.project_guid = "11111111-1111-1111-1111-111111111111";
        distribution.engine_version = "0.1.0";
        distribution.sdk_version = "0.1.0";
        distribution.target = {.id = server ? "windows-x64-headless" : "windows-x64-vulkan",
                               .platform = "windows",
                               .architecture = "x86_64",
                               .renderer = server ? "none" : "vulkan",
                               .configuration = "Shipping",
                               .cook_profile = server ? "windows-server" : "windows-client"};
        distribution.role = server ? "server" : "client";
        distribution.finalizer = "windows-directory";
        distribution.entrypoint = {.kind = "executable", .value = server ? "bin/Server.exe" : "bin/Game.exe"};
        if (!server) distribution.startup_scenes = {assets::to_string(scene)};
        distribution.runtime_config = "Config/runtime.json";
        distribution.cook_manifests = {cook_manifest_relative};

        const auto add_file = [&](const fs::path& file, std::string kind)
        {
            const auto hashed = assets::hash_file(file);
            REQUIRE(hashed);
            distribution.files.push_back({.path = fs::relative(file, root).generic_string(),
                                          .kind = std::move(kind),
                                          .size = fs::file_size(file),
                                          .sha256 = assets::to_string(hashed.value())});
        };
        add_file(root / distribution.entrypoint.value, "executable");
        add_file(root / distribution.runtime_config, "configuration");
        for (const auto& file : fs::recursive_directory_iterator(root / "Content"))
            if (file.is_regular_file()) add_file(file.path(), "content");
        const auto serialized = project::serialize_distribution_manifest(distribution);
        REQUIRE(serialized);
        write(root / "arc-distribution.json", serialized.value());
    }

    project::runtime_boot_context boot(bool server = false)
    {
        project::runtime_boot_request request;
        request.role = server ? project::runtime_launch_role::server : project::runtime_launch_role::client;
        request.host.distribution.kind = project::runtime_distribution_location_kind::directory;
        request.host.distribution.native_path = root;
        request.host.writable_storage_root = root / "UserData";
        request.host.capabilities.family = arc::framework::platform_family::windows;
        request.expected = {.engine_version = "0.1.0",
                            .sdk_version = "0.1.0",
                            .architecture = "x86_64",
                            .renderer = server ? "none" : "vulkan",
                            .configuration = "Shipping"};
        request.startup_bundles = {"Boot", "Startup/Common"};
        auto discovered = project::discover_runtime_boot(std::move(request));
        REQUIRE(discovered);
        return std::move(discovered).value();
    }

    void mount_to_memory(io::virtual_file_system& vfs, io::memory_file_provider& provider) const
    {
        std::vector<io::memory_provider_update> updates;
        for (const auto& file : fs::recursive_directory_iterator(root))
        {
            if (!file.is_regular_file()) continue;
            const auto relative = fs::relative(file.path(), root).generic_string();
            if (relative.starts_with("cache/")) continue;
            updates.push_back({.relative_path = "games/demo/" + relative, .bytes = bytes(read_text(file.path()))});
        }
        REQUIRE(provider.publish(updates).succeeded());
        REQUIRE(vfs.resolve("package://app/games/demo/arc-distribution.json").succeeded());
    }
};

void expect_failure(const project::runtime_catalog_request& request, project::project_error_code code,
                    std::string_view field)
{
    const auto result = project::bootstrap_runtime_catalog(request);
    REQUIRE_FALSE(result);
    CHECK(result.error().code == code);
    CHECK(result.error().field == field);
    CHECK_FALSE(result.error().message.empty());
}

} // namespace

TEST_CASE("D1.3 native catalog mounts cooked scene metadata without resident asset bytes")
{
    cooked_distribution_fixture fixture;
    fixture.stage();
    auto boot = fixture.boot();
    const auto before = fixture.files.telemetry().read_bytes;
    auto started = project::bootstrap_runtime_catalog({.boot = boot, .files = fixture.files,
                                                       .native_files = &fixture.native_files});
    REQUIRE(started);
    CHECK(started.value()->startup_scenes() == std::vector<assets::asset_guid>{fixture.scene});
    const std::vector<std::string> expected_bundles{"Boot", "Startup/Common"};
    CHECK(started.value()->startup_bundles() == expected_bundles);

    auto record = started.value()->catalog().database().query(fixture.scene, assets::asset_types::scene);
    REQUIRE(record);
    CHECK(record->residency == assets::asset_residency::metadata_only);
    CHECK(record->state == assets::asset_state::ready);
    CHECK(record->representations.front().schema == assets::artifact_schemas::scene);

    auto resolved = started.value()->catalog().resolve(
        {.asset = fixture.scene, .schema = assets::artifact_schemas::scene, .name = "startup/scene"});
    REQUIRE(resolved.succeeded());
    CHECK(resolved.value().size() == fixture.payload.size());
    CHECK(fixture.files.telemetry().read_bytes - before < fixture.payload.size() + 10000);

    const auto content = fixture.files.read_all(resolved.value()).get();
    REQUIRE(content.succeeded());
    CHECK(content.value() == fixture.payload);

    started.value().reset();
    CHECK_FALSE(fixture.files.resolve("package://runtime/Config/runtime.json").succeeded());
    CHECK_FALSE(fixture.files.resolve(resolved.value().path()).succeeded());
}

TEST_CASE("D1.3 headless server accepts an intentionally absent startup scene")
{
    cooked_distribution_fixture fixture;
    fixture.stage(true);
    auto boot = fixture.boot(true);
    auto session = project::bootstrap_runtime_catalog({.boot = boot, .files = fixture.files,
                                                       .native_files = &fixture.native_files});
    REQUIRE(session);
    CHECK(session.value()->startup_scenes().empty());
    CHECK(session.value()->startup_bundles().size() == 2);
}

TEST_CASE("D1.3 missing required Boot or Startup bundle rolls back all native mounts")
{
    cooked_distribution_fixture fixture;
    fixture.bundle_ids = {"Engine"};
    fixture.stage();
    auto boot = fixture.boot();
    expect_failure({.boot = boot, .files = fixture.files, .native_files = &fixture.native_files},
                   project::project_error_code::not_found, "startupBundles");
    CHECK_FALSE(fixture.files.resolve("package://runtime/Config/runtime.json").succeeded());
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.3 missing or wrong-type startup scene fails instead of creating an empty world")
{
    cooked_distribution_fixture fixture;
    fixture.stage(false, true);
    auto boot = fixture.boot();
    expect_failure({.boot = boot, .files = fixture.files, .native_files = &fixture.native_files},
                   project::project_error_code::invalid_scene, "startupScenes");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.3 refuses startup GUIDs whose cooked records are not scene assets")
{
    cooked_distribution_fixture fixture;
    fixture.stage(false, false, true);
    auto boot = fixture.boot();
    expect_failure({.boot = boot, .files = fixture.files, .native_files = &fixture.native_files},
                   project::project_error_code::invalid_scene, "startupScenes");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.3 verifies cooked target against packaged platform architecture renderer and configuration")
{
    cooked_distribution_fixture fixture;
    fixture.stage(false, false, false, true);
    auto boot = fixture.boot();
    expect_failure({.boot = boot, .files = fixture.files, .native_files = &fixture.native_files},
                   project::project_error_code::incompatible_engine, "target.renderer");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.3 a missing cooked package chunk triggers atomic rollback")
{
    cooked_distribution_fixture fixture;
    fixture.stage();
    auto boot = fixture.boot();
    REQUIRE(fs::remove(fixture.chunk));
    expect_failure({.boot = boot, .files = fixture.files, .native_files = &fixture.native_files},
                   project::project_error_code::invalid_descriptor, "content.cookManifests");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.3 rejects native bootstrap without a native file service")
{
    cooked_distribution_fixture fixture;
    fixture.stage();
    auto boot = fixture.boot();
    expect_failure({.boot = boot, .files = fixture.files},
                   project::project_error_code::invalid_descriptor, "host.distribution");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.3 mounted VFS distribution uses host provider without taking ownership")
{
    cooked_distribution_fixture fixture;
    fixture.stage();
    auto provider = std::make_shared<io::memory_file_provider>(fixture.jobs);
    REQUIRE(fixture.files.mount({.root = virtual_path("package://app"), .provider = provider}).succeeded());
    fixture.mount_to_memory(fixture.files, *provider);

    project::runtime_boot_request request;
    request.host.distribution.kind = project::runtime_distribution_location_kind::mounted_vfs;
    request.host.distribution.files = &fixture.files;
    request.host.distribution.virtual_root = virtual_path("package://app/games/demo");
    request.host.writable_storage_root = fixture.root / "UserData";
    request.host.capabilities.family = arc::framework::platform_family::windows;
    request.expected = {.engine_version = "0.1.0",
                        .sdk_version = "0.1.0",
                        .architecture = "x86_64",
                        .renderer = "vulkan",
                        .configuration = "Shipping"};
    request.startup_bundles = {"Boot", "Startup/Common"};
    auto boot = project::discover_runtime_boot(std::move(request));
    REQUIRE(boot);
    auto mounted = project::bootstrap_runtime_catalog({.boot = boot.value(), .files = fixture.files});
    REQUIRE(mounted);
    CHECK(mounted.value()->startup_scenes() == std::vector<assets::asset_guid>{fixture.scene});
    CHECK(mounted.value()->catalog().database().query(fixture.scene).has_value());
    CHECK(fixture.files.telemetry().mounted_providers == 2);
    mounted.value().reset();
    CHECK(fixture.files.telemetry().mounted_providers == 1);
    CHECK(fixture.files.resolve("package://app/games/demo/arc-distribution.json").succeeded());
}

TEST_CASE("D1.3 rejects an artifact namespace overlapping the distribution provider")
{
    cooked_distribution_fixture fixture;
    fixture.stage();
    auto boot = fixture.boot();
    expect_failure({.boot = boot, .files = fixture.files, .native_files = &fixture.native_files,
                    .artifact_mount = virtual_path("package://runtime")},
                   project::project_error_code::invalid_descriptor, "host.distribution");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}
