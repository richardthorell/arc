#include <arc/project/runtime_application.h>

#include <arc/assets/cook.h>
#include <arc/framework/runtime.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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
namespace assets = arc::assets;
namespace framework = arc::framework;
namespace fs = std::filesystem;

arc::io::file_buffer bytes(std::string_view source)
{
    const auto span = std::as_bytes(std::span(source.data(), source.size()));
    return {span.begin(), span.end()};
}

void write(const fs::path& path, std::string_view text)
{
    fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    REQUIRE(stream.good());
}

struct packaged_world_fixture
{
    fs::path root = fs::temp_directory_path() / ("arc-d1-4-" + assets::to_string(assets::generate_asset_guid()));
    assets::asset_guid first = assets::generate_asset_guid();
    assets::asset_guid second = assets::generate_asset_guid();
    arc::io::file_buffer scene_data = bytes("valid cooked scene");
    arc::jobs::job_system jobs{
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false}};
    arc::io::async_file_service native_files{jobs};
    arc::io::virtual_file_system files{jobs};
    fs::path cook_path;
    fs::path chunk_path;
    std::uint64_t first_offset{};

    ~packaged_world_fixture()
    {
        std::error_code error;
        fs::remove_all(root, error);
    }

    void stage(bool server = false, bool two_scenes = false)
    {
        assets::derived_data_cache cache({.root = root / "cache"});
        assets::cache_error error;
        const auto hash = assets::hash_bytes(scene_data);
        REQUIRE(cache.put_blob(hash, scene_data, error));

        assets::cook_manifest cooked;
        cooked.build_id = "game-build";
        cooked.bundle_ids = {"Boot", "Startup/Common"};
        cooked.target = assets::windows_vulkan_cook_target();
        if (server) cooked.target.renderer = assets::cook_renderer::none;

        const auto add_scene = [&](assets::asset_guid guid)
        {
            cooked.roots.push_back(guid);
            cooked.dependency_closure.push_back(guid);
            cooked.artifacts.push_back({.asset = guid,
                                        .type = assets::asset_types::scene,
                                        .name = "main/scene",
                                        .schema = assets::artifact_schemas::scene,
                                        .schema_version = 1,
                                        .hash = hash,
                                        .size = scene_data.size(),
                                        .chunk = "startup"});
        };
        if (!server || two_scenes) add_scene(first);
        if (two_scenes) add_scene(second);

        const auto result = assets::build_asset_packages(std::move(cooked), cache, root / "Content");
        REQUIRE(result.succeeded());
        cook_path = result.manifest_path;
        if (!result.chunks.empty())
        {
            chunk_path = result.chunks.front();
            const auto loaded = assets::load_cook_manifest(cook_path);
            REQUIRE(loaded);
            first_offset = loaded.value().artifacts.front().offset;
        }

        const auto executable = server ? "bin/Server.exe" : "bin/Game.exe";
        write(root / executable, "exe");
        write(root / "Config/runtime.json", "{}");

        project::distribution_manifest manifest;
        manifest.project_guid = "11111111-1111-1111-1111-111111111111";
        manifest.engine_version = "0.1.0";
        manifest.sdk_version = "0.1.0";
        manifest.target = {.id = server ? "server-windows" : "client-windows",
                           .platform = "windows",
                           .architecture = "x86_64",
                           .renderer = server ? "none" : "vulkan",
                           .configuration = "Shipping",
                           .cook_profile = server ? "server-windows" : "client-windows"};
        manifest.role = server ? "server" : "client";
        manifest.finalizer = "windows-directory";
        manifest.entrypoint = {.kind = "executable", .value = executable};
        if (!server || two_scenes) manifest.startup_scenes.push_back(assets::to_string(first));
        if (two_scenes) manifest.startup_scenes.push_back(assets::to_string(second));
        manifest.runtime_config = "Config/runtime.json";
        manifest.cook_manifests = {fs::relative(cook_path, root).generic_string()};

        const auto add_file = [&](const fs::path& file, std::string kind)
        {
            const auto digest = assets::hash_file(file);
            REQUIRE(digest);
            manifest.files.push_back({.path = fs::relative(file, root).generic_string(),
                                      .kind = std::move(kind),
                                      .size = fs::file_size(file),
                                      .sha256 = assets::to_string(digest.value())});
        };
        add_file(root / executable, "executable");
        add_file(root / manifest.runtime_config, "configuration");
        for (const auto& file : fs::recursive_directory_iterator(root / "Content"))
            if (file.is_regular_file()) add_file(file.path(), "content");
        const auto serialized = project::serialize_distribution_manifest(manifest);
        REQUIRE(serialized);
        write(root / "arc-distribution.json", serialized.value());
    }

    project::runtime_boot_context boot(bool server = false)
    {
        project::runtime_boot_request request;
        request.role = server ? project::runtime_launch_role::server : project::runtime_launch_role::client;
        request.host.distribution.kind = project::runtime_distribution_location_kind::directory;
        request.host.distribution.native_path = root;
        request.host.writable_storage_root = root / "Saved";
        request.host.capabilities.family = framework::platform_family::windows;
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

    std::unique_ptr<project::runtime_catalog_session> catalog(const project::runtime_boot_context& boot)
    {
        auto mounted =
            project::bootstrap_runtime_catalog({.boot = boot, .files = files, .native_files = &native_files});
        REQUIRE(mounted);
        return std::move(mounted).value();
    }

    void corrupt_scene_bytes()
    {
        REQUIRE_FALSE(chunk_path.empty());
        std::fstream stream(chunk_path, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(stream.good());
        stream.seekp(static_cast<std::streamoff>(first_offset));
        const std::string invalid(scene_data.size(), 'X');
        stream.write(invalid.data(), static_cast<std::streamsize>(invalid.size()));
        REQUIRE(stream.good());
    }
};

class recording_module final : public framework::module
{
public:
    explicit recording_module(std::vector<std::string>& events) : events_(&events) {}
    std::string_view name() const override
    {
        return "packaged.test.module";
    }
    void on_start(framework::module_context&) override
    {
        events_->push_back("module:start");
    }
    void on_shutdown(framework::module_context&) override
    {
        events_->push_back("module:stop");
    }

private:
    std::vector<std::string>* events_{};
};

} // namespace

TEST_CASE("D1.4 packaged client loads verified scene bytes then explicitly registers ordered worlds and modules")
{
    packaged_world_fixture fixture;
    fixture.stage(false, true);
    auto boot = fixture.boot();
    auto catalog = fixture.catalog(boot);

    std::vector<assets::asset_guid> materialized;
    std::vector<std::string> events;
    project::packaged_game_registrations registrations;
    registrations.services = [&](framework::runtime_service_registry&) { events.push_back("services:register"); };
    registrations.modules = [&](framework::module_registry& registry)
    {
        events.push_back("modules:register");
        registry.emplace<recording_module>(events);
    };
    registrations.materialize_scene = [&](framework::runtime_world& world, const project::packaged_scene_payload& scene)
    {
        materialized.push_back(scene.guid);
        CHECK(scene.bytes == fixture.scene_data);
        CHECK(scene.artifact_name == "main/scene");
        CHECK(world.entities().live_count() == 0);
        CHECK(world.entities().create().valid());
        events.push_back("scene:materialized");
        return project::project_status::success();
    };

    auto app = project::prepare_packaged_application(
        {.boot = boot, .catalog = *catalog, .registrations = std::move(registrations)});
    REQUIRE(app);
    CHECK(app.value()->scenes().size() == 2);
    CHECK(app.value()->role() == project::runtime_launch_role::client);
    framework::runtime host(*app.value());
    host.start();
    REQUIRE(host.worlds().size() == 2);
    CHECK(materialized == std::vector<assets::asset_guid>{fixture.first, fixture.second});
    for (const auto id : host.worlds().ordered_worlds())
    {
        const auto* world = host.worlds().find(id);
        REQUIRE(world);
        CHECK(world->role() == framework::runtime_world_role::client);
        CHECK(world->entities().live_count() == 1);
        CHECK(world->state() == framework::runtime_world_state::running);
    }
    CHECK(events.front() == "services:register");
    CHECK(events.back() == "module:start");
    host.advance(1.0 / 60.0);
    host.shutdown();
    CHECK(events.back() == "module:stop");
}

TEST_CASE("D1.4 dedicated server explicitly creates a headless world with no startup scenes")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    auto boot = fixture.boot(true);
    auto catalog = fixture.catalog(boot);
    std::size_t initialized{};
    auto app = project::prepare_packaged_application(
        {.boot = boot,
         .catalog = *catalog,
         .registrations = {.initialize_headless_world = [&](framework::runtime_world& world)
                           {
                               CHECK(world.role() == framework::runtime_world_role::server);
                               ++initialized;
                               return project::project_status::success();
                           }}});
    REQUIRE(app);
    CHECK(app.value()->configure().simulation.headless);
    CHECK_FALSE(app.value()->configure().simulation.presentation_enabled);
    CHECK_FALSE(app.value()->configure().visible);
    framework::runtime host(*app.value());
    host.start();
    REQUIRE(host.worlds().size() == 1);
    const auto id = host.worlds().ordered_worlds().front();
    CHECK(host.worlds().find(id)->role() == framework::runtime_world_role::server);
    CHECK(initialized == 1);
    host.advance(1.0 / 60.0);
    CHECK(host.current_tick().id.value == 1);
    host.shutdown();
}

TEST_CASE("D1.4 missing scene materializer fails before world creation")
{
    packaged_world_fixture fixture;
    fixture.stage();
    auto boot = fixture.boot();
    auto catalog = fixture.catalog(boot);
    const auto app = project::prepare_packaged_application({.boot = boot, .catalog = *catalog});
    REQUIRE_FALSE(app);
    CHECK(app.error().code == project::project_error_code::missing_module);
    CHECK(app.error().field == "materializeScene");
}

TEST_CASE("D1.4 incorrect startup GUID order fails instead of launching a different scene")
{
    packaged_world_fixture fixture;
    fixture.stage(false, true);
    auto boot = fixture.boot();
    auto catalog = fixture.catalog(boot);
    std::reverse(boot.launch.manifest.startup_scenes.begin(), boot.launch.manifest.startup_scenes.end());
    auto app = project::prepare_packaged_application(
        {.boot = boot,
         .catalog = *catalog,
         .registrations = {.materialize_scene = [](framework::runtime_world&, const project::packaged_scene_payload&)
                           { return project::project_status::success(); }}});
    REQUIRE_FALSE(app);
    CHECK(app.error().code == project::project_error_code::invalid_scene);
    CHECK(app.error().field == "startupScenes");
}

TEST_CASE("D1.4 rejects corrupted cooked scene payload before a runtime exists")
{
    packaged_world_fixture fixture;
    fixture.stage();
    auto boot = fixture.boot();
    auto catalog = fixture.catalog(boot);
    fixture.corrupt_scene_bytes();
    auto app = project::prepare_packaged_application(
        {.boot = boot,
         .catalog = *catalog,
         .registrations = {.materialize_scene = [](framework::runtime_world&, const project::packaged_scene_payload&)
                           { return project::project_status::success(); }}});
    REQUIRE_FALSE(app);
    CHECK(app.error().code == project::project_error_code::corrupt_content);
    CHECK(app.error().field == "startupScenes");
}

TEST_CASE("D1.4 scene materialization failure rolls back worlds and propagates structured error")
{
    packaged_world_fixture fixture;
    fixture.stage(false, true);
    auto boot = fixture.boot();
    auto catalog = fixture.catalog(boot);
    std::size_t attempted{};
    auto app = project::prepare_packaged_application(
        {.boot = boot,
         .catalog = *catalog,
         .registrations = {
             .materialize_scene = [&](framework::runtime_world& world, const project::packaged_scene_payload&)
             {
                 CHECK(world.entities().create().valid());
                 if (++attempted == 2)
                     return project::project_status::failure({.code = project::project_error_code::invalid_scene,
                                                              .field = "startupScenes",
                                                              .message = "test scene decode failure"});
                 return project::project_status::success();
             }}});
    REQUIRE(app);
    framework::runtime host(*app.value());
    try
    {
        host.start();
        FAIL("Materialization failure should reject packaged runtime startup");
    }
    catch (const project::packaged_runtime_startup_error& error)
    {
        CHECK(error.detail().code == project::project_error_code::invalid_scene);
        CHECK(error.detail().message == "test scene decode failure");
    }
    CHECK(attempted == 2);
    CHECK_FALSE(host.started());
    CHECK(host.worlds().size() == 0);
}

TEST_CASE("D1.4 server world initialization errors also roll back without fallback worlds")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    auto boot = fixture.boot(true);
    auto catalog = fixture.catalog(boot);
    auto app = project::prepare_packaged_application(
        {.boot = boot,
         .catalog = *catalog,
         .registrations = {.initialize_headless_world = [](framework::runtime_world&)
                           {
                               return project::project_status::failure(
                                   {.code = project::project_error_code::missing_module,
                                    .field = "server",
                                    .message = "Server bootstrap registration unavailable"});
                           }}});
    REQUIRE(app);
    framework::runtime host(*app.value());
    CHECK_THROWS_AS(host.start(), project::packaged_runtime_startup_error);
    CHECK(host.worlds().size() == 0);
    CHECK_FALSE(host.started());
}
