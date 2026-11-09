#include <arc/project/headless_server.h>
#include <arc/project/runtime_session.h>

#include <arc/assets/cook.h>
#include <arc/framework/runtime.h>
#include <arc/persistence/persistence.h>
#include <arc/scene/components.h>
#include <arc/scene/persistence.h>
#include <arc/scene/runtime_scene.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
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

class lifecycle_module final : public framework::module
{
public:
    lifecycle_module(std::vector<std::string>& events, bool throw_on_shutdown = false, bool throw_on_event = false)
        : events_(&events), throw_on_shutdown_(throw_on_shutdown), throw_on_event_(throw_on_event)
    {
    }

    std::string_view name() const override
    {
        return "packaged.lifecycle.test";
    }
    void on_start(framework::module_context&) override
    {
        events_->push_back("module:start");
    }
    void on_update(framework::module_context&, const framework::frame_time&) override
    {
        events_->push_back("module:update");
    }
    void on_event(framework::module_context&, const framework::event& event) override
    {
        events_->push_back("module:event:" + std::to_string(static_cast<int>(event.type)));
        if (throw_on_event_) throw std::runtime_error("deliberate event error");
    }
    void on_shutdown(framework::module_context&) override
    {
        events_->push_back("module:shutdown");
        if (throw_on_shutdown_) throw std::runtime_error("deliberate shutdown error");
    }

private:
    std::vector<std::string>* events_{};
    bool throw_on_shutdown_{};
    bool throw_on_event_{};
};

class failure_module final : public framework::module
{
public:
    failure_module(std::string name, std::vector<std::string>& events, bool fail_start = false,
                   bool fail_shutdown = false)
        : name_(std::move(name)), events_(&events), fail_start_(fail_start), fail_shutdown_(fail_shutdown)
    {
    }

    std::string_view name() const override
    {
        return name_;
    }
    void on_start(framework::module_context&) override
    {
        events_->push_back(name_ + ":start");
        if (fail_start_) throw std::runtime_error(name_ + ":startup failure");
    }
    void on_shutdown(framework::module_context&) override
    {
        events_->push_back(name_ + ":shutdown");
        if (fail_shutdown_) throw std::runtime_error(name_ + ":shutdown failure");
    }

private:
    std::string name_;
    std::vector<std::string>* events_{};
    bool fail_start_{};
    bool fail_shutdown_{};
};

class lifecycle_service final : public framework::runtime_service
{
public:
    explicit lifecycle_service(std::vector<std::string>& events) : events_(&events) {}

    framework::runtime_service_id id() const noexcept override
    {
        return framework::make_runtime_service_id("d1.5.lifecycle.service");
    }
    std::string_view name() const noexcept override
    {
        return "d1.5.lifecycle.service";
    }
    void on_start(framework::runtime_service_context&) override
    {
        events_->push_back("service:start");
    }
    void on_shutdown(framework::runtime_service_context&) noexcept override
    {
        events_->push_back("service:shutdown");
    }

private:
    std::vector<std::string>* events_{};
};

project::packaged_runtime_request request_for(packaged_world_fixture& fixture, bool server = false)
{
    project::runtime_boot_request boot;
    boot.role = server ? project::runtime_launch_role::server : project::runtime_launch_role::client;
    boot.host.distribution.kind = project::runtime_distribution_location_kind::directory;
    boot.host.distribution.native_path = fixture.root;
    boot.host.writable_storage_root = fixture.root / "Saved";
    boot.host.capabilities.family = framework::platform_family::windows;
    boot.expected = {.engine_version = "0.1.0",
                     .sdk_version = "0.1.0",
                     .architecture = "x86_64",
                     .renderer = server ? "none" : "vulkan",
                     .configuration = "Shipping"};
    boot.startup_bundles = {"Boot", "Startup/Common"};

    project::packaged_runtime_request request{
        .boot = std::move(boot), .files = fixture.files, .native_files = &fixture.native_files};
    request.registrations.materialize_scene =
        [](framework::runtime_world& world, const project::packaged_scene_payload& scene)
    {
        if (scene.bytes.empty())
            return project::project_status::failure(
                {.code = project::project_error_code::invalid_scene, .message = "Empty cooked scene"});
        if (!world.entities().create().valid())
            return project::project_status::failure(
                {.code = project::project_error_code::invalid_scene, .message = "Cannot create scene entity"});
        return project::project_status::success();
    };
    return request;
}

} // namespace

TEST_CASE("D1.5 client session forwards lifecycle events, suspends ticks, and shuts down in ownership order")
{
    packaged_world_fixture fixture;
    fixture.stage();
    std::vector<std::string> events;
    auto request = request_for(fixture);
    request.registrations.modules = [&](framework::module_registry& registry)
    { registry.emplace<lifecycle_module>(events); };
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE(started);
    auto session = std::move(started).value();
    REQUIRE(session->runtime());
    CHECK(session->state() == project::packaged_runtime_state::running);
    CHECK(session->boot().launch.role == project::runtime_launch_role::client);
    REQUIRE(session->catalog());
    REQUIRE(session->runtime()->worlds().size() == 1);
    CHECK(fixture.files.telemetry().mounted_providers == 2);

    auto frame = session->advance(1.0 / 60.0);
    REQUIRE(frame);
    CHECK(frame.value().completed_ticks == 1);
    const auto tick = session->runtime()->current_tick().id.value;

    REQUIRE(session->pause());
    REQUIRE(session->pause());
    CHECK(session->state() == project::packaged_runtime_state::suspended);
    frame = session->advance(4.0);
    REQUIRE(frame);
    CHECK(session->runtime()->current_tick().id.value == tick);
    REQUIRE(session->surface_lost());
    REQUIRE(session->surface_lost());
    REQUIRE(session->resume());
    CHECK(session->state() == project::packaged_runtime_state::suspended);
    frame = session->advance(2.0);
    REQUIRE(frame);
    CHECK(session->runtime()->current_tick().id.value == tick);
    CHECK_FALSE(session->surface_ready(640, 0));
    CHECK(session->state() == project::packaged_runtime_state::suspended);
    REQUIRE(session->surface_ready(1280, 720));
    REQUIRE(session->surface_ready(0, 0));
    CHECK(session->state() == project::packaged_runtime_state::running);
    REQUIRE(session->advance(1.0 / 60.0));
    CHECK(session->runtime()->current_tick().id.value == tick + 1);

    REQUIRE(session->dispatch({.type = framework::event_type::close_requested}));
    CHECK(session->state() == project::packaged_runtime_state::stop_requested);
    CHECK_FALSE(session->advance(1.0 / 60.0));
    REQUIRE(session->shutdown());
    REQUIRE(session->shutdown());
    CHECK(session->state() == project::packaged_runtime_state::stopped);
    CHECK(session->runtime() == nullptr);
    CHECK(session->catalog() == nullptr);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
    CHECK_FALSE(fixture.files.resolve("package://runtime/Config/runtime.json").succeeded());
    CHECK(events.front() == "module:start");
    CHECK(events.back() == "module:shutdown");
}

TEST_CASE("D1.5 repeated headless sessions reuse host file services without leaking package mounts")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    for (int iteration = 0; iteration < 3; ++iteration)
    {
        auto started = project::start_packaged_runtime(request_for(fixture, true));
        REQUIRE(started);
        auto session = std::move(started).value();
        CHECK(session->state() == project::packaged_runtime_state::running);
        REQUIRE(session->runtime());
        CHECK(session->runtime()->config().simulation.headless);
        REQUIRE(session->runtime()->worlds().size() == 1);
        CHECK_FALSE(session->surface_lost());
        CHECK(session->state() == project::packaged_runtime_state::running);
        auto frame = session->advance(1.0 / 60.0);
        REQUIRE(frame);
        CHECK(frame.value().completed_ticks == 1);
        REQUIRE(session->pause());
        CHECK(session->state() == project::packaged_runtime_state::suspended);
        REQUIRE(session->resume());
        CHECK(session->state() == project::packaged_runtime_state::running);
        session->request_stop();
        session->request_stop();
        CHECK(session->state() == project::packaged_runtime_state::stop_requested);
        CHECK_FALSE(session->dispatch({.type = framework::event_type::focus_lost}));
        REQUIRE(session->shutdown());
        CHECK(fixture.files.telemetry().mounted_providers == 0);
    }
}

TEST_CASE("D1.5 missing cooked startup scene materializer never leaks distribution mounts")
{
    packaged_world_fixture fixture;
    fixture.stage();
    auto request = request_for(fixture);
    request.registrations.materialize_scene = {};
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE_FALSE(started);
    CHECK(started.error().code == project::project_error_code::missing_module);
    CHECK(started.error().field == "materializeScene");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.5 corrupt packaged startup bytes fail closed before runtime startup")
{
    packaged_world_fixture fixture;
    fixture.stage();
    fixture.corrupt_scene_bytes();
    auto started = project::start_packaged_runtime(request_for(fixture));
    REQUIRE_FALSE(started);
    CHECK(started.error().code == project::project_error_code::corrupt_content);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.5 failed project scene setup propagates error and unmounts every owned provider")
{
    packaged_world_fixture fixture;
    fixture.stage();
    auto request = request_for(fixture);
    request.registrations.materialize_scene = [](framework::runtime_world&,
                                                 const project::packaged_scene_payload&) -> project::project_status
    { throw std::runtime_error("deliberate scene failure"); };
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE_FALSE(started);
    CHECK(started.error().code == project::project_error_code::process_failed);
    CHECK(started.error().field == "startup");
    CHECK(started.error().message.find("deliberate scene failure") != std::string::npos);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.5 shutdown exceptions return errors but still release runtime and cooked mounts")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    std::vector<std::string> events;
    auto request = request_for(fixture, true);
    request.registrations.modules = [&](framework::module_registry& registry)
    { registry.emplace<lifecycle_module>(events, true); };
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE(started);
    auto session = std::move(started).value();
    const auto shutdown = session->shutdown();
    REQUIRE_FALSE(shutdown);
    CHECK(shutdown.error().code == project::project_error_code::process_failed);
    CHECK(shutdown.error().field == "shutdown");
    CHECK(events.back() == "module:shutdown");
    CHECK(session->state() == project::packaged_runtime_state::stopped);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
    REQUIRE(session->shutdown());
}

TEST_CASE("D1.5 a throwing module cannot prevent other module shutdown callbacks")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    std::vector<std::string> events;
    auto request = request_for(fixture, true);
    request.registrations.modules = [&](framework::module_registry& registry)
    {
        registry.emplace<failure_module>("normal", events);
        registry.emplace<failure_module>("throwing", events, false, true);
    };
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE(started);
    auto session = std::move(started).value();
    auto shutdown = session->shutdown();
    REQUIRE_FALSE(shutdown);
    CHECK(shutdown.error().message.find("throwing:shutdown failure") != std::string::npos);
    REQUIRE(events.size() == 4);
    CHECK(events[0] == "normal:start");
    CHECK(events[1] == "throwing:start");
    CHECK(events[2] == "throwing:shutdown");
    CHECK(events[3] == "normal:shutdown");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.5 failing module startup preserves original error and still stops services")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    std::vector<std::string> events;
    auto request = request_for(fixture, true);
    request.registrations.services = [&](framework::runtime_service_registry& registry)
    { registry.emplace<lifecycle_service>(events); };
    request.registrations.modules = [&](framework::module_registry& registry)
    {
        registry.emplace<failure_module>("cleanup_throws", events, false, true);
        registry.emplace<failure_module>("start_throws", events, true);
    };
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE_FALSE(started);
    CHECK(started.error().code == project::project_error_code::process_failed);
    CHECK(started.error().message.find("start_throws:startup failure") != std::string::npos);
    REQUIRE(events.size() == 5);
    CHECK(events[0] == "service:start");
    CHECK(events[1] == "cleanup_throws:start");
    CHECK(events[2] == "start_throws:start");
    CHECK(events[3] == "cleanup_throws:shutdown");
    CHECK(events[4] == "service:shutdown");
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.5 faulted simulation world is reported and unmounted")
{
    packaged_world_fixture fixture;
    fixture.stage();
    auto request = request_for(fixture);
    request.registrations.materialize_scene =
        [](framework::runtime_world& world, const project::packaged_scene_payload&)
    {
        world.systems().add(
            {.name = "intentional fault",
             .phase = arc::ecs::system_phase::movement,
             .execute = [](arc::ecs::system_context&) { throw std::runtime_error("simulated system failure"); }});
        return project::project_status::success();
    };
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE(started);
    auto session = std::move(started).value();
    auto result = session->advance(1.0 / 60.0);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "worlds");
    CHECK(result.error().message.find("simulated system failure") != std::string::npos);
    CHECK(session->state() == project::packaged_runtime_state::stopped);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.5 module event exception is reported and forces safe teardown")
{
    packaged_world_fixture fixture;
    fixture.stage();
    std::vector<std::string> events;
    auto request = request_for(fixture);
    request.registrations.modules = [&](framework::module_registry& registry)
    { registry.emplace<lifecycle_module>(events, false, true); };
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE(started);
    auto session = std::move(started).value();
    const auto outcome = session->dispatch({.type = framework::event_type::key_down, .key_code = 65});
    REQUIRE_FALSE(outcome);
    CHECK(outcome.error().field == "dispatch");
    CHECK(session->state() == project::packaged_runtime_state::stopped);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.5 missing distribution fails before any catalog mount")
{
    packaged_world_fixture fixture;
    auto started = project::start_packaged_runtime(request_for(fixture));
    REQUIRE_FALSE(started);
    CHECK(started.error().code == project::project_error_code::not_found);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.6 packaged headless server boots staged D0 content and runs exactly the requested ticks")
{
    packaged_world_fixture fixture;
    fixture.stage(true);

    std::size_t world_initializations{};
    std::size_t simulated_ticks{};
    auto request = request_for(fixture, true);
    request.registrations.initialize_headless_world = [&](framework::runtime_world& world)
    {
        ++world_initializations;
        CHECK(world.role() == framework::runtime_world_role::server);
        const bool installed = world.systems().add({.name = "d1.6.server.tick",
                                                    .phase = arc::ecs::system_phase::movement,
                                                    .execute = [&](arc::ecs::system_context&) { ++simulated_ticks; }});
        if (!installed)
            return project::project_status::failure({.code = project::project_error_code::process_failed,
                                                     .message = "Cannot install the packaged server tick system"});
        return project::project_status::success();
    };

    const auto result = project::run_packaged_headless_server(
        {.runtime = std::move(request), .maximum_ticks = 3, .sleep_to_clock = false});
    REQUIRE(result);
    CHECK(result.value() == 3);
    CHECK(world_initializations == 1);
    CHECK(simulated_ticks == 3);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.6 headless server shutdown signal is polled at a safe frame boundary")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    std::size_t polls{};
    const auto result = project::run_packaged_headless_server({.runtime = request_for(fixture, true),
                                                               .sleep_to_clock = false,
                                                               .stop_requested = [&] { return ++polls == 2; }});
    REQUIRE(result);
    CHECK(result.value() == 1);
    CHECK(polls == 2);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.6 headless runner refuses client role and a non-headless compiled renderer")
{
    packaged_world_fixture fixture;
    auto client = request_for(fixture);
    const auto client_result = project::run_packaged_headless_server(
        {.runtime = std::move(client), .maximum_ticks = 1, .sleep_to_clock = false});
    REQUIRE_FALSE(client_result);
    CHECK(client_result.error().code == project::project_error_code::invalid_descriptor);
    CHECK(client_result.error().field == "role");

    auto server = request_for(fixture, true);
    server.boot.expected.renderer = "vulkan";
    const auto renderer_result = project::run_packaged_headless_server(
        {.runtime = std::move(server), .maximum_ticks = 1, .sleep_to_clock = false});
    REQUIRE_FALSE(renderer_result);
    CHECK(renderer_result.error().code == project::project_error_code::invalid_descriptor);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.6 server fails closed without a staged manifest and leaves host providers untouched")
{
    packaged_world_fixture fixture;
    const auto result = project::run_packaged_headless_server(
        {.runtime = request_for(fixture, true), .maximum_ticks = 1, .sleep_to_clock = false});
    REQUIRE_FALSE(result);
    CHECK(result.error().code == project::project_error_code::not_found);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.6 server propagates shutdown callback errors and still unmounts packages")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    std::vector<std::string> events;
    auto request = request_for(fixture, true);
    request.registrations.modules = [&](framework::module_registry& modules)
    { modules.emplace<failure_module>("shutdown_failure", events, false, true); };

    const auto result = project::run_packaged_headless_server(
        {.runtime = std::move(request), .maximum_ticks = 1, .sleep_to_clock = false});
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "shutdown");
    CHECK(result.error().message.find("shutdown_failure") != std::string::npos);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.6 failing host stop callback preserves diagnostic and shuts the server down")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    const auto result = project::run_packaged_headless_server(
        {.runtime = request_for(fixture, true), .sleep_to_clock = false, .stop_requested = []() -> bool {
             throw std::runtime_error("host stop signal failed");
         }});
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "serverLoop");
    CHECK(result.error().message.find("host stop signal failed") != std::string::npos);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.6 server surfaces faulted simulation worlds and destroys the session")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    auto request = request_for(fixture, true);
    request.registrations.initialize_headless_world = [](framework::runtime_world& world)
    {
        const bool installed = world.systems().add(
            {.name = "d1.6.intentional-fault",
             .phase = arc::ecs::system_phase::movement,
             .execute = [](arc::ecs::system_context&) { throw std::runtime_error("server simulation fault"); }});
        if (!installed)
            return project::project_status::failure(
                {.code = project::project_error_code::process_failed, .message = "Could not register system"});
        return project::project_status::success();
    };
    const auto result = project::run_packaged_headless_server(
        {.runtime = std::move(request), .maximum_ticks = 2, .sleep_to_clock = false});
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "worlds");
    CHECK(result.error().message.find("server simulation fault") != std::string::npos);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

namespace
{

arc::io::file_buffer cooked_startup_scene()
{
    arc::persistence::component_persistence_registry registry;
    REQUIRE(arc::scene::register_persistence_components(registry));

    arc::persistence::archive_document scene;
    scene.kind = arc::persistence::document_kind::scene;
    scene.id = {0x123, 0x456};
    scene.name = "D1.8 packaged integration scene";
    arc::persistence::archive_entity_record entity;
    entity.id = {0x789, 0xabc};
    const arc::scene::name_component name{"Relocated Player"};
    arc::persistence::archive_component_record component;
    REQUIRE(registry.encode(arc::ecs::component_metadata<arc::scene::name_component>().id, &name, component));
    entity.components.push_back(std::move(component));
    scene.entities.push_back(std::move(entity));

    auto cooked = arc::persistence::write_tagged_binary(
        scene, assets::canonical_cook_target(assets::windows_vulkan_cook_target()));
    REQUIRE(cooked);
    return std::move(cooked).value();
}

void relocate_distribution(packaged_world_fixture& fixture)
{
    // A staged executable must never depend on the cooker cache or source checkout.
    std::error_code error;
    fs::remove_all(fixture.root / "cache", error);
    REQUIRE_FALSE(error);
    const fs::path relocated =
        fs::temp_directory_path() / ("arc-relocated-d1-8-" + assets::to_string(assets::generate_asset_guid()));
    REQUIRE_FALSE(fs::exists(relocated));
    fs::rename(fixture.root, relocated, error);
    REQUIRE_FALSE(error);
    fixture.root = relocated;
    CHECK_FALSE(fs::exists(fixture.root / "cache"));
    CHECK_FALSE(fs::exists(fixture.root / "Source"));
    CHECK_FALSE(fs::exists(fixture.root / "Intermediate"));
}

void use_executable_discovery(project::packaged_runtime_request& request, const packaged_world_fixture& fixture,
                              bool server)
{
    request.boot.host.distribution.kind = project::runtime_distribution_location_kind::executable;
    request.boot.host.distribution.native_path = fixture.root / "bin" / (server ? "Server.exe" : "Game.exe");
    request.boot.host.writable_storage_root = fixture.root / "Saved";
    request.boot.expected.project_guid = "11111111-1111-1111-1111-111111111111";
}

} // namespace

TEST_CASE("D1.8 relocated packaged Windows client loads a real cooked scene through the shared session")
{
    packaged_world_fixture fixture;
    fixture.scene_data = cooked_startup_scene();
    fixture.stage();
    relocate_distribution(fixture);

    std::vector<std::string> events;
    auto request = request_for(fixture);
    use_executable_discovery(request, fixture, false);
    request.registrations.materialize_scene =
        [&](framework::runtime_world& world, const project::packaged_scene_payload& scene)
    {
        events.push_back("scene:materialize");
        const auto error = arc::scene::materialize_cooked_scene(world.entities(), scene.bytes);
        if (!error.empty())
            return project::project_status::failure(
                {.code = project::project_error_code::invalid_scene, .field = "startupScenes", .message = error});
        return project::project_status::success();
    };
    request.registrations.modules = [&](framework::module_registry& modules)
    { modules.emplace<lifecycle_module>(events); };
    request.registrations.services = [&](framework::runtime_service_registry& services)
    { services.emplace<lifecycle_service>(events); };

    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE(started);
    auto session = std::move(started).value();
    CHECK(session->boot().native_distribution_root == fs::canonical(fixture.root));
    REQUIRE(session->runtime());
    REQUIRE(session->runtime()->worlds().size() == 1);
    auto* world = session->runtime()->worlds().find(session->runtime()->worlds().ordered_worlds().front());
    REQUIRE(world);
    CHECK(world->role() == framework::runtime_world_role::client);
    CHECK(world->entities().live_count() == 1);
    const auto entities = world->entities().entities_snapshot();
    REQUIRE(entities.size() == 1);
    CHECK(world->entities().get<arc::scene::name_component>(entities.front()).value == "Relocated Player");
    CHECK(events.front() == "scene:materialize");

    REQUIRE(session->surface_ready(800, 600));
    REQUIRE(session->advance(1.0 / 60.0));
    REQUIRE(session->pause());
    CHECK(session->state() == project::packaged_runtime_state::suspended);
    REQUIRE(session->resume());
    session->request_stop();
    REQUIRE(session->shutdown());
    CHECK(fixture.files.telemetry().mounted_providers == 0);
    CHECK(events.back() == "service:shutdown" || events.back() == "module:shutdown");
}

TEST_CASE("D1.8 relocated dedicated server uses the same manifest and catalog startup without a render surface")
{
    packaged_world_fixture fixture;
    fixture.stage(true);
    relocate_distribution(fixture);

    auto request = request_for(fixture, true);
    use_executable_discovery(request, fixture, true);
    std::uint64_t ticks{};
    request.registrations.initialize_headless_world = [&](framework::runtime_world& world)
    {
        CHECK(world.role() == framework::runtime_world_role::server);
        CHECK_FALSE(world.entities().live_count());
        const bool added = world.systems().add({.name = "d1.8.packaged.server",
                                                .phase = arc::ecs::system_phase::movement,
                                                .execute = [&](arc::ecs::system_context&) { ++ticks; }});
        REQUIRE(added);
        return project::project_status::success();
    };
    const auto result = project::run_packaged_headless_server(
        {.runtime = std::move(request), .maximum_ticks = 4, .sleep_to_clock = false});
    REQUIRE(result);
    CHECK(result.value() == 4);
    CHECK(ticks == 4);
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.8 relocated client fails closed when required cooked package is missing")
{
    packaged_world_fixture fixture;
    fixture.scene_data = cooked_startup_scene();
    fixture.stage();
    relocate_distribution(fixture);
    REQUIRE_FALSE(fixture.chunk_path.empty());
    // chunk_path was recorded before relocation; remove the same relative file
    // from the new distribution, not from the stale original source location.
    const auto chunk = fixture.root / "Content" / fixture.chunk_path.filename();
    REQUIRE(fs::exists(chunk));
    REQUIRE(fs::remove(chunk));

    auto request = request_for(fixture);
    use_executable_discovery(request, fixture, false);
    auto started = project::start_packaged_runtime(std::move(request));
    REQUIRE_FALSE(started);
    CHECK_FALSE(started.error().message.empty());
    CHECK(fixture.files.telemetry().mounted_providers == 0);
}

TEST_CASE("D1.8 relocated client can restart and release mounts repeatedly with no source or SDK")
{
    packaged_world_fixture fixture;
    fixture.scene_data = cooked_startup_scene();
    fixture.stage();
    relocate_distribution(fixture);

    for (int iteration = 0; iteration < 3; ++iteration)
    {
        auto request = request_for(fixture);
        use_executable_discovery(request, fixture, false);
        request.registrations.materialize_scene =
            [](framework::runtime_world& world, const project::packaged_scene_payload& scene)
        {
            auto error = arc::scene::materialize_cooked_scene(world.entities(), scene.bytes);
            if (!error.empty())
                return project::project_status::failure({.code = project::project_error_code::invalid_scene,
                                                         .field = "startupScenes",
                                                         .message = std::move(error)});
            return project::project_status::success();
        };
        auto started = project::start_packaged_runtime(std::move(request));
        REQUIRE(started);
        auto session = std::move(started).value();
        CHECK(fixture.files.telemetry().mounted_providers == 2);
        REQUIRE(session->advance(1.0 / 60.0));
        session->request_stop();
        REQUIRE(session->shutdown());
        REQUIRE(session->shutdown());
        CHECK(fixture.files.telemetry().mounted_providers == 0);
    }
}
