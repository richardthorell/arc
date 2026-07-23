#include <arc/framework/framework.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <memory>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

class recording_application : public arc::application
{
public:
    explicit recording_application(arc::application_config config = {})
        : config_(std::move(config))
    {
    }

    arc::application_config configure() const override
    {
        return config_;
    }

    void on_start() override
    {
        calls.push_back("start");
    }

    void on_update(const arc::frame_time& time) override
    {
        calls.push_back("update");
        last_time = time;
    }

    void on_event(const arc::event& value) override
    {
        calls.push_back("event");
        last_event = value;
    }

    void on_shutdown() override
    {
        calls.push_back("shutdown");
    }

    std::vector<std::string> calls;
    arc::frame_time last_time{};
    arc::event last_event{};

private:
    arc::application_config config_;
};

class recording_module final : public arc::module
{
public:
    recording_module(std::string module_name, std::vector<std::string>* calls, std::vector<std::string> deps = {})
        : module_name_(std::move(module_name))
        , calls_(calls)
        , deps_(std::move(deps))
    {
    }

    std::string_view name() const override
    {
        return module_name_;
    }

    std::vector<std::string> dependencies() const override
    {
        return deps_;
    }

    void on_start(arc::module_context& context) override
    {
        REQUIRE(context.jobs().run_inline() == false);
        calls_->push_back(module_name_ + ":start");
    }

    void on_update(arc::module_context&, const arc::frame_time&) override
    {
        calls_->push_back(module_name_ + ":update");
    }

    void on_event(arc::module_context&, const arc::event&) override
    {
        calls_->push_back(module_name_ + ":event");
    }

    void on_shutdown(arc::module_context&) override
    {
        calls_->push_back(module_name_ + ":shutdown");
    }

private:
    std::string module_name_;
    std::vector<std::string>* calls_{};
    std::vector<std::string> deps_;
};

class modular_application final : public arc::application
{
public:
    void register_modules(arc::module_registry& registry) override
    {
        calls.push_back("register");
        registry.add(std::make_unique<recording_module>("graphics", &calls));
        registry.add(std::make_unique<recording_module>("physics", &calls, std::vector<std::string>{ "graphics" }));
    }

    void on_start() override
    {
        calls.push_back("app:start");
    }

    void on_update(const arc::frame_time&) override
    {
        calls.push_back("app:update");
    }

    void on_event(const arc::event&) override
    {
        calls.push_back("app:event");
    }

    void on_shutdown() override
    {
        calls.push_back("app:shutdown");
    }

    std::vector<std::string> calls;
};

struct counter_component
{
    int value{};
};

class headless_test_application final : public arc::application
{
public:
    arc::application_config configure() const override
    {
        arc::application_config result{};
        result.simulation.fixed_tick_rate = 60.0;
        return result;
    }
};

class failing_headless_application final : public arc::application
{
public:
    void register_worlds(arc::runtime_world_manager& worlds) override
    {
        arc::runtime_world& world = worlds.create({
            .name = "Failing Headless World",
            .role = arc::runtime_world_role::server,
            .install_placeholder_systems = false,
            .presentation_enabled = false
        });
        world.systems().add({
            .name = "Headless failure",
            .phase = arc::ecs::system_phase::movement,
            .execute = [](arc::ecs::system_context&) {
                throw std::runtime_error("intentional headless fault");
            }
        });
    }
};

class lifecycle_service final : public arc::runtime_service
{
public:
    lifecycle_service(
        arc::runtime_service_id service_id,
        std::string service_name,
        std::vector<std::string>* calls,
        std::vector<arc::runtime_service_id> dependencies = {})
        : id_(service_id)
        , name_(std::move(service_name))
        , calls_(calls)
        , dependencies_(std::move(dependencies))
    {
    }

    arc::runtime_service_id id() const noexcept override { return id_; }
    std::string_view name() const noexcept override { return name_; }
    std::vector<arc::runtime_service_id> dependencies() const override { return dependencies_; }
    void on_start(arc::runtime_service_context&) override { calls_->push_back(name_ + ":start"); }
    void on_shutdown(arc::runtime_service_context&) noexcept override { calls_->push_back(name_ + ":stop"); }

private:
    arc::runtime_service_id id_{};
    std::string name_;
    std::vector<std::string>* calls_{};
    std::vector<arc::runtime_service_id> dependencies_;
};

class deterministic_service final : public arc::runtime_service
{
public:
    static constexpr arc::runtime_service_id service_id =
        arc::make_runtime_service_id("tests.deterministic");

    arc::runtime_service_id id() const noexcept override { return service_id; }
    std::string_view name() const noexcept override { return "deterministic"; }
    bool has_deterministic_state() const noexcept override { return true; }

    bool capture_deterministic_state(
        std::uint64_t,
        std::vector<std::byte>& bytes,
        std::string&) const override
    {
        bytes = { static_cast<std::byte>(value) };
        return true;
    }

    bool validate_deterministic_state(
        std::uint64_t,
        std::uint32_t version,
        const std::vector<std::byte>& bytes,
        std::string& error) const override
    {
        if (version == 1 && bytes.size() == 1)
            return true;
        error = "invalid deterministic test service snapshot";
        return false;
    }

    void restore_deterministic_state(
        std::uint64_t,
        std::uint32_t,
        const std::vector<std::byte>& bytes) noexcept override
    {
        value = std::to_integer<int>(bytes.front());
    }

    int value{};
};

} // namespace

TEST_CASE("runtime normalizes application config")
{
    arc::application_config config{};
    config.title.clear();
    config.initial_width = 0;
    config.initial_height = 0;
    config.resizable = false;
    config.visible = false;

    const auto normalized = arc::runtime::normalize_config(config);
    REQUIRE(normalized.title == "ARC Application");
    REQUIRE(normalized.initial_width == 1280);
    REQUIRE(normalized.initial_height == 720);
    REQUIRE_FALSE(normalized.resizable);
    REQUIRE_FALSE(normalized.visible);
}

TEST_CASE("runtime calls lifecycle hooks in order")
{
    recording_application app;
    arc::runtime runtime(app);

    REQUIRE_FALSE(runtime.started());
    REQUIRE_FALSE(runtime.running());

    runtime.start();
    REQUIRE(runtime.started());
    REQUIRE(runtime.running());

    runtime.tick();
    runtime.shutdown();

    REQUIRE(app.calls.size() == 3);
    REQUIRE(app.calls[0] == "start");
    REQUIRE(app.calls[1] == "update");
    REQUIRE(app.calls[2] == "shutdown");
}

TEST_CASE("runtime dispatches events and close requests stop")
{
    recording_application app;
    arc::runtime runtime(app);

    runtime.start();
    arc::event resize{};
    resize.type = arc::event_type::resized;
    resize.width = 1920;
    resize.height = 1080;
    runtime.dispatch(resize);

    REQUIRE(runtime.running());
    REQUIRE(app.last_event.type == arc::event_type::resized);
    REQUIRE(app.last_event.width == 1920);
    REQUIRE(app.last_event.height == 1080);

    arc::event close{};
    close.type = arc::event_type::close_requested;
    runtime.dispatch(close);

    REQUIRE_FALSE(runtime.running());
    REQUIRE(app.last_event.type == arc::event_type::close_requested);
    runtime.shutdown();
}

TEST_CASE("runtime uses application provided config")
{
    arc::application_config config{};
    config.title = "Test App";
    config.initial_width = 800;
    config.initial_height = 600;

    recording_application app(config);
    arc::runtime runtime(app);

    REQUIRE(runtime.config().title == "Test App");
    REQUIRE(runtime.config().initial_width == 800);
    REQUIRE(runtime.config().initial_height == 600);
}

TEST_CASE("runtime starts, updates, dispatches, and shuts modules in order")
{
    modular_application app;
    arc::runtime runtime(app);

    runtime.start();
    runtime.tick();

    arc::event event{};
    event.type = arc::event_type::resized;
    runtime.dispatch(event);
    runtime.shutdown();

    REQUIRE(app.calls == std::vector<std::string>{
        "register",
        "graphics:start",
        "physics:start",
        "app:start",
        "graphics:update",
        "physics:update",
        "app:update",
        "graphics:event",
        "physics:event",
        "app:event",
        "app:shutdown",
        "physics:shutdown",
        "graphics:shutdown" });
}

TEST_CASE("module manager rejects dependency problems")
{
    arc::job_system jobs(arc::job_system::single_threaded_config());
    arc::module_context context(jobs, arc::default_logger(), arc::default_tracked_memory_resource());

    SECTION("unknown dependency")
    {
        arc::module_manager manager;
        std::vector<std::string> calls;
        manager.registry().add(std::make_unique<recording_module>("physics", &calls, std::vector<std::string>{ "graphics" }));

        REQUIRE_THROWS_AS(manager.start(context), std::invalid_argument);
    }

    SECTION("duplicate names")
    {
        arc::module_manager manager;
        std::vector<std::string> calls;
        manager.registry().add(std::make_unique<recording_module>("graphics", &calls));
        manager.registry().add(std::make_unique<recording_module>("graphics", &calls));

        REQUIRE_THROWS_AS(manager.start(context), std::invalid_argument);
    }

    SECTION("dependency cycles")
    {
        arc::module_manager manager;
        std::vector<std::string> calls;
        manager.registry().add(std::make_unique<recording_module>("a", &calls, std::vector<std::string>{ "b" }));
        manager.registry().add(std::make_unique<recording_module>("b", &calls, std::vector<std::string>{ "a" }));

        REQUIRE_THROWS_AS(manager.start(context), std::invalid_argument);
    }
}

TEST_CASE("runtime advances a deterministic fixed-step clock")
{
    recording_application app;
    arc::application_config config{};
    config.simulation.fixed_tick_rate = 60.0;
    config.simulation.maximum_catch_up_ticks = 8;
    arc::runtime host(app, config);

    const arc::frame_time first = host.advance(1.0 / 120.0);
    REQUIRE(first.completed_ticks == 0);
    REQUIRE(first.interpolation_alpha == Catch::Approx(0.5));

    const arc::frame_time second = host.advance(1.0 / 120.0);
    REQUIRE(second.completed_ticks == 1);
    REQUIRE(second.last_completed_tick.value == 1);
    REQUIRE(second.interpolation_alpha == Catch::Approx(0.0).margin(1e-8));
    host.shutdown();
}

TEST_CASE("runtime pause and single-step preserve fixed delta")
{
    recording_application app;
    arc::runtime host(app);
    host.start();
    host.pause();
    REQUIRE(host.paused());

    REQUIRE(host.advance(1.0).completed_ticks == 0);
    REQUIRE(host.step());
    REQUIRE(host.advance(0.0).completed_ticks == 1);
    REQUIRE(host.current_tick().id.value == 1);
    REQUIRE(host.paused());

    REQUIRE(host.set_time_scale(2.0));
    host.resume();
    REQUIRE(host.advance(1.0 / 120.0).completed_ticks == 1);
    host.shutdown();
}

TEST_CASE("runtime executes server worlds before client and preview worlds")
{
    recording_application app;
    arc::runtime host(app);
    std::vector<arc::runtime_world_role> order;

    auto add_world = [&](arc::runtime_world_role role, std::string name) {
        arc::runtime_world& world = host.worlds().create({
            .name = std::move(name),
            .role = role,
            .install_placeholder_systems = false
        });
        REQUIRE(world.systems().add({
            .name = std::string(world.name()) + " recorder",
            .phase = arc::ecs::system_phase::movement,
            .execute = [&order](arc::ecs::system_context& context) {
                order.push_back(context.world_role());
            }
        }));
    };

    add_world(arc::runtime_world_role::client, "Client");
    add_world(arc::runtime_world_role::editor_preview, "Preview");
    add_world(arc::runtime_world_role::server, "Server");
    host.start();
    host.advance(1.0 / 60.0);

    REQUIRE(order == std::vector<arc::runtime_world_role>{
        arc::runtime_world_role::server,
        arc::runtime_world_role::client,
        arc::runtime_world_role::editor_preview });
    host.shutdown();
}

TEST_CASE("systems flush structural commands at phase boundaries")
{
    recording_application app;
    arc::runtime host(app);
    arc::runtime_world& world = host.worlds().create({
        .name = "Structural world",
        .role = arc::runtime_world_role::client,
        .install_placeholder_systems = false
    });
    const arc::ecs::entity entity = world.entities().create();
    bool observed{};

    REQUIRE(world.systems().add({
        .name = "Add counter",
        .phase = arc::ecs::system_phase::gameplay_commands,
        .execute = [entity](arc::ecs::system_context& context) {
            context.commands().add<counter_component>(entity, counter_component{ 42 });
        }
    }));
    REQUIRE(world.systems().add({
        .name = "Read counter",
        .phase = arc::ecs::system_phase::movement,
        .components = { arc::ecs::reads<counter_component>() },
        .execute = [entity, &observed](arc::ecs::system_context& context) {
            const counter_component* value = context.read<counter_component>(entity);
            observed = value && value->value == 42;
        }
    }));

    host.start();
    host.advance(1.0 / 60.0);
    REQUIRE(observed);
    host.shutdown();
}

TEST_CASE("runtime executes fixed phases in order and presentation once per frame")
{
    recording_application app;
    arc::runtime host(app);
    arc::runtime_world& world = host.worlds().create({
        .name = "Phase world",
        .role = arc::runtime_world_role::client,
        .install_placeholder_systems = false
    });
    std::vector<arc::ecs::system_phase> phases;
    const auto record = [&world, &phases](arc::ecs::system_phase phase, const char* name) {
        REQUIRE(world.systems().add({
            .name = name,
            .phase = phase,
            .execute = [&phases, phase](arc::ecs::system_context&) { phases.push_back(phase); }
        }));
    };
    record(arc::ecs::system_phase::input, "Input");
    record(arc::ecs::system_phase::network_receive, "Network");
    record(arc::ecs::system_phase::gameplay_commands, "Commands");
    record(arc::ecs::system_phase::movement, "Movement");
    record(arc::ecs::system_phase::physics, "Physics");
    record(arc::ecs::system_phase::abilities, "Abilities");
    record(arc::ecs::system_phase::ai, "AI");
    record(arc::ecs::system_phase::replication, "Replication");
    record(arc::ecs::system_phase::presentation_extraction, "Presentation");

    host.start();
    REQUIRE(host.advance(0.0).completed_ticks == 0);
    REQUIRE(phases == std::vector<arc::ecs::system_phase>{
        arc::ecs::system_phase::presentation_extraction });

    phases.clear();
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    REQUIRE(phases == std::vector<arc::ecs::system_phase>{
        arc::ecs::system_phase::input,
        arc::ecs::system_phase::network_receive,
        arc::ecs::system_phase::gameplay_commands,
        arc::ecs::system_phase::movement,
        arc::ecs::system_phase::physics,
        arc::ecs::system_phase::abilities,
        arc::ecs::system_phase::ai,
        arc::ecs::system_phase::replication,
        arc::ecs::system_phase::presentation_extraction });
    host.shutdown();
}

TEST_CASE("input commands are sampled once and retained until a fixed tick")
{
    recording_application app;
    arc::runtime host(app);
    arc::runtime_world& world = host.worlds().create({
        .name = "Input world",
        .install_placeholder_systems = false
    });
    std::vector<arc::simulation_input_command> observed;
    std::uint64_t observed_revision{};
    REQUIRE(world.systems().add({
        .name = "Input consumer",
        .phase = arc::ecs::system_phase::input,
        .execute = [&observed, &observed_revision](arc::ecs::system_context& context) {
            observed.assign(context.input().commands.begin(), context.input().commands.end());
            observed_revision = context.input().revision;
        }
    }));

    host.start();
    host.dispatch({
        .type = arc::event_type::key_down,
        .key_code = 87,
        .modifiers = 2,
        .repeat = false
    });
    REQUIRE(host.advance(0.0).completed_ticks == 0);
    REQUIRE(observed.empty());
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    REQUIRE(observed.size() == 1);
    REQUIRE(observed.front().kind == arc::simulation_input_kind::key);
    REQUIRE(observed.front().action == arc::simulation_input_action::pressed);
    REQUIRE(observed.front().code == 87);
    REQUIRE(observed.front().modifiers == 2);
    REQUIRE(observed_revision == 1);

    observed.clear();
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    REQUIRE(observed.empty());
    REQUIRE(observed_revision == 1);
    host.shutdown();
}

TEST_CASE("a failed world does not prevent later interactive worlds from ticking")
{
    recording_application app;
    arc::runtime host(app);
    arc::runtime_world& server = host.worlds().create({
        .name = "Faulted server",
        .role = arc::runtime_world_role::server,
        .install_placeholder_systems = false
    });
    arc::runtime_world& client = host.worlds().create({
        .name = "Responsive client",
        .role = arc::runtime_world_role::client,
        .install_placeholder_systems = false
    });
    bool client_ran{};
    REQUIRE(server.systems().add({
        .name = "Failing movement",
        .phase = arc::ecs::system_phase::movement,
        .execute = [](arc::ecs::system_context&) { throw std::runtime_error("test fault"); }
    }));
    REQUIRE(client.systems().add({
        .name = "Client movement",
        .phase = arc::ecs::system_phase::movement,
        .execute = [&client_ran](arc::ecs::system_context&) { client_ran = true; }
    }));

    host.start();
    host.advance(1.0 / 60.0);
    REQUIRE(server.state() == arc::runtime_world_state::faulted);
    REQUIRE(server.fault_message().find("test fault") != std::string::npos);
    REQUIRE(client.state() == arc::runtime_world_state::running);
    REQUIRE(client_ran);
    REQUIRE(host.running());
    host.shutdown();
}

TEST_CASE("deterministic random streams are stable and independent")
{
    constexpr auto gameplay = arc::ecs::make_random_stream_id("tests.gameplay");
    constexpr auto effects = arc::ecs::make_random_stream_id("tests.effects");
    auto first = arc::ecs::make_random_stream(12, 3, { 8 }, gameplay, 44);
    auto second = arc::ecs::make_random_stream(12, 3, { 8 }, gameplay, 44);
    auto other = arc::ecs::make_random_stream(12, 3, { 8 }, effects, 44);

    for (int index = 0; index < 16; ++index)
        REQUIRE(first.next_u32() == second.next_u32());
    REQUIRE(first.next_u32() != other.next_u32());
}

TEST_CASE("runtime world snapshots restore state atomically")
{
    recording_application app;
    arc::application_config config{};
    config.simulation.snapshot_budget_bytes = 1024u * 1024u;
    arc::runtime host(app, config);
    arc::runtime_world& world = host.worlds().create({
        .name = "Snapshot world",
        .install_placeholder_systems = false
    });
    const arc::ecs::entity entity = world.entities().create();
    world.entities().emplace<counter_component>(entity, counter_component{ 7 });
    world.entities().prepare_query<counter_component>();
    host.start();

    const arc::world_snapshot_result captured = host.capture_snapshot(world.id(), "checkpoint");
    REQUIRE(captured.succeeded);
    world.entities().get<counter_component>(entity).value = 99;

    const std::uint64_t prior_epoch = world.epoch();
    const arc::world_snapshot_result restored = host.restore_snapshot(captured.metadata.id);
    REQUIRE(restored.succeeded);
    REQUIRE(world.epoch() == prior_epoch + 1);
    REQUIRE(std::as_const(world.entities()).get<counter_component>(entity).value == 7);
    REQUIRE_FALSE(world.entities().query<arc::ecs::query_read<counter_component>>().empty());
    host.shutdown();
}

TEST_CASE("runtime snapshots include registered deterministic service state")
{
    class snapshot_application final : public recording_application
    {
    public:
        using recording_application::recording_application;
        void register_services(arc::runtime_service_registry& services) override
        {
            state = &services.emplace<deterministic_service>();
        }
        deterministic_service* state{};
    } app;

    arc::application_config config{};
    config.simulation.snapshot_budget_bytes = 1024u * 1024u;
    arc::runtime host(app, config);
    arc::runtime_world& world = host.worlds().create({
        .name = "Service snapshot world",
        .install_placeholder_systems = false
    });
    host.start();
    REQUIRE(app.state != nullptr);
    app.state->value = 42;

    const arc::world_snapshot_result captured = host.capture_snapshot(world.id());
    REQUIRE(captured.succeeded);
    app.state->value = 7;
    REQUIRE(host.restore_snapshot(captured.metadata.id).succeeded);
    REQUIRE(app.state->value == 42);
    host.shutdown();
}

TEST_CASE("runtime snapshots are explicitly disabled by a zero budget")
{
    recording_application app;
    arc::runtime host(app);
    arc::runtime_world& world = host.worlds().create({
        .name = "No snapshots",
        .install_placeholder_systems = false
    });
    host.start();

    const arc::world_snapshot_result captured = host.capture_snapshot(world.id(), "disabled");
    REQUIRE_FALSE(captured.succeeded);
    REQUIRE(captured.error == "runtime snapshots are disabled");
    REQUIRE(host.worlds().snapshots().empty());
    host.shutdown();
}

TEST_CASE("headless simulation preserves catch-up debt")
{
    arc::application_config config{};
    config.simulation.headless = true;
    config.simulation.presentation_enabled = false;
    config.simulation.maximum_catch_up_ticks = 8;
    config.simulation.overrun_policy = arc::simulation_overrun_policy::preserve_debt;
    recording_application app(config);
    arc::runtime host(app, config);

    REQUIRE(host.advance(1.0).completed_ticks == 8);
    REQUIRE(host.discarded_ticks() == 0);
    for (int frame = 0; frame < 7; ++frame)
        host.advance(0.0);
    REQUIRE(host.current_tick().id.value == 60);
    REQUIRE(host.discarded_ticks() == 0);
    host.shutdown();
}

TEST_CASE("runtime services start in dependency order and stop in reverse")
{
    constexpr arc::runtime_service_id storage = arc::make_runtime_service_id("tests.storage");
    constexpr arc::runtime_service_id gameplay = arc::make_runtime_service_id("tests.gameplay");
    std::vector<std::string> calls;
    arc::runtime_service_registry services;
    REQUIRE(services.add(std::make_unique<lifecycle_service>(storage, "storage", &calls)));
    REQUIRE(services.add(std::make_unique<lifecycle_service>(
        gameplay,
        "gameplay",
        &calls,
        std::vector<arc::runtime_service_id>{ storage })));

    services.start();
    services.shutdown();
    REQUIRE(calls == std::vector<std::string>{
        "storage:start", "gameplay:start", "gameplay:stop", "storage:stop" });
}

TEST_CASE("headless runtime executes finite renderer-free runs")
{
    headless_test_application app;
    const arc::headless_runtime_result result = arc::run_headless(app, {
        .maximum_ticks = 5,
        .sleep_to_clock = false
    });
    REQUIRE(result.succeeded);
    REQUIRE(result.completed_ticks == 5);
}

TEST_CASE("headless runtime reports a world fault as process failure")
{
    failing_headless_application app;
    const arc::headless_runtime_result result = arc::run_headless(app, {
        .maximum_ticks = 5,
        .sleep_to_clock = false
    });
    REQUIRE_FALSE(result.succeeded);
    REQUIRE(result.completed_ticks == 1);
    REQUIRE(result.error.find("intentional headless fault") != std::string::npos);
}
