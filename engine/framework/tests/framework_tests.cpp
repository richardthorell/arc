#include <arc/framework/framework.h>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

class recording_application final : public arc::application
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
