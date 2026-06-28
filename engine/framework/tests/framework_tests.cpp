#include <arc/framework.h>

#include <catch2/catch_test_macros.hpp>

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
