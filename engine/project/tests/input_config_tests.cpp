#include <arc/input/gamepad.h>
#include <arc/input/touch.h>
#include <arc/project/input_config.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace
{

class temporary_input_config
{
public:
    temporary_input_config()
        : root_(std::filesystem::temp_directory_path() /
                ("arc-input-config-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directories(root_);
    }

    ~temporary_input_config()
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    [[nodiscard]] std::filesystem::path path(std::string_view name = "Input.json") const
    {
        return root_ / std::string(name);
    }

    std::filesystem::path write(std::string_view source, std::string_view name = "Input.json") const
    {
        const auto output_path = path(name);
        std::ofstream output(output_path, std::ios::binary);
        output << source;
        return output_path;
    }

    [[nodiscard]] std::string read(const std::filesystem::path& source) const
    {
        std::ifstream input(source, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

private:
    std::filesystem::path root_;
};

} // namespace

TEST_CASE("project input config drives ARC semantic action contexts")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [
        {
          "name": "Gameplay",
          "priority": 0,
          "actions": [
            {"name": "Jump", "bindings": [{"device": "keyboard", "control": "Space"}]},
            {"name": "Fire", "bindings": [{"device": "mouse", "control": "Left"}]}
          ]
        },
        {
          "name": "Menu",
          "priority": 100,
          "enabled": true,
          "actions": [
            {"name": "Jump", "bindings": [{"device": "keyboard", "control": "Enter"}]}
          ]
        }
      ]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    REQUIRE(loaded.succeeded);
    const std::vector<std::string> expected_actions{"Jump", "Fire"};
    CHECK(arc::project::input_action_names(loaded.config) == expected_actions);

    arc::input::input_system input;
    const auto keyboard = input.connect_device(
        {.type = arc::input::input_device_type::keyboard, .name = "Keyboard", .capabilities = {.buttons = true}});
    const auto mouse = input.connect_device(
        {.type = arc::input::input_device_type::mouse, .name = "Mouse", .capabilities = {.buttons = true}});
    REQUIRE(arc::project::apply_input_config(loaded.config, input, 0).succeeded);

    auto& player = input.player(0);
    input.begin_frame();
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::space), true));
    CHECK_FALSE(player.pressed("Jump"));

    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::enter), true));
    CHECK(player.pressed("Jump"));

    REQUIRE(input.submit_button(mouse, arc::input::make_mouse_button_control(arc::input::mouse_button::left), true));
    CHECK(player.pressed("Fire"));

    REQUIRE(player.set_context_enabled("Menu", false));
    CHECK(player.down("Jump"));
}

TEST_CASE("project input config drives scalar and 2D axes")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "axes": [{
          "name": "MoveForward",
          "bindings": [
            {"device": "keyboard", "control": "W", "contribution": 1.0},
            {"device": "keyboard", "control": "S", "contribution": -1.0}
          ]
        }, {
          "name": "LookX",
          "bindings": [{"device": "mouse", "control": "DeltaX", "processors": [{"type": "scale", "value": 0.5}] }]
        }],
        "axes2d": [{
          "name": "Move",
          "bindings": [
            {"device": "keyboard", "control": "W", "contribution": [0.0, 1.0]},
            {"device": "keyboard", "control": "D", "contribution": [1.0, 0.0]}
          ]
        }]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    REQUIRE(loaded.succeeded);
    REQUIRE(loaded.config.contexts.size() == 1);
    CHECK(loaded.config.contexts[0].axes.size() == 2);
    CHECK(loaded.config.contexts[0].axes2d.size() == 1);

    arc::input::input_system input;
    const auto keyboard = input.connect_device(
        {.type = arc::input::input_device_type::keyboard, .name = "Keyboard", .capabilities = {.buttons = true}});
    const auto mouse = input.connect_device({.type = arc::input::input_device_type::mouse,
                                             .name = "Mouse",
                                             .capabilities = {.axes = true, .pointer = true}});
    REQUIRE(input.assign_device(0, keyboard));
    REQUIRE(input.assign_device(0, mouse));
    REQUIRE(arc::project::apply_input_config(loaded.config, input, 0).succeeded);

    auto& player = input.player(0);
    input.begin_frame();
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::w), true));
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::d), true));
    REQUIRE(input.submit_axis(mouse, arc::input::make_mouse_axis_control(arc::input::mouse_axis::delta_x), 0.8f));

    CHECK(player.axis("MoveForward") == 1.0f);
    CHECK(player.axis("LookX") == 0.4f);
    const auto move = player.axis2d("Move");
    CHECK(move[0] == 1.0f);
    CHECK(move[1] == 1.0f);
}

TEST_CASE("project input config rejects malformed axis contributions")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "axes2d": [{"name": "Move", "bindings": [{"device": "keyboard", "control": "W", "contribution": [1.0]}]}]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.error.find("contribution") != std::string::npos);
}

TEST_CASE("project input config rejects unknown physical controls")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "actions": [{"name": "Jump", "bindings": [{"device": "keyboard", "control": "HyperSpace"}]}]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.error.find("HyperSpace") != std::string::npos);
}

TEST_CASE("project input config maps motion sensor axes")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "axes": [
          {"name": "Turn", "bindings": [{"device": "motion", "control": "GyroZ"}]},
          {"name": "Tilt", "bindings": [{"device": "sensor", "control": "AccelerometerX"}]}
        ]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    REQUIRE(loaded.succeeded);
    REQUIRE(loaded.config.contexts.size() == 1);
    REQUIRE(loaded.config.contexts[0].axes.size() == 2);
    CHECK(loaded.config.contexts[0].axes[0].bindings[0].binding.device ==
          arc::input::input_device_type::motion_controller);
    CHECK(loaded.config.contexts[0].axes[0].bindings[0].binding.control ==
          arc::input::make_sensor_axis_control(arc::input::sensor_axis::gyroscope_z));
    CHECK(loaded.config.contexts[0].axes[1].bindings[0].binding.control ==
          arc::input::make_sensor_axis_control(arc::input::sensor_axis::accelerometer_x));
}

TEST_CASE("project input config rejects unknown motion sensor controls")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "axes": [{"name": "Tilt", "bindings": [{"device": "motion", "control": "CompassX"}]}]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.error.find("CompassX") != std::string::npos);
}

TEST_CASE("project input config drives gamepad sensors and touch controls")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "actions": [
          {"name": "Accept", "bindings": [{"device": "gamepad", "control": "South"}]},
          {"name": "Touchpad", "bindings": [{"device": "controller", "control": "TouchPrimaryDown"}]},
          {"name": "ScreenTouch", "bindings": [{"device": "touch", "control": "PrimaryDown"}]}
        ],
        "axes": [
          {"name": "Throttle", "bindings": [{"device": "gamepad", "control": "LeftTrigger"}]},
          {"name": "Gyro", "bindings": [{"device": "gamepad", "control": "GyroZ"}]},
          {"name": "TouchPressure", "bindings": [{"device": "gamepad", "control": "TouchPrimaryPressure"}]}
        ],
        "axes2d": [
          {"name": "Move", "bindings": [
            {"device": "gamepad", "control": "LeftX", "contribution": [1.0, 0.0]},
            {"device": "gamepad", "control": "LeftY", "contribution": [0.0, 1.0]}
          ]},
          {"name": "ScreenPosition", "bindings": [
            {"device": "touchscreen", "control": "PrimaryX", "contribution": [1.0, 0.0]},
            {"device": "touchscreen", "control": "PrimaryY", "contribution": [0.0, 1.0]}
          ]}
        ]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    REQUIRE(loaded.succeeded);

    arc::input::input_system input;
    const auto gamepad = input.connect_device({.type = arc::input::input_device_type::gamepad,
                                               .name = "Gamepad",
                                               .capabilities = {.buttons = true,
                                                                .axes = true,
                                                                .gyroscope = true,
                                                                .touchpad = true}});
    const auto touch = input.connect_device({.type = arc::input::input_device_type::touch,
                                             .name = "Touchscreen",
                                             .capabilities = {.pointer = true, .touchpad = true}});
    REQUIRE(input.assign_device(0, gamepad));
    REQUIRE(input.assign_device(0, touch));
    REQUIRE(arc::project::apply_input_config(loaded.config, input, 0).succeeded);

    auto& player = input.player(0);
    input.begin_frame();
    REQUIRE(input.submit_button(gamepad,
                                arc::input::make_gamepad_button_control(arc::input::gamepad_button::south), true));
    REQUIRE(input.submit_axis(gamepad, arc::input::make_gamepad_axis_control(arc::input::gamepad_axis::left_x),
                              0.75f));
    REQUIRE(input.submit_axis(gamepad, arc::input::make_gamepad_axis_control(arc::input::gamepad_axis::left_y),
                              -0.25f));
    REQUIRE(input.submit_axis(gamepad,
                              arc::input::make_gamepad_axis_control(arc::input::gamepad_axis::left_trigger), 0.6f));
    REQUIRE(input.submit_axis(gamepad, arc::input::make_sensor_axis_control(arc::input::sensor_axis::gyroscope_z),
                              1.5f));
    REQUIRE(input.submit_touch_contacts(gamepad, {{.id = 4,
                                                   .surface = 0,
                                                   .position = {0.25f, 0.75f},
                                                   .pressure = 0.8f,
                                                   .pressure_available = true}}));
    REQUIRE(input.submit_touch_contacts(touch, {{.id = 2, .surface = 0, .position = {0.4f, 0.9f}}}));

    CHECK(player.pressed("Accept"));
    CHECK(player.pressed("Touchpad"));
    CHECK(player.pressed("ScreenTouch"));
    CHECK(player.axis("Throttle") == 0.6f);
    CHECK(player.axis("Gyro") == 1.5f);
    CHECK(player.axis("TouchPressure") == 0.8f);
    const auto move = player.axis2d("Move");
    CHECK(move[0] == 0.75f);
    CHECK(move[1] == -0.25f);
    const auto screen = player.axis2d("ScreenPosition");
    CHECK(screen[0] == 0.4f);
    CHECK(screen[1] == 0.9f);
}

TEST_CASE("project input config canonical save is deterministic and migrates legacy formatVersion")
{
    temporary_input_config temporary;
    const auto legacy_path = temporary.write(R"json({
      "formatVersion": 1,
      "contexts": [{
        "name": "Gameplay",
        "priority": 5,
        "enabled": true,
        "actions": [{
          "name": "Accept",
          "bindings": [{"device": "controller", "control": "south"}]
        }],
        "axes2d": [{
          "name": "Move",
          "bindings": [
            {"device": "gamepad", "control": "left_stick_x", "contribution": [1.0, 0.0]},
            {"device": "gamepad", "control": "left_stick_y", "contribution": [0.0, 1.0]}
          ]
        }]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(legacy_path);
    REQUIRE(loaded.succeeded);
    const auto first_path = temporary.path("Canonical.json");
    REQUIRE(arc::project::save_input_config(loaded.config, first_path).succeeded);
    const std::string first = temporary.read(first_path);
    CHECK(first.find("\"version\": 1") != std::string::npos);
    CHECK(first.find("formatVersion") == std::string::npos);
    CHECK(first.find("\"device\": \"gamepad\"") != std::string::npos);
    CHECK(first.find("\"control\": \"LeftX\"") != std::string::npos);

    const auto reloaded = arc::project::load_input_config(first_path);
    REQUIRE(reloaded.succeeded);
    const auto second_path = temporary.path("CanonicalAgain.json");
    REQUIRE(arc::project::save_input_config(reloaded.config, second_path).succeeded);
    CHECK(temporary.read(second_path) == first);
}

TEST_CASE("project input config rejects conflicting version fields")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({"version":1,"formatVersion":2,"contexts":[]})json");
    const auto loaded = arc::project::load_input_config(path);
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.error.find("disagree") != std::string::npos);
}

TEST_CASE("project input config rejects malformed processor values")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "axes": [{
          "name": "Look",
          "bindings": [{
            "device": "mouse",
            "control": "DeltaX",
            "processors": [{"type": "scale", "value": "fast"}]
          }]
        }]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.error.find("finite number") != std::string::npos);
}

TEST_CASE("project input config rejects duplicate semantic identifiers")
{
    temporary_input_config temporary;
    const auto path = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "actions": [
          {"name": "Jump", "bindings": [{"device": "keyboard", "control": "Space"}]},
          {"name": "Jump", "bindings": [{"device": "gamepad", "control": "South"}]}
        ]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(path);
    CHECK_FALSE(loaded.succeeded);
    CHECK(loaded.error.find("duplicate input action") != std::string::npos);
}
