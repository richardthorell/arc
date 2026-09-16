#include <arc/project/input_config.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{

class temporary_input_config
{
public:
    temporary_input_config()
        : root_(std::filesystem::temp_directory_path() /
                ("arc-input-config-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directories(root_);
    }

    ~temporary_input_config()
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    std::filesystem::path write(std::string_view source) const
    {
        const auto path = root_ / "Input.json";
        std::ofstream output(path, std::ios::binary);
        output << source;
        return path;
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
    const auto keyboard = input.connect_device({.type = arc::input::input_device_type::keyboard,
                                                .name = "Keyboard",
                                                .capabilities = {.buttons = true}});
    const auto mouse = input.connect_device({.type = arc::input::input_device_type::mouse,
                                             .name = "Mouse",
                                             .capabilities = {.buttons = true}});
    std::string error;
    REQUIRE(arc::project::apply_input_config(loaded.config, input, 0, &error));

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
