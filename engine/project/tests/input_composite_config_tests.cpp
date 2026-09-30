#include <arc/project/input_config.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace
{

class temporary_input_config
{
public:
    temporary_input_config()
        : root_(std::filesystem::temp_directory_path() /
                ("arc-input-composites-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directories(root_);
    }

    ~temporary_input_config()
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    [[nodiscard]] std::filesystem::path path(std::string_view name) const
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

private:
    std::filesystem::path root_;
};

} // namespace

TEST_CASE("input config persists modifier chords and vector composites")
{
    temporary_input_config temporary;
    const auto source = temporary.write(R"json({
      "version": 1,
      "contexts": [{
        "name": "Gameplay",
        "actions": [{
          "name": "Save",
          "bindings": [{
            "device": "keyboard",
            "control": "S",
            "modifiers": [{"device": "keyboard", "control": "LeftControl"}],
            "compositeProcessors": [{"type": "scale", "value": 1.0}]
          }]
        }],
        "axes2d": [{
          "name": "Move",
          "bindings": [
            {"device": "keyboard", "control": "W", "contribution": [0.0, 1.0]},
            {"device": "keyboard", "control": "S", "contribution": [0.0, -1.0]},
            {"device": "keyboard", "control": "A", "contribution": [-1.0, 0.0]},
            {"device": "keyboard", "control": "D", "contribution": [1.0, 0.0]}
          ]
        }]
      }]
    })json");

    const auto loaded = arc::project::load_input_config(source);
    REQUIRE(loaded.succeeded);
    REQUIRE(loaded.config.contexts.size() == 1);
    REQUIRE(loaded.config.contexts[0].actions.size() == 1);
    REQUIRE(loaded.config.contexts[0].actions[0].bindings.size() == 1);

    const auto& save_binding = loaded.config.contexts[0].actions[0].bindings[0];
    REQUIRE(save_binding.modifiers.size() == 1);
    CHECK(save_binding.modifiers[0].control == arc::input::make_key_control(arc::input::key::left_control));
    REQUIRE(save_binding.composite_processors.size() == 1);
    CHECK(save_binding.composite_processors[0].type == arc::input::input_processor_type::scale);
    REQUIRE(loaded.config.contexts[0].axes2d.size() == 1);
    CHECK(loaded.config.contexts[0].axes2d[0].bindings.size() == 4);

    arc::input::input_system input;
    const auto keyboard = input.connect_device({.type = arc::input::input_device_type::keyboard,
                                                .name = "Keyboard",
                                                .capabilities = {.buttons = true}});
    REQUIRE(arc::project::apply_input_config(loaded.config, input).succeeded);
    auto& player = input.player(0);

    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::s), true));
    CHECK_FALSE(player.down("Save"));
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::left_control), true));
    CHECK(player.down("Save"));

    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::w), true));
    const auto opposed = player.axis2d("Move");
    CHECK(opposed[0] == 0.0f);
    CHECK(opposed[1] == 0.0f);

    const auto saved_path = temporary.path("Saved.json");
    REQUIRE(arc::project::save_input_config(loaded.config, saved_path).succeeded);
    const auto round_trip = arc::project::load_input_config(saved_path);
    REQUIRE(round_trip.succeeded);
    const auto& round_trip_binding = round_trip.config.contexts[0].actions[0].bindings[0];
    REQUIRE(round_trip_binding.modifiers.size() == 1);
    REQUIRE(round_trip_binding.composite_processors.size() == 1);
    CHECK(round_trip.config.contexts[0].axes2d[0].bindings.size() == 4);
}

TEST_CASE("input config rejects incomplete modifier chords")
{
    temporary_input_config temporary;

    SECTION("empty modifier list")
    {
        const auto path = temporary.write(R"json({
          "version": 1,
          "contexts": [{
            "name": "Gameplay",
            "actions": [{
              "name": "Save",
              "bindings": [{"device": "keyboard", "control": "S", "modifiers": []}]
            }]
          }]
        })json");
        const auto loaded = arc::project::load_input_config(path);
        CHECK_FALSE(loaded.succeeded);
        CHECK(loaded.error.find("modifiers") != std::string::npos);
    }

    SECTION("modifier without control")
    {
        const auto path = temporary.write(R"json({
          "version": 1,
          "contexts": [{
            "name": "Gameplay",
            "actions": [{
              "name": "Save",
              "bindings": [{
                "device": "keyboard",
                "control": "S",
                "modifiers": [{"device": "keyboard"}]
              }]
            }]
          }]
        })json", "Invalid.json");
        const auto loaded = arc::project::load_input_config(path);
        CHECK_FALSE(loaded.succeeded);
        CHECK(loaded.error.find("modifier") != std::string::npos);
    }
}
