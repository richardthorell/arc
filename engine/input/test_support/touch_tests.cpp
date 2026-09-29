#include <arc/input/input.h>
#include <arc/input/touch.h>

#include <cmath>
#include <limits>
#include <vector>

namespace
{

int require(bool condition, int code)
{
    return condition ? 0 : code;
}

} // namespace

int main()
{
    arc::input::input_system input;
    const arc::input::input_device_id device{.value = 9001};
    input.connect_device({.id = device,
                          .type = arc::input::input_device_type::gamepad,
                          .name = "Touch Test",
                          .capabilities = {.touchpad = true}});
    if (const int error = require(input.assign_device(0, device), 1)) return error;

    auto& player = input.player(0);
    player.add_context("gameplay");
    player.bind_action("gameplay", "touch",
                       {.device = arc::input::input_device_type::gamepad,
                        .control = arc::input::make_touch_control(arc::input::touch_control::primary_down)});
    player.bind_axis2d("gameplay", "touch_position",
                       {.device = arc::input::input_device_type::gamepad,
                        .control = arc::input::make_touch_control(arc::input::touch_control::primary_x)},
                       {1.0f, 0.0f});
    player.bind_axis2d("gameplay", "touch_position",
                       {.device = arc::input::input_device_type::gamepad,
                        .control = arc::input::make_touch_control(arc::input::touch_control::primary_y)},
                       {0.0f, 1.0f});
    player.bind_axis("gameplay", "touch_pressure",
                     {.device = arc::input::input_device_type::gamepad,
                      .control = arc::input::make_touch_control(arc::input::touch_control::primary_pressure)});

    std::vector<arc::input::input_touch_contact> contacts{
        {.id = 3, .surface = 0, .position = {-0.25f, 1.25f}, .pressure = 2.0f, .pressure_available = true},
        {.id = 8,
         .surface = 0,
         .position = {0.5f, 0.25f},
         .pressure = std::numeric_limits<float>::quiet_NaN(),
         .pressure_available = true}};

    if (const int error = require(input.submit_touch_contacts(device, contacts), 2)) return error;
    const auto& stored = input.touch_contacts(device);
    if (const int error = require(stored.size() == 2, 3)) return error;
    if (const int error = require(stored[0].position[0] == 0.0f && stored[0].position[1] == 1.0f, 4)) return error;
    if (const int error = require(stored[0].pressure == 1.0f && stored[0].pressure_available, 5)) return error;
    if (const int error = require(stored[1].pressure == 0.0f && !stored[1].pressure_available, 6)) return error;
    if (const int error = require(player.pressed("touch") && player.down("touch"), 7)) return error;
    const auto position = player.axis2d("touch_position");
    if (const int error = require(position[0] == 0.0f && position[1] == 1.0f, 8)) return error;
    if (const int error = require(player.axis("touch_pressure") == 1.0f, 9)) return error;

    input.begin_frame();
    if (const int error = require(input.submit_touch_contacts(device, {}), 10)) return error;
    if (const int error = require(player.released("touch"), 11)) return error;
    const auto released_position = player.axis2d("touch_position");
    if (const int error = require(released_position[0] == 0.0f && released_position[1] == 0.0f, 12)) return error;

    if (const int error = require(input.disconnect_device(device), 13)) return error;
    if (const int error = require(input.touch_contacts(device).empty(), 14)) return error;
    if (const int error = require(!input.submit_touch_contacts(device, contacts), 15)) return error;
    if (const int error = require(input.submit_touch_contacts(device, {}), 16)) return error;

    input.connect_device({.id = device,
                          .type = arc::input::input_device_type::gamepad,
                          .name = "Touch Test",
                          .capabilities = {.touchpad = true}});
    if (const int error = require(input.touch_contacts(device).empty(), 17)) return error;
    return 0;
}
