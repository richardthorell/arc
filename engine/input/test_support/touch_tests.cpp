#include <arc/input/input.h>

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

    std::vector<arc::input::input_touch_contact> contacts{
        {.id = 3, .surface = 0, .position = {-0.25f, 1.25f}, .pressure = 2.0f, .pressure_available = true},
        {.id = 8,
         .surface = 0,
         .position = {0.5f, 0.25f},
         .pressure = std::numeric_limits<float>::quiet_NaN(),
         .pressure_available = true}};

    if (const int error = require(input.submit_touch_contacts(device, contacts), 1)) return error;
    const auto& stored = input.touch_contacts(device);
    if (const int error = require(stored.size() == 2, 2)) return error;
    if (const int error = require(stored[0].position[0] == 0.0f && stored[0].position[1] == 1.0f, 3)) return error;
    if (const int error = require(stored[0].pressure == 1.0f && stored[0].pressure_available, 4)) return error;
    if (const int error = require(stored[1].pressure == 0.0f && !stored[1].pressure_available, 5)) return error;

    if (const int error = require(input.disconnect_device(device), 6)) return error;
    if (const int error = require(input.touch_contacts(device).empty(), 7)) return error;
    if (const int error = require(!input.submit_touch_contacts(device, contacts), 8)) return error;
    if (const int error = require(input.submit_touch_contacts(device, {}), 9)) return error;

    input.connect_device({.id = device,
                          .type = arc::input::input_device_type::gamepad,
                          .name = "Touch Test",
                          .capabilities = {.touchpad = true}});
    if (const int error = require(input.touch_contacts(device).empty(), 10)) return error;
    return 0;
}
