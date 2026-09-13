#include <arc/input/input.h>

#include <limits>

int main()
{
    using namespace arc::input;

    input_system input;
    const input_device_id controller = input.connect_device({
        .id = {.value = 500},
        .type = input_device_type::gamepad,
        .backend = input_backend_type::game_input,
        .name = "Battery Controller",
        .capabilities = {.battery = true},
    });

    const input_device* device = input.device(controller);
    if (!device) return 1;
    if (device->battery_state().status != input_battery_status::unknown) return 2;
    if (device->battery_state().level_available) return 3;

    if (!input.submit_battery_state(controller,
                                    {.status = input_battery_status::charging, .level = 0.4f, .level_available = true}))
        return 4;
    if (!device->capabilities().battery) return 5;
    if (device->battery_state().status != input_battery_status::charging) return 6;
    if (!device->battery_state().level_available || device->battery_state().level != 0.4f) return 7;

    if (!input.submit_battery_state(
            controller, {.status = input_battery_status::discharging, .level = 1.5f, .level_available = true}))
        return 8;
    if (device->battery_state().level != 1.0f) return 9;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (!input.submit_battery_state(controller,
                                    {.status = input_battery_status::idle, .level = nan, .level_available = true}))
        return 10;
    if (device->battery_state().level_available || device->battery_state().level != 0.0f) return 11;

    if (!input.submit_battery_state(controller, {.status = input_battery_status::not_present})) return 12;
    if (device->capabilities().battery) return 13;
    if (device->battery_state().status != input_battery_status::not_present) return 14;

    if (!input.submit_battery_state(
            controller, {.status = input_battery_status::charging, .level = -0.5f, .level_available = true}))
        return 15;
    if (!device->capabilities().battery) return 16;
    if (device->battery_state().level != 0.0f) return 17;

    if (!input.disconnect_device(controller)) return 18;
    if (input.submit_battery_state(
            controller, {.status = input_battery_status::discharging, .level = 0.5f, .level_available = true}))
        return 19;

    return 0;
}
