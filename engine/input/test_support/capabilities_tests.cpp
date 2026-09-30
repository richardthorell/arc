#include <arc/input/input.h>

int main()
{
    using namespace arc::input;

    input_system input;
    const auto keyboard = input.connect_device({.type = input_device_type::keyboard,
                                                .name = "Keyboard",
                                                .capabilities = {.buttons = true, .button_count = 104}});
    const auto mouse = input.connect_device(
        {.type = input_device_type::mouse,
         .name = "Mouse",
         .capabilities = {.buttons = true, .axes = true, .pointer = true, .scroll = true, .button_count = 5}});
    const auto controller = input.connect_device({
        .type = input_device_type::gamepad,
        .name = "Controller",
        .capabilities = {.buttons = true,
                         .axes = true,
                         .rumble = true,
                         .haptics = true,
                         .gyroscope = true,
                         .accelerometer = true,
                         .touchpad = true,
                         .battery = true},
    });

    auto capabilities = input.capabilities();
    if (capabilities.connected_devices != 3) return 1;
    if (!capabilities.buttons || !capabilities.axes || !capabilities.pointer || !capabilities.scroll) return 2;
    if (!capabilities.rumble || !capabilities.haptics || !capabilities.gyroscope || !capabilities.accelerometer)
        return 3;
    if (!capabilities.touch || !capabilities.battery) return 4;

    if (!input.disconnect_device(controller)) return 5;
    capabilities = input.capabilities();
    if (capabilities.connected_devices != 2) return 6;
    if (capabilities.rumble || capabilities.haptics || capabilities.gyroscope || capabilities.accelerometer ||
        capabilities.touch || capabilities.battery)
        return 7;
    if (!capabilities.buttons || !capabilities.pointer) return 8;

    if (!input.disconnect_device(keyboard) || !input.disconnect_device(mouse)) return 9;
    if (input.capabilities().connected_devices != 0) return 10;
    return 0;
}
