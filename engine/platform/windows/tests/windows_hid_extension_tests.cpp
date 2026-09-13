#include "windows_controller_extension_host.h"
#include "windows_hid_extension_manager.h"

#include <arc/input/input.h>

#include <vector>

namespace
{

int require(bool condition, int code)
{
    return condition ? 0 : code;
}

arc::input::input_device_descriptor controller_descriptor(arc::input::input_device_id id,
                                                          arc::input::input_backend_type backend, std::uint16_t vendor,
                                                          std::uint16_t product, std::uint16_t version)
{
    return {.id = id,
            .type = arc::input::input_device_type::gamepad,
            .subtype = arc::input::input_device_subtype::standard_gamepad,
            .connectivity = arc::input::input_connectivity_type::usb,
            .backend = backend,
            .hardware_id = {.vendor_id = vendor, .product_id = product, .version = version},
            .backend_id = "test-controller",
            .name = "Test Controller",
            .capabilities = {.buttons = true, .axes = true, .button_count = 14, .axis_count = 6}};
}

int test_extension_lifecycle()
{
    arc::input::input_system input;
    const arc::input::input_device_id device{.value = 1001};
    input.connect_device(controller_descriptor(device, arc::input::input_backend_type::game_input, 0x054c, 0x0ce6, 1));
    if (const int error = require(input.assign_device(2, device), 1)) return error;

    arc::platform::windows::windows_controller_extension_host host(input);
    arc::platform::windows::windows_controller_extension_descriptor hid{};
    hid.backend = arc::input::input_backend_type::hid;
    hid.backend_id = "hid:test";
    hid.capabilities.touchpad = true;
    hid.capabilities.haptics = true;
    hid.capabilities.button_count = 20;

    const auto first = host.attach(device, hid);
    if (const int error = require(static_cast<bool>(first), 2)) return error;
    if (const int error = require(host.extension_count(device) == 1, 3)) return error;
    if (const int error = require(input.devices(true).size() == 1, 4)) return error;

    const arc::input::input_device* record = input.device(device);
    if (const int error = require(record && record->capabilities().touchpad, 5)) return error;
    if (const int error = require(record->capabilities().haptics, 6)) return error;
    if (const int error = require(record->capabilities().button_count == 20, 7)) return error;
    if (const int error = require(input.players_for_device(device).size() == 1, 8)) return error;

    input.connect_device(controller_descriptor(device, arc::input::input_backend_type::game_input, 0x054c, 0x0ce6, 1));
    if (const int error = require(!input.device(device)->capabilities().touchpad, 9)) return error;
    host.refresh(device);
    if (const int error = require(input.device(device)->capabilities().touchpad, 10)) return error;
    if (const int error = require(input.players_for_device(device).size() == 1, 11)) return error;

    arc::platform::windows::windows_controller_extension_descriptor lighting{};
    lighting.backend = arc::input::input_backend_type::hid;
    lighting.backend_id = "hid:test:lighting";
    lighting.capabilities.light = true;
    const auto second = host.attach(device, lighting);
    if (const int error = require(static_cast<bool>(second), 12)) return error;
    if (const int error = require(host.extension_count(device) == 2, 13)) return error;

    input.submit_battery_state(
        device, {.status = arc::input::input_battery_status::charging, .level = 0.75f, .level_available = true});
    if (const int error = require(input.device(device)->capabilities().battery, 14)) return error;

    if (const int error = require(host.detach(device, first), 15)) return error;
    record = input.device(device);
    if (const int error = require(record && !record->capabilities().touchpad, 16)) return error;
    if (const int error = require(!record->capabilities().haptics, 17)) return error;
    if (const int error = require(record->capabilities().light, 18)) return error;
    if (const int error = require(record->capabilities().battery, 19)) return error;
    if (const int error = require(record->capabilities().button_count == 14, 20)) return error;

    if (const int error = require(input.disconnect_device(device), 21)) return error;
    if (const int error = require(host.detach(device, second), 22)) return error;
    if (const int error = require(host.extension_count(device) == 0, 23)) return error;
    if (const int error = require(input.device(device) && !input.device(device)->connected(), 24)) return error;
    return 0;
}

int test_hid_matching()
{
    arc::input::input_system input;
    const arc::input::input_device_id game_input_device{.value = 2001};
    input.connect_device(
        controller_descriptor(game_input_device, arc::input::input_backend_type::game_input, 0x054c, 0x0ce6, 1));
    input.connect_device(
        controller_descriptor({.value = 2002}, arc::input::input_backend_type::xinput, 0x054c, 0x0ce6, 1));

    std::vector<arc::platform::windows::windows_hid_interface> interfaces{
        {.path = L"hid-a",
         .hardware_id = {.vendor_id = 0x054c, .product_id = 0x0ce6, .version = 1},
         .usage_page = 0x01,
         .usage = 0x05},
        {.path = L"hid-keyboard",
         .hardware_id = {.vendor_id = 0x054c, .product_id = 0x0ce6, .version = 1},
         .usage_page = 0x01,
         .usage = 0x06}};

    auto matches = arc::platform::windows::match_hid_extensions(interfaces, input);
    if (const int error = require(matches.size() == 1, 30)) return error;
    if (const int error = require(matches.front().device == game_input_device, 31)) return error;
    if (const int error = require(matches.front().path == L"hid-a", 32)) return error;

    input.connect_device(
        controller_descriptor({.value = 2003}, arc::input::input_backend_type::game_input, 0x054c, 0x0ce6, 1));
    matches = arc::platform::windows::match_hid_extensions(interfaces, input);
    if (const int error = require(matches.empty(), 33)) return error;

    arc::input::input_system version_input;
    version_input.connect_device(
        controller_descriptor({.value = 3001}, arc::input::input_backend_type::game_input, 0x1234, 0xabcd, 7));
    const std::vector<arc::platform::windows::windows_hid_interface> wrong_version{
        {.path = L"hid-version",
         .hardware_id = {.vendor_id = 0x1234, .product_id = 0xabcd, .version = 8},
         .usage_page = 0x01,
         .usage = 0x05}};
    if (const int error =
            require(arc::platform::windows::match_hid_extensions(wrong_version, version_input).empty(), 34))
        return error;
    return 0;
}

} // namespace

int main()
{
    if (const int error = test_extension_lifecycle()) return error;
    return test_hid_matching();
}
