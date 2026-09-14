#include "windows_dualsense_touch.h"

#include <arc/input/input.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace
{

int require(bool condition, int code)
{
    return condition ? 0 : code;
}

void write_touch(std::span<std::uint8_t, 4> point, std::uint8_t id, std::uint16_t x, std::uint16_t y, bool active)
{
    point[0] = static_cast<std::uint8_t>((active ? 0 : 0x80) | (id & 0x7f));
    point[1] = static_cast<std::uint8_t>(x & 0xff);
    point[2] = static_cast<std::uint8_t>(((x >> 8) & 0x0f) | ((y & 0x0f) << 4));
    point[3] = static_cast<std::uint8_t>((y >> 4) & 0xff);
}

int test_usb_report()
{
    std::array<std::uint8_t, 64> report{};
    report[0] = 0x01;
    write_touch(std::span<std::uint8_t, 4>(report.data() + 33, 4), 7, 960, 540, true);
    write_touch(std::span<std::uint8_t, 4>(report.data() + 37, 4), 9, 100, 200, false);

    std::vector<arc::input::input_touch_contact> contacts;
    if (const int error = require(arc::platform::windows::parse_dualsense_touch_report(report, contacts), 1))
        return error;
    if (const int error = require(contacts.size() == 1, 2)) return error;
    if (const int error = require(contacts[0].id == 7 && contacts[0].surface == 0, 3)) return error;
    if (const int error = require(std::abs(contacts[0].position[0] - (960.0f / 1919.0f)) < 0.0001f, 4)) return error;
    if (const int error = require(std::abs(contacts[0].position[1] - (540.0f / 1079.0f)) < 0.0001f, 5)) return error;
    if (const int error = require(!contacts[0].pressure_available, 6)) return error;
    return 0;
}

int test_bluetooth_report()
{
    std::array<std::uint8_t, 78> report{};
    report[0] = 0x31;
    write_touch(std::span<std::uint8_t, 4>(report.data() + 34, 4), 12, 1919, 1079, true);
    write_touch(std::span<std::uint8_t, 4>(report.data() + 38, 4), 13, 0, 0, true);

    std::vector<arc::input::input_touch_contact> contacts;
    if (const int error = require(arc::platform::windows::parse_dualsense_touch_report(report, contacts), 10))
        return error;
    if (const int error = require(contacts.size() == 2, 11)) return error;
    if (const int error = require(contacts[0].position[0] == 1.0f && contacts[0].position[1] == 1.0f, 12)) return error;
    if (const int error = require(contacts[1].position[0] == 0.0f && contacts[1].position[1] == 0.0f, 13)) return error;
    return 0;
}

int test_recognition()
{
    if (const int error =
            require(arc::platform::windows::is_dualsense_hardware({.vendor_id = 0x054c, .product_id = 0x0ce6}), 20))
        return error;
    if (const int error =
            require(arc::platform::windows::is_dualsense_hardware({.vendor_id = 0x054c, .product_id = 0x0df2}), 21))
        return error;
    if (const int error =
            require(!arc::platform::windows::is_dualsense_hardware({.vendor_id = 0x045e, .product_id = 0x0b13}), 22))
        return error;

    std::array<std::uint8_t, 10> minimal{};
    minimal[0] = 0x01;
    std::vector<arc::input::input_touch_contact> contacts;
    if (const int error = require(!arc::platform::windows::parse_dualsense_touch_report(minimal, contacts), 23))
        return error;
    return 0;
}

} // namespace

int main()
{
    if (const int error = test_usb_report()) return error;
    if (const int error = test_bluetooth_report()) return error;
    return test_recognition();
}
