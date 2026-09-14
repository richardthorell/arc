#include "windows_dualsense_output.h"

#include <arc/input/input.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace
{

int require(bool condition, int code)
{
    return condition ? 0 : code;
}

std::uint32_t crc32_update(std::uint32_t crc, std::uint8_t value) noexcept
{
    crc ^= value;
    for (int bit = 0; bit < 8; ++bit)
        crc = (crc >> 1U) ^ ((crc & 1U) != 0 ? 0xedb88320U : 0U);
    return crc;
}

std::uint32_t expected_bluetooth_crc(std::span<const std::uint8_t> report_without_crc) noexcept
{
    std::uint32_t crc = 0xffffffffU;
    crc = crc32_update(crc, 0xa2);
    for (std::uint8_t value : report_without_crc)
        crc = crc32_update(crc, value);
    return ~crc;
}

std::uint32_t stored_crc(const std::vector<std::uint8_t>& report) noexcept
{
    const std::size_t offset = report.size() - 4;
    return static_cast<std::uint32_t>(report[offset]) | (static_cast<std::uint32_t>(report[offset + 1]) << 8U) |
           (static_cast<std::uint32_t>(report[offset + 2]) << 16U) |
           (static_cast<std::uint32_t>(report[offset + 3]) << 24U);
}

int test_usb_light()
{
    using namespace arc::platform::windows;

    dualsense_output_command command{};
    command.update_light = true;
    command.light = {.red = 1.0f, .green = 0.5f, .blue = 0.0f};
    const std::vector<std::uint8_t> report = build_dualsense_output_report(dualsense_output_transport::usb, 0, command);

    if (const int error = require(report.size() == 63, 1)) return error;
    if (const int error = require(report[0] == 0x02, 2)) return error;
    if (const int error = require((report[2] & 0x04U) != 0, 3)) return error;
    if (const int error = require(report[45] == 255 && report[46] == 128 && report[47] == 0, 4)) return error;
    return 0;
}

int test_usb_triggers()
{
    using namespace arc::input;
    using namespace arc::platform::windows;

    dualsense_output_command command{};
    command.update_adaptive_triggers = true;
    command.adaptive_triggers.right = {.type = input_adaptive_trigger_effect_type::weapon,
                                       .start_position = 0.25f,
                                       .end_position = 0.75f,
                                       .strength = 1.0f};
    command.adaptive_triggers.left = {.type = input_adaptive_trigger_effect_type::vibration,
                                      .start_position = 0.5f,
                                      .end_position = 1.0f,
                                      .strength = 0.5f,
                                      .frequency_hz = 120.0f};
    const std::vector<std::uint8_t> report = build_dualsense_output_report(dualsense_output_transport::usb, 0, command);

    if (const int error = require((report[1] & 0x0cU) == 0x0cU, 10)) return error;
    if (const int error = require(report[11] == 0x25, 11)) return error;
    if (const int error = require(report[22] == 0x26, 12)) return error;
    if (const int error = require(report[31] == 120, 13)) return error;
    return 0;
}

int test_bluetooth_crc_and_sequence()
{
    using namespace arc::platform::windows;

    dualsense_output_command command{};
    command.update_light = true;
    command.light = {.red = 0.25f, .green = 0.5f, .blue = 0.75f};

    const std::vector<std::uint8_t> first =
        build_dualsense_output_report(dualsense_output_transport::bluetooth, 3, command);
    const std::vector<std::uint8_t> second =
        build_dualsense_output_report(dualsense_output_transport::bluetooth, 4, command);

    if (const int error = require(first.size() == 78, 20)) return error;
    if (const int error = require(first[0] == 0x31 && first[1] == 0x30 && first[2] == 0x10, 21)) return error;
    if (const int error = require(second[1] == 0x40, 22)) return error;

    const std::size_t crc_offset = first.size() - 4;
    if (const int error = require(stored_crc(first) == expected_bluetooth_crc({first.data(), crc_offset}), 23))
        return error;
    if (const int error = require(stored_crc(second) == expected_bluetooth_crc({second.data(), crc_offset}), 24))
        return error;
    if (const int error = require(stored_crc(first) != stored_crc(second), 25)) return error;
    return 0;
}

} // namespace

int main()
{
    if (const int error = test_usb_light()) return error;
    if (const int error = test_usb_triggers()) return error;
    return test_bluetooth_crc_and_sequence();
}
