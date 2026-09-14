#pragma once

#include <arc/input/input.h>

#include <cstdint>
#include <span>
#include <vector>

namespace arc::platform::windows
{

[[nodiscard]] bool is_dualsense_hardware(const input::input_device_hardware_id& hardware) noexcept;

/**
 * @brief Parse active DualSense touchpad contacts from one USB or Bluetooth HID report.
 * @return True when the report is a recognized full DualSense input report.
 */
[[nodiscard]] bool parse_dualsense_touch_report(std::span<const std::uint8_t> report,
                                                std::vector<input::input_touch_contact>& contacts);

} // namespace arc::platform::windows
