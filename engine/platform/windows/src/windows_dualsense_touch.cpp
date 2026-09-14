#include "windows_dualsense_touch.h"

#include <cstddef>

namespace arc::platform::windows
{
namespace
{

constexpr std::uint16_t sony_vendor_id = 0x054c;
constexpr std::uint16_t dualsense_product_id = 0x0ce6;
constexpr std::uint16_t dualsense_edge_product_id = 0x0df2;
constexpr std::uint8_t usb_report_id = 0x01;
constexpr std::uint8_t bluetooth_report_id = 0x31;
constexpr std::size_t usb_report_size = 64;
constexpr std::size_t bluetooth_report_size = 78;
constexpr std::size_t touch_offset_in_common_report = 32;
constexpr float touchpad_max_x = 1919.0f;
constexpr float touchpad_max_y = 1079.0f;
constexpr std::uint8_t inactive_contact_mask = 0x80;
constexpr std::uint8_t contact_id_mask = 0x7f;

void append_contact(std::span<const std::uint8_t> point, std::vector<input::input_touch_contact>& contacts)
{
    if (point.size() < 4 || (point[0] & inactive_contact_mask) != 0) return;

    const std::uint16_t x = static_cast<std::uint16_t>(point[1]) | (static_cast<std::uint16_t>(point[2] & 0x0f) << 8);
    const std::uint16_t y =
        static_cast<std::uint16_t>((point[2] >> 4) & 0x0f) | (static_cast<std::uint16_t>(point[3]) << 4);

    contacts.push_back({.id = static_cast<std::uint32_t>(point[0] & contact_id_mask),
                        .surface = 0,
                        .position = {static_cast<float>(x) / touchpad_max_x, static_cast<float>(y) / touchpad_max_y},
                        .pressure = 0.0f,
                        .pressure_available = false});
}

} // namespace

bool is_dualsense_hardware(const input::input_device_hardware_id& hardware) noexcept
{
    return hardware.vendor_id == sony_vendor_id &&
           (hardware.product_id == dualsense_product_id || hardware.product_id == dualsense_edge_product_id);
}

bool parse_dualsense_touch_report(std::span<const std::uint8_t> report,
                                  std::vector<input::input_touch_contact>& contacts)
{
    std::size_t common_offset = 0;
    if (report.size() >= usb_report_size && report[0] == usb_report_id)
        common_offset = 1;
    else if (report.size() >= bluetooth_report_size && report[0] == bluetooth_report_id)
        common_offset = 2;
    else
        return false;

    const std::size_t first_touch = common_offset + touch_offset_in_common_report;
    if (report.size() < first_touch + 8) return false;

    contacts.clear();
    contacts.reserve(2);
    append_contact(report.subspan(first_touch, 4), contacts);
    append_contact(report.subspan(first_touch + 4, 4), contacts);
    return true;
}

} // namespace arc::platform::windows
