#include "windows_dualsense_output.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace arc::platform::windows
{
namespace
{

constexpr std::size_t usb_report_size = 63;
constexpr std::size_t bluetooth_report_size = 78;
constexpr std::size_t common_size = 47;
constexpr std::uint8_t usb_report_id = 0x02;
constexpr std::uint8_t bluetooth_report_id = 0x31;
constexpr std::uint8_t bluetooth_tag = 0x10;
constexpr std::uint8_t output_crc_seed = 0xa2;
constexpr std::uint8_t valid_flag0_right_trigger = 0x04;
constexpr std::uint8_t valid_flag0_left_trigger = 0x08;
constexpr std::uint8_t valid_flag1_lightbar = 0x04;
constexpr std::uint8_t trigger_effect_off = 0x05;
constexpr std::uint8_t trigger_effect_feedback = 0x21;
constexpr std::uint8_t trigger_effect_weapon = 0x25;
constexpr std::uint8_t trigger_effect_vibration = 0x26;
constexpr std::size_t right_trigger_offset = 10;
constexpr std::size_t left_trigger_offset = 21;
constexpr std::size_t lightbar_red_offset = 44;
constexpr std::size_t lightbar_green_offset = 45;
constexpr std::size_t lightbar_blue_offset = 46;

std::uint8_t unit_byte(float value) noexcept
{
    if (!std::isfinite(value)) return 0;
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

std::uint8_t strength_level(float value) noexcept
{
    if (!std::isfinite(value) || value <= 0.0f) return 0;
    return static_cast<std::uint8_t>(std::clamp(std::lround(std::clamp(value, 0.0f, 1.0f) * 8.0f), 1l, 8l));
}

std::uint8_t position_zone(float value, std::uint8_t minimum, std::uint8_t maximum) noexcept
{
    if (!std::isfinite(value)) value = 0.0f;
    const long zone = std::lround(std::clamp(value, 0.0f, 1.0f) * 9.0f);
    return static_cast<std::uint8_t>(std::clamp(zone, static_cast<long>(minimum), static_cast<long>(maximum)));
}

void encode_zone_strengths(std::span<std::uint8_t, 11> output, std::uint8_t mode, std::uint8_t start_zone,
                           std::uint8_t end_zone, std::uint8_t strength, std::uint8_t frequency = 0) noexcept
{
    std::fill(output.begin(), output.end(), std::uint8_t{});
    if (strength == 0)
    {
        output[0] = trigger_effect_off;
        return;
    }

    std::uint16_t active_zones = 0;
    std::uint32_t strengths = 0;
    const std::uint32_t encoded_strength = static_cast<std::uint32_t>((strength - 1U) & 0x07U);
    for (std::uint8_t zone = start_zone; zone <= end_zone && zone < 10; ++zone)
    {
        active_zones |= static_cast<std::uint16_t>(1U << zone);
        strengths |= encoded_strength << (3U * zone);
    }

    output[0] = mode;
    output[1] = static_cast<std::uint8_t>(active_zones & 0xffU);
    output[2] = static_cast<std::uint8_t>((active_zones >> 8U) & 0xffU);
    output[3] = static_cast<std::uint8_t>(strengths & 0xffU);
    output[4] = static_cast<std::uint8_t>((strengths >> 8U) & 0xffU);
    output[5] = static_cast<std::uint8_t>((strengths >> 16U) & 0xffU);
    output[6] = static_cast<std::uint8_t>((strengths >> 24U) & 0xffU);
    output[9] = frequency;
}

void encode_trigger_effect(const input::input_adaptive_trigger_effect& effect, std::span<std::uint8_t, 11> output)
{
    std::fill(output.begin(), output.end(), std::uint8_t{});
    const std::uint8_t strength = strength_level(effect.strength);

    switch (effect.type)
    {
        case input::input_adaptive_trigger_effect_type::off:
            output[0] = trigger_effect_off;
            break;
        case input::input_adaptive_trigger_effect_type::resistance:
        {
            const std::uint8_t start = position_zone(effect.start_position, 0, 9);
            const std::uint8_t end = position_zone(effect.end_position, start, 9);
            encode_zone_strengths(output, trigger_effect_feedback, start, end, strength);
            break;
        }
        case input::input_adaptive_trigger_effect_type::weapon:
        {
            if (strength == 0)
            {
                output[0] = trigger_effect_off;
                break;
            }

            const std::uint8_t start = position_zone(effect.start_position, 2, 7);
            const std::uint8_t requested_end = position_zone(effect.end_position, 3, 8);
            const std::uint8_t end = std::max<std::uint8_t>(requested_end, static_cast<std::uint8_t>(start + 1));
            const std::uint16_t zones = static_cast<std::uint16_t>((1U << start) | (1U << end));
            output[0] = trigger_effect_weapon;
            output[1] = static_cast<std::uint8_t>(zones & 0xffU);
            output[2] = static_cast<std::uint8_t>((zones >> 8U) & 0xffU);
            output[3] = static_cast<std::uint8_t>(strength - 1U);
            break;
        }
        case input::input_adaptive_trigger_effect_type::vibration:
        {
            const auto frequency = static_cast<std::uint8_t>(
                std::clamp(std::lround(std::isfinite(effect.frequency_hz) ? effect.frequency_hz : 0.0f), 0l, 255l));
            if (frequency == 0)
            {
                output[0] = trigger_effect_off;
                break;
            }
            const std::uint8_t start = position_zone(effect.start_position, 0, 9);
            const std::uint8_t end = position_zone(effect.end_position, start, 9);
            encode_zone_strengths(output, trigger_effect_vibration, start, end, strength, frequency);
            break;
        }
    }
}

std::uint32_t crc32_update(std::uint32_t crc, std::uint8_t value) noexcept
{
    crc ^= value;
    for (int bit = 0; bit < 8; ++bit)
        crc = (crc >> 1U) ^ ((crc & 1U) != 0 ? 0xedb88320U : 0U);
    return crc;
}

std::uint32_t bluetooth_crc(std::span<const std::uint8_t> report_without_crc) noexcept
{
    std::uint32_t crc = 0xffffffffU;
    crc = crc32_update(crc, output_crc_seed);
    for (std::uint8_t value : report_without_crc)
        crc = crc32_update(crc, value);
    return ~crc;
}

} // namespace

std::vector<std::uint8_t> build_dualsense_output_report(dualsense_output_transport transport, std::uint8_t sequence,
                                                        const dualsense_output_command& command)
{
    const bool bluetooth = transport == dualsense_output_transport::bluetooth;
    std::vector<std::uint8_t> report(bluetooth ? bluetooth_report_size : usb_report_size, 0);
    const std::size_t common_offset = bluetooth ? 3 : 1;

    if (bluetooth)
    {
        report[0] = bluetooth_report_id;
        report[1] = static_cast<std::uint8_t>((sequence & 0x0fU) << 4U);
        report[2] = bluetooth_tag;
    }
    else
    {
        report[0] = usb_report_id;
    }

    std::span<std::uint8_t> common(report.data() + common_offset, common_size);
    if (command.update_light)
    {
        common[1] |= valid_flag1_lightbar;
        common[lightbar_red_offset] = unit_byte(command.light.red);
        common[lightbar_green_offset] = unit_byte(command.light.green);
        common[lightbar_blue_offset] = unit_byte(command.light.blue);
    }

    if (command.update_adaptive_triggers)
    {
        common[0] |= static_cast<std::uint8_t>(valid_flag0_right_trigger | valid_flag0_left_trigger);
        encode_trigger_effect(command.adaptive_triggers.right,
                              std::span<std::uint8_t, 11>(common.data() + right_trigger_offset, 11));
        encode_trigger_effect(command.adaptive_triggers.left,
                              std::span<std::uint8_t, 11>(common.data() + left_trigger_offset, 11));
    }

    if (bluetooth)
    {
        const std::size_t crc_offset = report.size() - sizeof(std::uint32_t);
        const std::uint32_t crc = bluetooth_crc(std::span<const std::uint8_t>(report.data(), crc_offset));
        report[crc_offset] = static_cast<std::uint8_t>(crc & 0xffU);
        report[crc_offset + 1] = static_cast<std::uint8_t>((crc >> 8U) & 0xffU);
        report[crc_offset + 2] = static_cast<std::uint8_t>((crc >> 16U) & 0xffU);
        report[crc_offset + 3] = static_cast<std::uint8_t>((crc >> 24U) & 0xffU);
    }

    return report;
}

windows_dualsense_output_sink::windows_dualsense_output_sink(input::input_device_id device, std::wstring path,
                                                             dualsense_output_transport transport)
    : device_(device), path_(std::move(path)), transport_(transport)
{
    handle_ = CreateFileW(path_.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0,
                          nullptr);
}

windows_dualsense_output_sink::~windows_dualsense_output_sink()
{
    if (handle_ != INVALID_HANDLE_VALUE)
    {
        static_cast<void>(set_adaptive_triggers(device_, {}));
        CloseHandle(handle_);
    }
}

bool windows_dualsense_output_sink::available() const noexcept
{
    return handle_ != INVALID_HANDLE_VALUE;
}

bool windows_dualsense_output_sink::set_light(input::input_device_id device, input::input_light_state state)
{
    if (device != device_) return false;
    dualsense_output_command command{};
    command.update_light = true;
    command.light = state;
    return send(command);
}

bool windows_dualsense_output_sink::set_adaptive_triggers(input::input_device_id device,
                                                          input::input_adaptive_trigger_state state)
{
    if (device != device_) return false;
    dualsense_output_command command{};
    command.update_adaptive_triggers = true;
    command.adaptive_triggers = state;
    return send(command);
}

bool windows_dualsense_output_sink::send(const dualsense_output_command& command)
{
    if (handle_ == INVALID_HANDLE_VALUE) return false;

    const std::vector<std::uint8_t> report = build_dualsense_output_report(transport_, sequence_, command);
    if (transport_ == dualsense_output_transport::bluetooth)
        sequence_ = static_cast<std::uint8_t>((sequence_ + 1U) & 0x0fU);

    DWORD written = 0;
    return WriteFile(handle_, report.data(), static_cast<DWORD>(report.size()), &written, nullptr) != FALSE &&
           written == static_cast<DWORD>(report.size());
}

} // namespace arc::platform::windows
