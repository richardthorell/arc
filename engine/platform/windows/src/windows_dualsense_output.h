#pragma once

#include <arc/input/input.h>

#include <windows.h>

#include <cstdint>
#include <string>
#include <vector>

namespace arc::platform::windows
{

enum class dualsense_output_transport : std::uint8_t
{
    usb,
    bluetooth
};

struct dualsense_output_command
{
    bool update_light{};
    input::input_light_state light{};
    bool update_adaptive_triggers{};
    input::input_adaptive_trigger_state adaptive_triggers{};
};

/**
 * @brief Build one native DualSense USB or Bluetooth HID output report.
 *
 * Bluetooth reports include the required sequence/tag header and CRC32.
 */
[[nodiscard]] std::vector<std::uint8_t> build_dualsense_output_report(dualsense_output_transport transport,
                                                                      std::uint8_t sequence,
                                                                      const dualsense_output_command& command);

/**
 * @brief Device-specific HID output sink for DualSense advanced features.
 */
class windows_dualsense_output_sink final : public input::input_advanced_output_sink
{
public:
    windows_dualsense_output_sink(input::input_device_id device, std::wstring path,
                                  dualsense_output_transport transport);
    ~windows_dualsense_output_sink() override;

    windows_dualsense_output_sink(const windows_dualsense_output_sink&) = delete;
    windows_dualsense_output_sink& operator=(const windows_dualsense_output_sink&) = delete;

    [[nodiscard]] bool available() const noexcept;

    bool set_light(input::input_device_id device, input::input_light_state state) override;
    bool set_adaptive_triggers(input::input_device_id device, input::input_adaptive_trigger_state state) override;

private:
    bool send(const dualsense_output_command& command);

    input::input_device_id device_{};
    std::wstring path_;
    dualsense_output_transport transport_{dualsense_output_transport::usb};
    HANDLE handle_{INVALID_HANDLE_VALUE};
    std::uint8_t sequence_{};
};

} // namespace arc::platform::windows
