#pragma once

#include "windows_controller_extension_host.h"

#include <arc/input/input.h>

#include <windows.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc::platform::windows
{

class windows_dualsense_output_sink;

struct windows_hid_interface
{
    std::wstring path;
    input::input_device_hardware_id hardware_id{};
    std::uint16_t usage_page{};
    std::uint16_t usage{};
};

struct windows_hid_extension_match
{
    std::wstring path;
    input::input_device_id device{};
    input::input_device_hardware_id hardware_id{};
};

/**
 * @brief Match controller-class HID interfaces to existing GameInput devices.
 *
 * Matching is intentionally conservative: hardware identity must resolve to one
 * connected GameInput device or the interface is left unattached.
 */
[[nodiscard]] std::vector<windows_hid_extension_match>
match_hid_extensions(const std::vector<windows_hid_interface>& interfaces, const input::input_system& input);

/**
 * @brief Discovers Windows HID controller interfaces and attaches them as extensions.
 *
 * HID interfaces augment GameInput-owned logical controllers instead of creating
 * another player-visible controller. Raw HID reports are routed to device-specific
 * parsers while discovery and logical controller ownership remain separate.
 */
class windows_hid_extension_manager final
{
public:
    windows_hid_extension_manager(input::input_system& input, windows_controller_extension_host& host) noexcept;
    ~windows_hid_extension_manager();

    windows_hid_extension_manager(const windows_hid_extension_manager&) = delete;
    windows_hid_extension_manager& operator=(const windows_hid_extension_manager&) = delete;

    [[nodiscard]] bool attach(HWND window);
    void process_message(UINT message, WPARAM wparam, LPARAM lparam);
    void poll();

private:
    struct attachment
    {
        input::input_device_id device{};
        input::input_device_hardware_id hardware_id{};
        windows_controller_extension_id extension{};
        std::unique_ptr<windows_dualsense_output_sink> advanced_output;
    };

    void detach_attachment(attachment& value);
    void handle_raw_input(HRAWINPUT raw_input);
    [[nodiscard]] static std::vector<windows_hid_interface> enumerate_interfaces();
    [[nodiscard]] static std::wstring device_path(HANDLE native_device);
    [[nodiscard]] static std::string utf8_path(const std::wstring& path);

    input::input_system* input_{};
    windows_controller_extension_host* host_{};
    std::unordered_map<std::wstring, attachment> attachments_;
    std::uint64_t next_scan_ticks_{};
    bool raw_input_attached_{};
};

} // namespace arc::platform::windows
