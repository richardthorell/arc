#pragma once

#include "windows_controller_extension_host.h"

#include <arc/input/input.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc::platform::windows
{

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
 * another player-visible controller. This milestone establishes discovery and
 * lifecycle only; device-specific HID report parsing is added by later extensions.
 */
class windows_hid_extension_manager final
{
public:
    windows_hid_extension_manager(input::input_system& input, windows_controller_extension_host& host) noexcept;
    ~windows_hid_extension_manager();

    windows_hid_extension_manager(const windows_hid_extension_manager&) = delete;
    windows_hid_extension_manager& operator=(const windows_hid_extension_manager&) = delete;

    void poll();

private:
    struct attachment
    {
        input::input_device_id device{};
        windows_controller_extension_id extension{};
    };

    [[nodiscard]] static std::vector<windows_hid_interface> enumerate_interfaces();
    [[nodiscard]] static std::string utf8_path(const std::wstring& path);

    input::input_system* input_{};
    windows_controller_extension_host* host_{};
    std::unordered_map<std::wstring, attachment> attachments_;
    std::uint64_t next_scan_ticks_{};
};

} // namespace arc::platform::windows
