#include "windows_hid_extension_manager.h"

#include "windows_dualsense_output.h"
#include "windows_dualsense_touch.h"

#include <windows.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::platform::windows
{
namespace
{

constexpr std::uint16_t generic_desktop_usage_page = 0x01;
constexpr std::uint16_t joystick_usage = 0x04;
constexpr std::uint16_t gamepad_usage = 0x05;
constexpr std::uint16_t multi_axis_usage = 0x08;
constexpr std::uint64_t hid_scan_interval_ms = 1000;

bool controller_usage(const windows_hid_interface& interface) noexcept
{
    if (interface.usage_page != generic_desktop_usage_page) return false;
    return interface.usage == joystick_usage || interface.usage == gamepad_usage || interface.usage == multi_axis_usage;
}

bool hardware_matches(const input::input_device_hardware_id& candidate,
                      const input::input_device_hardware_id& interface) noexcept
{
    if (candidate.vendor_id == 0 || candidate.product_id == 0 || interface.vendor_id == 0 || interface.product_id == 0)
        return false;
    if (candidate.vendor_id != interface.vendor_id || candidate.product_id != interface.product_id) return false;
    if (candidate.version != 0 && interface.version != 0 && candidate.version != interface.version) return false;
    return true;
}

dualsense_output_transport output_transport(const input::input_device* device) noexcept
{
    if (device && device->connectivity() == input::input_connectivity_type::wireless)
        return dualsense_output_transport::bluetooth;
    return dualsense_output_transport::usb;
}

} // namespace

std::vector<windows_hid_extension_match> match_hid_extensions(const std::vector<windows_hid_interface>& interfaces,
                                                              const input::input_system& input)
{
    std::vector<windows_hid_extension_match> matches;
    for (const windows_hid_interface& interface : interfaces)
    {
        if (!controller_usage(interface)) continue;

        input::input_device_id candidate{};
        bool ambiguous = false;
        for (input::input_device_id id : input.devices(true))
        {
            const input::input_device* device = input.device(id);
            if (!device || device->backend() != input::input_backend_type::game_input) continue;
            if (!hardware_matches(device->hardware_id(), interface.hardware_id)) continue;

            if (candidate)
            {
                ambiguous = true;
                break;
            }
            candidate = id;
        }

        if (candidate && !ambiguous)
            matches.push_back({.path = interface.path, .device = candidate, .hardware_id = interface.hardware_id});
    }
    return matches;
}

windows_hid_extension_manager::windows_hid_extension_manager(input::input_system& input,
                                                             windows_controller_extension_host& host) noexcept
    : input_(&input), host_(&host)
{
}

windows_hid_extension_manager::~windows_hid_extension_manager()
{
    for (auto& [_, value] : attachments_)
        detach_attachment(value);
}

bool windows_hid_extension_manager::attach(HWND window)
{
    RAWINPUTDEVICE devices[3]{};
    devices[0] = {.usUsagePage = generic_desktop_usage_page,
                  .usUsage = joystick_usage,
                  .dwFlags = RIDEV_DEVNOTIFY,
                  .hwndTarget = window};
    devices[1] = {.usUsagePage = generic_desktop_usage_page,
                  .usUsage = gamepad_usage,
                  .dwFlags = RIDEV_DEVNOTIFY,
                  .hwndTarget = window};
    devices[2] = {.usUsagePage = generic_desktop_usage_page,
                  .usUsage = multi_axis_usage,
                  .dwFlags = RIDEV_DEVNOTIFY,
                  .hwndTarget = window};

    raw_input_attached_ = RegisterRawInputDevices(devices, 3, sizeof(RAWINPUTDEVICE)) != FALSE;
    if (raw_input_attached_) next_scan_ticks_ = 0;
    return raw_input_attached_;
}

void windows_hid_extension_manager::process_message(UINT message, WPARAM, LPARAM lparam)
{
    if (!raw_input_attached_) return;

    if (message == WM_INPUT)
        handle_raw_input(reinterpret_cast<HRAWINPUT>(lparam));
    else if (message == WM_INPUT_DEVICE_CHANGE)
        next_scan_ticks_ = 0;
}

void windows_hid_extension_manager::poll()
{
    const std::uint64_t now = GetTickCount64();
    if (next_scan_ticks_ != 0 && now < next_scan_ticks_) return;
    next_scan_ticks_ = now + hid_scan_interval_ms;

    const std::vector<windows_hid_interface> interfaces = enumerate_interfaces();
    const std::vector<windows_hid_extension_match> matches = match_hid_extensions(interfaces, *input_);

    std::unordered_map<std::wstring, input::input_device_id> desired;
    desired.reserve(matches.size());
    for (const windows_hid_extension_match& match : matches)
        desired.insert_or_assign(match.path, match.device);

    for (auto it = attachments_.begin(); it != attachments_.end();)
    {
        const auto wanted = desired.find(it->first);
        const input::input_device* device = input_->device(it->second.device);
        const bool keep =
            wanted != desired.end() && wanted->second == it->second.device && device && device->connected();
        if (keep)
        {
            host_->refresh(it->second.device);
            ++it;
            continue;
        }

        detach_attachment(it->second);
        it = attachments_.erase(it);
    }

    for (const windows_hid_extension_match& match : matches)
    {
        if (attachments_.contains(match.path)) continue;

        const bool dualsense = is_dualsense_hardware(match.hardware_id);
        const input::input_device* device = input_->device(match.device);
        std::unique_ptr<windows_dualsense_output_sink> advanced_output;
        if (dualsense)
        {
            advanced_output =
                std::make_unique<windows_dualsense_output_sink>(match.device, match.path, output_transport(device));
            if (!advanced_output->available() || !input_->register_advanced_output_sink(match.device, *advanced_output))
                advanced_output.reset();
        }

        windows_controller_extension_descriptor descriptor{};
        descriptor.backend = input::input_backend_type::hid;
        descriptor.backend_id = utf8_path(match.path);
        descriptor.capabilities.touchpad = dualsense;
        descriptor.capabilities.light = advanced_output != nullptr;
        descriptor.capabilities.adaptive_triggers = advanced_output != nullptr;
        const windows_controller_extension_id extension = host_->attach(match.device, std::move(descriptor));
        if (!extension)
        {
            if (advanced_output) input_->unregister_advanced_output_sink(match.device, *advanced_output);
            continue;
        }

        attachments_.emplace(match.path, attachment{.device = match.device,
                                                    .hardware_id = match.hardware_id,
                                                    .extension = extension,
                                                    .advanced_output = std::move(advanced_output)});
    }
}

void windows_hid_extension_manager::detach_attachment(attachment& value)
{
    if (is_dualsense_hardware(value.hardware_id)) input_->submit_touch_contacts(value.device, {});
    if (value.advanced_output)
    {
        input_->unregister_advanced_output_sink(value.device, *value.advanced_output);
        value.advanced_output.reset();
    }
    host_->detach(value.device, value.extension);
}

void windows_hid_extension_manager::handle_raw_input(HRAWINPUT raw_input)
{
    UINT size = 0;
    if (GetRawInputData(raw_input, RID_INPUT, nullptr, &size, sizeof(RAWINPUTHEADER)) == static_cast<UINT>(-1) ||
        size == 0)
        return;

    std::vector<std::uint8_t> buffer(size);
    UINT read_size = size;
    if (GetRawInputData(raw_input, RID_INPUT, buffer.data(), &read_size, sizeof(RAWINPUTHEADER)) ==
        static_cast<UINT>(-1))
        return;
    if (read_size < sizeof(RAWINPUTHEADER)) return;

    const auto* input_report = reinterpret_cast<const RAWINPUT*>(buffer.data());
    if (input_report->header.dwType != RIM_TYPEHID) return;

    const std::wstring path = device_path(input_report->header.hDevice);
    const auto attachment_it = attachments_.find(path);
    if (attachment_it == attachments_.end() || !is_dualsense_hardware(attachment_it->second.hardware_id)) return;

    const RAWHID& hid = input_report->data.hid;
    const auto* data = reinterpret_cast<const std::uint8_t*>(hid.bRawData);
    std::vector<input::input_touch_contact> contacts;
    for (DWORD index = 0; index < hid.dwCount; ++index)
    {
        const std::span<const std::uint8_t> report(data + static_cast<std::size_t>(index) * hid.dwSizeHid,
                                                   hid.dwSizeHid);
        if (parse_dualsense_touch_report(report, contacts))
            input_->submit_touch_contacts(attachment_it->second.device, contacts);
    }
}

std::vector<windows_hid_interface> windows_hid_extension_manager::enumerate_interfaces()
{
    UINT count = 0;
    if (GetRawInputDeviceList(nullptr, &count, sizeof(RAWINPUTDEVICELIST)) != 0 || count == 0) return {};

    std::vector<RAWINPUTDEVICELIST> devices(count);
    const UINT result = GetRawInputDeviceList(devices.data(), &count, sizeof(RAWINPUTDEVICELIST));
    if (result == std::numeric_limits<UINT>::max()) return {};
    devices.resize(result);

    std::vector<windows_hid_interface> interfaces;
    for (const RAWINPUTDEVICELIST& raw : devices)
    {
        if (raw.dwType != RIM_TYPEHID) continue;

        RID_DEVICE_INFO info{};
        info.cbSize = sizeof(info);
        UINT info_size = sizeof(info);
        if (GetRawInputDeviceInfoW(raw.hDevice, RIDI_DEVICEINFO, &info, &info_size) == std::numeric_limits<UINT>::max())
            continue;

        std::wstring path = device_path(raw.hDevice);
        if (path.empty()) continue;

        interfaces.push_back({.path = std::move(path),
                              .hardware_id = {.vendor_id = static_cast<std::uint16_t>(info.hid.dwVendorId),
                                              .product_id = static_cast<std::uint16_t>(info.hid.dwProductId),
                                              .version = static_cast<std::uint16_t>(info.hid.dwVersionNumber)},
                              .usage_page = info.hid.usUsagePage,
                              .usage = info.hid.usUsage});
    }
    return interfaces;
}

std::wstring windows_hid_extension_manager::device_path(HANDLE native_device)
{
    UINT path_size = 0;
    if (GetRawInputDeviceInfoW(native_device, RIDI_DEVICENAME, nullptr, &path_size) != 0 || path_size == 0) return {};

    std::wstring path(path_size, L'\0');
    const UINT result = GetRawInputDeviceInfoW(native_device, RIDI_DEVICENAME, path.data(), &path_size);
    if (result == std::numeric_limits<UINT>::max()) return {};
    if (!path.empty() && path.back() == L'\0') path.pop_back();
    return path;
}

std::string windows_hid_extension_manager::utf8_path(const std::wstring& path)
{
    if (path.empty()) return {};

    const int required =
        WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), nullptr, 0, nullptr, nullptr);
    if (required <= 0) return {};

    std::string result(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), result.data(), required, nullptr,
                        nullptr);
    return result;
}

} // namespace arc::platform::windows
