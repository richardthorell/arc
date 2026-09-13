#include "windows_hid_extension_manager.h"

#include <windows.h>

#include <limits>
#include <unordered_map>
#include <utility>

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

        if (candidate && !ambiguous) matches.push_back({.path = interface.path, .device = candidate});
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
    for (const auto& [_, attachment] : attachments_)
        host_->detach(attachment.device, attachment.extension);
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

        host_->detach(it->second.device, it->second.extension);
        it = attachments_.erase(it);
    }

    for (const windows_hid_extension_match& match : matches)
    {
        if (attachments_.contains(match.path)) continue;

        windows_controller_extension_descriptor descriptor{};
        descriptor.backend = input::input_backend_type::hid;
        descriptor.backend_id = utf8_path(match.path);
        const windows_controller_extension_id extension = host_->attach(match.device, std::move(descriptor));
        if (extension) attachments_.emplace(match.path, attachment{.device = match.device, .extension = extension});
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

        UINT path_size = 0;
        if (GetRawInputDeviceInfoW(raw.hDevice, RIDI_DEVICENAME, nullptr, &path_size) != 0 || path_size == 0) continue;

        std::wstring path(path_size, L'\0');
        const UINT path_result = GetRawInputDeviceInfoW(raw.hDevice, RIDI_DEVICENAME, path.data(), &path_size);
        if (path_result == std::numeric_limits<UINT>::max()) continue;
        if (!path.empty() && path.back() == L'\0') path.pop_back();
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
