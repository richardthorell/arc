#include "windows_input_backend.h"

#include <windowsx.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cwctype>
#include <limits>
#include <string>
#include <vector>

namespace arc::platform::windows
{
namespace
{

constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

void submit_mouse_button(input::input_system& system, input::input_device_id device, USHORT flags, USHORT down_flag,
                         USHORT up_flag, input::mouse_button button)
{
    if ((flags & down_flag) != 0) system.submit_button(device, input::make_mouse_button_control(button), true);
    if ((flags & up_flag) != 0) system.submit_button(device, input::make_mouse_button_control(button), false);
}

} // namespace

windows_input_backend::windows_input_backend(input::input_system& input) noexcept : input_(&input) {}

bool windows_input_backend::attach(HWND window)
{
    window_ = window;

    RAWINPUTDEVICE devices[2]{};
    devices[0].usUsagePage = 0x01;
    devices[0].usUsage = 0x06;
    devices[0].dwFlags = RIDEV_DEVNOTIFY;
    devices[0].hwndTarget = window_;
    devices[1].usUsagePage = 0x01;
    devices[1].usUsage = 0x02;
    devices[1].dwFlags = RIDEV_DEVNOTIFY;
    devices[1].hwndTarget = window_;

    if (!RegisterRawInputDevices(devices, 2, sizeof(RAWINPUTDEVICE))) return false;

    enumerate_devices();
    return true;
}

void windows_input_backend::process_message(UINT message, WPARAM wparam, LPARAM lparam)
{
    switch (message)
    {
        case WM_INPUT:
            handle_raw_input(reinterpret_cast<HRAWINPUT>(lparam));
            break;
        case WM_INPUT_DEVICE_CHANGE:
            if (wparam == GIDC_ARRIVAL)
                register_device(reinterpret_cast<HANDLE>(lparam));
            else if (wparam == GIDC_REMOVAL)
                remove_device(reinterpret_cast<HANDLE>(lparam));
            break;
        case WM_MOUSEMOVE:
            update_pointer_position(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
            break;
        case WM_KILLFOCUS:
            input_->release_all();
            break;
        default:
            break;
    }
}

void windows_input_backend::enumerate_devices()
{
    UINT count = 0;
    if (GetRawInputDeviceList(nullptr, &count, sizeof(RAWINPUTDEVICELIST)) == static_cast<UINT>(-1) || count == 0)
        return;

    std::vector<RAWINPUTDEVICELIST> devices(count);
    if (GetRawInputDeviceList(devices.data(), &count, sizeof(RAWINPUTDEVICELIST)) == static_cast<UINT>(-1)) return;

    for (UINT index = 0; index < count; ++index)
    {
        if (devices[index].dwType == RIM_TYPEKEYBOARD || devices[index].dwType == RIM_TYPEMOUSE)
            register_device(devices[index].hDevice);
    }
}

void windows_input_backend::register_device(HANDLE native_device)
{
    RID_DEVICE_INFO info{};
    info.cbSize = sizeof(info);
    UINT info_size = sizeof(info);
    if (GetRawInputDeviceInfoW(native_device, RIDI_DEVICEINFO, &info, &info_size) == static_cast<UINT>(-1)) return;

    input::input_device_type type = input::input_device_type::unknown;
    input::input_device_capabilities capabilities{};
    std::string name;

    if (info.dwType == RIM_TYPEKEYBOARD)
    {
        type = input::input_device_type::keyboard;
        name = "Raw Input Keyboard";
        capabilities.buttons = true;
        capabilities.button_count = static_cast<std::uint16_t>(std::min<DWORD>(
            info.keyboard.dwNumberOfKeysTotal, static_cast<DWORD>(std::numeric_limits<std::uint16_t>::max())));
    }
    else if (info.dwType == RIM_TYPEMOUSE)
    {
        type = input::input_device_type::mouse;
        name = "Raw Input Mouse";
        capabilities.buttons = true;
        capabilities.axes = true;
        capabilities.pointer = true;
        capabilities.scroll = true;
        capabilities.button_count = static_cast<std::uint16_t>(std::min<DWORD>(
            info.mouse.dwNumberOfButtons, static_cast<DWORD>(std::numeric_limits<std::uint16_t>::max())));
        capabilities.axis_count = 6;
    }
    else
    {
        return;
    }

    const std::wstring path = device_path(native_device);
    const input::input_device_id id = stable_device_id(native_device, path, type);
    devices_[native_device] = id;
    input_->connect_device({.id = id,
                            .type = type,
                            .connectivity = connectivity_from_path(path),
                            .name = std::move(name),
                            .capabilities = capabilities});
}

void windows_input_backend::remove_device(HANDLE native_device)
{
    const auto found = devices_.find(native_device);
    if (found == devices_.end()) return;

    input_->disconnect_device(found->second);
    devices_.erase(found);
}

void windows_input_backend::handle_raw_input(HRAWINPUT raw_input)
{
    UINT size = 0;
    if (GetRawInputData(raw_input, RID_INPUT, nullptr, &size, sizeof(RAWINPUTHEADER)) == static_cast<UINT>(-1) ||
        size == 0)
        return;

    std::vector<std::byte> storage(size);
    if (GetRawInputData(raw_input, RID_INPUT, storage.data(), &size, sizeof(RAWINPUTHEADER)) != size) return;

    const auto* raw = reinterpret_cast<const RAWINPUT*>(storage.data());
    const input::input_device_id id = device_id(raw->header.hDevice);
    if (!id) return;

    if (raw->header.dwType == RIM_TYPEKEYBOARD)
        handle_keyboard(id, raw->data.keyboard);
    else if (raw->header.dwType == RIM_TYPEMOUSE)
        handle_mouse(id, raw->data.mouse);
}

void windows_input_backend::handle_keyboard(input::input_device_id device, const RAWKEYBOARD& keyboard)
{
    const input::key value = translate_key(keyboard);
    if (value == input::key::unknown) return;

    const bool down = (keyboard.Flags & RI_KEY_BREAK) == 0;
    input_->submit_button(device, input::make_key_control(value), down);
}

void windows_input_backend::handle_mouse(input::input_device_id device, const RAWMOUSE& mouse)
{
    if ((mouse.usFlags & MOUSE_MOVE_ABSOLUTE) == 0)
    {
        if (mouse.lLastX != 0)
            input_->submit_axis(device, input::make_mouse_axis_control(input::mouse_axis::delta_x),
                                static_cast<float>(mouse.lLastX));
        if (mouse.lLastY != 0)
            input_->submit_axis(device, input::make_mouse_axis_control(input::mouse_axis::delta_y),
                                static_cast<float>(mouse.lLastY));
    }

    const USHORT flags = mouse.usButtonFlags;
    submit_mouse_button(*input_, device, flags, RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_LEFT_BUTTON_UP,
                        input::mouse_button::left);
    submit_mouse_button(*input_, device, flags, RI_MOUSE_RIGHT_BUTTON_DOWN, RI_MOUSE_RIGHT_BUTTON_UP,
                        input::mouse_button::right);
    submit_mouse_button(*input_, device, flags, RI_MOUSE_MIDDLE_BUTTON_DOWN, RI_MOUSE_MIDDLE_BUTTON_UP,
                        input::mouse_button::middle);
    submit_mouse_button(*input_, device, flags, RI_MOUSE_BUTTON_4_DOWN, RI_MOUSE_BUTTON_4_UP, input::mouse_button::x1);
    submit_mouse_button(*input_, device, flags, RI_MOUSE_BUTTON_5_DOWN, RI_MOUSE_BUTTON_5_UP, input::mouse_button::x2);

    if ((flags & RI_MOUSE_WHEEL) != 0)
    {
        const float value =
            static_cast<float>(static_cast<SHORT>(mouse.usButtonData)) / static_cast<float>(WHEEL_DELTA);
        input_->submit_axis(device, input::make_mouse_axis_control(input::mouse_axis::wheel_y), value);
    }
    if ((flags & RI_MOUSE_HWHEEL) != 0)
    {
        const float value =
            static_cast<float>(static_cast<SHORT>(mouse.usButtonData)) / static_cast<float>(WHEEL_DELTA);
        input_->submit_axis(device, input::make_mouse_axis_control(input::mouse_axis::wheel_x), value);
    }
}

void windows_input_backend::update_pointer_position(int x, int y)
{
    const input::input_device_id mouse = primary_device(input::input_device_type::mouse);
    if (!mouse) return;

    input_->submit_axis(mouse, input::make_mouse_axis_control(input::mouse_axis::position_x), static_cast<float>(x));
    input_->submit_axis(mouse, input::make_mouse_axis_control(input::mouse_axis::position_y), static_cast<float>(y));
}

input::input_device_id windows_input_backend::device_id(HANDLE native_device)
{
    const auto found = devices_.find(native_device);
    if (found != devices_.end()) return found->second;

    register_device(native_device);
    const auto registered = devices_.find(native_device);
    return registered == devices_.end() ? input::input_device_id{} : registered->second;
}

input::input_device_id windows_input_backend::primary_device(input::input_device_type type) const
{
    const std::vector<input::input_device_id> devices = input_->devices(type);
    return devices.empty() ? input::input_device_id{} : devices.front();
}

std::wstring windows_input_backend::device_path(HANDLE native_device)
{
    UINT characters = 0;
    if (GetRawInputDeviceInfoW(native_device, RIDI_DEVICENAME, nullptr, &characters) == static_cast<UINT>(-1) ||
        characters == 0)
        return {};

    std::wstring result(characters, L'\0');
    if (GetRawInputDeviceInfoW(native_device, RIDI_DEVICENAME, result.data(), &characters) == static_cast<UINT>(-1))
        return {};

    while (!result.empty() && result.back() == L'\0')
        result.pop_back();
    return result;
}

input::input_connectivity_type windows_input_backend::connectivity_from_path(std::wstring_view path) noexcept
{
    std::wstring normalized(path);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](wchar_t value) { return static_cast<wchar_t>(std::towupper(value)); });

    if (normalized.find(L"BTH") != std::wstring::npos || normalized.find(L"BLUETOOTH") != std::wstring::npos)
        return input::input_connectivity_type::wireless;
    if (normalized.find(L"ACPI") != std::wstring::npos || normalized.find(L"I8042") != std::wstring::npos ||
        normalized.find(L"ROOT") != std::wstring::npos)
        return input::input_connectivity_type::builtin;
    if (normalized.find(L"USB") != std::wstring::npos || normalized.find(L"HID") != std::wstring::npos)
        return input::input_connectivity_type::usb;
    return input::input_connectivity_type::unknown;
}

input::input_device_id windows_input_backend::stable_device_id(HANDLE native_device, std::wstring_view path,
                                                               input::input_device_type type) noexcept
{
    std::uint64_t hash = fnv_offset_basis;
    if (!path.empty())
    {
        for (wchar_t value : path)
        {
            hash ^= static_cast<std::uint16_t>(value);
            hash *= fnv_prime;
        }
    }
    else
    {
        const auto value = reinterpret_cast<std::uintptr_t>(native_device);
        for (std::size_t index = 0; index < sizeof(value); ++index)
        {
            hash ^= static_cast<std::uint8_t>((value >> (index * 8U)) & 0xffU);
            hash *= fnv_prime;
        }
    }

    hash ^= static_cast<std::uint8_t>(type);
    hash *= fnv_prime;
    if (hash == 0) hash = 1;
    return {.value = hash};
}

input::key windows_input_backend::translate_key(const RAWKEYBOARD& keyboard) noexcept
{
    USHORT virtual_key = keyboard.VKey;
    if (virtual_key == 0 || virtual_key == 0xff) return input::key::unknown;

    if (virtual_key == VK_SHIFT)
        virtual_key = static_cast<USHORT>(MapVirtualKeyW(keyboard.MakeCode, MAPVK_VSC_TO_VK_EX));
    else if (virtual_key == VK_CONTROL)
        virtual_key = (keyboard.Flags & RI_KEY_E0) != 0 ? VK_RCONTROL : VK_LCONTROL;
    else if (virtual_key == VK_MENU)
        virtual_key = (keyboard.Flags & RI_KEY_E0) != 0 ? VK_RMENU : VK_LMENU;

    if (virtual_key >= 'A' && virtual_key <= 'Z')
        return static_cast<input::key>(static_cast<std::uint16_t>(input::key::a) + (virtual_key - 'A'));
    if (virtual_key >= '0' && virtual_key <= '9')
        return static_cast<input::key>(static_cast<std::uint16_t>(input::key::num0) + (virtual_key - '0'));

    switch (virtual_key)
    {
        case VK_ESCAPE:
            return input::key::escape;
        case VK_SPACE:
            return input::key::space;
        case VK_RETURN:
            return input::key::enter;
        case VK_TAB:
            return input::key::tab;
        case VK_BACK:
            return input::key::backspace;
        case VK_LSHIFT:
            return input::key::left_shift;
        case VK_RSHIFT:
            return input::key::right_shift;
        case VK_LCONTROL:
            return input::key::left_control;
        case VK_RCONTROL:
            return input::key::right_control;
        case VK_LMENU:
            return input::key::left_alt;
        case VK_RMENU:
            return input::key::right_alt;
        case VK_LEFT:
            return input::key::left;
        case VK_RIGHT:
            return input::key::right;
        case VK_UP:
            return input::key::up;
        case VK_DOWN:
            return input::key::down;
        case VK_INSERT:
            return input::key::insert;
        case VK_DELETE:
            return input::key::delete_key;
        case VK_HOME:
            return input::key::home;
        case VK_END:
            return input::key::end;
        case VK_PRIOR:
            return input::key::page_up;
        case VK_NEXT:
            return input::key::page_down;
        case VK_F1:
            return input::key::f1;
        case VK_F2:
            return input::key::f2;
        case VK_F3:
            return input::key::f3;
        case VK_F4:
            return input::key::f4;
        case VK_F5:
            return input::key::f5;
        case VK_F6:
            return input::key::f6;
        case VK_F7:
            return input::key::f7;
        case VK_F8:
            return input::key::f8;
        case VK_F9:
            return input::key::f9;
        case VK_F10:
            return input::key::f10;
        case VK_F11:
            return input::key::f11;
        case VK_F12:
            return input::key::f12;
        default:
            return input::key::unknown;
    }
}

} // namespace arc::platform::windows
