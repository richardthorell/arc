#pragma once

#include <arc/input/input.h>

#include <windows.h>

#include <string>
#include <string_view>
#include <unordered_map>

namespace arc::platform::windows
{

/**
 * @brief Win32 Raw Input adapter feeding normalized keyboard and mouse state into ARC input.
 */
class windows_input_backend final
{
public:
    explicit windows_input_backend(input::input_system& input) noexcept;

    [[nodiscard]] bool attach(HWND window);
    void process_message(UINT message, WPARAM wparam, LPARAM lparam);

private:
    void enumerate_devices();
    void register_device(HANDLE native_device);
    void remove_device(HANDLE native_device);
    void handle_raw_input(HRAWINPUT raw_input);
    void handle_keyboard(input::input_device_id device, const RAWKEYBOARD& keyboard);
    void handle_mouse(input::input_device_id device, const RAWMOUSE& mouse);
    void update_pointer_position(int x, int y);

    [[nodiscard]] input::input_device_id device_id(HANDLE native_device);
    [[nodiscard]] input::input_device_id primary_device(input::input_device_type type) const;
    [[nodiscard]] static std::wstring device_path(HANDLE native_device);
    [[nodiscard]] static input::input_connectivity_type connectivity_from_path(std::wstring_view path) noexcept;
    [[nodiscard]] static input::input_device_id stable_device_id(HANDLE native_device, std::wstring_view path,
                                                                 input::input_device_type type) noexcept;
    [[nodiscard]] static input::key translate_key(const RAWKEYBOARD& keyboard) noexcept;

    input::input_system* input_{};
    HWND window_{};
    std::unordered_map<HANDLE, input::input_device_id> devices_;
};

} // namespace arc::platform::windows
