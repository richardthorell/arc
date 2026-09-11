#include "windows_gamepad_backend.h"

#include <arc/input/gamepad.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace arc::platform::windows
{
namespace
{

constexpr std::uint64_t xinput_device_namespace = 0x58494e5000000000ull;

float normalize_stick(SHORT value, SHORT deadzone) noexcept
{
    const int signed_value = static_cast<int>(value);
    const int magnitude = std::abs(signed_value);
    if (magnitude <= static_cast<int>(deadzone)) return 0.0f;

    const int maximum = signed_value < 0 ? 32768 : 32767;
    const float normalized = static_cast<float>(magnitude - deadzone) / static_cast<float>(maximum - deadzone);
    return std::clamp(normalized, 0.0f, 1.0f) * (signed_value < 0 ? -1.0f : 1.0f);
}

float normalize_trigger(BYTE value) noexcept
{
    if (value <= XINPUT_GAMEPAD_TRIGGER_THRESHOLD) return 0.0f;

    const float normalized = static_cast<float>(value - XINPUT_GAMEPAD_TRIGGER_THRESHOLD) /
                             static_cast<float>(255 - XINPUT_GAMEPAD_TRIGGER_THRESHOLD);
    return std::clamp(normalized, 0.0f, 1.0f);
}

void submit_button(input::input_system& system, input::input_device_id device, WORD buttons, WORD mask,
                   input::gamepad_button button)
{
    system.submit_button(device, input::make_gamepad_button_control(button), (buttons & mask) != 0);
}

template <class Procedure> Procedure load_procedure(HMODULE module, const char* name) noexcept
{
    Procedure result{};
    if (!module) return result;

    const FARPROC source = GetProcAddress(module, name);
    if (!source) return result;

    static_assert(sizeof(result) == sizeof(source));
    std::memcpy(&result, &source, sizeof(result));
    return result;
}

WORD motor_speed(float value) noexcept
{
    constexpr float maximum = 65535.0f;
    return static_cast<WORD>(std::clamp(value, 0.0f, 1.0f) * maximum);
}

} // namespace

windows_gamepad_backend::windows_gamepad_backend(input::input_system& input) noexcept : input_(&input)
{
    module_ = load_xinput();
    get_state_ = load_procedure<get_state_fn>(module_, "XInputGetState");
    set_state_ = load_procedure<set_state_fn>(module_, "XInputSetState");
}

windows_gamepad_backend::~windows_gamepad_backend()
{
    for (input::input_device_id device : devices_)
    {
        if (!device) continue;
        if (set_state_) set_rumble(device, {});
        input_->unregister_output_sink(device, *this);
    }

    if (module_) FreeLibrary(module_);
}

bool windows_gamepad_backend::available() const noexcept
{
    return get_state_ != nullptr;
}

void windows_gamepad_backend::poll()
{
    if (!get_state_) return;

    for (DWORD user_index = 0; user_index < XUSER_MAX_COUNT; ++user_index)
    {
        XINPUT_STATE state{};
        const DWORD result = get_state_(user_index, &state);
        input::input_device_id device = devices_[user_index];
        const input::input_device* record = device ? input_->device(device) : nullptr;

        if (result == ERROR_SUCCESS)
        {
            if (!record || !record->connected())
            {
                connect(user_index);
                device = devices_[user_index];
            }
            submit_state(user_index, state.Gamepad);
        }
        else if (result == ERROR_DEVICE_NOT_CONNECTED && record && record->connected())
        {
            disconnect(user_index);
        }
    }
}

bool windows_gamepad_backend::set_rumble(input::input_device_id device, input::input_rumble_state state)
{
    if (!set_state_) return false;

    const DWORD index = user_index(device);
    if (index >= XUSER_MAX_COUNT) return false;

    XINPUT_VIBRATION vibration{};
    vibration.wLeftMotorSpeed = motor_speed(state.low_frequency);
    vibration.wRightMotorSpeed = motor_speed(state.high_frequency);
    return set_state_(index, &vibration) == ERROR_SUCCESS;
}

HMODULE windows_gamepad_backend::load_xinput() noexcept
{
    constexpr const wchar_t* libraries[]{L"xinput1_4.dll", L"xinput1_3.dll", L"xinput9_1_0.dll"};
    for (const wchar_t* library : libraries)
    {
        if (HMODULE module = LoadLibraryW(library)) return module;
    }
    return nullptr;
}

input::input_device_id windows_gamepad_backend::stable_device_id(DWORD user_index) noexcept
{
    return {.value = xinput_device_namespace | (static_cast<std::uint64_t>(user_index) + 1ull)};
}

DWORD windows_gamepad_backend::user_index(input::input_device_id device) const noexcept
{
    for (DWORD index = 0; index < XUSER_MAX_COUNT; ++index)
    {
        if (devices_[index] == device) return index;
    }
    return XUSER_MAX_COUNT;
}

void windows_gamepad_backend::connect(DWORD user_index)
{
    const input::input_device_id device = stable_device_id(user_index);
    devices_[user_index] = device;
    input_->connect_device(
        {.id = device,
         .type = input::input_device_type::gamepad,
         .connectivity = input::input_connectivity_type::unknown,
         .name = "XInput Gamepad " + std::to_string(user_index + 1),
         .capabilities = {
             .buttons = true, .axes = true, .rumble = set_state_ != nullptr, .button_count = 14, .axis_count = 6}});

    if (set_state_) input_->register_output_sink(device, *this);
}

void windows_gamepad_backend::disconnect(DWORD user_index)
{
    const input::input_device_id device = devices_[user_index];
    if (device) input_->disconnect_device(device);
}

void windows_gamepad_backend::submit_state(DWORD user_index, const XINPUT_GAMEPAD& state)
{
    const input::input_device_id device = devices_[user_index];
    if (!device) return;

    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_A, input::gamepad_button::south);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_B, input::gamepad_button::east);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_X, input::gamepad_button::west);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_Y, input::gamepad_button::north);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_DPAD_UP, input::gamepad_button::dpad_up);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_DPAD_DOWN, input::gamepad_button::dpad_down);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_DPAD_LEFT, input::gamepad_button::dpad_left);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_DPAD_RIGHT, input::gamepad_button::dpad_right);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_LEFT_SHOULDER, input::gamepad_button::left_shoulder);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_RIGHT_SHOULDER,
                  input::gamepad_button::right_shoulder);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_LEFT_THUMB, input::gamepad_button::left_stick);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_RIGHT_THUMB, input::gamepad_button::right_stick);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_BACK, input::gamepad_button::view);
    submit_button(*input_, device, state.wButtons, XINPUT_GAMEPAD_START, input::gamepad_button::menu);

    input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::left_x),
                        normalize_stick(state.sThumbLX, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE));
    input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::left_y),
                        normalize_stick(state.sThumbLY, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE));
    input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::right_x),
                        normalize_stick(state.sThumbRX, XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE));
    input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::right_y),
                        normalize_stick(state.sThumbRY, XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE));
    input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::left_trigger),
                        normalize_trigger(state.bLeftTrigger));
    input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::right_trigger),
                        normalize_trigger(state.bRightTrigger));
}

} // namespace arc::platform::windows
