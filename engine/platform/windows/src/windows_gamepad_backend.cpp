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
constexpr ULONGLONG disconnected_probe_interval_ms = 500;
constexpr ULONGLONG battery_poll_interval_ms = 5000;

struct xinput_device_classification
{
    input::input_device_type type{input::input_device_type::gamepad};
    input::input_device_subtype subtype{input::input_device_subtype::standard_gamepad};
    const char* name{"XInput Gamepad"};
};

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

input::input_battery_state normalized_battery_state(const XINPUT_BATTERY_INFORMATION& battery) noexcept
{
    input::input_battery_state state{};
    switch (battery.BatteryType)
    {
        case BATTERY_TYPE_DISCONNECTED:
        case BATTERY_TYPE_WIRED:
            state.status = input::input_battery_status::not_present;
            return state;
        case BATTERY_TYPE_ALKALINE:
        case BATTERY_TYPE_NIMH:
            state.status = input::input_battery_status::discharging;
            break;
        case BATTERY_TYPE_UNKNOWN:
        default:
            state.status = input::input_battery_status::unknown;
            break;
    }

    state.level_available = true;
    switch (battery.BatteryLevel)
    {
        case BATTERY_LEVEL_EMPTY:
            state.level = 0.0f;
            break;
        case BATTERY_LEVEL_LOW:
            state.level = 1.0f / 3.0f;
            break;
        case BATTERY_LEVEL_MEDIUM:
            state.level = 2.0f / 3.0f;
            break;
        case BATTERY_LEVEL_FULL:
            state.level = 1.0f;
            break;
        default:
            state.level = 0.0f;
            state.level_available = false;
            break;
    }
    return state;
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

xinput_device_classification classify_device(BYTE subtype) noexcept
{
    switch (subtype)
    {
        case XINPUT_DEVSUBTYPE_WHEEL:
            return {.type = input::input_device_type::wheel,
                    .subtype = input::input_device_subtype::wheel,
                    .name = "XInput Wheel"};
        case XINPUT_DEVSUBTYPE_FLIGHT_STICK:
            return {.type = input::input_device_type::flight_stick,
                    .subtype = input::input_device_subtype::flight_stick,
                    .name = "XInput Flight Stick"};
        case XINPUT_DEVSUBTYPE_ARCADE_STICK:
            return {.subtype = input::input_device_subtype::arcade_stick, .name = "XInput Arcade Stick"};
        case XINPUT_DEVSUBTYPE_DANCE_PAD:
            return {.subtype = input::input_device_subtype::dance_pad, .name = "XInput Dance Pad"};
        case XINPUT_DEVSUBTYPE_GUITAR:
            return {.subtype = input::input_device_subtype::guitar, .name = "XInput Guitar"};
        case XINPUT_DEVSUBTYPE_GUITAR_ALTERNATE:
            return {.subtype = input::input_device_subtype::guitar, .name = "XInput Guitar Alternate"};
        case XINPUT_DEVSUBTYPE_DRUM_KIT:
            return {.subtype = input::input_device_subtype::drum_kit, .name = "XInput Drum Kit"};
        case XINPUT_DEVSUBTYPE_GUITAR_BASS:
            return {.subtype = input::input_device_subtype::guitar, .name = "XInput Bass Guitar"};
        case XINPUT_DEVSUBTYPE_ARCADE_PAD:
            return {.subtype = input::input_device_subtype::arcade_pad, .name = "XInput Arcade Pad"};
        case XINPUT_DEVSUBTYPE_GAMEPAD:
        default:
            return {};
    }
}

} // namespace

windows_gamepad_backend::windows_gamepad_backend(input::input_system& input) noexcept : input_(&input)
{
    module_ = load_xinput();
    get_state_ = load_procedure<get_state_fn>(module_, "XInputGetState");
    set_state_ = load_procedure<set_state_fn>(module_, "XInputSetState");
    get_capabilities_ = load_procedure<get_capabilities_fn>(module_, "XInputGetCapabilities");
    get_battery_information_ = load_procedure<get_battery_information_fn>(module_, "XInputGetBatteryInformation");
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

    const ULONGLONG now = GetTickCount64();
    for (DWORD user_index = 0; user_index < XUSER_MAX_COUNT; ++user_index)
    {
        input::input_device_id device = devices_[user_index];
        const input::input_device* record = device ? input_->device(device) : nullptr;
        const bool connected = record && record->connected();

        if (!connected && next_probe_ticks_[user_index] != 0 && now < next_probe_ticks_[user_index]) continue;

        XINPUT_STATE state{};
        const DWORD result = get_state_(user_index, &state);
        if (result == ERROR_SUCCESS)
        {
            const bool newly_connected = !connected;
            if (newly_connected)
            {
                connect(user_index);
                device = devices_[user_index];
            }

            next_probe_ticks_[user_index] = 0;
            if (newly_connected || !packet_valid_[user_index] || packet_numbers_[user_index] != state.dwPacketNumber)
                submit_state(user_index, state.Gamepad);

            packet_numbers_[user_index] = state.dwPacketNumber;
            packet_valid_[user_index] = true;

            if (get_battery_information_ && now >= next_battery_ticks_[user_index])
            {
                submit_battery_state(user_index);
                next_battery_ticks_[user_index] = now + battery_poll_interval_ms;
            }
        }
        else if (result == ERROR_DEVICE_NOT_CONNECTED)
        {
            if (connected) disconnect(user_index);
            packet_valid_[user_index] = false;
            next_probe_ticks_[user_index] = now + disconnected_probe_interval_ms;
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
    XINPUT_CAPABILITIES native_capabilities{};
    const bool has_capabilities =
        get_capabilities_ && get_capabilities_(user_index, XINPUT_FLAG_GAMEPAD, &native_capabilities) == ERROR_SUCCESS;
    const xinput_device_classification classification =
        classify_device(has_capabilities ? native_capabilities.SubType : XINPUT_DEVSUBTYPE_GAMEPAD);

    const bool wireless = has_capabilities && (native_capabilities.Flags & XINPUT_CAPS_WIRELESS) != 0;
    const bool force_feedback = !has_capabilities || (native_capabilities.Flags & XINPUT_CAPS_FFB_SUPPORTED) != 0;

    XINPUT_BATTERY_INFORMATION native_battery{};
    const bool has_battery_state =
        get_battery_information_ &&
        get_battery_information_(user_index, BATTERY_DEVTYPE_GAMEPAD, &native_battery) == ERROR_SUCCESS;
    const input::input_battery_state battery =
        has_battery_state ? normalized_battery_state(native_battery) : input::input_battery_state{};
    const bool supports_battery = has_battery_state && battery.status != input::input_battery_status::not_present;

    const input::input_device_id device = stable_device_id(user_index);
    devices_[user_index] = device;
    input_->connect_device(
        {.id = device,
         .type = classification.type,
         .subtype = classification.subtype,
         .connectivity = wireless ? input::input_connectivity_type::wireless : input::input_connectivity_type::unknown,
         .backend = input::input_backend_type::xinput,
         .backend_id = "xinput:" + std::to_string(user_index),
         .name = std::string(classification.name) + " " + std::to_string(user_index + 1),
         .capabilities = {.buttons = true,
                          .axes = true,
                          .rumble = set_state_ != nullptr && force_feedback,
                          .battery = supports_battery,
                          .button_count = 14,
                          .axis_count = 6}});

    if (has_battery_state) input_->submit_battery_state(device, battery);
    next_battery_ticks_[user_index] = GetTickCount64() + battery_poll_interval_ms;
    if (set_state_ && force_feedback) input_->register_output_sink(device, *this);
}

void windows_gamepad_backend::disconnect(DWORD user_index)
{
    const input::input_device_id device = devices_[user_index];
    if (device) input_->disconnect_device(device);
    packet_valid_[user_index] = false;
    next_battery_ticks_[user_index] = 0;
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

void windows_gamepad_backend::submit_battery_state(DWORD user_index)
{
    if (!get_battery_information_) return;

    const input::input_device_id device = devices_[user_index];
    if (!device) return;

    XINPUT_BATTERY_INFORMATION native_battery{};
    if (get_battery_information_(user_index, BATTERY_DEVTYPE_GAMEPAD, &native_battery) != ERROR_SUCCESS) return;
    input_->submit_battery_state(device, normalized_battery_state(native_battery));
}

} // namespace arc::platform::windows