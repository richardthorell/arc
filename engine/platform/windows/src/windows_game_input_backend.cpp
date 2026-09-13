#include "windows_game_input_backend.h"

#include <arc/input/gamepad.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef ARC_WINDOWS_HAS_GAMEINPUT
#define ARC_WINDOWS_HAS_GAMEINPUT 0
#endif

#if ARC_WINDOWS_HAS_GAMEINPUT
#include <GameInput.h>
#endif

#if ARC_WINDOWS_HAS_GAMEINPUT && defined(GAMEINPUT_API_VERSION) && GAMEINPUT_API_VERSION >= 3
#define ARC_WINDOWS_HAS_GAMEINPUT_V3 1
using namespace GameInput::v3;
#else
#define ARC_WINDOWS_HAS_GAMEINPUT_V3 0
#endif

namespace arc::platform::windows
{
namespace
{

#if ARC_WINDOWS_HAS_GAMEINPUT_V3

constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;
constexpr std::uint64_t game_input_namespace = 0x4749000000000000ull;
constexpr std::uint64_t device_payload_mask = 0x0000ffffffffffffull;
constexpr float standard_gravity_meters_per_second_squared = 9.80665f;
constexpr std::uint32_t fallback_gamepad_layout = 0x00003fffu;
constexpr auto battery_poll_interval = std::chrono::seconds(5);

float normalized_axis(float value, float minimum, float maximum) noexcept
{
    return std::clamp(value, minimum, maximum);
}

float acceleration_meters_per_second_squared(float acceleration_g) noexcept
{
    return acceleration_g * standard_gravity_meters_per_second_squared;
}

bool supports_rumble_motor(GameInputRumbleMotors supported, GameInputRumbleMotors motor) noexcept
{
    return (supported & motor) != GameInputRumbleNone;
}

input::input_battery_status battery_status(GameInputBatteryStatus status) noexcept
{
    switch (status)
    {
        case GameInputBatteryNotPresent:
            return input::input_battery_status::not_present;
        case GameInputBatteryDischarging:
            return input::input_battery_status::discharging;
        case GameInputBatteryIdle:
            return input::input_battery_status::idle;
        case GameInputBatteryCharging:
            return input::input_battery_status::charging;
        case GameInputBatteryUnknown:
        default:
            return input::input_battery_status::unknown;
    }
}

input::input_battery_state normalized_battery_state(const GameInputBatteryState& state) noexcept
{
    input::input_battery_state result{.status = battery_status(state.status)};
    if (std::isfinite(state.remainingCapacity) && std::isfinite(state.fullChargeCapacity) &&
        state.fullChargeCapacity > 0.0f && state.remainingCapacity >= 0.0f)
    {
        result.level = std::clamp(state.remainingCapacity / state.fullChargeCapacity, 0.0f, 1.0f);
        result.level_available = true;
    }
    return result;
}

std::uint16_t supported_button_count(GameInputGamepadButtons layout, GameInputSystemButtons system_buttons) noexcept
{
    const auto layout_bits = static_cast<std::uint32_t>(layout);
    const auto system_bits = static_cast<std::uint32_t>(system_buttons);
    return static_cast<std::uint16_t>(std::popcount(layout_bits) + std::popcount(system_bits));
}

input::input_connectivity_type connectivity_from_path(const char* path)
{
    if (!path || *path == '\0') return input::input_connectivity_type::unknown;

    std::string normalized(path);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char value) { return static_cast<char>(std::toupper(value)); });

    if (normalized.find("BTH") != std::string::npos || normalized.find("BLUETOOTH") != std::string::npos)
        return input::input_connectivity_type::wireless;
    if (normalized.find("USB") != std::string::npos || normalized.find("HID") != std::string::npos)
        return input::input_connectivity_type::usb;
    return input::input_connectivity_type::unknown;
}

input::input_device_id stable_device_id(const APP_LOCAL_DEVICE_ID& native_id) noexcept
{
    std::uint64_t hash = fnv_offset_basis;
    const auto* bytes = reinterpret_cast<const std::byte*>(&native_id);
    for (std::size_t index = 0; index < sizeof(native_id); ++index)
    {
        hash ^= std::to_integer<std::uint8_t>(bytes[index]);
        hash *= fnv_prime;
    }

    const std::uint64_t payload = hash & device_payload_mask;
    return {.value = game_input_namespace | (payload == 0 ? 1ull : payload)};
}

std::string backend_id(input::input_device_id id)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result = "gameinput:";
    result.reserve(result.size() + 16);
    for (int shift = 60; shift >= 0; shift -= 4)
        result.push_back(digits[(id.value >> static_cast<unsigned>(shift)) & 0x0full]);
    return result;
}

void submit_button(input::input_system& system, input::input_device_id device, GameInputGamepadButtons buttons,
                   GameInputGamepadButtons supported, GameInputGamepadButtons mask, input::gamepad_button button)
{
    if ((supported & mask) == GameInputGamepadNone) return;
    system.submit_button(device, input::make_gamepad_button_control(button), (buttons & mask) != GameInputGamepadNone);
}

#endif

} // namespace

struct windows_game_input_backend::implementation
{
    explicit implementation(input::input_system& input) : input_(&input)
    {
#if ARC_WINDOWS_HAS_GAMEINPUT_V3
        if (FAILED(GameInputCreate(&game_input_)) || !game_input_) return;

        const HRESULT device_result = game_input_->RegisterDeviceCallback(
            nullptr, GameInputKindGamepad, GameInputDeviceConnected, GameInputBlockingEnumeration, this,
            &implementation::device_callback, &device_callback_token_);
        if (FAILED(device_result))
        {
            game_input_->Release();
            game_input_ = nullptr;
            return;
        }

        device_callback_registered_ = true;
        const auto system_filter =
            static_cast<GameInputSystemButtons>(GameInputSystemButtonGuide | GameInputSystemButtonShare);
        const HRESULT system_result = game_input_->RegisterSystemButtonCallback(
            nullptr, system_filter, this, &implementation::system_button_callback, &system_button_callback_token_);
        system_button_callback_registered_ = SUCCEEDED(system_result);
        available_ = true;
#endif
    }

    ~implementation()
    {
#if ARC_WINDOWS_HAS_GAMEINPUT_V3
        if (game_input_ && system_button_callback_registered_)
            game_input_->UnregisterCallback(system_button_callback_token_);
        if (game_input_ && device_callback_registered_) game_input_->UnregisterCallback(device_callback_token_);

        {
            std::scoped_lock lock(pending_mutex_);
            for (const pending_device_event& event : pending_device_events_)
                if (event.device) event.device->Release();
            for (const pending_system_button_event& event : pending_system_button_events_)
                if (event.device) event.device->Release();
            pending_device_events_.clear();
            pending_system_button_events_.clear();
        }

        for (auto& [native, record] : devices_)
        {
            (void)native;
            if (owner_) input_->unregister_output_sink(record.id, *owner_);
            if (record.device)
            {
                GameInputRumbleParams stopped{};
                record.device->SetRumbleState(&stopped);
                record.device->Release();
            }
        }
        devices_.clear();

        if (game_input_) game_input_->Release();
#endif
    }

    [[nodiscard]] bool available() const noexcept
    {
        return available_;
    }

    void poll()
    {
#if ARC_WINDOWS_HAS_GAMEINPUT_V3
        if (!available_) return;
        drain_pending_events();

        const auto now = std::chrono::steady_clock::now();
        for (auto& [native, record] : devices_)
        {
            (void)native;
            poll_gamepad(record);
            poll_sensors(record);
            poll_battery(record, now);
        }
#endif
    }

    bool set_rumble(input::input_device_id device, input::input_rumble_state state)
    {
#if ARC_WINDOWS_HAS_GAMEINPUT_V3
        for (auto& [native, record] : devices_)
        {
            (void)native;
            if (record.id != device) continue;

            const GameInputRumbleMotors supported = record.supported_rumble_motors;
            GameInputRumbleParams params{};
            if (supports_rumble_motor(supported, GameInputRumbleLowFrequency))
                params.lowFrequency = std::clamp(state.low_frequency, 0.0f, 1.0f);
            if (supports_rumble_motor(supported, GameInputRumbleHighFrequency))
                params.highFrequency = std::clamp(state.high_frequency, 0.0f, 1.0f);
            if (supports_rumble_motor(supported, GameInputRumbleLeftTrigger))
                params.leftTrigger = std::clamp(state.left_trigger, 0.0f, 1.0f);
            if (supports_rumble_motor(supported, GameInputRumbleRightTrigger))
                params.rightTrigger = std::clamp(state.right_trigger, 0.0f, 1.0f);
            record.device->SetRumbleState(&params);
            return true;
        }
#else
        (void)device;
        (void)state;
#endif
        return false;
    }

    void set_owner(input::input_output_sink& owner) noexcept
    {
#if ARC_WINDOWS_HAS_GAMEINPUT_V3
        owner_ = &owner;
#else
        (void)owner;
#endif
    }

private:
#if ARC_WINDOWS_HAS_GAMEINPUT_V3
    struct pending_device_event
    {
        IGameInputDevice* device{};
        bool connected{};
    };

    struct pending_system_button_event
    {
        IGameInputDevice* device{};
        GameInputSystemButtons buttons{GameInputSystemButtonNone};
    };

    struct device_record
    {
        IGameInputDevice* device{};
        input::input_device_id id{};
        GameInputGamepadButtons supported_layout{GameInputGamepadNone};
        GameInputSystemButtons supported_system_buttons{GameInputSystemButtonNone};
        GameInputRumbleMotors supported_rumble_motors{GameInputRumbleNone};
        std::uint64_t last_gamepad_timestamp{};
        std::uint64_t last_sensor_timestamp{};
        std::chrono::steady_clock::time_point next_battery_poll{};
        bool gyroscope{};
        bool accelerometer{};
    };

    static void CALLBACK device_callback(GameInputCallbackToken, void* context, IGameInputDevice* device, std::uint64_t,
                                         GameInputDeviceStatus current_status, GameInputDeviceStatus)
    {
        if (!context || !device) return;
        auto& self = *static_cast<implementation*>(context);
        device->AddRef();
        std::scoped_lock lock(self.pending_mutex_);
        self.pending_device_events_.push_back(
            {.device = device, .connected = (current_status & GameInputDeviceConnected) != 0});
    }

    static void CALLBACK system_button_callback(GameInputCallbackToken, void* context, IGameInputDevice* device,
                                                std::uint64_t, GameInputSystemButtons current_buttons,
                                                GameInputSystemButtons)
    {
        if (!context || !device) return;
        auto& self = *static_cast<implementation*>(context);
        device->AddRef();
        std::scoped_lock lock(self.pending_mutex_);
        self.pending_system_button_events_.push_back({.device = device, .buttons = current_buttons});
    }

    void drain_pending_events()
    {
        std::vector<pending_device_event> device_events;
        std::vector<pending_system_button_event> system_button_events;
        {
            std::scoped_lock lock(pending_mutex_);
            device_events.swap(pending_device_events_);
            system_button_events.swap(pending_system_button_events_);
        }

        for (pending_device_event& event : device_events)
        {
            if (event.connected)
                connect_device(event.device);
            else
                disconnect_device(event.device);

            if (event.device) event.device->Release();
        }

        for (pending_system_button_event& event : system_button_events)
        {
            const auto found = devices_.find(event.device);
            if (found != devices_.end()) submit_system_button_state(found->second, event.buttons);
            if (event.device) event.device->Release();
        }
    }

    void connect_device(IGameInputDevice* device)
    {
        if (devices_.contains(device)) return;

        const GameInputDeviceInfo* info = nullptr;
        if (FAILED(device->GetDeviceInfo(&info)) || !info) return;

        const GameInputSensorsKind supported_sensors =
            info->sensorsInfo ? info->sensorsInfo->supportedSensors : GameInputSensorsNone;
        const bool gyroscope = (supported_sensors & GameInputSensorsGyrometer) != GameInputSensorsNone;
        const bool accelerometer = (supported_sensors & GameInputSensorsAccelerometer) != GameInputSensorsNone;
        const input::input_device_id id = stable_device_id(info->deviceId);
        const GameInputRumbleMotors supported_rumble_motors = info->supportedRumbleMotors;
        const bool supports_rumble = supported_rumble_motors != GameInputRumbleNone;
        const bool supports_trigger_rumble =
            supports_rumble_motor(supported_rumble_motors, GameInputRumbleLeftTrigger) ||
            supports_rumble_motor(supported_rumble_motors, GameInputRumbleRightTrigger);
        const GameInputGamepadButtons supported_layout =
            info->gamepadInfo ? info->gamepadInfo->supportedLayout
                              : static_cast<GameInputGamepadButtons>(fallback_gamepad_layout);
        const GameInputSystemButtons supported_system_buttons =
            system_button_callback_registered_ ? info->supportedSystemButtons : GameInputSystemButtonNone;
        const std::uint16_t button_count = supported_button_count(supported_layout, supported_system_buttons);
        GameInputBatteryState native_battery{};
        device->GetBatteryState(&native_battery);
        const input::input_battery_state initial_battery = normalized_battery_state(native_battery);
        const bool supports_battery = initial_battery.status != input::input_battery_status::not_present;
        std::string name = info->displayName && *info->displayName ? info->displayName : "GameInput Gamepad";
        input_->connect_device({.id = id,
                                .type = input::input_device_type::gamepad,
                                .subtype = input::input_device_subtype::standard_gamepad,
                                .connectivity = connectivity_from_path(info->pnpPath),
                                .backend = input::input_backend_type::game_input,
                                .hardware_id = {.vendor_id = info->vendorId,
                                                .product_id = info->productId,
                                                .version = info->revisionNumber},
                                .backend_id = backend_id(id),
                                .name = std::move(name),
                                .capabilities = {.buttons = button_count != 0,
                                                 .axes = true,
                                                 .rumble = supports_rumble,
                                                 .trigger_rumble = supports_trigger_rumble,
                                                 .gyroscope = gyroscope,
                                                 .accelerometer = accelerometer,
                                                 .battery = supports_battery,
                                                 .button_count = button_count,
                                                 .axis_count = 6}});

        device->AddRef();
        devices_.emplace(device,
                         device_record{.device = device,
                                       .id = id,
                                       .supported_layout = supported_layout,
                                       .supported_system_buttons = supported_system_buttons,
                                       .supported_rumble_motors = supported_rumble_motors,
                                       .next_battery_poll = std::chrono::steady_clock::now() + battery_poll_interval,
                                       .gyroscope = gyroscope,
                                       .accelerometer = accelerometer});
        input_->submit_battery_state(id, initial_battery);
        if (supports_rumble && owner_) input_->register_output_sink(id, *owner_);
    }

    void disconnect_device(IGameInputDevice* device)
    {
        const auto found = devices_.find(device);
        if (found == devices_.end()) return;

        input_->disconnect_device(found->second.id);
        if (owner_) input_->unregister_output_sink(found->second.id, *owner_);
        found->second.device->Release();
        devices_.erase(found);
    }

    void poll_gamepad(device_record& record)
    {
        IGameInputReading* reading = nullptr;
        if (FAILED(game_input_->GetCurrentReading(GameInputKindGamepad, record.device, &reading)) || !reading) return;

        const std::uint64_t timestamp = reading->GetTimestamp();
        if (timestamp != record.last_gamepad_timestamp)
        {
            GameInputGamepadState state{};
            if (reading->GetGamepadState(&state)) submit_gamepad_state(record, state);
            record.last_gamepad_timestamp = timestamp;
        }
        reading->Release();
    }

    void poll_sensors(device_record& record)
    {
        if (!record.gyroscope && !record.accelerometer) return;

        IGameInputReading* reading = nullptr;
        if (FAILED(game_input_->GetCurrentReading(GameInputKindSensors, record.device, &reading)) || !reading) return;

        const std::uint64_t timestamp = reading->GetTimestamp();
        if (timestamp != record.last_sensor_timestamp)
        {
            GameInputSensorsState state{};
            if (reading->GetSensorsState(&state)) submit_sensor_state(record, state);
            record.last_sensor_timestamp = timestamp;
        }
        reading->Release();
    }

    void poll_battery(device_record& record, std::chrono::steady_clock::time_point now)
    {
        if (now < record.next_battery_poll) return;

        GameInputBatteryState state{};
        record.device->GetBatteryState(&state);
        input_->submit_battery_state(record.id, normalized_battery_state(state));
        record.next_battery_poll = now + battery_poll_interval;
    }

    void submit_gamepad_state(const device_record& record, const GameInputGamepadState& state)
    {
        const input::input_device_id device = record.id;
        const GameInputGamepadButtons supported = record.supported_layout;
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadA, input::gamepad_button::south);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadB, input::gamepad_button::east);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadX, input::gamepad_button::west);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadY, input::gamepad_button::north);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadC, input::gamepad_button::auxiliary_1);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadZ, input::gamepad_button::auxiliary_2);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadDPadUp,
                      input::gamepad_button::dpad_up);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadDPadDown,
                      input::gamepad_button::dpad_down);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadDPadLeft,
                      input::gamepad_button::dpad_left);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadDPadRight,
                      input::gamepad_button::dpad_right);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadLeftShoulder,
                      input::gamepad_button::left_shoulder);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadRightShoulder,
                      input::gamepad_button::right_shoulder);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadLeftTriggerButton,
                      input::gamepad_button::left_trigger_button);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadRightTriggerButton,
                      input::gamepad_button::right_trigger_button);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadLeftThumbstick,
                      input::gamepad_button::left_stick);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadRightThumbstick,
                      input::gamepad_button::right_stick);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadLeftThumbstickUp,
                      input::gamepad_button::left_stick_up);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadLeftThumbstickDown,
                      input::gamepad_button::left_stick_down);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadLeftThumbstickLeft,
                      input::gamepad_button::left_stick_left);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadLeftThumbstickRight,
                      input::gamepad_button::left_stick_right);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadRightThumbstickUp,
                      input::gamepad_button::right_stick_up);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadRightThumbstickDown,
                      input::gamepad_button::right_stick_down);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadRightThumbstickLeft,
                      input::gamepad_button::right_stick_left);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadRightThumbstickRight,
                      input::gamepad_button::right_stick_right);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadPaddleLeft1,
                      input::gamepad_button::paddle_left_1);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadPaddleLeft2,
                      input::gamepad_button::paddle_left_2);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadPaddleRight1,
                      input::gamepad_button::paddle_right_1);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadPaddleRight2,
                      input::gamepad_button::paddle_right_2);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadView, input::gamepad_button::view);
        submit_button(*input_, device, state.buttons, supported, GameInputGamepadMenu, input::gamepad_button::menu);

        input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::left_x),
                            normalized_axis(state.leftThumbstickX, -1.0f, 1.0f));
        input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::left_y),
                            normalized_axis(state.leftThumbstickY, -1.0f, 1.0f));
        input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::right_x),
                            normalized_axis(state.rightThumbstickX, -1.0f, 1.0f));
        input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::right_y),
                            normalized_axis(state.rightThumbstickY, -1.0f, 1.0f));
        input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::left_trigger),
                            normalized_axis(state.leftTrigger, 0.0f, 1.0f));
        input_->submit_axis(device, input::make_gamepad_axis_control(input::gamepad_axis::right_trigger),
                            normalized_axis(state.rightTrigger, 0.0f, 1.0f));
    }

    void submit_system_button_state(const device_record& record, GameInputSystemButtons buttons)
    {
        if ((record.supported_system_buttons & GameInputSystemButtonGuide) != GameInputSystemButtonNone)
            input_->submit_button(record.id, input::make_gamepad_button_control(input::gamepad_button::guide),
                                  (buttons & GameInputSystemButtonGuide) != GameInputSystemButtonNone);
        if ((record.supported_system_buttons & GameInputSystemButtonShare) != GameInputSystemButtonNone)
            input_->submit_button(record.id, input::make_gamepad_button_control(input::gamepad_button::share),
                                  (buttons & GameInputSystemButtonShare) != GameInputSystemButtonNone);
    }

    void submit_sensor_state(const device_record& record, const GameInputSensorsState& state)
    {
        if (record.gyroscope)
        {
            input_->submit_axis(record.id, input::make_sensor_axis_control(input::sensor_axis::gyroscope_x),
                                state.angularVelocityInRadPerSecX);
            input_->submit_axis(record.id, input::make_sensor_axis_control(input::sensor_axis::gyroscope_y),
                                state.angularVelocityInRadPerSecY);
            input_->submit_axis(record.id, input::make_sensor_axis_control(input::sensor_axis::gyroscope_z),
                                state.angularVelocityInRadPerSecZ);
        }

        if (record.accelerometer)
        {
            input_->submit_axis(record.id, input::make_sensor_axis_control(input::sensor_axis::accelerometer_x),
                                acceleration_meters_per_second_squared(state.accelerationInGX));
            input_->submit_axis(record.id, input::make_sensor_axis_control(input::sensor_axis::accelerometer_y),
                                acceleration_meters_per_second_squared(state.accelerationInGY));
            input_->submit_axis(record.id, input::make_sensor_axis_control(input::sensor_axis::accelerometer_z),
                                acceleration_meters_per_second_squared(state.accelerationInGZ));
        }
    }

    input::input_system* input_{};
    input::input_output_sink* owner_{};
    IGameInput* game_input_{};
    GameInputCallbackToken device_callback_token_{};
    GameInputCallbackToken system_button_callback_token_{};
    bool device_callback_registered_{};
    bool system_button_callback_registered_{};
    bool available_{};
    std::mutex pending_mutex_;
    std::vector<pending_device_event> pending_device_events_;
    std::vector<pending_system_button_event> pending_system_button_events_;
    std::unordered_map<IGameInputDevice*, device_record> devices_;
#else
    input::input_system* input_{};
    bool available_{};
#endif
};

windows_game_input_backend::windows_game_input_backend(input::input_system& input)
    : impl_(std::make_unique<implementation>(input))
{
    impl_->set_owner(*this);
}

windows_game_input_backend::~windows_game_input_backend() = default;

bool windows_game_input_backend::available() const noexcept
{
    return impl_->available();
}

void windows_game_input_backend::poll()
{
    impl_->poll();
}

bool windows_game_input_backend::set_rumble(input::input_device_id device, input::input_rumble_state state)
{
    return impl_->set_rumble(device, state);
}

} // namespace arc::platform::windows