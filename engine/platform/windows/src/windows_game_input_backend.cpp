#include "windows_game_input_backend.h"

#include <arc/input/gamepad.h>

#include <algorithm>
#include <cctype>
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

float normalized_axis(float value, float minimum, float maximum) noexcept
{
    return std::clamp(value, minimum, maximum);
}

float acceleration_meters_per_second_squared(float acceleration_g) noexcept
{
    return acceleration_g * standard_gravity_meters_per_second_squared;
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
                   GameInputGamepadButtons mask, input::gamepad_button button)
{
    system.submit_button(device, input::make_gamepad_button_control(button), (buttons & mask) != 0);
}

#endif

} // namespace

struct windows_game_input_backend::implementation
{
    explicit implementation(input::input_system& input) : input_(&input)
    {
#if ARC_WINDOWS_HAS_GAMEINPUT_V3
        if (FAILED(GameInputCreate(&game_input_)) || !game_input_) return;

        const HRESULT result = game_input_->RegisterDeviceCallback(
            nullptr, GameInputKindGamepad, GameInputDeviceConnected, GameInputBlockingEnumeration, this,
            &implementation::device_callback, &callback_token_);
        if (FAILED(result))
        {
            game_input_->Release();
            game_input_ = nullptr;
            return;
        }

        callback_registered_ = true;
        available_ = true;
#endif
    }

    ~implementation()
    {
#if ARC_WINDOWS_HAS_GAMEINPUT_V3
        if (game_input_ && callback_registered_) game_input_->UnregisterCallback(callback_token_);

        {
            std::scoped_lock lock(pending_mutex_);
            for (const pending_device_event& event : pending_events_)
                if (event.device) event.device->Release();
            pending_events_.clear();
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
        drain_device_events();

        for (auto& [native, record] : devices_)
        {
            (void)native;
            poll_gamepad(record);
            poll_sensors(record);
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

            GameInputRumbleParams params{};
            params.lowFrequency = std::clamp(state.low_frequency, 0.0f, 1.0f);
            params.highFrequency = std::clamp(state.high_frequency, 0.0f, 1.0f);
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

    struct device_record
    {
        IGameInputDevice* device{};
        input::input_device_id id{};
        std::uint64_t last_gamepad_timestamp{};
        std::uint64_t last_sensor_timestamp{};
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
        self.pending_events_.push_back(
            {.device = device, .connected = (current_status & GameInputDeviceConnected) != 0});
    }

    void drain_device_events()
    {
        std::vector<pending_device_event> events;
        {
            std::scoped_lock lock(pending_mutex_);
            events.swap(pending_events_);
        }

        for (pending_device_event& event : events)
        {
            if (event.connected)
                connect_device(event.device);
            else
                disconnect_device(event.device);

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
        const bool supports_rumble = info->supportedRumbleMotors != GameInputRumbleNone;
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
                                .capabilities = {.buttons = true,
                                                 .axes = true,
                                                 .rumble = supports_rumble,
                                                 .gyroscope = gyroscope,
                                                 .accelerometer = accelerometer,
                                                 .button_count = 14,
                                                 .axis_count = 6}});

        device->AddRef();
        devices_.emplace(
            device, device_record{.device = device, .id = id, .gyroscope = gyroscope, .accelerometer = accelerometer});
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
            if (reading->GetGamepadState(&state)) submit_gamepad_state(record.id, state);
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

    void submit_gamepad_state(input::input_device_id device, const GameInputGamepadState& state)
    {
        submit_button(*input_, device, state.buttons, GameInputGamepadA, input::gamepad_button::south);
        submit_button(*input_, device, state.buttons, GameInputGamepadB, input::gamepad_button::east);
        submit_button(*input_, device, state.buttons, GameInputGamepadX, input::gamepad_button::west);
        submit_button(*input_, device, state.buttons, GameInputGamepadY, input::gamepad_button::north);
        submit_button(*input_, device, state.buttons, GameInputGamepadDPadUp, input::gamepad_button::dpad_up);
        submit_button(*input_, device, state.buttons, GameInputGamepadDPadDown, input::gamepad_button::dpad_down);
        submit_button(*input_, device, state.buttons, GameInputGamepadDPadLeft, input::gamepad_button::dpad_left);
        submit_button(*input_, device, state.buttons, GameInputGamepadDPadRight, input::gamepad_button::dpad_right);
        submit_button(*input_, device, state.buttons, GameInputGamepadLeftShoulder,
                      input::gamepad_button::left_shoulder);
        submit_button(*input_, device, state.buttons, GameInputGamepadRightShoulder,
                      input::gamepad_button::right_shoulder);
        submit_button(*input_, device, state.buttons, GameInputGamepadLeftThumbstick,
                      input::gamepad_button::left_stick);
        submit_button(*input_, device, state.buttons, GameInputGamepadRightThumbstick,
                      input::gamepad_button::right_stick);
        submit_button(*input_, device, state.buttons, GameInputGamepadView, input::gamepad_button::view);
        submit_button(*input_, device, state.buttons, GameInputGamepadMenu, input::gamepad_button::menu);

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
    GameInputCallbackToken callback_token_{};
    bool callback_registered_{};
    bool available_{};
    std::mutex pending_mutex_;
    std::vector<pending_device_event> pending_events_;
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
