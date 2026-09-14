#include <arc/input/input.h>

#include <limits>

namespace
{

class fake_advanced_output_sink final : public arc::input::input_advanced_output_sink
{
public:
    bool set_light(arc::input::input_device_id device, arc::input::input_light_state state) override
    {
        light_device = device;
        light = state;
        ++light_calls;
        return accept_light;
    }

    bool set_adaptive_triggers(arc::input::input_device_id device,
                               arc::input::input_adaptive_trigger_state state) override
    {
        trigger_device = device;
        triggers = state;
        ++trigger_calls;
        return accept_triggers;
    }

    arc::input::input_device_id light_device{};
    arc::input::input_device_id trigger_device{};
    arc::input::input_light_state light{};
    arc::input::input_adaptive_trigger_state triggers{};
    int light_calls{};
    int trigger_calls{};
    bool accept_light{true};
    bool accept_triggers{true};
};

int require(bool condition, int code)
{
    return condition ? 0 : code;
}

} // namespace

int main()
{
    using namespace arc::input;

    input_system input;
    const input_device_id device{.value = 712};
    input.connect_device({.id = device,
                          .type = input_device_type::gamepad,
                          .name = "Advanced Output Test",
                          .capabilities = {.light = true, .adaptive_triggers = true}});

    fake_advanced_output_sink sink;
    if (const int error = require(input.register_advanced_output_sink(device, sink), 1)) return error;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (const int error = require(input.set_light(device, {.red = -1.0f, .green = 2.0f, .blue = nan}), 2)) return error;
    if (const int error = require(sink.light_calls == 1 && sink.light_device == device, 3)) return error;
    if (const int error = require(sink.light.red == 0.0f && sink.light.green == 1.0f && sink.light.blue == 0.0f, 4))
        return error;

    input_adaptive_trigger_state trigger_state{};
    trigger_state.left = {.type = input_adaptive_trigger_effect_type::resistance,
                          .start_position = -0.5f,
                          .end_position = 2.0f,
                          .strength = 1.5f,
                          .frequency_hz = nan};
    trigger_state.right = {.type = input_adaptive_trigger_effect_type::vibration,
                           .start_position = 0.8f,
                           .end_position = 0.2f,
                           .strength = -1.0f,
                           .frequency_hz = 500.0f};
    if (const int error = require(input.set_adaptive_triggers(device, trigger_state), 5)) return error;
    if (const int error = require(sink.trigger_calls == 1 && sink.trigger_device == device, 6)) return error;
    if (const int error =
            require(sink.triggers.left.start_position == 0.0f && sink.triggers.left.end_position == 1.0f &&
                        sink.triggers.left.strength == 1.0f && sink.triggers.left.frequency_hz == 0.0f,
                    7))
        return error;
    if (const int error =
            require(sink.triggers.right.start_position == 0.8f && sink.triggers.right.end_position == 0.8f &&
                        sink.triggers.right.strength == 0.0f && sink.triggers.right.frequency_hz == 255.0f,
                    8))
        return error;

    input.assign_device(3, device);
    if (const int error = require(input.player(3).set_light({.red = 0.25f, .green = 0.5f, .blue = 0.75f}), 9))
        return error;
    if (const int error = require(sink.light_calls == 2, 10)) return error;

    if (const int error = require(input.unregister_advanced_output_sink(device, sink), 11)) return error;
    if (const int error = require(!input.set_light(device, {}), 12)) return error;

    input.disconnect_device(device);
    if (const int error = require(!input.set_adaptive_triggers(device, {}), 13)) return error;

    return 0;
}
