#include <arc/input/input.h>

#include <limits>

namespace
{

class test_output_sink final : public arc::input::input_output_sink
{
public:
    bool set_rumble(arc::input::input_device_id device, arc::input::input_rumble_state state) override
    {
        last_device = device;
        last_state = state;
        ++calls;
        return true;
    }

    arc::input::input_device_id last_device{};
    arc::input::input_rumble_state last_state{};
    int calls{};
};

} // namespace

int main()
{
    using namespace arc::input;

    input_system input;
    const input_device_id controller = input.connect_device({
        .id = {.value = 400},
        .type = input_device_type::gamepad,
        .subtype = input_device_subtype::standard_gamepad,
        .backend = input_backend_type::game_input,
        .name = "Four Motor Controller",
        .capabilities = {.rumble = true, .trigger_rumble = true},
    });

    const input_device* device = input.device(controller);
    if (!device || !device->capabilities().rumble || !device->capabilities().trigger_rumble) return 1;

    test_output_sink output;
    if (!input.register_output_sink(controller, output)) return 2;
    if (!input.assign_device(1, controller)) return 3;

    input_player& player = input.player(1);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (!player.set_rumble(
            {.low_frequency = -0.25f, .high_frequency = 1.25f, .left_trigger = nan, .right_trigger = 0.75f}))
        return 4;

    if (output.last_device != controller) return 5;
    if (output.last_state.low_frequency != 0.0f) return 6;
    if (output.last_state.high_frequency != 1.0f) return 7;
    if (output.last_state.left_trigger != 0.0f) return 8;
    if (output.last_state.right_trigger != 0.75f) return 9;

    if (!input.set_rumble(controller,
                          {.low_frequency = 0.1f, .high_frequency = 0.2f, .left_trigger = 0.3f, .right_trigger = 2.0f}))
        return 10;
    if (output.last_state.low_frequency != 0.1f || output.last_state.high_frequency != 0.2f ||
        output.last_state.left_trigger != 0.3f || output.last_state.right_trigger != 1.0f)
        return 11;

    if (!player.stop_rumble()) return 12;
    if (output.last_state != input_rumble_state{}) return 13;

    if (!input.set_rumble(controller,
                          {.low_frequency = 0.4f, .high_frequency = 0.5f, .left_trigger = 0.6f, .right_trigger = 0.7f}))
        return 14;
    const int calls_before_disconnect = output.calls;
    if (!input.disconnect_device(controller)) return 15;
    if (output.calls != calls_before_disconnect + 1) return 16;
    if (output.last_state != input_rumble_state{}) return 17;

    return 0;
}
