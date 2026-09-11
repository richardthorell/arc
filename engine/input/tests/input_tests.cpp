#include <arc/input/gamepad.h>
#include <arc/input/input.h>

#include <algorithm>
#include <cassert>

namespace
{

arc::input::input_binding key_binding(arc::input::key value)
{
    return {.device = arc::input::input_device_type::keyboard,
            .control = arc::input::make_key_control(value),
            .processors = {}};
}

arc::input::input_binding gamepad_button_binding(arc::input::gamepad_button value)
{
    return {.device = arc::input::input_device_type::gamepad,
            .control = arc::input::make_gamepad_button_control(value),
            .processors = {}};
}

arc::input::input_binding gamepad_axis_binding(arc::input::gamepad_axis value)
{
    return {.device = arc::input::input_device_type::gamepad,
            .control = arc::input::make_gamepad_axis_control(value),
            .processors = {}};
}

bool contains_device(const std::vector<arc::input::input_device_id>& devices, arc::input::input_device_id id)
{
    return std::find(devices.begin(), devices.end(), id) != devices.end();
}

class test_output_sink final : public arc::input::input_output_sink
{
public:
    bool set_rumble(arc::input::input_device_id device, arc::input::input_rumble_state state) override
    {
        last_device = device;
        last_state = state;
        ++calls;
        return accept;
    }

    arc::input::input_device_id last_device{};
    arc::input::input_rumble_state last_state{};
    int calls{};
    bool accept{true};
};

} // namespace

int main()
{
    using namespace arc::input;

    input_system input;
    const input_device_id keyboard = input.connect_device({
        .id = {.value = 100},
        .type = input_device_type::keyboard,
        .connectivity = input_connectivity_type::builtin,
        .name = "Test Keyboard",
        .capabilities = {.buttons = true, .button_count = 104},
    });
    const input_device_id mouse = input.connect_device({
        .id = {.value = 200},
        .type = input_device_type::mouse,
        .connectivity = input_connectivity_type::usb,
        .name = "Test Mouse",
        .capabilities = {.buttons = true, .axes = true, .pointer = true, .scroll = true, .button_count = 5},
    });
    const input_device_id controller = input.connect_device({
        .id = {.value = 300},
        .type = input_device_type::gamepad,
        .subtype = input_device_subtype::standard_gamepad,
        .connectivity = input_connectivity_type::wireless,
        .backend = input_backend_type::hid,
        .hardware_id = {.vendor_id = 0x1234, .product_id = 0xabcd, .version = 0x0002},
        .backend_id = "hid:test-controller-300",
        .name = "Future Controller",
        .capabilities = {.buttons = true, .axes = true, .rumble = true, .gyroscope = true, .accelerometer = true},
    });

    assert(input.device(keyboard));
    assert(input.device(keyboard)->connectivity() == input_connectivity_type::builtin);
    assert(input.device(mouse)->connectivity() == input_connectivity_type::usb);
    assert(input.device(controller)->capabilities().gyroscope);
    assert(input.device(controller)->subtype() == input_device_subtype::standard_gamepad);
    assert(input.device(controller)->backend() == input_backend_type::hid);
    const input_device_hardware_id controller_hardware = input.device(controller)->hardware_id();
    assert(controller_hardware.vendor_id == 0x1234);
    assert(controller_hardware.product_id == 0xabcd);
    assert(controller_hardware.version == 0x0002);
    assert(input.device(controller)->backend_id() == "hid:test-controller-300");
    assert(input.device_events().size() == 3);

    const auto player_zero_devices = input.devices_for_player(0);
    assert(contains_device(player_zero_devices, keyboard));
    assert(contains_device(player_zero_devices, mouse));
    assert(!contains_device(player_zero_devices, controller));

    input_player& player_zero = input.player(0);
    input_player& player_one = input.player(1);
    assert(input.assign_device(1, keyboard));
    assert(input.assign_device(1, controller));
    assert(input.players_for_device(keyboard).size() == 2);

    test_output_sink output;
    assert(input.register_output_sink(controller, output));
    assert(!input.set_rumble(keyboard, {.low_frequency = 1.0f, .high_frequency = 1.0f}));
    assert(!player_zero.set_rumble({.low_frequency = 1.0f, .high_frequency = 1.0f}));
    assert(player_one.set_rumble({.low_frequency = -0.25f, .high_frequency = 1.5f}));
    assert(output.last_device == controller);
    assert(output.last_state.low_frequency == 0.0f);
    assert(output.last_state.high_frequency == 1.0f);
    assert(player_one.stop_rumble());
    assert(output.last_state == input_rumble_state{});

    player_zero.add_context("gameplay", 0);
    player_zero.bind_action("gameplay", "jump", key_binding(key::space));
    player_zero.bind_axis2d("gameplay", "move", key_binding(key::d), {1.0f, 0.0f});
    player_zero.bind_axis2d("gameplay", "move", key_binding(key::a), {-1.0f, 0.0f});
    player_zero.bind_axis2d("gameplay", "move", key_binding(key::w), {0.0f, 1.0f});
    player_zero.bind_axis2d("gameplay", "move", key_binding(key::s), {0.0f, -1.0f});

    player_one.add_context("gameplay", 0);
    player_one.bind_action("gameplay", "jump", key_binding(key::enter));
    player_one.bind_action("gameplay", "accept", gamepad_button_binding(gamepad_button::south));
    player_one.bind_axis2d("gameplay", "move", gamepad_axis_binding(gamepad_axis::left_x), {1.0f, 0.0f});
    player_one.bind_axis2d("gameplay", "move", gamepad_axis_binding(gamepad_axis::left_y), {0.0f, 1.0f});

    input.begin_frame();
    assert(input.device_events().empty());
    assert(input.submit_button(controller, make_gamepad_button_control(gamepad_button::south), true));
    assert(input.submit_axis(controller, make_gamepad_axis_control(gamepad_axis::left_x), 0.75f));
    assert(input.submit_axis(controller, make_gamepad_axis_control(gamepad_axis::left_y), -0.25f));
    assert(player_one.pressed("accept"));
    assert(player_one.down("accept"));
    const arc::math::vector2f gamepad_move = player_one.axis2d("move");
    assert(gamepad_move[0] == 0.75f);
    assert(gamepad_move[1] == -0.25f);

    input.begin_frame();
    assert(!player_one.pressed("accept"));
    assert(player_one.down("accept"));
    assert(input.submit_button(controller, make_gamepad_button_control(gamepad_button::south), false));
    assert(player_one.released("accept"));

    input.begin_frame();
    assert(input.submit_button(keyboard, make_key_control(key::space), true));
    assert(player_zero.pressed("jump"));
    assert(player_zero.down("jump"));
    assert(!player_one.down("jump"));

    input.begin_frame();
    assert(!player_zero.pressed("jump"));
    assert(player_zero.down("jump"));
    assert(input.submit_button(keyboard, make_key_control(key::space), false));
    assert(player_zero.released("jump"));

    input.begin_frame();
    input.submit_button(keyboard, make_key_control(key::w), true);
    input.submit_button(keyboard, make_key_control(key::d), true);
    const arc::math::vector2f move = player_zero.axis2d("move");
    assert(move[0] == 1.0f);
    assert(move[1] == 1.0f);

    player_zero.add_context("ui", 100);
    player_zero.bind_action("ui", "jump", key_binding(key::enter));
    input.submit_button(keyboard, make_key_control(key::enter), true);
    assert(player_zero.down("jump"));
    input.submit_button(keyboard, make_key_control(key::enter), false);
    assert(!player_zero.down("jump"));
    assert(player_zero.set_context_enabled("ui", false));

    input.begin_frame();
    input.submit_axis(mouse, make_mouse_axis_control(mouse_axis::delta_x), 4.0f);
    input.submit_axis(mouse, make_mouse_axis_control(mouse_axis::delta_y), -2.0f);
    player_zero.bind_axis2d(
        "gameplay", "look",
        {.device = input_device_type::mouse, .control = make_mouse_axis_control(mouse_axis::delta_x), .processors = {}},
        {1.0f, 0.0f});
    player_zero.bind_axis2d(
        "gameplay", "look",
        {.device = input_device_type::mouse, .control = make_mouse_axis_control(mouse_axis::delta_y), .processors = {}},
        {0.0f, 1.0f});
    const arc::math::vector2f look = player_zero.axis2d("look");
    assert(look[0] == 4.0f);
    assert(look[1] == -2.0f);

    input.begin_frame();
    const arc::math::vector2f reset_look = player_zero.axis2d("look");
    assert(reset_look[0] == 0.0f);
    assert(reset_look[1] == 0.0f);

    input.submit_button(keyboard, make_key_control(key::space), true);
    input.begin_frame();
    assert(input.disconnect_device(keyboard));
    assert(!input.device(keyboard)->connected());
    assert(player_zero.released("jump"));
    assert(input.device_events().size() == 1);
    assert(input.device_events().front().type == input_device_event_type::disconnected);
    assert(contains_device(input.devices_for_player(0), keyboard));

    input.begin_frame();
    input.connect_device({
        .id = keyboard,
        .type = input_device_type::keyboard,
        .connectivity = input_connectivity_type::builtin,
        .name = "Test Keyboard",
        .capabilities = {.buttons = true, .button_count = 104},
    });
    assert(input.device(keyboard)->connected());
    assert(contains_device(input.devices_for_player(0), keyboard));
    assert(contains_device(input.devices_for_player(1), keyboard));
    assert(input.device_events().size() == 1);
    assert(input.device_events().front().type == input_device_event_type::connected);

    assert(input.set_rumble(controller, {.low_frequency = 0.4f, .high_frequency = 0.8f}));
    const int calls_before_disconnect = output.calls;
    input.begin_frame();
    assert(input.disconnect_device(controller));
    assert(output.calls == calls_before_disconnect + 1);
    assert(output.last_state == input_rumble_state{});
    assert(input.device(controller)->backend() == input_backend_type::hid);
    assert(input.device(controller)->backend_id() == "hid:test-controller-300");
    assert(contains_device(input.devices_for_player(1), controller));

    input.begin_frame();
    input.connect_device({
        .id = controller,
        .type = input_device_type::gamepad,
        .subtype = input_device_subtype::standard_gamepad,
        .connectivity = input_connectivity_type::wireless,
        .backend = input_backend_type::game_input,
        .hardware_id = {.vendor_id = 0x1234, .product_id = 0xabcd, .version = 0x0003},
        .backend_id = "gameinput:test-controller-300",
        .name = "Future Controller",
        .capabilities = {.buttons = true, .axes = true, .rumble = true, .gyroscope = true, .accelerometer = true},
    });
    assert(input.device(controller)->backend() == input_backend_type::game_input);
    assert(input.device(controller)->hardware_id().version == 0x0003);
    assert(input.device(controller)->backend_id() == "gameinput:test-controller-300");
    assert(player_one.set_rumble({.low_frequency = 0.2f, .high_frequency = 0.6f}));
    assert(output.last_state.low_frequency == 0.2f);
    assert(output.last_state.high_frequency == 0.6f);
    assert(input.unregister_output_sink(controller, output));
    assert(!input.set_rumble(controller, {.low_frequency = 1.0f, .high_frequency = 1.0f}));

    return 0;
}