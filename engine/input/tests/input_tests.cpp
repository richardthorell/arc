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

bool contains_device(const std::vector<arc::input::input_device_id>& devices, arc::input::input_device_id id)
{
    return std::find(devices.begin(), devices.end(), id) != devices.end();
}

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
        .connectivity = input_connectivity_type::wireless,
        .name = "Future Controller",
        .capabilities = {.buttons = true, .axes = true, .rumble = true, .gyroscope = true, .accelerometer = true},
    });

    assert(input.device(keyboard));
    assert(input.device(keyboard)->connectivity() == input_connectivity_type::builtin);
    assert(input.device(mouse)->connectivity() == input_connectivity_type::usb);
    assert(input.device(controller)->capabilities().gyroscope);
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

    player_zero.add_context("gameplay", 0);
    player_zero.bind_action("gameplay", "jump", key_binding(key::space));
    player_zero.bind_axis2d("gameplay", "move", key_binding(key::d), {1.0f, 0.0f});
    player_zero.bind_axis2d("gameplay", "move", key_binding(key::a), {-1.0f, 0.0f});
    player_zero.bind_axis2d("gameplay", "move", key_binding(key::w), {0.0f, 1.0f});
    player_zero.bind_axis2d("gameplay", "move", key_binding(key::s), {0.0f, -1.0f});

    player_one.add_context("gameplay", 0);
    player_one.bind_action("gameplay", "jump", key_binding(key::enter));

    input.begin_frame();
    assert(input.device_events().empty());
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

    return 0;
}
