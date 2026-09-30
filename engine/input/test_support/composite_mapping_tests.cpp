#include <arc/input/gamepad.h>
#include <arc/input/input.h>

namespace
{

int require(bool condition, int code)
{
    return condition ? 0 : code;
}

arc::input::input_binding key_binding(arc::input::key key)
{
    return {.device = arc::input::input_device_type::keyboard, .control = arc::input::make_key_control(key)};
}

arc::input::input_binding gamepad_binding(arc::input::gamepad_button button)
{
    return {.device = arc::input::input_device_type::gamepad,
            .control = arc::input::make_gamepad_button_control(button)};
}

} // namespace

int main()
{
    using namespace arc::input;

    input_system input;
    const auto keyboard = input.connect_device(
        {.type = input_device_type::keyboard, .name = "Keyboard", .capabilities = {.buttons = true}});
    const auto gamepad = input.connect_device(
        {.type = input_device_type::gamepad, .name = "Gamepad", .capabilities = {.buttons = true}});
    if (const int error = require(input.assign_device(0, gamepad), 1)) return error;

    auto& player = input.player(0);
    player.add_context("gameplay");

    auto save = key_binding(key::s);
    save.modifiers.push_back(key_binding(key::left_control));
    player.bind_action("gameplay", "Save", save);

    if (const int error = require(input.submit_button(keyboard, make_key_control(key::s), true), 2)) return error;
    if (const int error = require(!player.down("Save"), 3)) return error;
    if (const int error = require(input.submit_button(keyboard, make_key_control(key::left_control), true), 4))
        return error;
    if (const int error = require(player.down("Save"), 5)) return error;
    if (const int error = require(input.submit_button(keyboard, make_key_control(key::left_control), false), 6))
        return error;
    if (const int error = require(!player.down("Save"), 7)) return error;

    auto special = gamepad_binding(gamepad_button::south);
    special.modifiers.push_back(key_binding(key::left_shift));
    player.bind_action("gameplay", "Special", special);
    if (const int error = require(input.submit_button(gamepad, make_gamepad_button_control(gamepad_button::south), true), 8))
        return error;
    if (const int error = require(!player.down("Special"), 9)) return error;
    if (const int error = require(input.submit_button(keyboard, make_key_control(key::left_shift), true), 10))
        return error;
    if (const int error = require(player.down("Special"), 11)) return error;

    auto precision = key_binding(key::w);
    precision.modifiers.push_back(key_binding(key::left_shift));
    precision.composite_processors.push_back({.type = input_processor_type::scale, .value = 0.25f});
    player.bind_axis("gameplay", "PrecisionForward", precision);
    if (const int error = require(input.submit_button(keyboard, make_key_control(key::w), true), 12)) return error;
    if (const int error = require(player.axis("PrecisionForward") == 0.25f, 13)) return error;

    player.bind_axis2d("gameplay", "Move", key_binding(key::w), {0.0f, 1.0f});
    player.bind_axis2d("gameplay", "Move", key_binding(key::s), {0.0f, -1.0f});
    player.bind_axis2d("gameplay", "Move", key_binding(key::a), {-1.0f, 0.0f});
    player.bind_axis2d("gameplay", "Move", key_binding(key::d), {1.0f, 0.0f});

    if (const int error = require(input.submit_button(keyboard, make_key_control(key::d), true), 14)) return error;
    const auto opposed = player.axis2d("Move");
    if (const int error = require(opposed[0] == 1.0f && opposed[1] == 0.0f, 15)) return error;

    player.add_context("low", 1);
    player.add_context("high", 10);
    player.bind_axis("low", "PriorityAxis", key_binding(key::w));
    player.bind_axis("high", "PriorityAxis", key_binding(key::d));
    if (const int error = require(input.submit_button(keyboard, make_key_control(key::d), false), 16)) return error;
    if (const int error = require(player.axis("PriorityAxis") == 0.0f, 17)) return error;
    if (const int error = require(input.submit_button(keyboard, make_key_control(key::d), true), 18)) return error;
    if (const int error = require(player.axis("PriorityAxis") == 1.0f, 19)) return error;

    return 0;
}
