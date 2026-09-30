#include <arc/input/composite.h>

#include <array>

namespace
{
arc::input::input_binding key_binding(arc::input::key value)
{
    return {.device = arc::input::input_device_type::keyboard,
            .control = arc::input::make_key_control(value),
            .processors = {}};
}

int require(bool condition, int code)
{
    return condition ? 0 : code;
}
} // namespace

int main()
{
    using namespace arc::input;

    const input_chord save{
        .primary = key_binding(key::s),
        .modifiers = {key_binding(key::left_control)},
    };
    const std::array save_active{true, true};
    const std::array save_inactive{true, false};
    if (const int error = require(evaluate_chord(save, save_active), 1)) return error;
    if (const int error = require(!evaluate_chord(save, save_inactive), 2)) return error;

    const input_chord ordered{
        .primary = key_binding(key::s),
        .modifiers = {key_binding(key::left_control), key_binding(key::left_shift)},
        .ordered = true,
    };
    const std::array ordered_active{true, true, true};
    const std::array<std::uint64_t, 3> correct_order{1, 2, 3};
    const std::array<std::uint64_t, 3> wrong_order{2, 1, 3};
    if (const int error = require(evaluate_chord(ordered, ordered_active, correct_order), 3)) return error;
    if (const int error = require(!evaluate_chord(ordered, ordered_active, wrong_order), 4)) return error;

    const input_digital_axis2d_composite wasd{
        .parts =
            {
                {.binding = key_binding(key::w), .contribution = {0.0f, 1.0f}},
                {.binding = key_binding(key::s), .contribution = {0.0f, -1.0f}},
                {.binding = key_binding(key::a), .contribution = {-1.0f, 0.0f}},
                {.binding = key_binding(key::d), .contribution = {1.0f, 0.0f}},
            },
    };

    const std::array diagonal{true, false, false, true};
    const auto diagonal_value = evaluate_digital_axis2d(wasd, diagonal);
    if (const int error = require(diagonal_value[0] == 1.0f && diagonal_value[1] == 1.0f, 5)) return error;

    const std::array opposing{true, true, true, true};
    const auto opposing_value = evaluate_digital_axis2d(wasd, opposing);
    if (const int error = require(opposing_value[0] == 0.0f && opposing_value[1] == 0.0f, 6)) return error;

    const std::array invalid{true};
    const auto invalid_value = evaluate_digital_axis2d(wasd, invalid);
    if (const int error = require(invalid_value[0] == 0.0f && invalid_value[1] == 0.0f, 7)) return error;

    return 0;
}
