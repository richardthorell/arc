#include <arc/input/composite.h>

#include <array>
#include <cassert>

namespace
{
arc::input::input_binding key_binding(arc::input::key value)
{
    return {.device = arc::input::input_device_type::keyboard,
            .control = arc::input::make_key_control(value),
            .processors = {}};
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
    assert(evaluate_chord(save, save_active));
    assert(!evaluate_chord(save, save_inactive));

    const input_chord ordered{
        .primary = key_binding(key::s),
        .modifiers = {key_binding(key::left_control), key_binding(key::left_shift)},
        .ordered = true,
    };
    const std::array ordered_active{true, true, true};
    const std::array<std::uint64_t, 3> correct_order{1, 2, 3};
    const std::array<std::uint64_t, 3> wrong_order{2, 1, 3};
    assert(evaluate_chord(ordered, ordered_active, correct_order));
    assert(!evaluate_chord(ordered, ordered_active, wrong_order));

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
    assert(diagonal_value[0] == 1.0f);
    assert(diagonal_value[1] == 1.0f);

    const std::array opposing{true, true, true, true};
    const auto opposing_value = evaluate_digital_axis2d(wasd, opposing);
    assert(opposing_value[0] == 0.0f);
    assert(opposing_value[1] == 0.0f);

    const std::array invalid{true};
    const auto invalid_value = evaluate_digital_axis2d(wasd, invalid);
    assert(invalid_value[0] == 0.0f);
    assert(invalid_value[1] == 0.0f);

    return 0;
}
