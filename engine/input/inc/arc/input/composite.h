#pragma once

#include <arc/input/input.h>

#include <algorithm>
#include <span>
#include <vector>

namespace arc::input
{

/**
 * @brief Platform-neutral chord definition layered on top of a primary binding.
 *
 * Modifiers are intentionally ordinary input bindings. This keeps chord data
 * independent of platform APIs and allows keyboard and gamepad controls to use
 * the same representation.
 */
struct input_chord
{
    input_binding primary;
    std::vector<input_binding> modifiers;
    bool ordered{};
};

/**
 * @brief One digital contribution to a two-dimensional composite.
 */
struct input_digital_axis2d_part
{
    input_binding binding;
    math::vector2f contribution{};
};

/**
 * @brief A deterministic digital vector composite such as WASD or a D-pad.
 */
struct input_digital_axis2d_composite
{
    std::vector<input_digital_axis2d_part> parts;
};

/**
 * @brief Evaluate a chord from already-resolved digital binding states.
 *
 * The first state is the primary binding and the remaining states correspond
 * to modifiers in authored order. Ordered chords additionally require the
 * caller-provided activation order to be strictly increasing, which lets the
 * runtime preserve press ordering without putting timestamps in authored data.
 */
[[nodiscard]] inline bool evaluate_chord(const input_chord& chord, std::span<const bool> states,
                                         std::span<const std::uint64_t> activation_order = {}) noexcept
{
    const std::size_t expected = chord.modifiers.size() + 1;
    if (states.size() != expected || !std::all_of(states.begin(), states.end(), [](bool active) { return active; }))
    {
        return false;
    }

    if (!chord.ordered)
    {
        return true;
    }

    if (activation_order.size() != expected)
    {
        return false;
    }

    for (std::size_t index = 1; index < activation_order.size(); ++index)
    {
        if (activation_order[index - 1] >= activation_order[index])
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Combine active digital parts into one deterministic 2D value.
 *
 * Opposing directions cancel naturally. Each component is clamped to [-1, 1]
 * so duplicate active parts cannot amplify a digital composite beyond its
 * normalized range.
 */
[[nodiscard]] inline math::vector2f evaluate_digital_axis2d(const input_digital_axis2d_composite& composite,
                                                            std::span<const bool> states) noexcept
{
    math::vector2f result{};
    if (states.size() != composite.parts.size())
    {
        return result;
    }

    for (std::size_t index = 0; index < composite.parts.size(); ++index)
    {
        if (!states[index])
        {
            continue;
        }
        result[0] += composite.parts[index].contribution[0];
        result[1] += composite.parts[index].contribution[1];
    }

    result[0] = std::clamp(result[0], -1.0f, 1.0f);
    result[1] = std::clamp(result[1], -1.0f, 1.0f);
    return result;
}

} // namespace arc::input
