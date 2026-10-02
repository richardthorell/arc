#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace arc::input
{

enum class input_action_phase : std::uint8_t
{
    none,
    started,
    performed,
    canceled
};

enum class input_trigger_type : std::uint8_t
{
    press,
    release,
    hold,
    tap,
    double_tap,
    threshold
};

struct input_trigger_config
{
    input_trigger_type type{input_trigger_type::press};
    float actuation_threshold{0.5F};
    float hold_seconds{0.5F};
    float tap_seconds{0.25F};
    float double_tap_seconds{0.3F};
};

struct input_trigger_state
{
    bool actuated{};
    bool performed{};
    bool waiting_for_second_tap{};
    float actuated_seconds{};
    float since_first_tap_seconds{};
};

struct input_trigger_result
{
    input_action_phase phase{input_action_phase::none};
    input_trigger_state state{};
};

/**
 * @brief Advance one semantic-action trigger by a caller supplied frame delta.
 *
 * The evaluator is platform neutral and contains no global state. Callers own one
 * state value per player/action/trigger, which prevents trigger history leaking
 * between players or mapping contexts. A context becoming inactive should discard
 * its state rather than continuing an old gesture when it is enabled again.
 */
[[nodiscard]] inline input_trigger_result evaluate_action_trigger(input_trigger_state state,
                                                                  const input_trigger_config& config,
                                                                  float value,
                                                                  float delta_seconds) noexcept
{
    const float dt = std::max(0.0F, delta_seconds);
    const float threshold = std::max(0.0F, config.actuation_threshold);
    const bool is_actuated = std::fabs(value) >= threshold;
    const bool pressed = is_actuated && !state.actuated;
    const bool released = !is_actuated && state.actuated;
    input_action_phase phase = input_action_phase::none;

    if (state.waiting_for_second_tap)
    {
        state.since_first_tap_seconds += dt;
        if (state.since_first_tap_seconds > std::max(0.0F, config.double_tap_seconds))
        {
            state.waiting_for_second_tap = false;
            state.since_first_tap_seconds = 0.0F;
        }
    }

    if (is_actuated)
        state.actuated_seconds = state.actuated ? state.actuated_seconds + dt : 0.0F;

    switch (config.type)
    {
    case input_trigger_type::press:
        if (pressed)
            phase = input_action_phase::performed;
        break;
    case input_trigger_type::release:
        if (pressed)
            phase = input_action_phase::started;
        else if (released)
            phase = input_action_phase::performed;
        break;
    case input_trigger_type::hold:
        if (pressed)
            phase = input_action_phase::started;
        else if (is_actuated && !state.performed && state.actuated_seconds >= std::max(0.0F, config.hold_seconds))
        {
            state.performed = true;
            phase = input_action_phase::performed;
        }
        else if (released && !state.performed)
            phase = input_action_phase::canceled;
        break;
    case input_trigger_type::tap:
        if (pressed)
            phase = input_action_phase::started;
        else if (released)
            phase = state.actuated_seconds <= std::max(0.0F, config.tap_seconds)
                        ? input_action_phase::performed
                        : input_action_phase::canceled;
        break;
    case input_trigger_type::double_tap:
        if (released && state.actuated_seconds <= std::max(0.0F, config.tap_seconds))
        {
            if (state.waiting_for_second_tap)
            {
                state.waiting_for_second_tap = false;
                state.since_first_tap_seconds = 0.0F;
                phase = input_action_phase::performed;
            }
            else
            {
                state.waiting_for_second_tap = true;
                state.since_first_tap_seconds = 0.0F;
                phase = input_action_phase::started;
            }
        }
        break;
    case input_trigger_type::threshold:
        if (pressed)
            phase = input_action_phase::started;
        else if (released)
            phase = input_action_phase::canceled;
        else if (is_actuated)
            phase = input_action_phase::performed;
        break;
    }

    if (released)
    {
        state.actuated_seconds = 0.0F;
        state.performed = false;
    }
    state.actuated = is_actuated;
    return {.phase = phase, .state = state};
}

} // namespace arc::input
