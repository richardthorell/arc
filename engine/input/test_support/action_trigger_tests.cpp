#include <arc/input/action_trigger.h>

#include <cassert>

using namespace arc::input;

namespace
{
input_trigger_result step(input_trigger_state state, input_trigger_type type, float value, float dt,
                          float hold = 0.5F)
{
    input_trigger_config config{};
    config.type = type;
    config.hold_seconds = hold;
    return evaluate_action_trigger(state, config, value, dt);
}
} // namespace

int main()
{
    input_trigger_state state{};

    auto result = step(state, input_trigger_type::press, 1.0F, 0.016F);
    assert(result.phase == input_action_phase::performed);

    state = {};
    result = step(state, input_trigger_type::hold, 1.0F, 0.1F, 0.5F);
    assert(result.phase == input_action_phase::started);
    result = step(result.state, input_trigger_type::hold, 1.0F, 0.5F, 0.5F);
    assert(result.phase == input_action_phase::performed);

    state = {};
    result = step(state, input_trigger_type::hold, 1.0F, 0.1F, 0.5F);
    result = step(result.state, input_trigger_type::hold, 0.0F, 0.1F, 0.5F);
    assert(result.phase == input_action_phase::canceled);

    input_trigger_config tap{};
    tap.type = input_trigger_type::tap;
    tap.tap_seconds = 0.25F;
    result = evaluate_action_trigger({}, tap, 1.0F, 0.0F);
    result = evaluate_action_trigger(result.state, tap, 0.0F, 0.2F);
    assert(result.phase == input_action_phase::performed);

    result = evaluate_action_trigger({}, tap, 1.0F, 0.0F);
    result = evaluate_action_trigger(result.state, tap, 0.0F, 0.3F);
    assert(result.phase == input_action_phase::canceled);

    input_trigger_config threshold{};
    threshold.type = input_trigger_type::threshold;
    threshold.actuation_threshold = 0.7F;
    result = evaluate_action_trigger({}, threshold, 0.69F, 0.016F);
    assert(result.phase == input_action_phase::none);
    result = evaluate_action_trigger(result.state, threshold, 0.71F, 0.016F);
    assert(result.phase == input_action_phase::started);
    result = evaluate_action_trigger(result.state, threshold, 0.8F, 0.016F);
    assert(result.phase == input_action_phase::performed);
    result = evaluate_action_trigger(result.state, threshold, 0.2F, 0.016F);
    assert(result.phase == input_action_phase::canceled);

    input_trigger_config double_tap{};
    double_tap.type = input_trigger_type::double_tap;
    result = evaluate_action_trigger({}, double_tap, 1.0F, 0.0F);
    result = evaluate_action_trigger(result.state, double_tap, 0.0F, 0.1F);
    assert(result.phase == input_action_phase::started);
    result = evaluate_action_trigger(result.state, double_tap, 1.0F, 0.1F);
    result = evaluate_action_trigger(result.state, double_tap, 0.0F, 0.1F);
    assert(result.phase == input_action_phase::performed);

    return 0;
}
