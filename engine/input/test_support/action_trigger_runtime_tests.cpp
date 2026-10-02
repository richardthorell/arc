#include <arc/input/action_trigger_runtime.h>

#include <cassert>

using namespace arc::input;

int main()
{
    input_action_trigger_runtime runtime;
    input_trigger_config hold{};
    hold.type = input_trigger_type::hold;
    hold.hold_seconds = 0.5F;

    // Trigger history is independent for each player even when action/context names match.
    auto player_zero = runtime.evaluate(0, "ChargeAttack", "Gameplay", hold, 1.0F, 0.0F);
    assert(player_zero.phase == input_action_phase::started);
    player_zero = runtime.evaluate(0, "ChargeAttack", "Gameplay", hold, 1.0F, 0.5F);
    assert(player_zero.phase == input_action_phase::performed);

    auto player_one = runtime.evaluate(1, "ChargeAttack", "Gameplay", hold, 1.0F, 0.0F);
    assert(player_one.phase == input_action_phase::started);
    assert(runtime.state(0, "ChargeAttack") != nullptr);
    assert(runtime.state(1, "ChargeAttack") != nullptr);
    assert(runtime.state(0, "ChargeAttack")->performed);
    assert(!runtime.state(1, "ChargeAttack")->performed);

    // Resolving the same action through a different priority context starts fresh.
    auto switched = runtime.evaluate(0, "ChargeAttack", "Menu", hold, 1.0F, 0.0F);
    assert(switched.phase == input_action_phase::started);
    assert(!switched.state.performed);
    assert(switched.state.actuated_seconds == 0.0F);

    // Disabling/reprioritizing a context can explicitly clear only state sourced from it.
    runtime.reset_context(0, "Menu");
    assert(runtime.state(0, "ChargeAttack") == nullptr);
    assert(runtime.state(1, "ChargeAttack") != nullptr);

    // A reset player cannot inherit a partially completed tap/hold sequence.
    input_trigger_config double_tap{};
    double_tap.type = input_trigger_type::double_tap;
    auto tap = runtime.evaluate(1, "Dodge", "Gameplay", double_tap, 1.0F, 0.0F);
    tap = runtime.evaluate(1, "Dodge", "Gameplay", double_tap, 0.0F, 0.1F);
    assert(tap.phase == input_action_phase::started);
    runtime.reset_player(1);
    assert(runtime.state(1, "Dodge") == nullptr);
    tap = runtime.evaluate(1, "Dodge", "Gameplay", double_tap, 1.0F, 0.0F);
    tap = runtime.evaluate(1, "Dodge", "Gameplay", double_tap, 0.0F, 0.1F);
    assert(tap.phase == input_action_phase::started);

    runtime.reset();
    assert(runtime.state(1, "Dodge") == nullptr);

    return 0;
}
