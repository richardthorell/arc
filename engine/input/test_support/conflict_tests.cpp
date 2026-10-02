#include <arc/input/conflict.h>
#include <arc/input/gamepad.h>

#include <cassert>

namespace
{
arc::input::input_binding key_binding(arc::input::key key)
{
    return {.device = arc::input::input_device_type::keyboard, .control = arc::input::make_key_control(key)};
}

arc::input::input_binding gamepad_binding(arc::input::gamepad_button button)
{
    return {.device = arc::input::input_device_type::gamepad,
            .control = arc::input::make_gamepad_button_control(button)};
}

arc::input::input_mapping_binding mapping(std::string context, int priority, std::string action, std::string id,
                                          arc::input::input_binding binding)
{
    return {.context = std::move(context),
            .context_priority = priority,
            .context_enabled = true,
            .action = std::move(action),
            .binding_id = std::move(id),
            .binding = std::move(binding)};
}
} // namespace

int main()
{
    using namespace arc::input;

    const input_mapping_binding jump = mapping("gameplay", 0, "jump", "jump-space", key_binding(key::space));
    const input_mapping_binding interact = mapping("gameplay", 0, "interact", "interact-space", key_binding(key::space));

    auto conflicts = enumerate_binding_conflicts({jump}, interact);
    assert(conflicts.size() == 1);
    assert(conflicts.front().kind == input_conflict_kind::same_context);
    assert(conflicts.front().ambiguous);
    assert(conflicts.front().existing_action == "jump");
    assert(conflicts.front().candidate_action == "interact");

    const auto rejected = apply_binding_conflict_policy({jump}, interact, input_conflict_policy::reject);
    assert(!rejected.accepted);
    assert(rejected.bindings.size() == 1);

    const auto replaced = apply_binding_conflict_policy({jump}, interact, input_conflict_policy::replace);
    assert(replaced.accepted);
    assert(replaced.bindings.size() == 1);
    assert(replaced.bindings.front().binding_id == "interact-space");

    const input_mapping_binding menu = mapping("menu", 100, "accept", "menu-space", key_binding(key::space));
    conflicts = enumerate_binding_conflicts({jump}, menu);
    assert(conflicts.size() == 1);
    assert(conflicts.front().kind == input_conflict_kind::layered_context);
    assert(!conflicts.front().ambiguous);
    const auto layered = apply_binding_conflict_policy({jump}, menu, input_conflict_policy::reject);
    assert(layered.accepted);
    assert(layered.bindings.size() == 2);

    input_binding chord = key_binding(key::s);
    chord.modifiers.push_back(key_binding(key::left_control));
    const input_mapping_binding save = mapping("editor", 0, "save", "save-chord", chord);
    const input_mapping_binding crouch = mapping("editor", 0, "crouch", "ctrl", key_binding(key::left_control));
    conflicts = enumerate_binding_conflicts({crouch}, save);
    assert(conflicts.size() == 1);
    assert(conflicts.front().kind == input_conflict_kind::chord_component);
    assert(conflicts.front().ambiguous);

    const input_mapping_binding pad_accept =
        mapping("gameplay", 0, "accept", "pad-accept", gamepad_binding(gamepad_button::south));
    assert(enumerate_binding_conflicts({jump}, pad_accept).empty());

    input_mapping_binding disabled = interact;
    disabled.context_enabled = false;
    assert(enumerate_binding_conflicts({jump}, disabled).empty());

    return 0;
}
