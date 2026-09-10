#include <arc/project/project_module.h>

#include <cstddef>
#include <iterator>

namespace
{
std::uint32_t phase{};

bool execute(void*, arc::project::game_system_context_v1* context)
{
    if (!context) return false;
    bool saw_key{};
    bool saw_pointer{};
    bool saw_focus_lost{};
    for (std::size_t index = 0; index < context->input_command_count; ++index)
    {
        const auto& input = context->input_commands[index];
        if (input.kind == arc::project::game_input_kind_v1::key &&
            input.action == arc::project::game_input_action_v1::pressed && input.code == 'W' &&
            (input.modifiers & arc::project::game_input_modifier_shift_v1) != 0u)
            saw_key = true;
        if (input.kind == arc::project::game_input_kind_v1::mouse_position && input.x == 12 && input.y == 34)
            saw_pointer = true;
        if (input.kind == arc::project::game_input_kind_v1::focus &&
            input.action == arc::project::game_input_action_v1::changed && input.value == 0.0f)
            saw_focus_lost = true;
    }
    if (phase == 0)
    {
        if (!saw_key || !saw_pointer || context->input_revision == 0) return false;
        ++phase;
    }
    else if (phase == 1)
    {
        if (!saw_focus_lost) return false;
        ++phase;
    }
    return true;
}

bool start(const arc::project::game_module_host_v1*)
{
    phase = 0;
    return true;
}
void stop() {}

constexpr arc::project::game_system_descriptor_v1 input_system{
    .phase = arc::project::game_system_phase_v1::input,
    .priority = arc::project::game_system_priority_v1::critical,
    .unrestricted_native_world_access = false,
    .execute = execute,
};
constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.input", "Fixture Runtime Input",
     &input_system},
};
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 10,
    .registrations = registrations,
    .registration_count = std::size(registrations),
    .start = start,
    .stop = stop,
};
} // namespace

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1()
{
    return &descriptor;
}
