#include <arc/project/project_module.h>

#include <cstdint>
#include <iterator>

namespace
{
bool session_active{};
bool lifecycle_valid{true};
std::uint64_t active_world_id{};
std::uint32_t begin_count{};
std::uint32_t end_count{};

bool begin_play(void*, const arc::project::game_play_context_v1* context)
{
    if (!context || context->structure_size < sizeof(arc::project::game_play_context_v1) || context->world_id == 0 ||
        session_active)
        return false;
    session_active = true;
    active_world_id = context->world_id;
    ++begin_count;
    lifecycle_valid = lifecycle_valid && begin_count == end_count + 1;
    return lifecycle_valid;
}

void end_play(void*, const arc::project::game_play_context_v1* context)
{
    if (!context || !session_active || context->world_id != active_world_id)
    {
        lifecycle_valid = false;
        return;
    }
    session_active = false;
    active_world_id = 0;
    ++end_count;
    lifecycle_valid = lifecycle_valid && begin_count == end_count;
}

bool execute(void*, arc::project::game_system_context_v1* context)
{
    return context && lifecycle_valid && session_active && context->world_id == active_world_id &&
           begin_count == end_count + 1;
}

bool start(const arc::project::game_module_host_v1*)
{
    session_active = false;
    lifecycle_valid = true;
    active_world_id = 0;
    begin_count = 0;
    end_count = 0;
    return true;
}

bool prepare_reload()
{
    return !session_active;
}

void stop()
{
    lifecycle_valid = lifecycle_valid && !session_active && begin_count == end_count;
}

constexpr arc::project::game_system_descriptor_v1 lifecycle_probe_system{
    .phase = arc::project::game_system_phase_v1::gameplay_commands,
    .priority = arc::project::game_system_priority_v1::normal,
    .unrestricted_native_world_access = false,
    .execute = execute,
};

constexpr arc::project::game_play_lifecycle_descriptor_v1 play_lifecycle{
    .begin_play = begin_play,
    .end_play = end_play,
};

constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.lifecycle-probe",
     "Fixture Lifecycle Probe", &lifecycle_probe_system},
    {arc::project::game_registration_kind_v1::play_lifecycle, "fixture.runtime.play-lifecycle",
     "Fixture Play Lifecycle", &play_lifecycle},
};

constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 11,
    .registrations = registrations,
    .registration_count = std::size(registrations),
    .start = start,
    .prepare_reload = prepare_reload,
    .stop = stop,
};
} // namespace

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1()
{
    return &descriptor;
}