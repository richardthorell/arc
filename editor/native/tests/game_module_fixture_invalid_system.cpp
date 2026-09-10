#include <arc/project/project_module.h>

#include <iterator>

namespace
{
bool execute_invalid_system(void*, arc::project::game_system_context_v1*)
{
    return true;
}
bool start(const arc::project::game_module_host_v1*)
{
    return true;
}
bool prepare_reload()
{
    return true;
}
void stop() {}

constexpr arc::project::game_system_descriptor_v1 invalid_system{
    .phase = arc::project::game_system_phase_v1::gameplay_commands,
    .priority = static_cast<arc::project::game_system_priority_v1>(0xffu),
    .execute = execute_invalid_system,
};
constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.invalid", "Invalid Runtime System",
     &invalid_system},
};
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 6,
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
