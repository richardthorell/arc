#include <arc/project/project_module.h>

#include <iterator>

namespace
{
constexpr arc::project::game_field_descriptor_v1 fields[]{
    {1, "value", "Value", "Runtime", "Value", arc::project::game_field_kind_v1::floating_point,
     arc::project::game_field_flags_v1::serialized, "0.0"},
};
constexpr arc::project::game_component_descriptor_v1 components[]{
    {"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "runtime_probe", "Runtime Probe", "Runtime", "Probe", 1, fields,
     std::size(fields)},
};
constexpr arc::project::game_system_component_access_v1 accesses[]{
    {"dddddddddddddddddddddddddddddddd", arc::project::game_system_component_access_mode_v1::write},
};
bool execute(void*, arc::project::game_system_context_v1*)
{
    return true;
}
bool start(const arc::project::game_module_host_v1*)
{
    return true;
}
void stop() {}
constexpr arc::project::game_system_descriptor_v1 system{
    .component_accesses = accesses,
    .component_access_count = std::size(accesses),
    .unrestricted_native_world_access = false,
    .execute = execute,
};
constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.invalid_access", "Invalid Access", &system},
};
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 8,
    .components = components,
    .component_count = std::size(components),
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
