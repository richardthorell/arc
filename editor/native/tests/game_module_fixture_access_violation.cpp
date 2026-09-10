#include <arc/project/project_module.h>

#include <iterator>

namespace
{
constexpr const char* component_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr arc::project::game_field_descriptor_v1 fields[]{
    {1, "value", "Value", "Runtime", "Value", arc::project::game_field_kind_v1::floating_point,
     arc::project::game_field_flags_v1::serialized, "1.0"},
};
constexpr arc::project::game_component_descriptor_v1 components[]{
    {component_id, "runtime_probe", "Runtime Probe", "Runtime", "Probe", 1, fields, std::size(fields)},
};

bool violate(void* user_data, arc::project::game_entity_v1 entity, const char*)
{
    auto& context = *static_cast<arc::project::game_system_context_v1*>(user_data);
    (void)context.patch_project_component_json(context.project_component_user_data, entity, component_id,
                                               "{\"value\":9.0}");
    return true;
}

bool execute(void*, arc::project::game_system_context_v1* context)
{
    if (!context || !context->for_each_project_component) return false;
    return context->for_each_project_component(context->project_component_user_data, component_id, context, violate);
}
bool start(const arc::project::game_module_host_v1*)
{
    return true;
}
void stop() {}
constexpr arc::project::game_system_component_access_v1 accesses[]{
    {component_id, arc::project::game_system_component_access_mode_v1::read},
};
constexpr arc::project::game_system_descriptor_v1 system{
    .component_accesses = accesses,
    .component_access_count = std::size(accesses),
    .unrestricted_native_world_access = false,
    .execute = execute,
};
constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.access_violation", "Access Violation",
     &system},
};
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 9,
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
