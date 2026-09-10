#include <arc/project/project_module.h>

#include <iterator>
#include <string_view>

namespace
{
constexpr std::string_view runtime_component_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr std::string_view secondary_component_id = "cccccccccccccccccccccccccccccccc";

constexpr arc::project::game_field_descriptor_v1 runtime_fields[]{
    {0x2222222222222222ull, "value", "Value", "Runtime", "Runtime component probe value",
     arc::project::game_field_kind_v1::floating_point,
     arc::project::game_field_flags_v1::editable | arc::project::game_field_flags_v1::serialized, "1.0"},
};
constexpr arc::project::game_component_descriptor_v1 runtime_components[]{
    {runtime_component_id.data(), "runtime_probe", "Runtime Probe", "Runtime", "Runtime project component probe", 1,
     runtime_fields, std::size(runtime_fields)},
    {secondary_component_id.data(), "runtime_secondary", "Runtime Secondary", "Runtime", "Secondary scheduling probe",
     1, runtime_fields, std::size(runtime_fields)},
};

struct visit_state
{
    arc::project::game_system_context_v1* context{};
    bool visited{};
};

bool patch_component(void* user_data, arc::project::game_entity_v1 entity, const char* json)
{
    auto& state = *static_cast<visit_state*>(user_data);
    if (!json || std::string_view(json).find("\"value\":") == std::string_view::npos) return false;
    state.visited = true;
    return state.context->patch_project_component_json(state.context->project_component_user_data, entity,
                                                       runtime_component_id.data(), "{\"value\":2.0}");
}

bool execute_component_system(void*, arc::project::game_system_context_v1* context)
{
    if (!context || context->native_context || !context->project_component_user_data ||
        !context->for_each_project_component || !context->patch_project_component_json)
        return false;
    visit_state state{.context = context};
    return context->for_each_project_component(context->project_component_user_data, runtime_component_id.data(),
                                               &state, patch_component) &&
           state.visited;
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

constexpr arc::project::game_system_component_access_v1 component_accesses[]{
    {runtime_component_id.data(), arc::project::game_system_component_access_mode_v1::write},
    {secondary_component_id.data(), arc::project::game_system_component_access_mode_v1::read},
};
constexpr arc::project::game_system_descriptor_v1 component_system{
    .phase = arc::project::game_system_phase_v1::gameplay_commands,
    .priority = arc::project::game_system_priority_v1::normal,
    .component_accesses = component_accesses,
    .component_access_count = std::size(component_accesses),
    .unrestricted_native_world_access = false,
    .execute = execute_component_system,
};
constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.components", "Fixture Runtime Components",
     &component_system},
};
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 7,
    .components = runtime_components,
    .component_count = std::size(runtime_components),
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
