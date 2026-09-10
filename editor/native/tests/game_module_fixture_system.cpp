#include <arc/ecs/ecs.h>
#include <arc/project/project_module.h>
#include <arc/scene/scene.h>

#include <iterator>
#include <string_view>

namespace
{
constexpr std::string_view runtime_component_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

constexpr arc::project::game_field_descriptor_v1 runtime_fields[]{
    {0x2222222222222222ull, "value", "Value", "Runtime", "Runtime component probe value",
     arc::project::game_field_kind_v1::floating_point,
     arc::project::game_field_flags_v1::editable | arc::project::game_field_flags_v1::serialized, "1.0"},
};
constexpr arc::project::game_component_descriptor_v1 runtime_components[]{
    {runtime_component_id.data(), "runtime_probe", "Runtime Probe", "Runtime", "Runtime project component probe", 1,
     runtime_fields, std::size(runtime_fields)},
};

bool execute_visibility_system(void*, arc::project::game_system_context_v1* context)
{
    if (!context || !context->native_context) return false;
    auto& native = *static_cast<arc::ecs::system_context*>(context->native_context);
    auto& world = native.owner();
    for (const auto entity : world.entities())
    {
        const auto* name = world.try_get<arc::scene::name_component>(entity);
        if (!name || std::string_view(name->value) != "Runtime System Probe") continue;
        if (!context->project_component_user_data || !context->has_project_component ||
            !context->read_project_component_json || !context->patch_project_component_json)
            return false;

        const arc::project::game_entity_v1 runtime_entity{entity.index, entity.generation};
        if (!context->has_project_component(context->project_component_user_data, runtime_entity,
                                            runtime_component_id.data()))
            return false;
        const char* before = context->read_project_component_json(context->project_component_user_data, runtime_entity,
                                                                  runtime_component_id.data());
        if (!before || std::string_view(before).find("\"value\":1.0") == std::string_view::npos) return false;
        if (!context->patch_project_component_json(context->project_component_user_data, runtime_entity,
                                                   runtime_component_id.data(), "{\"value\":2.0}"))
            return false;
        const char* after = context->read_project_component_json(context->project_component_user_data, runtime_entity,
                                                                 runtime_component_id.data());
        if (!after || std::string_view(after).find("\"value\":2.0") == std::string_view::npos) return false;

        if (auto* mesh = world.try_get<arc::scene::mesh_renderer_component>(entity)) mesh->visible = false;
        if (auto* active = world.try_get<arc::scene::active_component>(entity)) active->active = false;
    }
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

constexpr arc::project::game_system_component_access_v1 visibility_accesses[]{
    {runtime_component_id.data(), arc::project::game_system_component_access_mode_v1::write},
};
constexpr arc::project::game_system_descriptor_v1 visibility_system{
    .phase = arc::project::game_system_phase_v1::gameplay_commands,
    .priority = arc::project::game_system_priority_v1::normal,
    .component_accesses = visibility_accesses,
    .component_access_count = std::size(visibility_accesses),
    .execute = execute_visibility_system,
};
constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.visibility", "Fixture Runtime Visibility",
     &visibility_system},
};
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 5,
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
