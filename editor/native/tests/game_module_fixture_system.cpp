#include <arc/ecs/ecs.h>
#include <arc/project/project_module.h>
#include <arc/scene/scene.h>

#include <iterator>
#include <string_view>

namespace
{
bool execute_visibility_system(void*, arc::project::game_system_context_v1* context)
{
    if (!context || !context->native_context) return false;
    auto& native = *static_cast<arc::ecs::system_context*>(context->native_context);
    auto& world = native.owner();
    for (const auto entity : world.entities())
    {
        const auto* name = world.try_get<arc::scene::name_component>(entity);
        if (!name || std::string_view(name->value) != "Runtime System Probe") continue;
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

constexpr arc::project::game_system_descriptor_v1 visibility_system{
    .phase = arc::project::game_system_phase_v1::gameplay_commands,
    .priority = arc::project::game_system_priority_v1::normal,
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
