#include <arc/project/project_module.h>

#include <iterator>
#include <string_view>

namespace
{
constexpr std::string_view runtime_component_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr std::string_view probe_name = "Runtime System Probe";

constexpr arc::project::game_field_descriptor_v1 runtime_fields[]{
    {0x2222222222222222ull, "value", "Value", "Runtime", "Runtime component probe value",
     arc::project::game_field_kind_v1::floating_point,
     arc::project::game_field_flags_v1::editable | arc::project::game_field_flags_v1::serialized, "1.0"},
};
constexpr arc::project::game_component_descriptor_v1 runtime_components[]{
    {runtime_component_id.data(), "runtime_probe", "Runtime Probe", "Runtime", "Runtime project component probe", 1,
     runtime_fields, std::size(runtime_fields)},
};

struct probe_search
{
    const arc::project::game_world_api_v1* world{};
    arc::project::game_entity_v1 entity{};
};

bool find_probe(void* user_data, arc::project::game_entity_v1 entity)
{
    auto& search = *static_cast<probe_search*>(user_data);
    const auto name = search.world->read_name(search.world->user_data, entity);
    if (name.valid() && std::string_view(name.data, name.size) == probe_name) search.entity = entity;
    return true;
}

bool execute_visibility_system(void*, arc::project::game_system_context_v1* context)
{
    if (!context || !context->world || context->world->structure_size < sizeof(arc::project::game_world_api_v1))
        return false;
    const auto& world = *context->world;
    if (!world.user_data || !world.query_entities || !world.read_name || !world.read_transform || !world.read_active ||
        !world.set_active || !world.create_entity || !world.destroy_entity || !world.set_name || !world.set_transform ||
        !world.set_tag || !world.entity_count || world.entity_count(world.user_data) == 0)
        return false;

    arc::project::game_world_query_v1 query{
        .required_core_components = arc::project::game_core_component_bit_v1(arc::project::game_core_component_v1::name) |
                                    arc::project::game_core_component_bit_v1(arc::project::game_core_component_v1::active)};
    probe_search search{.world = &world};
    if (!world.query_entities(world.user_data, &query, &search, find_probe) || !search.entity.valid()) return false;

    if (!context->project_component_user_data || !context->has_project_component ||
        !context->read_project_component_json || !context->patch_project_component_json)
        return false;
    if (!context->has_project_component(context->project_component_user_data, search.entity, runtime_component_id.data()))
        return false;
    const char* before = context->read_project_component_json(context->project_component_user_data, search.entity,
                                                              runtime_component_id.data());
    if (!before || std::string_view(before).find("\"value\":1.0") == std::string_view::npos) return false;
    if (!context->patch_project_component_json(context->project_component_user_data, search.entity,
                                               runtime_component_id.data(), "{\"value\":2.0}"))
        return false;
    const char* after = context->read_project_component_json(context->project_component_user_data, search.entity,
                                                             runtime_component_id.data());
    if (!after || std::string_view(after).find("\"value\":2.0") == std::string_view::npos) return false;

    arc::project::game_transform_v1 transform;
    bool active{};
    if (!world.read_transform(world.user_data, search.entity, &transform) ||
        !world.read_active(world.user_data, search.entity, &active) || !active)
        return false;
    if (!world.set_active(world.user_data, arc::project::game_entity_target_v1{.entity = search.entity}, false))
        return false;

    const auto scratch = world.create_entity(world.user_data);
    if (!scratch.valid()) return false;
    constexpr std::string_view scratch_name = "Runtime Scratch";
    constexpr std::string_view scratch_tag = "m3.5";
    arc::project::game_transform_v1 scratch_transform;
    scratch_transform.position = {1.0f, 2.0f, 3.0f};
    if (!world.set_name(world.user_data, scratch, scratch_name.data(), scratch_name.size()) ||
        !world.set_transform(world.user_data, scratch, &scratch_transform) ||
        !world.set_tag(world.user_data, scratch, scratch_tag.data(), scratch_tag.size()) ||
        !world.set_active(world.user_data, scratch, true) || !world.destroy_entity(world.user_data, scratch))
        return false;

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
constexpr arc::project::game_core_component_access_v1 core_accesses[]{
    {arc::project::game_core_component_v1::name, arc::project::game_core_component_access_mode_v1::write},
    {arc::project::game_core_component_v1::transform, arc::project::game_core_component_access_mode_v1::write},
    {arc::project::game_core_component_v1::tag, arc::project::game_core_component_access_mode_v1::write},
    {arc::project::game_core_component_v1::active, arc::project::game_core_component_access_mode_v1::write},
};
constexpr arc::project::game_system_descriptor_v1 visibility_system{
    .phase = arc::project::game_system_phase_v1::gameplay_commands,
    .priority = arc::project::game_system_priority_v1::normal,
    .component_accesses = visibility_accesses,
    .component_access_count = std::size(visibility_accesses),
    .unrestricted_native_world_access = false,
    .execute = execute_visibility_system,
    .core_component_accesses = core_accesses,
    .core_component_access_count = std::size(core_accesses),
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
    .generation = 6,
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
