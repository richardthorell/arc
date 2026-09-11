#include <arc/project/project_module.h>

#include <cstdint>
#include <iterator>
#include <string_view>

namespace
{
bool session_active{};
bool lifecycle_valid{true};
std::uint64_t active_world_id{};
std::uint32_t begin_count{};
std::uint32_t end_count{};
arc::project::game_entity_v1 session_entity{};
constexpr std::string_view session_name = "M3.5 Lifecycle Entity";

struct session_search
{
    const arc::project::game_world_api_v1* world{};
    bool found{};
};

bool find_session_entity(void* user_data, arc::project::game_entity_v1 entity)
{
    auto& search = *static_cast<session_search*>(user_data);
    const auto name = search.world->read_name(search.world->user_data, entity);
    if (name.valid() && std::string_view(name.data, name.size) == session_name) search.found = true;
    return true;
}

bool begin_play(void*, const arc::project::game_play_context_v1* context)
{
    if (!context || context->structure_size < sizeof(arc::project::game_play_context_v1) || context->world_id == 0 ||
        !context->world || context->world->structure_size < sizeof(arc::project::game_world_api_v1) || session_active)
        return false;
    const auto& world = *context->world;
    if (!world.user_data || !world.create_entity || !world.destroy_entity || !world.entity_alive ||
        !world.query_entities || !world.has_core_component || !world.read_name || !world.set_name ||
        !world.read_transform || !world.set_transform || !world.read_tag || !world.set_tag || !world.read_active ||
        !world.set_active)
        return false;

    const auto created = world.create_entity(world.user_data);
    if (!created.valid() || created.is_deferred || !created.entity.valid()) return false;
    session_entity = created.entity;

    arc::project::game_transform_v1 transform;
    transform.position = {4.0f, 5.0f, 6.0f};
    constexpr std::string_view tag = "runtime";
    if (!world.set_name(world.user_data, created, session_name.data(), session_name.size()) ||
        !world.set_transform(world.user_data, created, &transform) ||
        !world.set_tag(world.user_data, created, tag.data(), tag.size()) ||
        !world.set_active(world.user_data, created, true))
        return false;

    arc::project::game_transform_v1 read_transform;
    bool active{};
    const auto read_name = world.read_name(world.user_data, session_entity);
    const auto read_tag = world.read_tag(world.user_data, session_entity);
    if (!read_name.valid() || std::string_view(read_name.data, read_name.size) != session_name || !read_tag.valid() ||
        std::string_view(read_tag.data, read_tag.size) != tag ||
        !world.has_core_component(world.user_data, session_entity, arc::project::game_core_component_v1::transform) ||
        !world.read_transform(world.user_data, session_entity, &read_transform) || read_transform.position.x != 4.0f ||
        read_transform.position.y != 5.0f || read_transform.position.z != 6.0f ||
        !world.read_active(world.user_data, session_entity, &active) || !active)
        return false;

    arc::project::game_world_query_v1 query{
        .required_core_components =
            arc::project::game_core_component_bit_v1(arc::project::game_core_component_v1::name) |
            arc::project::game_core_component_bit_v1(arc::project::game_core_component_v1::transform) |
            arc::project::game_core_component_bit_v1(arc::project::game_core_component_v1::active)};
    session_search search{.world = &world};
    if (!world.query_entities(world.user_data, &query, &search, find_session_entity) || !search.found) return false;

    const auto scratch = world.create_entity(world.user_data);
    if (!scratch.valid() || scratch.is_deferred || !world.destroy_entity(world.user_data, scratch) ||
        world.entity_alive(world.user_data, scratch.entity))
        return false;

    session_active = true;
    active_world_id = context->world_id;
    ++begin_count;
    lifecycle_valid = lifecycle_valid && begin_count == end_count + 1;
    return lifecycle_valid;
}

void end_play(void*, const arc::project::game_play_context_v1* context)
{
    if (!context || !context->world || !session_active || context->world_id != active_world_id)
    {
        lifecycle_valid = false;
        return;
    }
    const auto& world = *context->world;
    if (!world.entity_alive(world.user_data, session_entity) ||
        !world.destroy_entity(world.user_data, arc::project::game_entity_target_v1{.entity = session_entity}) ||
        world.entity_alive(world.user_data, session_entity))
        lifecycle_valid = false;
    session_entity = {};
    session_active = false;
    active_world_id = 0;
    ++end_count;
    lifecycle_valid = lifecycle_valid && begin_count == end_count;
}

bool execute(void*, arc::project::game_system_context_v1* context)
{
    return context && context->world && lifecycle_valid && session_active && context->world_id == active_world_id &&
           context->world->entity_alive(context->world->user_data, session_entity) && begin_count == end_count + 1;
}

bool start(const arc::project::game_module_host_v1*)
{
    session_active = false;
    lifecycle_valid = true;
    active_world_id = 0;
    begin_count = 0;
    end_count = 0;
    session_entity = {};
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
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.lifecycle-probe", "Fixture Lifecycle Probe",
     &lifecycle_probe_system},
    {arc::project::game_registration_kind_v1::play_lifecycle, "fixture.runtime.play-lifecycle",
     "Fixture Play Lifecycle", &play_lifecycle},
};

constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 12,
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
