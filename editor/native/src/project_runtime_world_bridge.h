#pragma once

#include <arc/ecs/system.h>
#include <arc/project/runtime_world_api.h>
#include <arc/scene/components.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace arc::editor
{

struct runtime_world_bridge_context
{
    ecs::world* world{};
    ecs::entity_command_buffer* commands{};
    const std::vector<project::game_core_component_access_v1>* declared_accesses{};
    bool unrestricted_access{};
    bool violation{};
    std::string violation_message;
};

inline const char* core_component_name(project::game_core_component_v1 component) noexcept
{
    switch (component)
    {
        case project::game_core_component_v1::name:
            return "name";
        case project::game_core_component_v1::transform:
            return "transform";
        case project::game_core_component_v1::tag:
            return "tag";
        case project::game_core_component_v1::active:
            return "active";
    }
    return "unknown";
}

inline bool valid_core_component(project::game_core_component_v1 component) noexcept
{
    switch (component)
    {
        case project::game_core_component_v1::name:
        case project::game_core_component_v1::transform:
        case project::game_core_component_v1::tag:
        case project::game_core_component_v1::active:
            return true;
    }
    return false;
}

inline bool valid_core_component_access_mode(project::game_core_component_access_mode_v1 mode) noexcept
{
    return mode == project::game_core_component_access_mode_v1::read ||
           mode == project::game_core_component_access_mode_v1::write;
}

inline bool allows_core_component_access(const runtime_world_bridge_context& bridge,
                                         project::game_core_component_v1 component,
                                         project::game_core_component_access_mode_v1 requested) noexcept
{
    if (!valid_core_component(component)) return false;
    if (bridge.unrestricted_access) return true;
    if (!bridge.declared_accesses) return false;
    const auto found = std::find_if(bridge.declared_accesses->begin(), bridge.declared_accesses->end(),
                                    [&](const project::game_core_component_access_v1& access)
                                    { return access.component == component; });
    if (found == bridge.declared_accesses->end()) return false;
    return requested == project::game_core_component_access_mode_v1::read ||
           found->mode == project::game_core_component_access_mode_v1::write;
}

inline bool require_core_component_access(runtime_world_bridge_context& bridge,
                                          project::game_core_component_v1 component,
                                          project::game_core_component_access_mode_v1 requested) noexcept
{
    if (allows_core_component_access(bridge, component, requested)) return true;
    bridge.violation = true;
    bridge.violation_message =
        std::string(requested == project::game_core_component_access_mode_v1::write ? "write" : "read") +
        " access to undeclared core component '" + core_component_name(component) + "'";
    return false;
}

[[nodiscard]] inline ecs::entity native_entity(project::game_entity_v1 entity) noexcept
{
    return {entity.index, entity.generation};
}

[[nodiscard]] inline project::game_entity_v1 game_entity(ecs::entity entity) noexcept
{
    return {entity.index, entity.generation};
}

inline bool has_core_component_unchecked(const ecs::world& world, ecs::entity entity,
                                         project::game_core_component_v1 component) noexcept
{
    switch (component)
    {
        case project::game_core_component_v1::name:
            return world.has<scene::name_component>(entity);
        case project::game_core_component_v1::transform:
            return world.has<scene::transform_component>(entity);
        case project::game_core_component_v1::tag:
            return world.has<scene::tag_component>(entity);
        case project::game_core_component_v1::active:
            return world.has<scene::active_component>(entity);
    }
    return false;
}

inline project::game_entity_target_v1 create_runtime_entity(void* user_data) noexcept
{
    if (!user_data) return {};
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!bridge.world) return {};
    if (bridge.commands)
    {
        const auto entity = bridge.commands->create();
        return {.deferred = {entity.buffer, entity.ordinal}, .is_deferred = true};
    }
    return {.entity = game_entity(bridge.world->create())};
}

inline bool destroy_runtime_entity(void* user_data, project::game_entity_target_v1 target) noexcept
{
    if (!user_data || !target.valid()) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!bridge.world) return false;
    if (bridge.commands)
    {
        if (target.is_deferred)
            bridge.commands->destroy(ecs::deferred_entity{target.deferred.buffer, target.deferred.ordinal});
        else
        {
            const auto entity = native_entity(target.entity);
            if (!bridge.world->alive(entity)) return false;
            bridge.commands->destroy(entity);
        }
        return true;
    }
    return !target.is_deferred && bridge.world->destroy(native_entity(target.entity));
}

inline bool runtime_entity_alive(void* user_data, project::game_entity_v1 entity) noexcept
{
    if (!user_data || !entity.valid()) return false;
    const auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    return bridge.world && bridge.world->alive(native_entity(entity));
}

inline std::size_t runtime_entity_count(void* user_data) noexcept
{
    if (!user_data) return 0;
    const auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    return bridge.world ? bridge.world->live_count() : 0;
}

inline bool query_runtime_entities(void* user_data, const project::game_world_query_v1* query, void* visitor_user_data,
                                   project::game_visit_entity_v1 visitor) noexcept
{
    if (!user_data || !query || query->structure_size < sizeof(project::game_world_query_v1) || !visitor) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!bridge.world) return false;

    constexpr auto known_mask = project::game_core_component_bit_v1(project::game_core_component_v1::name) |
                                project::game_core_component_bit_v1(project::game_core_component_v1::transform) |
                                project::game_core_component_bit_v1(project::game_core_component_v1::tag) |
                                project::game_core_component_bit_v1(project::game_core_component_v1::active);
    if (((query->required_core_components | query->excluded_core_components) & ~known_mask) != 0) return false;

    constexpr project::game_core_component_v1 components[]{
        project::game_core_component_v1::name, project::game_core_component_v1::transform,
        project::game_core_component_v1::tag, project::game_core_component_v1::active};
    for (const auto component : components)
    {
        const auto bit = project::game_core_component_bit_v1(component);
        if (((query->required_core_components | query->excluded_core_components) & bit) != 0 &&
            !require_core_component_access(bridge, component, project::game_core_component_access_mode_v1::read))
            return false;
    }

    for (const auto entity : bridge.world->entities())
    {
        bool matches = true;
        for (const auto component : components)
        {
            const auto bit = project::game_core_component_bit_v1(component);
            const bool present = has_core_component_unchecked(*bridge.world, entity, component);
            if ((query->required_core_components & bit) != 0 && !present) matches = false;
            if ((query->excluded_core_components & bit) != 0 && present) matches = false;
        }
        if (matches && !visitor(visitor_user_data, game_entity(entity))) break;
    }
    return true;
}

inline bool has_runtime_core_component(void* user_data, project::game_entity_v1 entity,
                                       project::game_core_component_v1 component) noexcept
{
    if (!user_data || !entity.valid()) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!bridge.world ||
        !require_core_component_access(bridge, component, project::game_core_component_access_mode_v1::read))
        return false;
    const auto value = native_entity(entity);
    return bridge.world->alive(value) && has_core_component_unchecked(*bridge.world, value, component);
}

template <class Component>
inline bool set_runtime_core_component(runtime_world_bridge_context& bridge, project::game_entity_target_v1 target,
                                       Component component) noexcept
{
    if (!bridge.world || !target.valid()) return false;
    if (bridge.commands)
    {
        if (target.is_deferred)
        {
            bridge.commands->add<Component>(ecs::deferred_entity{target.deferred.buffer, target.deferred.ordinal},
                                            std::move(component));
            return true;
        }
        const auto entity = native_entity(target.entity);
        if (!bridge.world->alive(entity)) return false;
        if (auto* existing = bridge.world->try_get<Component>(entity))
        {
            *existing = std::move(component);
            return true;
        }
        bridge.commands->add<Component>(entity, std::move(component));
        return true;
    }
    if (target.is_deferred) return false;
    const auto entity = native_entity(target.entity);
    if (!bridge.world->alive(entity)) return false;
    bridge.world->emplace<Component>(entity, std::move(component));
    return true;
}

template <class Component>
inline bool remove_runtime_core_component_value(runtime_world_bridge_context& bridge,
                                                project::game_entity_target_v1 target) noexcept
{
    if (!bridge.world || !target.valid()) return false;
    if (bridge.commands)
    {
        if (target.is_deferred)
            bridge.commands->remove<Component>(ecs::deferred_entity{target.deferred.buffer, target.deferred.ordinal});
        else
        {
            const auto entity = native_entity(target.entity);
            if (!bridge.world->alive(entity) || !bridge.world->has<Component>(entity)) return false;
            bridge.commands->remove<Component>(entity);
        }
        return true;
    }
    return !target.is_deferred && bridge.world->remove<Component>(native_entity(target.entity));
}

inline bool remove_runtime_core_component(void* user_data, project::game_entity_target_v1 target,
                                          project::game_core_component_v1 component) noexcept
{
    if (!user_data) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!require_core_component_access(bridge, component, project::game_core_component_access_mode_v1::write))
        return false;
    switch (component)
    {
        case project::game_core_component_v1::name:
            return remove_runtime_core_component_value<scene::name_component>(bridge, target);
        case project::game_core_component_v1::transform:
            return remove_runtime_core_component_value<scene::transform_component>(bridge, target);
        case project::game_core_component_v1::tag:
            return remove_runtime_core_component_value<scene::tag_component>(bridge, target);
        case project::game_core_component_v1::active:
            return remove_runtime_core_component_value<scene::active_component>(bridge, target);
    }
    return false;
}

inline project::game_string_view_v1 read_runtime_name(void* user_data, project::game_entity_v1 entity) noexcept
{
    if (!user_data || !entity.valid()) return {};
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!bridge.world || !require_core_component_access(bridge, project::game_core_component_v1::name,
                                                        project::game_core_component_access_mode_v1::read))
        return {};
    const auto& world = std::as_const(*bridge.world);
    const auto* value = world.try_get<scene::name_component>(native_entity(entity));
    return value ? project::game_string_view_v1{value->value.data(), value->value.size()}
                 : project::game_string_view_v1{};
}

inline bool set_runtime_name(void* user_data, project::game_entity_target_v1 entity, const char* value,
                             std::size_t value_size) noexcept
{
    if (!user_data || (!value && value_size != 0)) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!require_core_component_access(bridge, project::game_core_component_v1::name,
                                       project::game_core_component_access_mode_v1::write))
        return false;
    std::string text;
    if (value_size != 0) text.assign(value, value_size);
    return set_runtime_core_component(bridge, entity, scene::name_component{std::move(text)});
}

inline bool read_runtime_transform(void* user_data, project::game_entity_v1 entity,
                                   project::game_transform_v1* value) noexcept
{
    if (!user_data || !entity.valid() || !value) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!bridge.world || !require_core_component_access(bridge, project::game_core_component_v1::transform,
                                                        project::game_core_component_access_mode_v1::read))
        return false;
    const auto& world = std::as_const(*bridge.world);
    const auto* transform = world.try_get<scene::transform_component>(native_entity(entity));
    if (!transform) return false;
    *value = {.position = {transform->position[0], transform->position[1], transform->position[2]},
              .rotation = {transform->rotation.x(), transform->rotation.y(), transform->rotation.z(),
                           transform->rotation.w()},
              .scale = {transform->scale[0], transform->scale[1], transform->scale[2]}};
    return true;
}

inline bool set_runtime_transform(void* user_data, project::game_entity_target_v1 entity,
                                  const project::game_transform_v1* value) noexcept
{
    if (!user_data || !value) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!require_core_component_access(bridge, project::game_core_component_v1::transform,
                                       project::game_core_component_access_mode_v1::write))
        return false;
    scene::transform_component transform;
    transform.position = {value->position.x, value->position.y, value->position.z};
    transform.rotation = {value->rotation.x, value->rotation.y, value->rotation.z, value->rotation.w};
    transform.scale = {value->scale.x, value->scale.y, value->scale.z};
    transform.dirty = true;
    return set_runtime_core_component(bridge, entity, std::move(transform));
}

inline project::game_string_view_v1 read_runtime_tag(void* user_data, project::game_entity_v1 entity) noexcept
{
    if (!user_data || !entity.valid()) return {};
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!bridge.world || !require_core_component_access(bridge, project::game_core_component_v1::tag,
                                                        project::game_core_component_access_mode_v1::read))
        return {};
    const auto& world = std::as_const(*bridge.world);
    const auto* value = world.try_get<scene::tag_component>(native_entity(entity));
    return value ? project::game_string_view_v1{value->value.data(), value->value.size()}
                 : project::game_string_view_v1{};
}

inline bool set_runtime_tag(void* user_data, project::game_entity_target_v1 entity, const char* value,
                            std::size_t value_size) noexcept
{
    if (!user_data || (!value && value_size != 0)) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!require_core_component_access(bridge, project::game_core_component_v1::tag,
                                       project::game_core_component_access_mode_v1::write))
        return false;
    std::string text;
    if (value_size != 0) text.assign(value, value_size);
    return set_runtime_core_component(bridge, entity, scene::tag_component{std::move(text)});
}

inline bool read_runtime_active(void* user_data, project::game_entity_v1 entity, bool* value) noexcept
{
    if (!user_data || !entity.valid() || !value) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!bridge.world || !require_core_component_access(bridge, project::game_core_component_v1::active,
                                                        project::game_core_component_access_mode_v1::read))
        return false;
    const auto& world = std::as_const(*bridge.world);
    const auto* active = world.try_get<scene::active_component>(native_entity(entity));
    if (!active) return false;
    *value = active->active;
    return true;
}

inline bool set_runtime_active(void* user_data, project::game_entity_target_v1 entity, bool value) noexcept
{
    if (!user_data) return false;
    auto& bridge = *static_cast<runtime_world_bridge_context*>(user_data);
    if (!require_core_component_access(bridge, project::game_core_component_v1::active,
                                       project::game_core_component_access_mode_v1::write))
        return false;
    return set_runtime_core_component(bridge, entity, scene::active_component{value});
}

inline project::game_world_api_v1 make_runtime_world_api(runtime_world_bridge_context& bridge) noexcept
{
    return {.user_data = &bridge,
            .create_entity = create_runtime_entity,
            .destroy_entity = destroy_runtime_entity,
            .entity_alive = runtime_entity_alive,
            .entity_count = runtime_entity_count,
            .query_entities = query_runtime_entities,
            .has_core_component = has_runtime_core_component,
            .remove_core_component = remove_runtime_core_component,
            .read_name = read_runtime_name,
            .set_name = set_runtime_name,
            .read_transform = read_runtime_transform,
            .set_transform = set_runtime_transform,
            .read_tag = read_runtime_tag,
            .set_tag = set_runtime_tag,
            .read_active = read_runtime_active,
            .set_active = set_runtime_active};
}

inline ecs::component_access scheduler_core_access(const project::game_core_component_access_v1& access) noexcept
{
    ecs::component_type_id component{};
    switch (access.component)
    {
        case project::game_core_component_v1::name:
            component = ecs::component_type<scene::name_component>();
            break;
        case project::game_core_component_v1::transform:
            component = ecs::component_type<scene::transform_component>();
            break;
        case project::game_core_component_v1::tag:
            component = ecs::component_type<scene::tag_component>();
            break;
        case project::game_core_component_v1::active:
            component = ecs::component_type<scene::active_component>();
            break;
    }
    return {component, access.mode == project::game_core_component_access_mode_v1::write
                           ? ecs::component_access_mode::write
                           : ecs::component_access_mode::read};
}

} // namespace arc::editor
