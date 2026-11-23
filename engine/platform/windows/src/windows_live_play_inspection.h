#pragma once

/** L1.5a: read-only Development inspection of the actual game runtime ECS.
 * Access only on the host frame thread, never from socket callbacks.
 */
#include <arc/framework/runtime.h>
#include <arc/network/live_play_dev_inspection.h>
#include <arc/scene/components.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace arc::platform::windows
{
class live_play_runtime_inspector final
{
public:
    using entity_ref = network::live_play_dev::inspection_entity;
    using hierarchy_item = network::live_play_dev::hierarchy_item;
    using hierarchy_snapshot = network::live_play_dev::hierarchy_snapshot;
    using inspection_request = network::live_play_dev::inspection_request;
    using inspection_outcome = network::live_play_dev::inspection_outcome;
    using inspection_property = network::live_play_dev::inspection_property;
    using inspection_value_kind = network::live_play_dev::inspection_value_kind;

    /** Stable for an ECS entity lifetime, namespaced by runtime world epoch.
     * The Live Play envelope additionally scopes it to session and instance.
     */
    [[nodiscard]] static std::string runtime_id(const framework::runtime_world& world, ecs::entity entity)
    {
        return "w" + std::to_string(world.id().value) + "-x" + std::to_string(world.epoch()) + "-e" +
               std::to_string(entity.index) + "-g" + std::to_string(entity.generation);
    }

    [[nodiscard]] static entity_ref reference(const framework::runtime_world& world, ecs::entity entity)
    {
        const auto& entities = static_cast<const ecs::world&>(world.entities());
        if (const auto* authored = entities.try_get<scene::persistent_id_component>(entity))
            if (authored->value.valid()) return {true, ecs::to_string(authored->value)};
        return {false, runtime_id(world, entity)};
    }

    /** Returns changed hierarchy snapshots capped at 4096 runtime entities.
     * Oversized scenes are explicitly marked incomplete, never silently accepted.
     */
    [[nodiscard]] std::optional<hierarchy_snapshot> sample(const framework::runtime& runtime)
    {
        if (runtime.worlds().executing()) return std::nullopt;
        hierarchy_snapshot snapshot;
        for (const auto id : runtime.worlds().ordered_worlds())
        {
            const auto* world = runtime.worlds().find(id);
            if (!world || world->role() != framework::runtime_world_role::client ||
                (world->state() != framework::runtime_world_state::running &&
                 world->state() != framework::runtime_world_state::paused))
                continue;
            const auto& entities = static_cast<const ecs::world&>(world->entities());
            for (const auto entity : entities.entities())
            {
                if (snapshot.entities.size() == 4096)
                {
                    snapshot.complete = false;
                    break;
                }
                hierarchy_item item;
                item.entity = reference(*world, entity);
                if (const auto* links = entities.try_get<scene::hierarchy_component>(entity))
                    if (entities.alive(links->parent)) item.parent = reference(*world, links->parent);
                if (const auto* name = entities.try_get<scene::name_component>(entity))
                    item.name = name->value.substr(0, 256);
                snapshot.entities.push_back(std::move(item));
            }
            if (!snapshot.complete) break;
        }
        if (initialized_ && previous_ == snapshot.entities && complete_ == snapshot.complete) return std::nullopt;
        initialized_ = true;
        previous_ = snapshot.entities;
        complete_ = snapshot.complete;
        snapshot.world_revision = ++world_revision_;
        return snapshot;
    }

    /** Inspect one component at a safe frame boundary, never serialize
     * opaque gameplay component pointers or raw ECS memory.
     */
    [[nodiscard]] inspection_outcome inspect(const framework::runtime& runtime, const inspection_request& request) const
    {
        using namespace network::live_play_dev;
        const auto reject = [&](std::string code) -> inspection_outcome
        { return {.request_id = request.request_id, .error_code = std::move(code)}; };
        if (runtime.worlds().executing()) return reject("busy");
        if (request.schema_version != 1) return reject("schema_mismatch");

        const framework::runtime_world* selected_world = nullptr;
        ecs::entity selected{};
        std::size_t visited = 0;
        for (const auto id : runtime.worlds().ordered_worlds())
        {
            const auto* world = runtime.worlds().find(id);
            if (!world || world->role() != framework::runtime_world_role::client ||
                (world->state() != framework::runtime_world_state::running &&
                 world->state() != framework::runtime_world_state::paused))
                continue;
            const auto& entities = static_cast<const ecs::world&>(world->entities());
            for (const auto entity : entities.entities())
            {
                if (++visited > 16384) return reject("busy");
                if (reference(*world, entity) != request.entity) continue;
                if (selected_world) return reject("resync_required");
                selected_world = world;
                selected = entity;
            }
        }
        if (!selected_world || !selected.valid()) return reject("unknown_entity");
        const auto& entities = static_cast<const ecs::world&>(selected_world->entities());
        component_inspection value;
        value.request = request;
        value.runtime_revision = std::min<std::uint64_t>(runtime.current_tick().id.value, 9007199254740991ULL);
        if (request.component == "arc.transform")
        {
            const auto* transform = entities.try_get<scene::transform_component>(selected);
            if (!transform) return reject("unsupported_operation");
            value.properties = {
                {.path = "position",
                 .numbers = {transform->position[0], transform->position[1], transform->position[2]}},
                {.path = "rotation",
                 .numbers = {transform->rotation[0], transform->rotation[1], transform->rotation[2],
                             transform->rotation[3]}},
                {.path = "scale", .numbers = {transform->scale[0], transform->scale[1], transform->scale[2]}},
            };
        }
        else if (request.component == "arc.scene.name")
        {
            const auto* name = entities.try_get<scene::name_component>(selected);
            if (!name) return reject("unsupported_operation");
            value.properties = {
                {.path = "value", .kind = inspection_value_kind::text, .text = name->value.substr(0, 2048)}};
        }
        else if (request.component == "arc.scene.tag")
        {
            const auto* tag = entities.try_get<scene::tag_component>(selected);
            if (!tag) return reject("unsupported_operation");
            value.properties = {
                {.path = "value", .kind = inspection_value_kind::text, .text = tag->value.substr(0, 2048)}};
        }
        else if (request.component == "arc.scene.active")
        {
            const auto* active = entities.try_get<scene::active_component>(selected);
            if (!active) return reject("unsupported_operation");
            value.properties = {{.path = "active", .kind = inspection_value_kind::boolean, .flag = active->active}};
        }
        else
            return reject("unsupported_operation");
        return {.request_id = request.request_id, .result = std::move(value)};
    }

    void reset() noexcept
    {
        previous_.clear();
        complete_ = true;
        initialized_ = false;
        world_revision_ = 0;
    }

private:
    std::vector<hierarchy_item> previous_;
    std::uint64_t world_revision_{};
    bool initialized_{};
    bool complete_{true};
};
} // namespace arc::platform::windows
