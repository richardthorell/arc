#pragma once

/** Development-only Windows host runtime-thread application of authored edits.
 * Socket I/O is in arc-live-play-dev; this file is never built in Shipping.
 */
#include <arc/framework/runtime.h>
#include <arc/network/live_play_dev_edits.h>
#include <arc/scene/components.h>
#include <arc/scene/hierarchy.h>
#include <arc/scene/physics_transform_authority.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

namespace arc::platform::windows
{

class live_play_property_applier final
{
public:
    [[nodiscard]] network::live_play_dev::edit_result
    apply(framework::runtime& runtime, const network::live_play_dev::property_edit& edit, bool baseline_active)
    {
        using namespace network::live_play_dev;
        const auto reject = [&](std::string code) -> edit_result
        { return {.request_id = edit.request_id, .applied = false, .code = std::move(code)}; };
        if (!baseline_active) return reject("restart_required");
        if (runtime.worlds().executing()) return reject("busy");
        if (edit.operation != edit_operation::property) return apply_structural(runtime, edit);
        const bool is_transform = edit.component == "arc.transform";
        const bool directional = edit.component == "arc.light.directional";
        const bool point = edit.component == "arc.light.point";
        const bool spot = edit.component == "arc.light.spot";
        const bool area = edit.component == "arc.light.area";
        const bool is_light = directional || point || spot || area;
        if ((!is_transform && !is_light) || edit.schema_version != 1) return reject("schema_mismatch");
        // Begin/cancel are deliberate, authenticated transform authority transitions.
        // Light property changes retain the original update/commit-only contract.
        const bool acquire = is_transform && edit.phase == "begin";
        const bool release = is_transform && edit.phase == "cancel";
        if (edit.phase != "update" && edit.phase != "commit" && !acquire && !release)
            return reject("unsupported_operation");
        if (is_transform)
        {
            if ((edit.property != "position" && edit.property != "rotation" && edit.property != "scale") ||
                edit.value_count != (edit.property == "rotation" ? 4 : 3))
                return reject("unsupported_operation");
        }
        else
        {
            if ((edit.property != "intensity" && edit.property != "color") ||
                edit.value_count != (edit.property == "intensity" ? 1 : 3))
                return reject("unsupported_operation");
            for (std::uint8_t i = 0; i < edit.value_count; ++i)
                if (!std::isfinite(edit.values[i]) || edit.values[i] < 0.0f || edit.values[i] > 1000000.0f)
                    return reject("invalid_request");
        }
        if (edit.expected_runtime_revision > applied_revision_ || edit.authoring_revision == 0 ||
            edit.authoring_revision > 9007199254740991ULL)
            return reject("stale_revision");

        const auto entity_guid = ecs::parse_entity_guid(edit.entity_guid);
        if (!entity_guid) return reject("invalid_request");
        framework::runtime_world_id found_world_id{};
        ecs::world* found_world = nullptr;
        ecs::entity found_entity{};
        for (const auto id : runtime.worlds().ordered_worlds())
        {
            auto* world = runtime.worlds().find(id);
            if (!world || world->role() != framework::runtime_world_role::client ||
                (world->state() != framework::runtime_world_state::running &&
                 world->state() != framework::runtime_world_state::paused))
                continue;
            for (const auto entity : world->entities().entities())
            {
                const auto* identity = world->entities().try_get<scene::persistent_id_component>(entity);
                if (!identity || identity->value != *entity_guid) continue;
                if (found_world) return reject("invalid_request"); // ambiguous identity across worlds
                found_world_id = id;
                found_world = &world->entities();
                found_entity = entity;
            }
        }
        if (!found_world) return reject("unknown_entity");
        // A deleted/recreated entity may reuse an authored GUID; never let a
        // command from this baseline silently switch to the replacement ECS ID.
        const auto identity_key = ecs::to_string(*entity_guid);
        const auto structural = structural_entities_.find(identity_key);
        if (structural != structural_entities_.end() &&
            (structural->second.deleted || edit.authoring_revision <= structural->second.create_revision))
            return reject("stale_revision");
        const auto bound = identity_bindings_.find(identity_key);
        if (bound != identity_bindings_.end() &&
            (bound->second.world != found_world || bound->second.entity != found_entity))
            return reject("restart_required");
        const auto* current = found_world->try_get<scene::transform_component>(found_entity);
        if (is_transform && !current) return reject("unsupported_operation");
        if ((directional && !found_world->has<scene::directional_light_component>(found_entity)) ||
            (point && !found_world->has<scene::point_light_component>(found_entity)) ||
            (spot && !found_world->has<scene::spot_light_component>(found_entity)) ||
            (area && !found_world->has<scene::area_light_component>(found_entity)))
            return reject("unsupported_operation");
        const auto* mobility = found_world->try_get<scene::mobility_component>(found_entity);
        if (mobility && mobility->value != render::render_mobility::movable && !release)
            return reject("permission_denied");

        const std::string key = ecs::to_string(*entity_guid) + ":" + edit.component + ":" + edit.property;
        const auto previous = applied_authoring_.find(key);
        if (previous != applied_authoring_.end())
        {
            const auto& applied = previous->second;
            if (edit.authoring_revision < applied.revision) return reject("stale_revision");
            if (edit.authoring_revision == applied.revision)
            {
                const bool same_value =
                    edit.value_count == applied.value_count &&
                    std::equal(edit.values.begin(), edit.values.begin() + edit.value_count, applied.value.begin());
                if (!same_value) return reject("stale_revision");
                if (applied.phase == edit.phase)
                {
                    // A final commit's ACK may be lost; acknowledge identical
                    // replay without applying ECS twice or advancing revision.
                    return {
                        .request_id = edit.request_id, .applied = true, .applied_revision = applied.applied_revision};
                }
                if (applied.phase != "update" || edit.phase != "commit") return reject("stale_revision");
            }
        }
        // Gameplay and physics ownership use the same explicit hold policy.
        auto* physics = runtime.services().find<scene::physics_transform_authority_service>(
            scene::physics_transform_authority_service::service_id);
        const auto owner =
            physics ? physics->owner(found_world_id, found_entity) : scene::physics_transform_owner::unbound;
        const auto hold = held_transforms_.find(identity_key);
        if (is_transform)
        {
            if (release && hold == held_transforms_.end()) return reject("permission_denied");
            if (hold != held_transforms_.end() &&
                (hold->second.world != found_world || hold->second.entity != found_entity))
                return reject("restart_required");
            if (hold != held_transforms_.end() && hold->second.physics && !hold->second.physics->valid())
            {
                held_transforms_.erase(hold);
                return reject("restart_required");
            }
            if (!release && owner == scene::physics_transform_owner::static_body) return reject("permission_denied");
            if (!release && owner != scene::physics_transform_owner::unbound && edit.property == "scale")
                return reject("unsupported_operation");
            const auto* flow = found_world->try_get<scene::flow_component>(found_entity);
            const bool simulated = owner == scene::physics_transform_owner::simulated;
            if ((simulated || (flow && flow->enabled)) && !acquire && hold == held_transforms_.end())
                return reject("permission_denied");
            if (!release && simulated && hold != held_transforms_.end() && !hold->second.physics)
                return reject("restart_required");
            if (acquire && hold == held_transforms_.end() && held_transforms_.size() >= maximum_holds)
                return reject("busy");
        }
        if (applied_revision_ >= 9007199254740991ULL) return reject("resync_required");

        // ECS mutation occurs on the Windows game frame thread, outside system
        // iteration, after the network worker's bounded mailbox handoff.
        if (is_transform && !release)
        {
            auto updated = hold == held_transforms_.end() ? *current : hold->second.transform;
            if (edit.property == "rotation")
            {
                updated.set_rotation(math::quatf{edit.values[0], edit.values[1], edit.values[2], edit.values[3]});
            }
            else
            {
                const math::vector3f value{edit.values[0], edit.values[1], edit.values[2]};
                if (edit.property == "position")
                    updated.set_position(value);
                else
                    updated.set_scale(value);
            }
            std::unique_ptr<scene::physics_transform_hold> physics_hold;
            if (hold != held_transforms_.end() && hold->second.physics)
            {
                if (!hold->second.physics->set_pose(updated)) return reject("busy");
            }
            else if (acquire && owner == scene::physics_transform_owner::simulated)
            {
                physics_hold = physics->acquire(found_world_id, found_entity, updated);
                if (!physics_hold || !physics_hold->valid()) return reject("busy");
            }
            found_world->emplace<scene::transform_component>(found_entity, updated);
            scene::mark_transform_subtree_dirty(*found_world, found_entity);
            scene::update_world_transforms(*found_world);
            if (hold != held_transforms_.end())
                hold->second.transform = updated;
            else if (acquire)
                held_transforms_.emplace(identity_key,
                                         held_transform{found_world, found_entity, updated, std::move(physics_hold)});
        }
        else if (!release)
        {
            // Only these four known ECS components are mutable. Never write
            // arbitrary Inspector property paths or rebuild GPU objects on
            // the socket worker. Render extraction reads ECS on the next frame.
            const auto set_light = [&]<typename Light>() -> bool
            {
                const auto* component = found_world->try_get<Light>(found_entity);
                if (!component) return false;
                auto updated = *component;
                if (edit.property == "intensity")
                    updated.intensity = edit.values[0];
                else
                    updated.color = math::vector3f{edit.values[0], edit.values[1], edit.values[2]};
                found_world->emplace<Light>(found_entity, updated);
                return true;
            };
            const bool applied = directional ? set_light.template operator()<scene::directional_light_component>()
                                 : point     ? set_light.template operator()<scene::point_light_component>()
                                 : spot      ? set_light.template operator()<scene::spot_light_component>()
                                             : set_light.template operator()<scene::area_light_component>();
            if (!applied) return reject("unsupported_operation");
        }
        // Release preserves the final authored pose in ECS, then returns
        // ownership to gameplay on the next simulation tick. No source save.
        if (release) held_transforms_.erase(identity_key);
        identity_bindings_.try_emplace(identity_key, entity_binding{found_world, found_entity});
        const auto applied_revision = ++applied_revision_;
        applied_authoring_[key] = {.revision = edit.authoring_revision,
                                   .phase = edit.phase,
                                   .value = edit.values,
                                   .value_count = edit.value_count,
                                   .applied_revision = applied_revision};
        return {.request_id = edit.request_id, .applied = true, .applied_revision = applied_revision};
    }

    /** Reassert explicit holds after gameplay has advanced, before render/inspection.
     * Must run on the owning Windows runtime frame thread, never on socket I/O.
     * Dead worlds, entity generations and authored identities fail closed.
     */
    void after_simulation(framework::runtime& runtime, bool baseline_active)
    {
        if (!baseline_active)
        {
            clear();
            return;
        }
        if (runtime.worlds().executing()) return;
        for (auto it = held_transforms_.begin(); it != held_transforms_.end();)
        {
            auto& held = it->second;
            bool world_alive = false;
            for (const auto id : runtime.worlds().ordered_worlds())
            {
                auto* candidate = runtime.worlds().find(id);
                if (candidate && &candidate->entities() == held.world &&
                    candidate->role() == framework::runtime_world_role::client &&
                    (candidate->state() == framework::runtime_world_state::running ||
                     candidate->state() == framework::runtime_world_state::paused))
                {
                    world_alive = true;
                    break;
                }
            }
            if (!world_alive || !held.world->alive(held.entity) || (held.physics && !held.physics->valid()))
            {
                it = held_transforms_.erase(it);
                continue;
            }
            const auto* identity = held.world->try_get<scene::persistent_id_component>(held.entity);
            const auto* current = held.world->try_get<scene::transform_component>(held.entity);
            const auto* mobility = held.world->try_get<scene::mobility_component>(held.entity);
            if (!identity || ecs::to_string(identity->value) != it->first || !current ||
                (mobility && mobility->value != render::render_mobility::movable))
            {
                it = held_transforms_.erase(it);
                continue;
            }
            bool changed = false;
            for (int i = 0; i < 3; ++i)
                changed = changed || current->position[i] != held.transform.position[i] ||
                          current->scale[i] != held.transform.scale[i];
            for (int i = 0; i < 4; ++i)
                changed = changed || current->rotation[i] != held.transform.rotation[i];
            if (changed)
            {
                held.world->emplace<scene::transform_component>(held.entity, held.transform);
                scene::mark_transform_subtree_dirty(*held.world, held.entity);
                scene::update_world_transforms(*held.world);
            }
            ++it;
        }
    }

    [[nodiscard]] std::size_t held_entity_count() const noexcept
    {
        return held_transforms_.size();
    }

    void clear() noexcept
    {
        held_transforms_.clear();
        identity_bindings_.clear();
        applied_authoring_.clear();
        // Keep structural identity tombstones across a transport reconnect.
        // Their ACK revisions belong to the old baseline and cannot be replayed.
        ++structural_epoch_;
        applied_revision_ = 0;
    }

private:
    struct structural_entity
    {
        framework::runtime_world_id world_id{};
        ecs::entity entity{};
        std::uint64_t create_revision{};
        std::uint64_t create_ack{};
        std::uint64_t delete_expected{};
        std::uint64_t delete_ack{};
        bool deleted{};
        std::uint64_t epoch{};
    };

    [[nodiscard]] network::live_play_dev::edit_result
    apply_structural(framework::runtime& runtime, const network::live_play_dev::property_edit& edit)
    {
        using namespace network::live_play_dev;
        const auto reject = [&](std::string code) -> edit_result
        { return {.request_id = edit.request_id, .applied = false, .code = std::move(code)}; };
        const auto ack = [&](std::uint64_t revision) -> edit_result
        { return {.request_id = edit.request_id, .applied = true, .applied_revision = revision}; };
        const auto guid = ecs::parse_entity_guid(edit.entity_guid);
        if (!guid) return reject("invalid_request");
        const auto key = ecs::to_string(*guid);
        if (applied_revision_ >= 9007199254740991ULL) return reject("resync_required");
        const auto existing = structural_entities_.find(key);
        if (edit.operation == edit_operation::create)
        {
            if (edit.authoring_revision == 0 || edit.authoring_revision > 9007199254740991ULL)
                return reject("stale_revision");
            if (existing != structural_entities_.end())
            {
                const auto& record = existing->second;
                auto* world = runtime.worlds().find(record.world_id);
                if (record.epoch == structural_epoch_ && !record.deleted && record.create_revision == edit.authoring_revision &&
                    world && world->entities().alive(record.entity))
                    return ack(record.create_ack);
                return reject("restart_required");
            }
            if (structural_entities_.size() >= maximum_structural_entities || identity_bindings_.contains(key))
                return reject("restart_required");
            framework::runtime_world* target = nullptr;
            for (const auto id : runtime.worlds().ordered_worlds())
            {
                auto* world = runtime.worlds().find(id);
                if (!world) continue;
                for (const auto entity : world->entities().entities())
                {
                    const auto* identity = world->entities().try_get<scene::persistent_id_component>(entity);
                    if (identity && identity->value == *guid) return reject("invalid_request");
                }
                if (world->role() != framework::runtime_world_role::client ||
                    (world->state() != framework::runtime_world_state::running &&
                     world->state() != framework::runtime_world_state::paused))
                    continue;
                if (target) return reject("unsupported_operation");
                target = world;
            }
            if (!target) return reject("restart_required");
            auto& world = target->entities();
            const auto entity = world.create();
            world.emplace<scene::persistent_id_component>(entity, scene::persistent_id_component{*guid});
            world.emplace<scene::transform_component>(entity);
            world.emplace<scene::name_component>(entity, scene::name_component{"Entity"});
            scene::update_world_transforms(world);
            const auto revision = ++applied_revision_;
            structural_entities_.emplace(key, structural_entity{.world_id = target->id(),
                                                                 .entity = entity,
                                                                 .create_revision = edit.authoring_revision,
                                                                 .create_ack = revision,
                                                                 .epoch = structural_epoch_});
            identity_bindings_.try_emplace(key, entity_binding{&world, entity});
            return ack(revision);
        }
        if (edit.operation != edit_operation::erase) return reject("unsupported_operation");
        if (existing == structural_entities_.end()) return reject("unsupported_operation");
        auto& record = existing->second;
        if (record.deleted)
        {
            if (record.epoch == structural_epoch_ && edit.expected_runtime_revision == record.delete_expected)
                return ack(record.delete_ack);
            return reject("restart_required");
        }
        if (edit.expected_runtime_revision != applied_revision_) return reject("stale_revision");
        auto* target = runtime.worlds().find(record.world_id);
        if (!target || (target->state() != framework::runtime_world_state::running &&
                        target->state() != framework::runtime_world_state::paused))
            return reject("restart_required");
        auto& world = target->entities();
        if (!world.alive(record.entity)) return reject("restart_required");
        const auto* identity = world.try_get<scene::persistent_id_component>(record.entity);
        if (!identity || identity->value != *guid) return reject("restart_required");
        const auto* hierarchy = world.try_get<scene::hierarchy_component>(record.entity);
        if ((hierarchy && world.alive(hierarchy->parent)) || !scene::children(world, record.entity).empty() ||
            held_transforms_.contains(key))
            return reject("unsupported_operation");
        auto* physics = runtime.services().find<scene::physics_transform_authority_service>(
            scene::physics_transform_authority_service::service_id);
        if (physics && physics->owner(record.world_id, record.entity) != scene::physics_transform_owner::unbound)
            return reject("permission_denied");
        if (world.has<scene::flow_component>(record.entity)) return reject("permission_denied");
        scene::detach(world, record.entity);
        if (!world.destroy(record.entity)) return reject("restart_required");
        record.deleted = true;
        record.epoch = structural_epoch_;
        record.delete_expected = edit.expected_runtime_revision;
        record.delete_ack = ++applied_revision_;
        return ack(record.delete_ack);
    }

    struct applied_property
    {
        std::uint64_t revision{};
        std::string phase;
        std::array<float, 4> value{};
        std::uint8_t value_count{};
        std::uint64_t applied_revision{};
    };

    struct entity_binding
    {
        ecs::world* world{};
        ecs::entity entity{};
    };
    struct held_transform
    {
        ecs::world* world{};
        ecs::entity entity{};
        scene::transform_component transform{};
        std::unique_ptr<scene::physics_transform_hold> physics;
    };

    static constexpr std::size_t maximum_holds = 64;
    static constexpr std::size_t maximum_structural_entities = 256;
    std::unordered_map<std::string, structural_entity> structural_entities_;
    std::uint64_t structural_epoch_{};
    std::unordered_map<std::string, entity_binding> identity_bindings_;
    std::unordered_map<std::string, held_transform> held_transforms_;
    std::unordered_map<std::string, applied_property> applied_authoring_;
    std::uint64_t applied_revision_{};
};

} // namespace arc::platform::windows
