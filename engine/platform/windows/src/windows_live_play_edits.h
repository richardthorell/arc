#pragma once

/** Development-only Windows host runtime-thread application of authored edits.
 * Socket I/O is in arc-live-play-dev; this file is never built in Shipping.
 */
#include <arc/framework/runtime.h>
#include <arc/network/live_play_dev_edits.h>
#include <arc/scene/components.h>
#include <arc/scene/hierarchy.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>

namespace arc::platform::windows
{

class live_play_transform_applier final
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
        if (edit.component != "arc.transform" || edit.schema_version != 1) return reject("schema_mismatch");
        if (edit.phase != "update" && edit.phase != "commit") return reject("unsupported_operation");
        if ((edit.property != "position" && edit.property != "rotation" && edit.property != "scale") ||
            edit.value_count != (edit.property == "rotation" ? 4 : 3))
            return reject("unsupported_operation");
        if (edit.expected_runtime_revision > applied_revision_ || edit.authoring_revision == 0 ||
            edit.authoring_revision > 9007199254740991ULL)
            return reject("stale_revision");

        const auto entity_guid = ecs::parse_entity_guid(edit.entity_guid);
        if (!entity_guid) return reject("invalid_request");
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
                found_world = &world->entities();
                found_entity = entity;
            }
        }
        if (!found_world) return reject("unknown_entity");
        const auto* current = found_world->try_get<scene::transform_component>(found_entity);
        if (!current) return reject("unsupported_operation");
        const auto* mobility = found_world->try_get<scene::mobility_component>(found_entity);
        if (mobility && mobility->value != render::render_mobility::movable) return reject("permission_denied");

        const std::string key = ecs::to_string(*entity_guid) + ":" + edit.property;
        const auto previous = applied_authoring_.find(key);
        if (previous != applied_authoring_.end())
        {
            const auto& applied = previous->second;
            if (edit.authoring_revision < applied.revision) return reject("stale_revision");
            if (edit.authoring_revision == applied.revision)
            {
                const bool same_value = edit.value_count == applied.value_count &&
                                        std::equal(edit.values.begin(),
                                                   edit.values.begin() + edit.value_count,
                                                   applied.value.begin());
                if (!same_value) return reject("stale_revision");
                if (applied.phase == edit.phase)
                {
                    // A final commit's ACK may be lost; acknowledge identical
                    // replay without applying ECS twice or advancing revision.
                    return {.request_id = edit.request_id,
                            .applied = true,
                            .applied_revision = applied.applied_revision};
                }
                if (applied.phase != "update" || edit.phase != "commit") return reject("stale_revision");
            }
        }
        if (applied_revision_ >= 9007199254740991ULL) return reject("resync_required");

        // ECS mutation occurs on the Windows game frame thread, outside system
        // iteration, after the network worker's bounded mailbox handoff.
        auto updated = *current;
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
        found_world->emplace<scene::transform_component>(found_entity, updated);
        scene::mark_transform_subtree_dirty(*found_world, found_entity);
        scene::update_world_transforms(*found_world);
        const auto applied_revision = ++applied_revision_;
        applied_authoring_[key] = {.revision = edit.authoring_revision,
                                   .phase = edit.phase,
                                   .value = edit.values,
                                   .value_count = edit.value_count,
                                   .applied_revision = applied_revision};
        return {.request_id = edit.request_id, .applied = true, .applied_revision = applied_revision};
    }

    void clear() noexcept
    {
        applied_authoring_.clear();
        applied_revision_ = 0;
    }

private:
    struct applied_property
    {
        std::uint64_t revision{};
        std::string phase;
        std::array<float, 4> value{};
        std::uint8_t value_count{};
        std::uint64_t applied_revision{};
    };

    std::unordered_map<std::string, applied_property> applied_authoring_;
    std::uint64_t applied_revision_{};
};

} // namespace arc::platform::windows
