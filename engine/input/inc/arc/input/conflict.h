#pragma once

#include <arc/input/input.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace arc::input
{

enum class input_conflict_kind : std::uint8_t
{
    same_context,
    layered_context,
    chord_component
};

enum class input_conflict_policy : std::uint8_t
{
    allow,
    warn,
    reject,
    replace,
    unbind_previous
};

struct input_mapping_binding
{
    std::string context;
    int context_priority{};
    bool context_enabled{true};
    std::string action;
    std::string binding_id;
    input_binding binding;
};

struct input_binding_conflict
{
    std::string existing_binding_id;
    std::string candidate_binding_id;
    std::string existing_context;
    std::string candidate_context;
    std::string existing_action;
    std::string candidate_action;
    input_control control{};
    input_conflict_kind kind{input_conflict_kind::same_context};
    bool ambiguous{};
};

struct input_conflict_resolution
{
    bool accepted{};
    std::vector<input_mapping_binding> bindings;
    std::vector<input_binding_conflict> conflicts;
};

namespace detail
{
struct binding_control
{
    input_device_type device{input_device_type::unknown};
    input_control control{};
    bool modifier{};
};

inline void collect_binding_controls(const input_binding& binding, std::vector<binding_control>& controls,
                                     bool modifier = false)
{
    controls.push_back({binding.device, binding.control, modifier});
    for (const input_binding& nested : binding.modifiers)
    {
        collect_binding_controls(nested, controls, true);
    }
}

[[nodiscard]] inline bool same_physical_control(const binding_control& lhs, const binding_control& rhs) noexcept
{
    return lhs.device == rhs.device && lhs.control == rhs.control;
}
} // namespace detail

/**
 * @brief Enumerate conflicts for one candidate without mutating the effective mapping set.
 *
 * Disabled contexts do not conflict. Reuse across different-priority contexts is reported as
 * intentional layered overlap rather than ambiguity. Chord modifiers participate in analysis,
 * while physically distinct device families never alias each other.
 */
[[nodiscard]] inline std::vector<input_binding_conflict>
enumerate_binding_conflicts(const std::vector<input_mapping_binding>& mappings, const input_mapping_binding& candidate)
{
    std::vector<input_binding_conflict> result;
    if (!candidate.context_enabled)
    {
        return result;
    }

    std::vector<detail::binding_control> candidate_controls;
    detail::collect_binding_controls(candidate.binding, candidate_controls);

    for (const input_mapping_binding& existing : mappings)
    {
        if (!existing.context_enabled || existing.binding_id == candidate.binding_id)
        {
            continue;
        }

        std::vector<detail::binding_control> existing_controls;
        detail::collect_binding_controls(existing.binding, existing_controls);
        for (const detail::binding_control& lhs : existing_controls)
        {
            for (const detail::binding_control& rhs : candidate_controls)
            {
                if (!detail::same_physical_control(lhs, rhs))
                {
                    continue;
                }

                const bool same_context = existing.context == candidate.context;
                const bool same_priority = existing.context_priority == candidate.context_priority;
                const input_conflict_kind kind = (lhs.modifier || rhs.modifier)
                                                     ? input_conflict_kind::chord_component
                                                     : (same_context ? input_conflict_kind::same_context
                                                                     : input_conflict_kind::layered_context);
                result.push_back({existing.binding_id,
                                  candidate.binding_id,
                                  existing.context,
                                  candidate.context,
                                  existing.action,
                                  candidate.action,
                                  rhs.control,
                                  kind,
                                  same_context || same_priority});
            }
        }
    }

    return result;
}

/**
 * @brief Apply an explicit conflict policy deterministically.
 *
 * Reject only blocks ambiguous conflicts. Replace/unbind_previous remove prior bindings that
 * ambiguously collide with the candidate; intentional priority layering is preserved.
 */
[[nodiscard]] inline input_conflict_resolution
apply_binding_conflict_policy(const std::vector<input_mapping_binding>& mappings, input_mapping_binding candidate,
                              input_conflict_policy policy)
{
    input_conflict_resolution resolution;
    resolution.bindings = mappings;
    resolution.conflicts = enumerate_binding_conflicts(mappings, candidate);
    const bool ambiguous = std::any_of(resolution.conflicts.begin(), resolution.conflicts.end(),
                                       [](const input_binding_conflict& conflict) { return conflict.ambiguous; });

    if (policy == input_conflict_policy::reject && ambiguous)
    {
        return resolution;
    }

    if ((policy == input_conflict_policy::replace || policy == input_conflict_policy::unbind_previous) && ambiguous)
    {
        resolution.bindings.erase(
            std::remove_if(resolution.bindings.begin(), resolution.bindings.end(), [&](const input_mapping_binding& item) {
                return std::any_of(resolution.conflicts.begin(), resolution.conflicts.end(),
                                   [&](const input_binding_conflict& conflict) {
                                       return conflict.ambiguous && conflict.existing_binding_id == item.binding_id;
                                   });
            }),
            resolution.bindings.end());
    }

    resolution.bindings.push_back(std::move(candidate));
    resolution.accepted = true;
    return resolution;
}

} // namespace arc::input
