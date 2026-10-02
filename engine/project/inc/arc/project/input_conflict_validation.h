#pragma once

#include <arc/input/conflict.h>
#include <arc/project/input_config.h>

#include <cstddef>
#include <string>
#include <vector>

namespace arc::project
{

/**
 * @brief Flatten project-authored action bindings into the shared input conflict model.
 *
 * Stable synthetic binding IDs are derived from context/action/index so diagnostics can be
 * correlated back to Config/Input.json without introducing a second conflict implementation.
 */
[[nodiscard]] inline std::vector<input::input_mapping_binding>
input_config_conflict_bindings(const input_config& config)
{
    std::vector<input::input_mapping_binding> mappings;
    for (const input_context_config& context : config.contexts)
    {
        for (const input_action_config& action : context.actions)
        {
            for (std::size_t index = 0; index < action.bindings.size(); ++index)
            {
                mappings.push_back({context.name, context.priority, context.enabled, action.name,
                                    context.name + "/" + action.name + "/" + std::to_string(index),
                                    action.bindings[index]});
            }
        }
    }
    return mappings;
}

/**
 * @brief Enumerate conflicts in project-authored action mappings using the runtime conflict API.
 *
 * The function is validation-only and never mutates the project configuration. Each unordered
 * binding pair is reported once, preserving deterministic config order.
 */
[[nodiscard]] inline std::vector<input::input_binding_conflict>
validate_input_config_conflicts(const input_config& config)
{
    const std::vector<input::input_mapping_binding> mappings = input_config_conflict_bindings(config);
    std::vector<input::input_binding_conflict> conflicts;

    for (std::size_t index = 0; index < mappings.size(); ++index)
    {
        const std::vector<input::input_mapping_binding> prior(mappings.begin(), mappings.begin() + index);
        std::vector<input::input_binding_conflict> current = input::enumerate_binding_conflicts(prior, mappings[index]);
        conflicts.insert(conflicts.end(), current.begin(), current.end());
    }

    return conflicts;
}

} // namespace arc::project
