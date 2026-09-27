#pragma once

#include "project_module_loader.h"

#include <algorithm>
#include <string>
#include <vector>

namespace arc::editor
{

enum class module_reload_reason : std::uint8_t
{
    none,
    component_removed,
    component_schema_downgraded,
    field_kind_changed
};

struct module_reload_diagnostic
{
    module_reload_classification classification{module_reload_classification::safe_hot_reload};
    module_reload_reason reason{module_reload_reason::none};
    std::string component_id;
    std::string component_name;
    std::uint64_t field_id{};
    std::string field_name;
    std::string message;
};

[[nodiscard]] inline module_reload_diagnostic
analyze_project_module_reload(const std::vector<project_component_schema>& previous,
                              const std::vector<project_component_schema>& next)
{
    if (previous.empty())
        return {.classification = module_reload_classification::initial_load,
                .message = "Project module is being loaded for the first time"};

    for (const auto& old_component : previous)
    {
        const auto component = std::find_if(next.begin(), next.end(), [&](const auto& candidate)
                                            { return candidate.stable_id == old_component.stable_id; });
        if (component == next.end())
            return {.classification = module_reload_classification::native_host_restart_required,
                    .reason = module_reload_reason::component_removed,
                    .component_id = old_component.stable_id,
                    .component_name = old_component.display_name,
                    .message = "Component '" + old_component.display_name +
                               "' was removed; restart the native host before loading this module generation"};

        if (component->schema_version < old_component.schema_version)
            return {.classification = module_reload_classification::native_host_restart_required,
                    .reason = module_reload_reason::component_schema_downgraded,
                    .component_id = old_component.stable_id,
                    .component_name = old_component.display_name,
                    .message = "Component '" + old_component.display_name + "' schema version decreased from " +
                               std::to_string(old_component.schema_version) + " to " +
                               std::to_string(component->schema_version) +
                               "; restart the native host before loading this module generation"};

        for (const auto& old_field : old_component.fields)
        {
            const auto field = std::find_if(component->fields.begin(), component->fields.end(), [&](const auto& candidate)
                                            { return candidate.stable_id == old_field.stable_id; });
            if (field != component->fields.end() && field->kind != old_field.kind)
                return {.classification = module_reload_classification::play_session_restart_required,
                        .reason = module_reload_reason::field_kind_changed,
                        .component_id = old_component.stable_id,
                        .component_name = old_component.display_name,
                        .field_id = old_field.stable_id,
                        .field_name = old_field.display_name,
                        .message = "Field '" + old_component.display_name + "." + old_field.display_name +
                                   "' changed type; restart the active Play session to use this module generation"};
        }
    }

    return {.classification = module_reload_classification::safe_hot_reload,
            .message = "Project module schema is compatible with hot reload"};
}

} // namespace arc::editor
