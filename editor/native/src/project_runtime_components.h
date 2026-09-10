#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace arc::editor
{

struct project_runtime_component_value
{
    std::string stable_id;
    std::string canonical_name;
    std::uint32_t schema_version{1};
    std::string json;
};

/** Play-only component bag attached to the runtime copy of an authored entity. */
struct project_runtime_component_set
{
    std::vector<project_runtime_component_value> values;
};

inline project_runtime_component_value* find_project_runtime_component(project_runtime_component_set& set,
                                                                       std::string_view identity) noexcept
{
    for (auto& value : set.values)
        if (value.stable_id == identity || value.canonical_name == identity) return &value;
    return nullptr;
}

inline const project_runtime_component_value* find_project_runtime_component(const project_runtime_component_set& set,
                                                                             std::string_view identity) noexcept
{
    for (const auto& value : set.values)
        if (value.stable_id == identity || value.canonical_name == identity) return &value;
    return nullptr;
}

} // namespace arc::editor
