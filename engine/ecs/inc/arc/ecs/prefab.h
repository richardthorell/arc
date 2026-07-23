#pragma once

#include <arc/ecs/hierarchy.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc::ecs
{

struct prefab_component_record
{
    component_type_id type{};
    std::uint32_t schema_version{ 1 };
    std::vector<std::byte> payload;
};

struct prefab_entity_record
{
    entity_guid local_id{};
    entity_guid parent_local_id{};
    std::uint32_t sibling_order{};
    std::vector<prefab_component_record> components;
};

struct prefab_asset
{
    static constexpr std::uint32_t current_format_version = 1;
    std::uint32_t format_version{ current_format_version };
    entity_guid guid{};
    std::string name;
    std::string project_relative_path;
    std::vector<prefab_entity_record> entities;
};

enum class prefab_override_kind : std::uint8_t
{
    field,
    component_added,
    component_removed,
    source_child_removed,
    instance_child_added
};

struct prefab_override_key
{
    entity_guid source_entity{};
    component_type_id component{};
    component_field_id field{};
    prefab_override_kind kind{ prefab_override_kind::field };

    friend bool operator==(const prefab_override_key&, const prefab_override_key&) = default;
};

struct prefab_override
{
    prefab_override_key key;
    std::vector<std::byte> value;
};

struct prefab_instance_component
{
    entity_guid prefab_guid{};
    std::string prefab_path;
    entity_guid source_root{};
    std::vector<std::pair<entity_guid, entity_guid>> source_to_instance;
    std::vector<prefab_override> overrides;
    bool source_missing{};
};

template <>
struct component_traits<prefab_instance_component>
{
    static constexpr bool reflected = true;
    static constexpr std::string_view canonical_name = "arc.ecs.prefab_instance";
    static constexpr component_type_id id{ 0xa7c0000000000000ull, 0x0000000000000003ull };
    static constexpr std::array<component_field_descriptor, 6> fields{{
        { 1, "prefab_guid", "Prefab GUID", reflected_field_kind::structure,
            reflected_field_flags::serialized },
        { 2, "prefab_path", "Prefab", reflected_field_kind::asset_reference,
            reflected_field_flags::serialized | reflected_field_flags::editable },
        { 3, "source_root", "Source Root", reflected_field_kind::structure,
            reflected_field_flags::serialized },
        { 4, "source_to_instance", "Entity Mapping", reflected_field_kind::sequence,
            reflected_field_flags::serialized },
        { 5, "overrides", "Overrides", reflected_field_kind::sequence,
            reflected_field_flags::serialized },
        { 6, "source_missing", "Source Missing", reflected_field_kind::boolean,
            reflected_field_flags::transient }
    }};
    static constexpr component_descriptor descriptor{
        id, canonical_name, "Prefab Instance", 1, sizeof(prefab_instance_component),
        alignof(prefab_instance_component), fields, true, false
    };
};

struct prefab_propagation_result
{
    std::size_t fields_updated{};
    std::size_t entities_added{};
    std::size_t entities_removed{};
    std::size_t overrides_preserved{};
    std::vector<std::string> diagnostics;

    bool succeeded() const noexcept { return diagnostics.empty(); }
};

inline bool has_prefab_override(
    const prefab_instance_component& instance,
    const prefab_override_key& key) noexcept
{
    for (const prefab_override& override_value : instance.overrides)
        if (override_value.key == key)
            return true;
    return false;
}

inline bool set_prefab_override(prefab_instance_component& instance, prefab_override value)
{
    for (prefab_override& existing : instance.overrides)
    {
        if (existing.key == value.key)
        {
            existing.value = std::move(value.value);
            return false;
        }
    }
    instance.overrides.push_back(std::move(value));
    return true;
}

inline bool revert_prefab_override(
    prefab_instance_component& instance,
    const prefab_override_key& key)
{
    const auto found = std::find_if(instance.overrides.begin(), instance.overrides.end(),
        [&key](const prefab_override& value) { return value.key == key; });
    if (found == instance.overrides.end())
        return false;
    instance.overrides.erase(found);
    return true;
}

} // namespace arc::ecs
