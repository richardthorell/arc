#pragma once

#include <arc/core/id.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace arc::ecs
{

struct entity_guid_tag;

/** Stable 128-bit identity used by documents, prefabs, and cross-world references. */
using entity_guid = core::uuid<entity_guid_tag>;
static_assert(sizeof(entity_guid) == 16);
static_assert(std::is_standard_layout_v<entity_guid>);
static_assert(std::is_trivially_copyable_v<entity_guid>);

struct entity_guid_hash
{
    std::size_t operator()(entity_guid value) const noexcept
    {
        return core::uuid_hash<entity_guid_tag>{}(value);
    }
};

entity_guid generate_entity_guid() noexcept;
std::string to_string(entity_guid value);
std::optional<entity_guid> parse_entity_guid(std::string_view value) noexcept;

} // namespace arc::ecs
