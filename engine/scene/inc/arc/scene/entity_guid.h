#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace arc::scene
{

/** Stable identity used by scene documents and cross-entity references. */
struct entity_guid
{
    std::uint64_t high{};
    std::uint64_t low{};

    constexpr bool valid() const noexcept { return high != 0 || low != 0; }
    friend constexpr bool operator==(entity_guid, entity_guid) noexcept = default;
};

entity_guid generate_entity_guid() noexcept;
std::string to_string(entity_guid value);
std::optional<entity_guid> parse_entity_guid(std::string_view value) noexcept;

} // namespace arc::scene
