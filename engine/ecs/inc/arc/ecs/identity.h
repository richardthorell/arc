#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace arc::ecs
{

/** Stable 128-bit identity used by documents, prefabs, and cross-world references. */
struct entity_guid
{
    std::uint64_t high{};
    std::uint64_t low{};

    constexpr bool valid() const noexcept { return high != 0 || low != 0; }
    friend constexpr bool operator==(entity_guid, entity_guid) noexcept = default;
};

struct entity_guid_hash
{
    std::size_t operator()(entity_guid value) const noexcept
    {
        return static_cast<std::size_t>(value.high ^ (value.low + 0x9e3779b97f4a7c15ull +
            (value.high << 6u) + (value.high >> 2u)));
    }
};

entity_guid generate_entity_guid() noexcept;
std::string to_string(entity_guid value);
std::optional<entity_guid> parse_entity_guid(std::string_view value) noexcept;

} // namespace arc::ecs
