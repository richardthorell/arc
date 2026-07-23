#pragma once

#include <cstdint>
#include <functional>

namespace arc::ecs
{

/** Transient, generation-checked entity handle. Persisted references use entity_guid. */
struct entity
{
    std::uint32_t index{ invalid_index };
    std::uint32_t generation{};

    static constexpr std::uint32_t invalid_index = 0xffffffffu;

    constexpr bool valid() const noexcept { return index != invalid_index; }
    friend constexpr bool operator==(entity, entity) noexcept = default;
};

struct entity_hash
{
    std::size_t operator()(entity value) const noexcept
    {
        return (static_cast<std::size_t>(value.generation) << 32u) ^ value.index;
    }
};

} // namespace arc::ecs
