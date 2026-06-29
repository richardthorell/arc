#pragma once

#include <cstdint>

namespace arc::scene
{

/**
 * @brief Stale-detectable ECS entity identifier.
 */
struct entity
{
    std::uint32_t index{ invalid_index };
    std::uint32_t generation{};

    static constexpr std::uint32_t invalid_index = 0xffffffffu;

    /**
     * @brief Return whether this value may reference an entity slot.
     */
    constexpr bool valid() const noexcept
    {
        return index != invalid_index;
    }

    friend constexpr bool operator==(entity lhs, entity rhs) noexcept
    {
        return lhs.index == rhs.index && lhs.generation == rhs.generation;
    }

    friend constexpr bool operator!=(entity lhs, entity rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

} // namespace arc::scene
