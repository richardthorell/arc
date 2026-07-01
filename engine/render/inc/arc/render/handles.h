#pragma once

#include <cstdint>
#include <vector>

namespace arc::render
{

/**
 * @brief Stable generational handle used for renderer-owned resources.
 */
struct resource_handle
{
    std::uint32_t index{ invalid_index };
    std::uint32_t generation{};

    static constexpr std::uint32_t invalid_index = 0xffffffffu;

    /**
     * @brief Return whether this handle references a possible resource slot.
     */
    constexpr bool valid() const noexcept
    {
        return index != invalid_index;
    }

    friend constexpr bool operator==(resource_handle lhs, resource_handle rhs) noexcept
    {
        return lhs.index == rhs.index && lhs.generation == rhs.generation;
    }

    friend constexpr bool operator!=(resource_handle lhs, resource_handle rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

/**
 * @brief Small resource pool that produces stale-detectable handles.
 */
class handle_pool
{
public:
    /**
     * @brief Allocate a new handle slot.
     */
    resource_handle allocate();

    /**
     * @brief Release a handle if it still matches the live generation.
     */
    bool release(resource_handle handle);

    /**
     * @brief Return whether a handle currently references a live slot.
     */
    bool alive(resource_handle handle) const;

    /**
     * @brief Return the number of live handles.
     */
    std::uint32_t live_count() const noexcept;

private:
    struct slot
    {
        std::uint32_t generation{ 1 };
        bool alive{};
    };

    std::vector<slot> slots_;
    std::vector<std::uint32_t> free_list_;
    std::uint32_t live_count_{};
};

using buffer_handle = resource_handle;
using texture_handle = resource_handle;
using sampler_handle = resource_handle;
using shader_handle = resource_handle;
using pipeline_handle = resource_handle;
using material_handle = resource_handle;
using mesh_handle = resource_handle;
using environment_handle = resource_handle;

/**
 * @brief Stable editor/game object id written into picking and outline buffers.
 */
struct render_object_id
{
    std::uint32_t index{ resource_handle::invalid_index };
    std::uint32_t generation{};

    /**
     * @brief Return whether this id can identify a scene object.
     */
    constexpr bool valid() const noexcept
    {
        return index != resource_handle::invalid_index;
    }

    friend constexpr bool operator==(render_object_id lhs, render_object_id rhs) noexcept
    {
        return lhs.index == rhs.index && lhs.generation == rhs.generation;
    }

    friend constexpr bool operator!=(render_object_id lhs, render_object_id rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

} // namespace arc::render
