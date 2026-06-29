#pragma once

#include <arc/render/handles.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc::render
{

/**
 * @brief Descriptor/resource slot categories used by backend descriptor managers.
 */
enum class descriptor_resource_type : std::uint8_t
{
    sampled_image,
    storage_image,
    uniform_buffer,
    storage_buffer,
    sampler
};

/**
 * @brief Stable descriptor slot returned by a backend descriptor manager.
 */
struct descriptor_slot
{
    descriptor_resource_type type{ descriptor_resource_type::sampled_image };
    std::uint32_t index{ resource_handle::invalid_index };
    std::uint32_t generation{};

    /**
     * @brief Return whether this slot may reference a live descriptor.
     */
    constexpr bool valid() const noexcept
    {
        return index != resource_handle::invalid_index;
    }
};

/**
 * @brief Backend-neutral descriptor slot allocator with stale-slot detection.
 */
class descriptor_slot_pool
{
public:
    /**
     * @brief Allocate one descriptor slot of the requested type.
     */
    descriptor_slot allocate(descriptor_resource_type type);

    /**
     * @brief Release a slot if its generation is still current.
     */
    bool release(descriptor_slot slot);

    /**
     * @brief Return whether a slot is currently live.
     */
    bool alive(descriptor_slot slot) const;

    /**
     * @brief Return the number of live slots.
     */
    std::uint32_t live_count() const noexcept;

private:
    struct slot_state
    {
        descriptor_resource_type type{ descriptor_resource_type::sampled_image };
        std::uint32_t generation{ 1 };
        bool alive{};
    };

    std::vector<slot_state> slots_;
    std::vector<std::uint32_t> free_list_;
    std::uint32_t live_count_{};
};

/**
 * @brief Queue for resource destruction after the GPU has retired a frame.
 */
class deferred_resource_releaser
{
public:
    using release_fn = std::function<void()>;

    /**
     * @brief Schedule a release callback after the given frame has completed.
     */
    void defer(std::uint64_t retire_after_frame, release_fn release);

    /**
     * @brief Run callbacks whose retire frame is no newer than completed_frame.
     */
    std::size_t collect(std::uint64_t completed_frame);

    /**
     * @brief Return how many callbacks are waiting for retirement.
     */
    std::size_t pending_count() const noexcept;

private:
    struct pending_release
    {
        std::uint64_t retire_after_frame{};
        release_fn release;
    };

    std::vector<pending_release> pending_;
};

/**
 * @brief Simple bump allocator for per-frame transient CPU data.
 */
class frame_allocator
{
public:
    explicit frame_allocator(std::size_t capacity = 0);

    /**
     * @brief Allocate aligned bytes from the frame arena.
     */
    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t));

    /**
     * @brief Reset all transient allocations.
     */
    void reset() noexcept;

    /**
     * @brief Return used bytes in this frame.
     */
    std::size_t used() const noexcept;

    /**
     * @brief Return total reserved bytes.
     */
    std::size_t capacity() const noexcept;

private:
    std::vector<std::byte> storage_;
    std::size_t offset_{};
};

/**
 * @brief Stable key for backend graphics pipeline caching.
 */
struct graphics_pipeline_key
{
    shader_handle vertex_shader{};
    shader_handle fragment_shader{};
    std::string vertex_layout;
    std::string color_format;
    std::string depth_format;
    bool depth_test{};
    bool depth_write{};
    bool wireframe{};
    bool alpha_blend{};

    friend bool operator==(const graphics_pipeline_key& lhs, const graphics_pipeline_key& rhs) noexcept;
};

/**
 * @brief In-memory pipeline handle cache shared by backend implementations.
 */
class pipeline_handle_cache
{
public:
    /**
     * @brief Return a cached pipeline handle or an invalid handle.
     */
    pipeline_handle find(const graphics_pipeline_key& key) const;

    /**
     * @brief Insert or replace a pipeline key mapping.
     */
    void insert(graphics_pipeline_key key, pipeline_handle handle);

    /**
     * @brief Remove all cached mappings.
     */
    void clear();

    /**
     * @brief Return cached pipeline count.
     */
    std::size_t size() const noexcept;

private:
    struct key_hash
    {
        std::size_t operator()(const graphics_pipeline_key& key) const noexcept;
    };

    std::unordered_map<graphics_pipeline_key, pipeline_handle, key_hash> cache_;
};

} // namespace arc::render
