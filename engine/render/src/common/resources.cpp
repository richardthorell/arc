#include <arc/render/resources.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <utility>

namespace arc::render
{

descriptor_slot descriptor_slot_pool::allocate(descriptor_resource_type type)
{
    std::uint32_t index{};
    if (!free_list_.empty())
    {
        index = free_list_.back();
        free_list_.pop_back();
    }
    else
    {
        index = static_cast<std::uint32_t>(slots_.size());
        slots_.push_back({});
    }

    auto& slot = slots_[index];
    slot.type = type;
    slot.alive = true;
    ++live_count_;
    return { .type = type, .index = index, .generation = slot.generation };
}

bool descriptor_slot_pool::release(descriptor_slot slot)
{
    if (!alive(slot))
        return false;

    auto& state = slots_[slot.index];
    state.alive = false;
    ++state.generation;
    free_list_.push_back(slot.index);
    --live_count_;
    return true;
}

bool descriptor_slot_pool::alive(descriptor_slot slot) const
{
    if (!slot.valid() || slot.index >= slots_.size())
        return false;

    const auto& state = slots_[slot.index];
    return state.alive && state.generation == slot.generation && state.type == slot.type;
}

std::uint32_t descriptor_slot_pool::live_count() const noexcept
{
    return live_count_;
}

void deferred_resource_releaser::defer(std::uint64_t retire_after_frame, release_fn release)
{
    if (!release)
        return;
    pending_.push_back({ .retire_after_frame = retire_after_frame, .release = std::move(release) });
}

std::size_t deferred_resource_releaser::collect(std::uint64_t completed_frame)
{
    std::size_t released = 0;
    auto iterator = pending_.begin();
    while (iterator != pending_.end())
    {
        if (iterator->retire_after_frame <= completed_frame)
        {
            iterator->release();
            iterator = pending_.erase(iterator);
            ++released;
        }
        else
        {
            ++iterator;
        }
    }
    return released;
}

std::size_t deferred_resource_releaser::pending_count() const noexcept
{
    return pending_.size();
}

frame_allocator::frame_allocator(std::size_t capacity)
    : arena_(capacity)
{
}

void* frame_allocator::allocate(std::size_t size, std::size_t alignment)
{
    return arena_.allocate(size, alignment);
}

void frame_allocator::reset() noexcept
{
    arena_.reset();
}

std::size_t frame_allocator::used() const noexcept
{
    return arena_.used();
}

std::size_t frame_allocator::capacity() const noexcept
{
    return arena_.capacity();
}

bool operator==(const graphics_pipeline_key& lhs, const graphics_pipeline_key& rhs) noexcept
{
    return lhs.vertex_shader == rhs.vertex_shader &&
        lhs.fragment_shader == rhs.fragment_shader &&
        lhs.vertex_layout == rhs.vertex_layout &&
        lhs.color_format == rhs.color_format &&
        lhs.depth_format == rhs.depth_format &&
        lhs.depth_test == rhs.depth_test &&
        lhs.depth_write == rhs.depth_write &&
        lhs.wireframe == rhs.wireframe &&
        lhs.alpha_blend == rhs.alpha_blend &&
        lhs.permutation == rhs.permutation;
}

std::size_t pipeline_handle_cache::key_hash::operator()(const graphics_pipeline_key& key) const noexcept
{
    auto combine = [](std::size_t seed, std::size_t value) {
        return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u));
    };

    std::size_t seed = 0;
    seed = combine(seed, std::hash<std::uint32_t>{}(key.vertex_shader.index));
    seed = combine(seed, std::hash<std::uint32_t>{}(key.vertex_shader.generation));
    seed = combine(seed, std::hash<std::uint32_t>{}(key.fragment_shader.index));
    seed = combine(seed, std::hash<std::uint32_t>{}(key.fragment_shader.generation));
    seed = combine(seed, std::hash<std::string>{}(key.vertex_layout));
    seed = combine(seed, std::hash<std::string>{}(key.color_format));
    seed = combine(seed, std::hash<std::string>{}(key.depth_format));
    seed = combine(seed, std::hash<bool>{}(key.depth_test));
    seed = combine(seed, std::hash<bool>{}(key.depth_write));
    seed = combine(seed, std::hash<bool>{}(key.wireframe));
    seed = combine(seed, std::hash<bool>{}(key.alpha_blend));
    seed = combine(seed, hash_shader_permutation_key(key.permutation));
    return seed;
}

pipeline_handle pipeline_handle_cache::find(const graphics_pipeline_key& key) const
{
    const auto found = cache_.find(key);
    if (found == cache_.end())
        return {};
    return found->second;
}

void pipeline_handle_cache::insert(graphics_pipeline_key key, pipeline_handle handle)
{
    cache_[std::move(key)] = handle;
}

void pipeline_handle_cache::clear()
{
    cache_.clear();
}

std::size_t pipeline_handle_cache::size() const noexcept
{
    return cache_.size();
}

} // namespace arc::render
