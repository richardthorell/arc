#include <arc/render/virtual_shadow.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace arc::render
{
namespace
{

std::uint64_t physical_page_bytes(virtual_shadow_depth_format format) noexcept
{
    const std::uint64_t physical_extent = virtual_shadow_page_texels + virtual_shadow_page_guard_texels * 2u;
    const std::uint64_t bytes_per_texel = format == virtual_shadow_depth_format::d16_unorm ? 2u : 4u;
    // One physical slot reserves matching static and dynamic overlay tiles.
    return physical_extent * physical_extent * bytes_per_texel * 2u;
}

bool same_address_space(virtual_shadow_address_space_handle lhs, virtual_shadow_address_space_handle rhs) noexcept
{
    return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

} // namespace

struct virtual_shadow_cache::address_space_slot
{
    virtual_shadow_address_space_descriptor descriptor{};
    std::uint32_t generation{1};
    bool occupied{};
};

struct virtual_shadow_cache::physical_page_slot
{
    std::uint32_t generation{1};
    bool occupied{};
};

struct virtual_shadow_cache::page_key_less
{
    bool operator()(const virtual_shadow_page_key& lhs, const virtual_shadow_page_key& rhs) const noexcept
    {
        return lhs < rhs;
    }
};

virtual_shadow_cache::~virtual_shadow_cache() = default;
virtual_shadow_cache::virtual_shadow_cache(virtual_shadow_cache&&) noexcept = default;
virtual_shadow_cache& virtual_shadow_cache::operator=(virtual_shadow_cache&&) noexcept = default;

virtual_shadow_cache::virtual_shadow_cache(std::uint64_t requested_budget_bytes, std::uint64_t device_budget_bytes,
                                           virtual_shadow_depth_format format)
    : depth_format_(format)
{
    const std::uint64_t device_cap =
        device_budget_bytes == 0 ? requested_budget_bytes : device_budget_bytes * 8u / 100u;
    budget_bytes_ = std::min(requested_budget_bytes, device_cap);
    const std::uint64_t bytes_per_page = physical_page_bytes(format);
    const auto capacity = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        budget_bytes_ / std::max<std::uint64_t>(bytes_per_page, 1u), std::numeric_limits<std::uint32_t>::max()));
    physical_pages_.resize(capacity);
    free_physical_pages_.reserve(capacity);
    for (std::uint32_t index = capacity; index > 0; --index)
        free_physical_pages_.push_back(index - 1u);
    cumulative_.physical_page_capacity = capacity;
    cumulative_.physical_memory_bytes = static_cast<std::uint64_t>(capacity) * bytes_per_page;
}

std::optional<virtual_shadow_address_space_handle>
virtual_shadow_cache::create_address_space(const virtual_shadow_address_space_descriptor& requested)
{
    virtual_shadow_address_space_descriptor descriptor = requested;
    descriptor.virtual_resolution = std::max(virtual_shadow_page_texels, descriptor.virtual_resolution);
    descriptor.level_count = std::max<std::uint8_t>(1, descriptor.level_count);
    descriptor.face_count = descriptor.light_kind == shadow_light_kind::point ? point_shadow_face_count : 1u;
    if (descriptor.light_kind == shadow_light_kind::directional)
        descriptor.level_count = virtual_shadow_directional_clip_levels;

    std::uint32_t index{};
    if (free_address_spaces_.empty())
    {
        index = static_cast<std::uint32_t>(address_spaces_.size());
        address_spaces_.push_back({});
    }
    else
    {
        index = free_address_spaces_.back();
        free_address_spaces_.pop_back();
    }
    auto& slot = address_spaces_[index];
    slot.occupied = true;
    slot.descriptor = descriptor;
    return virtual_shadow_address_space_handle{index, slot.generation};
}

bool virtual_shadow_cache::destroy_address_space(virtual_shadow_address_space_handle handle) noexcept
{
    if (!address_space(handle)) return false;
    for (std::size_t index = mappings_.size(); index > 0; --index)
        if (same_address_space(mappings_[index - 1u].key.address_space, handle))
            release_mapping(mappings_[index - 1u].key);
    auto& slot = address_spaces_[handle.index];
    slot.occupied = false;
    slot.descriptor = {};
    if (++slot.generation == 0) slot.generation = 1;
    free_address_spaces_.push_back(handle.index);
    return true;
}

const virtual_shadow_address_space_descriptor*
virtual_shadow_cache::address_space(virtual_shadow_address_space_handle handle) const noexcept
{
    if (!handle.valid() || handle.index >= address_spaces_.size()) return nullptr;
    const auto& slot = address_spaces_[handle.index];
    return slot.occupied && slot.generation == handle.generation ? &slot.descriptor : nullptr;
}

virtual_shadow_page_mapping* virtual_shadow_cache::find_mutable(const virtual_shadow_page_key& key) noexcept
{
    const auto found = std::lower_bound(mappings_.begin(), mappings_.end(), key,
                                        [](const virtual_shadow_page_mapping& mapping,
                                           const virtual_shadow_page_key& value) { return mapping.key < value; });
    return found != mappings_.end() && found->key == key ? &*found : nullptr;
}

const virtual_shadow_page_mapping* virtual_shadow_cache::find(const virtual_shadow_page_key& key) const noexcept
{
    const auto found = std::lower_bound(mappings_.begin(), mappings_.end(), key,
                                        [](const virtual_shadow_page_mapping& mapping,
                                           const virtual_shadow_page_key& value) { return mapping.key < value; });
    return found != mappings_.end() && found->key == key ? &*found : nullptr;
}

const virtual_shadow_page_mapping*
virtual_shadow_cache::find_resident_or_ancestor(const virtual_shadow_page_key& requested) const noexcept
{
    auto key = requested;
    const auto* descriptor = address_space(key.address_space);
    if (!descriptor) return nullptr;
    for (std::uint8_t level = key.coordinate.level; level < descriptor->level_count; ++level)
    {
        if (const auto* mapping = find(key); mapping && mapping->resident) return mapping;
        key.coordinate = virtual_shadow_parent_page(key.coordinate);
    }
    return nullptr;
}

std::span<const virtual_shadow_page_mapping> virtual_shadow_cache::mappings() const noexcept
{
    return mappings_;
}

std::optional<std::uint32_t> virtual_shadow_cache::eviction_candidate(std::uint64_t frame_index) const noexcept
{
    std::optional<std::uint32_t> candidate;
    for (std::uint32_t index = 0; index < mappings_.size(); ++index)
    {
        const auto& mapping = mappings_[index];
        const bool recently_used = frame_index < mapping.last_used_frame + virtual_shadow_page_protection_frames;
        if (mapping.pinned || mapping.in_flight || recently_used) continue;
        if (!candidate || mapping.last_used_frame < mappings_[*candidate].last_used_frame ||
            (mapping.last_used_frame == mappings_[*candidate].last_used_frame &&
             mapping.key < mappings_[*candidate].key))
            candidate = index;
    }
    return candidate;
}

std::optional<virtual_shadow_physical_page_handle>
virtual_shadow_cache::allocate_physical_page(std::uint64_t frame_index)
{
    if (free_physical_pages_.empty())
    {
        const auto candidate = eviction_candidate(frame_index);
        if (!candidate) return std::nullopt;
        release_mapping(mappings_[*candidate].key);
        ++cumulative_.eviction_count;
    }
    if (free_physical_pages_.empty()) return std::nullopt;
    const std::uint32_t index = free_physical_pages_.back();
    free_physical_pages_.pop_back();
    auto& slot = physical_pages_[index];
    slot.occupied = true;
    ++cumulative_.allocation_count;
    return virtual_shadow_physical_page_handle{index, slot.generation};
}

void virtual_shadow_cache::release_mapping(const virtual_shadow_page_key& key) noexcept
{
    const auto found = std::lower_bound(mappings_.begin(), mappings_.end(), key,
                                        [](const virtual_shadow_page_mapping& mapping,
                                           const virtual_shadow_page_key& value) { return mapping.key < value; });
    if (found == mappings_.end() || found->key != key) return;
    const auto physical = found->physical_page;
    if (physical.valid() && physical.index < physical_pages_.size())
    {
        auto& slot = physical_pages_[physical.index];
        if (slot.occupied && slot.generation == physical.generation)
        {
            slot.occupied = false;
            if (++slot.generation == 0) slot.generation = 1;
            free_physical_pages_.push_back(physical.index);
        }
    }
    mappings_.erase(found);
}

virtual_shadow_request_result
virtual_shadow_cache::resolve_requests(std::span<const virtual_shadow_page_request> requests, std::uint64_t frame_index)
{
    std::vector<virtual_shadow_page_request> ordered(requests.begin(), requests.end());
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const virtual_shadow_page_request& lhs, const virtual_shadow_page_request& rhs)
                     {
                         if (lhs.coarse_page != rhs.coarse_page) return lhs.coarse_page > rhs.coarse_page;
                         if (lhs.light_priority != rhs.light_priority) return lhs.light_priority > rhs.light_priority;
                         if (lhs.projected_coverage != rhs.projected_coverage)
                             return lhs.projected_coverage > rhs.projected_coverage;
                         if (lhs.key.coordinate.level != rhs.key.coordinate.level)
                             return lhs.key.coordinate.level > rhs.key.coordinate.level;
                         return lhs.key < rhs.key;
                     });
    ordered.erase(std::unique(ordered.begin(), ordered.end(),
                              [](const auto& lhs, const auto& rhs) { return lhs.key == rhs.key; }),
                  ordered.end());

    virtual_shadow_request_result result{};
    for (const auto& request : ordered)
    {
        if (!address_space(request.key.address_space))
        {
            ++result.failed_requests;
            ++cumulative_.failed_requests;
            continue;
        }
        if (auto* mapping = find_mutable(request.key))
        {
            mapping->last_used_frame = frame_index;
            mapping->pinned = mapping->pinned || request.coarse_page;
            if (mapping->content_revision != request.content_revision)
                mapping->dirty_reason = virtual_shadow_invalidation_reason::geometry;
            if (mapping->dirty()) result.render_pages.push_back(*mapping);
            ++result.cache_hits;
            ++cumulative_.cache_hits;
            continue;
        }

        const auto physical = allocate_physical_page(frame_index);
        if (!physical)
        {
            if (find_resident_or_ancestor(request.key))
            {
                ++result.parent_fallbacks;
                ++cumulative_.parent_fallbacks;
            }
            else
            {
                ++result.failed_requests;
                ++cumulative_.failed_requests;
            }
            continue;
        }
        virtual_shadow_page_mapping mapping{.key = request.key,
                                            .physical_page = *physical,
                                            .content_revision = request.content_revision,
                                            .last_used_frame = frame_index,
                                            .dirty_reason = virtual_shadow_invalidation_reason::newly_allocated,
                                            .resident = false,
                                            .pinned = request.coarse_page};
        const auto insertion = std::lower_bound(mappings_.begin(), mappings_.end(), mapping.key,
                                                [](const virtual_shadow_page_mapping& value,
                                                   const virtual_shadow_page_key& key) { return value.key < key; });
        mappings_.insert(insertion, mapping);
        result.render_pages.push_back(mapping);
        ++cumulative_.cache_misses;
    }
    return result;
}

bool virtual_shadow_cache::publish(const virtual_shadow_page_key& key, std::uint64_t content_revision) noexcept
{
    auto* mapping = find_mutable(key);
    if (!mapping) return false;
    mapping->resident = true;
    mapping->in_flight = false;
    mapping->content_revision = content_revision;
    mapping->dirty_reason = virtual_shadow_invalidation_reason::none;
    return true;
}

bool virtual_shadow_cache::set_in_flight(const virtual_shadow_page_key& key, bool in_flight) noexcept
{
    auto* mapping = find_mutable(key);
    if (!mapping) return false;
    mapping->in_flight = in_flight;
    return true;
}

std::uint32_t virtual_shadow_cache::invalidate(virtual_shadow_address_space_handle handle,
                                               virtual_shadow_invalidation_reason reason,
                                               std::optional<virtual_shadow_page_coordinate> coordinate) noexcept
{
    if (!address_space(handle) || reason == virtual_shadow_invalidation_reason::none) return 0;
    std::uint32_t count{};
    for (auto& mapping : mappings_)
    {
        if (!same_address_space(mapping.key.address_space, handle)) continue;
        if (coordinate && mapping.key.coordinate != *coordinate) continue;
        mapping.dirty_reason = reason;
        ++count;
    }
    return count;
}

void virtual_shadow_cache::clear() noexcept
{
    mappings_.clear();
    free_physical_pages_.clear();
    for (std::uint32_t index = static_cast<std::uint32_t>(physical_pages_.size()); index > 0; --index)
    {
        auto& slot = physical_pages_[index - 1u];
        slot.occupied = false;
        if (++slot.generation == 0) slot.generation = 1;
        free_physical_pages_.push_back(index - 1u);
    }
}

virtual_shadow_cache_statistics virtual_shadow_cache::statistics() const noexcept
{
    auto result = cumulative_;
    result.address_space_count = static_cast<std::uint32_t>(address_spaces_.size() - free_address_spaces_.size());
    result.resident_pages = 0;
    result.pinned_pages = 0;
    result.dirty_pages = 0;
    for (const auto& mapping : mappings_)
    {
        result.resident_pages += mapping.resident ? 1u : 0u;
        result.pinned_pages += mapping.pinned ? 1u : 0u;
        result.dirty_pages += mapping.dirty() ? 1u : 0u;
    }
    return result;
}

std::uint64_t virtual_shadow_cache::budget_bytes() const noexcept
{
    return budget_bytes_;
}

std::uint32_t virtual_shadow_cache::physical_page_capacity() const noexcept
{
    return static_cast<std::uint32_t>(physical_pages_.size());
}

virtual_shadow_depth_format virtual_shadow_cache::depth_format() const noexcept
{
    return depth_format_;
}

math::vector2f snap_virtual_shadow_clipmap_origin(const math::vector2f& origin, float world_units_per_texel) noexcept
{
    if (!std::isfinite(world_units_per_texel) || world_units_per_texel <= 0.0f) return origin;
    const float page_world_size = world_units_per_texel * static_cast<float>(virtual_shadow_page_texels);
    return {std::floor(origin[0] / page_world_size) * page_world_size,
            std::floor(origin[1] / page_world_size) * page_world_size};
}

} // namespace arc::render
