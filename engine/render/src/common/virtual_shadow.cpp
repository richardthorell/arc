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
    const std::uint64_t physical_extent = virtual_shadow_physical_page_texels;
    const std::uint64_t bytes_per_texel = format == virtual_shadow_depth_format::d16_unorm ? 2u : 4u;
    // One physical slot reserves matching static and dynamic overlay tiles.
    return physical_extent * physical_extent * bytes_per_texel * 2u;
}

std::uint32_t floor_square_root(std::uint64_t value) noexcept
{
    if (value == 0) return 0;
    auto root = static_cast<std::uint64_t>(std::sqrt(static_cast<long double>(value)));
    while ((root + 1u) <= std::numeric_limits<std::uint32_t>::max() && (root + 1u) * (root + 1u) <= value)
        ++root;
    while (root * root > value)
        --root;
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(root, std::numeric_limits<std::uint32_t>::max()));
}

gpu_virtual_shadow_physical_mapping invalid_gpu_mapping() noexcept
{
    return {};
}

gpu_virtual_shadow_page_table_entry invalid_gpu_page_entry() noexcept
{
    return {.static_depth = invalid_gpu_mapping(), .dynamic_depth = invalid_gpu_mapping()};
}

math::matrix4f look_at_rh(const math::vector3f& eye, const math::vector3f& center, const math::vector3f& up) noexcept
{
    const auto forward = math::normalize(math::sub(center, eye), 0.0f);
    const auto right = math::normalize(math::cross(forward, up), 0.0f);
    const auto corrected_up = math::cross(right, forward);
    math::matrix4f result = math::identity<float, 4>();
    result(0, 0) = right[0];
    result(0, 1) = right[1];
    result(0, 2) = right[2];
    result(1, 0) = corrected_up[0];
    result(1, 1) = corrected_up[1];
    result(1, 2) = corrected_up[2];
    result(2, 0) = -forward[0];
    result(2, 1) = -forward[1];
    result(2, 2) = -forward[2];
    result(0, 3) = -math::dot(right, eye);
    result(1, 3) = -math::dot(corrected_up, eye);
    result(2, 3) = math::dot(forward, eye);
    return result;
}

math::matrix4f orthographic_rh_zo(float extent, float near_plane, float far_plane) noexcept
{
    const float half = std::max(extent * 0.5f, 0.001f);
    const float depth = std::max(far_plane - near_plane, 0.001f);
    math::matrix4f result{};
    result(0, 0) = 1.0f / half;
    result(1, 1) = 1.0f / half;
    result(2, 2) = -1.0f / depth;
    result(2, 3) = -near_plane / depth;
    result(3, 3) = 1.0f;
    return result;
}

math::matrix4f perspective_rh_zo(float vertical_fov, float aspect, float near_plane, float far_plane) noexcept
{
    vertical_fov = std::clamp(vertical_fov, 0.001f, math::pi<float> - 0.001f);
    aspect = std::max(aspect, 0.001f);
    near_plane = std::max(near_plane, 0.001f);
    far_plane = std::max(far_plane, near_plane + 0.001f);
    const float inverse_tangent = 1.0f / std::tan(vertical_fov * 0.5f);
    math::matrix4f result{};
    result(0, 0) = inverse_tangent / aspect;
    result(1, 1) = inverse_tangent;
    result(2, 2) = far_plane / (near_plane - far_plane);
    result(2, 3) = (far_plane * near_plane) / (near_plane - far_plane);
    result(3, 2) = -1.0f;
    return result;
}

math::vector3f safe_direction(math::vector3f direction, math::vector3f fallback) noexcept
{
    direction = math::normalize(direction, 0.0f);
    return math::length_squared(direction) < 1.0e-6f ? fallback : direction;
}

math::vector3f safe_up_for_direction(const math::vector3f& direction) noexcept
{
    return std::abs(math::dot(direction, math::vector3f{0.0f, 1.0f, 0.0f})) > 0.95f ? math::vector3f{0.0f, 0.0f, 1.0f}
                                                                                    : math::vector3f{0.0f, 1.0f, 0.0f};
}

bool same_address_space(virtual_shadow_address_space_handle lhs, virtual_shadow_address_space_handle rhs) noexcept
{
    return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

} // namespace

virtual_shadow_physical_pool_layout
resolve_virtual_shadow_physical_pool(std::uint64_t budget_bytes, std::uint32_t maximum_texture_dimension_2d,
                                     virtual_shadow_depth_format_support formats) noexcept
{
    virtual_shadow_physical_pool_layout result{};
    result.budget_bytes = budget_bytes;
    if (budget_bytes == 0 || maximum_texture_dimension_2d < virtual_shadow_physical_page_texels || !formats.any())
        return result;

    result.format = formats.d16_unorm ? virtual_shadow_depth_format::d16_unorm : virtual_shadow_depth_format::d32_float;
    const auto bytes_per_slot = physical_page_bytes(result.format);
    const auto budget_axis = floor_square_root(budget_bytes / bytes_per_slot);
    const auto dimension_axis = maximum_texture_dimension_2d / virtual_shadow_physical_page_texels;
    result.pages_per_axis = std::min(budget_axis, dimension_axis);
    if (result.pages_per_axis == 0) return result;
    result.atlas_extent = result.pages_per_axis * virtual_shadow_physical_page_texels;
    result.physical_page_capacity = result.pages_per_axis * result.pages_per_axis;
    result.allocated_bytes = static_cast<std::uint64_t>(result.physical_page_capacity) * bytes_per_slot;
    return result;
}

struct virtual_shadow_cache::address_space_slot
{
    virtual_shadow_address_space_descriptor descriptor{};
    std::uint32_t generation{1};
    std::uint32_t page_table_base{};
    std::uint32_t page_table_count{};
    std::uint32_t view_base{};
    std::uint32_t view_count{};
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

struct virtual_shadow_cache::free_range
{
    std::uint32_t base{};
    std::uint32_t count{};
};

virtual_shadow_cache::~virtual_shadow_cache() = default;
virtual_shadow_cache::virtual_shadow_cache(virtual_shadow_cache&&) noexcept = default;
virtual_shadow_cache& virtual_shadow_cache::operator=(virtual_shadow_cache&&) noexcept = default;

virtual_shadow_cache::virtual_shadow_cache(const virtual_shadow_cache_config& config)
    : budget_bytes_(config.physical_pool.budget_bytes), depth_format_(config.physical_pool.format),
      physical_pool_layout_(config.physical_pool), page_table_entry_capacity_(config.page_table_entry_capacity),
      view_capacity_(config.view_capacity)
{
    const auto capacity = physical_pool_layout_.valid() ? physical_pool_layout_.physical_page_capacity : 0u;
    physical_pages_.resize(capacity);
    free_physical_pages_.reserve(capacity);
    for (std::uint32_t index = capacity; index > 0; --index)
        free_physical_pages_.push_back(index - 1u);
    if (page_table_entry_capacity_ != 0) free_page_table_ranges_.push_back({0u, page_table_entry_capacity_});
    if (view_capacity_ != 0) free_view_ranges_.push_back({0u, view_capacity_});
    cumulative_.physical_page_capacity = capacity;
    cumulative_.physical_memory_bytes = physical_pool_layout_.allocated_bytes;
}

virtual_shadow_cache::virtual_shadow_cache(std::uint64_t requested_budget_bytes, std::uint64_t device_budget_bytes,
                                           virtual_shadow_depth_format format)
    : virtual_shadow_cache(virtual_shadow_cache_config{
          .physical_pool = resolve_virtual_shadow_physical_pool(
              std::min(requested_budget_bytes,
                       device_budget_bytes == 0 ? requested_budget_bytes : device_budget_bytes * 8u / 100u),
              std::numeric_limits<std::uint32_t>::max(),
              {.d16_unorm = format == virtual_shadow_depth_format::d16_unorm,
               .d32_float = format == virtual_shadow_depth_format::d32_float})})
{
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

    const auto page_count_64 = virtual_shadow_page_table_entry_count(descriptor);
    const auto view_count_64 = static_cast<std::uint64_t>(descriptor.face_count) * descriptor.level_count;
    if (page_count_64 == 0 || page_count_64 > std::numeric_limits<std::uint32_t>::max() || view_count_64 == 0 ||
        view_count_64 > std::numeric_limits<std::uint32_t>::max())
        return std::nullopt;
    const auto page_count = static_cast<std::uint32_t>(page_count_64);
    const auto view_count = static_cast<std::uint32_t>(view_count_64);
    const auto page_base = allocate_range(free_page_table_ranges_, page_count);
    if (!page_base) return std::nullopt;
    const auto view_base = allocate_range(free_view_ranges_, view_count);
    if (!view_base)
    {
        release_range(free_page_table_ranges_, *page_base, page_count);
        return std::nullopt;
    }

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
    slot.page_table_base = *page_base;
    slot.page_table_count = page_count;
    slot.view_base = *view_base;
    slot.view_count = view_count;

    if (gpu_address_spaces_.size() <= index) gpu_address_spaces_.resize(static_cast<std::size_t>(index) + 1u);
    const auto required_views = static_cast<std::size_t>(*view_base) + view_count;
    if (gpu_views_.size() < required_views) gpu_views_.resize(required_views);
    const auto required_pages = static_cast<std::size_t>(*page_base) + page_count;
    if (gpu_page_table_.size() < required_pages) gpu_page_table_.resize(required_pages, invalid_gpu_page_entry());
    std::fill(gpu_page_table_.begin() + *page_base, gpu_page_table_.begin() + *page_base + page_count,
              invalid_gpu_page_entry());

    std::uint32_t view_index{};
    for (std::uint8_t face = 0; face < descriptor.face_count; ++face)
    {
        for (std::uint8_t level = 0; level < descriptor.level_count; ++level)
        {
            const auto offset = virtual_shadow_dense_page_offset(descriptor, {0, 0, level, face}).value_or(0u);
            auto& output = gpu_views_[*view_base + view_index++];
            output = {};
            for (std::uint32_t diagonal = 0; diagonal < 4; ++diagonal)
                output.world_to_shadow_clip[diagonal * 4u + diagonal] = 1.0f;
            output.page_table_offset = offset;
            output.pages_per_axis = virtual_shadow_pages_per_axis(descriptor, level);
            output.face = face;
            output.level = level;
        }
    }
    rebuild_gpu_address_space(index);
    ++gpu_revision_;
    return virtual_shadow_address_space_handle{index, slot.generation};
}

bool virtual_shadow_cache::destroy_address_space(virtual_shadow_address_space_handle handle) noexcept
{
    if (!address_space(handle)) return false;
    for (std::size_t index = mappings_.size(); index > 0; --index)
        if (same_address_space(mappings_[index - 1u].key.address_space, handle))
            release_mapping(mappings_[index - 1u].key);
    auto& slot = address_spaces_[handle.index];
    std::fill(gpu_page_table_.begin() + slot.page_table_base,
              gpu_page_table_.begin() + slot.page_table_base + slot.page_table_count, invalid_gpu_page_entry());
    std::fill(gpu_views_.begin() + slot.view_base, gpu_views_.begin() + slot.view_base + slot.view_count,
              gpu_virtual_shadow_view_record{});
    release_range(free_page_table_ranges_, slot.page_table_base, slot.page_table_count);
    release_range(free_view_ranges_, slot.view_base, slot.view_count);
    slot.occupied = false;
    slot.descriptor = {};
    slot.page_table_base = 0;
    slot.page_table_count = 0;
    slot.view_base = 0;
    slot.view_count = 0;
    if (++slot.generation == 0) slot.generation = 1;
    rebuild_gpu_address_space(handle.index);
    free_address_spaces_.push_back(handle.index);
    ++gpu_revision_;
    return true;
}

const virtual_shadow_address_space_descriptor*
virtual_shadow_cache::address_space(virtual_shadow_address_space_handle handle) const noexcept
{
    if (!handle.valid() || handle.index >= address_spaces_.size()) return nullptr;
    const auto& slot = address_spaces_[handle.index];
    return slot.occupied && slot.generation == handle.generation ? &slot.descriptor : nullptr;
}

std::optional<std::uint32_t> virtual_shadow_cache::allocate_range(std::vector<free_range>& ranges,
                                                                  std::uint32_t count) noexcept
{
    if (count == 0) return std::nullopt;
    for (auto iterator = ranges.begin(); iterator != ranges.end(); ++iterator)
    {
        if (iterator->count < count) continue;
        const auto base = iterator->base;
        iterator->base += count;
        iterator->count -= count;
        if (iterator->count == 0) ranges.erase(iterator);
        return base;
    }
    return std::nullopt;
}

void virtual_shadow_cache::release_range(std::vector<free_range>& ranges, std::uint32_t base,
                                         std::uint32_t count) noexcept
{
    if (count == 0) return;
    const auto position =
        std::lower_bound(ranges.begin(), ranges.end(), base,
                         [](const free_range& range, std::uint32_t value) { return range.base < value; });
    auto inserted = ranges.insert(position, {base, count});
    if (inserted != ranges.begin())
    {
        auto previous = inserted - 1;
        if (static_cast<std::uint64_t>(previous->base) + previous->count == inserted->base)
        {
            previous->count += inserted->count;
            inserted = ranges.erase(inserted);
            inserted = previous;
        }
    }
    auto next = inserted + 1;
    if (next != ranges.end() && static_cast<std::uint64_t>(inserted->base) + inserted->count == next->base)
    {
        inserted->count += next->count;
        ranges.erase(next);
    }
}

void virtual_shadow_cache::rebuild_gpu_address_space(std::uint32_t index) noexcept
{
    if (gpu_address_spaces_.size() <= index) gpu_address_spaces_.resize(static_cast<std::size_t>(index) + 1u);
    const auto& slot = address_spaces_[index];
    auto& output = gpu_address_spaces_[index];
    output = {};
    output.generation = slot.generation;
    if (!slot.occupied) return;
    output.light_kind = static_cast<std::uint32_t>(slot.descriptor.light_kind);
    output.virtual_resolution = slot.descriptor.virtual_resolution;
    output.topology = static_cast<std::uint32_t>(slot.descriptor.level_count) |
                      (static_cast<std::uint32_t>(slot.descriptor.face_count) << 16u);
    output.view_base = slot.view_base;
    output.view_count = slot.view_count;
    output.page_table_base = slot.page_table_base;
    output.page_table_count = slot.page_table_count;
}

bool virtual_shadow_cache::update_address_space_views(virtual_shadow_address_space_handle handle,
                                                      std::span<const virtual_shadow_view_descriptor> views) noexcept
{
    if (!address_space(handle)) return false;
    auto& slot = address_spaces_[handle.index];
    if (views.size() != slot.view_count) return false;
    std::vector<bool> written(slot.view_count, false);
    std::vector<gpu_virtual_shadow_view_record> packed(slot.view_count);
    for (const auto& input : views)
    {
        if (input.face >= slot.descriptor.face_count || input.level >= slot.descriptor.level_count ||
            input.pages_per_axis !=
                virtual_shadow_pages_per_axis(slot.descriptor, static_cast<std::uint8_t>(input.level)) ||
            !std::isfinite(input.world_units_per_texel) || input.world_units_per_texel < 0.0f ||
            !std::isfinite(input.snapped_origin[0]) || !std::isfinite(input.snapped_origin[1]) ||
            !std::isfinite(input.snapped_origin[2]))
            return false;
        const auto local_index = static_cast<std::uint32_t>(input.face) * slot.descriptor.level_count + input.level;
        if (written[local_index]) return false;
        written[local_index] = true;
        auto& output = packed[local_index];
        output = {};
        for (std::uint32_t row = 0; row < 4; ++row)
            for (std::uint32_t column = 0; column < 4; ++column)
            {
                const float value = input.world_to_shadow_clip(row, column);
                if (!std::isfinite(value)) return false;
                output.world_to_shadow_clip[row * 4u + column] = value;
            }
        output.snapped_origin_world_units[0] = input.snapped_origin[0];
        output.snapped_origin_world_units[1] = input.snapped_origin[1];
        output.snapped_origin_world_units[2] = input.snapped_origin[2];
        output.snapped_origin_world_units[3] = input.world_units_per_texel;
        const auto offset = virtual_shadow_dense_page_offset(
            slot.descriptor, {0, 0, static_cast<std::uint8_t>(input.level), static_cast<std::uint8_t>(input.face)});
        if (!offset) return false;
        output.page_table_offset = *offset;
        output.pages_per_axis = input.pages_per_axis;
        output.face = input.face;
        output.level = input.level;
    }
    std::copy(packed.begin(), packed.end(), gpu_views_.begin() + slot.view_base);
    ++gpu_revision_;
    return true;
}

std::optional<std::uint32_t> virtual_shadow_cache::dense_page_index(const virtual_shadow_page_key& key) const noexcept
{
    const auto* descriptor = address_space(key.address_space);
    if (!descriptor) return std::nullopt;
    const auto offset = virtual_shadow_dense_page_offset(*descriptor, key.coordinate);
    if (!offset) return std::nullopt;
    const auto& slot = address_spaces_[key.address_space.index];
    if (*offset >= slot.page_table_count) return std::nullopt;
    return slot.page_table_base + *offset;
}

virtual_shadow_gpu_snapshot virtual_shadow_cache::gpu_snapshot() const noexcept
{
    return {.address_spaces = gpu_address_spaces_,
            .views = gpu_views_,
            .page_table = gpu_page_table_,
            .revision = gpu_revision_};
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
    // Directional levels are independent equal-resolution clip grids. A coarser
    // fallback must reproject the receiver through that level's view and cannot
    // be derived by halving a page coordinate.
    if (descriptor->light_kind == shadow_light_kind::directional)
    {
        const auto* mapping = find(key);
        return mapping && mapping->resident ? mapping : nullptr;
    }
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
    clear_gpu_mapping(key);
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
    ++gpu_revision_;
}

void virtual_shadow_cache::clear_gpu_mapping(const virtual_shadow_page_key& key) noexcept
{
    const auto index = dense_page_index(key);
    if (!index || *index >= gpu_page_table_.size()) return;
    auto& entry = gpu_page_table_[*index];
    if (key.layer == virtual_shadow_page_layer::static_depth)
        entry.static_depth = invalid_gpu_mapping();
    else
        entry.dynamic_depth = invalid_gpu_mapping();
}

void virtual_shadow_cache::publish_gpu_mapping(const virtual_shadow_page_mapping& mapping) noexcept
{
    const auto index = dense_page_index(mapping.key);
    if (!index || *index >= gpu_page_table_.size() || !mapping.physical_page.valid()) return;
    const gpu_virtual_shadow_physical_mapping output{
        .physical_page = mapping.physical_page.index,
        .physical_generation = mapping.physical_page.generation,
        .content_revision_low = static_cast<std::uint32_t>(mapping.content_revision),
        .content_revision_high = static_cast<std::uint32_t>(mapping.content_revision >> 32u)};
    auto& entry = gpu_page_table_[*index];
    if (mapping.key.layer == virtual_shadow_page_layer::static_depth)
        entry.static_depth = output;
    else
        entry.dynamic_depth = output;
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
        if (!address_space(request.key.address_space) || !dense_page_index(request.key))
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
    publish_gpu_mapping(*mapping);
    ++gpu_revision_;
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
    for (const auto& mapping : mappings_)
        clear_gpu_mapping(mapping.key);
    mappings_.clear();
    free_physical_pages_.clear();
    for (std::uint32_t index = static_cast<std::uint32_t>(physical_pages_.size()); index > 0; --index)
    {
        auto& slot = physical_pages_[index - 1u];
        slot.occupied = false;
        if (++slot.generation == 0) slot.generation = 1;
        free_physical_pages_.push_back(index - 1u);
    }
    ++gpu_revision_;
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

const virtual_shadow_physical_pool_layout& virtual_shadow_cache::physical_pool_layout() const noexcept
{
    return physical_pool_layout_;
}

std::vector<virtual_shadow_view_descriptor>
make_directional_virtual_shadow_views(const virtual_shadow_address_space_descriptor& requested,
                                      const directional_shadow_camera& camera, const math::vector3f& camera_position,
                                      const math::vector3f& authored_light_direction, float shadow_distance) noexcept
{
    auto descriptor = requested;
    descriptor.light_kind = shadow_light_kind::directional;
    descriptor.face_count = 1;
    descriptor.level_count = virtual_shadow_directional_clip_levels;
    descriptor.virtual_resolution = std::max(virtual_shadow_page_texels, descriptor.virtual_resolution);
    shadow_distance = std::max(shadow_distance, std::max(camera.near_plane, 0.01f));

    const auto light_direction =
        safe_direction(authored_light_direction, math::normalize(math::vector3f{0.35f, -0.85f, -0.4f}, 0.0f));
    const auto up = safe_up_for_direction(light_direction);
    const auto light_basis = look_at_rh(math::mul(light_direction, -1.0f), math::vector3f::zero, up);
    const auto light_center = math::transform_point(light_basis, camera_position);

    std::vector<virtual_shadow_view_descriptor> result;
    result.reserve(descriptor.level_count);
    for (std::uint8_t level = 0; level < descriptor.level_count; ++level)
    {
        const auto coarser_levels = static_cast<int>(descriptor.level_count - level - 1u);
        const float half_extent = shadow_distance / std::pow(2.0f, static_cast<float>(coarser_levels));
        const float world_units_per_texel = (half_extent * 2.0f) / static_cast<float>(descriptor.virtual_resolution);
        const auto snapped =
            snap_virtual_shadow_clipmap_origin({light_center[0], light_center[1]}, world_units_per_texel);
        const float delta_x = snapped[0] - light_center[0];
        const float delta_y = snapped[1] - light_center[1];
        const auto center =
            math::add(camera_position, math::vector3f{light_basis(0, 0) * delta_x + light_basis(1, 0) * delta_y,
                                                      light_basis(0, 1) * delta_x + light_basis(1, 1) * delta_y,
                                                      light_basis(0, 2) * delta_x + light_basis(1, 2) * delta_y});
        const auto view = look_at_rh(math::sub(center, math::mul(light_direction, shadow_distance * 2.5f)), center, up);
        const float guard_extent = half_extent * 2.0f + world_units_per_texel * 4.0f;
        const auto projection = orthographic_rh_zo(guard_extent, 0.01f, shadow_distance * 5.0f);
        result.push_back({.world_to_shadow_clip = math::matmul(projection, view),
                          .snapped_origin = {snapped[0], snapped[1], 0.0f},
                          .world_units_per_texel = world_units_per_texel,
                          .pages_per_axis = virtual_shadow_pages_per_axis(descriptor, level),
                          .face = 0,
                          .level = level});
    }
    return result;
}

std::vector<virtual_shadow_view_descriptor>
make_point_virtual_shadow_views(const virtual_shadow_address_space_descriptor& requested,
                                const math::vector3f& light_position, float light_range) noexcept
{
    auto descriptor = requested;
    descriptor.light_kind = shadow_light_kind::point;
    descriptor.face_count = point_shadow_face_count;
    descriptor.level_count = std::max<std::uint8_t>(1, descriptor.level_count);
    descriptor.virtual_resolution = std::max(virtual_shadow_page_texels, descriptor.virtual_resolution);
    light_range = std::max(light_range, 0.01f);
    constexpr std::array<math::vector3f, point_shadow_face_count> directions{{{1.0f, 0.0f, 0.0f},
                                                                              {-1.0f, 0.0f, 0.0f},
                                                                              {0.0f, 1.0f, 0.0f},
                                                                              {0.0f, -1.0f, 0.0f},
                                                                              {0.0f, 0.0f, 1.0f},
                                                                              {0.0f, 0.0f, -1.0f}}};
    constexpr std::array<math::vector3f, point_shadow_face_count> up_vectors{{{0.0f, -1.0f, 0.0f},
                                                                              {0.0f, -1.0f, 0.0f},
                                                                              {0.0f, 0.0f, 1.0f},
                                                                              {0.0f, 0.0f, -1.0f},
                                                                              {0.0f, -1.0f, 0.0f},
                                                                              {0.0f, -1.0f, 0.0f}}};
    const auto projection = perspective_rh_zo(math::pi<float> * 0.5f, 1.0f, 0.01f, light_range);

    std::vector<virtual_shadow_view_descriptor> result;
    result.reserve(static_cast<std::size_t>(descriptor.face_count) * descriptor.level_count);
    for (std::uint8_t face = 0; face < descriptor.face_count; ++face)
    {
        const auto view = look_at_rh(light_position, math::add(light_position, directions[face]), up_vectors[face]);
        for (std::uint8_t level = 0; level < descriptor.level_count; ++level)
            result.push_back({.world_to_shadow_clip = math::matmul(projection, view),
                              .snapped_origin = light_position,
                              .world_units_per_texel = 0.0f,
                              .pages_per_axis = virtual_shadow_pages_per_axis(descriptor, level),
                              .face = face,
                              .level = level});
    }
    return result;
}

std::vector<virtual_shadow_view_descriptor>
make_spot_virtual_shadow_views(const virtual_shadow_address_space_descriptor& requested,
                               const math::vector3f& light_position, const math::vector3f& authored_light_direction,
                               float outer_cone_radians, float light_range) noexcept
{
    auto descriptor = requested;
    descriptor.light_kind = shadow_light_kind::spot;
    descriptor.face_count = 1;
    descriptor.level_count = std::max<std::uint8_t>(1, descriptor.level_count);
    descriptor.virtual_resolution = std::max(virtual_shadow_page_texels, descriptor.virtual_resolution);
    light_range = std::max(light_range, 0.01f);
    const auto direction = safe_direction(authored_light_direction, math::vector3f{0.0f, 0.0f, -1.0f});
    const auto view =
        look_at_rh(light_position, math::add(light_position, direction), safe_up_for_direction(direction));
    const auto projection = perspective_rh_zo(outer_cone_radians * 2.0f, 1.0f, 0.01f, light_range);

    std::vector<virtual_shadow_view_descriptor> result;
    result.reserve(descriptor.level_count);
    for (std::uint8_t level = 0; level < descriptor.level_count; ++level)
        result.push_back({.world_to_shadow_clip = math::matmul(projection, view),
                          .snapped_origin = light_position,
                          .world_units_per_texel = 0.0f,
                          .pages_per_axis = virtual_shadow_pages_per_axis(descriptor, level),
                          .face = 0,
                          .level = level});
    return result;
}

std::optional<virtual_shadow_page_coordinate>
virtual_shadow_page_for_world(const virtual_shadow_view_descriptor& view, const math::vector3f& world_position) noexcept
{
    if (view.pages_per_axis == 0 || view.face > std::numeric_limits<std::uint8_t>::max() ||
        view.level > std::numeric_limits<std::uint8_t>::max())
        return std::nullopt;
    const auto clip = math::transform_point(view.world_to_shadow_clip, world_position);
    const float u = clip[0] * 0.5f + 0.5f;
    const float v = clip[1] * 0.5f + 0.5f;
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0f || v < 0.0f || u >= 1.0f || v >= 1.0f) return std::nullopt;
    const auto x = static_cast<std::uint32_t>(std::floor(u * static_cast<float>(view.pages_per_axis)));
    const auto y = static_cast<std::uint32_t>(std::floor(v * static_cast<float>(view.pages_per_axis)));
    if (x > std::numeric_limits<std::uint16_t>::max() || y > std::numeric_limits<std::uint16_t>::max())
        return std::nullopt;
    return virtual_shadow_page_coordinate{static_cast<std::uint16_t>(x), static_cast<std::uint16_t>(y),
                                          static_cast<std::uint8_t>(view.level), static_cast<std::uint8_t>(view.face)};
}

math::vector2f snap_virtual_shadow_clipmap_origin(const math::vector2f& origin, float world_units_per_texel) noexcept
{
    if (!std::isfinite(world_units_per_texel) || world_units_per_texel <= 0.0f) return origin;
    const float page_world_size = world_units_per_texel * static_cast<float>(virtual_shadow_page_texels);
    return {std::floor(origin[0] / page_world_size) * page_world_size,
            std::floor(origin[1] / page_world_size) * page_world_size};
}

} // namespace arc::render
