#pragma once

#include <arc/math/math.h>
#include <arc/render/shadow.h>

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace arc::render
{

/** @brief Width and height, in virtual texels, represented by one VSM page. */
inline constexpr std::uint32_t virtual_shadow_page_texels = 128;
/** @brief Texels replicated around every physical VSM page for filtered sampling. */
inline constexpr std::uint32_t virtual_shadow_page_guard_texels = 4;
/** @brief Width and height of one guarded physical atlas tile. */
inline constexpr std::uint32_t virtual_shadow_physical_page_texels =
    virtual_shadow_page_texels + virtual_shadow_page_guard_texels * 2u;
/** @brief Directional-light clip levels used by ARC's Ultra quality profile. */
inline constexpr std::uint32_t virtual_shadow_directional_clip_levels = 5;
/** @brief Number of frames for which a recently sampled page cannot be evicted. */
inline constexpr std::uint32_t virtual_shadow_page_protection_frames = 30;
/** @brief Default device-local memory budget for the Ultra VSM pool. */
inline constexpr std::uint64_t default_virtual_shadow_budget_bytes = 512ull * 1024ull * 1024ull;
/** @brief Default upper bound for the dense GPU page-table address space. */
inline constexpr std::uint64_t default_virtual_shadow_page_table_bytes = 64ull * 1024ull * 1024ull;
/** @brief Default number of per-face/per-level view records retained by the cache. */
inline constexpr std::uint32_t default_virtual_shadow_view_capacity = 4096;
inline constexpr std::uint32_t invalid_virtual_shadow_index = 0xffffffffu;

/** @brief Physical depth format selected for the VSM page pool. */
enum class virtual_shadow_depth_format : std::uint8_t
{
    d16_unorm,
    d32_float
};

/** @brief Sampled depth formats supported by a backend for the VSM atlas pair. */
struct virtual_shadow_depth_format_support
{
    bool d16_unorm{};
    bool d32_float{};

    [[nodiscard]] constexpr bool any() const noexcept
    {
        return d16_unorm || d32_float;
    }
};

/** @brief Complete backend-neutral physical pool layout shared by the graph and backend. */
struct virtual_shadow_physical_pool_layout
{
    virtual_shadow_depth_format format{virtual_shadow_depth_format::d16_unorm};
    std::uint32_t pages_per_axis{};
    std::uint32_t atlas_extent{};
    std::uint32_t physical_page_capacity{};
    std::uint64_t budget_bytes{};
    std::uint64_t allocated_bytes{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return pages_per_axis != 0 && atlas_extent != 0 && physical_page_capacity != 0 && allocated_bytes != 0 &&
               allocated_bytes <= budget_bytes;
    }
};

/** @brief Resolve one square paired-atlas allocation without using backend-native types. */
[[nodiscard]] virtual_shadow_physical_pool_layout
resolve_virtual_shadow_physical_pool(std::uint64_t budget_bytes, std::uint32_t maximum_texture_dimension_2d,
                                     virtual_shadow_depth_format_support formats) noexcept;

/** @brief Per-light-kind executable VSM support. */
struct virtual_shadow_light_support
{
    bool directional{};
    bool point{};
    bool spot{};

    [[nodiscard]] constexpr bool supports(shadow_light_kind kind) const noexcept
    {
        switch (kind)
        {
            case shadow_light_kind::directional:
                return directional;
            case shadow_light_kind::point:
                return point;
            case shadow_light_kind::spot:
                return spot;
        }
        return false;
    }

    [[nodiscard]] constexpr bool any() const noexcept
    {
        return directional || point || spot;
    }
};

/** @brief Static or dynamic depth layer represented by a virtual page. */
enum class virtual_shadow_page_layer : std::uint8_t
{
    static_depth,
    dynamic_depth
};

/** @brief Reason a cached VSM page must be rendered again. */
enum class virtual_shadow_invalidation_reason : std::uint8_t
{
    none,
    newly_allocated,
    light_changed,
    caster_transform,
    geometry,
    material_alpha,
    terrain,
    vegetation,
    prefab,
    world_epoch,
    address_space_moved
};

/** @brief Generational handle for a light's virtual shadow address space. */
struct virtual_shadow_address_space_handle
{
    static constexpr std::uint32_t invalid_index = 0xffffffffu;
    std::uint32_t index{invalid_index};
    std::uint32_t generation{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index != invalid_index;
    }
    friend constexpr bool operator==(virtual_shadow_address_space_handle,
                                     virtual_shadow_address_space_handle) noexcept = default;
    friend constexpr auto operator<=>(virtual_shadow_address_space_handle,
                                      virtual_shadow_address_space_handle) noexcept = default;
};

/** @brief Generational handle for one resident physical VSM page. */
struct virtual_shadow_physical_page_handle
{
    static constexpr std::uint32_t invalid_index = 0xffffffffu;
    std::uint32_t index{invalid_index};
    std::uint32_t generation{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index != invalid_index;
    }
    friend constexpr bool operator==(virtual_shadow_physical_page_handle,
                                     virtual_shadow_physical_page_handle) noexcept = default;
    friend constexpr auto operator<=>(virtual_shadow_physical_page_handle,
                                      virtual_shadow_physical_page_handle) noexcept = default;
};

/** @brief Stable virtual coordinate inside a light's shadow address space. */
struct virtual_shadow_page_coordinate
{
    std::uint16_t x{};
    std::uint16_t y{};
    std::uint8_t level{};
    std::uint8_t face{};

    friend constexpr bool operator==(virtual_shadow_page_coordinate, virtual_shadow_page_coordinate) noexcept = default;
    friend constexpr auto operator<=>(virtual_shadow_page_coordinate,
                                      virtual_shadow_page_coordinate) noexcept = default;
};

/** @brief Complete identity of one static or dynamic virtual shadow page. */
struct virtual_shadow_page_key
{
    virtual_shadow_address_space_handle address_space{};
    virtual_shadow_page_coordinate coordinate{};
    virtual_shadow_page_layer layer{virtual_shadow_page_layer::static_depth};

    friend constexpr bool operator==(virtual_shadow_page_key, virtual_shadow_page_key) noexcept = default;
    friend constexpr auto operator<=>(virtual_shadow_page_key, virtual_shadow_page_key) noexcept = default;
};

/** @brief Creation parameters for one directional, point, or spot VSM address space. */
struct virtual_shadow_address_space_descriptor
{
    shadow_light_kind light_kind{shadow_light_kind::directional};
    std::uint64_t light_key{};
    render_mobility mobility{render_mobility::movable};
    std::uint32_t virtual_resolution{16384};
    std::uint8_t level_count{virtual_shadow_directional_clip_levels};
    std::uint8_t face_count{1};
    std::uint16_t priority{128};
};

/** @brief CPU-side projection and page-grid data for one face and level. */
struct virtual_shadow_view_descriptor
{
    math::matrix4f world_to_shadow_clip{math::identity<float, 4>()};
    math::vector3f snapped_origin{};
    float world_units_per_texel{};
    std::uint32_t pages_per_axis{};
    std::uint16_t face{};
    std::uint16_t level{};
};

/** @brief Stable GPU address-space header. */
struct alignas(16) gpu_virtual_shadow_address_space_record
{
    std::uint32_t generation{};
    std::uint32_t light_kind{};
    std::uint32_t virtual_resolution{};
    std::uint32_t topology{}; // Low 16 bits: level count. High 16 bits: face count.
    std::uint32_t view_base{};
    std::uint32_t view_count{};
    std::uint32_t page_table_base{};
    std::uint32_t page_table_count{};
};

/** @brief Stable GPU projection and dense-table range for one face and level. */
struct alignas(16) gpu_virtual_shadow_view_record
{
    float world_to_shadow_clip[16]{};      // Row-major; shader helpers perform the matching multiply.
    float snapped_origin_world_units[4]{}; // xyz origin, w world-units-per-texel.
    std::uint32_t page_table_offset{};
    std::uint32_t pages_per_axis{};
    std::uint32_t face{};
    std::uint32_t level{};
};

/** @brief One published physical mapping inside a dense virtual page entry. */
struct alignas(16) gpu_virtual_shadow_physical_mapping
{
    std::uint32_t physical_page{invalid_virtual_shadow_index};
    std::uint32_t physical_generation{};
    std::uint32_t content_revision_low{};
    std::uint32_t content_revision_high{};
};

/** @brief Static and dynamic mappings sharing one virtual page-table coordinate. */
struct alignas(16) gpu_virtual_shadow_page_table_entry
{
    gpu_virtual_shadow_physical_mapping static_depth{};
    gpu_virtual_shadow_physical_mapping dynamic_depth{};
};

static_assert(sizeof(gpu_virtual_shadow_address_space_record) == 32);
static_assert(sizeof(gpu_virtual_shadow_view_record) == 96);
static_assert(sizeof(gpu_virtual_shadow_physical_mapping) == 16);
static_assert(sizeof(gpu_virtual_shadow_page_table_entry) == 32);
static_assert(std::is_standard_layout_v<gpu_virtual_shadow_address_space_record>);
static_assert(std::is_standard_layout_v<gpu_virtual_shadow_view_record>);
static_assert(std::is_standard_layout_v<gpu_virtual_shadow_page_table_entry>);

/** @brief Borrowed cache-owned tables for a backend upload. */
struct virtual_shadow_gpu_snapshot
{
    std::span<const gpu_virtual_shadow_address_space_record> address_spaces;
    std::span<const gpu_virtual_shadow_view_record> views;
    std::span<const gpu_virtual_shadow_page_table_entry> page_table;
    std::uint64_t revision{};
};

/** @brief Complete construction contract for a backend-neutral VSM cache. */
struct virtual_shadow_cache_config
{
    virtual_shadow_physical_pool_layout physical_pool{};
    std::uint32_t page_table_entry_capacity{static_cast<std::uint32_t>(default_virtual_shadow_page_table_bytes /
                                                                       sizeof(gpu_virtual_shadow_page_table_entry))};
    std::uint32_t view_capacity{default_virtual_shadow_view_capacity};
};

/** @brief Request emitted by receiver or caster page marking. */
struct virtual_shadow_page_request
{
    virtual_shadow_page_key key{};
    std::uint64_t frame_index{};
    std::uint64_t content_revision{};
    float projected_coverage{};
    std::uint16_t light_priority{128};
    bool coarse_page{};
};

/** @brief Resolved page-table entry consumed by rendering backends. */
struct virtual_shadow_page_mapping
{
    virtual_shadow_page_key key{};
    virtual_shadow_physical_page_handle physical_page{};
    std::uint64_t content_revision{};
    std::uint64_t last_used_frame{};
    virtual_shadow_invalidation_reason dirty_reason{virtual_shadow_invalidation_reason::newly_allocated};
    bool resident{};
    bool pinned{};
    bool in_flight{};

    [[nodiscard]] constexpr bool dirty() const noexcept
    {
        return dirty_reason != virtual_shadow_invalidation_reason::none;
    }
};

/** @brief Aggregate state for tooling and renderer diagnostics. */
struct virtual_shadow_cache_statistics
{
    std::uint32_t address_space_count{};
    std::uint32_t physical_page_capacity{};
    std::uint32_t resident_pages{};
    std::uint32_t pinned_pages{};
    std::uint32_t dirty_pages{};
    std::uint32_t allocation_count{};
    std::uint32_t eviction_count{};
    std::uint32_t cache_hits{};
    std::uint32_t cache_misses{};
    std::uint32_t parent_fallbacks{};
    std::uint32_t failed_requests{};
    std::uint64_t physical_memory_bytes{};
};

/** @brief Result of resolving one deterministic request batch. */
struct [[nodiscard]] virtual_shadow_request_result
{
    std::vector<virtual_shadow_page_mapping> render_pages;
    std::uint32_t cache_hits{};
    std::uint32_t parent_fallbacks{};
    std::uint32_t failed_requests{};
};

/**
 * @brief Persistent backend-neutral virtual shadow page allocator and cache.
 *
 * The cache owns address-space and physical-page generations but no graphics
 * API objects. Backends mirror its mappings into GPU page tables and publish a
 * page only after rendering and border replication have completed.
 */
class virtual_shadow_cache
{
public:
    explicit virtual_shadow_cache(const virtual_shadow_cache_config& config);
    explicit virtual_shadow_cache(std::uint64_t requested_budget_bytes = default_virtual_shadow_budget_bytes,
                                  std::uint64_t device_budget_bytes = 0,
                                  virtual_shadow_depth_format format = virtual_shadow_depth_format::d16_unorm);
    ~virtual_shadow_cache();

    virtual_shadow_cache(const virtual_shadow_cache&) = delete;
    virtual_shadow_cache& operator=(const virtual_shadow_cache&) = delete;
    virtual_shadow_cache(virtual_shadow_cache&&) noexcept;
    virtual_shadow_cache& operator=(virtual_shadow_cache&&) noexcept;

    [[nodiscard]] std::optional<virtual_shadow_address_space_handle>
    create_address_space(const virtual_shadow_address_space_descriptor& descriptor);
    [[nodiscard]] bool destroy_address_space(virtual_shadow_address_space_handle handle) noexcept;
    [[nodiscard]] const virtual_shadow_address_space_descriptor*
    address_space(virtual_shadow_address_space_handle handle) const noexcept;
    [[nodiscard]] bool update_address_space_views(virtual_shadow_address_space_handle handle,
                                                  std::span<const virtual_shadow_view_descriptor> views) noexcept;
    [[nodiscard]] std::optional<std::uint32_t> dense_page_index(const virtual_shadow_page_key& key) const noexcept;
    /** @brief Borrow all GPU tables until the cache is next mutated. */
    [[nodiscard]] virtual_shadow_gpu_snapshot gpu_snapshot() const noexcept;

    [[nodiscard]] virtual_shadow_request_result resolve_requests(std::span<const virtual_shadow_page_request> requests,
                                                                 std::uint64_t frame_index);
    [[nodiscard]] const virtual_shadow_page_mapping* find(const virtual_shadow_page_key& key) const noexcept;
    [[nodiscard]] const virtual_shadow_page_mapping*
    find_resident_or_ancestor(const virtual_shadow_page_key& key) const noexcept;
    /** @brief Borrow all current mappings until the cache is next mutated. */
    [[nodiscard]] std::span<const virtual_shadow_page_mapping> mappings() const noexcept;

    [[nodiscard]] bool publish(const virtual_shadow_page_key& key, std::uint64_t content_revision) noexcept;
    [[nodiscard]] bool set_in_flight(const virtual_shadow_page_key& key, bool in_flight) noexcept;
    std::uint32_t invalidate(virtual_shadow_address_space_handle handle, virtual_shadow_invalidation_reason reason,
                             std::optional<virtual_shadow_page_coordinate> coordinate = std::nullopt) noexcept;
    void clear() noexcept;

    [[nodiscard]] virtual_shadow_cache_statistics statistics() const noexcept;
    [[nodiscard]] std::uint64_t budget_bytes() const noexcept;
    [[nodiscard]] std::uint32_t physical_page_capacity() const noexcept;
    [[nodiscard]] virtual_shadow_depth_format depth_format() const noexcept;
    [[nodiscard]] const virtual_shadow_physical_pool_layout& physical_pool_layout() const noexcept;

private:
    struct address_space_slot;
    struct physical_page_slot;
    struct page_key_less;
    struct free_range;

    [[nodiscard]] virtual_shadow_page_mapping* find_mutable(const virtual_shadow_page_key& key) noexcept;
    [[nodiscard]] std::optional<virtual_shadow_physical_page_handle> allocate_physical_page(std::uint64_t frame_index);
    [[nodiscard]] std::optional<std::uint32_t> eviction_candidate(std::uint64_t frame_index) const noexcept;
    void release_mapping(const virtual_shadow_page_key& key) noexcept;
    [[nodiscard]] std::optional<std::uint32_t> allocate_range(std::vector<free_range>& ranges,
                                                              std::uint32_t count) noexcept;
    void release_range(std::vector<free_range>& ranges, std::uint32_t base, std::uint32_t count) noexcept;
    void rebuild_gpu_address_space(std::uint32_t index) noexcept;
    void clear_gpu_mapping(const virtual_shadow_page_key& key) noexcept;
    void publish_gpu_mapping(const virtual_shadow_page_mapping& mapping) noexcept;

    std::uint64_t budget_bytes_{};
    virtual_shadow_depth_format depth_format_{virtual_shadow_depth_format::d16_unorm};
    virtual_shadow_physical_pool_layout physical_pool_layout_{};
    std::uint32_t page_table_entry_capacity_{};
    std::uint32_t view_capacity_{};
    std::vector<address_space_slot> address_spaces_;
    std::vector<std::uint32_t> free_address_spaces_;
    std::vector<physical_page_slot> physical_pages_;
    std::vector<std::uint32_t> free_physical_pages_;
    std::vector<virtual_shadow_page_mapping> mappings_;
    std::vector<free_range> free_page_table_ranges_;
    std::vector<free_range> free_view_ranges_;
    std::vector<gpu_virtual_shadow_address_space_record> gpu_address_spaces_;
    std::vector<gpu_virtual_shadow_view_record> gpu_views_;
    std::vector<gpu_virtual_shadow_page_table_entry> gpu_page_table_;
    std::uint64_t gpu_revision_{1};
    virtual_shadow_cache_statistics cumulative_{};
};

/** @brief Number of pages on one axis for a normalized face/level topology. */
[[nodiscard]] constexpr std::uint32_t
virtual_shadow_pages_per_axis(const virtual_shadow_address_space_descriptor& descriptor, std::uint8_t level) noexcept
{
    const std::uint32_t base =
        (descriptor.virtual_resolution + virtual_shadow_page_texels - 1u) / virtual_shadow_page_texels;
    if (descriptor.light_kind == shadow_light_kind::directional) return std::max(1u, base);
    return std::max(1u, base >> level);
}

/** @brief Number of dense entries required by one complete address space. */
[[nodiscard]] constexpr std::uint64_t
virtual_shadow_page_table_entry_count(const virtual_shadow_address_space_descriptor& descriptor) noexcept
{
    std::uint64_t per_face{};
    for (std::uint8_t level = 0; level < descriptor.level_count; ++level)
    {
        const auto axis = virtual_shadow_pages_per_axis(descriptor, level);
        per_face += static_cast<std::uint64_t>(axis) * axis;
    }
    return per_face * descriptor.face_count;
}

/** @brief Dense level-relative offset using face-major, level-major, row-major ordering. */
[[nodiscard]] constexpr std::optional<std::uint32_t>
virtual_shadow_dense_page_offset(const virtual_shadow_address_space_descriptor& descriptor,
                                 virtual_shadow_page_coordinate coordinate) noexcept
{
    if (coordinate.face >= descriptor.face_count || coordinate.level >= descriptor.level_count) return std::nullopt;
    std::uint64_t per_face{};
    for (std::uint8_t level = 0; level < descriptor.level_count; ++level)
    {
        const auto axis = virtual_shadow_pages_per_axis(descriptor, level);
        per_face += static_cast<std::uint64_t>(axis) * axis;
    }
    std::uint64_t offset = static_cast<std::uint64_t>(coordinate.face) * per_face;
    for (std::uint8_t level = 0; level < coordinate.level; ++level)
    {
        const auto axis = virtual_shadow_pages_per_axis(descriptor, level);
        offset += static_cast<std::uint64_t>(axis) * axis;
    }
    const auto axis = virtual_shadow_pages_per_axis(descriptor, coordinate.level);
    if (coordinate.x >= axis || coordinate.y >= axis) return std::nullopt;
    offset += static_cast<std::uint64_t>(coordinate.y) * axis + coordinate.x;
    if (offset > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
    return static_cast<std::uint32_t>(offset);
}

/** @brief Build the five stable equal-grid views for one directional clipmap. */
[[nodiscard]] std::vector<virtual_shadow_view_descriptor>
make_directional_virtual_shadow_views(const virtual_shadow_address_space_descriptor& descriptor,
                                      const directional_shadow_camera& camera, const math::vector3f& camera_position,
                                      const math::vector3f& light_direction, float shadow_distance) noexcept;

/** @brief Build canonical +X/-X/+Y/-Y/+Z/-Z point-light face views for every level. */
[[nodiscard]] std::vector<virtual_shadow_view_descriptor>
make_point_virtual_shadow_views(const virtual_shadow_address_space_descriptor& descriptor,
                                const math::vector3f& light_position, float light_range) noexcept;

/** @brief Build the perspective view hierarchy for one spot light. */
[[nodiscard]] std::vector<virtual_shadow_view_descriptor>
make_spot_virtual_shadow_views(const virtual_shadow_address_space_descriptor& descriptor,
                               const math::vector3f& light_position, const math::vector3f& light_direction,
                               float outer_cone_radians, float light_range) noexcept;

/** @brief Reproject a world position into one view's page grid. */
[[nodiscard]] std::optional<virtual_shadow_page_coordinate>
virtual_shadow_page_for_world(const virtual_shadow_view_descriptor& view,
                              const math::vector3f& world_position) noexcept;

/** @brief Returns the next coarser page containing the supplied coordinate. */
[[nodiscard]] constexpr virtual_shadow_page_coordinate
virtual_shadow_parent_page(virtual_shadow_page_coordinate coordinate) noexcept
{
    coordinate.x = static_cast<std::uint16_t>(coordinate.x / 2u);
    coordinate.y = static_cast<std::uint16_t>(coordinate.y / 2u);
    ++coordinate.level;
    return coordinate;
}

/** @brief Snaps a directional clipmap origin to page-sized world increments. */
[[nodiscard]] math::vector2f snap_virtual_shadow_clipmap_origin(const math::vector2f& origin,
                                                                float world_units_per_texel) noexcept;

} // namespace arc::render
