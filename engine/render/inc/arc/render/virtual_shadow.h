#pragma once

#include <arc/math/math.h>
#include <arc/render/shadow.h>

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace arc::render
{

/** @brief Width and height, in virtual texels, represented by one VSM page. */
inline constexpr std::uint32_t virtual_shadow_page_texels = 128;
/** @brief Texels replicated around every physical VSM page for filtered sampling. */
inline constexpr std::uint32_t virtual_shadow_page_guard_texels = 4;
/** @brief Directional-light clip levels used by ARC's Ultra quality profile. */
inline constexpr std::uint32_t virtual_shadow_directional_clip_levels = 5;
/** @brief Number of frames for which a recently sampled page cannot be evicted. */
inline constexpr std::uint32_t virtual_shadow_page_protection_frames = 30;
/** @brief Default device-local memory budget for the Ultra VSM pool. */
inline constexpr std::uint64_t default_virtual_shadow_budget_bytes = 512ull * 1024ull * 1024ull;

/** @brief Physical depth format selected for the VSM page pool. */
enum class virtual_shadow_depth_format : std::uint8_t
{
    d16_unorm,
    d32_float
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

    friend constexpr bool operator==(virtual_shadow_page_coordinate, virtual_shadow_page_coordinate) noexcept =
        default;
    friend constexpr auto operator<=>(virtual_shadow_page_coordinate, virtual_shadow_page_coordinate) noexcept =
        default;
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
struct virtual_shadow_request_result
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

private:
    struct address_space_slot;
    struct physical_page_slot;
    struct page_key_less;

    [[nodiscard]] virtual_shadow_page_mapping* find_mutable(const virtual_shadow_page_key& key) noexcept;
    [[nodiscard]] std::optional<virtual_shadow_physical_page_handle> allocate_physical_page(std::uint64_t frame_index);
    [[nodiscard]] std::optional<std::uint32_t> eviction_candidate(std::uint64_t frame_index) const noexcept;
    void release_mapping(const virtual_shadow_page_key& key) noexcept;

    std::uint64_t budget_bytes_{};
    virtual_shadow_depth_format depth_format_{virtual_shadow_depth_format::d16_unorm};
    std::vector<address_space_slot> address_spaces_;
    std::vector<std::uint32_t> free_address_spaces_;
    std::vector<physical_page_slot> physical_pages_;
    std::vector<std::uint32_t> free_physical_pages_;
    std::vector<virtual_shadow_page_mapping> mappings_;
    virtual_shadow_cache_statistics cumulative_{};
};

/** @brief Number of virtual pages on one axis at the requested level. */
[[nodiscard]] constexpr std::uint32_t virtual_shadow_pages_per_axis(std::uint32_t virtual_resolution,
                                                                    std::uint8_t level) noexcept
{
    const std::uint32_t base =
        (virtual_resolution + virtual_shadow_page_texels - 1u) / virtual_shadow_page_texels;
    return std::max(1u, base >> level);
}

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
