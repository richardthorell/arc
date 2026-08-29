#pragma once

#include <arc/math/math.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace arc::render
{

inline constexpr std::uint32_t maximum_directional_shadow_cascades = 4;
inline constexpr std::uint32_t point_shadow_face_count = 6;

enum class render_mobility : std::uint8_t
{
    static_object,
    stationary,
    movable
};

enum class shadow_cache_mode : std::uint8_t
{
    automatic,
    always_update,
    static_only
};

/** @brief Shadow-map representation requested by an authored light. */
enum class shadow_map_method : std::uint8_t
{
    auto_select,
    conventional,
    virtualized
};

enum class shadow_light_kind : std::uint8_t
{
    directional,
    point,
    spot
};

struct directional_shadow_settings
{
    std::uint32_t cascade_count{maximum_directional_shadow_cascades};
    float maximum_distance{200.0f};
    float split_lambda{0.65f};
    float blend_fraction{0.10f};
    bool stable{true};
};

struct shadow_atlas_rect
{
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t size{};
    std::uint32_t guard{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return size > guard * 2u;
    }
    [[nodiscard]] constexpr std::uint32_t content_x() const noexcept
    {
        return x + guard;
    }
    [[nodiscard]] constexpr std::uint32_t content_y() const noexcept
    {
        return y + guard;
    }
    [[nodiscard]] constexpr std::uint32_t content_size() const noexcept
    {
        return size - guard * 2u;
    }
};

struct shadow_allocation_handle
{
    static constexpr std::uint32_t invalid_index = 0xffffffffu;
    std::uint32_t index{invalid_index};
    std::uint32_t generation{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index != invalid_index;
    }
    friend constexpr bool operator==(shadow_allocation_handle, shadow_allocation_handle) noexcept = default;
};

struct shadow_atlas_request
{
    shadow_light_kind kind{shadow_light_kind::spot};
    std::uint64_t light_key{};
    std::uint32_t requested_resolution{512};
    std::uint32_t minimum_resolution{256};
    std::uint16_t priority{128};
    std::uint64_t frame_index{};
};

struct shadow_atlas_allocation
{
    shadow_allocation_handle handle{};
    shadow_light_kind kind{shadow_light_kind::spot};
    std::uint64_t light_key{};
    std::array<shadow_atlas_rect, point_shadow_face_count> faces{};
    std::uint32_t face_count{};
    std::uint32_t resolved_resolution{};
    std::uint16_t priority{};
    std::uint64_t last_used_frame{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return handle.valid() && face_count > 0 && faces[0].valid();
    }
};

struct shadow_atlas_statistics
{
    std::uint32_t atlas_size{};
    std::uint32_t minimum_tile_size{};
    std::uint32_t allocation_count{};
    std::uint32_t occupied_texels{};
    std::uint32_t eviction_count{};
    std::uint32_t resolution_reduction_count{};
};

/**
 * @brief Persistent deterministic power-of-two atlas allocator.
 *
 * Allocations remain stable until explicitly released or evicted. Point lights
 * reserve all six faces atomically. The allocator never exposes backend
 * objects and can therefore be shared by Vulkan, D3D12, Metal and tests.
 */
class shadow_atlas_allocator
{
public:
    explicit shadow_atlas_allocator(std::uint32_t atlas_size = 4096, std::uint32_t minimum_tile_size = 128,
                                    std::uint32_t guard_texels = 2);
    ~shadow_atlas_allocator();

    shadow_atlas_allocator(const shadow_atlas_allocator&) = delete;
    shadow_atlas_allocator& operator=(const shadow_atlas_allocator&) = delete;
    shadow_atlas_allocator(shadow_atlas_allocator&&) noexcept;
    shadow_atlas_allocator& operator=(shadow_atlas_allocator&&) noexcept;

    [[nodiscard]] std::optional<shadow_atlas_allocation> allocate(const shadow_atlas_request& request);
    [[nodiscard]] const shadow_atlas_allocation* find(shadow_allocation_handle handle) const noexcept;
    [[nodiscard]] const shadow_atlas_allocation* find_light(shadow_light_kind kind,
                                                            std::uint64_t light_key) const noexcept;
    bool touch(shadow_allocation_handle handle, std::uint64_t frame_index, std::uint16_t priority) noexcept;
    bool release(shadow_allocation_handle handle) noexcept;
    void clear() noexcept;

    [[nodiscard]] shadow_atlas_statistics statistics() const noexcept;
    [[nodiscard]] std::uint32_t atlas_size() const noexcept
    {
        return atlas_size_;
    }

private:
    struct slot;

    [[nodiscard]] std::optional<shadow_atlas_allocation> try_allocate(const shadow_atlas_request& request,
                                                                      std::uint32_t resolution);
    [[nodiscard]] bool region_free(std::uint32_t x, std::uint32_t y, std::uint32_t cells) const noexcept;
    void mark_region(const shadow_atlas_rect& rect, bool occupied) noexcept;
    [[nodiscard]] std::optional<std::size_t> eviction_candidate(std::uint16_t incoming_priority,
                                                                std::uint64_t protected_light_key) const noexcept;

    std::uint32_t atlas_size_{};
    std::uint32_t minimum_tile_size_{};
    std::uint32_t guard_texels_{};
    std::uint32_t cells_per_axis_{};
    std::vector<std::uint8_t> occupancy_;
    std::vector<slot> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t eviction_count_{};
    std::uint32_t resolution_reduction_count_{};
};

struct directional_shadow_camera
{
    math::matrix4f inverse_view_projection{math::identity<float, 4>()};
    float near_plane{0.01f};
    float far_plane{1000.0f};
};

struct directional_shadow_cascade
{
    math::matrix4f light_view_projection{math::identity<float, 4>()};
    math::vector3f center{};
    float radius{};
    float near_depth{};
    float split_depth{};
    float blend_start_depth{};
    float texel_world_size{};
};

struct directional_shadow_layout
{
    std::array<directional_shadow_cascade, maximum_directional_shadow_cascades> cascades{};
    std::uint32_t cascade_count{};
};

[[nodiscard]] directional_shadow_layout fit_directional_shadow_cascades(const directional_shadow_camera& camera,
                                                                        const math::vector3f& light_direction,
                                                                        const directional_shadow_settings& settings,
                                                                        std::uint32_t resolution) noexcept;

} // namespace arc::render
