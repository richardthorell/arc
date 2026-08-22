#pragma once

#include <arc/core/id.h>
#include <arc/render/handles.h>

#include <cstdint>
#include <span>
#include <vector>

namespace arc::render
{

/** @brief Maximum number of bounded pipeline bins used by the portable GPU-driven path. */
inline constexpr std::uint32_t default_gpu_pipeline_bin_capacity = 512u;
/** @brief Conservative frame delay before a tombstoned GPU table slot may be reused. */
inline constexpr std::uint64_t default_gpu_table_slot_reuse_delay_frames = 4u;

/** @brief Resource table addressed by a GPU-visible runtime index. */
enum class gpu_resource_table_kind : std::uint8_t
{
    /** Shared conventional and specialized geometry metadata. */
    geometry,
    /** Packed material constants and texture references. */
    material,
    /** Sampled-image descriptor slots. */
    texture,
    /** Sampler descriptor slots. */
    sampler,
    /** Current and previous skin-palette records. */
    skin_palette,
    /** Persistent per-world GPU Scene instances. */
    instance,
    /** Per-view compacted visible records. */
    visible_draw
};

/** @brief Geometry execution path selected for one GPU Scene record. */
enum class gpu_geometry_path : std::uint8_t
{
    /** Indexed geometry stored in shared vertex and index heaps. */
    conventional,
    /** Visible-only GPU-deformed indexed geometry. */
    skinned,
    /** Heightfield patches selected from a terrain hierarchy. */
    terrain,
    /** Cluster hierarchy selected from virtual geometry. */
    virtual_geometry
};

/** @brief Resource binding model selected by feature resolution. */
enum class gpu_resource_binding_model : std::uint8_t
{
    /** Bind resources explicitly for each compatible batch. */
    classic,
    /** Resolve resources through stable non-uniform GPU table indices. */
    bindless
};

/** @brief Runtime GPU geometry-table index tag. */
struct gpu_geometry_index_tag;
/** @brief Runtime GPU material-table index tag. */
struct gpu_material_index_tag;
/** @brief Runtime GPU texture-table index tag. */
struct gpu_texture_index_tag;
/** @brief Runtime GPU sampler-table index tag. */
struct gpu_sampler_index_tag;
/** @brief Runtime GPU skin-palette-table index tag. */
struct gpu_skin_palette_index_tag;
/** @brief Runtime GPU pipeline-bin index tag. */
struct gpu_pipeline_bin_index_tag;
/** @brief Runtime compacted visible-draw index tag. */
struct gpu_visible_draw_index_tag;

/** @brief Stable runtime index into the geometry table. */
using gpu_geometry_index = core::strong_id<gpu_geometry_index_tag, std::uint32_t>;
/** @brief Stable runtime index into the material table. */
using gpu_material_index = core::strong_id<gpu_material_index_tag, std::uint32_t>;
/** @brief Stable runtime index into the texture table. */
using gpu_texture_index = core::strong_id<gpu_texture_index_tag, std::uint32_t>;
/** @brief Stable runtime index into the sampler table. */
using gpu_sampler_index = core::strong_id<gpu_sampler_index_tag, std::uint32_t>;
/** @brief Stable runtime index into the skin-palette table. */
using gpu_skin_palette_index = core::strong_id<gpu_skin_palette_index_tag, std::uint32_t>;
/** @brief Stable runtime index into the pipeline-bin table. */
using gpu_pipeline_bin_index = core::strong_id<gpu_pipeline_bin_index_tag, std::uint32_t>;
/** @brief Stable runtime index into the compacted visible-draw table. */
using gpu_visible_draw_index = core::strong_id<gpu_visible_draw_index_tag, std::uint32_t>;

/**
 * @brief GPU-visible table reference protected by a generation value.
 * @tparam Index Strong index type naming the target table.
 */
template <class Index> struct gpu_table_reference
{
    /** Runtime table slot. */
    Index index{};
    /** Generation mirrored in the GPU validation table. */
    std::uint32_t generation{};

    /** @return Whether this reference names a possible table slot. */
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index.valid() && generation != 0u;
    }

    friend constexpr bool operator==(const gpu_table_reference&, const gpu_table_reference&) noexcept = default;
};

/** @brief One coalesced range of table slots changed by an update batch. */
struct gpu_table_dirty_range
{
    /** First changed table slot. */
    std::uint32_t first{};
    /** Number of adjacent changed slots. */
    std::uint32_t count{};

    /** @return One-past-the-last changed slot. */
    [[nodiscard]] constexpr std::uint32_t end() const noexcept
    {
        return first + count;
    }

    friend constexpr bool operator==(const gpu_table_dirty_range&, const gpu_table_dirty_range&) noexcept = default;
};

/** @brief Mutation applied to one persistent GPU resource-table slot. */
enum class gpu_table_update_kind : std::uint8_t
{
    /** Publish or replace a live record at the named slot and generation. */
    upsert,
    /** Invalidate a slot while deferring its reuse until frame retirement. */
    tombstone,
    /** Discard the complete table generation before applying later updates. */
    reset
};

/** @brief Backend-neutral metadata for one packed table mutation. */
struct gpu_table_update
{
    /** Resource table receiving the mutation. */
    gpu_resource_table_kind table{gpu_resource_table_kind::geometry};
    /** Mutation semantics. */
    gpu_table_update_kind kind{gpu_table_update_kind::upsert};
    /** Stable table slot. */
    std::uint32_t slot{};
    /** Generation validated by GPU consumers. */
    std::uint32_t generation{};
    /** Byte offset into the frame-owned packed update payload. */
    std::uint32_t payload_offset{};
    /** Number of payload bytes for an upsert operation. */
    std::uint32_t payload_size{};
};

/** @brief Sparse mutations and capacity state for one table publication. */
struct gpu_table_update_batch
{
    /** Resource table receiving the batch. */
    gpu_resource_table_kind table{gpu_resource_table_kind::geometry};
    /** Monotonic table storage generation. */
    std::uint32_t table_generation{};
    /** Required number of addressable slots after publication. */
    std::uint32_t capacity{};
    /** Earliest completed frame after which tombstoned slots may be reused. */
    std::uint64_t reuse_after_frame{};
    /** Individual slot mutations. */
    std::vector<gpu_table_update> updates;
    /** Sorted coalesced table ranges suitable for sparse copies. */
    std::vector<gpu_table_dirty_range> dirty_ranges;
    /** Packed immutable bytes referenced by update payload offsets. */
    std::vector<std::byte> payload;
};

/** @brief Compact draw metadata consumed by indirect-command generation. */
struct gpu_draw_record
{
    /** Persistent GPU Scene instance slot. */
    std::uint32_t instance_index{};
    /** Runtime geometry-table slot. */
    std::uint32_t geometry_index{resource_handle::invalid_index};
    /** Runtime material-table slot. */
    std::uint32_t material_index{resource_handle::invalid_index};
    /** Bounded pipeline bin selected for the draw. */
    std::uint32_t pipeline_bin{};
    /** Stable back-to-front key used by transparent radix sorting. */
    std::uint64_t sort_key{};
};

/** @brief Backend-neutral key used to place a draw into a compatible PSO bin. */
struct gpu_pipeline_bin_key
{
    /** Render pass identifier local to the renderer. */
    std::uint16_t render_pass{};
    /** Vertex-layout identifier. */
    std::uint8_t vertex_layout{};
    /** @ref gpu_geometry_path representation. */
    gpu_geometry_path geometry_path{gpu_geometry_path::conventional};
    /** Material alpha-mode identifier. */
    std::uint8_t alpha_mode{};
    /** Material shading-model identifier. */
    std::uint8_t shading_model{};
    /** Nonzero for double-sided rasterization. */
    std::uint8_t double_sided{};
    /** Reserved feature/permutation flags. */
    std::uint8_t flags{};

    friend constexpr bool operator==(const gpu_pipeline_bin_key&, const gpu_pipeline_bin_key&) noexcept = default;
};

/** @brief Per-view visibility and command-generation counters. */
struct gpu_visibility_statistics
{
    /** Instances considered by the view. */
    std::uint32_t candidates{};
    /** Records surviving all visibility tests. */
    std::uint32_t visible{};
    /** Records rejected by the frustum. */
    std::uint32_t frustum_rejected{};
    /** Records rejected by authored distance. */
    std::uint32_t distance_rejected{};
    /** Records conservatively rejected by HZB. */
    std::uint32_t occlusion_rejected{};
    /** Pipeline bins containing commands. */
    std::uint32_t active_bins{};
    /** Generated indexed or mesh-task commands. */
    std::uint32_t indirect_commands{};
    /** Records participating in transparent sorting. */
    std::uint32_t transparent_records{};
    /** Records routed to a correctness fallback. */
    std::uint32_t overflow_records{};
    /** CPU draw submissions used by the selected fallback. */
    std::uint32_t cpu_submissions{};
};

/**
 * @brief Coalesce unordered changed slot indices into ascending adjacent ranges.
 * @param indices Changed slots; duplicates are permitted.
 * @return Sorted non-overlapping ranges suitable for sparse uploads.
 */
[[nodiscard]] std::vector<gpu_table_dirty_range>
coalesce_gpu_table_dirty_ranges(std::span<const std::uint32_t> indices);

/**
 * @brief Build a stable descending transparent sort key.
 * @param normalized_depth Device depth clamped to `0..1`.
 * @param pipeline_bin Bounded pipeline-bin index.
 * @param stable_instance_index Stable tie breaker.
 * @return Key ordered back-to-front by ordinary ascending integer sort.
 */
[[nodiscard]] std::uint64_t make_gpu_transparent_sort_key(float normalized_depth, std::uint16_t pipeline_bin,
                                                          std::uint32_t stable_instance_index) noexcept;

} // namespace arc::render
