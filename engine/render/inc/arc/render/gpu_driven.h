#pragma once

#include <arc/core/id.h>
#include <arc/render/handles.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
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
    /** Byte stride of one complete GPU-visible table record. */
    std::uint32_t element_stride{};
    /** Earliest completed frame after which tombstoned slots may be reused. */
    std::uint64_t reuse_after_frame{};
    /** Individual slot mutations. */
    std::vector<gpu_table_update> updates;
    /** Sorted coalesced table ranges suitable for sparse copies. */
    std::vector<gpu_table_dirty_range> dirty_ranges;
    /** Packed immutable bytes referenced by update payload offsets. */
    std::vector<std::byte> payload;
    /** Monotonic generation of the shared geometry heaps. */
    std::uint32_t geometry_heap_generation{};
    /** Required vertex heap capacity after applying this batch. */
    std::uint64_t vertex_heap_capacity{};
    /** Required index heap capacity after applying this batch. */
    std::uint64_t index_heap_capacity{};
    /** Sparse byte updates targeting the shared vertex/index heaps. */
    struct heap_update
    {
        /** False targets the vertex heap; true targets the index heap. */
        bool index_heap{};
        /** Destination byte offset in the selected heap. */
        std::uint64_t destination_offset{};
        /** Byte offset into @ref heap_payload. */
        std::uint32_t payload_offset{};
        /** Number of bytes copied by this update. */
        std::uint32_t payload_size{};
    };
    std::vector<heap_update> heap_updates;
    /** Immutable bytes referenced by @ref heap_updates. */
    std::vector<std::byte> heap_payload;
};

/** @brief GPU record describing conventional indexed geometry in shared heaps. */
struct alignas(16) gpu_geometry_table_record
{
    std::uint32_t generation{};
    std::uint32_t flags{};
    std::uint64_t vertex_offset{};
    std::uint64_t index_offset{};
    std::uint32_t vertex_count{};
    std::uint32_t index_count{};
    std::uint32_t vertex_stride{};
    std::uint32_t index_stride{sizeof(std::uint32_t)};
    std::uint32_t reserved[2]{};
};

/** @brief GPU record describing a stable sampled-image table entry. */
struct alignas(16) gpu_texture_table_record
{
    std::uint32_t generation{};
    std::uint32_t descriptor_index{resource_handle::invalid_index};
    std::uint32_t descriptor_generation{};
    std::uint32_t flags{};
    std::uint32_t mip_window_base{};
    std::uint32_t mip_count{1};
    std::uint32_t width{};
    std::uint32_t height{};
};

/** @brief GPU record describing one immutable sampler descriptor. */
struct alignas(16) gpu_sampler_table_record
{
    std::uint32_t generation{};
    std::uint32_t descriptor_index{resource_handle::invalid_index};
    std::uint32_t descriptor_generation{};
    std::uint32_t flags{};
};

/** @brief GPU record describing current and previous skin-palette ranges. */
struct alignas(16) gpu_skin_palette_table_record
{
    std::uint32_t generation{};
    std::uint32_t joint_count{};
    std::uint64_t current_offset{};
    std::uint64_t previous_offset{};
    std::uint64_t byte_size{};
};

/** @brief GPU record containing packed material constants and stable texture references. */
struct alignas(16) gpu_material_table_record
{
    static constexpr std::size_t texture_slot_count = 12;

    std::uint32_t generation{};
    std::uint32_t flags{};
    float base_color[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float emissive[4]{};
    float surface[4]{};
    std::array<std::uint32_t, texture_slot_count> texture_indices{};
    std::array<std::uint32_t, texture_slot_count> texture_generations{};
    std::uint32_t reserved[2]{};
};

static_assert(sizeof(gpu_geometry_table_record) == 48);
static_assert(sizeof(gpu_texture_table_record) == 32);
static_assert(sizeof(gpu_sampler_table_record) == 16);
static_assert(sizeof(gpu_skin_palette_table_record) == 32);
static_assert(sizeof(gpu_material_table_record) == 160);

/** @brief Untyped stable slot reference returned by the resource-table authority. */
struct gpu_resource_table_reference
{
    std::uint32_t index{resource_handle::invalid_index};
    std::uint32_t generation{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index != resource_handle::invalid_index && generation != 0u;
    }

    friend constexpr bool operator==(gpu_resource_table_reference, gpu_resource_table_reference) noexcept = default;
};

/** @brief Occupancy and generation state for one renderer-owned stable table. */
struct gpu_resource_table_snapshot
{
    gpu_resource_table_kind table{gpu_resource_table_kind::geometry};
    std::uint32_t table_generation{};
    std::uint32_t capacity{};
    std::uint32_t live_entries{};
    std::uint32_t tombstones{};
    std::uint32_t element_stride{};
    std::uint64_t sparse_upload_bytes{};
};

/** @brief Capacity state for the renderer-owned shared conventional-geometry heaps. */
struct gpu_geometry_heap_snapshot
{
    std::uint32_t generation{};
    std::uint64_t vertex_bytes{};
    std::uint64_t index_bytes{};
    std::uint64_t live_vertex_bytes{};
    std::uint64_t live_index_bytes{};
    std::uint32_t live_allocations{};
};

/**
 * @brief Renderer-owned authority for generational GPU resource tables and shared geometry heaps.
 *
 * Table indices intentionally match renderer handle indices. A replacement record carries the new handle generation,
 * while backends retain prior table-buffer generations until their frames retire. This keeps scene handles stable and
 * prevents an in-flight frame from observing a recycled record.
 */
class gpu_resource_tables
{
public:
    gpu_resource_tables();

    [[nodiscard]] gpu_table_update_batch publish_geometry(resource_handle handle, std::span<const std::byte> vertices,
                                                          std::uint32_t vertex_stride,
                                                          std::span<const std::byte> indices,
                                                          std::uint32_t index_stride, std::uint64_t frame_index);
    [[nodiscard]] gpu_table_update_batch
    publish_material(resource_handle handle, const gpu_material_table_record& record, std::uint64_t frame_index);
    [[nodiscard]] gpu_table_update_batch publish_texture(resource_handle handle, const gpu_texture_table_record& record,
                                                         std::uint64_t frame_index);
    [[nodiscard]] gpu_table_update_batch publish_sampler(resource_handle handle, const gpu_sampler_table_record& record,
                                                         std::uint64_t frame_index);
    [[nodiscard]] gpu_table_update_batch publish_skin_palette(resource_handle handle,
                                                              const gpu_skin_palette_table_record& record,
                                                              std::uint64_t frame_index);
    [[nodiscard]] gpu_table_update_batch tombstone(gpu_resource_table_kind table, resource_handle handle,
                                                   std::uint64_t frame_index);

    [[nodiscard]] std::optional<gpu_resource_table_reference> find(gpu_resource_table_kind table,
                                                                   resource_handle handle) const noexcept;
    [[nodiscard]] gpu_resource_table_snapshot snapshot(gpu_resource_table_kind table) const noexcept;
    [[nodiscard]] gpu_geometry_heap_snapshot geometry_heap_snapshot() const noexcept;
    void reset();

private:
    struct table_slot
    {
        std::uint32_t generation{};
        bool live{};
    };

    struct table_state
    {
        gpu_resource_table_kind kind{gpu_resource_table_kind::geometry};
        std::uint32_t generation{1};
        std::uint32_t stride{};
        std::uint32_t live_entries{};
        std::uint32_t tombstones{};
        std::uint64_t sparse_upload_bytes{};
        std::vector<table_slot> slots;
    };

    struct heap_range
    {
        std::uint64_t offset{};
        std::uint64_t size{};
    };

    struct geometry_allocation
    {
        heap_range vertices;
        heap_range indices;
    };

    [[nodiscard]] gpu_table_update_batch publish_record(gpu_resource_table_kind table, resource_handle handle,
                                                        std::span<const std::byte> record, std::uint64_t frame_index);
    [[nodiscard]] heap_range allocate_heap_range(std::vector<std::byte>& heap, std::vector<heap_range>& free_ranges,
                                                 std::uint64_t size, std::uint64_t alignment);
    void release_heap_range(std::vector<heap_range>& free_ranges, heap_range range);
    [[nodiscard]] static std::size_t table_offset(gpu_resource_table_kind table) noexcept;

    std::array<table_state, 7> tables_;
    std::vector<std::byte> vertex_heap_;
    std::vector<std::byte> index_heap_;
    std::vector<heap_range> free_vertex_ranges_;
    std::vector<heap_range> free_index_ranges_;
    std::unordered_map<std::uint64_t, geometry_allocation> geometry_allocations_;
    std::uint32_t geometry_heap_generation_{1};
    std::uint64_t live_vertex_bytes_{};
    std::uint64_t live_index_bytes_{};
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

/** @brief Stable CPU reference for the GPU count/prefix/scatter compaction stages. */
struct gpu_draw_compaction_result
{
    /** Draw records in pipeline-bin order, preserving input order within each bin. */
    std::vector<gpu_draw_record> visible_draws;
    /** First compacted record for every configured pipeline bin. */
    std::vector<std::uint32_t> bin_offsets;
    /** Number of compacted records in every configured pipeline bin. */
    std::vector<std::uint32_t> bin_counts;
    /** Records rejected because either the output or pipeline-bin capacity was exceeded. */
    std::vector<gpu_draw_record> overflow_draws;
    /** Aggregate counters matching the executable compute path. */
    gpu_visibility_statistics statistics;
};

/**
 * @brief Count, prefix, and stably scatter visible records into bounded pipeline bins.
 * @details Invalid bins and records beyond @p maximum_visible_draws are returned for the classic correctness path.
 */
[[nodiscard]] gpu_draw_compaction_result compact_gpu_draw_records(std::span<const gpu_draw_record> records,
                                                                  std::uint32_t pipeline_bin_capacity,
                                                                  std::uint32_t maximum_visible_draws);

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
