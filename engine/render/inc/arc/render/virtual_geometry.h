#pragma once

#include <arc/render/virtual_mesh.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace arc::render
{

/** @brief Residency state of one independently streamable virtual-geometry page. */
enum class virtual_geometry_page_state : std::uint8_t
{
    nonresident,
    requested,
    loading,
    resident,
    failed
};

/** @brief GPU-visible residency flags for one virtual-geometry page-table entry. */
enum class virtual_geometry_gpu_page_flag : std::uint32_t
{
    none = 0,
    resident = 1u << 0u,
    root = 1u << 1u,
    loading = 1u << 2u,
    failed = 1u << 3u
};

/** @brief Immutable GPU table record for one realized virtual-geometry resource. */
struct virtual_geometry_gpu_resource_record
{
    std::uint32_t first_node{};
    std::uint32_t node_count{};
    std::uint32_t first_cluster{};
    std::uint32_t cluster_count{};
    std::uint32_t first_child{};
    std::uint32_t child_count{};
    std::uint32_t first_page{};
    std::uint32_t page_count{};
    std::uint32_t first_root{};
    std::uint32_t root_count{};
    std::uint32_t generation{};
    std::uint32_t flags{};
};

/** @brief Compact GPU hierarchy node using object-space spheres and normal cones. */
struct virtual_geometry_gpu_node_record
{
    std::array<float, 4> sphere{};
    std::array<float, 4> normal_cone{};
    float geometric_error{};
    std::uint32_t first_cluster{};
    std::uint32_t cluster_count{};
    std::uint32_t first_child{};
    std::uint32_t child_count{};
    std::uint32_t page_index{};
    std::uint32_t level{};
    std::uint32_t flags{};
};

/** @brief Compact GPU metadata for one independently rasterizable cluster. */
struct virtual_geometry_gpu_cluster_record
{
    std::array<float, 4> sphere{};
    std::array<float, 4> normal_cone{};
    std::array<float, 4> bounds_min_error{};
    std::array<float, 4> bounds_max{};
    std::uint32_t page_index{};
    std::uint32_t page_byte_offset{};
    std::uint32_t vertex_count{};
    std::uint32_t triangle_count{};
    std::uint32_t material_section{};
    std::uint32_t hierarchy_node{};
    std::uint32_t flags{};
    std::uint32_t reserved{};
};

/** @brief Frame-safe GPU page-table entry addressed through a renderer-owned page heap. */
struct virtual_geometry_gpu_page_record
{
    std::uint32_t heap_index{invalid_virtual_geometry_index};
    std::uint32_t heap_byte_offset{};
    std::uint32_t stored_size{};
    std::uint32_t decoded_size{};
    std::uint32_t resource_generation{};
    virtual_geometry_gpu_page_flag flags{virtual_geometry_gpu_page_flag::none};
    std::uint32_t last_used_frame{};
    std::uint32_t reserved{};
};

/** @brief Generation-stamped feedback emitted by GPU hierarchy traversal for a missing page. */
struct virtual_geometry_gpu_page_request
{
    std::uint32_t resource_index{};
    std::uint32_t handle_generation{};
    std::uint32_t resource_generation{};
    std::uint32_t page_index{};
    float projected_error{};
    float screen_coverage{};
    float distance{};
    std::uint32_t flags{};
};

/** @brief One resident cluster selected by GPU traversal for visibility or shadow rasterization. */
struct virtual_geometry_visible_cluster_record
{
    std::uint32_t instance_index{};
    std::uint32_t resource_index{};
    std::uint32_t cluster_index{};
    std::uint32_t page_index{};
    std::uint32_t material_index{};
    std::uint32_t hierarchy_level{};
    std::uint32_t flags{};
    float view_depth{};
};

/** @brief Tile/raster bin emitted before software visibility rasterization. */
struct virtual_geometry_raster_bin_record
{
    std::uint32_t first_cluster{};
    std::uint32_t cluster_count{};
    std::uint32_t tile_x{};
    std::uint32_t tile_y{};
};

/** @brief Bounded traversal counters and correctness-fallback information. */
struct virtual_geometry_overflow_record
{
    std::uint32_t visible_cluster_overflow{};
    std::uint32_t page_request_overflow{};
    std::uint32_t raster_bin_overflow{};
    std::uint32_t fallback_instance_count{};
};

/** @brief Sparse renderer update for virtual-geometry GPU metadata or page residency. */
struct virtual_geometry_gpu_table_update
{
    virtual_mesh_handle resource{};
    std::uint32_t resource_generation{};
    std::uint32_t first_record{};
    std::vector<virtual_geometry_gpu_resource_record> resources;
    std::vector<virtual_geometry_gpu_node_record> nodes;
    std::vector<virtual_geometry_gpu_cluster_record> clusters;
    std::vector<std::uint32_t> children;
    std::vector<std::uint32_t> roots;
    std::vector<virtual_geometry_gpu_page_record> pages;
};

/** @brief Decoded page bytes ready for frame-safe publication into the GPU page heap. */
struct virtual_geometry_page_upload
{
    virtual_mesh_handle resource{};
    std::uint32_t resource_generation{};
    std::uint32_t page_index{};
    std::shared_ptr<const std::vector<std::byte>> decoded_bytes;
    std::uint32_t compressed_cpu_bytes{};
};

/** @brief Asynchronous feedback copied from one completed GPU traversal frame. */
struct virtual_geometry_feedback_readback
{
    std::uint64_t frame_index{};
    std::vector<virtual_geometry_gpu_page_request> page_requests;
    virtual_geometry_overflow_record overflow{};
};

/** @brief Bounded list capacities shared by GPU traversal and its deterministic validation path. */
struct virtual_geometry_traversal_limits
{
    std::uint32_t maximum_visible_clusters{1u << 20u};
    std::uint32_t maximum_page_requests{4096u};
};

/** @brief Generation-stamped validation output matching the GPU traversal buffers. */
struct [[nodiscard]] virtual_geometry_gpu_reference_result
{
    std::vector<virtual_geometry_visible_cluster_record> visible_clusters;
    virtual_geometry_feedback_readback feedback;
    std::uint32_t frustum_rejected{};
    std::uint32_t cone_rejected{};
    std::uint32_t hzb_rejected{};
    std::uint32_t projected_size_rejected{};
};

/** @brief Runtime page-cache policy selected by the resolved quality tier. */
struct virtual_geometry_residency_config
{
    std::uint64_t gpu_budget_bytes{512ull * 1024ull * 1024ull};
    std::uint64_t compressed_cpu_budget_bytes{256ull * 1024ull * 1024ull};
    std::uint32_t maximum_requests_per_frame{4096};
    std::uint32_t protected_frame_count{30};
};

/** @brief One GPU-generated or CPU-reference request for a missing geometry page. */
struct virtual_geometry_page_request
{
    virtual_mesh_handle resource{};
    std::uint32_t resource_generation{};
    std::uint32_t page_index{};
    float projected_error{};
    float screen_coverage{};
    float distance{};
    bool visible_child{};
    bool shadow_view{};
};

/** @brief Page request selected for asynchronous range IO and decompression. */
struct virtual_geometry_page_load
{
    virtual_mesh_handle resource{};
    std::uint32_t resource_generation{};
    std::uint32_t page_index{};
    std::uint32_t byte_offset{};
    std::uint32_t byte_size{};
    float priority{};
};

/** @brief Aggregate residency diagnostics for editor and telemetry consumers. */
struct virtual_geometry_residency_snapshot
{
    std::uint64_t frame_index{};
    std::uint64_t gpu_budget_bytes{};
    std::uint64_t gpu_resident_bytes{};
    std::uint64_t compressed_cpu_budget_bytes{};
    std::uint64_t compressed_cpu_resident_bytes{};
    std::uint32_t resource_count{};
    std::uint32_t page_count{};
    std::uint32_t resident_pages{};
    std::uint32_t requested_pages{};
    std::uint32_t failed_pages{};
    std::uint32_t evictions{};
    std::uint32_t deduplicated_requests{};
    std::uint32_t parent_fallbacks{};
    std::uint32_t stale_requests{};
    std::uint32_t protected_pages{};
};

/** @brief Object-space traversal inputs used by tests and backend validation. */
struct virtual_geometry_reference_view
{
    /** Object-space frustum planes stored as `(normal.xyz, distance)`. */
    std::array<math::vector4f, 6> frustum_planes{};
    math::vector3f camera_position{};
    /** Converts object-space error and radius to pixels at unit distance. */
    float projection_scale{1.0f};
    float geometric_error_threshold{1.0f};
    float minimum_projected_radius{0.5f};
    bool camera_cut{};
    bool double_sided{};
    /** Optional conservative previous-frame HZB callback. */
    bool (*occluded)(const math::vector3f& center, float radius, void* user_data){};
    void* occlusion_user_data{};
};

/** @brief Deterministic reference result for validating GPU hierarchy traversal. */
struct [[nodiscard]] virtual_geometry_reference_result
{
    std::vector<std::uint32_t> visible_clusters;
    std::vector<std::uint32_t> requested_pages;
    std::uint32_t frustum_rejected{};
    std::uint32_t cone_rejected{};
    std::uint32_t hzb_rejected{};
    std::uint32_t projected_size_rejected{};
    std::uint32_t parent_fallbacks{};
};

/**
 * @brief Traverses one cooked hierarchy using the same conservative policy required of GPU paths.
 * @param geometry Immutable cooked virtual-geometry metadata.
 * @param resident_pages Byte flags indexed by page ID; missing entries are nonresident.
 * @param view Object-space view, error, and optional occlusion state.
 * @return Selected resident clusters and requested missing child pages.
 */
[[nodiscard]] virtual_geometry_reference_result
traverse_virtual_geometry_reference(const virtual_mesh_data& geometry, std::span<const std::uint8_t> resident_pages,
                                    const virtual_geometry_reference_view& view);

/** @brief Build compact backend-neutral GPU metadata for one virtual-mesh generation. */
[[nodiscard]] virtual_geometry_gpu_table_update
make_virtual_geometry_gpu_table_update(virtual_mesh_handle resource, const virtual_mesh_data& geometry,
                                       std::uint32_t resource_generation);

/**
 * @brief Produce the exact bounded records expected from one GPU hierarchy traversal.
 * @details This path is used by deterministic tests and correctness fallback diagnostics; production Ultra frames
 * execute the equivalent algorithm in compute or mesh shaders.
 */
[[nodiscard]] virtual_geometry_gpu_reference_result traverse_virtual_geometry_gpu_reference(
    virtual_mesh_handle resource, std::uint32_t resource_generation, std::uint32_t instance_index,
    std::uint32_t material_index, const virtual_mesh_data& geometry, std::span<const std::uint8_t> resident_pages,
    const virtual_geometry_reference_view& view, virtual_geometry_traversal_limits limits = {});

/**
 * @brief Render-thread authority for virtual-geometry page requests and eviction.
 *
 * Asset and IO adapters consume load requests and return completed page sizes.
 * The manager never performs filesystem or backend operations itself.
 */
class virtual_geometry_residency_manager
{
public:
    explicit virtual_geometry_residency_manager(virtual_geometry_residency_config config = {});
    ~virtual_geometry_residency_manager();
    virtual_geometry_residency_manager(virtual_geometry_residency_manager&&) noexcept;
    virtual_geometry_residency_manager& operator=(virtual_geometry_residency_manager&&) noexcept;
    virtual_geometry_residency_manager(const virtual_geometry_residency_manager&) = delete;
    virtual_geometry_residency_manager& operator=(const virtual_geometry_residency_manager&) = delete;

    void configure(virtual_geometry_residency_config config);
    void register_resource(virtual_mesh_handle resource, const virtual_mesh_data& data, std::uint32_t generation);
    void unregister_resource(virtual_mesh_handle resource);
    void begin_frame(std::uint64_t frame_index);
    void request(std::span<const virtual_geometry_page_request> requests);
    /** @brief Validate and ingest asynchronous feedback produced by GPU hierarchy traversal. */
    void request_gpu(std::span<const virtual_geometry_gpu_page_request> requests);
    [[nodiscard]] std::vector<virtual_geometry_page_load> take_load_requests();
    void mark_loading(virtual_mesh_handle resource, std::uint32_t generation, std::uint32_t page_index);
    void publish(virtual_mesh_handle resource, std::uint32_t generation, std::uint32_t page_index,
                 std::uint32_t gpu_bytes, std::uint32_t compressed_cpu_bytes);
    void fail(virtual_mesh_handle resource, std::uint32_t generation, std::uint32_t page_index);
    void touch(virtual_mesh_handle resource, std::uint32_t generation, std::uint32_t page_index);
    [[nodiscard]] bool resident(virtual_mesh_handle resource, std::uint32_t generation,
                                std::uint32_t page_index) const noexcept;
    void note_parent_fallback() noexcept;
    [[nodiscard]] virtual_geometry_residency_snapshot snapshot() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace arc::render
