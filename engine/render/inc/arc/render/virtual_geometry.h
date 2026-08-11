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
