#pragma once

#include <arc/geometric/box.h>
#include <arc/math/matrix.h>
#include <arc/math/vector.h>
#include <arc/render/handles.h>
#include <arc/render/material.h>
#include <arc/render/mesh.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace arc::render
{

/** @brief Authored indirect-lighting policy resolved against the active quality tier. */
enum class indirect_lighting_method : std::uint8_t
{
    auto_select,
    baked_probe,
    screen_space,
    software,
    hybrid_hardware
};

/** @brief Concrete trace path executed for an indirect-lighting view. */
enum class lighting_trace_path : std::uint8_t
{
    disabled,
    baked_probe,
    screen_space,
    software_distance_field,
    hybrid_hardware
};

/** @brief Source that resolved one diffuse or reflection trace. */
enum class lighting_trace_source : std::uint8_t
{
    none,
    screen_space,
    software_distance_field,
    hardware_ray_query,
    radiance_probe,
    environment
};

/** @brief Whether a cooked field stores signed or conservative unsigned distances. */
enum class distance_field_mode : std::uint8_t
{
    signed_distance,
    two_sided_unsigned_distance
};

using lighting_geometry_handle = resource_handle;
using lighting_scene_instance_handle = resource_handle;
using surface_cache_page_handle = resource_handle;
using radiance_probe_handle = resource_handle;

/** @brief One orthographic material capture generated around a mesh section. */
struct surface_card_descriptor
{
    math::vector3f center{};
    math::vector3f normal{0.0f, 0.0f, 1.0f};
    math::vector3f tangent{1.0f, 0.0f, 0.0f};
    math::vector2f extent{1.0f, 1.0f};
    float depth_extent{1.0f};
    float texel_density{32.0f};
    float geometric_error{};
    std::uint32_t material_section{};
    std::uint32_t fallback_card{};
};

/** @brief Independently streamable 8x8x8 brick within a mesh distance field. */
struct mesh_distance_field_brick
{
    std::array<std::uint16_t, 3> coordinate{};
    std::uint16_t reserved{};
    std::uint32_t page_index{};
    std::uint32_t page_offset{};
    std::uint32_t byte_size{};
    float minimum_distance{};
    float maximum_distance{};
};

/** @brief Cooked sparse mesh distance-field metadata and independently encoded pages. */
struct mesh_distance_field_descriptor
{
    static constexpr std::uint32_t brick_dimension = 8;
    static constexpr std::uint32_t page_size = 64u * 1024u;

    geometric::box3f bounds{};
    std::array<std::uint32_t, 3> dimensions{};
    math::vector3f voxel_size = math::vector3f::one;
    /** Metres represented by an encoded signed-normalized value of one. */
    float distance_scale{1.0f};
    distance_field_mode mode{distance_field_mode::two_sided_unsigned_distance};
    std::vector<mesh_distance_field_brick> bricks;
    std::vector<std::uint32_t> page_offsets;
    std::vector<std::byte> pages;
    std::uint64_t content_hash{};
};

/** @brief Backend-neutral lighting representation cooked from one geometry subasset. */
struct lighting_geometry_descriptor
{
    std::string name;
    geometric::box3f bounds{};
    std::vector<surface_card_descriptor> cards;
    mesh_distance_field_descriptor distance_field;
    std::uint32_t material_section_count{1};
    std::uint32_t generation{1};
    bool opaque{true};
    bool masked{};
    bool double_sided{};
};

/** @brief Deterministic cooker controls for cards and sparse distance fields. */
struct lighting_geometry_build_options
{
    std::uint32_t minimum_distance_field_resolution{16};
    std::uint32_t maximum_distance_field_resolution{64};
    float distance_field_resolution_scale{1.0f};
    float card_texel_density{32.0f};
    float narrow_band_voxels{2.0f};
};

/** @brief Diagnostics emitted while constructing a lighting representation. */
struct lighting_geometry_build_statistics
{
    std::uint32_t source_triangles{};
    std::uint32_t rejected_triangles{};
    std::uint32_t card_count{};
    std::uint32_t brick_count{};
    std::uint32_t page_count{};
    std::uint64_t encoded_bytes{};
    bool watertight{};
};

/** @brief Complete deterministic lighting-geometry build result. */
struct [[nodiscard]] lighting_geometry_build_result
{
    lighting_geometry_descriptor geometry;
    lighting_geometry_build_statistics statistics;
    std::vector<std::string> diagnostics;
};

/** @brief Build surface cards and a sparse distance field from conventional proxy geometry. */
[[nodiscard]] lighting_geometry_build_result
build_lighting_geometry(const mesh_data& mesh, const lighting_geometry_build_options& options = {});

/** @brief One renderable instance tracked independently from visible triangle submission. */
struct lighting_scene_instance
{
    std::uint64_t stable_id{};
    lighting_geometry_handle geometry{};
    material_handle material{};
    math::matrix4f model{math::identity<float, 4>()};
    geometric::box3f world_bounds{};
    std::uint64_t transform_revision{};
    std::uint64_t material_revision{};
    std::uint32_t geometry_generation{1};
    float card_density_bias{1.0f};
    float distance_field_resolution_bias{1.0f};
    bool static_object{};
    bool affects_indirect_lighting{true};
    bool visible_in_hardware_tracing{true};
};

/** @brief Incremental Lighting Scene mutation emitted after synchronization. */
enum class lighting_scene_update_kind : std::uint8_t
{
    reset,
    upsert,
    destroy
};

/** @brief One generation-checked Lighting Scene mutation. */
struct lighting_scene_update
{
    lighting_scene_update_kind kind{lighting_scene_update_kind::upsert};
    lighting_scene_instance_handle handle{};
    lighting_scene_instance instance{};
    bool transform_dirty{};
    bool material_dirty{};
    bool geometry_dirty{};
};

/** @brief Complete update batch for one world and frame. */
struct lighting_scene_update_batch
{
    std::uint64_t frame_index{};
    std::uint64_t world_id{};
    std::uint64_t world_epoch{};
    std::uint32_t active_instances{};
    std::vector<lighting_scene_update> updates;
    std::vector<geometric::box3f> dirty_world_regions;
};

/** @brief Runtime budgets for shared world-space lighting data. */
struct lighting_scene_config
{
    std::uint64_t gpu_budget_bytes{384ull * 1024ull * 1024ull};
    std::uint64_t compressed_cpu_budget_bytes{128ull * 1024ull * 1024ull};
    std::uint32_t maximum_surface_updates_per_frame{256};
    std::uint32_t maximum_radiance_probe_updates_per_frame{64};
    std::uint32_t protected_frame_count{30};
};

/** @brief Renderer and editor diagnostics for the current Lighting Scene generation. */
struct lighting_scene_snapshot
{
    std::uint64_t frame_index{};
    std::uint64_t world_id{};
    std::uint64_t world_epoch{};
    std::uint64_t cache_generation{};
    std::uint64_t gpu_budget_bytes{};
    std::uint64_t gpu_resident_bytes{};
    std::uint32_t active_instances{};
    std::uint32_t surface_cards{};
    std::uint32_t resident_surface_pages{};
    std::uint32_t resident_distance_field_pages{};
    std::uint32_t dirty_regions{};
    std::uint32_t surface_updates{};
    std::uint32_t radiance_probe_updates{};
    std::uint32_t evictions{};
};

/**
 * @brief CPU authority for stable Lighting Scene instances and dirty-region propagation.
 *
 * GPU caches and acceleration structures consume the returned update batch. No backend
 * object is stored or exposed by this class.
 */
class lighting_scene
{
public:
    explicit lighting_scene(lighting_scene_config config = {});
    ~lighting_scene();
    lighting_scene(lighting_scene&&) noexcept;
    lighting_scene& operator=(lighting_scene&&) noexcept;
    lighting_scene(const lighting_scene&) = delete;
    lighting_scene& operator=(const lighting_scene&) = delete;

    void configure(lighting_scene_config config);
    [[nodiscard]] lighting_scene_update_batch synchronize(std::uint64_t world_id, std::uint64_t world_epoch,
                                                          std::uint64_t frame_index,
                                                          std::span<const lighting_scene_instance> instances);
    void reset();
    [[nodiscard]] const lighting_scene_instance* find(lighting_scene_instance_handle handle) const noexcept;
    [[nodiscard]] lighting_scene_snapshot snapshot() const noexcept;
    /** @brief Publish backend/cache residency counters without exposing backend resources. */
    void update_residency_statistics(std::uint32_t surface_cards, std::uint32_t surface_pages,
                                     std::uint32_t distance_field_pages, std::uint64_t resident_bytes,
                                     std::uint32_t evictions = 0) noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

/** @brief Input ray for deterministic software-trace validation. */
struct lighting_trace_ray
{
    math::vector3f origin{};
    math::vector3f direction{0.0f, 0.0f, -1.0f};
    float minimum_distance{0.01f};
    float maximum_distance{100.0f};
};

/** @brief CPU reference hit produced from one cooked mesh distance field. */
struct [[nodiscard]] lighting_trace_result
{
    bool hit{};
    float distance{};
    math::vector3f position{};
    lighting_trace_source source{lighting_trace_source::none};
    std::uint32_t steps{};
};

/** @brief Sphere-trace a cooked local-space distance field for backend validation. */
[[nodiscard]] lighting_trace_result trace_mesh_distance_field(const mesh_distance_field_descriptor& field,
                                                              const lighting_trace_ray& ray,
                                                              std::uint32_t maximum_steps = 128);

} // namespace arc::render
