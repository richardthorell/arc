#pragma once

#include <arc/render/handles.h>
#include <arc/render/material.h>
#include <arc/geometric/box.h>
#include <arc/math/matrix.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace arc::render
{

struct render_camera;

/** @brief Terrain quadtree child sentinel. */
inline constexpr std::uint32_t invalid_terrain_node = 0xffffffffu;

/** @brief Terrain LOD policy authored independently from renderer quality. */
struct terrain_lod_settings
{
    /** Number of quads along one side of the reusable patch topology. */
    std::uint32_t patch_quads{32};
    /** Optional hierarchy depth limit; zero selects the complete hierarchy. */
    std::uint32_t maximum_hierarchy_depth{};
    /** Unitless multiplier applied to projected geometric error. */
    float geometric_error_multiplier{1.0f};
};

/** @brief Inclusive sample rectangle used by terrain uploads and hierarchy updates. */
struct terrain_sample_region
{
    std::uint32_t min_x{};
    std::uint32_t min_z{};
    std::uint32_t max_x{};
    std::uint32_t max_z{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return min_x <= max_x && min_z <= max_z;
    }
    [[nodiscard]] constexpr std::uint32_t width() const noexcept
    {
        return max_x - min_x + 1u;
    }
    [[nodiscard]] constexpr std::uint32_t height() const noexcept
    {
        return max_z - min_z + 1u;
    }
};

/** @brief One deterministic node in a flat terrain quadtree. */
struct terrain_hierarchy_node
{
    terrain_sample_region samples{};
    geometric::box3f local_bounds{};
    float minimum_height{};
    float maximum_height{};
    float geometric_error{};
    std::uint32_t depth{};
    std::array<std::uint32_t, 4> children{invalid_terrain_node, invalid_terrain_node, invalid_terrain_node,
                                          invalid_terrain_node};

    [[nodiscard]] constexpr bool leaf() const noexcept
    {
        return children[0] == invalid_terrain_node;
    }
};

/** @brief CPU hierarchy retained by a terrain render resource. */
struct terrain_hierarchy
{
    std::vector<terrain_hierarchy_node> nodes;
    std::uint32_t root{invalid_terrain_node};
    std::uint32_t leaf_count{};
    std::uint32_t maximum_depth{};
    std::uint32_t patch_quads{32};
};

/** @brief Complete immutable input used to create a terrain render resource. */
struct terrain_resource_descriptor
{
    std::uint32_t sample_resolution{};
    float width{1.0f};
    float depth{1.0f};
    std::vector<float> heights;
    std::vector<std::array<std::uint8_t, 4>> weights;
    terrain_hierarchy hierarchy;
    geometric::box3f local_bounds{};
    material_handle material{};
    terrain_lod_settings lod{};
    std::uint64_t content_revision{};
    std::string name;
};

/** @brief Partial row-major float-height update. */
struct terrain_height_region_update
{
    terrain_sample_region region{};
    std::uint32_t row_stride{};
    std::vector<float> values;
    std::uint64_t content_revision{};
};

/** @brief Partial row-major RGBA8 layer-weight update. */
struct terrain_weight_region_update
{
    terrain_sample_region region{};
    std::uint32_t row_stride{};
    std::vector<std::array<std::uint8_t, 4>> values;
    std::uint64_t content_revision{};
};

/** @brief One selected terrain patch consumed by backend draw generation. */
struct terrain_patch
{
    terrain_handle terrain{};
    std::uint32_t node_index{invalid_terrain_node};
    terrain_sample_region samples{};
    std::uint32_t lod{};
    std::uint8_t stitch_mask{};
    float projected_error{};
};

/** @brief Statistics from one view-dependent terrain traversal. */
struct terrain_selection_statistics
{
    std::uint32_t hierarchy_nodes{};
    std::uint32_t selected_patches{};
    std::uint32_t culled_nodes{};
    std::uint64_t rendered_triangles{};
    std::array<std::uint32_t, 16> patches_per_lod{};
};

/** @brief Result of selecting terrain patches for one view. */
struct [[nodiscard]] terrain_selection_result
{
    std::vector<terrain_patch> patches;
    terrain_selection_statistics statistics{};
};

/** @brief Read-only terrain resource state for diagnostics. */
struct terrain_resource_snapshot
{
    terrain_handle handle{};
    std::uint32_t sample_resolution{};
    std::uint32_t hierarchy_nodes{};
    std::uint32_t hierarchy_leaves{};
    geometric::box3f local_bounds{};
    std::uint64_t height_bytes{};
    std::uint64_t weight_bytes{};
    std::uint64_t uploaded_height_bytes{};
    std::uint64_t uploaded_weight_bytes{};
    std::uint64_t content_revision{};
    bool valid{};
};

/** @brief Reusable traversal scratch owned by a view and retained between frames. */
struct terrain_selection_scratch
{
    std::vector<std::uint32_t> traversal_stack;
    std::vector<std::uint32_t> previous_nodes;
};

/** @brief Std430-compatible terrain hierarchy node consumed by GPU traversal backends. */
struct alignas(16) gpu_terrain_node_record
{
    float bounds_min[4]{};
    float bounds_max[4]{};
    std::uint32_t samples[4]{};
    std::uint32_t children[4]{invalid_terrain_node, invalid_terrain_node, invalid_terrain_node,
                              invalid_terrain_node};
    float geometric_error{};
    std::uint32_t depth{};
    std::uint32_t leaf{};
    std::uint32_t reserved{};
};

/** @brief Immutable packed hierarchy plus its stable root index. */
struct terrain_gpu_hierarchy
{
    std::vector<gpu_terrain_node_record> nodes;
    std::uint32_t root{invalid_terrain_node};
    std::uint32_t leaf_count{};
    std::uint32_t maximum_depth{};
    std::uint32_t patch_quads{32};

    [[nodiscard]] bool valid() const noexcept
    {
        return root < nodes.size() && !nodes.empty();
    }
};

/** @brief Result of bounded terrain traversal used to validate GPU overflow fallback. */
struct bounded_terrain_selection
{
    terrain_selection_result selection;
    std::uint32_t capacity{};
    bool overflowed{};
    bool use_conventional_fallback{};
};

static_assert(sizeof(gpu_terrain_node_record) == 80);

/**
 * @brief Build the deterministic hierarchy for square row-major height data.
 * @param heights Authoritative row-major height samples.
 * @param sample_resolution Number of samples along both axes.
 * @param width Local-space terrain width in metres.
 * @param depth Local-space terrain depth in metres.
 * @param settings Authored hierarchy and patch policy.
 * @return Flat hierarchy, or an empty hierarchy when the input is invalid.
 */
[[nodiscard]] terrain_hierarchy build_terrain_hierarchy(std::span<const float> heights, std::uint32_t sample_resolution,
                                                        float width, float depth,
                                                        const terrain_lod_settings& settings = {});

/** @brief Pack a CPU terrain hierarchy into a deterministic Vulkan/Metal/D3D-neutral table. */
[[nodiscard]] terrain_gpu_hierarchy make_terrain_gpu_hierarchy(const terrain_hierarchy& hierarchy);

/**
 * @brief Incrementally recompute nodes affected by a height edit.
 * @return `true` when the dirty region and hierarchy were valid.
 */
bool update_terrain_hierarchy(terrain_hierarchy& hierarchy, std::span<const float> heights,
                              std::uint32_t sample_resolution, float width, float depth,
                              terrain_sample_region dirty_region, const terrain_lod_settings& settings = {});

/**
 * @brief Select and balance visible terrain patches using projected geometric error.
 * @param scratch Optional view-owned storage retaining traversal history; not shared between threads.
 */
[[nodiscard]] terrain_selection_result
select_terrain_patches(terrain_handle terrain, const terrain_hierarchy& hierarchy, const math::matrix4f& model,
                       const render_camera& camera, float geometry_error_threshold, float terrain_error_bias = 1.0f,
                       terrain_selection_scratch* scratch = nullptr);

/**
 * @brief Run the reference selector under a GPU-equivalent output capacity.
 *
 * Overflow deliberately discards the partial result and requests conventional
 * fallback for the whole terrain instance so the rendered surface remains hole-free.
 */
[[nodiscard]] bounded_terrain_selection
select_terrain_patches_bounded(terrain_handle terrain, const terrain_hierarchy& hierarchy,
                               const math::matrix4f& model, const render_camera& camera,
                               float geometry_error_threshold, std::uint32_t capacity,
                               float terrain_error_bias = 1.0f, terrain_selection_scratch* scratch = nullptr);

/**
 * @brief Build one shared grid index variant.
 * @param patch_quads Number of quads along each patch axis.
 * @param stitch_mask Mask bits identify left, right, top, and bottom coarse neighbors.
 * @return Triangle-list indices with degenerate edge triangles removed.
 */
[[nodiscard]] std::vector<std::uint32_t> make_terrain_patch_indices(std::uint32_t patch_quads,
                                                                    std::uint8_t stitch_mask);

} // namespace arc::render
