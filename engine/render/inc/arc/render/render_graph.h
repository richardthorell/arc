#pragma once

#include <arc/core/result.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace arc::render
{

class command_encoder;
class render_pass_context;
using render_pass_record_fn = void (*)(render_pass_context& context);

/** @brief Allocation-free, graph-owned data copied with a pass declaration. */
struct render_pass_payload
{
    static constexpr std::size_t capacity = 64;

    /** @brief Copy a small trivially-copyable value into inline pass storage. */
    template <typename T> [[nodiscard]] static render_pass_payload from(const T& value) noexcept
    {
        static_assert(std::is_trivially_copyable_v<T>);
        static_assert(sizeof(T) <= capacity);
        render_pass_payload result;
        std::memcpy(result.storage.data(), &value, sizeof(T));
        result.size = static_cast<std::uint8_t>(sizeof(T));
        return result;
    }

    /** @brief Read a copied payload value, returning an empty value on a size mismatch. */
    template <typename T> [[nodiscard]] T get() const noexcept
    {
        static_assert(std::is_trivially_copyable_v<T>);
        T result{};
        if (size == sizeof(T)) std::memcpy(&result, storage.data(), sizeof(T));
        return result;
    }

    alignas(std::max_align_t) std::array<std::byte, capacity> storage{};
    std::uint8_t size{};
};

/**
 * @brief Queue class requested by a render graph pass.
 */
enum class render_queue_type : std::uint8_t
{
    graphics,
    compute,
    transfer
};

/**
 * @brief High-level pass behavior.
 */
enum class render_pass_kind : std::uint8_t
{
    clear,
    compute,
    depth_prepass,
    gbuffer,
    lighting,
    post_process,
    imgui,
    present,
    custom
};

/** @brief Stable engine pass identity implemented by every capable backend. */
enum class builtin_render_pass : std::uint8_t
{
    none,
    depth_prepass,
    gbuffer,
    deferred_lighting,
    forward_opaque,
    forward_transparent,
    presentation,
    gpu_scene_upload,
    gpu_visibility_clear,
    gpu_frustum_distance_cull,
    gpu_hzb_occlusion_cull,
    gpu_lod_selection,
    gpu_skinning,
    gpu_terrain_traversal,
    gpu_visibility_compact,
    gpu_draw_bin_count,
    gpu_draw_bin_prefix_sum,
    gpu_draw_bin_scatter,
    gpu_transparent_sort,
    gpu_indirect_command_generation,
    gpu_visibility_overflow,
    virtual_geometry_hierarchy_traversal,
    virtual_geometry_page_requests,
    virtual_geometry_cluster_binning,
    virtual_geometry_software_depth,
    virtual_geometry_visibility_resolve,
    virtual_geometry_material_resolve,
    virtual_geometry_mesh_shader_visibility,
    virtual_geometry_shadow_traversal,
    atmosphere_transmittance,
    atmosphere_multi_scattering,
    atmosphere_sky_view,
    cloud_shadow,
    sky_composite,
    environment_prefilter,
    environment_equirect_to_cube,
    environment_irradiance,
    environment_specular_prefilter,
    brdf_integration,
    luminance_histogram,
    exposure_resolve,
    subsurface_diffusion,
    transmission_refraction,
    directional_shadow_static,
    directional_shadow_dynamic,
    point_shadow,
    spot_shadow,
    depth_pyramid,
    screen_space_shadow,
    screen_space_shadow_filter,
    surface_card_capture,
    surface_cache_relight,
    distance_field_composition,
    screen_space_gi,
    software_gi_trace,
    hardware_gi_trace,
    radiance_cache_update,
    screen_space_reflections,
    software_reflections,
    hardware_reflections,
    indirect_lighting_temporal,
    reflection_temporal,
    lighting_variance,
    lighting_spatial_filter,
    indirect_lighting_composite,
    velocity_dilation,
    reactive_mask,
    disocclusion_mask,
    temporal_antialiasing,
    temporal_upscale,
    fxaa,
    spatial_sharpen,
    output_transform,
    debug_overlay,
    editor_overlay
};

/**
 * @brief High-level graph resource category.
 */
enum class render_resource_kind : std::uint8_t
{
    unknown,
    color_texture,
    depth_texture,
    buffer,
    swapchain_image
};

/** @brief Dimensionality of a graph texture resource. */
enum class render_texture_dimension : std::uint8_t
{
    texture_1d,
    texture_2d,
    texture_3d,
    texture_cube
};

/** @brief Intended allocation lifetime and ownership of a graph resource. */
enum class render_resource_lifetime_class : std::uint8_t
{
    transient,
    per_view,
    per_world,
    external
};

/** @brief Preferred memory placement for a graph resource. */
enum class render_memory_class : std::uint8_t
{
    device_local,
    upload,
    readback
};

/** @brief Backend-neutral pipeline stages participating in a resource access. */
enum class render_pipeline_stage : std::uint32_t
{
    none = 0,
    draw_indirect = 1u << 0u,
    vertex_input = 1u << 1u,
    vertex_shader = 1u << 2u,
    fragment_shader = 1u << 3u,
    early_depth = 1u << 4u,
    late_depth = 1u << 5u,
    color_output = 1u << 6u,
    compute_shader = 1u << 7u,
    transfer = 1u << 8u,
    host = 1u << 9u,
    present = 1u << 10u,
    all_graphics = (1u << 1u) | (1u << 2u) | (1u << 3u) | (1u << 4u) | (1u << 5u) | (1u << 6u),
    all_commands = 0xffffffffu
};

[[nodiscard]] constexpr render_pipeline_stage operator|(render_pipeline_stage lhs, render_pipeline_stage rhs) noexcept
{
    return static_cast<render_pipeline_stage>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

[[nodiscard]] constexpr render_pipeline_stage operator&(render_pipeline_stage lhs, render_pipeline_stage rhs) noexcept
{
    return static_cast<render_pipeline_stage>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

/** @brief Texture aspects addressed by a graph access. */
enum class render_texture_aspect : std::uint8_t
{
    automatic,
    color,
    depth,
    stencil,
    depth_stencil
};

/** @brief Mip/layer subset addressed by a graph access. */
struct render_subresource_range
{
    render_texture_aspect aspect{render_texture_aspect::automatic};
    std::uint32_t first_mip{};
    std::uint32_t mip_count{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t first_layer{};
    std::uint32_t layer_count{std::numeric_limits<std::uint32_t>::max()};
};

/**
 * @brief Backend-neutral formats used by render-graph resources.
 */
enum class render_format : std::uint8_t
{
    unknown,
    rgba8_unorm,
    rgba8_srgb,
    rgba16_float,
    rg16_float,
    rg32_float,
    r8_unorm,
    r32_uint,
    r32_float,
    d24_unorm_s8_uint,
    d32_float
};

/** @brief Which generation of a persistent history resource a pass accesses. */
enum class render_history_access : std::uint8_t
{
    current,
    previous
};

/** @brief Conditions that invalidate a persistent history resource. */
enum class render_history_reset : std::uint32_t
{
    none = 0,
    camera_cut = 1u << 0u,
    resize = 1u << 1u,
    render_scale_change = 1u << 2u,
    world_epoch_change = 1u << 3u,
    debug_view_change = 1u << 4u,
    projection_change = 1u << 5u
};

/** @brief Min/max device-depth pair stored in ARC's shared hierarchical depth pyramid. */
struct hzb_depth_range
{
    float nearest{1.0f};
    float farthest{1.0f};
};

/** @brief Device-depth convention used by shared hierarchical depth resources. */
enum class hzb_depth_convention : std::uint8_t
{
    /** Near depth is zero and far/clear depth is one. */
    conventional_zero_to_one
};

/** @brief Backend-neutral description of a per-view min/max depth pyramid. */
struct hzb_descriptor
{
    hzb_depth_convention convention{hzb_depth_convention::conventional_zero_to_one};
    render_format format{render_format::rg32_float};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t mip_count{};
};

/** @brief Runtime validity and generation state for a per-view HZB. */
struct hzb_snapshot
{
    hzb_descriptor descriptor{};
    std::uint64_t current_generation{};
    std::uint64_t previous_generation{};
    bool current_valid{};
    bool previous_valid{};
    std::string invalidation_reason;
};

/** @brief Return the complete mip count required for a depth pyramid of the supplied extent. */
[[nodiscard]] constexpr std::uint32_t hzb_mip_count(std::uint32_t width, std::uint32_t height) noexcept
{
    std::uint32_t largest = width > height ? width : height;
    std::uint32_t levels = largest == 0u ? 0u : 1u;
    while (largest > 1u)
    {
        largest /= 2u;
        ++levels;
    }
    return levels;
}

/** @brief Conservatively combine conventional-Z depth ranges for the next HZB mip. */
[[nodiscard]] constexpr hzb_depth_range reduce_hzb_depth(hzb_depth_range a, hzb_depth_range b) noexcept
{
    return {.nearest = a.nearest < b.nearest ? a.nearest : b.nearest,
            .farthest = a.farthest > b.farthest ? a.farthest : b.farthest};
}

[[nodiscard]] constexpr render_history_reset operator|(render_history_reset lhs, render_history_reset rhs) noexcept
{
    return static_cast<render_history_reset>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

[[nodiscard]] constexpr render_history_reset operator&(render_history_reset lhs, render_history_reset rhs) noexcept
{
    return static_cast<render_history_reset>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

/**
 * @brief Stable reference to one logical resource in a render graph.
 */
struct render_graph_resource_handle
{
    static constexpr std::uint32_t invalid_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t index{invalid_index};

    constexpr bool valid() const noexcept
    {
        return index != invalid_index;
    }
    friend constexpr bool operator==(render_graph_resource_handle, render_graph_resource_handle) noexcept = default;
};

/**
 * @brief How a graph resource extent is resolved for a view.
 */
enum class render_extent_mode : std::uint8_t
{
    absolute,
    relative_to_view,
    relative_to_output
};

/**
 * @brief Backend-neutral resource usage requested by a graph pass.
 */
enum class render_resource_usage : std::uint8_t
{
    unknown,
    color_attachment,
    depth_attachment,
    sampled,
    storage,
    transfer_src,
    transfer_dst,
    vertex_buffer,
    index_buffer,
    uniform_buffer,
    storage_buffer,
    indirect_buffer,
    present
};

/**
 * @brief Attachment load behavior requested by a graph pass.
 */
enum class render_load_op : std::uint8_t
{
    load,
    clear,
    dont_care
};

/**
 * @brief Attachment store behavior requested by a graph pass.
 */
enum class render_store_op : std::uint8_t
{
    store,
    dont_care
};

/**
 * @brief Texture dimensions used by typed graph resources.
 */
struct render_extent
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t depth{1};
};

/**
 * @brief Logical graph resource declaration.
 */
struct render_graph_resource
{
    std::string name;
    render_resource_kind kind{render_resource_kind::unknown};
    render_texture_dimension dimension{render_texture_dimension::texture_2d};
    render_extent extent{};
    render_extent_mode extent_mode{render_extent_mode::relative_to_view};
    float width_scale{1.0f};
    float height_scale{1.0f};
    render_format format{render_format::unknown};
    std::uint32_t mip_levels{1};
    std::uint32_t array_layers{1};
    std::uint32_t sample_count{1};
    std::uint64_t byte_size{};
    std::uint32_t element_stride{};
    render_memory_class memory{render_memory_class::device_local};
    render_resource_lifetime_class lifetime{render_resource_lifetime_class::transient};
    std::string persistent_key;
    std::uint8_t history_length{1};
    render_history_reset history_reset{render_history_reset::none};
    bool imported{};
    bool exported{};
    bool persistent{};
    bool allow_aliasing{true};
};

/**
 * @brief Logical resource access declared by a render graph pass.
 */
struct render_resource_access
{
    render_graph_resource_handle handle{};
    // Transitional label-based access for external graph producers. Compiled
    // accesses always contain both the strong handle and canonical name.
    std::string resource;
    render_resource_kind kind{render_resource_kind::unknown};
    render_resource_usage usage{render_resource_usage::unknown};
    render_pipeline_stage stages{render_pipeline_stage::none};
    render_subresource_range subresources{};
    bool write{};
    render_load_op load_op{render_load_op::load};
    render_store_op store_op{render_store_op::store};
    render_history_access history{render_history_access::current};
    float clear_color[4]{};
    float clear_depth{1.0f};
};

/**
 * @brief One pass declaration in the render graph.
 */
struct render_graph_pass
{
    std::string name;
    render_queue_type queue{render_queue_type::graphics};
    render_pass_kind kind{render_pass_kind::custom};
    builtin_render_pass builtin{builtin_render_pass::none};
    std::vector<render_resource_access> reads;
    std::vector<render_resource_access> writes;
    render_pass_record_fn record{};
    render_pass_payload payload{};
    bool side_effect{};
};

/**
 * @brief Compiled pass metadata.
 */
struct compiled_render_pass
{
    std::uint32_t source_index{};
    std::string name;
    render_queue_type queue{render_queue_type::graphics};
    render_pass_kind kind{render_pass_kind::custom};
    builtin_render_pass builtin{builtin_render_pass::none};
    std::vector<render_resource_access> reads;
    std::vector<render_resource_access> writes;
    render_pass_record_fn record{};
    render_pass_payload payload{};
};

/**
 * @brief Backend-neutral resource transition emitted by graph compilation.
 */
struct render_resource_transition
{
    render_graph_resource_handle handle{};
    std::string resource;
    render_resource_usage before{render_resource_usage::unknown};
    render_resource_usage after{render_resource_usage::unknown};
    render_history_access before_history{render_history_access::current};
    render_history_access after_history{render_history_access::current};
    render_pipeline_stage before_stages{render_pipeline_stage::none};
    render_pipeline_stage after_stages{render_pipeline_stage::none};
    render_subresource_range subresources{};
    bool release{};
    bool acquire{};
    std::uint32_t before_pass{};
    std::uint32_t after_pass{};
    render_queue_type before_queue{render_queue_type::graphics};
    render_queue_type after_queue{render_queue_type::graphics};
};

/** @brief Cross-queue timeline dependency for one compiled submission. */
struct render_queue_wait
{
    render_queue_type queue{render_queue_type::graphics};
    std::uint64_t value{};
};

/** @brief Executable batch of adjacent graph passes targeting one queue. */
struct compiled_queue_submission
{
    render_queue_type queue{render_queue_type::graphics};
    std::vector<std::uint32_t> passes;
    std::vector<render_queue_wait> waits;
    std::uint64_t signal_value{};
};

/** @brief Rotation operation performed after a persistent history generation completes. */
struct render_history_rotation
{
    render_graph_resource_handle handle{};
    std::string persistent_key;
    std::uint8_t history_length{1};
    render_history_reset reset{render_history_reset::none};
    bool invalidated{};
};

/**
 * @brief Lifetime and physical-allocation assignment for one logical resource.
 */
struct render_resource_lifetime
{
    render_graph_resource_handle handle{};
    std::uint32_t first_pass{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t last_pass{};
    std::uint32_t physical_resource{};
    std::uint64_t estimated_bytes{};
    bool aliased{};
};

/** @brief One physical allocation shared by compatible logical resources. */
struct render_physical_resource
{
    std::uint32_t index{};
    render_resource_kind kind{render_resource_kind::unknown};
    std::uint64_t estimated_bytes{};
    std::vector<render_graph_resource_handle> logical_resources;
};

/** @brief Reason a pass was removed from the executable plan. */
struct render_culled_pass
{
    std::uint32_t source_index{};
    std::string name;
};

/**
 * @brief Frame/view inputs used to specialize a reusable graph declaration.
 *
 * Relative resource extents and queue placement are resolved from this value
 * during compilation. The value is copied into the immutable execution plan.
 */
struct render_graph_compile_options
{
    std::uint64_t view_id{1};
    render_extent output_extent{};
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 14
#pragma GCC diagnostic push
// GCC 14 diagnoses this intentionally symmetric type/member spelling as
// -Wchanges-meaning. Keep the stable public field name without weakening the
// warning for any other declaration in this header.
#pragma GCC diagnostic ignored "-Wchanges-meaning"
#endif
    render_extent render_extent{};
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 14
#pragma GCC diagnostic pop
#endif
    std::uint64_t frame_index{};
    std::uint64_t world_epoch{};
    render_history_reset temporal_reset{render_history_reset::none};
    bool compute_queue_available{true};
    bool transfer_queue_available{true};
};

/** @brief Recoverable graph compilation error category. */
enum class render_graph_compile_error_code : std::uint8_t
{
    invalid_resource,
    invalid_access,
    read_before_write,
    dependency_cycle,
    invalid_history,
    invalid_attachment
};

/** @brief Structured graph compilation failure. */
struct render_graph_compile_error
{
    render_graph_compile_error_code code{render_graph_compile_error_code::invalid_resource};
    std::string message;
    std::string pass;
    std::string resource;
};

/**
 * @brief Result of render graph compilation.
 */
struct compiled_render_graph
{
    render_graph_compile_options view{};
    std::vector<compiled_render_pass> passes;
    std::vector<render_graph_resource> resources;
    std::vector<render_resource_transition> transitions;
    std::vector<render_resource_lifetime> lifetimes;
    std::vector<render_physical_resource> physical_resources;
    std::vector<compiled_queue_submission> submissions;
    std::vector<render_history_rotation> history_rotations;
    std::vector<render_culled_pass> culled_passes;
};

using render_graph_compile_result = core::result<compiled_render_graph, render_graph_compile_error>;

/**
 * @brief Minimal render graph that orders passes by declared resource dependencies.
 */
class render_graph
{
public:
    /**
     * @brief Declare a graph resource and return its index.
     */
    render_graph_resource_handle add_resource(render_graph_resource resource);

    /**
     * @brief Find a declared resource by name.
     */
    [[nodiscard]] const render_graph_resource* find_resource(std::string_view name) const noexcept;

    /**
     * @brief Return a declared resource by strong graph handle.
     */
    [[nodiscard]] const render_graph_resource* find_resource(render_graph_resource_handle handle) const noexcept;

    /**
     * @brief Add a pass declaration and return its index.
     */
    std::uint32_t add_pass(render_graph_pass pass);

    /**
     * @brief Compile pass order and validate dependencies.
     */
    [[nodiscard]] render_graph_compile_result compile(const render_graph_compile_options& options = {}) const;

    /**
     * @brief Remove all pass declarations.
     */
    void clear();

    /**
     * @brief Return declared passes.
     */
    [[nodiscard]] const std::vector<render_graph_pass>& passes() const noexcept;

    /**
     * @brief Return declared resources.
     */
    [[nodiscard]] const std::vector<render_graph_resource>& resources() const noexcept;

private:
    std::vector<render_graph_resource> resources_;
    std::vector<render_graph_pass> passes_;
};

/**
 * @brief Build the initial clear/present render graph for viewport bring-up.
 */
render_graph make_clear_present_graph(std::string_view target_name);

/**
 * @brief Return a stable display label for a typed format.
 */
[[nodiscard]] std::string_view render_format_name(render_format format) noexcept;

} // namespace arc::render
