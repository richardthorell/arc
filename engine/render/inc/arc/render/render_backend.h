#pragma once

#include <arc/core/core.h>
#include <arc/render/events.h>
#include <arc/render/render_graph.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace arc::render
{

/**
 * @brief Render API backend family.
 */
enum class render_backend_type : std::uint8_t
{
    vulkan,
    d3d12,
    metal
};

/**
 * @brief Renderer quality policy selected by a project or resolved from hardware.
 */
enum class render_quality_tier : std::uint8_t
{
    auto_select,
    low,
    medium,
    high
};

/**
 * @brief Backend-neutral raster path used for a view.
 */
enum class render_path : std::uint8_t
{
    auto_select,
    forward_plus,
    deferred
};

inline constexpr float default_target_frame_time_ms = 1000.0f / 60.0f;
inline constexpr float dynamic_resolution_scale_step = 1.0f / 16.0f;
inline constexpr float dynamic_resolution_over_budget_ratio = 1.04f;
inline constexpr float dynamic_resolution_under_budget_ratio = 0.82f;
inline constexpr float dynamic_resolution_smoothing = 0.2f;
inline constexpr std::uint32_t dynamic_resolution_over_budget_frames = 3;
inline constexpr std::uint32_t dynamic_resolution_under_budget_frames = 8;

/** @brief Immutable renderer limits associated with one implemented quality tier. */
struct render_quality_profile
{
    render_quality_tier quality{ render_quality_tier::medium };
    render_path default_path{ render_path::deferred };
    float minimum_render_scale{ 0.67f };
    float maximum_render_scale{ 1.0f };
    std::uint32_t max_point_lights{ 64 };
    std::uint32_t max_spot_lights{ 64 };
    std::uint32_t directional_shadow_cascades{ 4 };
    std::uint32_t directional_shadow_resolution{ 2048 };
    float directional_shadow_distance{ 200.0f };
    std::uint32_t local_shadow_atlas_resolution{ 4096 };
    std::uint32_t max_shadowed_point_lights{ 4 };
    std::uint32_t max_shadowed_spot_lights{ 8 };
    std::uint32_t max_local_shadow_resolution{ 1024 };
    bool screen_space_shadows{ true };
    float screen_space_shadow_scale{ 0.5f };
};

inline constexpr render_quality_profile low_render_quality_profile{
    .quality = render_quality_tier::low,
    .default_path = render_path::forward_plus,
    .minimum_render_scale = 0.5f,
    .maximum_render_scale = 1.0f,
    .max_point_lights = 32,
    .max_spot_lights = 32,
    .directional_shadow_cascades = 2,
    .directional_shadow_resolution = 1024,
    .directional_shadow_distance = 80.0f,
    .local_shadow_atlas_resolution = 2048,
    .max_shadowed_point_lights = 0,
    .max_shadowed_spot_lights = 2,
    .max_local_shadow_resolution = 512,
    .screen_space_shadows = false,
    .screen_space_shadow_scale = 0.0f
};

inline constexpr render_quality_profile standard_render_quality_profile{};

inline constexpr render_quality_profile high_render_quality_profile{
    .quality = render_quality_tier::high,
    .default_path = render_path::deferred,
    .minimum_render_scale = 0.67f,
    .maximum_render_scale = 1.0f,
    .max_point_lights = 64,
    .max_spot_lights = 64,
    .directional_shadow_cascades = 4,
    .directional_shadow_resolution = 4096,
    .directional_shadow_distance = 300.0f,
    .local_shadow_atlas_resolution = 8192,
    .max_shadowed_point_lights = 8,
    .max_shadowed_spot_lights = 16,
    .max_local_shadow_resolution = 2048,
    .screen_space_shadows = true,
    .screen_space_shadow_scale = 1.0f
};

[[nodiscard]] constexpr const render_quality_profile& quality_profile(render_quality_tier quality) noexcept
{
    if (quality == render_quality_tier::low)
        return low_render_quality_profile;
    if (quality == render_quality_tier::high)
        return high_render_quality_profile;
    return standard_render_quality_profile;
}

/**
 * @brief Optional backend features exposed through capability queries.
 */
struct render_capabilities
{
    render_backend_type backend{ render_backend_type::vulkan };
    std::uint32_t api_major{};
    std::uint32_t api_minor{};
    std::string adapter_name;
    std::string driver_name;
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint64_t driver_version{};
    bool discrete_gpu{};
    bool integrated_gpu{};
    std::uint64_t dedicated_video_memory{};
    std::uint64_t shared_system_memory{};
    std::uint64_t memory_budget{};
    std::uint64_t memory_usage{};
    std::uint32_t max_texture_dimension_2d{};
    std::uint32_t max_color_attachments{};
    std::uint32_t max_compute_workgroup_invocations{};
    bool graphics_queue{};
    bool compute_queue{};
    bool transfer_queue{};
    bool presentation{};
    bool gpu_timestamps{};
    bool draw_indirect{};
    bool draw_indirect_count{};
    bool sampler_anisotropy{};
    bool texture_compression_bc{};
    bool synchronization2{};
    bool timeline_semaphores{};
    bool dynamic_rendering{};
    bool descriptor_indexing{};
    bool descriptor_buffer{};
    bool mesh_shaders{};
    bool ray_tracing{};
    bool variable_rate_shading{};
    bool fill_mode_non_solid{};
};

/**
 * @brief Optional features enabled for the active renderer path.
 *
 * Capabilities describe immutable adapter facts. This structure describes the
 * subset the renderer deliberately enabled and may therefore use.
 */
struct render_feature_set
{
    bool dynamic_rendering{};
    bool synchronization2{};
    bool timeline_semaphores{};
    bool descriptor_indexing{};
    bool descriptor_buffer{};
    bool draw_indirect{};
    bool draw_indirect_count{};
    bool sampler_anisotropy{};
    bool texture_compression_bc{};
    bool mesh_shaders{};
    bool ray_tracing{};
    bool variable_rate_shading{};
};

/**
 * @brief Concrete settings selected after applying project policy to hardware.
 */
struct resolved_render_config
{
    render_quality_tier requested_quality{ render_quality_tier::auto_select };
    render_quality_tier quality{ render_quality_tier::medium };
    render_path requested_path{ render_path::auto_select };
    render_path path{ render_path::deferred };
    render_feature_set features{};
    float target_frame_time_ms{ default_target_frame_time_ms };
    float minimum_render_scale{ standard_render_quality_profile.minimum_render_scale };
    float maximum_render_scale{ standard_render_quality_profile.maximum_render_scale };
    float render_scale{ 1.0f };
    std::uint32_t max_point_lights{ standard_render_quality_profile.max_point_lights };
    std::uint32_t max_spot_lights{ standard_render_quality_profile.max_spot_lights };
    std::uint32_t directional_shadow_cascades{ standard_render_quality_profile.directional_shadow_cascades };
    std::uint32_t directional_shadow_resolution{ standard_render_quality_profile.directional_shadow_resolution };
    float directional_shadow_distance{ standard_render_quality_profile.directional_shadow_distance };
    std::uint32_t local_shadow_atlas_resolution{ standard_render_quality_profile.local_shadow_atlas_resolution };
    std::uint32_t max_shadowed_point_lights{ standard_render_quality_profile.max_shadowed_point_lights };
    std::uint32_t max_shadowed_spot_lights{ standard_render_quality_profile.max_shadowed_spot_lights };
    std::uint32_t max_local_shadow_resolution{ standard_render_quality_profile.max_local_shadow_resolution };
    bool screen_space_shadows{ standard_render_quality_profile.screen_space_shadows };
    float screen_space_shadow_scale{ standard_render_quality_profile.screen_space_shadow_scale };
    std::vector<std::string> fallback_reasons;
};

/**
 * @brief Abstract renderer device.
 */
class render_device
{
public:
    virtual ~render_device() = default;
};

/**
 * @brief Abstract swapchain owned by a renderer backend.
 */
class render_swapchain
{
public:
    virtual ~render_swapchain() = default;
};

/**
 * @brief Abstract GPU queue.
 */
class render_queue
{
public:
    virtual ~render_queue() = default;
};

/**
 * @brief Abstract command encoder.
 */
class command_encoder
{
public:
    virtual ~command_encoder() = default;

    virtual void resource_barrier(const render_resource_transition& transition) = 0;
    virtual void begin_pass(const compiled_render_pass& pass) = 0;
    virtual void end_pass() = 0;
};

/**
 * @brief Execute a compiled plan through a backend-neutral command encoder.
 */
void execute_render_graph(const compiled_render_graph& graph, command_encoder& encoder);

/**
 * @brief Abstract resource allocator.
 */
class resource_allocator
{
public:
    virtual ~resource_allocator() = default;
};

/**
 * @brief Abstract pipeline cache.
 */
class pipeline_cache
{
public:
    virtual ~pipeline_cache() = default;
};

/**
 * @brief Abstract shader library.
 */
class shader_library
{
public:
    virtual ~shader_library() = default;
};

/** @brief Recoverable frame-submission failure categories. */
enum class render_submit_error_code : std::uint8_t
{
    /// No backend is attached or capable of accepting the frame.
    backend_unavailable,
    /// The compiled graph does not satisfy backend execution requirements.
    invalid_render_graph,
    /// The graphics device was lost during submission.
    device_lost,
    /// An unspecified backend operation failed.
    backend_failure
};

/** @brief Recoverable error returned when a backend cannot submit a frame. */
struct render_submit_error
{
    render_submit_error_code code{ render_submit_error_code::backend_failure };
    std::string message;
};

/** @brief Status returned after submitting a frame packet to a backend. */
using render_submit_result = core::status<render_submit_error>;

/**
 * @brief Failure categories for presenting a backend-owned surface.
 */
enum class surface_frame_error_code : std::uint8_t
{
    unsupported,
    unavailable,
    out_of_date,
    device_lost,
    backend_failure
};

/**
 * @brief Recoverable failure returned by surface presentation.
 */
struct surface_frame_error
{
    surface_frame_error_code code{ surface_frame_error_code::backend_failure };
    std::string message;
};

/** @brief Result of presenting one backend-owned surface frame. */
using surface_frame_result = core::status<surface_frame_error>;

/**
 * @brief One named GPU/backend timing sample in milliseconds.
 */
struct render_pass_timing
{
    std::string name;
    double milliseconds{};
};

/**
 * @brief Lightweight clustered-light culling summary for editor diagnostics.
 */
struct clustered_light_grid_profile
{
    std::uint32_t tile_size_pixels{ 32 };
    std::uint32_t tiles_x{};
    std::uint32_t tiles_y{};
    std::uint32_t depth_slices{ 16 };
    std::uint32_t cluster_count{};
    std::uint32_t point_light_references{};
    std::uint32_t spot_light_references{};
    std::uint32_t overflow_count{};
    bool available{};
};

/**
 * @brief Resolved world-environment state reported by the active backend.
 *
 * The graph can describe a higher-quality path before a backend implements it,
 * so these fields intentionally report what the backend actually executed.
 */
struct render_environment_profile
{
    bool enabled{};
    bool sky_visible{};
    bool affects_lighting{};
    std::string source;
    std::string quality_path;
    std::string atmosphere_lut_state;
    std::string environment_lighting_state;
    std::uint32_t cloud_shadow_resolution{};
    std::string fallback_reason;
};

/** @brief Actual shadow allocation/cache state executed for one frame. */
struct render_shadow_profile
{
    std::uint32_t directional_cascade_count{};
    std::uint32_t directional_resolution{};
    std::uint32_t local_atlas_resolution{};
    std::uint32_t local_allocation_count{};
    std::uint32_t local_occupied_texels{};
    std::uint32_t local_eviction_count{};
    std::uint32_t local_resolution_reductions{};
    std::uint32_t shadowed_point_lights{};
    std::uint32_t shadowed_spot_lights{};
    std::uint32_t static_caster_count{};
    std::uint32_t dynamic_caster_count{};
    std::uint32_t local_cache_hits{};
    std::uint32_t local_cache_misses{};
    bool static_cache_hit{};
    bool screen_space_shadows{};
    std::string fallback_reason;
};

/**
 * @brief Backend frame profile data exposed to tools such as the editor profiler.
 */
struct render_backend_frame_profile
{
    std::uint64_t frame_index{};
    std::vector<render_pass_timing> pass_timings;
    std::string summary;
    compiled_render_graph graph;
    clustered_light_grid_profile clustered_lights;
    render_environment_profile environment;
    render_shadow_profile shadows;
    resolved_render_config configuration;
};

/**
 * @brief One asynchronous editor ObjectID picking request.
 */
struct render_object_pick_request
{
    std::uint64_t request_id{};
    std::uint32_t x{};
    std::uint32_t y{};
};

/**
 * @brief Result from the latest asynchronous ObjectID pick readback.
 */
struct [[nodiscard]] render_object_pick_result
{
    std::uint64_t request_id{};
    bool available{};
    bool hit{};
    render_object_id object{};
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint64_t frame_index{};
};

enum class render_capture_channel : std::uint8_t
{
    output_color,
    scene_color,
    linear_depth,
    object_id,
    world_normal,
    base_color,
    material_properties,
    emissive
};

enum class render_capture_format : std::uint8_t
{
    rgba8_unorm,
    bgra8_unorm,
    rgba16_float,
    r32_float,
    r32_uint
};

struct render_frame_capture_request
{
    std::uint64_t capture_id{};
    std::vector<render_capture_channel> channels{ render_capture_channel::output_color };
};

struct render_capture_image
{
    render_capture_channel channel{ render_capture_channel::output_color };
    render_capture_format format{ render_capture_format::rgba8_unorm };
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::byte> data;
};

struct render_capture_object
{
    std::uint32_t encoded_id{};
    render_object_id object{};
};

struct render_capture_camera_state
{
    math::matrix4f view_projection{ math::identity<float, 4>() };
    math::matrix4f inverse_view_projection{ math::identity<float, 4>() };
    math::matrix4f projection{ math::identity<float, 4>() };
    math::vector3f position{};
    math::vector3f forward{ 0.0f, 0.0f, -1.0f };
    math::vector3f up{ 0.0f, 1.0f, 0.0f };
    float near_plane{ 0.01f };
    float far_plane{ 1000.0f };
    std::uint32_t render_width{};
    std::uint32_t render_height{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
};

struct [[nodiscard]] render_frame_capture_result
{
    std::uint64_t capture_id{};
    std::uint64_t frame_index{};
    bool available{};
    bool succeeded{};
    render_capture_camera_state camera{};
    std::vector<render_capture_image> images;
    std::vector<render_capture_object> objects;
    std::vector<std::string> diagnostics;
};

/**
 * @brief Opaque UI-facing texture exported by a backend.
 */
struct render_viewport_texture
{
    std::uint64_t id{};
    std::uint32_t width{};
    std::uint32_t height{};

    /**
     * @brief Return whether this texture can be shown by an editor UI.
     */
    bool valid() const noexcept
    {
        return id != 0 && width > 0 && height > 0;
    }
};

/**
 * @brief Backend-neutral render API implementation.
 */
class render_backend
{
public:
    virtual ~render_backend() = default;

    /**
     * @brief Return the backend family.
     */
    virtual render_backend_type type() const noexcept = 0;

    /**
     * @brief Return optional feature support.
     */
    virtual const render_capabilities& capabilities() const noexcept = 0;

    /**
     * @brief Apply the renderer's resolved feature and quality policy.
     */
    virtual void configure(const resolved_render_config& config);

    /**
     * @brief Submit one immutable frame packet and compiled graph.
     */
    virtual render_submit_result submit(const render_frame_packet& packet, const compiled_render_graph& graph) = 0;

    /**
     * @brief Present the latest submitted frame to the backend-owned surface.
     * @param width Output width in physical pixels.
     * @param height Output height in physical pixels.
     * @return Success, or a structured recoverable presentation error.
     * @note Must be called from the backend's render thread.
     */
    [[nodiscard]] virtual surface_frame_result present_surface_frame(
        std::uint32_t width,
        std::uint32_t height);

    /**
     * @brief Resize the backend-owned editor/game viewport target.
     */
    virtual void resize_viewport(std::uint32_t width, std::uint32_t height);

    /**
     * @brief Return an opaque texture identifier for editor display.
     */
    virtual render_viewport_texture viewport_texture() const noexcept;

    /**
     * @brief Return the most recent backend frame profile.
     */
    virtual render_backend_frame_profile last_frame_profile() const;

    /**
     * @brief Request an async ObjectID readback at viewport pixel coordinates.
     */
    virtual void request_object_pick(render_object_pick_request request);

    /**
     * @brief Return the latest async ObjectID readback result.
     */
    virtual render_object_pick_result last_object_pick() const;

    /**
     * @brief Queue an asynchronous capture of coherent channels from one rendered frame.
     */
    virtual void request_frame_capture(render_frame_capture_request request);

    /**
     * @brief Return the latest completed frame capture.
     */
    virtual render_frame_capture_result last_frame_capture() const;
};

/** @brief Render-backend creation failure categories. */
enum class render_backend_create_error_code : std::uint8_t
{
    /// The platform graphics loader could not be initialized.
    loader_unavailable,
    /// The backend API instance could not be created.
    instance_creation_failed,
    /// The requested presentation surface could not be created.
    surface_creation_failed,
    /// No adapter met the backend's required baseline.
    adapter_unavailable,
    /// Logical device creation failed.
    device_creation_failed,
    /// Backend GPU-memory allocator creation failed.
    memory_allocator_creation_failed
};

/** @brief Structured failure returned by a render-backend factory. */
struct render_backend_create_error
{
    render_backend_create_error_code code{
        render_backend_create_error_code::device_creation_failed
    };
    std::string message;
};

/** @brief Value-or-error result returned by a render-backend factory. */
using render_backend_create_result =
    core::result<std::unique_ptr<render_backend>, render_backend_create_error>;

} // namespace arc::render
