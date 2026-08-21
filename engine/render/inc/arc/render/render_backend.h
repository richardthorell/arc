#pragma once

#include <arc/core/core.h>
#include <arc/render/events.h>
#include <arc/render/render_graph.h>
#include <arc/render/lighting_scene.h>
#include <arc/render/virtual_mesh.h>

#include <array>
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
    high,
    ultra
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

/** @brief Anti-aliasing and temporal reconstruction policy requested for a rendered view. */
enum class anti_aliasing_method : std::uint8_t
{
    auto_select,
    disabled,
    fxaa,
    taa,
    taau
};

/** @brief Camera-level override of the project anti-aliasing policy. */
enum class camera_anti_aliasing_override : std::uint8_t
{
    inherit,
    disabled,
    fxaa,
    taa,
    taau
};

/** @brief Authored tuning shared by TAA and temporal upscaling. */
struct temporal_settings
{
    float history_weight{0.9f};
    float disocclusion_threshold{0.01f};
    float reactive_response{1.0f};
    float sharpening{0.2f};
    std::uint8_t jitter_sample_count{8};
};

inline constexpr float default_target_frame_time_ms = 1000.0f / 60.0f;
inline constexpr float dynamic_resolution_scale_step = 1.0f / 16.0f;
inline constexpr float dynamic_resolution_over_budget_ratio = 1.04f;
inline constexpr float dynamic_resolution_under_budget_ratio = 0.82f;
inline constexpr float dynamic_resolution_smoothing = 0.2f;
inline constexpr std::uint32_t dynamic_resolution_over_budget_frames = 3;
inline constexpr std::uint32_t dynamic_resolution_under_budget_frames = 8;

/** @brief Backend-neutral draw submission strategy selected for a view. */
enum class gpu_submission_path : std::uint8_t
{
    cpu_direct,
    indirect,
    indirect_count
};

/** @brief Immutable renderer limits associated with one implemented quality tier. */
struct render_quality_profile
{
    render_quality_tier quality{render_quality_tier::medium};
    render_path default_path{render_path::deferred};
    float minimum_render_scale{0.67f};
    float maximum_render_scale{1.0f};
    std::uint32_t max_point_lights{64};
    std::uint32_t max_spot_lights{64};
    std::uint32_t directional_shadow_cascades{4};
    std::uint32_t directional_shadow_resolution{2048};
    float directional_shadow_distance{200.0f};
    std::uint32_t local_shadow_atlas_resolution{4096};
    std::uint32_t max_shadowed_point_lights{4};
    std::uint32_t max_shadowed_spot_lights{8};
    std::uint32_t max_local_shadow_resolution{1024};
    bool screen_space_shadows{true};
    float screen_space_shadow_scale{0.5f};
    float target_frame_time_ms{default_target_frame_time_ms};
    float geometry_error_threshold{1.0f};
    float minimum_geometry_error_threshold{0.5f};
    float maximum_geometry_error_threshold{4.0f};
    float minimum_shadow_resolution_scale{0.5f};
    float maximum_shadow_resolution_scale{1.0f};
    float minimum_volumetric_resolution_scale{0.5f};
    float maximum_volumetric_resolution_scale{1.0f};
    std::uint32_t gi_trace_budget{1};
    std::uint32_t reflection_ray_budget{1};
    float lighting_trace_scale{0.5f};
    std::uint32_t surface_cache_update_budget{128};
    std::uint32_t radiance_probe_update_budget{32};
    gpu_submission_path preferred_submission{gpu_submission_path::indirect_count};
    bool prefer_gpu_driven{true};
    bool prefer_hzb_occlusion{true};
    bool prefer_temporal_upscaling{true};
    bool prefer_async_compute{true};
};

inline constexpr render_quality_profile low_render_quality_profile{.quality = render_quality_tier::low,
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
                                                                   .screen_space_shadow_scale = 0.0f,
                                                                   .target_frame_time_ms = default_target_frame_time_ms,
                                                                   .geometry_error_threshold = 2.0f,
                                                                   .minimum_geometry_error_threshold = 1.0f,
                                                                   .maximum_geometry_error_threshold = 6.0f,
                                                                   .minimum_shadow_resolution_scale = 0.5f,
                                                                   .minimum_volumetric_resolution_scale = 0.35f,
                                                                   .gi_trace_budget = 0,
                                                                   .reflection_ray_budget = 0,
                                                                   .lighting_trace_scale = 0.0f,
                                                                   .surface_cache_update_budget = 0,
                                                                   .radiance_probe_update_budget = 0,
                                                                   .preferred_submission =
                                                                       gpu_submission_path::indirect,
                                                                   .prefer_gpu_driven = false,
                                                                   .prefer_hzb_occlusion = false,
                                                                   .prefer_temporal_upscaling = false,
                                                                   .prefer_async_compute = false};

inline constexpr render_quality_profile standard_render_quality_profile{};

inline constexpr render_quality_profile high_render_quality_profile{.quality = render_quality_tier::high,
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
                                                                    .screen_space_shadow_scale = 1.0f,
                                                                    .target_frame_time_ms =
                                                                        default_target_frame_time_ms,
                                                                    .geometry_error_threshold = 0.75f,
                                                                    .minimum_geometry_error_threshold = 0.35f,
                                                                    .maximum_geometry_error_threshold = 3.0f,
                                                                    .minimum_shadow_resolution_scale = 0.67f,
                                                                    .minimum_volumetric_resolution_scale = 0.5f,
                                                                    .gi_trace_budget = 2,
                                                                    .reflection_ray_budget = 2,
                                                                    .lighting_trace_scale = 0.5f,
                                                                    .surface_cache_update_budget = 256,
                                                                    .radiance_probe_update_budget = 64};

inline constexpr render_quality_profile ultra_render_quality_profile{.quality = render_quality_tier::ultra,
                                                                     .default_path = render_path::deferred,
                                                                     .minimum_render_scale = 0.75f,
                                                                     .maximum_render_scale = 1.0f,
                                                                     .max_point_lights = 128,
                                                                     .max_spot_lights = 128,
                                                                     .directional_shadow_cascades = 4,
                                                                     .directional_shadow_resolution = 4096,
                                                                     .directional_shadow_distance = 400.0f,
                                                                     .local_shadow_atlas_resolution = 8192,
                                                                     .max_shadowed_point_lights = 12,
                                                                     .max_shadowed_spot_lights = 24,
                                                                     .max_local_shadow_resolution = 2048,
                                                                     .screen_space_shadows = true,
                                                                     .screen_space_shadow_scale = 1.0f,
                                                                     .target_frame_time_ms = 1000.0f / 30.0f,
                                                                     .geometry_error_threshold = 0.5f,
                                                                     .minimum_geometry_error_threshold = 0.25f,
                                                                     .maximum_geometry_error_threshold = 2.0f,
                                                                     .minimum_shadow_resolution_scale = 0.75f,
                                                                     .minimum_volumetric_resolution_scale = 0.67f,
                                                                     .gi_trace_budget = 4,
                                                                     .reflection_ray_budget = 4,
                                                                     .lighting_trace_scale = 1.0f,
                                                                     .surface_cache_update_budget = 512,
                                                                     .radiance_probe_update_budget = 128};

[[nodiscard]] constexpr const render_quality_profile& quality_profile(render_quality_tier quality) noexcept
{
    if (quality == render_quality_tier::low) return low_render_quality_profile;
    if (quality == render_quality_tier::high) return high_render_quality_profile;
    if (quality == render_quality_tier::ultra) return ultra_render_quality_profile;
    return standard_render_quality_profile;
}

/**
 * @brief Optional backend features exposed through capability queries.
 */
struct render_capabilities
{
    render_backend_type backend{render_backend_type::vulkan};
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
    bool dedicated_compute_queue{};
    bool presentation{};
    bool gpu_timestamps{};
    bool draw_indirect{};
    bool draw_indirect_count{};
    bool compute_shaders{};
    bool storage_buffers{};
    bool storage_images{};
    bool shader_draw_parameters{};
    /** @brief Backend has an executable GPU Scene visibility and indirect draw pipeline. */
    bool gpu_scene_indirect{};
    /** @brief GPU Scene bins can be submitted with a GPU-generated draw count. */
    bool gpu_scene_indirect_count{};
    /** @brief Backend can build and sample a cross-frame hierarchical depth buffer. */
    bool hzb_occlusion{};
    /** @brief Backend can execute ARC's temporal resolve pipeline. */
    bool temporal_resolve{};
    /** @brief Backend can resolve lower-resolution input into an output-resolution temporal history. */
    bool temporal_upscale{};
    /** @brief Backend can execute ARC's lightweight post-transform anti-aliasing path. */
    bool fxaa{};
    /** @brief Backend can execute ARC's compute traversal and software cluster rasterizer. */
    bool virtual_geometry_compute{};
    /** @brief Backend can execute cluster rasterization with mesh shaders. */
    bool virtual_geometry_mesh_shader{};
    /** @brief Backend can safely request, upload, and retire virtual-geometry pages. */
    bool virtual_geometry_streaming{};
    /** @brief Backend can execute ARC's HZB screen-space GI and reflection traces. */
    bool screen_space_indirect_lighting{};
    /** @brief Backend can capture and relight paged surface cards. */
    bool surface_cache{};
    /** @brief Backend can update and sample ARC's cascaded radiance cache. */
    bool radiance_cache{};
    bool software_ray_tracing{};
    /** @brief Backend has an executable inline hardware ray-query path. */
    bool hardware_ray_query{};
    bool sampler_anisotropy{};
    bool texture_compression_bc{};
    bool synchronization2{};
    bool timeline_semaphores{};
    bool dynamic_rendering{};
    bool descriptor_indexing{};
    bool descriptor_buffer{};
    bool mesh_shaders{};
    bool ray_tracing{};
    bool sparse_resources{};
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
    bool gpu_driven_rendering{};
    bool hzb_occlusion{};
    bool fxaa{};
    bool temporal_antialiasing{};
    bool temporal_upscaling{};
    anti_aliasing_method anti_aliasing{anti_aliasing_method::disabled};
    bool async_compute{};
    bool virtual_geometry{};
    virtual_geometry_raster_path virtual_geometry_path{virtual_geometry_raster_path::unavailable};
    bool software_ray_tracing{};
    bool hardware_ray_tracing{};
    bool screen_space_gi{};
    bool screen_space_reflections{};
    bool surface_cache{};
    bool radiance_cache{};
    bool software_gi{};
    bool software_reflections{};
    bool hardware_gi{};
    bool hardware_reflections{};
    bool sparse_resources{};
    bool sampler_anisotropy{};
    bool texture_compression_bc{};
    bool mesh_shaders{};
    bool ray_tracing{};
    bool variable_rate_shading{};
    gpu_submission_path submission{gpu_submission_path::cpu_direct};
};

/**
 * @brief Concrete settings selected after applying project policy to hardware.
 */
struct resolved_render_config
{
    render_quality_tier requested_quality{render_quality_tier::auto_select};
    render_quality_tier quality{render_quality_tier::medium};
    render_path requested_path{render_path::auto_select};
    render_path path{render_path::deferred};
    anti_aliasing_method requested_anti_aliasing{anti_aliasing_method::auto_select};
    anti_aliasing_method anti_aliasing{anti_aliasing_method::disabled};
    temporal_settings temporal{};
    render_feature_set features{};
    float target_frame_time_ms{default_target_frame_time_ms};
    float minimum_render_scale{standard_render_quality_profile.minimum_render_scale};
    float maximum_render_scale{standard_render_quality_profile.maximum_render_scale};
    float render_scale{1.0f};
    std::uint32_t max_point_lights{standard_render_quality_profile.max_point_lights};
    std::uint32_t max_spot_lights{standard_render_quality_profile.max_spot_lights};
    std::uint32_t directional_shadow_cascades{standard_render_quality_profile.directional_shadow_cascades};
    std::uint32_t directional_shadow_resolution{standard_render_quality_profile.directional_shadow_resolution};
    float directional_shadow_distance{standard_render_quality_profile.directional_shadow_distance};
    std::uint32_t local_shadow_atlas_resolution{standard_render_quality_profile.local_shadow_atlas_resolution};
    std::uint32_t max_shadowed_point_lights{standard_render_quality_profile.max_shadowed_point_lights};
    std::uint32_t max_shadowed_spot_lights{standard_render_quality_profile.max_shadowed_spot_lights};
    std::uint32_t max_local_shadow_resolution{standard_render_quality_profile.max_local_shadow_resolution};
    bool screen_space_shadows{standard_render_quality_profile.screen_space_shadows};
    float screen_space_shadow_scale{standard_render_quality_profile.screen_space_shadow_scale};
    float geometry_error_threshold{standard_render_quality_profile.geometry_error_threshold};
    float shadow_resolution_scale{1.0f};
    float volumetric_resolution_scale{1.0f};
    std::uint32_t gi_trace_budget{standard_render_quality_profile.gi_trace_budget};
    std::uint32_t reflection_ray_budget{standard_render_quality_profile.reflection_ray_budget};
    float lighting_trace_scale{standard_render_quality_profile.lighting_trace_scale};
    std::uint32_t surface_cache_update_budget{standard_render_quality_profile.surface_cache_update_budget};
    std::uint32_t radiance_probe_update_budget{standard_render_quality_profile.radiance_probe_update_budget};
    std::uint64_t lighting_scene_gpu_budget_bytes{};
    lighting_trace_path indirect_lighting_path{lighting_trace_path::baked_probe};
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
    virtual void begin_submission(const compiled_queue_submission&) {}
    virtual void end_submission(const compiled_queue_submission&) {}
    virtual void begin_pass(const compiled_render_pass& pass) = 0;
    virtual void end_pass() = 0;

    /** @brief Dispatch backend-neutral compute work for the active pass. */
    virtual void dispatch(std::uint32_t, std::uint32_t, std::uint32_t) {}

    /** @brief Draw a fixed-capacity indexed indirect command buffer. */
    virtual void draw_indexed_indirect(render_graph_resource_handle, std::uint64_t, std::uint32_t, std::uint32_t) {}

    /** @brief Draw an indexed indirect buffer using a GPU-generated count. */
    virtual void draw_indexed_indirect_count(render_graph_resource_handle, std::uint64_t, render_graph_resource_handle,
                                             std::uint64_t, std::uint32_t, std::uint32_t)
    {
    }
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
    render_submit_error_code code{render_submit_error_code::backend_failure};
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
    surface_frame_error_code code{surface_frame_error_code::backend_failure};
    std::string message;
};

/** @brief Result of presenting one backend-owned surface frame. */
using surface_frame_result = core::status<surface_frame_error>;

/** @brief Presentation transport selected for an editor viewport. */
enum class viewport_output_type : std::uint8_t
{
    /// Present through a platform window and swapchain.
    native_window,
    /// Present through an externally shareable GPU texture.
    shared_texture
};

/**
 * @brief Frame-local context supplied to a render-graph pass recorder.
 *
 * The context borrows the immutable compiled plan and active encoder. It is
 * valid only for the duration of the callback and must not be retained.
 */
class render_pass_context
{
public:
    render_pass_context(const compiled_render_graph& graph, std::uint32_t pass_index,
                        command_encoder& encoder) noexcept
        : graph_(&graph), pass_index_(pass_index), encoder_(&encoder)
    {
    }

    /** @brief Return the backend-neutral command encoder for this pass. */
    [[nodiscard]] command_encoder& encoder() const noexcept
    {
        return *encoder_;
    }

    /** @brief Return the active compiled pass. */
    [[nodiscard]] const compiled_render_pass& pass() const noexcept
    {
        return graph_->passes[pass_index_];
    }

    /** @brief Read the active pass's graph-owned inline payload. */
    template <typename T>
    [[nodiscard]] T payload() const noexcept
    {
        return pass().payload.template get<T>();
    }

    /** @brief Resolve a logical graph handle to its physical allocation index. */
    [[nodiscard]] std::uint32_t physical_resource(render_graph_resource_handle handle) const noexcept
    {
        return handle.valid() && handle.index < graph_->lifetimes.size()
                   ? graph_->lifetimes[handle.index].physical_resource
                   : render_graph_resource_handle::invalid_index;
    }

    /** @brief Return the immutable compiled graph owning this pass. */
    [[nodiscard]] const compiled_render_graph& graph() const noexcept
    {
        return *graph_;
    }

private:
    const compiled_render_graph* graph_{};
    std::uint32_t pass_index_{};
    command_encoder* encoder_{};
};

/** @brief Backend-neutral pixel formats supported by external viewport images. */
enum class viewport_pixel_format : std::uint8_t
{
    bgra8_unorm,
    rgba8_unorm,
    rgba16_float
};

/** @brief Platform handle representation carried by an exported GPU resource. */
enum class external_gpu_handle_type : std::uint8_t
{
    none,
    win32_nt_handle,
    posix_file_descriptor,
    io_surface
};

/**
 * @brief Opaque platform handle exported by a rendering backend.
 *
 * The numeric payload is meaningful only to the platform integration layer in
 * the process that owns it. Consumers must duplicate or transfer it explicitly
 * before crossing a process boundary.
 */
struct external_gpu_handle
{
    external_gpu_handle_type type{external_gpu_handle_type::none};
    std::uint64_t payload{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return type != external_gpu_handle_type::none && payload != 0;
    }
};

/** @brief Producer/consumer state of one shared viewport frame slot. */
enum class shared_viewport_frame_state : std::uint8_t
{
    available,
    rendering,
    ready,
    consumer_owned
};

/** @brief Synchronization guarantee attached to an exported viewport frame. */
struct shared_viewport_frame_sync
{
    /// The producer GPU fence completed before the frame was published.
    bool producer_complete{};
    /// Monotonic producer sequence associated with the completed submission.
    std::uint64_t value{};
};

/** @brief Description used to create or recreate one backend viewport output. */
struct viewport_output_descriptor
{
    std::string id{"viewport-1"};
    viewport_output_type type{viewport_output_type::native_window};
    std::uint32_t width{1};
    std::uint32_t height{1};
    bool visible{true};
};

/** @brief Immutable metadata for a shared GPU viewport frame. */
struct shared_viewport_frame
{
    std::string viewport_id;
    std::uint64_t frame_id{};
    std::uint64_t generation{};
    std::uint32_t width{};
    std::uint32_t height{};
    viewport_pixel_format format{viewport_pixel_format::bgra8_unorm};
    external_gpu_handle texture;
    shared_viewport_frame_sync synchronization;
};

/** @brief Result of polling a completed shared viewport frame. */
using shared_viewport_frame_result = core::result<std::optional<shared_viewport_frame>, surface_frame_error>;

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
    std::uint32_t tile_size_pixels{32};
    std::uint32_t tiles_x{};
    std::uint32_t tiles_y{};
    std::uint32_t depth_slices{16};
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

/** @brief Persistent GPU Scene and visibility work executed for one frame. */
struct render_gpu_scene_profile
{
    bool enabled{};
    bool hzb_occlusion{};
    bool history_valid{};
    gpu_submission_path submission{gpu_submission_path::cpu_direct};
    std::uint32_t capacity{};
    std::uint32_t active_instances{};
    std::uint32_t uploaded_instances{};
    std::uint32_t destroyed_instances{};
    std::uint64_t uploaded_bytes{};
    std::uint32_t frustum_rejected{};
    std::uint32_t distance_rejected{};
    std::uint32_t occlusion_rejected{};
    std::uint32_t visible_instances{};
    std::uint32_t indirect_commands{};
    std::string fallback_reason;
};

/** @brief Virtual-geometry traversal, raster, and residency work executed for one frame. */
struct render_virtual_geometry_profile
{
    bool enabled{};
    virtual_geometry_raster_path raster_path{virtual_geometry_raster_path::unavailable};
    std::uint32_t visible_clusters{};
    std::uint64_t visible_triangles{};
    std::uint32_t frustum_rejected{};
    std::uint32_t cone_rejected{};
    std::uint32_t hzb_rejected{};
    std::uint32_t projected_size_rejected{};
    std::uint32_t requested_pages{};
    std::uint32_t loaded_pages{};
    std::uint32_t failed_pages{};
    std::uint32_t parent_fallbacks{};
    std::uint64_t resident_bytes{};
    std::uint64_t residency_budget_bytes{};
    double decompression_milliseconds{};
    double upload_milliseconds{};
    std::string fallback_reason;
};

/** @brief Dynamic indirect-lighting, tracing, cache, and denoising work executed for one frame. */
struct render_indirect_lighting_profile
{
    bool enabled{};
    lighting_trace_path trace_path{lighting_trace_path::baked_probe};
    float trace_scale{};
    std::uint32_t gi_rays{};
    std::uint32_t reflection_rays{};
    std::uint32_t surface_cards{};
    std::uint32_t resident_surface_pages{};
    std::uint32_t resident_distance_field_pages{};
    std::uint32_t dirty_regions{};
    std::uint32_t surface_updates{};
    std::uint32_t radiance_probe_updates{};
    std::uint64_t resident_bytes{};
    std::uint64_t budget_bytes{};
    float screen_hit_rate{};
    float software_hit_rate{};
    float hardware_hit_rate{};
    std::string fallback_reason;
};

/** @brief Temporal history state used by TAA and temporal upscaling. */
struct render_temporal_profile
{
    bool enabled{};
    bool upscaling{};
    bool fxaa{};
    bool history_valid{};
    bool camera_cut{};
    anti_aliasing_method effective_method{anti_aliasing_method::disabled};
    std::uint32_t hzb_mip_count{};
    std::uint32_t rejected_history_samples{};
    std::uint32_t accepted_history_samples{};
    math::vector2f jitter{};
    std::string reset_reason;
    std::string fallback_reason;
};

/** @brief Per-frame terrain hierarchy, selection, residency, and upload telemetry. */
struct render_terrain_profile
{
    std::uint32_t hierarchy_nodes{};
    std::uint32_t selected_patches{};
    std::uint32_t culled_nodes{};
    std::uint64_t rendered_triangles{};
    std::uint64_t height_bytes{};
    std::uint64_t weight_bytes{};
    std::uint64_t uploaded_height_bytes{};
    std::uint64_t uploaded_weight_bytes{};
    std::array<std::uint32_t, 16> patches_per_lod{};
    double selection_milliseconds{};
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
    render_gpu_scene_profile gpu_scene;
    render_virtual_geometry_profile virtual_geometry;
    render_terrain_profile terrain;
    render_indirect_lighting_profile indirect_lighting;
    render_temporal_profile temporal;
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
    emissive,
    indirect_diffuse,
    reflections,
    trace_source,
    mesh_distance_field,
    temporal_confidence
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
    std::vector<render_capture_channel> channels{render_capture_channel::output_color};
};

struct render_capture_image
{
    render_capture_channel channel{render_capture_channel::output_color};
    render_capture_format format{render_capture_format::rgba8_unorm};
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
    math::matrix4f view_projection{math::identity<float, 4>()};
    math::matrix4f inverse_view_projection{math::identity<float, 4>()};
    math::matrix4f projection{math::identity<float, 4>()};
    math::vector3f position{};
    math::vector3f forward{0.0f, 0.0f, -1.0f};
    math::vector3f up{0.0f, 1.0f, 0.0f};
    float near_plane{0.01f};
    float far_plane{1000.0f};
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
    [[nodiscard]] bool valid() const noexcept
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
    [[nodiscard]] virtual render_backend_type type() const noexcept = 0;

    /**
     * @brief Return optional feature support.
     */
    [[nodiscard]] virtual const render_capabilities& capabilities() const noexcept = 0;

    /**
     * @brief Apply the renderer's resolved feature and quality policy.
     */
    virtual void configure(const resolved_render_config& config);

    /**
     * @brief Submit one immutable frame packet and compiled graph.
     */
    [[nodiscard]] virtual render_submit_result submit(const render_frame_packet& packet,
                                                      const compiled_render_graph& graph) = 0;

    /**
     * @brief Present the latest submitted frame to the backend-owned surface.
     * @param width Output width
     * in physical pixels.
     * @param height Output height in physical pixels.
     * @return Success, or a
     * structured recoverable presentation error.
     * @note Must be called from the backend's render thread.
     */
    [[nodiscard]] virtual surface_frame_result present_surface_frame(std::uint32_t width, std::uint32_t height);

    /** @brief Create a viewport presentation output. */
    [[nodiscard]] virtual surface_frame_result create_viewport_output(const viewport_output_descriptor& descriptor);

    /** @brief Resize an existing viewport output and advance its generation. */
    [[nodiscard]] virtual surface_frame_result resize_viewport_output(std::string_view viewport_id,
                                                                      std::uint32_t width, std::uint32_t height);

    /** @brief Submit the latest rendered frame to a named viewport output. */
    [[nodiscard]] virtual surface_frame_result present_viewport_output(std::string_view viewport_id);

    /** @brief Poll one producer-complete frame without blocking the CPU. */
    [[nodiscard]] virtual shared_viewport_frame_result poll_viewport_output(std::string_view viewport_id);

    /** @brief Return a consumer-owned frame slot to the producer. */
    virtual void release_viewport_frame(std::string_view viewport_id, std::uint64_t generation,
                                        std::uint64_t frame_id);

    /** @brief Change whether a viewport should produce frames. */
    virtual void set_viewport_output_visible(std::string_view viewport_id, bool visible);

    /** @brief Destroy a viewport output after consumer-owned frames are released. */
    virtual void destroy_viewport_output(std::string_view viewport_id);

    /**
     * @brief Resize the backend-owned editor/game viewport target.
     */
    virtual void resize_viewport(std::uint32_t width, std::uint32_t height);

    /**
     * @brief Return an opaque texture identifier for editor display.
     */
    [[nodiscard]] virtual render_viewport_texture viewport_texture() const noexcept;

    /**
     * @brief Return the most recent backend frame profile.
     */
    [[nodiscard]] virtual render_backend_frame_profile last_frame_profile() const;

    /**
     * @brief Request an async ObjectID readback at viewport pixel coordinates.
     */
    virtual void request_object_pick(render_object_pick_request request);

    /**
     * @brief Return the latest async ObjectID readback result.
     */
    [[nodiscard]] virtual render_object_pick_result last_object_pick() const;

    /**
     * @brief Queue an asynchronous capture of coherent channels from one rendered frame.
     */
    virtual void request_frame_capture(render_frame_capture_request request);

    /**
     * @brief Return the latest completed frame capture.
     */
    [[nodiscard]] virtual render_frame_capture_result last_frame_capture() const;
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
    render_backend_create_error_code code{render_backend_create_error_code::device_creation_failed};
    std::string message;
};

/** @brief Value-or-error result returned by a render-backend factory. */
using render_backend_create_result = core::result<std::unique_ptr<render_backend>, render_backend_create_error>;

} // namespace arc::render
