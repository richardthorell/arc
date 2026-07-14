#pragma once

#include <arc/render/events.h>
#include <arc/render/render_graph.h>

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
    float target_frame_time_ms{ 16.6667f };
    float minimum_render_scale{ 0.67f };
    float maximum_render_scale{ 1.0f };
    float render_scale{ 1.0f };
    std::uint32_t max_point_lights{ 64 };
    std::uint32_t max_spot_lights{ 64 };
    std::uint32_t directional_shadow_cascades{ 4 };
    std::uint32_t directional_shadow_resolution{ 2048 };
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

/**
 * @brief Result from submitting a frame packet to a backend.
 */
struct render_submit_result
{
    bool submitted{};
    std::string message;
};

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
 * @brief Backend frame profile data exposed to tools such as the editor profiler.
 */
struct render_backend_frame_profile
{
    std::uint64_t frame_index{};
    std::vector<render_pass_timing> pass_timings;
    std::string summary;
    compiled_render_graph graph;
    clustered_light_grid_profile clustered_lights;
    resolved_render_config configuration;
};

/**
 * @brief One asynchronous editor ObjectID picking request.
 */
struct render_object_pick_request
{
    std::uint32_t x{};
    std::uint32_t y{};
};

/**
 * @brief Result from the latest asynchronous ObjectID pick readback.
 */
struct render_object_pick_result
{
    bool available{};
    bool hit{};
    render_object_id object{};
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint64_t frame_index{};
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
};

/**
 * @brief Factory result for backend creation.
 */
struct render_backend_create_result
{
    std::unique_ptr<render_backend> backend;
    std::string message;

    /**
     * @brief Return whether a backend was created.
     */
    bool succeeded() const noexcept
    {
        return static_cast<bool>(backend);
    }
};

} // namespace arc::render
