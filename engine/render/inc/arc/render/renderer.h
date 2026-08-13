#pragma once

#include <arc/framework/module.h>
#include <arc/render/events.h>
#include <arc/render/gpu_scene.h>
#include <arc/render/handles.h>
#include <arc/render/lighting.h>
#include <arc/render/material.h>
#include <arc/render/mesh.h>
#include <arc/render/render_backend.h>
#include <arc/render/render_graph.h>
#include <arc/render/virtual_mesh.h>
#include <arc/render/virtual_geometry.h>
#include <arc/render/terrain.h>

#include <memory>
#include <unordered_map>

namespace arc::render
{

/**
 * @brief Runtime renderer configuration.
 */
struct renderer_config
{
    render_backend_type preferred_backend{render_backend_type::vulkan};
    bool enable_validation{};
    render_quality_tier quality{render_quality_tier::auto_select};
    render_path path{render_path::auto_select};
    std::uint32_t adapter_index{resource_handle::invalid_index};
    /** Zero selects the target defined by the resolved quality profile. */
    float target_frame_time_ms{};
    bool enable_dynamic_resolution{true};
    bool force_disable_optional_features{};
    bool force_cpu_submission{};
    bool force_disable_gpu_driven{};
    bool force_disable_async_compute{};
    bool force_disable_temporal{};
    bool force_disable_dynamic_gi{};
    bool force_disable_hardware_ray_tracing{};
};

/**
 * @brief Resolve project rendering policy against immutable adapter support.
 */
resolved_render_config resolve_render_config(const renderer_config& config, const render_capabilities& capabilities);

/**
 * @brief Per-frame quality controls emitted by the frame-budget controller.
 */
struct frame_budget_settings
{
    float render_scale{1.0f};
    float geometry_error_threshold{1.0f};
    float shadow_resolution_scale{1.0f};
    float volumetric_resolution_scale{1.0f};
    std::uint32_t gi_trace_budget{1};
    std::uint32_t reflection_ray_budget{1};
    float lighting_trace_scale{0.5f};
    std::uint32_t surface_cache_update_budget{128};
    std::uint32_t radiance_probe_update_budget{32};
};

/** @brief Quality control most recently adjusted to meet the frame target. */
enum class frame_budget_change : std::uint8_t
{
    none,
    render_scale,
    geometry_error,
    shadow_resolution,
    gi_traces,
    reflection_rays,
    lighting_trace_resolution,
    surface_cache_updates,
    radiance_probe_updates,
    volumetric_resolution
};

/**
 * @brief Smoothed frame-time controller shared by all scalable render systems.
 */
class frame_budget_controller
{
public:
    void reset(const render_quality_profile& profile, float target_frame_time_ms) noexcept;
    const frame_budget_settings& update(float gpu_frame_time_ms) noexcept;
    const frame_budget_settings& settings() const noexcept;
    frame_budget_change last_change() const noexcept;
    float smoothed_frame_time_ms() const noexcept;

private:
    float target_frame_time_ms_{default_target_frame_time_ms};
    float minimum_scale_{low_render_quality_profile.minimum_render_scale};
    float maximum_scale_{1.0f};
    float minimum_geometry_error_{0.5f};
    float maximum_geometry_error_{4.0f};
    float minimum_shadow_scale_{0.5f};
    float maximum_shadow_scale_{1.0f};
    float minimum_volumetric_scale_{0.5f};
    float maximum_volumetric_scale_{1.0f};
    std::uint32_t maximum_gi_trace_budget_{1};
    std::uint32_t maximum_reflection_ray_budget_{1};
    float maximum_lighting_trace_scale_{0.5f};
    std::uint32_t maximum_surface_cache_update_budget_{128};
    std::uint32_t maximum_radiance_probe_update_budget_{32};
    frame_budget_settings settings_{};
    frame_budget_change last_change_{frame_budget_change::none};
    float smoothed_frame_time_ms_{default_target_frame_time_ms};
    std::uint32_t over_budget_frames_{};
    std::uint32_t under_budget_frames_{};
};

/**
 * @brief Backend-neutral renderer facade.
 */
class renderer
{
public:
    explicit renderer(renderer_config config = {});

    /**
     * @brief Attach a backend implementation.
     */
    void set_backend(std::unique_ptr<render_backend> backend);

    /**
     * @brief Return the active backend, if any.
     */
    render_backend* backend() noexcept;
    const render_backend* backend() const noexcept;

    /**
     * @brief Return the immutable renderer configuration.
     */
    const renderer_config& config() const noexcept;

    /**
     * @brief Return the concrete path and feature set selected for the backend.
     */
    const resolved_render_config& resolved_config() const noexcept;

    /**
     * @brief Return the queue used by producers to submit render events.
     */
    render_frame_queue& frame_queue() noexcept;

    /**
     * @brief Create a renderer-owned mesh resource and enqueue its upload.
     */
    [[nodiscard]] mesh_handle create_mesh(mesh_data mesh);

    /** @brief Replace mesh vertices while retaining its handle and topology. */
    bool update_mesh_vertices(mesh_handle handle, std::vector<mesh_vertex> vertices);

    /** @brief Retire a mesh handle and enqueue backend cleanup. */
    bool destroy_mesh(mesh_handle handle);

    /**
     * @brief Create a renderer-owned virtual mesh resource and enqueue its upload.
     */
    [[nodiscard]] virtual_mesh_handle create_virtual_mesh(virtual_mesh_data mesh);

    /** @brief Replace virtual-geometry metadata and pages while retaining its handle. */
    bool update_virtual_mesh(virtual_mesh_handle handle, virtual_mesh_data mesh);

    /** @brief Retire a virtual-geometry resource and all of its streamed pages. */
    bool destroy_virtual_mesh(virtual_mesh_handle handle);

    /** @brief Create a renderer-owned heightfield terrain resource. */
    [[nodiscard]] terrain_handle create_terrain(terrain_resource_descriptor terrain);
    /** @brief Replace terrain material and LOD configuration without changing its handle. */
    bool update_terrain(terrain_handle handle, material_handle material, terrain_lod_settings settings,
                        std::uint64_t content_revision);
    /** @brief Update a rectangular row-major height region and its hierarchy. */
    bool update_terrain_heights(terrain_handle handle, terrain_height_region_update update);
    /** @brief Update a rectangular row-major weight region. */
    bool update_terrain_weights(terrain_handle handle, terrain_weight_region_update update);
    /** @brief Retire a terrain resource. */
    bool destroy_terrain(terrain_handle handle);
    /** @brief Return whether a terrain handle is live. */
    [[nodiscard]] bool terrain_alive(terrain_handle handle) const noexcept;
    /** @brief Return retained terrain data used for view selection and tooling. */
    [[nodiscard]] const terrain_resource_descriptor* terrain_data_for(terrain_handle handle) const noexcept;
    /** @brief Return terrain resource diagnostics. */
    [[nodiscard]] terrain_resource_snapshot terrain_snapshot(terrain_handle handle) const noexcept;

    /** @brief Realize conventional LODs and virtual pages from one cooked geometry artifact. */
    [[nodiscard]] geometry_resource_handle create_geometry_resource(virtual_mesh_data geometry,
                                                                    std::uint32_t asset_generation = 1);

    /** @brief Retire every renderer resource owned by a unified geometry binding. */
    bool destroy_geometry_resource(const geometry_resource_handle& geometry);

    /**
     * @brief Create a renderer-owned texture resource and enqueue its upload.
     */
    [[nodiscard]] texture_handle create_texture(texture_data texture);

    /**
     * @brief Replace an existing renderer texture without changing its handle.
     */
    bool update_texture(texture_handle handle, texture_data texture);

    /**
     * @brief Create a renderer-owned material resource and enqueue its upload.
     */
    [[nodiscard]] material_handle create_material(material_descriptor material);

    /**
     * @brief Replace an existing renderer material description without changing its handle.
     */
    bool update_material(material_handle handle, material_descriptor material);

    /**
     * @brief Create a renderer-owned environment resource.
     */
    [[nodiscard]] environment_handle create_environment(environment_descriptor environment);

    /** @brief Replace an existing environment without changing its handle. */
    bool update_environment(environment_handle handle, environment_descriptor environment);

    /** @brief Retire an environment handle and enqueue backend cleanup. */
    bool destroy_environment(environment_handle handle);

    /**
     * @brief Return whether a mesh handle still references a live renderer mesh.
     */
    bool mesh_alive(mesh_handle handle) const;

    /**
     * @brief Return retained CPU mesh data for tooling and backend-neutral queries.
     */
    const mesh_data* mesh_data_for(mesh_handle handle) const;

    /** @brief Return the lighting representation generated for a static mesh. */
    [[nodiscard]] lighting_geometry_handle lighting_geometry_for(mesh_handle handle) const noexcept;

    /** @brief Return retained cards and SDF metadata for a lighting geometry handle. */
    [[nodiscard]] const lighting_geometry_descriptor*
    lighting_geometry_data_for(lighting_geometry_handle handle) const noexcept;

    /**
     * @brief Return whether a virtual mesh handle still references a live renderer virtual mesh.
     */
    bool virtual_mesh_alive(virtual_mesh_handle handle) const;

    /**
     * @brief Return CPU-side virtual mesh metadata needed for cluster extraction.
     */
    const virtual_mesh_data* virtual_mesh_data_for(virtual_mesh_handle handle) const;

    /** @brief Current content generation used to reject stale asynchronous page completions. */
    [[nodiscard]] std::uint32_t virtual_mesh_content_generation(virtual_mesh_handle handle) const noexcept;

    /** @return Renderer-owned virtual-geometry residency authority. */
    virtual_geometry_residency_manager& virtual_geometry_residency() noexcept;
    /** @return Read-only renderer-owned virtual-geometry residency authority. */
    const virtual_geometry_residency_manager& virtual_geometry_residency() const noexcept;

    /** @return Renderer-owned Lighting Scene authority shared by all views of a world. */
    lighting_scene& indirect_lighting_scene() noexcept;
    /** @return Read-only renderer-owned Lighting Scene authority. */
    const lighting_scene& indirect_lighting_scene() const noexcept;

    /**
     * @brief Return whether a texture handle still references a live renderer texture.
     */
    bool texture_alive(texture_handle handle) const;

    /**
     * @brief Return whether a material handle still references a live renderer material.
     */
    bool material_alive(material_handle handle) const;

    /**
     * @brief Return whether an environment handle still references a live renderer environment.
     */
    bool environment_alive(environment_handle handle) const;

    /**
     * @brief Resize the backend-owned viewport render target.
     */
    void resize_viewport(std::uint32_t width, std::uint32_t height);

    /**
     * @brief Return the current backend-owned viewport texture.
     */
    render_viewport_texture viewport_texture() const noexcept;

    /**
     * @brief Return the latest backend frame profile, if any.
     */
    render_backend_frame_profile last_frame_profile() const;

    /**
     * @brief Request an async ObjectID readback at viewport pixel coordinates.
     */
    void request_object_pick(std::uint64_t request_id, std::uint32_t x, std::uint32_t y);

    /**
     * @brief Return the latest async ObjectID readback result.
     */
    render_object_pick_result last_object_pick() const;

    void request_frame_capture(render_frame_capture_request request);
    render_frame_capture_result last_frame_capture() const;

    /**
     * @brief Build and submit one frame.
     */
    render_submit_result render_frame(std::uint64_t frame_index, const render_graph& graph);

private:
    struct temporal_view_state
    {
        math::matrix4f view_projection{math::identity<float, 4>()};
        math::vector3f position{};
        math::vector3f forward{0.0f, 0.0f, -1.0f};
        std::uint64_t world_epoch{};
        std::uint32_t width{};
        std::uint32_t height{};
        bool valid{};
    };

    renderer_config config_{};
    resolved_render_config resolved_config_{};
    std::unique_ptr<render_backend> backend_;
    render_frame_queue frame_queue_;
    gpu_scene gpu_scene_;
    virtual_geometry_residency_manager virtual_geometry_residency_;
    lighting_scene lighting_scene_;
    handle_pool mesh_handles_;
    handle_pool virtual_mesh_handles_;
    handle_pool terrain_handles_;
    handle_pool lighting_geometry_handles_;
    handle_pool texture_handles_;
    handle_pool material_handles_;
    handle_pool environment_handles_;
    std::uint32_t viewport_width_{};
    std::uint32_t viewport_height_{};
    frame_budget_controller frame_budget_;
    std::unordered_map<std::uint64_t, std::shared_ptr<const virtual_mesh_data>> virtual_mesh_data_;
    std::unordered_map<std::uint64_t, std::shared_ptr<terrain_resource_descriptor>> terrain_data_;
    std::unordered_map<std::uint64_t, terrain_resource_snapshot> terrain_snapshots_;
    std::unordered_map<std::uint64_t, terrain_selection_scratch> terrain_selection_scratch_;
    std::unordered_map<std::uint64_t, std::uint32_t> virtual_mesh_content_generations_;
    std::unordered_map<std::uint64_t, std::shared_ptr<const mesh_data>> mesh_data_;
    std::unordered_map<std::uint64_t, lighting_geometry_handle> mesh_lighting_geometry_;
    std::unordered_map<std::uint64_t, std::shared_ptr<const lighting_geometry_descriptor>> lighting_geometry_data_;
    std::unordered_map<std::uint64_t, temporal_view_state> temporal_views_;
};

/**
 * @brief Engine module that owns renderer lifecycle.
 */
class renderer_module final : public framework::module
{
public:
    explicit renderer_module(renderer_config config = {});

    /**
     * @brief Return the renderer service.
     */
    renderer& service() noexcept;

    std::string_view name() const override;
    void on_start(framework::module_context& context) override;
    void on_update(framework::module_context& context, const framework::frame_time& time) override;
    void on_shutdown(framework::module_context& context) override;

private:
    renderer renderer_;
    render_graph graph_;
    bool missing_backend_reported_{};
};

} // namespace arc::render
