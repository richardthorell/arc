#pragma once

#include <arc/framework/module.h>
#include <arc/render/events.h>
#include <arc/render/handles.h>
#include <arc/render/lighting.h>
#include <arc/render/material.h>
#include <arc/render/mesh.h>
#include <arc/render/render_backend.h>
#include <arc/render/render_graph.h>
#include <arc/render/virtual_mesh.h>

#include <memory>
#include <unordered_map>

namespace arc::render
{

/**
 * @brief Runtime renderer configuration.
 */
struct renderer_config
{
    render_backend_type preferred_backend{ render_backend_type::vulkan };
    bool enable_validation{};
    render_quality_tier quality{ render_quality_tier::auto_select };
    render_path path{ render_path::auto_select };
    std::uint32_t adapter_index{ resource_handle::invalid_index };
    float target_frame_time_ms{ 16.6667f };
    bool enable_dynamic_resolution{ true };
    bool force_disable_optional_features{};
};

/**
 * @brief Resolve project rendering policy against immutable adapter support.
 */
resolved_render_config resolve_render_config(
    const renderer_config& config,
    const render_capabilities& capabilities);

/**
 * @brief Smoothed, quantized dynamic-resolution policy shared by render paths.
 */
class dynamic_resolution_controller
{
public:
    void reset(float target_frame_time_ms, float minimum_scale, float maximum_scale) noexcept;
    float update(float gpu_frame_time_ms) noexcept;
    float scale() const noexcept;

private:
    float target_frame_time_ms_{ 16.6667f };
    float minimum_scale_{ 0.5f };
    float maximum_scale_{ 1.0f };
    float scale_{ 1.0f };
    float smoothed_frame_time_ms_{ 16.6667f };
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
    mesh_handle create_mesh(mesh_data mesh);

    /**
     * @brief Create a renderer-owned virtual mesh resource and enqueue its upload.
     */
    virtual_mesh_handle create_virtual_mesh(virtual_mesh_data mesh);

    /**
     * @brief Create a renderer-owned texture resource and enqueue its upload.
     */
    texture_handle create_texture(texture_data texture);

    /**
     * @brief Replace an existing renderer texture without changing its handle.
     */
    bool update_texture(texture_handle handle, texture_data texture);

    /**
     * @brief Create a renderer-owned material resource and enqueue its upload.
     */
    material_handle create_material(material_desc material);

    /**
     * @brief Replace an existing renderer material description without changing its handle.
     */
    bool update_material(material_handle handle, material_desc material);

    /**
     * @brief Create a renderer-owned environment resource.
     */
    environment_handle create_environment(environment_desc environment);

    /** @brief Replace an existing environment without changing its handle. */
    bool update_environment(environment_handle handle, environment_desc environment);

    /** @brief Retire an environment handle and enqueue backend cleanup. */
    bool destroy_environment(environment_handle handle);

    /**
     * @brief Return whether a mesh handle still references a live renderer mesh.
     */
    bool mesh_alive(mesh_handle handle) const;

    /**
     * @brief Return whether a virtual mesh handle still references a live renderer virtual mesh.
     */
    bool virtual_mesh_alive(virtual_mesh_handle handle) const;

    /**
     * @brief Return CPU-side virtual mesh metadata needed for cluster extraction.
     */
    const virtual_mesh_data* virtual_mesh_data_for(virtual_mesh_handle handle) const;

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
    void request_object_pick(std::uint32_t x, std::uint32_t y);

    /**
     * @brief Return the latest async ObjectID readback result.
     */
    render_object_pick_result last_object_pick() const;

    /**
     * @brief Build and submit one frame.
     */
    render_submit_result render_frame(std::uint64_t frame_index, const render_graph& graph);

private:
    renderer_config config_{};
    resolved_render_config resolved_config_{};
    std::unique_ptr<render_backend> backend_;
    render_frame_queue frame_queue_;
    handle_pool mesh_handles_;
    handle_pool virtual_mesh_handles_;
    handle_pool texture_handles_;
    handle_pool material_handles_;
    handle_pool environment_handles_;
    std::uint32_t viewport_width_{};
    std::uint32_t viewport_height_{};
    dynamic_resolution_controller dynamic_resolution_;
    std::unordered_map<std::uint64_t, std::shared_ptr<const virtual_mesh_data>> virtual_mesh_data_;
};

/**
 * @brief Engine module that owns renderer lifecycle.
 */
class renderer_module final : public arc::module
{
public:
    explicit renderer_module(renderer_config config = {});

    /**
     * @brief Return the renderer service.
     */
    renderer& service() noexcept;

    std::string_view name() const override;
    void on_start(module_context& context) override;
    void on_update(module_context& context, const frame_time& time) override;
    void on_shutdown(module_context& context) override;

private:
    renderer renderer_;
    render_graph graph_;
    bool missing_backend_reported_{};
};

} // namespace arc::render
