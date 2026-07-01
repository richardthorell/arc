#pragma once

#include <arc/framework/module.h>
#include <arc/render/events.h>
#include <arc/render/handles.h>
#include <arc/render/lighting.h>
#include <arc/render/material.h>
#include <arc/render/mesh.h>
#include <arc/render/render_backend.h>
#include <arc/render/render_graph.h>

#include <memory>

namespace arc::render
{

/**
 * @brief Runtime renderer configuration.
 */
struct renderer_config
{
    render_backend_type preferred_backend{ render_backend_type::vulkan };
    bool enable_validation{};
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

    /**
     * @brief Return the immutable renderer configuration.
     */
    const renderer_config& config() const noexcept;

    /**
     * @brief Return the queue used by producers to submit render events.
     */
    render_frame_queue& frame_queue() noexcept;

    /**
     * @brief Create a renderer-owned mesh resource and enqueue its upload.
     */
    mesh_handle create_mesh(mesh_data mesh);

    /**
     * @brief Create a renderer-owned texture resource and enqueue its upload.
     */
    texture_handle create_texture(texture_data texture);

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

    /**
     * @brief Return whether a mesh handle still references a live renderer mesh.
     */
    bool mesh_alive(mesh_handle handle) const;

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
    std::unique_ptr<render_backend> backend_;
    render_frame_queue frame_queue_;
    handle_pool mesh_handles_;
    handle_pool texture_handles_;
    handle_pool material_handles_;
    handle_pool environment_handles_;
    std::uint32_t viewport_width_{};
    std::uint32_t viewport_height_{};
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
