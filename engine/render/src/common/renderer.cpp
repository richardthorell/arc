#include <arc/render/renderer.h>

#include <arc/application.h>
#include <arc/log.h>

#include <utility>

namespace arc::render
{

void render_backend::resize_viewport(std::uint32_t, std::uint32_t)
{
}

render_viewport_texture render_backend::viewport_texture() const noexcept
{
    return {};
}

renderer::renderer(renderer_config config)
    : config_(config)
{
}

void renderer::set_backend(std::unique_ptr<render_backend> backend)
{
    backend_ = std::move(backend);
}

render_backend* renderer::backend() noexcept
{
    return backend_.get();
}

const renderer_config& renderer::config() const noexcept
{
    return config_;
}

render_frame_queue& renderer::frame_queue() noexcept
{
    return frame_queue_;
}

mesh_handle renderer::create_mesh(mesh_data mesh)
{
    const mesh_handle handle = mesh_handles_.allocate();
    auto shared_mesh = std::make_shared<mesh_data>(std::move(mesh));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.mesh_upload(handle, shared_mesh, shared_mesh->name);
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::mesh_alive(mesh_handle handle) const
{
    return mesh_handles_.alive(handle);
}

void renderer::resize_viewport(std::uint32_t width, std::uint32_t height)
{
    viewport_width_ = width;
    viewport_height_ = height;
    if (backend_)
        backend_->resize_viewport(width, height);
}

render_viewport_texture renderer::viewport_texture() const noexcept
{
    if (!backend_)
        return {};
    return backend_->viewport_texture();
}

render_submit_result renderer::render_frame(std::uint64_t frame_index, const render_graph& graph)
{
    auto packet = frame_queue_.commit(frame_index);
    const auto compiled = graph.compile();

    if (!backend_)
        return { .submitted = false, .message = "no render backend attached" };

    for (const auto& event : packet.events)
    {
        if (const auto* resize = std::get_if<viewport_resize_event>(&event.payload))
            resize_viewport(resize->width, resize->height);
    }

    return backend_->submit(packet, compiled);
}

renderer_module::renderer_module(renderer_config config)
    : renderer_(config)
    , graph_(make_clear_present_graph("viewport"))
{
}

renderer& renderer_module::service() noexcept
{
    return renderer_;
}

std::string_view renderer_module::name() const
{
    return "renderer";
}

void renderer_module::on_start(module_context&)
{
    arc::info("render", "Renderer module started");
}

void renderer_module::on_update(module_context&, const frame_time& time)
{
    if (!renderer_.backend())
    {
        if (!missing_backend_reported_)
        {
            arc::debug("render", "no render backend attached");
            missing_backend_reported_ = true;
        }
        return;
    }

    const auto result = renderer_.render_frame(time.frame_index, graph_);
    if (!result.submitted && !result.message.empty())
        arc::debug("render", result.message);
}

void renderer_module::on_shutdown(module_context&)
{
    arc::info("render", "Renderer module shutdown");
}

} // namespace arc::render
