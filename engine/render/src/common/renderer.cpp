#include <arc/render/renderer.h>

#include <arc/framework/application.h>
#include <arc/diagnostics/log.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace arc::render
{
namespace
{

constexpr std::uint64_t gibibyte = 1024ull * 1024ull * 1024ull;

std::uint64_t renderer_resource_key(resource_handle handle) noexcept
{
    return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
}

} // namespace

resolved_render_config resolve_render_config(
    const renderer_config& config,
    const render_capabilities& capabilities)
{
    resolved_render_config result{};
    result.requested_quality = config.quality;
    result.requested_path = config.path;
    result.target_frame_time_ms = config.target_frame_time_ms > 0.0f
        ? config.target_frame_time_ms
        : 16.6667f;

    if (config.quality == render_quality_tier::auto_select)
    {
        const bool constrained_memory = capabilities.integrated_gpu ||
            (capabilities.dedicated_video_memory != 0 &&
                capabilities.dedicated_video_memory < 2ull * gibibyte);
        result.quality = constrained_memory ? render_quality_tier::low : render_quality_tier::medium;
        result.fallback_reasons.push_back(constrained_memory
            ? "auto-selected low quality for an integrated or memory-constrained adapter"
            : "auto-selected standard quality");
    }
    else if (config.quality == render_quality_tier::high)
    {
        result.quality = render_quality_tier::medium;
        result.fallback_reasons.push_back("high quality currently resolves to the standard renderer");
    }
    else
    {
        result.quality = config.quality;
    }

    if (config.path == render_path::auto_select)
        result.path = result.quality == render_quality_tier::low ? render_path::forward_plus : render_path::deferred;
    else
        result.path = config.path;

    if (result.quality == render_quality_tier::low)
    {
        result.minimum_render_scale = config.enable_dynamic_resolution ? 0.5f : 1.0f;
        result.max_point_lights = 32;
        result.max_spot_lights = 32;
        result.directional_shadow_cascades = 2;
        result.directional_shadow_resolution = 1024;
    }
    else
    {
        result.minimum_render_scale = config.enable_dynamic_resolution ? 0.67f : 1.0f;
    }

    result.maximum_render_scale = 1.0f;
    const bool optional_features = !config.force_disable_optional_features;
    result.features = {
        .dynamic_rendering = optional_features && capabilities.dynamic_rendering,
        .synchronization2 = optional_features && capabilities.synchronization2,
        .timeline_semaphores = optional_features && capabilities.timeline_semaphores,
        .descriptor_indexing = optional_features && capabilities.descriptor_indexing,
        .descriptor_buffer = optional_features && capabilities.descriptor_buffer,
        .draw_indirect = capabilities.draw_indirect,
        .draw_indirect_count = optional_features && capabilities.draw_indirect_count,
        .sampler_anisotropy = optional_features && capabilities.sampler_anisotropy,
        .texture_compression_bc = capabilities.texture_compression_bc,
        .mesh_shaders = optional_features && capabilities.mesh_shaders,
        .ray_tracing = optional_features && capabilities.ray_tracing,
        .variable_rate_shading = optional_features && capabilities.variable_rate_shading
    };

    if (config.force_disable_optional_features)
        result.fallback_reasons.push_back("optional GPU features were disabled by renderer configuration");
    if (!capabilities.dynamic_rendering)
        result.fallback_reasons.push_back("dynamic rendering is unavailable; use the compatibility render-pass path");
    if (!capabilities.synchronization2)
        result.fallback_reasons.push_back("synchronization2 is unavailable; use legacy barriers and submission");
    if (!capabilities.timeline_semaphores)
        result.fallback_reasons.push_back("timeline semaphores are unavailable; use per-frame fences");
    if (!capabilities.descriptor_indexing)
        result.fallback_reasons.push_back("descriptor indexing is unavailable; use classic descriptor sets");

    return result;
}

void dynamic_resolution_controller::reset(
    float target_frame_time_ms,
    float minimum_scale,
    float maximum_scale) noexcept
{
    target_frame_time_ms_ = std::max(1.0f, target_frame_time_ms);
    minimum_scale_ = std::clamp(minimum_scale, 0.25f, 1.0f);
    maximum_scale_ = std::clamp(maximum_scale, minimum_scale_, 1.0f);
    scale_ = maximum_scale_;
    smoothed_frame_time_ms_ = target_frame_time_ms_;
    over_budget_frames_ = 0;
    under_budget_frames_ = 0;
}

float dynamic_resolution_controller::update(float gpu_frame_time_ms) noexcept
{
    if (!(gpu_frame_time_ms > 0.0f) || !std::isfinite(gpu_frame_time_ms))
        return scale_;

    smoothed_frame_time_ms_ += (gpu_frame_time_ms - smoothed_frame_time_ms_) * 0.2f;
    if (smoothed_frame_time_ms_ > target_frame_time_ms_ * 1.04f)
    {
        ++over_budget_frames_;
        under_budget_frames_ = 0;
        if (over_budget_frames_ >= 3)
        {
            scale_ = std::max(minimum_scale_, scale_ - 1.0f / 16.0f);
            over_budget_frames_ = 0;
        }
    }
    else if (smoothed_frame_time_ms_ < target_frame_time_ms_ * 0.82f)
    {
        ++under_budget_frames_;
        over_budget_frames_ = 0;
        if (under_budget_frames_ >= 8)
        {
            scale_ = std::min(maximum_scale_, scale_ + 1.0f / 16.0f);
            under_budget_frames_ = 0;
        }
    }
    else
    {
        over_budget_frames_ = 0;
        under_budget_frames_ = 0;
    }
    return scale_;
}

float dynamic_resolution_controller::scale() const noexcept
{
    return scale_;
}

void render_backend::resize_viewport(std::uint32_t, std::uint32_t)
{
}

void render_backend::configure(const resolved_render_config&)
{
}

render_viewport_texture render_backend::viewport_texture() const noexcept
{
    return {};
}

render_backend_frame_profile render_backend::last_frame_profile() const
{
    return {};
}

void render_backend::request_object_pick(render_object_pick_request)
{
}

render_object_pick_result render_backend::last_object_pick() const
{
    return {};
}

void execute_render_graph(const compiled_render_graph& graph, command_encoder& encoder)
{
    for (std::uint32_t pass_index = 0; pass_index < graph.passes.size(); ++pass_index)
    {
        for (const auto& transition : graph.transitions)
        {
            if (transition.after_pass == pass_index)
                encoder.resource_barrier(transition);
        }

        const auto& pass = graph.passes[pass_index];
        encoder.begin_pass(pass);
        if (pass.record)
            pass.record(encoder, pass.user_data);
        encoder.end_pass();
    }
}

renderer::renderer(renderer_config config)
    : config_(config)
{
}

void renderer::set_backend(std::unique_ptr<render_backend> backend)
{
    backend_ = std::move(backend);
    if (backend_)
    {
        resolved_config_ = resolve_render_config(config_, backend_->capabilities());
        dynamic_resolution_.reset(
            resolved_config_.target_frame_time_ms,
            resolved_config_.minimum_render_scale,
            resolved_config_.maximum_render_scale);
        resolved_config_.render_scale = dynamic_resolution_.scale();
        backend_->configure(resolved_config_);
    }
}

render_backend* renderer::backend() noexcept
{
    return backend_.get();
}

const render_backend* renderer::backend() const noexcept
{
    return backend_.get();
}

const renderer_config& renderer::config() const noexcept
{
    return config_;
}

const resolved_render_config& renderer::resolved_config() const noexcept
{
    return resolved_config_;
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

virtual_mesh_handle renderer::create_virtual_mesh(virtual_mesh_data mesh)
{
    const virtual_mesh_handle handle = virtual_mesh_handles_.allocate();
    auto shared_mesh = std::make_shared<virtual_mesh_data>(std::move(mesh));
    virtual_mesh_data_[renderer_resource_key(handle)] = shared_mesh;

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.virtual_mesh_upload(handle, shared_mesh, "virtual mesh");
    frame_queue_.submit(std::move(buffer));
    return handle;
}

texture_handle renderer::create_texture(texture_data texture)
{
    const texture_handle handle = texture_handles_.allocate();
    auto shared_texture = std::make_shared<texture_data>(std::move(texture));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.texture_upload(handle, shared_texture, shared_texture->name);
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_texture(texture_handle handle, texture_data texture)
{
    if (!texture_handles_.alive(handle))
        return false;

    auto shared_texture = std::make_shared<texture_data>(std::move(texture));
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.texture_upload(handle, shared_texture, shared_texture->name);
    frame_queue_.submit(std::move(buffer));
    return true;
}

material_handle renderer::create_material(material_desc material)
{
    const material_handle handle = material_handles_.allocate();
    material.handle = handle;
    auto shared_material = std::make_shared<material_desc>(std::move(material));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.material_upload(handle, shared_material, shared_material->name);
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_material(material_handle handle, material_desc material)
{
    if (!material_handles_.alive(handle))
        return false;

    material.handle = handle;
    auto shared_material = std::make_shared<material_desc>(std::move(material));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.material_upload(handle, shared_material, shared_material->name);
    frame_queue_.submit(std::move(buffer));
    return true;
}

environment_handle renderer::create_environment(environment_desc environment)
{
    const environment_handle handle = environment_handles_.allocate();
    environment.handle = handle;
    auto shared_environment = std::make_shared<environment_desc>(std::move(environment));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.environment_upload(handle, shared_environment, shared_environment->name);
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_environment(environment_handle handle, environment_desc environment)
{
    if (!environment_handles_.alive(handle))
        return false;
    environment.handle = handle;
    auto shared_environment = std::make_shared<environment_desc>(std::move(environment));
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.environment_upload(handle, shared_environment, shared_environment->name);
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::destroy_environment(environment_handle handle)
{
    if (!environment_handles_.release(handle))
        return false;
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.environment_destroy(handle);
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::mesh_alive(mesh_handle handle) const
{
    return mesh_handles_.alive(handle);
}

bool renderer::virtual_mesh_alive(virtual_mesh_handle handle) const
{
    return virtual_mesh_handles_.alive(handle);
}

const virtual_mesh_data* renderer::virtual_mesh_data_for(virtual_mesh_handle handle) const
{
    if (!virtual_mesh_handles_.alive(handle))
        return nullptr;
    const auto found = virtual_mesh_data_.find(renderer_resource_key(handle));
    return found == virtual_mesh_data_.end() ? nullptr : found->second.get();
}

bool renderer::texture_alive(texture_handle handle) const
{
    return texture_handles_.alive(handle);
}

bool renderer::material_alive(material_handle handle) const
{
    return material_handles_.alive(handle);
}

bool renderer::environment_alive(environment_handle handle) const
{
    return environment_handles_.alive(handle);
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

render_backend_frame_profile renderer::last_frame_profile() const
{
    if (!backend_)
        return {};
    return backend_->last_frame_profile();
}

void renderer::request_object_pick(std::uint32_t x, std::uint32_t y)
{
    if (backend_)
        backend_->request_object_pick({ .x = x, .y = y });
}

render_object_pick_result renderer::last_object_pick() const
{
    if (!backend_)
        return {};
    return backend_->last_object_pick();
}

render_submit_result renderer::render_frame(std::uint64_t frame_index, const render_graph& graph)
{
    auto packet = frame_queue_.commit(frame_index);
    const auto compiled = graph.compile();

    if (!backend_)
        return { .submitted = false, .message = "no render backend attached" };

    if (config_.enable_dynamic_resolution)
    {
        float gpu_frame_time_ms{};
        for (const auto& timing : backend_->last_frame_profile().pass_timings)
            gpu_frame_time_ms += static_cast<float>(timing.milliseconds);
        if (gpu_frame_time_ms > 0.0f)
        {
            const float previous_scale = resolved_config_.render_scale;
            resolved_config_.render_scale = dynamic_resolution_.update(gpu_frame_time_ms);
            if (resolved_config_.render_scale != previous_scale)
                backend_->configure(resolved_config_);
        }
    }

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
