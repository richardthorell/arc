#include <arc/render/events.h>

#include <utility>

namespace arc::render
{

render_event_type render_event::type() const noexcept
{
    if (std::holds_alternative<mesh_upload_event>(payload))
        return render_event_type::mesh_upload;
    if (std::holds_alternative<viewport_resize_event>(payload))
        return render_event_type::viewport_resize;
    if (std::holds_alternative<draw_mesh_event>(payload))
        return render_event_type::draw;
    if (std::holds_alternative<directional_light_event>(payload))
        return render_event_type::directional_light;
    if (std::holds_alternative<point_light_event>(payload))
        return render_event_type::point_light;
    if (std::holds_alternative<spot_light_event>(payload))
        return render_event_type::spot_light;
    if (std::holds_alternative<render_world_event>(payload))
        return render_event_type::render_world;
    return render_event_type::debug_marker;
}

void render_event_buffer::push(render_event event)
{
    events_.push_back(std::move(event));
}

void render_event_buffer::clear()
{
    events_.clear();
}

const std::vector<render_event>& render_event_buffer::events() const noexcept
{
    return events_;
}

bool render_event_buffer::empty() const noexcept
{
    return events_.empty();
}

render_event_writer::render_event_writer(render_event_buffer& buffer) noexcept
    : buffer_(&buffer)
{
}

void render_event_writer::push(render_event event)
{
    buffer_->push(std::move(event));
}

void render_event_writer::viewport_resize(std::uint32_t width, std::uint32_t height)
{
    render_event event{};
    event.payload = viewport_resize_event{ .width = width, .height = height };
    buffer_->push(std::move(event));
}

void render_event_writer::mesh_upload(mesh_handle handle, std::shared_ptr<const mesh_data> mesh, std::string label)
{
    render_event event{};
    event.payload = mesh_upload_event{ .handle = handle, .mesh = std::move(mesh), .label = std::move(label) };
    buffer_->push(std::move(event));
}

void render_event_writer::draw_mesh(
    mesh_handle mesh,
    material_handle material,
    const math::matrix4f& model,
    const math::matrix4f& view_projection,
    std::string label)
{
    draw_mesh(
        mesh,
        material,
        model,
        view_projection,
        render_mode::shaded,
        mesh_visualization_mode::standard,
        false,
        math::vector4f{ 0.25f, 0.65f, 1.0f, 1.0f },
        std::move(label));
}

void render_event_writer::draw_mesh(
    mesh_handle mesh,
    material_handle material,
    const math::matrix4f& model,
    const math::matrix4f& view_projection,
    render_mode mode,
    mesh_visualization_mode visualization,
    bool selected,
    const math::vector4f& wire_color,
    std::string label)
{
    render_event event{};
    event.payload = draw_mesh_event{
        .mesh = mesh,
        .material = material,
        .model = model,
        .view_projection = view_projection,
        .mode = mode,
        .visualization = visualization,
        .selected = selected,
        .wire_color = wire_color,
        .label = std::move(label)
    };
    buffer_->push(std::move(event));
}

void render_event_writer::directional_light(
    const math::vector3f& direction,
    const math::vector3f& color,
    float intensity,
    bool casts_shadows,
    std::string label)
{
    render_event event{};
    event.payload = directional_light_event{
        .direction = direction,
        .color = color,
        .intensity = intensity,
        .casts_shadows = casts_shadows,
        .label = std::move(label)
    };
    buffer_->push(std::move(event));
}

void render_event_writer::point_light(
    const math::vector3f& position,
    const math::vector3f& color,
    float intensity,
    float range,
    bool casts_shadows,
    std::string label)
{
    render_event event{};
    event.payload = point_light_event{
        .position = position,
        .color = color,
        .intensity = intensity,
        .range = range,
        .casts_shadows = casts_shadows,
        .label = std::move(label)
    };
    buffer_->push(std::move(event));
}

void render_event_writer::spot_light(
    const math::vector3f& position,
    const math::vector3f& direction,
    const math::vector3f& color,
    float intensity,
    float range,
    float inner_angle,
    float outer_angle,
    bool casts_shadows,
    std::string label)
{
    render_event event{};
    event.payload = spot_light_event{
        .position = position,
        .direction = direction,
        .color = color,
        .intensity = intensity,
        .range = range,
        .inner_angle = inner_angle,
        .outer_angle = outer_angle,
        .casts_shadows = casts_shadows,
        .label = std::move(label)
    };
    buffer_->push(std::move(event));
}

void render_event_writer::debug_marker(std::string label)
{
    render_event event{};
    event.payload = debug_marker_event{ .label = std::move(label) };
    buffer_->push(std::move(event));
}

void render_event_writer::render_world(std::shared_ptr<const render_world_packet> packet, std::string label)
{
    render_event event{};
    event.payload = render_world_event{ .packet = std::move(packet), .label = std::move(label) };
    buffer_->push(std::move(event));
}

void render_frame_queue::submit(render_event_buffer buffer)
{
    if (buffer.empty())
        return;

    std::lock_guard lock(mutex_);
    pending_.push_back(std::move(buffer));
}

render_frame_packet render_frame_queue::commit(std::uint64_t frame_index)
{
    std::vector<render_event_buffer> pending;
    {
        std::lock_guard lock(mutex_);
        pending.swap(pending_);
    }

    render_frame_packet packet{};
    packet.frame_index = frame_index;

    std::size_t event_count = 0;
    for (const auto& buffer : pending)
        event_count += buffer.events().size();
    packet.events.reserve(event_count);

    for (const auto& buffer : pending)
    {
        for (const auto& event : buffer.events())
            packet.events.push_back(event);
    }

    return packet;
}

std::size_t render_frame_queue::pending_buffer_count() const
{
    std::lock_guard lock(mutex_);
    return pending_.size();
}

} // namespace arc::render
