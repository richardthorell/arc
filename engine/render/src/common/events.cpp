#include <arc/render/events.h>

#include <utility>

namespace arc::render
{

render_event_type render_event::type() const noexcept
{
    if (std::holds_alternative<mesh_upload_event>(payload))
        return render_event_type::mesh_upload;
    if (std::holds_alternative<mesh_destroy_event>(payload))
        return render_event_type::mesh_destroy;
    if (std::holds_alternative<virtual_mesh_upload_event>(payload))
        return render_event_type::virtual_mesh_upload;
    if (std::holds_alternative<texture_upload_event>(payload))
        return render_event_type::texture_upload;
    if (std::holds_alternative<material_upload_event>(payload))
        return render_event_type::material_upload;
    if (std::holds_alternative<environment_upload_event>(payload))
        return render_event_type::environment_upload;
    if (std::holds_alternative<environment_destroy_event>(payload))
        return render_event_type::environment_destroy;
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
    if (std::holds_alternative<area_light_event>(payload))
        return render_event_type::area_light;
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

void render_event_writer::mesh_destroy(mesh_handle handle)
{
    render_event event{};
    event.payload = mesh_destroy_event{ .handle = handle };
    buffer_->push(std::move(event));
}

void render_event_writer::virtual_mesh_upload(
    virtual_mesh_handle handle,
    std::shared_ptr<const virtual_mesh_data> mesh,
    std::string label)
{
    render_event event{};
    event.payload = virtual_mesh_upload_event{ .handle = handle, .mesh = std::move(mesh), .label = std::move(label) };
    buffer_->push(std::move(event));
}

void render_event_writer::texture_upload(texture_handle handle, std::shared_ptr<const texture_data> texture, std::string label)
{
    render_event event{};
    event.payload = texture_upload_event{ .handle = handle, .texture = std::move(texture), .label = std::move(label) };
    buffer_->push(std::move(event));
}

void render_event_writer::material_upload(material_handle handle, std::shared_ptr<const material_desc> material, std::string label)
{
    render_event event{};
    event.payload = material_upload_event{ .handle = handle, .material = std::move(material), .label = std::move(label) };
    buffer_->push(std::move(event));
}

void render_event_writer::environment_upload(
    environment_handle handle,
    std::shared_ptr<const environment_desc> environment,
    std::string label)
{
    render_event event{};
    event.payload = environment_upload_event{ .handle = handle, .environment = std::move(environment), .label = std::move(label) };
    buffer_->push(std::move(event));
}

void render_event_writer::environment_destroy(environment_handle handle)
{
    render_event event{};
    event.payload = environment_destroy_event{ .handle = handle };
    buffer_->push(std::move(event));
}

void render_event_writer::draw_mesh(
    mesh_handle mesh,
    material_handle material,
    const math::matrix4f& model,
    const math::matrix4f& view_projection,
    std::string label)
{
    draw_mesh_tinted(
        mesh,
        material,
        model,
        view_projection,
        render_mode::shaded,
        mesh_visualization_mode::standard,
        false,
        math::vector4f{ 1.0f, 1.0f, 1.0f, 1.0f },
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
    draw_mesh_tinted(
        mesh,
        material,
        model,
        view_projection,
        mode,
        visualization,
        selected,
        math::vector4f{ 1.0f, 1.0f, 1.0f, 1.0f },
        wire_color,
        std::move(label));
}

void render_event_writer::draw_mesh_tinted(
    mesh_handle mesh,
    material_handle material,
    const math::matrix4f& model,
    const math::matrix4f& view_projection,
    render_mode mode,
    mesh_visualization_mode visualization,
    bool selected,
    const math::vector4f& base_color_tint,
    const math::vector4f& wire_color,
    std::string label)
{
    render_event event{};
    event.payload = draw_mesh_event{
        .mesh = mesh,
        .material = material,
        .model = model,
        .previous_model = model,
        .view_projection = view_projection,
        .previous_view_projection = view_projection,
        .mode = mode,
        .visualization = visualization,
        .selected = selected,
        .base_color_tint = base_color_tint,
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
    std::string label,
    bool enabled,
    bool use_color_temperature,
    float temperature_kelvin,
    light_intensity_unit intensity_unit,
    texture_handle cookie_texture,
    shadow_settings shadow)
{
    render_event event{};
    event.payload = directional_light_event{
        .direction = direction,
        .color = color,
        .intensity = intensity,
        .casts_shadows = casts_shadows,
        .enabled = enabled,
        .use_color_temperature = use_color_temperature,
        .temperature_kelvin = temperature_kelvin,
        .intensity_unit = intensity_unit,
        .cookie_texture = cookie_texture,
        .shadow = shadow,
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
    std::string label,
    bool enabled,
    bool use_color_temperature,
    float temperature_kelvin,
    light_intensity_unit intensity_unit,
    texture_handle cookie_texture,
    shadow_settings shadow)
{
    render_event event{};
    event.payload = point_light_event{
        .position = position,
        .color = color,
        .intensity = intensity,
        .range = range,
        .casts_shadows = casts_shadows,
        .enabled = enabled,
        .use_color_temperature = use_color_temperature,
        .temperature_kelvin = temperature_kelvin,
        .intensity_unit = intensity_unit,
        .cookie_texture = cookie_texture,
        .shadow = shadow,
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
    std::string label,
    bool enabled,
    bool use_color_temperature,
    float temperature_kelvin,
    light_intensity_unit intensity_unit,
    texture_handle cookie_texture,
    shadow_settings shadow)
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
        .enabled = enabled,
        .use_color_temperature = use_color_temperature,
        .temperature_kelvin = temperature_kelvin,
        .intensity_unit = intensity_unit,
        .cookie_texture = cookie_texture,
        .shadow = shadow,
        .label = std::move(label)
    };
    buffer_->push(std::move(event));
}

void render_event_writer::area_light(
    const math::vector3f& position,
    const math::vector3f& direction,
    const math::vector3f& tangent,
    const math::vector3f& color,
    float intensity,
    float width,
    float height,
    area_light_shape shape,
    bool two_sided,
    bool casts_shadows,
    std::string label,
    bool enabled,
    bool use_color_temperature,
    float temperature_kelvin,
    light_intensity_unit intensity_unit,
    shadow_settings shadow)
{
    render_event event{};
    event.payload = area_light_event{
        .position = position,
        .direction = direction,
        .tangent = tangent,
        .color = color,
        .intensity = intensity,
        .width = width,
        .height = height,
        .shape = shape,
        .two_sided = two_sided,
        .casts_shadows = casts_shadows,
        .enabled = enabled,
        .use_color_temperature = use_color_temperature,
        .temperature_kelvin = temperature_kelvin,
        .intensity_unit = intensity_unit,
        .shadow = shadow,
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
