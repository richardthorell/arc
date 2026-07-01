#pragma once

#include <arc/render/handles.h>
#include <arc/render/material.h>
#include <arc/render/mesh.h>
#include <arc/math/matrix.h>
#include <arc/math/vector.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace arc::render
{

enum class light_intensity_unit : std::uint8_t;
struct environment_desc;
struct render_world_packet;

/**
 * @brief Kinds of renderer events produced by game/editor threads.
 */
enum class render_event_type : std::uint8_t
{
    mesh_upload,
    texture_upload,
    material_upload,
    environment_upload,
    viewport_resize,
    draw,
    directional_light,
    point_light,
    spot_light,
    render_world,
    debug_marker
};

/**
 * @brief Mesh rendering mode requested by scene/editor extraction.
 */
enum class render_mode : std::uint8_t
{
    shaded,
    wireframe
};

/**
 * @brief Mesh visualization path used by debug viewport render modes.
 */
enum class mesh_visualization_mode : std::uint8_t
{
    standard,
    albedo,
    opacity,
    world_normal,
    specularity,
    gloss,
    metalness,
    ao,
    emission,
    lighting,
    uv0,
    cascade_debug,
    shadow_mask,
    light_complexity
};

/**
 * @brief Editor overlay policy layered on top of the primary render mode.
 */
enum class editor_overlay_mode : std::uint8_t
{
    none,
    selected_wireframe,
    all_wireframe
};

/**
 * @brief Shadow filtering requested by a light.
 */
enum class shadow_filter : std::uint8_t
{
    none,
    pcf_3x3,
    pcf_5x5,
    pcss
};

/**
 * @brief Per-light shadow authoring settings.
 */
struct shadow_settings
{
    bool enabled{ true };
    std::uint32_t resolution{ 2048 };
    float bias{ 0.0015f };
    float normal_bias{ 0.01f };
    float strength{ 0.75f };
    shadow_filter filter{ shadow_filter::pcf_3x3 };
};

/**
 * @brief Upload a static mesh into backend-owned GPU resources.
 */
struct mesh_upload_event
{
    mesh_handle handle{};
    std::shared_ptr<const mesh_data> mesh;
    std::string label;
};

/**
 * @brief Upload a texture into backend-owned GPU resources.
 */
struct texture_upload_event
{
    texture_handle handle{};
    std::shared_ptr<const texture_data> texture;
    std::string label;
};

/**
 * @brief Upload or replace a renderer material description.
 */
struct material_upload_event
{
    material_handle handle{};
    std::shared_ptr<const material_desc> material;
    std::string label;
};

/**
 * @brief Upload or replace an environment description.
 */
struct environment_upload_event
{
    environment_handle handle{};
    std::shared_ptr<const environment_desc> environment;
    std::string label;
};

/**
 * @brief Resize the backend-owned viewport render target.
 */
struct viewport_resize_event
{
    std::uint32_t width{};
    std::uint32_t height{};
};

/**
 * @brief Draw a static mesh with one model and camera matrix.
 */
struct draw_mesh_event
{
    mesh_handle mesh{};
    material_handle material{};
    math::matrix4f model{ math::identity<float, 4>() };
    math::matrix4f view_projection{ math::identity<float, 4>() };
    render_mode mode{ render_mode::shaded };
    mesh_visualization_mode visualization{ mesh_visualization_mode::standard };
    render_object_id object_id{};
    bool selected{};
    math::vector4f base_color_tint{ 1.0f, 1.0f, 1.0f, 1.0f };
    math::vector4f wire_color{ 0.25f, 0.65f, 1.0f, 1.0f };
    std::string label;
};

/**
 * @brief Submit one directional light to the renderer.
 */
struct directional_light_event
{
    math::vector3f direction{ 0.0f, -1.0f, 0.0f };
    math::vector3f color{ 1.0f, 1.0f, 1.0f };
    float intensity{ 1.0f };
    bool casts_shadows{};
    bool enabled{ true };
    bool use_color_temperature{};
    float temperature_kelvin{ 6500.0f };
    light_intensity_unit intensity_unit{};
    texture_handle cookie_texture{};
    shadow_settings shadow{};
    std::string label;
};

/**
 * @brief Submit one point light to the renderer.
 */
struct point_light_event
{
    math::vector3f position{};
    math::vector3f color{ 1.0f, 1.0f, 1.0f };
    float intensity{ 1.0f };
    float range{ 10.0f };
    bool casts_shadows{};
    bool enabled{ true };
    bool use_color_temperature{};
    float temperature_kelvin{ 6500.0f };
    light_intensity_unit intensity_unit{};
    texture_handle cookie_texture{};
    shadow_settings shadow{ .enabled = false };
    std::string label;
};

/**
 * @brief Submit one spot light to the renderer.
 */
struct spot_light_event
{
    math::vector3f position{};
    math::vector3f direction{ 0.0f, -1.0f, 0.0f };
    math::vector3f color{ 1.0f, 1.0f, 1.0f };
    float intensity{ 1.0f };
    float range{ 10.0f };
    float inner_angle{ 0.35f };
    float outer_angle{ 0.75f };
    bool casts_shadows{};
    bool enabled{ true };
    bool use_color_temperature{};
    float temperature_kelvin{ 6500.0f };
    light_intensity_unit intensity_unit{};
    texture_handle cookie_texture{};
    shadow_settings shadow{ .enabled = false };
    std::string label;
};

/**
 * @brief Insert a renderer debug marker.
 */
struct debug_marker_event
{
    std::string label;
};

/**
 * @brief Submit a prepared scene render packet to the backend.
 */
struct render_world_event
{
    std::shared_ptr<const render_world_packet> packet;
    std::string label;
};

using render_event_payload = std::variant<
    mesh_upload_event,
    texture_upload_event,
    material_upload_event,
    environment_upload_event,
    viewport_resize_event,
    draw_mesh_event,
    directional_light_event,
    point_light_event,
    spot_light_event,
    render_world_event,
    debug_marker_event>;

/**
 * @brief Thread-producible typed render event.
 */
struct render_event
{
    render_event_payload payload{ debug_marker_event{} };

    /**
     * @brief Return the event kind without exposing variant internals.
     */
    render_event_type type() const noexcept;
};

/**
 * @brief Per-thread append-only render event buffer.
 */
class render_event_buffer
{
public:
    /**
     * @brief Append one event to this buffer.
     */
    void push(render_event event);

    /**
     * @brief Remove all events.
     */
    void clear();

    /**
     * @brief Return buffered events.
     */
    const std::vector<render_event>& events() const noexcept;

    /**
     * @brief Return whether the buffer contains no events.
     */
    bool empty() const noexcept;

private:
    std::vector<render_event> events_;
};

/**
 * @brief Convenience writer used by producer systems.
 */
class render_event_writer
{
public:
    explicit render_event_writer(render_event_buffer& buffer) noexcept;

    /**
     * @brief Append one event.
     */
    void push(render_event event);

    /**
     * @brief Append a viewport resize event.
     */
    void viewport_resize(std::uint32_t width, std::uint32_t height);

    /**
     * @brief Append a static mesh upload request.
     */
    void mesh_upload(mesh_handle handle, std::shared_ptr<const mesh_data> mesh, std::string label = {});

    /**
     * @brief Append a texture upload request.
     */
    void texture_upload(texture_handle handle, std::shared_ptr<const texture_data> texture, std::string label = {});

    /**
     * @brief Append a material upload request.
     */
    void material_upload(material_handle handle, std::shared_ptr<const material_desc> material, std::string label = {});

    /**
     * @brief Append an environment upload request.
     */
    void environment_upload(environment_handle handle, std::shared_ptr<const environment_desc> environment, std::string label = {});

    /**
     * @brief Append a static mesh draw request.
     */
    void draw_mesh(
        mesh_handle mesh,
        material_handle material,
        const math::matrix4f& model,
        const math::matrix4f& view_projection,
        std::string label);

    /**
     * @brief Append a static mesh draw request with editor render state.
     */
    void draw_mesh(
        mesh_handle mesh,
        material_handle material,
        const math::matrix4f& model,
        const math::matrix4f& view_projection,
        render_mode mode = render_mode::shaded,
        mesh_visualization_mode visualization = mesh_visualization_mode::standard,
        bool selected = false,
        const math::vector4f& wire_color = math::vector4f{ 0.25f, 0.65f, 1.0f, 1.0f },
        std::string label = {});

    /**
     * @brief Append a static mesh draw request with editor render state and entity tint.
     */
    void draw_mesh_tinted(
        mesh_handle mesh,
        material_handle material,
        const math::matrix4f& model,
        const math::matrix4f& view_projection,
        render_mode mode = render_mode::shaded,
        mesh_visualization_mode visualization = mesh_visualization_mode::standard,
        bool selected = false,
        const math::vector4f& base_color_tint = math::vector4f{ 1.0f, 1.0f, 1.0f, 1.0f },
        const math::vector4f& wire_color = math::vector4f{ 0.25f, 0.65f, 1.0f, 1.0f },
        std::string label = {});

    /**
     * @brief Append a directional light.
     */
    void directional_light(
        const math::vector3f& direction,
        const math::vector3f& color,
        float intensity,
        bool casts_shadows,
        std::string label = {},
        bool enabled = true,
        bool use_color_temperature = false,
        float temperature_kelvin = 6500.0f,
        light_intensity_unit intensity_unit = {},
        texture_handle cookie_texture = {},
        shadow_settings shadow = {});

    /**
     * @brief Append a point light.
     */
    void point_light(
        const math::vector3f& position,
        const math::vector3f& color,
        float intensity,
        float range,
        bool casts_shadows,
        std::string label = {},
        bool enabled = true,
        bool use_color_temperature = false,
        float temperature_kelvin = 6500.0f,
        light_intensity_unit intensity_unit = {},
        texture_handle cookie_texture = {},
        shadow_settings shadow = { .enabled = false });

    /**
     * @brief Append a spot light.
     */
    void spot_light(
        const math::vector3f& position,
        const math::vector3f& direction,
        const math::vector3f& color,
        float intensity,
        float range,
        float inner_angle,
        float outer_angle,
        bool casts_shadows,
        std::string label = {},
        bool enabled = true,
        bool use_color_temperature = false,
        float temperature_kelvin = 6500.0f,
        light_intensity_unit intensity_unit = {},
        texture_handle cookie_texture = {},
        shadow_settings shadow = { .enabled = false });

    /**
     * @brief Append a debug marker event.
     */
    void debug_marker(std::string label);

    /**
     * @brief Append a prepared scene render packet.
     */
    void render_world(std::shared_ptr<const render_world_packet> packet, std::string label = {});

private:
    render_event_buffer* buffer_{};
};

/**
 * @brief Immutable packet consumed by the renderer for one frame.
 */
struct render_frame_packet
{
    std::uint64_t frame_index{};
    std::vector<render_event> events;
};

/**
 * @brief Lock-light frame submission queue for render event buffers.
 */
class render_frame_queue
{
public:
    /**
     * @brief Submit a complete producer buffer for the next committed packet.
     */
    void submit(render_event_buffer buffer);

    /**
     * @brief Commit all submitted buffers into an immutable frame packet.
     */
    render_frame_packet commit(std::uint64_t frame_index);

    /**
     * @brief Return the number of pending producer buffers.
     */
    std::size_t pending_buffer_count() const;

private:
    mutable std::mutex mutex_;
    std::vector<render_event_buffer> pending_;
};

} // namespace arc::render
