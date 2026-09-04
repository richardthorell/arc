#pragma once

#include <arc/render/handles.h>
#include <arc/render/gpu_scene.h>
#include <arc/render/material.h>
#include <arc/render/mesh.h>
#include <arc/render/lighting_scene.h>
#include <arc/render/shadow.h>
#include <arc/render/virtual_mesh.h>
#include <arc/render/virtual_geometry.h>
#include <arc/render/terrain.h>
#include <arc/render/texture_streaming.h>
#include <arc/math/matrix.h>
#include <arc/math/vector.h>
#include <arc/geometric/box.h>

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
enum class area_light_shape : std::uint8_t;
struct environment_descriptor;
struct render_world_packet;

/**
 * @brief Kinds of renderer events produced by game/editor threads.
 */
enum class render_event_type : std::uint8_t
{
    mesh_upload,
    mesh_destroy,
    virtual_mesh_upload,
    virtual_mesh_destroy,
    virtual_geometry_page_upload,
    terrain_upload,
    terrain_height_update,
    terrain_weight_update,
    terrain_destroy,
    lighting_geometry_upload,
    lighting_geometry_destroy,
    texture_upload,
    texture_stream_register,
    texture_stream_upload,
    texture_stream_evict,
    texture_destroy,
    material_upload,
    environment_upload,
    environment_destroy,
    viewport_resize,
    draw,
    directional_light,
    point_light,
    spot_light,
    area_light,
    gpu_scene_update,
    lighting_scene_update,
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
    light_complexity,
    cluster_debug,
    virtual_hierarchy_level,
    virtual_geometric_error,
    virtual_page_residency,
    virtual_overdraw,
    virtual_triangles_per_pixel,
    surface_cards,
    surface_card_residency,
    surface_material_cache,
    surface_radiance_cache,
    mesh_distance_fields,
    global_distance_field,
    radiance_probes,
    lighting_trace_source,
    lighting_hit_distance,
    lighting_temporal_confidence,
    indirect_diffuse,
    reflections,
    denoiser_variance,
    terrain_patch_boundaries,
    terrain_lod_level,
    terrain_hierarchy_nodes,
    terrain_geometric_error,
    terrain_culled_nodes,
    terrain_triangle_density,
    terrain_bounds,
    hzb_minimum_depth,
    hzb_maximum_depth,
    motion_vectors,
    temporal_reactive_mask,
    temporal_disocclusion,
    temporal_confidence,
    temporal_rejection,
    temporal_sample_weight,
    texture_desired_mip,
    texture_resident_mip,
    virtual_texture_page_residency,
    virtual_texture_recent_requests
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
    bool enabled{true};
    std::uint32_t resolution{2048};
    float bias{0.0015f};
    float normal_bias{0.01f};
    float strength{0.75f};
    shadow_filter filter{shadow_filter::pcf_3x3};
    std::uint16_t priority{128};
    bool contact_shadows{true};
    float contact_shadow_length{0.5f};
    shadow_cache_mode cache_mode{shadow_cache_mode::automatic};
    shadow_map_method map_method{shadow_map_method::auto_select};
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

/** @brief Retire a renderer-owned static or dynamic mesh. */
struct mesh_destroy_event
{
    mesh_handle handle{};
};

/**
 * @brief Upload a virtual mesh into backend-owned GPU resources.
 */
struct virtual_mesh_upload_event
{
    virtual_mesh_handle handle{};
    std::shared_ptr<const virtual_mesh_data> mesh;
    std::uint32_t resource_generation{1};
    std::string label;
};

/** @brief Retire renderer-owned virtual geometry after in-flight frames complete. */
struct virtual_mesh_destroy_event
{
    virtual_mesh_handle handle{};
};

/** @brief Publish one decoded virtual-geometry page on render affinity. */
struct virtual_geometry_page_upload_event
{
    virtual_geometry_page_upload upload;
};

/** @brief Upload or fully replace a renderer-owned terrain resource. */
struct terrain_upload_event
{
    terrain_handle handle{};
    std::shared_ptr<const terrain_resource_descriptor> terrain;
    std::string label;
};

/** @brief Upload a rectangular terrain height region. */
struct terrain_height_update_event
{
    terrain_handle handle{};
    std::shared_ptr<const terrain_height_region_update> update;
};

/** @brief Upload a rectangular terrain weight region. */
struct terrain_weight_update_event
{
    terrain_handle handle{};
    std::shared_ptr<const terrain_weight_region_update> update;
};

/** @brief Retire a renderer-owned terrain resource. */
struct terrain_destroy_event
{
    terrain_handle handle{};
};

/** @brief Publish cooked cards and distance-field pages for one conventional mesh. */
struct lighting_geometry_upload_event
{
    lighting_geometry_handle handle{};
    std::shared_ptr<const lighting_geometry_descriptor> geometry;
    std::string label;
};

/** @brief Retire a Lighting Scene geometry generation after in-flight frames complete. */
struct lighting_geometry_destroy_event
{
    lighting_geometry_handle handle{};
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

/** @brief Register immutable streamable-texture metadata with the backend. */
struct texture_stream_register_event
{
    texture_handle handle{};
    std::shared_ptr<const streamed_texture_descriptor> descriptor;
    std::string label;
};

/** @brief Publish one validated mip or virtual tile on render affinity. */
struct texture_stream_upload_event
{
    texture_stream_upload upload;
};

/** @brief Retire one non-pinned streamed subresource after frame protection. */
struct texture_stream_evict_event
{
    texture_stream_eviction eviction;
};

/** @brief Retire a conventional or streamable texture resource. */
struct texture_destroy_event
{
    texture_handle handle{};
};

/**
 * @brief Upload or replace a renderer material description.
 */
struct material_upload_event
{
    material_handle handle{};
    std::shared_ptr<const material_descriptor> material;
    std::string label;
};

/**
 * @brief Upload or replace an environment description.
 */
struct environment_upload_event
{
    environment_handle handle{};
    std::shared_ptr<const environment_descriptor> environment;
    std::string label;
};

/** @brief Retire a renderer-owned environment description. */
struct environment_destroy_event
{
    environment_handle handle{};
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
    gpu_scene_instance_handle gpu_scene_instance{};
    mesh_handle mesh{};
    material_handle material{};
    math::matrix4f model{math::identity<float, 4>()};
    math::matrix4f previous_model{math::identity<float, 4>()};
    math::matrix4f view_projection{math::identity<float, 4>()};
    math::matrix4f previous_view_projection{math::identity<float, 4>()};
    geometric::box3f world_bounds{};
    render_mode mode{render_mode::shaded};
    mesh_visualization_mode visualization{mesh_visualization_mode::standard};
    render_object_id object_id{};
    bool selected{};
    bool casts_shadows{true};
    bool receives_shadows{true};
    render_mobility mobility{render_mobility::movable};
    float shadow_lod_bias{};
    float maximum_shadow_distance{};
    math::vector4f base_color_tint = math::vector4f::one;
    math::vector4f wire_color{0.25f, 0.65f, 1.0f, 1.0f};
    std::string label;
};

/**
 * @brief Submit one directional light to the renderer.
 */
struct directional_light_event
{
    render_object_id object_id{};
    math::vector3f direction{0.0f, -1.0f, 0.0f};
    math::vector3f color = math::vector3f::one;
    float intensity{1.0f};
    bool casts_shadows{};
    bool enabled{true};
    bool use_color_temperature{};
    float temperature_kelvin{6500.0f};
    light_intensity_unit intensity_unit{};
    texture_handle cookie_texture{};
    shadow_settings shadow{};
    directional_shadow_settings cascades{};
    render_mobility mobility{render_mobility::movable};
    std::string label;
};

/**
 * @brief Submit one point light to the renderer.
 */
struct point_light_event
{
    render_object_id object_id{};
    math::vector3f position{};
    math::vector3f color = math::vector3f::one;
    float intensity{1.0f};
    float range{10.0f};
    bool casts_shadows{};
    bool enabled{true};
    bool use_color_temperature{};
    float temperature_kelvin{6500.0f};
    light_intensity_unit intensity_unit{};
    texture_handle cookie_texture{};
    shadow_settings shadow{.enabled = false};
    render_mobility mobility{render_mobility::movable};
    std::string label;
};

/**
 * @brief Submit one spot light to the renderer.
 */
struct spot_light_event
{
    render_object_id object_id{};
    math::vector3f position{};
    math::vector3f direction{0.0f, -1.0f, 0.0f};
    math::vector3f color = math::vector3f::one;
    float intensity{1.0f};
    float range{10.0f};
    float inner_angle{0.35f};
    float outer_angle{0.75f};
    bool casts_shadows{};
    bool enabled{true};
    bool use_color_temperature{};
    float temperature_kelvin{6500.0f};
    light_intensity_unit intensity_unit{};
    texture_handle cookie_texture{};
    shadow_settings shadow{.enabled = false};
    render_mobility mobility{render_mobility::movable};
    std::string label;
};

/**
 * @brief Rectangle/disk area light evaluated through a raster approximation.
 */
struct area_light_event
{
    render_object_id object_id{};
    math::vector3f position{};
    math::vector3f direction{0.0f, -1.0f, 0.0f};
    math::vector3f tangent{1.0f, 0.0f, 0.0f};
    math::vector3f color = math::vector3f::one;
    float intensity{100.0f};
    float width{1.0f};
    float height{1.0f};
    area_light_shape shape{};
    bool two_sided{};
    bool casts_shadows{};
    bool enabled{true};
    bool use_color_temperature{};
    float temperature_kelvin{6500.0f};
    light_intensity_unit intensity_unit{};
    shadow_settings shadow{.enabled = false};
    render_mobility mobility{render_mobility::movable};
    std::string label;
};

/**
 * @brief Insert a renderer debug marker.
 */
struct debug_marker_event
{
    std::string label;
};

/** @brief Incremental persistent GPU Scene mutations for one submitted frame. */
struct gpu_scene_update_event
{
    std::shared_ptr<const gpu_scene_update_batch> batch;
};

/** @brief Incremental backend-neutral Lighting Scene mutations for one submitted frame. */
struct lighting_scene_update_event
{
    std::shared_ptr<const lighting_scene_update_batch> batch;
};

/**
 * @brief Submit a prepared scene render packet to the backend.
 */
struct render_world_event
{
    std::shared_ptr<const render_world_packet> packet;
    std::string label;
};

using render_event_payload =
    std::variant<mesh_upload_event, mesh_destroy_event, virtual_mesh_upload_event, virtual_mesh_destroy_event,
                 virtual_geometry_page_upload_event, terrain_upload_event, terrain_height_update_event,
                 terrain_weight_update_event, terrain_destroy_event, lighting_geometry_upload_event,
                 lighting_geometry_destroy_event, texture_upload_event, texture_stream_register_event,
                 texture_stream_upload_event, texture_stream_evict_event, texture_destroy_event, material_upload_event,
                 environment_upload_event, environment_destroy_event, viewport_resize_event, draw_mesh_event,
                 directional_light_event, point_light_event, spot_light_event, area_light_event, gpu_scene_update_event,
                 lighting_scene_update_event, render_world_event, debug_marker_event>;

/**
 * @brief Thread-producible typed render event.
 */
struct render_event
{
    render_event_payload payload{debug_marker_event{}};

    /**
     * @brief Return the event kind without exposing variant internals.
     */
    [[nodiscard]] render_event_type type() const noexcept;
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
    [[nodiscard]] const std::vector<render_event>& events() const noexcept;

    /**
     * @brief Return whether the buffer contains no events.
     */
    [[nodiscard]] bool empty() const noexcept;

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

    /** @brief Append a mesh retirement request. */
    void mesh_destroy(mesh_handle handle);

    /**
     * @brief Append a virtual mesh upload request.
     */
    void virtual_mesh_upload(virtual_mesh_handle handle, std::shared_ptr<const virtual_mesh_data> mesh,
                             std::uint32_t resource_generation, std::string label = {});

    /** @brief Append a virtual-geometry retirement request. */
    void virtual_mesh_destroy(virtual_mesh_handle handle);

    /** @brief Append one generation-safe decoded virtual page publication. */
    void virtual_geometry_page_upload(virtual_geometry_page_upload upload);

    /** @brief Append a complete terrain upload. */
    void terrain_upload(terrain_handle handle, std::shared_ptr<const terrain_resource_descriptor> terrain,
                        std::string label = {});
    /** @brief Append a partial height update. */
    void terrain_height_update(terrain_handle handle, std::shared_ptr<const terrain_height_region_update> update);
    /** @brief Append a partial weight update. */
    void terrain_weight_update(terrain_handle handle, std::shared_ptr<const terrain_weight_region_update> update);
    /** @brief Append terrain retirement. */
    void terrain_destroy(terrain_handle handle);

    /**
     * @brief Append a texture upload request.
     */
    void texture_upload(texture_handle handle, std::shared_ptr<const texture_data> texture, std::string label = {});

    /** @brief Append a streamable texture registration. */
    void texture_stream_register(texture_handle handle, std::shared_ptr<const streamed_texture_descriptor> descriptor,
                                 std::string label = {});

    /** @brief Append a validated streamable subresource upload. */
    void texture_stream_upload(arc::render::texture_stream_upload upload);

    /** @brief Append a streamed subresource eviction. */
    void texture_stream_evict(arc::render::texture_stream_eviction eviction);

    /** @brief Append a texture retirement request. */
    void texture_destroy(texture_handle handle);

    /**
     * @brief Append a material upload request.
     */
    void material_upload(material_handle handle, std::shared_ptr<const material_descriptor> material,
                         std::string label = {});

    /**
     * @brief Append an environment upload request.
     */
    void environment_upload(environment_handle handle, std::shared_ptr<const environment_descriptor> environment,
                            std::string label = {});

    /** @brief Append an environment retirement request. */
    void environment_destroy(environment_handle handle);

    /**
     * @brief Append a static mesh draw request.
     */
    void draw_mesh(mesh_handle mesh, material_handle material, const math::matrix4f& model,
                   const math::matrix4f& view_projection, std::string label);

    /**
     * @brief Append a static mesh draw request with editor render state.
     */
    void draw_mesh(mesh_handle mesh, material_handle material, const math::matrix4f& model,
                   const math::matrix4f& view_projection, render_mode mode = render_mode::shaded,
                   mesh_visualization_mode visualization = mesh_visualization_mode::standard, bool selected = false,
                   const math::vector4f& wire_color = math::vector4f{0.25f, 0.65f, 1.0f, 1.0f}, std::string label = {});

    /**
     * @brief Append a static mesh draw request with editor render state and entity tint.
     */
    void draw_mesh_tinted(mesh_handle mesh, material_handle material, const math::matrix4f& model,
                          const math::matrix4f& view_projection, render_mode mode = render_mode::shaded,
                          mesh_visualization_mode visualization = mesh_visualization_mode::standard,
                          bool selected = false, const math::vector4f& base_color_tint = math::vector4f::one,
                          const math::vector4f& wire_color = math::vector4f{0.25f, 0.65f, 1.0f, 1.0f},
                          std::string label = {});

    /**
     * @brief Append a directional light.
     */
    void directional_light(const math::vector3f& direction, const math::vector3f& color, float intensity,
                           bool casts_shadows, std::string label = {}, bool enabled = true,
                           bool use_color_temperature = false, float temperature_kelvin = 6500.0f,
                           light_intensity_unit intensity_unit = {}, texture_handle cookie_texture = {},
                           shadow_settings shadow = {});

    /**
     * @brief Append a point light.
     */
    void point_light(const math::vector3f& position, const math::vector3f& color, float intensity, float range,
                     bool casts_shadows, std::string label = {}, bool enabled = true,
                     bool use_color_temperature = false, float temperature_kelvin = 6500.0f,
                     light_intensity_unit intensity_unit = {}, texture_handle cookie_texture = {},
                     shadow_settings shadow = {.enabled = false});

    /**
     * @brief Append a spot light.
     */
    void spot_light(const math::vector3f& position, const math::vector3f& direction, const math::vector3f& color,
                    float intensity, float range, float inner_angle, float outer_angle, bool casts_shadows,
                    std::string label = {}, bool enabled = true, bool use_color_temperature = false,
                    float temperature_kelvin = 6500.0f, light_intensity_unit intensity_unit = {},
                    texture_handle cookie_texture = {}, shadow_settings shadow = {.enabled = false});

    /**
     * @brief Append a rectangle/disk area light.
     */
    void area_light(const math::vector3f& position, const math::vector3f& direction, const math::vector3f& tangent,
                    const math::vector3f& color, float intensity, float width, float height, area_light_shape shape,
                    bool two_sided, bool casts_shadows, std::string label = {}, bool enabled = true,
                    bool use_color_temperature = false, float temperature_kelvin = 6500.0f,
                    light_intensity_unit intensity_unit = {}, shadow_settings shadow = {.enabled = false});

    /**
     * @brief Append a debug marker event.
     */
    void debug_marker(std::string label);

    /** @brief Append persistent GPU Scene mutations. */
    void gpu_scene_update(std::shared_ptr<const gpu_scene_update_batch> batch);

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
    [[nodiscard]] std::size_t pending_buffer_count() const;

private:
    mutable std::mutex mutex_;
    std::vector<render_event_buffer> pending_;
};

} // namespace arc::render
