#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace arc::editor
{

struct host_entity_id
{
    std::uint32_t index{ invalid_index };
    std::uint32_t generation{};

    static constexpr std::uint32_t invalid_index = 0xffffffffu;

    constexpr bool valid() const noexcept
    {
        return index != invalid_index;
    }

    friend constexpr bool operator==(host_entity_id lhs, host_entity_id rhs) noexcept
    {
        return lhs.index == rhs.index && lhs.generation == rhs.generation;
    }
};

struct host_vec3
{
    float x{};
    float y{};
    float z{};
};

struct host_quat
{
    float x{};
    float y{};
    float z{};
    float w{ 1.0f };
};

struct host_transform
{
    host_vec3 position{};
    host_quat rotation{};
    host_vec3 scale{ 1.0f, 1.0f, 1.0f };
};

enum class host_event_type : std::uint8_t
{
    host_started,
    host_shutdown,
    project_opened,
    project_closed,
    scene_changed,
    entity_created,
    entity_deleted,
    entity_selected,
    component_changed,
    command_failed,
    viewport_error
};

enum class host_entity_kind : std::uint8_t
{
    camera,
    light,
    environment,
    mesh,
    primitive,
    imported,
    unknown
};

enum class host_component_kind : std::uint8_t
{
    transform,
    mesh_renderer,
    directional_light,
    point_light,
    spot_light,
    sky_atmosphere,
    height_fog,
    terrain,
    water,
    vegetation,
    decal
};

enum class host_create_entity_kind : std::uint8_t
{
    plane,
    cube,
    sphere,
    cylinder,
    world_environment,
    terrain,
    water,
    grass_patch,
    decal
};

enum class host_camera_projection : std::uint8_t
{
    perspective,
    orthographic
};

enum class host_render_mode : std::uint8_t
{
    shaded,
    wireframe
};

enum class host_visualization_mode : std::uint8_t
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
    cluster_debug
};

enum class host_overlay_mode : std::uint8_t
{
    none,
    selected_wireframe,
    all_wireframe
};

struct host_environment_visibility
{
    bool sky{ true };
    bool fog{ true };
    bool terrain{ true };
    bool water{ true };
    bool vegetation{ true };
    bool decals{ true };
};

struct host_component_snapshot
{
    host_component_kind kind{ host_component_kind::transform };
    std::string label;
    bool editable{ true };
};

struct host_scene_entity_snapshot
{
    host_entity_id entity{};
    std::string name;
    host_entity_kind kind{ host_entity_kind::unknown };
    bool active{ true };
    bool selected{};
};

struct host_scene_snapshot
{
    std::vector<host_scene_entity_snapshot> entities;
};

struct host_selected_entity_snapshot
{
    host_entity_id entity{};
    std::string name;
    std::string tag;
    bool active{ true };
    std::optional<host_transform> transform;
    std::vector<host_component_snapshot> components;
};

struct host_asset_snapshot
{
    std::string path;
    std::string kind;
    bool imported{};
    bool import_running{};
};

struct host_project_assets_snapshot
{
    std::string project_name;
    std::filesystem::path project_root;
    std::filesystem::path asset_root;
    std::string default_mesh_path;
    bool default_mesh_loaded{};
    std::string default_mesh_message;
    std::vector<host_asset_snapshot> assets;
};

struct host_event
{
    std::uint64_t sequence{};
    host_event_type event_type{};
    host_entity_id entity{};
    std::string message;
    std::string payload_json;
};

struct host_open_project_command
{
    std::string name;
    std::filesystem::path root;
};

struct host_close_project_command
{
};

struct host_open_scene_command
{
    std::filesystem::path path;
    bool append{};
};

struct host_create_entity_command
{
    host_create_entity_kind kind{ host_create_entity_kind::cube };
};

struct host_delete_entity_command
{
    host_entity_id entity{};
};

struct host_rename_entity_command
{
    host_entity_id entity{};
    std::string name;
};

struct host_select_entity_command
{
    host_entity_id entity{};
};

struct host_clear_selection_command
{
};

struct host_set_active_command
{
    host_entity_id entity{};
    bool active{ true };
};

struct host_set_tag_command
{
    host_entity_id entity{};
    std::string tag;
};

struct host_set_transform_command
{
    host_entity_id entity{};
    host_transform transform;
};

struct host_set_camera_projection_command
{
    host_camera_projection projection{ host_camera_projection::perspective };
};

struct host_viewport_attach_command
{
    std::uint64_t native_handle{};
    std::int32_t x{};
    std::int32_t y{};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct host_viewport_resize_command
{
    std::int32_t x{};
    std::int32_t y{};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct host_viewport_set_camera_mode_command
{
    host_camera_projection projection{ host_camera_projection::perspective };
};

struct host_viewport_set_render_options_command
{
    host_render_mode render_mode{ host_render_mode::shaded };
    host_visualization_mode visualization{ host_visualization_mode::standard };
    host_overlay_mode overlay{ host_overlay_mode::selected_wireframe };
    bool shadows{ true };
    host_environment_visibility environment{};
};

struct host_viewport_camera_input_command
{
    float orbit_x{};
    float orbit_y{};
    float pan_x{};
    float pan_y{};
    float forward{};
    float zoom{};
    bool focus_selected{};
};

using host_command_payload = std::variant<
    host_open_project_command,
    host_close_project_command,
    host_open_scene_command,
    host_create_entity_command,
    host_delete_entity_command,
    host_rename_entity_command,
    host_select_entity_command,
    host_clear_selection_command,
    host_set_active_command,
    host_set_tag_command,
    host_set_transform_command,
    host_set_camera_projection_command,
    host_viewport_attach_command,
    host_viewport_resize_command,
    host_viewport_set_camera_mode_command,
    host_viewport_set_render_options_command,
    host_viewport_camera_input_command>;

struct host_command_envelope
{
    std::uint64_t request_id{};
    std::string command_type;
    host_command_payload payload{ host_close_project_command{} };
};

struct host_scene_hierarchy_query
{
};

struct host_selected_entity_query
{
};

struct host_project_assets_query
{
};

struct host_viewport_state_query
{
};

using host_query_payload = std::variant<
    host_scene_hierarchy_query,
    host_selected_entity_query,
    host_project_assets_query,
    host_viewport_state_query>;

struct host_query_envelope
{
    std::uint64_t request_id{};
    std::string query_type;
    host_query_payload payload{ host_scene_hierarchy_query{} };
};

struct host_response
{
    std::uint64_t request_id{};
    bool succeeded{};
    std::string error;
    std::string payload_json;
};

struct host_viewport_request
{
    std::uint64_t frame_index{};
    std::uint32_t width{};
    std::uint32_t height{};
    host_render_mode render_mode{ host_render_mode::shaded };
    host_visualization_mode visualization{ host_visualization_mode::standard };
    host_overlay_mode overlay{ host_overlay_mode::selected_wireframe };
    bool shadows{ true };
    host_environment_visibility environment{};
};

struct host_viewport_frame
{
    bool submitted{};
    std::string message;
    std::string payload_json;
};

const char* to_string(host_event_type value) noexcept;
const char* to_string(host_entity_kind value) noexcept;
const char* to_string(host_component_kind value) noexcept;
const char* to_string(host_create_entity_kind value) noexcept;
const char* to_string(host_camera_projection value) noexcept;
const char* to_string(host_render_mode value) noexcept;
const char* to_string(host_visualization_mode value) noexcept;
const char* to_string(host_overlay_mode value) noexcept;

std::string command_type(const host_command_payload& payload);
std::string query_type(const host_query_payload& payload);

std::string to_json(const host_command_envelope& envelope);
std::string to_json(const host_query_envelope& envelope);
std::string to_json(const host_response& response);
std::string to_json(const host_event& event);
std::string to_json(const host_scene_snapshot& snapshot);
std::string to_json(const host_selected_entity_snapshot& snapshot);
std::string to_json(const host_project_assets_snapshot& snapshot);
std::string to_json(const host_entity_id& entity);
std::string to_json(const host_transform& transform);

bool from_json(std::string_view json, host_command_envelope& envelope, std::string& error);
bool from_json(std::string_view json, host_query_envelope& envelope, std::string& error);

} // namespace arc::editor
