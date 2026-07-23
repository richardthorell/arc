#pragma once

#include <array>
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

    friend constexpr bool operator==(const host_vec3&, const host_vec3&) noexcept = default;
};

struct host_vec4
{
    float x{};
    float y{};
    float z{};
    float w{};

    friend constexpr bool operator==(const host_vec4&, const host_vec4&) noexcept = default;
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
    viewport_error,
    profiler_snapshot,
    terrain_tool_changed,
    terrain_stroke_committed,
    runtime_state_changed,
    runtime_tick_completed,
    runtime_fault
};

enum class host_runtime_state : std::uint8_t
{
    stopped,
    running,
    paused,
    faulted
};

struct host_runtime_snapshot
{
    host_runtime_state state{ host_runtime_state::stopped };
    std::uint64_t tick_id{};
    std::uint64_t revision{};
    std::uint64_t discarded_ticks{};
    double time_scale{ 1.0 };
    double interpolation_alpha{};
    std::uint32_t world_count{};
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
    camera,
    mesh_renderer,
    directional_light,
    point_light,
    spot_light,
    world_environment,
    sky_atmosphere,
    celestial_sky,
    cloud_layers,
    environment_lighting,
    height_fog,
    terrain,
    water,
    vegetation,
    decal,
    prefab_instance
};

inline constexpr std::uint32_t host_default_render_layer = 1u << 0u;
inline constexpr std::uint32_t host_environment_render_layer = 1u << 1u;

enum class host_sky_source : std::uint8_t { physical_atmosphere, hdri, solid_color };
enum class host_sun_position_mode : std::uint8_t { manual_light, geographic };
enum class host_celestial_time_mode : std::uint8_t { fixed, simulated, system_clock };
enum class host_environment_lighting_source : std::uint8_t { follow_sky, hdri, constant_color };
enum class host_world_environment_preset : std::uint8_t
{
    clear_day,
    alpine_late_morning,
    golden_hour,
    overcast,
    night,
    indoor_neutral
};

enum class host_create_entity_kind : std::uint8_t
{
    empty,
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

enum class host_viewport_tool : std::uint8_t { select, translate, rotate, scale, terrain };
enum class host_terrain_brush_tool : std::uint8_t { sculpt, smooth, flatten, paint };
enum class host_coordinate_space : std::uint8_t { world, local };
enum class host_edit_phase : std::uint8_t { none, begin, update, commit, cancel };

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

struct host_camera_snapshot
{
    host_camera_projection projection{ host_camera_projection::perspective };
    float fov_y_degrees{ 60.0f };
    float orthographic_height{ 10.0f };
    float near_plane{ 0.01f };
    float far_plane{ 1000.0f };
    bool active{ true };
    host_vec4 clear_color{ 0.10f, 0.22f, 0.34f, 1.0f };

    friend constexpr bool operator==(const host_camera_snapshot&, const host_camera_snapshot&) noexcept = default;
};

struct host_mesh_renderer_snapshot
{
    bool visible{ true };
    host_vec4 base_color_tint{ 1.0f, 1.0f, 1.0f, 1.0f };
    bool has_material{};
    bool asset_backed_material{};
    std::string material_name;
    std::string material_path;

    friend constexpr bool operator==(const host_mesh_renderer_snapshot&, const host_mesh_renderer_snapshot&) noexcept = default;
};

struct host_terrain_snapshot
{
    bool enabled{ true };
    float size{ 180.0f };
    std::uint32_t resolution{ 257u };
    std::uint32_t chunk_quads{ 128u };
    bool receive_shadows{ true };
    std::uint64_t content_revision{};
    host_terrain_brush_tool brush_tool{ host_terrain_brush_tool::sculpt };
    float brush_radius{ 6.0f };
    float brush_strength{ 0.25f };
    float brush_falloff{ 1.0f };
    std::uint32_t active_layer{};
    std::array<std::string, 4> layer_names{ "Grass", "Dirt", "Rock", "Sand" };
    std::array<std::string, 4> layer_base_color_paths{};
};

struct host_terrain_tool_snapshot
{
    host_entity_id entity{};
    bool active{};
    bool hover_visible{};
    host_terrain_brush_tool tool{ host_terrain_brush_tool::sculpt };
    float radius{ 6.0f };
    float strength{ 0.25f };
    float falloff{ 1.0f };
    std::uint32_t active_layer{};
};

struct host_prefab_snapshot
{
    std::string prefab_guid;
    std::string prefab_path;
    std::size_t override_count{};
    bool source_missing{};
};

struct host_scene_entity_snapshot
{
    host_entity_id entity{};
    std::string guid;
    std::string parent_guid;
    std::uint32_t sibling_order{};
    std::string name;
    host_entity_kind kind{ host_entity_kind::unknown };
    bool active{ true };
    bool selected{};
};

struct host_scene_snapshot
{
    std::vector<host_scene_entity_snapshot> entities;
    std::string scene_guid;
    std::string scene_name;
    std::string active_scene_path;
    bool dirty{};
    bool can_undo{};
    bool can_redo{};
    std::string undo_label;
    std::string redo_label;
};

struct host_selected_entity_snapshot
{
    host_entity_id entity{};
    std::string name;
    std::string tag;
    bool active{ true };
    std::uint32_t render_layer_mask{ host_default_render_layer };
    std::optional<host_transform> transform;
    std::optional<host_camera_snapshot> camera;
    std::optional<host_mesh_renderer_snapshot> mesh_renderer;
    std::optional<host_terrain_snapshot> terrain;
    std::optional<host_prefab_snapshot> prefab;
    std::vector<host_component_snapshot> components;
};

struct host_cloud_layer
{
    bool enabled{ true };
    float coverage{};
    float density{};
    float altitude{};
    float thickness{};
    float scale{};
    float detail{};
    float softness{};
    float wind_x{ 1.0f };
    float wind_y{};
    float wind_speed{};
    float lighting_strength{ 1.0f };
    float silver_lining{};

    friend constexpr bool operator==(const host_cloud_layer&, const host_cloud_layer&) noexcept = default;
};

struct host_world_environment_snapshot
{
    host_entity_id entity{};
    bool enabled{ true };
    bool sky_visible{ true };
    bool affect_lighting{ true };
    host_sky_source sky_source{ host_sky_source::physical_atmosphere };
    host_vec3 solid_color{ 0.08f, 0.13f, 0.22f };
    std::string hdri_path;
    float hdri_rotation_degrees{};
    float radiance_intensity{ 1.0f };
    float planet_radius{ 6360.0f };
    float atmosphere_radius{ 6420.0f };
    float rayleigh_strength{ 1.0f };
    float mie_strength{ 0.35f };
    float ozone_strength{ 0.15f };
    host_vec3 atmosphere_tint{ 0.56f, 0.72f, 1.0f };
    host_vec3 ground_albedo{ 0.18f, 0.18f, 0.18f };
    float mie_anisotropy{ 0.8f };
    float rayleigh_scale_height{ 8.0f };
    float mie_scale_height{ 1.2f };
    float multi_scattering_factor{ 1.0f };
    float exposure{ 1.0f };
    float sun_disk_size{ 0.025f };
    float sun_disk_intensity{ 1.4f };
    host_sun_position_mode sun_mode{ host_sun_position_mode::manual_light };
    host_celestial_time_mode time_mode{ host_celestial_time_mode::fixed };
    float latitude_degrees{ 46.8f };
    float longitude_degrees{ 8.2f };
    float north_offset_degrees{};
    std::int32_t year{ 2026 };
    std::int32_t month{ 7 };
    std::int32_t day{ 14 };
    float local_time_hours{ 10.5f };
    float utc_offset_hours{ 2.0f };
    bool playing{};
    bool loop_day{ true };
    float time_scale{ 60.0f };
    bool automatic_sun_light{ true };
    float sun_intensity_multiplier{ 1.0f };
    float sun_temperature_multiplier{ 1.0f };
    bool moon_enabled{ true };
    bool automatic_moon_phase{ true };
    float moon_phase{ 0.65f };
    float moon_intensity{ 0.22f };
    float moon_angular_radius_degrees{ 0.2725f };
    bool stars_enabled{ true };
    float star_density{ 0.42f };
    float star_intensity{ 0.75f };
    float star_twinkle{ 0.08f };
    bool clouds_enabled{ true };
    bool cloud_shadows{ true };
    host_cloud_layer cumulus;
    host_cloud_layer cirrus;
    bool fog_enabled{ true };
    host_vec3 fog_color{ 0.58f, 0.67f, 0.76f };
    float fog_density{ 0.035f };
    float fog_height_falloff{ 0.12f };
    float fog_start_distance{ 8.0f };
    float fog_max_opacity{ 0.55f };
    float fog_sun_scattering{ 0.25f };
    bool lighting_enabled{ true };
    host_environment_lighting_source lighting_source{ host_environment_lighting_source::follow_sky };
    host_vec3 lighting_color{ 0.18f, 0.23f, 0.29f };
    float diffuse_intensity{ 1.0f };
    float specular_intensity{ 1.0f };

    friend bool operator==(
        const host_world_environment_snapshot&,
        const host_world_environment_snapshot&) noexcept = default;
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

struct host_asset_thumbnail_snapshot
{
    std::string path;
    std::uint32_t width{};
    std::uint32_t height{};
    std::string data_url;
};

struct host_event
{
    std::uint64_t sequence{};
    host_event_type event_type{};
    host_entity_id entity{};
    std::string message;
    std::string payload_json;
};

struct host_job_profile_sample
{
    std::uint64_t sequence{};
    std::string name;
    std::string priority;
    std::string affinity;
    std::string status;
    std::uint64_t thread_id{};
    std::uint64_t queued_nanoseconds{};
    std::uint64_t started_nanoseconds{};
    std::uint64_t completed_nanoseconds{};
};

struct host_memory_domain_sample
{
    std::string domain;
    std::uint64_t bytes_outstanding{};
    std::uint64_t peak_bytes{};
    std::uint64_t soft_limit{};
    std::uint64_t hard_limit{};
    bool pressure{};
};

struct host_memory_allocation_group
{
    std::string domain;
    std::string tag;
    std::uint64_t world_id{};
    std::uint64_t thread_id{};
    std::uint64_t stack_id{};
    std::uint64_t allocation_count{};
    std::uint64_t bytes_outstanding{};
};

struct host_profiler_snapshot
{
    std::uint64_t timestamp_nanoseconds{};
    std::uint64_t memory_bytes{};
    std::uint64_t memory_soft_limit{};
    std::uint64_t memory_hard_limit{};
    std::uint64_t memory_pressure_events{};
    std::uint64_t jobs_submitted{};
    std::uint64_t jobs_completed{};
    std::uint64_t jobs_stolen{};
    std::uint64_t jobs_cancelled{};
    std::uint64_t jobs_failed{};
    std::uint64_t jobs_queued{};
    std::uint64_t dropped_profile_events{};
    std::vector<host_memory_domain_sample> memory_domains;
    std::vector<host_memory_allocation_group> allocation_groups;
    std::vector<host_job_profile_sample> jobs;
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

struct host_new_scene_command { std::string name{ "Untitled" }; };
struct host_save_scene_command {};
struct host_save_scene_as_command { std::filesystem::path path; };

struct host_create_entity_command
{
    host_create_entity_kind kind{ host_create_entity_kind::cube };
    host_entity_id parent{};
};

struct host_delete_entity_command
{
    host_entity_id entity{};
};

struct host_duplicate_entity_command { host_entity_id entity{}; };
struct host_create_prefab_command { host_entity_id entity{}; std::filesystem::path path; };
struct host_instantiate_prefab_command { std::filesystem::path path; host_entity_id parent{}; };
struct host_apply_prefab_command { host_entity_id entity{}; };
struct host_revert_prefab_command { host_entity_id entity{}; };
struct host_unpack_prefab_command { host_entity_id entity{}; };
struct host_reparent_entity_command
{
    host_entity_id entity{};
    host_entity_id parent{};
    host_entity_id before_sibling{};
    bool preserve_world{ true };
};
struct host_reorder_entity_command { host_entity_id entity{}; host_entity_id before_sibling{}; };

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

struct host_set_render_layer_command
{
    host_entity_id entity{};
    std::uint32_t render_layer_mask{ host_default_render_layer };
};

struct host_set_camera_command
{
    host_entity_id entity{};
    host_camera_snapshot camera;
};

struct host_set_mesh_renderer_command
{
    host_entity_id entity{};
    bool visible{ true };
    host_vec4 base_color_tint{ 1.0f, 1.0f, 1.0f, 1.0f };
};

struct host_set_terrain_command
{
    host_entity_id entity{};
    bool enabled{ true };
    bool receive_shadows{ true };
};

struct host_set_terrain_brush_command
{
    host_entity_id entity{};
    host_terrain_brush_tool tool{ host_terrain_brush_tool::sculpt };
    float radius{ 6.0f };
    float strength{ 0.25f };
    float falloff{ 1.0f };
    std::uint32_t active_layer{};
};

struct host_set_terrain_layer_command
{
    host_entity_id entity{};
    std::uint32_t layer{};
    std::filesystem::path path;
};

struct host_terrain_stroke_command
{
    host_entity_id entity{};
    std::uint32_t x{};
    std::uint32_t y{};
    host_edit_phase phase{ host_edit_phase::begin };
    bool invert{};
};

struct host_terrain_hover_command
{
    host_entity_id entity{};
    std::uint32_t x{};
    std::uint32_t y{};
    bool clear{};
};

struct host_set_entity_material_command
{
    host_entity_id entity{};
    std::filesystem::path path;
};

struct host_set_world_environment_command
{
    host_world_environment_snapshot environment;
};

struct host_apply_world_environment_preset_command
{
    host_entity_id entity{};
    host_world_environment_preset preset{ host_world_environment_preset::alpine_late_morning };
};

struct host_set_environment_hdri_command
{
    host_entity_id entity{};
    std::filesystem::path path;
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

struct host_history_undo_command {};
struct host_history_redo_command {};
struct host_runtime_resume_command {};
struct host_runtime_pause_command {};
struct host_runtime_stop_command {};
struct host_runtime_step_command { std::uint32_t ticks{ 1 }; };
struct host_runtime_set_time_scale_command { double value{ 1.0 }; };
struct host_runtime_capture_snapshot_command { std::string label; };
struct host_runtime_restore_snapshot_command { std::uint64_t snapshot_id{}; };
struct host_viewport_set_tool_command
{
    host_viewport_tool tool{ host_viewport_tool::select };
    host_coordinate_space coordinate_space{ host_coordinate_space::world };
    bool snapping{};
    float translation_snap{ 0.25f };
    float rotation_snap_degrees{ 15.0f };
    float scale_snap{ 0.1f };
};
struct host_viewport_pick_command { std::uint32_t x{}; std::uint32_t y{}; };

using host_command_payload = std::variant<
    host_open_project_command,
    host_close_project_command,
    host_open_scene_command,
    host_new_scene_command,
    host_save_scene_command,
    host_save_scene_as_command,
    host_create_entity_command,
    host_delete_entity_command,
    host_duplicate_entity_command,
    host_create_prefab_command,
    host_instantiate_prefab_command,
    host_apply_prefab_command,
    host_revert_prefab_command,
    host_unpack_prefab_command,
    host_reparent_entity_command,
    host_reorder_entity_command,
    host_rename_entity_command,
    host_select_entity_command,
    host_clear_selection_command,
    host_set_active_command,
    host_set_tag_command,
    host_set_transform_command,
    host_set_render_layer_command,
    host_set_camera_command,
    host_set_mesh_renderer_command,
    host_set_terrain_command,
    host_set_terrain_brush_command,
    host_set_terrain_layer_command,
    host_terrain_stroke_command,
    host_terrain_hover_command,
    host_set_entity_material_command,
    host_set_world_environment_command,
    host_apply_world_environment_preset_command,
    host_set_environment_hdri_command,
    host_set_camera_projection_command,
    host_viewport_attach_command,
    host_viewport_resize_command,
    host_viewport_set_camera_mode_command,
    host_viewport_set_render_options_command,
    host_viewport_camera_input_command,
    host_history_undo_command,
    host_history_redo_command,
    host_runtime_resume_command,
    host_runtime_pause_command,
    host_runtime_stop_command,
    host_runtime_step_command,
    host_runtime_set_time_scale_command,
    host_runtime_capture_snapshot_command,
    host_runtime_restore_snapshot_command,
    host_viewport_set_tool_command,
    host_viewport_pick_command>;

struct host_edit_transaction
{
    std::uint64_t id{};
    host_edit_phase phase{ host_edit_phase::none };
    std::string label;
};

struct host_command_envelope
{
    std::uint64_t request_id{};
    std::string command_type;
    host_command_payload payload{ host_close_project_command{} };
    std::optional<host_edit_transaction> edit;
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

struct host_asset_thumbnail_query
{
    std::string path;
    std::uint32_t max_size{ 96 };
};

struct host_viewport_state_query
{
};

struct host_world_environment_query
{
    host_entity_id entity{};
};
struct host_history_state_query {};
struct host_runtime_state_query {};
struct host_terrain_tool_state_query {};

using host_query_payload = std::variant<
    host_scene_hierarchy_query,
    host_selected_entity_query,
    host_project_assets_query,
    host_asset_thumbnail_query,
    host_viewport_state_query,
    host_world_environment_query,
    host_history_state_query,
    host_runtime_state_query,
    host_terrain_tool_state_query>;

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
const char* to_string(host_sky_source value) noexcept;
const char* to_string(host_sun_position_mode value) noexcept;
const char* to_string(host_celestial_time_mode value) noexcept;
const char* to_string(host_environment_lighting_source value) noexcept;
const char* to_string(host_world_environment_preset value) noexcept;
const char* to_string(host_runtime_state value) noexcept;

std::string command_type(const host_command_payload& payload);
std::string query_type(const host_query_payload& payload);

std::string to_json(const host_command_envelope& envelope);
std::string to_json(const host_query_envelope& envelope);
std::string to_json(const host_response& response);
std::string to_json(const host_event& event);
std::string to_json(const host_scene_snapshot& snapshot);
std::string to_json(const host_selected_entity_snapshot& snapshot);
std::string to_json(const host_project_assets_snapshot& snapshot);
std::string to_json(const host_asset_thumbnail_snapshot& snapshot);
std::string to_json(const host_entity_id& entity);
std::string to_json(const host_transform& transform);
std::string to_json(const host_camera_snapshot& camera);
std::string to_json(const host_mesh_renderer_snapshot& mesh_renderer);
std::string to_json(const host_world_environment_snapshot& environment);
std::string to_json(const host_profiler_snapshot& snapshot);
std::string to_json(const host_runtime_snapshot& snapshot);
std::string to_json(const host_terrain_tool_snapshot& snapshot);
std::string to_json_string(std::string_view value);

bool from_json(std::string_view json, host_command_envelope& envelope, std::string& error);
bool from_json(std::string_view json, host_query_envelope& envelope, std::string& error);
bool from_json(std::string_view json, host_world_environment_snapshot& environment, std::string& error);

} // namespace arc::editor
