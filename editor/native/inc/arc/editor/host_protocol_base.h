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
    std::uint32_t index{invalid_index};
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
    float w{1.0f};
};

struct host_transform
{
    host_vec3 position{};
    host_quat rotation{};
    host_vec3 scale{1.0f, 1.0f, 1.0f};
};

enum class host_event_type : std::uint8_t
{
    host_started,
    host_shutdown,
    project_opened,
    project_closed,
    project_module_reloaded,
    scene_changed,
    entity_created,
    entity_deleted,
    entity_selected,
    component_changed,
    command_failed,
    viewport_error,
    viewport_frame_ready,
    profiler_snapshot,
    terrain_tool_changed,
    terrain_stroke_committed,
    terrain_operation_changed,
    runtime_state_changed,
    runtime_tick_completed,
    runtime_fault,
    asset_changed
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
    host_runtime_state state{host_runtime_state::stopped};
    std::uint64_t tick_id{};
    std::uint64_t revision{};
    std::uint64_t discarded_ticks{};
    double time_scale{1.0};
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
    area_light,
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

enum class host_sky_source : std::uint8_t
{
    physical_atmosphere,
    hdri,
    solid_color
};
enum class host_sun_position_mode : std::uint8_t
{
    manual_light,
    geographic
};
enum class host_celestial_time_mode : std::uint8_t
{
    fixed,
    simulated,
    system_clock
};
enum class host_environment_lighting_source : std::uint8_t
{
    follow_sky,
    hdri,
    constant_color
};
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
    cone,
    capsule,
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

enum class host_mobility : std::uint8_t
{
    static_object,
    stationary,
    movable
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
    temporal_sample_weight
};
enum class host_indirect_lighting_method : std::uint8_t
{
    auto_select,
    baked_probe,
    screen_space,
    software,
    hybrid_hardware
};

enum class host_overlay_mode : std::uint8_t
{
    none,
    selected_wireframe,
    all_wireframe
};

enum class host_viewport_tool : std::uint8_t
{
    select,
    translate,
    rotate,
    scale,
    terrain
};
enum class host_terrain_brush_tool : std::uint8_t
{
    sculpt,
    smooth,
    flatten,
    paint
};
enum class host_coordinate_space : std::uint8_t
{
    world,
    local
};
enum class host_edit_phase : std::uint8_t
{
    none,
    begin,
    update,
    commit,
    cancel
};

struct host_environment_visibility
{
    bool sky{true};
    bool fog{true};
    bool terrain{true};
    bool water{true};
    bool vegetation{true};
    bool decals{true};
};

struct host_component_snapshot
{
    host_component_kind kind{host_component_kind::transform};
    std::string type_id;
    std::string label;
    std::uint64_t revision{};
    std::uint64_t dirty_fields{};
    bool editable{true};
};

struct host_project_component_snapshot
{
    std::string type_id;
    std::string canonical_name;
    std::string display_name;
    std::uint32_t schema_version{1};
    std::string values_json{"{}"};
};

struct host_bounds_snapshot
{
    host_vec3 minimum{};
    host_vec3 maximum{};
};

enum class host_exposure_mode : std::uint8_t
{
    manual,
    automatic
};

enum class host_exposure_metering_mode : std::uint8_t
{
    average,
    center_weighted
};

enum class host_camera_anti_aliasing : std::uint8_t
{
    inherit,
    disabled,
    fxaa,
    taa,
    taau
};

struct host_camera_snapshot
{
    host_camera_projection projection{host_camera_projection::perspective};
    float fov_y_degrees{60.0f};
    float orthographic_height{10.0f};
    float near_plane{0.01f};
    float far_plane{1000.0f};
    bool active{true};
    host_vec4 clear_color{0.10f, 0.22f, 0.34f, 1.0f};
    host_exposure_mode exposure_mode{host_exposure_mode::automatic};
    host_exposure_metering_mode exposure_metering{host_exposure_metering_mode::average};
    float manual_ev100{10.0f};
    float exposure_compensation{};
    float minimum_ev100{-8.0f};
    float maximum_ev100{20.0f};
    float brighten_speed{3.0f};
    float darken_speed{1.0f};
    host_camera_anti_aliasing anti_aliasing{host_camera_anti_aliasing::inherit};

    friend constexpr bool operator==(const host_camera_snapshot&, const host_camera_snapshot&) noexcept = default;
};

enum class host_light_kind : std::uint8_t
{
    directional,
    point,
    spot,
    rectangle,
    disk
};

enum class host_light_unit : std::uint8_t
{
    unitless,
    lumen,
    candela,
    lux,
    nit
};

struct host_light_snapshot
{
    host_light_kind kind{host_light_kind::point};
    host_light_unit unit{host_light_unit::lumen};
    host_vec3 color{1.0f, 1.0f, 1.0f};
    float intensity{1000.0f};
    float range{10.0f};
    float inner_angle_degrees{20.0f};
    float outer_angle_degrees{40.0f};
    float width{1.0f};
    float height{1.0f};
    bool two_sided{};
    bool enabled{true};
    bool casts_shadows{};
    std::uint32_t shadow_resolution{2048};
    std::uint16_t shadow_priority{128};
    float shadow_strength{0.75f};
    float shadow_bias{0.0015f};
    float shadow_normal_bias{0.01f};
    std::uint8_t shadow_filter{1};
    bool contact_shadows{true};
    float contact_shadow_length{0.5f};
    std::uint8_t shadow_cache_mode{};
    std::uint8_t shadow_map_method{};
    std::uint32_t cascade_count{4};
    float shadow_distance{200.0f};
    float cascade_split_lambda{0.65f};
    float cascade_blend_fraction{0.10f};
    bool stable_cascades{true};
    bool use_color_temperature{};
    float temperature_kelvin{6500.0f};

    friend constexpr bool operator==(const host_light_snapshot&, const host_light_snapshot&) noexcept = default;
};

struct host_mesh_renderer_snapshot
{
    std::uint8_t representation{};
    bool visible{true};
    bool casts_shadows{true};
    bool receives_shadows{true};
    float shadow_lod_bias{};
    float maximum_shadow_distance{};
    host_vec4 base_color_tint{1.0f, 1.0f, 1.0f, 1.0f};
    bool has_material{};
    bool asset_backed_material{};
    std::string material_name;
    std::string material_path;

    friend constexpr bool operator==(const host_mesh_renderer_snapshot&,
                                     const host_mesh_renderer_snapshot&) noexcept = default;
};

struct host_terrain_snapshot
{
    bool enabled{true};
    float size{180.0f};
    float minimum_elevation{};
    float maximum_elevation{};
    std::uint32_t resolution{257u};
    std::uint32_t chunk_quads{128u};
    std::uint32_t patch_quads{32u};
    std::uint32_t maximum_hierarchy_depth{};
    float geometric_error_multiplier{1.0f};
    bool receive_shadows{true};
    bool cast_shadows{true};
    float shadow_lod_bias{};
    float maximum_shadow_distance{};
    std::uint64_t content_revision{};
    std::string material_guid;
    std::string material_path;
    std::uint32_t hierarchy_nodes{};
    std::uint32_t hierarchy_depth{};
    std::uint32_t source_patches{};
    std::uint32_t visible_patches{};
    std::uint64_t rendered_triangles{};
    std::uint64_t cpu_memory_bytes{};
    std::uint64_t gpu_memory_bytes{};
    std::uint64_t uploaded_bytes{};
    host_terrain_brush_tool brush_tool{host_terrain_brush_tool::sculpt};
    float brush_radius{6.0f};
    float brush_strength{0.25f};
    float brush_falloff{1.0f};
    std::uint32_t active_layer{};
    std::array<std::string, 4> layer_names{"Grass", "Dirt", "Rock", "Sand"};
    std::array<std::string, 4> layer_base_color_paths{};
};

enum class host_terrain_operation_state : std::uint8_t
{
    queued,
    running,
    completed,
    cancelled,
    failed
};

struct host_terrain_operation_snapshot
{
    std::uint64_t operation_id{};
    host_terrain_operation_state state{host_terrain_operation_state::queued};
    float progress{};
    std::string label;
    std::string message;
    host_entity_id entity{};
};

struct host_terrain_tool_snapshot
{
    host_entity_id entity{};
    bool active{};
    bool hover_visible{};
    host_terrain_brush_tool tool{host_terrain_brush_tool::sculpt};
    float radius{6.0f};
    float strength{0.25f};
    float falloff{1.0f};
    std::uint32_t active_layer{};
};

struct host_prefab_snapshot
{
    struct override_snapshot
    {
        std::string source_entity;
        std::string component_id;
        std::uint64_t field_id{};
        std::string kind;
    };
    std::string prefab_guid;
    std::string prefab_path;
    std::size_t override_count{};
    bool source_missing{};
    std::vector<override_snapshot> overrides;
};

struct host_scene_entity_snapshot
{
    host_entity_id entity{};
    std::string guid;
    std::string parent_guid;
    std::uint32_t sibling_order{};
    std::string name;
    host_entity_kind kind{host_entity_kind::unknown};
    std::string document_guid;
    std::string editor_folder;
    std::string collection;
    std::string layer{"Default"};
    bool active{true};
    bool locked{};
    bool visible{true};
    bool pickable{true};
    std::optional<host_transform> transform;
    std::size_t prefab_override_count{};
    bool selected{};
};

struct host_scene_snapshot
{
    std::vector<host_scene_entity_snapshot> entities;
    std::string scene_guid;
    std::string scene_name;
    std::string active_scene_path;
    std::uint64_t scene_revision{};
    std::uint64_t world_epoch{};
    std::uint64_t frame_revision{};
    std::size_t total_entity_count{};
    std::size_t offset{};
    bool has_more{};
    bool dirty{};
    bool can_undo{};
    bool can_redo{};
    std::string undo_label;
    std::string redo_label;
};

struct host_selected_entity_snapshot
{
    host_entity_id entity{};
    std::size_t selection_count{};
    std::vector<std::string> selected_guids;
    std::string guid;
    std::string name;
    std::string tag;
    bool active{true};
    std::uint32_t render_layer_mask{host_default_render_layer};
    host_mobility mobility{host_mobility::movable};
    std::optional<host_transform> transform;
    std::optional<host_bounds_snapshot> bounds;
    std::optional<host_camera_snapshot> camera;
    std::optional<host_light_snapshot> light;
    std::optional<host_mesh_renderer_snapshot> mesh_renderer;
    std::optional<host_terrain_snapshot> terrain;
    std::optional<host_prefab_snapshot> prefab;
    std::vector<host_component_snapshot> components;
    std::vector<host_project_component_snapshot> project_components;
};

struct host_cloud_layer
{
    bool enabled{true};
    float coverage{};
    float density{};
    float altitude{};
    float thickness{};
    float scale{};
    float detail{};
    float softness{};
    float wind_x{1.0f};
    float wind_y{};
    float wind_speed{};
    float lighting_strength{1.0f};
    float silver_lining{};

    friend constexpr bool operator==(const host_cloud_layer&, const host_cloud_layer&) noexcept = default;
};

struct host_world_environment_snapshot
{
    host_entity_id entity{};
    bool enabled{true};
    bool sky_visible{true};
    bool affect_lighting{true};
    host_sky_source sky_source{host_sky_source::physical_atmosphere};
    host_vec3 solid_color{0.08f, 0.13f, 0.22f};
    std::string hdri_path;
    float hdri_rotation_degrees{};
    float radiance_intensity{1.0f};
    float planet_radius{6360.0f};
    float atmosphere_radius{6420.0f};
    float rayleigh_strength{1.0f};
    float mie_strength{0.35f};
    float ozone_strength{0.15f};
    host_vec3 atmosphere_tint{0.56f, 0.72f, 1.0f};
    host_vec3 ground_albedo{0.18f, 0.18f, 0.18f};
    float mie_anisotropy{0.8f};
    float rayleigh_scale_height{8.0f};
    float mie_scale_height{1.2f};
    float multi_scattering_factor{1.0f};
    float exposure{1.0f};
    float sun_disk_size{0.025f};
    float sun_disk_intensity{1.4f};
    host_sun_position_mode sun_mode{host_sun_position_mode::manual_light};
    host_celestial_time_mode time_mode{host_celestial_time_mode::fixed};
    float latitude_degrees{46.8f};
    float longitude_degrees{8.2f};
    float north_offset_degrees{};
    std::int32_t year{2026};
    std::int32_t month{7};
    std::int32_t day{14};
    float local_time_hours{10.5f};
    float utc_offset_hours{2.0f};
    bool playing{};
    bool loop_day{true};
    float time_scale{60.0f};
    bool automatic_sun_light{true};
    float sun_intensity_multiplier{1.0f};
    float sun_temperature_multiplier{1.0f};
    bool moon_enabled{true};
    bool automatic_moon_phase{true};
    float moon_phase{0.65f};
    float moon_intensity{0.22f};
    float moon_angular_radius_degrees{0.2725f};
    bool stars_enabled{true};
    float star_density{0.42f};
    float star_intensity{0.75f};
    float star_twinkle{0.08f};
    bool clouds_enabled{true};
    bool cloud_shadows{true};
    host_cloud_layer cumulus;
    host_cloud_layer cirrus;
    bool fog_enabled{true};
    host_vec3 fog_color{0.58f, 0.67f, 0.76f};
    float fog_density{0.035f};
    float fog_height_falloff{0.12f};
    float fog_start_distance{8.0f};
    float fog_max_opacity{0.55f};
    float fog_sun_scattering{0.25f};
    bool lighting_enabled{true};
    host_environment_lighting_source lighting_source{host_environment_lighting_source::follow_sky};
    host_vec3 lighting_color{0.18f, 0.23f, 0.29f};
    float diffuse_intensity{1.0f};
    float specular_intensity{1.0f};
    bool indirect_lighting_enabled{true};
    host_indirect_lighting_method indirect_lighting_method{host_indirect_lighting_method::auto_select};
    float indirect_diffuse_intensity{1.0f};
    float reflection_intensity{1.0f};
    float emissive_contribution{1.0f};
    float maximum_trace_distance{100.0f};
    float surface_cache_detail{1.0f};
    bool allow_hardware_ray_tracing{true};

    friend bool operator==(const host_world_environment_snapshot&,
                           const host_world_environment_snapshot&) noexcept = default;
};

struct host_asset_snapshot
{
    std::string guid;
    std::string path;
    std::string scope{"project"};
    std::string kind;
    std::string type_id;
    std::string importer_id;
    std::string state;
    std::string residency;
    std::uint64_t generation{};
    std::uint32_t strong_references{};
    std::uint32_t pins{};
    std::string diagnostic;
    std::vector<std::string> dependencies;
    std::vector<std::string> reverse_dependencies;
    bool read_only{};
    bool imported{};
    bool import_running{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::string texture_format;
    std::uint32_t mip_count{};
    std::uint32_t tile_count{};
    std::string streaming_mode{"resident"};
    std::uint32_t settings_version{1};
    std::uint64_t artifact_size{};
    std::string streaming_eligibility_error;
};

struct host_project_assets_snapshot
{
    std::string project_name;
    std::filesystem::path project_root;
    std::filesystem::path asset_root;
    std::string default_mesh_path;
    bool default_mesh_loaded{};
    std::string default_mesh_message;
    std::filesystem::path cache_root;
    std::uint64_t cache_local_bytes{};
    std::uint64_t cache_local_hits{};
    std::uint64_t cache_local_misses{};
    std::uint64_t cache_shared_hits{};
    std::uint64_t cache_shared_misses{};
    std::uint64_t cache_corrupt_entries{};
    std::uint64_t cache_evictions{};
    double cache_hit_rate{};
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
    std::filesystem::path descriptor_path;
    std::vector<std::filesystem::path> content_roots;
    std::vector<std::filesystem::path> builtin_content_roots;
    std::filesystem::path cache_root;
    std::filesystem::path default_scene;
    std::string project_guid;
    std::string engine_version;
    std::string editor_module_id;
    std::filesystem::path editor_module_path;
    bool read_only{};
};

struct host_close_project_command
{
};

struct host_reload_project_module_command
{
    std::filesystem::path path;
    std::string engine_version;
    std::string project_guid;
    std::string module_id;
};

struct host_open_scene_command
{
    std::filesystem::path path;
    bool append{};
};

struct host_new_scene_command
{
    std::string name{"Untitled"};
};
struct host_save_scene_command
{
};
struct host_save_scene_as_command
{
    std::filesystem::path path;
};
struct host_autosave_scene_command
{
    std::filesystem::path path;
};
struct host_open_recovery_scene_command
{
    std::filesystem::path path;
    std::filesystem::path original_path;
};
struct host_asset_reimport_command
{
    std::string guid;
};
struct host_set_texture_streaming_mode_command
{
    std::string guid;
    std::string mode;
};

struct host_patch_texture_settings_command
{
    std::string guid;
    std::string preset;
    std::string semantic;
    std::string color_space;
    std::string streaming_mode;
    std::string compression;
    std::string power_of_two;
    std::string min_filter;
    std::string mag_filter;
    std::string mip_filter;
    std::string wrap_u;
    std::string wrap_v;
    std::string mip_generation_filter;
    std::string channel_r;
    std::string channel_g;
    std::string channel_b;
    std::string channel_a;
    std::string curve_master;
    std::string curve_r;
    std::string curve_g;
    std::string curve_b;
    std::string curve_a;
    std::optional<std::uint32_t> max_size;
    std::optional<float> anisotropy;
    std::optional<float> lod_bias;
    std::optional<float> minimum_lod;
    std::optional<float> maximum_lod;
    std::optional<float> alpha_coverage_threshold;
    std::optional<float> mip_sharpen;
    std::optional<float> deband_strength;
    std::optional<bool> dither_mips;
    std::optional<bool> deband_mips;
    std::optional<float> brightness;
    std::optional<float> gamma;
    std::optional<float> contrast;
    std::optional<float> saturation;
    std::optional<float> vibrance;
    std::optional<float> tint_r;
    std::optional<float> tint_g;
    std::optional<float> tint_b;
    std::optional<float> input_black;
    std::optional<float> input_white;
    std::optional<float> output_black;
    std::optional<float> output_white;
    std::optional<bool> invert_r;
    std::optional<bool> invert_g;
    std::optional<bool> invert_b;
    std::optional<bool> invert_a;
    std::optional<bool> curves_enabled;
    std::optional<bool> generate_mips;
    std::optional<bool> preserve_alpha_coverage;
};
struct host_shader_compile_command
{
    std::filesystem::path path;
    std::string source;
    std::string entry_point{"main"};
    std::string stage{"fragment"};
    std::string domain{"surface"};
};
struct host_asset_cancel_import_command
{
    std::string guid;
};
struct host_asset_move_command
{
    std::string guid;
    std::filesystem::path path;
};
struct host_asset_rename_command
{
    std::string guid;
    std::string name;
};

struct host_create_entity_command
{
    host_create_entity_kind kind{host_create_entity_kind::cube};
    host_entity_id parent{};
};

struct host_delete_entity_command
{
    host_entity_id entity{};
};

struct host_duplicate_entity_command
{
    host_entity_id entity{};
};
struct host_create_prefab_command
{
    host_entity_id entity{};
    std::filesystem::path path;
};
struct host_instantiate_prefab_command
{
    std::filesystem::path path;
    host_entity_id parent{};
};
struct host_apply_prefab_command
{
    host_entity_id entity{};
};
struct host_revert_prefab_command
{
    host_entity_id entity{};
};
struct host_unpack_prefab_command
{
    host_entity_id entity{};
};
struct host_revert_prefab_override_command
{
    host_entity_id entity{};
    std::string source_entity;
    std::string component_id;
    std::uint64_t field_id{};
    std::string kind{"field"};
};
struct host_reparent_entity_command
{
    host_entity_id entity{};
    host_entity_id parent{};
    host_entity_id before_sibling{};
    bool preserve_world{true};
};
struct host_reorder_entity_command
{
    host_entity_id entity{};
    host_entity_id before_sibling{};
};

struct host_rename_entity_command
{
    host_entity_id entity{};
    std::string name;
};

struct host_select_entity_command
{
    host_entity_id entity{};
    bool additive{};
    bool toggle{};
};

struct host_clear_selection_command
{
};

struct host_set_active_command
{
    host_entity_id entity{};
    bool active{true};
    bool apply_to_selection{};
};

struct host_set_tag_command
{
    host_entity_id entity{};
    std::string tag;
    bool apply_to_selection{};
};

struct host_set_transform_command
{
    host_entity_id entity{};
    host_transform transform;
    bool apply_to_selection{};
};

struct host_set_render_layer_command
{
    host_entity_id entity{};
    std::uint32_t render_layer_mask{host_default_render_layer};
    bool apply_to_selection{};
};

struct host_set_mobility_command
{
    host_entity_id entity{};
    host_mobility mobility{host_mobility::movable};
    bool apply_to_selection{};
};

struct host_set_camera_command
{
    host_entity_id entity{};
    host_camera_snapshot camera;
    bool apply_to_selection{};
};

struct host_set_light_command
{
    host_entity_id entity{};
    host_light_snapshot light;
    bool apply_to_selection{};
};

struct host_set_mesh_renderer_command
{
    host_entity_id entity{};
    std::uint8_t representation{};
    bool visible{true};
    bool casts_shadows{true};
    bool receives_shadows{true};
    float shadow_lod_bias{};
    float maximum_shadow_distance{};
    host_vec4 base_color_tint{1.0f, 1.0f, 1.0f, 1.0f};
    bool apply_to_selection{};
};

struct host_set_terrain_command
{
    host_entity_id entity{};
    bool enabled{true};
    bool receive_shadows{true};
    bool cast_shadows{true};
    std::uint32_t patch_quads{32u};
    std::uint32_t maximum_hierarchy_depth{};
    float geometric_error_multiplier{1.0f};
    float shadow_lod_bias{};
    float maximum_shadow_distance{};
};

struct host_set_terrain_brush_command
{
    host_entity_id entity{};
    host_terrain_brush_tool tool{host_terrain_brush_tool::sculpt};
    float radius{6.0f};
    float strength{0.25f};
    float falloff{1.0f};
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
    host_edit_phase phase{host_edit_phase::begin};
    bool invert{};
    float elapsed_seconds{1.0f / 60.0f};
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
    bool apply_to_selection{};
};

enum class host_component_operation : std::uint8_t
{
    add,
    remove,
    reset
};

struct host_component_operation_command
{
    host_component_operation operation{host_component_operation::reset};
    std::string component;
};

struct host_patch_project_component_command
{
    std::string component;
    std::string field;
    std::string value_json;
};

struct host_set_world_environment_command
{
    host_world_environment_snapshot environment;
};

struct host_apply_world_environment_preset_command
{
    host_entity_id entity{};
    host_world_environment_preset preset{host_world_environment_preset::alpine_late_morning};
};

struct host_set_environment_hdri_command
{
    host_entity_id entity{};
    std::filesystem::path path;
};

struct host_set_camera_projection_command
{
    host_camera_projection projection{host_camera_projection::perspective};
};

struct host_viewport_attach_command
{
    std::string viewport_id{"viewport-1"};
    std::uint64_t native_handle{};
    std::int32_t x{};
    std::int32_t y{};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct host_viewport_resize_command
{
    std::string viewport_id{"viewport-1"};
    std::int32_t x{};
    std::int32_t y{};
    std::uint32_t width{};
    std::uint32_t height{};
};

enum class host_viewport_output_type : std::uint8_t
{
    native_window,
    shared_texture
};

struct host_viewport_create_command
{
    std::string viewport_id{"viewport-1"};
    host_viewport_output_type output{host_viewport_output_type::shared_texture};
    std::uint64_t consumer_process_id{};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct host_create_terrain_command
{
    float size{180.0f};
    float minimum_elevation{};
    float maximum_elevation{48.0f};
    std::uint32_t resolution{257u};
    std::uint32_t patch_quads{32u};
    std::string source{"flat"};
    std::uint64_t seed{1u};
    host_entity_id parent{};
};

struct host_generate_terrain_command
{
    host_entity_id entity{};
    std::string generator_id{"arc.terrain.domain_warped.v1"};
    std::uint64_t seed{1u};
    float minimum_elevation{};
    float maximum_elevation{48.0f};
};

struct host_import_terrain_heightmap_command
{
    host_entity_id entity{};
    std::filesystem::path path;
    std::uint32_t raw_width{};
    std::uint32_t raw_height{};
    std::uint32_t target_resolution{257u};
    float physical_size{180.0f};
    float minimum_elevation{};
    float maximum_elevation{48.0f};
    bool flip_x{};
    bool flip_z{};
    bool normalize{true};
};

struct host_export_terrain_heightmap_command
{
    host_entity_id entity{};
    std::filesystem::path path;
    float minimum_elevation{};
    float maximum_elevation{48.0f};
};

struct host_resample_terrain_command
{
    host_entity_id entity{};
    std::uint32_t resolution{257u};
    float physical_size{180.0f};
};

struct host_cancel_terrain_operation_command
{
    std::uint64_t operation_id{};
};

struct host_viewport_detach_command
{
    std::string viewport_id{"viewport-1"};
};

struct host_viewport_frame_released_command
{
    std::string viewport_id{"viewport-1"};
    std::uint64_t generation{};
    std::uint64_t frame_id{};
    std::string consumer_handle;
};

struct host_viewport_set_visibility_command
{
    std::string viewport_id{"viewport-1"};
    bool visible{true};
};

enum class host_viewport_pointer_phase : std::uint8_t
{
    down,
    move,
    up,
    wheel,
    leave,
    cancel
};

struct host_viewport_pointer_command
{
    std::string viewport_id{"viewport-1"};
    host_viewport_pointer_phase phase{host_viewport_pointer_phase::move};
    std::int32_t x{};
    std::int32_t y{};
    std::int32_t button{};
    float wheel{};
    bool alt{};
    bool shift{};
    bool control{};
};

struct host_viewport_key_command
{
    std::string viewport_id{"viewport-1"};
    std::string key;
    bool down{true};
    bool repeat{};
    bool alt{};
    bool shift{};
    bool control{};
};

struct host_viewport_set_camera_mode_command
{
    std::string viewport_id{"viewport-1"};
    host_camera_projection projection{host_camera_projection::perspective};
};

struct host_viewport_set_render_options_command
{
    std::string viewport_id{"viewport-1"};
    host_render_mode render_mode{host_render_mode::shaded};
    host_visualization_mode visualization{host_visualization_mode::standard};
    host_overlay_mode overlay{host_overlay_mode::selected_wireframe};
    bool shadows{true};
    bool grid{true};
    bool realtime{true};
    float camera_speed{4.0f};
    host_camera_anti_aliasing anti_aliasing{host_camera_anti_aliasing::inherit};
    host_environment_visibility environment{};
};

struct host_viewport_camera_input_command
{
    std::string viewport_id{"viewport-1"};
    float orbit_x{};
    float orbit_y{};
    float look_x{};
    float look_y{};
    float pan_x{};
    float pan_y{};
    float forward{};
    float move_right{};
    float move_up{};
    float move_forward{};
    float zoom{};
    bool focus_selected{};
};
struct host_viewport_set_pose_command
{
    std::string viewport_id{"viewport-1"};
    host_vec3 position{};
    host_vec3 target{};
};

struct host_history_undo_command
{
};
struct host_history_redo_command
{
};
struct host_history_begin_transaction_command
{
    std::uint64_t id{};
    std::string label;
};
struct host_history_commit_transaction_command
{
    std::uint64_t id{};
};
struct host_history_cancel_transaction_command
{
    std::uint64_t id{};
};
struct host_runtime_resume_command
{
};
struct host_runtime_pause_command
{
};
struct host_runtime_stop_command
{
};
struct host_runtime_step_command
{
    std::uint32_t ticks{1};
};
struct host_runtime_set_time_scale_command
{
    double value{1.0};
};
struct host_runtime_capture_snapshot_command
{
    std::string label;
};
struct host_runtime_restore_snapshot_command
{
    std::uint64_t snapshot_id{};
};
struct host_viewport_set_tool_command
{
    host_viewport_tool tool{host_viewport_tool::select};
    host_coordinate_space coordinate_space{host_coordinate_space::world};
    bool snapping{};
    float translation_snap{0.25f};
    float rotation_snap_degrees{15.0f};
    float scale_snap{0.1f};
};
struct host_viewport_pick_command
{
    std::string viewport_id{"viewport-1"};
    std::uint32_t x{};
    std::uint32_t y{};
};
struct host_viewport_capture_command
{
    std::uint64_t capture_id{};
    bool color{true};
    bool depth{true};
    bool object_id{true};
    bool normals{true};
    bool scene_color{};
    bool base_color{};
    bool material_properties{};
    bool emissive{};
    bool indirect_diffuse{};
    bool reflections{};
    bool trace_source{};
    bool distance_field{};
    bool temporal_confidence{};
};

using host_command_payload = std::variant<
    host_open_project_command, host_close_project_command, host_reload_project_module_command, host_open_scene_command,
    host_new_scene_command, host_save_scene_command, host_save_scene_as_command, host_autosave_scene_command,
    host_open_recovery_scene_command, host_asset_reimport_command, host_set_texture_streaming_mode_command,
    host_patch_texture_settings_command, host_shader_compile_command, host_asset_cancel_import_command,
    host_asset_move_command, host_asset_rename_command, host_create_entity_command, host_delete_entity_command,
    host_duplicate_entity_command, host_create_prefab_command, host_instantiate_prefab_command,
    host_apply_prefab_command, host_revert_prefab_command, host_unpack_prefab_command,
    host_revert_prefab_override_command, host_reparent_entity_command, host_reorder_entity_command,
    host_rename_entity_command, host_select_entity_command, host_clear_selection_command, host_set_active_command,
    host_set_tag_command, host_set_transform_command, host_set_render_layer_command, host_set_mobility_command,
    host_set_camera_command, host_set_light_command, host_set_mesh_renderer_command, host_set_terrain_command,
    host_set_terrain_brush_command, host_set_terrain_layer_command, host_create_terrain_command,
    host_generate_terrain_command, host_import_terrain_heightmap_command, host_export_terrain_heightmap_command,
    host_resample_terrain_command, host_cancel_terrain_operation_command, host_terrain_stroke_command,
    host_terrain_hover_command, host_set_entity_material_command, host_component_operation_command,
    host_patch_project_component_command, host_set_world_environment_command,
    host_apply_world_environment_preset_command, host_set_environment_hdri_command, host_set_camera_projection_command,
    host_viewport_attach_command, host_viewport_create_command, host_viewport_resize_command,
    host_viewport_detach_command, host_viewport_frame_released_command, host_viewport_set_visibility_command,
    host_viewport_pointer_command, host_viewport_key_command, host_viewport_set_camera_mode_command,
    host_viewport_set_render_options_command, host_viewport_camera_input_command, host_viewport_set_pose_command,
    host_history_undo_command, host_history_redo_command, host_history_begin_transaction_command,
    host_history_commit_transaction_command, host_history_cancel_transaction_command, host_runtime_resume_command,
    host_runtime_pause_command, host_runtime_stop_command, host_runtime_step_command,
    host_runtime_set_time_scale_command, host_runtime_capture_snapshot_command, host_runtime_restore_snapshot_command,
    host_viewport_set_tool_command, host_viewport_pick_command, host_viewport_capture_command>;

struct host_edit_transaction
{
    std::uint64_t id{};
    host_edit_phase phase{host_edit_phase::none};
    std::string label;
};

struct host_command_envelope
{
    std::uint64_t request_id{};
    std::string command_type;
    host_command_payload payload{host_close_project_command{}};
    std::optional<host_edit_transaction> edit;
    std::optional<std::uint64_t> expected_scene_revision;
};

struct host_scene_hierarchy_query
{
};

struct host_selected_entity_query
{
};

struct host_scene_entities_query
{
    std::string search;
    std::size_t offset{};
    std::size_t limit{100};
};

struct host_entity_by_guid_query
{
    std::string guid;
};

enum class host_spatial_query_kind : std::uint8_t
{
    raycast,
    nearby,
    bounds,
    frustum
};

struct host_scene_spatial_query
{
    host_spatial_query_kind kind{host_spatial_query_kind::nearby};
    host_vec3 origin{};
    host_vec3 direction{0.0f, 0.0f, -1.0f};
    host_vec3 center{};
    host_vec3 extent{1.0f, 1.0f, 1.0f};
    float radius{10.0f};
    std::size_t limit{100};
};

struct host_component_schema_query
{
};

struct host_workspace_documents_query
{
};

struct host_gateway_diagnostics_query
{
};

struct host_viewport_capture_query
{
    std::uint64_t capture_id{};
};

struct host_project_assets_query
{
};

struct host_asset_thumbnail_query
{
    std::string path;
    std::uint32_t max_size{96};
};

struct host_texture_settings_query
{
    std::string guid;
};

struct host_viewport_state_query
{
    std::string viewport_id{"viewport-1"};
};

struct host_world_environment_query
{
    host_entity_id entity{};
};
struct host_history_state_query
{
};
struct host_runtime_state_query
{
};
struct host_terrain_tool_state_query
{
};

using host_query_payload =
    std::variant<host_scene_hierarchy_query, host_selected_entity_query, host_scene_entities_query,
                 host_entity_by_guid_query, host_scene_spatial_query, host_component_schema_query,
                 host_workspace_documents_query, host_gateway_diagnostics_query, host_viewport_capture_query,
                 host_project_assets_query, host_asset_thumbnail_query, host_texture_settings_query,
                 host_viewport_state_query, host_world_environment_query, host_history_state_query,
                 host_runtime_state_query, host_terrain_tool_state_query>;

struct host_query_envelope
{
    std::uint64_t request_id{};
    std::string query_type;
    host_query_payload payload{host_scene_hierarchy_query{}};
};

struct host_response
{
    std::uint64_t request_id{};
    bool succeeded{};
    std::string error;
    std::string payload_json;
    std::uint64_t scene_revision{};
    std::uint64_t world_epoch{};
    std::uint64_t frame_revision{};
};

struct host_viewport_request
{
    std::string viewport_id{"viewport-1"};
    std::uint64_t frame_index{};
    std::uint32_t width{};
    std::uint32_t height{};
    host_render_mode render_mode{host_render_mode::shaded};
    host_visualization_mode visualization{host_visualization_mode::standard};
    host_overlay_mode overlay{host_overlay_mode::selected_wireframe};
    bool shadows{true};
    bool grid{true};
    bool realtime{true};
    float camera_speed{4.0f};
    host_camera_anti_aliasing anti_aliasing{host_camera_anti_aliasing::inherit};
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
const char* to_string(host_mobility value) noexcept;
const char* to_string(host_render_mode value) noexcept;
const char* to_string(host_visualization_mode value) noexcept;
const char* to_string(host_overlay_mode value) noexcept;
const char* to_string(host_sky_source value) noexcept;
const char* to_string(host_sun_position_mode value) noexcept;
const char* to_string(host_celestial_time_mode value) noexcept;
const char* to_string(host_environment_lighting_source value) noexcept;
const char* to_string(host_indirect_lighting_method value) noexcept;
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
