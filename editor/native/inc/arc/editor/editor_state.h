#pragma once

#include <arc/render/render.h>
#include <arc/scene/scene.h>
#include <arc/jobs/jobs.h>
#include <arc/editor/material_library.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <cstdint>
#include <string>
#include <vector>

namespace arc::editor
{

struct editor_asset_state
{
    std::filesystem::path root;
    std::filesystem::path default_mesh_path;
    std::filesystem::path default_vertex_shader_path;
    std::filesystem::path default_fragment_shader_path;
    render::mesh_data default_mesh;
    std::vector<render::texture_data> default_textures;
    std::vector<render::material_import> default_materials;
    std::string default_mesh_message;
    bool default_mesh_loaded{};
};

editor_asset_state load_default_editor_assets(const std::filesystem::path& asset_root);

struct editor_scene_state
{
    scene::registry scene;
    render::mesh_handle default_mesh;
    render::material_handle default_material;
    render::material_handle primitive_material;
    render::material_handle terrain_material;
    render::material_desc terrain_material_desc;
    std::array<std::filesystem::path, 4> terrain_layer_paths{};
    render::material_handle water_material;
    render::material_handle vegetation_material;
    render::environment_handle environment_lighting_resource;
    std::filesystem::path world_environment_hdri_path;
    std::vector<render::texture_handle> default_textures;
    scene::entity camera_entity;
    scene::entity game_camera_entity;
    scene::entity sun_entity;
    scene::entity world_environment_entity;
    scene::entity mesh_entity;
    scene::entity terrain_entity;
    scene::entity water_entity;
    scene::entity vegetation_entity;
    scene::entity selected_entity;
    std::vector<scene::entity> primitive_entities;
    std::vector<scene::entity> imported_scene_entities;
    std::vector<scene::entity> world_feature_entities;
    struct asset_binding
    {
        scene::entity_guid entity;
        std::string source_kind;
        std::filesystem::path source_path;
        std::string subresource;
        std::filesystem::path material_path;
    };
    scene::entity_guid scene_guid{};
    std::string scene_name{ "Untitled" };
    std::filesystem::path active_scene_path;
    std::vector<asset_binding> asset_bindings;
    std::vector<std::pair<scene::entity_guid, std::string>> unknown_component_records;
    editor_material_library material_library;
    material_editor_state material_editor;
    scene::render_scene_result last_render;
    std::uint32_t primitive_serial{};
    bool mesh_uploaded{};
    bool focus_imported_scene_requested{};
};

void ensure_scene_authoring_metadata(editor_scene_state& scene);
scene::entity find_entity_by_guid(const editor_scene_state& scene, scene::entity_guid guid) noexcept;
scene::entity_guid entity_guid_of(const editor_scene_state& scene, scene::entity entity) noexcept;
editor_scene_state::asset_binding* find_asset_binding(editor_scene_state& scene, scene::entity_guid guid) noexcept;
const editor_scene_state::asset_binding* find_asset_binding(const editor_scene_state& scene, scene::entity_guid guid) noexcept;

enum class editor_scene_open_mode : std::uint8_t
{
    replace,
    append
};

struct editor_scene_open_result
{
    bool succeeded{};
    std::size_t entity_count{};
    std::string message;
};

enum class editor_scene_import_status : std::uint8_t
{
    idle,
    running,
    succeeded,
    failed,
    cancelled
};

struct editor_scene_import_shared_state
{
    mutable std::mutex mutex;
    float progress{};
    render::scene_import_stage stage{ render::scene_import_stage::loading };
    std::string message;
    std::vector<std::string> diagnostics;
    std::atomic_bool cancel_requested{};
};

struct editor_scene_import_state
{
    editor_scene_import_status status{ editor_scene_import_status::idle };
    editor_scene_open_mode mode{ editor_scene_open_mode::replace };
    std::filesystem::path source_path;
    std::shared_ptr<editor_scene_import_shared_state> shared;
    cancellation_source cancellation;
    job_future<render::scene_import_result> task;
    render::scene_import_result result;
    bool modal_open{};
    bool result_ready{};
};

enum class editor_primitive_type : std::uint8_t
{
    plane,
    cube,
    sphere,
    cylinder
};

struct editor_project_state
{
    std::string name;
    std::filesystem::path root;
};

struct editor_build_state
{
    std::string configuration;
    std::string platform;
    bool background_job_running{};
    float background_job_progress{};
};

struct editor_source_control_state
{
    std::string branch;
    bool has_changes{};
    bool available{};
};

struct editor_metrics_state
{
    float fps{};
    float frame_ms{};
    std::size_t draw_calls{};
    bool gpu_time_available{};
    float gpu_ms{};
};

const char* selected_entity_name(const editor_scene_state& scene, const char* fallback = "None");

const char* primitive_type_name(editor_primitive_type type) noexcept;

scene::entity add_primitive_to_scene(
    editor_scene_state& scene,
    render::renderer& renderer,
    editor_primitive_type type);

scene::entity add_world_environment_to_scene(editor_scene_state& scene);

scene::entity add_terrain_to_scene(
    editor_scene_state& scene,
    render::renderer& renderer,
    render::material_handle material = {});

render::material_handle create_default_terrain_material(
    editor_scene_state& scene,
    render::renderer& renderer,
    const std::filesystem::path& asset_root);

bool rebuild_terrain_chunks(
    editor_scene_state& scene,
    render::renderer& renderer,
    scene::entity entity,
    const scene::terrain_dirty_region* dirty_region = nullptr);

scene::entity add_water_to_scene(editor_scene_state& scene, render::renderer& renderer);

scene::entity add_grass_patch_to_scene(editor_scene_state& scene, render::renderer& renderer);

scene::entity add_decal_to_scene(editor_scene_state& scene);

editor_scene_open_result open_scene_asset_in_editor(
    editor_scene_state& scene,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    editor_scene_open_mode mode);

editor_scene_open_result apply_scene_import_result_to_editor(
    editor_scene_state& scene,
    render::renderer& renderer,
    const std::filesystem::path& source_path,
    render::scene_import_result imported,
    editor_scene_open_mode mode);

bool start_scene_import(
    editor_scene_import_state& state,
    job_system& jobs,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    editor_scene_open_mode mode);

bool poll_scene_import(editor_scene_import_state& state);

void reset_scene_import(editor_scene_import_state& state);

const char* scene_import_stage_label(render::scene_import_stage stage) noexcept;

} // namespace arc::editor
