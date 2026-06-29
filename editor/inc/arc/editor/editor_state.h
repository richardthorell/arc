#pragma once

#include <arc/render/render.h>
#include <arc/scene/scene.h>

#include <filesystem>
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

struct editor_scene_state
{
    scene::registry scene;
    render::mesh_handle default_mesh;
    render::material_handle default_material;
    std::vector<render::texture_handle> default_textures;
    scene::entity camera_entity;
    scene::entity sun_entity;
    scene::entity mesh_entity;
    scene::entity selected_entity;
    scene::render_scene_result last_render;
    bool mesh_uploaded{};
    bool camera_created{};
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

} // namespace arc::editor
