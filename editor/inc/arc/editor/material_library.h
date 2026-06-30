#pragma once

#include <arc/editor/material_asset.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#include <filesystem>
#include <string>
#include <vector>

namespace arc::editor
{

struct editor_ray;

enum class material_texture_slot
{
    base_color,
    metallic_roughness,
    normal,
    ao,
    emissive,
    height
};

struct editor_material_record
{
    std::filesystem::path path;
    material_asset asset;
    render::material_handle material;
};

struct material_editor_state
{
    bool open{};
    bool dirty{};
    material_asset working;
    material_asset saved;
    render::material_handle material;
    render::mesh_handle preview_sphere;
    render::mesh_handle preview_plane;
    render::mesh_handle preview_cube;
    int preview_shape{};
};

struct editor_material_library
{
    std::vector<editor_material_record> materials;
    std::vector<std::pair<std::filesystem::path, render::texture_handle>> textures;
};

bool is_material_asset_path(const std::filesystem::path& path);

bool is_texture_asset_path(const std::filesystem::path& path);

bool assign_texture_to_material_slot(
    material_editor_state& editor,
    material_texture_slot slot,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& texture_path,
    std::string* message = nullptr);

bool create_default_material_asset(
    const std::filesystem::path& path,
    const std::filesystem::path& asset_root,
    std::string& message);

render::material_handle load_material_for_editor(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    material_asset* out_asset = nullptr);

bool open_material_editor(
    material_editor_state& editor,
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    std::string& message);

bool save_material_editor(
    material_editor_state& editor,
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    std::string& message);

bool update_material_editor_live_material(
    material_editor_state& editor,
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    std::string* message = nullptr);

bool apply_material_to_selected(
    scene::registry& scene,
    scene::entity selected,
    render::material_handle material);

bool apply_material_asset_to_entity(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& material_path,
    scene::registry& scene,
    scene::entity entity,
    std::string* message = nullptr);

scene::entity apply_material_asset_to_viewport_hit(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& material_path,
    scene::registry& scene,
    const editor_ray& ray,
    scene::entity& selected,
    std::string* message = nullptr);

} // namespace arc::editor
