#pragma once

#include <arc/math/quaternion.h>
#include <arc/math/vector.h>
#include <arc/render/material.h>
#include <arc/jobs/jobs.h>

#include <cstdint>
#include <atomic>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace arc::render
{

enum class mesh_usage : std::uint8_t
{
    static_gpu,
    dynamic_per_frame
};

/**
 * @brief Static mesh vertex used by the first renderer mesh path.
 */
struct mesh_vertex
{
    float position[3]{};
    float normal[3]{};
    float tangent[4]{ 1.0f, 0.0f, 0.0f, 1.0f };
    float texcoord[2]{};
    float color[4]{ 1.0f, 1.0f, 1.0f, 1.0f };
};

/**
 * @brief CPU-side mesh data ready for GPU upload.
 */
struct mesh_data
{
    std::string name;
    mesh_usage usage{ mesh_usage::static_gpu };
    std::vector<mesh_vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::size_t material_index{ std::numeric_limits<std::size_t>::max() };
};

/**
 * @brief Texture indices referenced by one imported glTF material.
 */
struct material_texture_indices
{
    static constexpr std::size_t invalid = std::numeric_limits<std::size_t>::max();

    std::size_t base_color{ invalid };
    std::size_t metallic_roughness{ invalid };
    std::size_t normal{ invalid };
    std::size_t occlusion{ invalid };
    std::size_t emissive{ invalid };
    std::size_t clear_coat{ invalid };
    std::size_t clear_coat_roughness{ invalid };
    std::size_t clear_coat_normal{ invalid };
    std::size_t anisotropy{ invalid };
    std::size_t thickness{ invalid };
    std::size_t transmission{ invalid };
};

/**
 * @brief Asset-root-relative texture paths for an imported material.
 */
struct material_texture_paths
{
    std::string base_color;
    std::string metallic_roughness;
    std::string normal;
    std::string occlusion;
    std::string emissive;
    std::string clear_coat;
    std::string clear_coat_roughness;
    std::string clear_coat_normal;
    std::string anisotropy;
    std::string thickness;
    std::string transmission;
};

/**
 * @brief Imported material plus texture index references before renderer handles exist.
 */
struct material_import
{
    material_desc material;
    material_texture_indices textures;
    material_texture_paths texture_paths;
    std::filesystem::path asset_path;
};

/**
 * @brief High-level scene import stage for progress reporting.
 */
enum class scene_import_stage : std::uint8_t
{
    loading,
    extracting_textures,
    building_materials,
    building_meshes,
    finalizing
};

/**
 * @brief Progress snapshot reported by a scene importer.
 */
struct scene_import_progress
{
    scene_import_stage stage{ scene_import_stage::loading };
    float progress{};
    std::string message;
};

/**
 * @brief Callback invoked by long-running scene importers.
 *
 * Return false to request cancellation.
 */
using scene_import_progress_callback = std::function<bool(const scene_import_progress&)>;

/**
 * @brief Scene import settings.
 */
struct scene_import_options
{
    std::filesystem::path asset_root;
    std::filesystem::path import_directory;
    bool copy_assets{ true };
    bool normalize_axes{ true };
    bool normalize_units{ true };
    const std::atomic_bool* cancel_requested{};
};

/**
 * @brief Result from loading a mesh file.
 */
struct mesh_load_result
{
    mesh_data mesh;
    std::vector<texture_data> textures;
    std::vector<material_import> materials;
    std::string message;

    /**
     * @brief Return whether mesh data was loaded.
     */
    bool succeeded() const noexcept
    {
        return !mesh.vertices.empty() && !mesh.indices.empty();
    }
};

/**
 * @brief Imported scene node that references CPU mesh/material records.
 */
struct scene_import_node
{
    std::string name;
    std::size_t mesh_index{ std::numeric_limits<std::size_t>::max() };
    std::size_t material_index{ std::numeric_limits<std::size_t>::max() };
    math::vector3f position{};
    math::quatf rotation{};
    math::vector3f scale{ 1.0f, 1.0f, 1.0f };
};

/**
 * @brief Imported static scene data before renderer handles are created.
 */
struct scene_import_result
{
    std::vector<mesh_data> meshes;
    std::vector<texture_data> textures;
    std::vector<material_import> materials;
    std::vector<scene_import_node> nodes;
    std::vector<std::string> diagnostics;
    std::filesystem::path import_directory;
    std::filesystem::path manifest_path;
    std::string message;

    /**
     * @brief Return whether any renderable nodes were imported.
     */
    bool succeeded() const noexcept
    {
        return !meshes.empty() && !nodes.empty();
    }
};

/**
 * @brief Load the first static triangle mesh from a glTF binary file.
 */
mesh_load_result load_gltf_mesh(const std::filesystem::path& path);

job_future<mesh_load_result> load_gltf_mesh_async(
    job_system& jobs,
    std::filesystem::path path,
    cancellation_token cancellation = {});

/**
 * @brief Load a supported static scene asset by extension.
 *
 * GLB files use the existing mesh importer and are wrapped as a one-node scene.
 * FBX files require the optional ufbx dependency; without it this returns a
 * clear unsupported-format diagnostic.
 */
scene_import_result load_scene_asset(
    const std::filesystem::path& path,
    const scene_import_options& options,
    scene_import_progress_callback progress = {});

/**
 * @brief Load a supported static scene asset by extension with default options.
 */
scene_import_result load_scene_asset(const std::filesystem::path& path);

job_future<scene_import_result> load_scene_asset_async(
    job_system& jobs,
    std::filesystem::path path,
    scene_import_options options = {},
    scene_import_progress_callback progress = {},
    cancellation_token cancellation = {});

} // namespace arc::render
