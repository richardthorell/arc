#pragma once

#include <arc/render/material.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace arc::render
{

/**
 * @brief Static mesh vertex used by the first renderer mesh path.
 */
struct mesh_vertex
{
    float position[3]{};
    float normal[3]{};
    float texcoord[2]{};
    float color[4]{ 1.0f, 1.0f, 1.0f, 1.0f };
};

/**
 * @brief CPU-side mesh data ready for GPU upload.
 */
struct mesh_data
{
    std::string name;
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
};

/**
 * @brief Imported material plus texture index references before renderer handles exist.
 */
struct material_import
{
    material_desc material;
    material_texture_indices textures;
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
 * @brief Load the first static triangle mesh from a glTF binary file.
 */
mesh_load_result load_gltf_mesh(const std::filesystem::path& path);

} // namespace arc::render
