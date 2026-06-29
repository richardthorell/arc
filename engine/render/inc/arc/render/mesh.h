#pragma once

#include <cstdint>
#include <filesystem>
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
};

/**
 * @brief Result from loading a mesh file.
 */
struct mesh_load_result
{
    mesh_data mesh;
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
