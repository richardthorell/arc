#include <arc/render/mesh_artifact.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iterator>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace arc::render
{
namespace
{
constexpr std::uint32_t max_meshes = 65536;
constexpr std::uint32_t max_lods = 64;
constexpr std::uint64_t max_name_bytes = 4096;
constexpr std::string_view magic = "ARC_MESH_2";

static_assert(std::is_trivially_copyable_v<mesh_vertex>);
static_assert(std::is_trivially_copyable_v<std::uint32_t>);

struct reader
{
    std::span<const std::byte> bytes;
    std::size_t offset{};

    template <class T> bool read(T& output)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (sizeof(T) > bytes.size() - offset) return false;
        std::memcpy(&output, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return true;
    }

    bool read_string(std::string& output)
    {
        std::uint64_t length{};
        if (!read(length) || length > max_name_bytes || length > bytes.size() - offset) return false;
        output.assign(reinterpret_cast<const char*>(bytes.data() + offset), static_cast<std::size_t>(length));
        offset += static_cast<std::size_t>(length);
        return true;
    }

    bool read_blob(std::span<const std::byte>& output, std::size_t element_size)
    {
        std::uint64_t length{};
        if (!read(length) || length > bytes.size() - offset || length % element_size != 0) return false;
        output = bytes.subspan(offset, static_cast<std::size_t>(length));
        offset += static_cast<std::size_t>(length);
        return true;
    }
};

bool valid_vertex(const mesh_vertex& vertex) noexcept
{
    const auto finite = [](float value) { return std::isfinite(value); };
    return std::all_of(std::begin(vertex.position), std::end(vertex.position), finite) &&
           std::all_of(std::begin(vertex.normal), std::end(vertex.normal), finite) &&
           std::all_of(std::begin(vertex.tangent), std::end(vertex.tangent), finite) &&
           std::all_of(std::begin(vertex.texcoord), std::end(vertex.texcoord), finite) &&
           std::all_of(std::begin(vertex.color), std::end(vertex.color), finite);
}
} // namespace

core::result<decoded_conventional_mesh, std::string> decode_conventional_mesh_artifact(std::span<const std::byte> bytes,
                                                                                       std::string_view subresource)
{
    reader input{bytes};
    std::string signature;
    if (!input.read_string(signature) || signature != magic)
        return core::result<decoded_conventional_mesh, std::string>::failure("Invalid .arcmesh schema-v2 magic");

    std::uint32_t mesh_count{};
    if (!input.read(mesh_count) || mesh_count == 0 || mesh_count > max_meshes)
        return core::result<decoded_conventional_mesh, std::string>::failure("Invalid .arcmesh mesh count");
    if (subresource.empty() && mesh_count != 1)
        return core::result<decoded_conventional_mesh, std::string>::failure(
            "Multi-mesh artifact requires an explicit mesh subresource name");

    std::unordered_set<std::string> names;
    decoded_conventional_mesh selected;
    bool found = false;
    for (std::uint32_t m = 0; m < mesh_count; ++m)
    {
        std::string name;
        std::uint64_t material_index{};
        std::uint32_t lod_count{};
        if (!input.read_string(name) || name.empty() || !names.insert(name).second || !input.read(material_index) ||
            !input.read(lod_count) || lod_count == 0 || lod_count > max_lods)
            return core::result<decoded_conventional_mesh, std::string>::failure("Invalid .arcmesh mesh record");

        const bool wanted = subresource.empty() || name == subresource;
        for (std::uint32_t lod = 0; lod < lod_count; ++lod)
        {
            float ratio{}, geometric_error{};
            std::span<const std::byte> vertices, indices;
            if (!input.read(ratio) || !input.read(geometric_error) || !std::isfinite(ratio) || ratio <= 0.0f ||
                ratio > 1.0f || !std::isfinite(geometric_error) || geometric_error < 0.0f ||
                !input.read_blob(vertices, sizeof(mesh_vertex)) || !input.read_blob(indices, sizeof(std::uint32_t)) ||
                vertices.empty() || indices.empty() || indices.size() / sizeof(std::uint32_t) % 3 != 0 ||
                vertices.size() / sizeof(mesh_vertex) > std::numeric_limits<std::uint32_t>::max())
                return core::result<decoded_conventional_mesh, std::string>::failure("Invalid .arcmesh LOD payload");

            const auto vertex_count = vertices.size() / sizeof(mesh_vertex);
            const auto index_count = indices.size() / sizeof(std::uint32_t);
            // Validate all LODs, not only the selected one, to reject corrupt packages.
            for (std::size_t v = 0; v < vertex_count; ++v)
            {
                mesh_vertex vertex;
                std::memcpy(&vertex, vertices.data() + v * sizeof(vertex), sizeof(vertex));
                if (!valid_vertex(vertex))
                    return core::result<decoded_conventional_mesh, std::string>::failure(
                        "Non-finite .arcmesh vertex data");
            }
            for (std::size_t i = 0; i < index_count; ++i)
            {
                std::uint32_t index{};
                std::memcpy(&index, indices.data() + i * sizeof(index), sizeof(index));
                if (index >= vertex_count)
                    return core::result<decoded_conventional_mesh, std::string>::failure(
                        "Out-of-range .arcmesh triangle index");
            }

            if (wanted && lod == 0)
            {
                auto& mesh = selected.mesh;
                mesh.name = name;
                mesh.material_index = static_cast<std::size_t>(material_index);
                if (static_cast<std::uint64_t>(mesh.material_index) != material_index)
                    return core::result<decoded_conventional_mesh, std::string>::failure(
                        "Invalid .arcmesh material index");
                mesh.vertices.resize(vertex_count);
                mesh.indices.resize(index_count);
                std::memcpy(mesh.vertices.data(), vertices.data(), vertices.size());
                std::memcpy(mesh.indices.data(), indices.data(), indices.size());
                selected.lod_ratio = ratio;
                selected.geometric_error = geometric_error;
                found = true;
            }
        }
    }
    if (input.offset != bytes.size())
        return core::result<decoded_conventional_mesh, std::string>::failure("Trailing .arcmesh bytes");
    if (!found)
        return core::result<decoded_conventional_mesh, std::string>::failure(
            "Requested .arcmesh subresource not found");
    return core::result<decoded_conventional_mesh, std::string>::success(std::move(selected));
}
} // namespace arc::render
