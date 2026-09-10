#include <arc/scene/terrain_render_geometry.h>

#include <arc/render/mesh.h>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace arc::scene
{
namespace
{

constexpr float normal_epsilon = 1.0e-12f;

math::vector3f stable_tangent(const math::vector3f& normal) noexcept
{
    const math::vector3f x_axis{1.0f, 0.0f, 0.0f};
    const math::vector3f z_axis{0.0f, 0.0f, 1.0f};
    math::vector3f tangent = math::sub(x_axis, math::mul(normal, math::dot(normal, x_axis)));
    if (math::length_squared(tangent) <= normal_epsilon)
        tangent = math::sub(z_axis, math::mul(normal, math::dot(normal, z_axis)));
    return math::length_squared(tangent) > normal_epsilon ? math::normalize(tangent) : x_axis;
}

std::optional<render::virtual_mesh_data> build_geometry(const terrain_surface_ir& surface,
                                                        std::span<const math::vector3f> supplied_normals,
                                                        const render::virtual_mesh_build_options& options)
{
    const auto canonical = canonicalize_terrain_surface_geometry(surface);
    if (!canonical) return std::nullopt;
    if (!supplied_normals.empty() && supplied_normals.size() != canonical->positions.size()) return std::nullopt;

    render::mesh_data source;
    source.name = "terrain";
    source.usage = render::mesh_usage::static_gpu;
    source.vertices.resize(canonical->positions.size());
    source.indices = canonical->indices;

    std::vector<math::vector3f> accumulated_normals;
    if (supplied_normals.empty())
    {
        accumulated_normals.resize(canonical->positions.size());
        for (std::size_t index = 0; index + 2u < canonical->indices.size(); index += 3u)
        {
            const auto i0 = canonical->indices[index + 0u];
            const auto i1 = canonical->indices[index + 1u];
            const auto i2 = canonical->indices[index + 2u];
            const auto edge0 = math::sub(canonical->positions[i1], canonical->positions[i0]);
            const auto edge1 = math::sub(canonical->positions[i2], canonical->positions[i0]);
            const auto face = math::cross(edge0, edge1);
            if (math::length_squared(face) <= normal_epsilon) continue;
            accumulated_normals[i0] = math::add(accumulated_normals[i0], face);
            accumulated_normals[i1] = math::add(accumulated_normals[i1], face);
            accumulated_normals[i2] = math::add(accumulated_normals[i2], face);
        }
    }

    const float extent_x = static_cast<float>(surface.local_bounds.max_x - surface.local_bounds.min_x);
    const float extent_z = static_cast<float>(surface.local_bounds.max_z - surface.local_bounds.min_z);
    for (std::size_t index = 0; index < canonical->positions.size(); ++index)
    {
        const auto& position = canonical->positions[index];
        const auto normal_source = supplied_normals.empty() ? accumulated_normals[index] : supplied_normals[index];
        const auto normal = math::length_squared(normal_source) > normal_epsilon ? math::normalize(normal_source)
                                                                                 : math::vector3f{0.0f, 1.0f, 0.0f};
        const auto tangent = stable_tangent(normal);
        auto& vertex = source.vertices[index];
        vertex.position[0] = position[0];
        vertex.position[1] = position[1];
        vertex.position[2] = position[2];
        vertex.normal[0] = normal[0];
        vertex.normal[1] = normal[1];
        vertex.normal[2] = normal[2];
        vertex.tangent[0] = tangent[0];
        vertex.tangent[1] = tangent[1];
        vertex.tangent[2] = tangent[2];
        vertex.tangent[3] = 1.0f;
        vertex.texcoord[0] = std::abs(extent_x) > 1.0e-8f
                                 ? (position[0] - static_cast<float>(surface.local_bounds.min_x)) / extent_x
                                 : 0.0f;
        vertex.texcoord[1] = std::abs(extent_z) > 1.0e-8f
                                 ? (position[2] - static_cast<float>(surface.local_bounds.min_z)) / extent_z
                                 : 0.0f;
    }

    auto result = render::build_virtual_mesh(source, options);
    if (result.clusters.empty() || result.root_nodes.empty() || result.pages.empty()) return std::nullopt;
    if (options.build_conventional_lods && result.conventional_lods.empty()) return std::nullopt;
    return result;
}

} // namespace

std::optional<render::virtual_mesh_data>
build_terrain_render_geometry(const terrain_surface_ir& surface, const render::virtual_mesh_build_options& options)
{
    return build_geometry(surface, {}, options);
}

std::optional<render::virtual_mesh_data>
build_terrain_render_region_geometry(const terrain_surface_ir& surface, std::span<const math::vector3f> vertex_normals,
                                     const render::virtual_mesh_build_options& options)
{
    return build_geometry(surface, vertex_normals, options);
}

} // namespace arc::scene
