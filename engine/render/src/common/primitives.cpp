#include <arc/render/primitives.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace arc::render
{
namespace
{

constexpr float pi = 3.14159265358979323846f;

float clamp01(float value) noexcept
{
    return std::clamp(value, 0.0f, 1.0f);
}

float smoothstep(float edge0, float edge1, float value) noexcept
{
    const float t = clamp01((value - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

float terrain_height_at(float x, float z, float size, float height_scale) noexcept
{
    const float broad = std::sin(x * 0.18f + std::cos(z * 0.09f) * 1.7f) * 0.62f;
    const float ridge = (1.0f - std::abs(std::sin(x * 0.31f - z * 0.24f))) * 0.46f;
    const float rolling = std::sin((x + z) * 0.42f) * std::cos((x - z) * 0.21f) * 0.24f;
    const float detail = std::sin(x * 1.35f + z * 0.77f) * std::sin(z * 1.11f) * 0.075f;
    const float distance = std::sqrt(x * x + z * z) / std::max(size * 0.5f, 0.001f);
    const float edge_falloff = smoothstep(0.72f, 1.0f, distance) * 0.34f;
    return (broad + ridge + rolling + detail - edge_falloff) * height_scale;
}

mesh_vertex vertex(
    float x,
    float y,
    float z,
    float nx,
    float ny,
    float nz,
    float u,
    float v,
    float r,
    float g,
    float b);

mesh_vertex terrain_vertex(
    float x,
    float y,
    float z,
    float nx,
    float ny,
    float nz,
    float u,
    float v,
    float height01,
    float slope)
{
    const float lowland = smoothstep(0.02f, 0.22f, height01);
    const float highland = smoothstep(0.58f, 0.90f, height01);
    const float rock = clamp01(smoothstep(0.20f, 0.62f, slope) + highland * 0.35f);

    const float grass_r = 0.20f + 0.15f * lowland + 0.05f * std::sin((x + z) * 0.42f);
    const float grass_g = 0.34f + 0.26f * lowland + 0.05f * std::sin(x * 0.71f);
    const float grass_b = 0.16f + 0.08f * lowland;
    const float rock_r = 0.43f + 0.11f * highland;
    const float rock_g = 0.42f + 0.10f * highland;
    const float rock_b = 0.38f + 0.08f * highland;

    return vertex(
        x,
        y,
        z,
        nx,
        ny,
        nz,
        u,
        v,
        grass_r * (1.0f - rock) + rock_r * rock,
        grass_g * (1.0f - rock) + rock_g * rock,
        grass_b * (1.0f - rock) + rock_b * rock);
}

mesh_vertex vertex(
    float x,
    float y,
    float z,
    float nx,
    float ny,
    float nz,
    float u,
    float v,
    float r = 1.0f,
    float g = 1.0f,
    float b = 1.0f)
{
    return {
        .position = { x, y, z },
        .normal = { nx, ny, nz },
        .texcoord = { u, v },
        .color = { r, g, b, 1.0f }
    };
}

void append_face(
    mesh_data& mesh,
    const mesh_vertex& a,
    const mesh_vertex& b,
    const mesh_vertex& c,
    const mesh_vertex& d)
{
    const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(a);
    mesh.vertices.push_back(b);
    mesh.vertices.push_back(c);
    mesh.vertices.push_back(d);
    mesh.indices.insert(mesh.indices.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
}

} // namespace

mesh_data make_plane_mesh(float size)
{
    const float half = std::max(0.001f, size) * 0.5f;
    mesh_data mesh;
    mesh.name = "Plane";
    mesh.vertices = {
        vertex(-half, 0.0f, -half, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f),
        vertex(half, 0.0f, -half, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f),
        vertex(half, 0.0f, half, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f),
        vertex(-half, 0.0f, half, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f)
    };
    mesh.indices = { 0, 1, 2, 0, 2, 3 };
    return mesh;
}

mesh_data make_cube_mesh(float size)
{
    const float half = std::max(0.001f, size) * 0.5f;
    mesh_data mesh;
    mesh.name = "Cube";

    append_face(
        mesh,
        vertex(-half, -half, half, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f),
        vertex(half, -half, half, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f),
        vertex(half, half, half, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
        vertex(-half, half, half, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f));
    append_face(
        mesh,
        vertex(half, -half, -half, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f),
        vertex(-half, -half, -half, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f),
        vertex(-half, half, -half, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f),
        vertex(half, half, -half, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f));
    append_face(
        mesh,
        vertex(-half, -half, -half, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f),
        vertex(-half, -half, half, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f),
        vertex(-half, half, half, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f),
        vertex(-half, half, -half, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f));
    append_face(
        mesh,
        vertex(half, -half, half, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f),
        vertex(half, -half, -half, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f),
        vertex(half, half, -half, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f),
        vertex(half, half, half, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f));
    append_face(
        mesh,
        vertex(-half, half, half, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f),
        vertex(half, half, half, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f),
        vertex(half, half, -half, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f),
        vertex(-half, half, -half, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f));
    append_face(
        mesh,
        vertex(-half, -half, -half, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f),
        vertex(half, -half, -half, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f),
        vertex(half, -half, half, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f),
        vertex(-half, -half, half, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f));

    return mesh;
}

mesh_data make_uv_sphere_mesh(float radius, std::uint32_t slices, std::uint32_t stacks)
{
    radius = std::max(0.001f, radius);
    slices = std::max<std::uint32_t>(3, slices);
    stacks = std::max<std::uint32_t>(2, stacks);

    mesh_data mesh;
    mesh.name = "Sphere";
    for (std::uint32_t stack = 0; stack <= stacks; ++stack)
    {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * pi;
        const float y = std::cos(phi);
        const float ring = std::sin(phi);
        for (std::uint32_t slice = 0; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * pi * 2.0f;
            const float x = std::cos(theta) * ring;
            const float z = std::sin(theta) * ring;
            mesh.vertices.push_back(vertex(x * radius, y * radius, z * radius, x, y, z, u, v));
        }
    }

    const std::uint32_t stride = slices + 1;
    for (std::uint32_t stack = 0; stack < stacks; ++stack)
    {
        for (std::uint32_t slice = 0; slice < slices; ++slice)
        {
            const std::uint32_t a = stack * stride + slice;
            const std::uint32_t b = a + 1;
            const std::uint32_t c = a + stride + 1;
            const std::uint32_t d = a + stride;
            mesh.indices.insert(mesh.indices.end(), { a, b, c, a, c, d });
        }
    }

    return mesh;
}

mesh_data make_cylinder_mesh(float radius, float height, std::uint32_t segments)
{
    radius = std::max(0.001f, radius);
    height = std::max(0.001f, height);
    segments = std::max<std::uint32_t>(3, segments);

    mesh_data mesh;
    mesh.name = "Cylinder";
    const float half_height = height * 0.5f;
    for (std::uint32_t segment = 0; segment <= segments; ++segment)
    {
        const float u = static_cast<float>(segment) / static_cast<float>(segments);
        const float theta = u * pi * 2.0f;
        const float x = std::cos(theta);
        const float z = std::sin(theta);
        mesh.vertices.push_back(vertex(x * radius, -half_height, z * radius, x, 0.0f, z, u, 0.0f));
        mesh.vertices.push_back(vertex(x * radius, half_height, z * radius, x, 0.0f, z, u, 1.0f));
    }

    for (std::uint32_t segment = 0; segment < segments; ++segment)
    {
        const std::uint32_t a = segment * 2;
        const std::uint32_t b = a + 1;
        const std::uint32_t c = a + 3;
        const std::uint32_t d = a + 2;
        mesh.indices.insert(mesh.indices.end(), { a, b, c, a, c, d });
    }

    const std::uint32_t top_center = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(vertex(0.0f, half_height, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f));
    const std::uint32_t bottom_center = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(vertex(0.0f, -half_height, 0.0f, 0.0f, -1.0f, 0.0f, 0.5f, 0.5f));

    for (std::uint32_t segment = 0; segment < segments; ++segment)
    {
        const std::uint32_t next = segment + 1;
        const std::uint32_t bottom = segment * 2;
        const std::uint32_t top = bottom + 1;
        const std::uint32_t next_bottom = next * 2;
        const std::uint32_t next_top = next_bottom + 1;
        mesh.indices.insert(mesh.indices.end(), { top_center, top, next_top, bottom_center, next_bottom, bottom });
    }

    return mesh;
}

mesh_data make_terrain_grid_mesh(float size, std::uint32_t subdivisions, float height_scale)
{
    size = std::max(1.0f, size);
    subdivisions = std::clamp<std::uint32_t>(subdivisions, 1, 256);
    height_scale = std::max(0.0f, height_scale);

    const std::uint32_t vertices_per_side = subdivisions + 1;
    const float half = size * 0.5f;
    const float step = size / static_cast<float>(subdivisions);

    mesh_data mesh;
    mesh.name = "Terrain";
    mesh.vertices.reserve(static_cast<std::size_t>(vertices_per_side) * vertices_per_side);
    mesh.indices.reserve(static_cast<std::size_t>(subdivisions) * subdivisions * 6);

    std::vector<float> heights(static_cast<std::size_t>(vertices_per_side) * vertices_per_side);
    float min_height = std::numeric_limits<float>::max();
    float max_height = std::numeric_limits<float>::lowest();

    for (std::uint32_t z = 0; z < vertices_per_side; ++z)
    {
        for (std::uint32_t x = 0; x < vertices_per_side; ++x)
        {
            const float px = -half + static_cast<float>(x) * step;
            const float pz = -half + static_cast<float>(z) * step;
            const float py = terrain_height_at(px, pz, size, height_scale);
            heights[static_cast<std::size_t>(z) * vertices_per_side + x] = py;
            min_height = std::min(min_height, py);
            max_height = std::max(max_height, py);
        }
    }

    const auto sample_height = [&](std::uint32_t x, std::uint32_t z) {
        x = std::min(x, subdivisions);
        z = std::min(z, subdivisions);
        return heights[static_cast<std::size_t>(z) * vertices_per_side + x];
    };

    for (std::uint32_t z = 0; z < vertices_per_side; ++z)
    {
        for (std::uint32_t x = 0; x < vertices_per_side; ++x)
        {
            const float px = -half + static_cast<float>(x) * step;
            const float pz = -half + static_cast<float>(z) * step;
            const float py = sample_height(x, z);
            const float left = sample_height(x == 0 ? 0 : x - 1, z);
            const float right = sample_height(x + 1, z);
            const float down = sample_height(x, z == 0 ? 0 : z - 1);
            const float up = sample_height(x, z + 1);
            const float nx = left - right;
            const float ny = step * 2.0f;
            const float nz = down - up;
            const float normal_length = std::sqrt(nx * nx + ny * ny + nz * nz);
            const float inv_normal_length = normal_length > 0.0f ? 1.0f / normal_length : 1.0f;
            const float normal_y = ny * inv_normal_length;
            const float height01 = (py - min_height) / std::max(max_height - min_height, 0.001f);
            const float slope = 1.0f - normal_y;
            mesh.vertices.push_back(terrain_vertex(
                px,
                py,
                pz,
                nx * inv_normal_length,
                normal_y,
                nz * inv_normal_length,
                static_cast<float>(x) / static_cast<float>(subdivisions),
                static_cast<float>(z) / static_cast<float>(subdivisions),
                height01,
                slope));
        }
    }

    for (std::uint32_t z = 0; z < subdivisions; ++z)
    {
        for (std::uint32_t x = 0; x < subdivisions; ++x)
        {
            const std::uint32_t a = z * vertices_per_side + x;
            const std::uint32_t b = a + 1;
            const std::uint32_t c = a + vertices_per_side + 1;
            const std::uint32_t d = a + vertices_per_side;
            mesh.indices.insert(mesh.indices.end(), { a, b, c, a, c, d });
        }
    }

    return mesh;
}

mesh_data make_grass_patch_mesh(float patch_size, std::uint32_t blade_count, float height)
{
    patch_size = std::max(0.1f, patch_size);
    blade_count = std::clamp<std::uint32_t>(blade_count, 1, 1024);
    height = std::max(0.05f, height);

    mesh_data mesh;
    mesh.name = "Grass Patch";
    mesh.vertices.reserve(static_cast<std::size_t>(blade_count) * 8);
    mesh.indices.reserve(static_cast<std::size_t>(blade_count) * 12);

    const auto random01 = [](std::uint32_t value) {
        value ^= value << 13u;
        value ^= value >> 17u;
        value ^= value << 5u;
        return static_cast<float>(value & 0xffffu) / 65535.0f;
    };

    const float half_patch = patch_size * 0.5f;
    for (std::uint32_t blade = 0; blade < blade_count; ++blade)
    {
        const float x = (random01(blade * 7477u + 13u) - 0.5f) * patch_size;
        const float z = (random01(blade * 9151u + 71u) - 0.5f) * patch_size;
        const float blade_height = height * (0.65f + random01(blade * 4261u + 9u) * 0.55f);
        const float width = 0.025f + random01(blade * 1999u + 31u) * 0.035f;
        const float bend = (random01(blade * 1237u + 5u) - 0.5f) * 0.12f;
        const float clump = std::max(0.45f, 1.0f - (std::abs(x) + std::abs(z)) / std::max(0.001f, half_patch * 2.0f));
        const float g = 0.55f + 0.28f * clump;

        const auto add_blade_quad = [&](float dx, float dz) {
            const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(vertex(x - dx, 0.0f, z - dz, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.18f, g, 0.14f));
            mesh.vertices.push_back(vertex(x + dx, 0.0f, z + dz, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.18f, g, 0.14f));
            mesh.vertices.push_back(vertex(x + bend + dx * 0.18f, blade_height, z + bend + dz * 0.18f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.36f, std::min(g + 0.18f, 1.0f), 0.20f));
            mesh.vertices.push_back(vertex(x + bend - dx * 0.18f, blade_height, z + bend - dz * 0.18f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.36f, std::min(g + 0.18f, 1.0f), 0.20f));
            mesh.indices.insert(mesh.indices.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
        };

        add_blade_quad(width, 0.0f);
        add_blade_quad(0.0f, width);
    }

    return mesh;
}

} // namespace arc::render
