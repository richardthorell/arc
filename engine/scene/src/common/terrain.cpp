#include <arc/scene/terrain.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace arc::scene
{
namespace
{

float saturate(float value) noexcept { return std::clamp(value, 0.0f, 1.0f); }
float smoothstep(float a, float b, float value) noexcept
{
    const float t = saturate((value - a) / std::max(b - a, 0.00001f));
    return t * t * (3.0f - 2.0f * t);
}

std::uint32_t hash(std::uint32_t x, std::uint32_t z) noexcept
{
    std::uint32_t value = x * 0x8da6b343u ^ z * 0xd8163841u;
    value ^= value >> 13u;
    value *= 0x85ebca6bu;
    return value ^ (value >> 16u);
}

float random_signed(std::int32_t x, std::int32_t z) noexcept
{
    return static_cast<float>(hash(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(z)) & 0xffffu) /
        32767.5f - 1.0f;
}

float value_noise(float x, float z) noexcept
{
    const auto ix = static_cast<std::int32_t>(std::floor(x));
    const auto iz = static_cast<std::int32_t>(std::floor(z));
    const float fx = x - static_cast<float>(ix);
    const float fz = z - static_cast<float>(iz);
    const float sx = fx * fx * (3.0f - 2.0f * fx);
    const float sz = fz * fz * (3.0f - 2.0f * fz);
    const float a = std::lerp(random_signed(ix, iz), random_signed(ix + 1, iz), sx);
    const float b = std::lerp(random_signed(ix, iz + 1), random_signed(ix + 1, iz + 1), sx);
    return std::lerp(a, b, sz);
}

float fbm(float x, float z, std::uint32_t octaves) noexcept
{
    float result{};
    float amplitude{ 0.5f };
    for (std::uint32_t octave = 0; octave < octaves; ++octave)
    {
        result += value_noise(x, z) * amplitude;
        x = x * 2.03f + 17.1f;
        z = z * 2.01f - 11.7f;
        amplitude *= 0.5f;
    }
    return result;
}

std::size_t sample_index(const terrain_component& terrain, std::uint32_t x, std::uint32_t z) noexcept
{
    const auto resolution = terrain.subdivisions + 1u;
    return static_cast<std::size_t>(z) * resolution + x;
}

float height_at(const terrain_component& terrain, std::uint32_t x, std::uint32_t z) noexcept
{
    x = std::min(x, terrain.subdivisions);
    z = std::min(z, terrain.subdivisions);
    return terrain.heights[sample_index(terrain, x, z)];
}

math::vector3f normal_at(const terrain_component& terrain, std::uint32_t x, std::uint32_t z) noexcept
{
    const float spacing = terrain.size / static_cast<float>(terrain.subdivisions);
    const float left = height_at(terrain, x > 0 ? x - 1u : x, z);
    const float right = height_at(terrain, std::min(x + 1u, terrain.subdivisions), z);
    const float down = height_at(terrain, x, z > 0 ? z - 1u : z);
    const float up = height_at(terrain, x, std::min(z + 1u, terrain.subdivisions));
    return math::normalize(math::vector3f{ left - right, spacing * 2.0f, down - up });
}

std::array<std::uint8_t, 4> normalized_weights(const std::array<float, 4>& input) noexcept
{
    float total{};
    for (const float value : input) total += std::max(value, 0.0f);
    std::array<std::uint8_t, 4> result{};
    if (total <= 0.00001f)
    {
        result[0] = 255;
        return result;
    }
    std::uint32_t remaining{ 255u };
    for (std::size_t index = 0; index < 3; ++index)
    {
        const auto encoded = static_cast<std::uint32_t>(std::clamp(
            std::lround(std::max(input[index], 0.0f) / total * 255.0f), 0l, 255l));
        result[index] = static_cast<std::uint8_t>(std::min(encoded, remaining));
        remaining -= result[index];
    }
    result[3] = static_cast<std::uint8_t>(remaining);
    return result;
}

bool intersect_triangle(
    const math::vector3f& origin,
    const math::vector3f& direction,
    const math::vector3f& a,
    const math::vector3f& b,
    const math::vector3f& c,
    float& distance) noexcept
{
    const auto edge1 = math::sub(b, a);
    const auto edge2 = math::sub(c, a);
    const auto p = math::cross(direction, edge2);
    const float determinant = math::dot(edge1, p);
    if (std::abs(determinant) < 0.000001f) return false;
    const float inverse = 1.0f / determinant;
    const auto t = math::sub(origin, a);
    const float u = math::dot(t, p) * inverse;
    if (u < 0.0f || u > 1.0f) return false;
    const auto q = math::cross(t, edge1);
    const float v = math::dot(direction, q) * inverse;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float hit = math::dot(edge2, q) * inverse;
    if (hit < 0.0f) return false;
    distance = hit;
    return true;
}

}

bool terrain_heightfield_valid(const terrain_component& terrain) noexcept
{
    if (terrain.subdivisions == 0 || terrain.size <= 0.0f) return false;
    const auto resolution = static_cast<std::size_t>(terrain.subdivisions) + 1u;
    const auto count = resolution * resolution;
    return terrain.heights.size() == count && terrain.layer_weights.size() == count;
}

void generate_terrain_heightfield(terrain_component& terrain)
{
    terrain.subdivisions = std::clamp<std::uint32_t>(terrain.subdivisions, 1u, 1024u);
    terrain.chunk_quads = std::clamp<std::uint32_t>(terrain.chunk_quads, 1u, terrain.subdivisions);
    terrain.size = std::max(terrain.size, 1.0f);
    terrain.height_scale = std::max(terrain.height_scale, 0.0f);
    const auto resolution = terrain.subdivisions + 1u;
    const auto count = static_cast<std::size_t>(resolution) * resolution;
    terrain.heights.resize(count);
    terrain.layer_weights.resize(count);

    float min_height = std::numeric_limits<float>::max();
    float max_height = std::numeric_limits<float>::lowest();
    for (std::uint32_t z = 0; z < resolution; ++z)
    {
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const float nx = static_cast<float>(x) / terrain.subdivisions * 2.0f - 1.0f;
            const float nz = static_cast<float>(z) / terrain.subdivisions * 2.0f - 1.0f;
            const float warp_x = fbm(nx * 1.7f + 7.0f, nz * 1.7f - 4.0f, 3) * 0.18f;
            const float warp_z = fbm(nx * 1.7f - 9.0f, nz * 1.7f + 3.0f, 3) * 0.18f;
            const float wx = nx + warp_x;
            const float wz = nz + warp_z;
            const float rear = std::exp(-std::pow((wz + 0.58f) / 0.30f, 2.0f));
            const float mountains = std::pow(saturate(fbm(wx * 2.25f, wz * 2.25f, 6) * 0.72f + 0.48f), 1.65f);
            const float ridges = std::pow(1.0f - std::abs(fbm(wx * 4.2f + 13.0f, wz * 4.2f, 4)), 3.0f);
            const float basin_dx = nx - 0.18f;
            const float basin_dz = nz - 0.18f;
            const float basin = std::exp(-(basin_dx * basin_dx / 0.10f + basin_dz * basin_dz / 0.13f));
            const float edge = smoothstep(0.82f, 1.28f, std::sqrt(nx * nx + nz * nz));
            float height = (0.035f + mountains * (0.22f + rear * 0.80f) + ridges * rear * 0.16f -
                basin * 0.16f - edge * 0.12f) * terrain.height_scale;
            // Stable staging area for the hero formation.
            const float pad = std::exp(-((nx + 0.18f) * (nx + 0.18f) + (nz - 0.08f) * (nz - 0.08f)) / 0.012f);
            height = std::lerp(height, terrain.height_scale * 0.075f, pad * 0.72f);
            terrain.heights[sample_index(terrain, x, z)] = height;
            min_height = std::min(min_height, height);
            max_height = std::max(max_height, height);
        }
    }

    for (std::uint32_t z = 0; z < resolution; ++z)
    {
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const auto index = sample_index(terrain, x, z);
            const float height01 = (terrain.heights[index] - min_height) / std::max(max_height - min_height, 0.001f);
            const float slope = 1.0f - normal_at(terrain, x, z)[1];
            const float shore = 1.0f - smoothstep(terrain.height_scale * 0.015f, terrain.height_scale * 0.10f, terrain.heights[index]);
            const float rock = saturate(smoothstep(0.10f, 0.38f, slope) + smoothstep(0.62f, 0.92f, height01));
            const float sand = shore * (1.0f - rock);
            const float dirt = saturate(smoothstep(0.04f, 0.22f, slope) * (1.0f - rock) +
                std::abs(fbm(static_cast<float>(x) * 0.025f, static_cast<float>(z) * 0.025f, 3)) * 0.22f);
            const float grass = std::max(0.0f, 1.0f - rock - sand - dirt * 0.55f);
            terrain.layer_weights[index] = normalized_weights({ grass, dirt, rock, sand });
        }
    }
    ++terrain.content_revision;
}

float sample_terrain_height(const terrain_component& terrain, float local_x, float local_z) noexcept
{
    if (!terrain_heightfield_valid(terrain)) return 0.0f;
    const float gx = saturate(local_x / terrain.size + 0.5f) * terrain.subdivisions;
    const float gz = saturate(local_z / terrain.size + 0.5f) * terrain.subdivisions;
    const auto x0 = static_cast<std::uint32_t>(std::floor(gx));
    const auto z0 = static_cast<std::uint32_t>(std::floor(gz));
    const auto x1 = std::min(x0 + 1u, terrain.subdivisions);
    const auto z1 = std::min(z0 + 1u, terrain.subdivisions);
    const float tx = gx - x0;
    const float tz = gz - z0;
    return std::lerp(std::lerp(height_at(terrain, x0, z0), height_at(terrain, x1, z0), tx),
        std::lerp(height_at(terrain, x0, z1), height_at(terrain, x1, z1), tx), tz);
}

math::vector3f sample_terrain_normal(const terrain_component& terrain, float local_x, float local_z) noexcept
{
    if (!terrain_heightfield_valid(terrain)) return { 0.0f, 1.0f, 0.0f };
    const auto x = static_cast<std::uint32_t>(std::round(saturate(local_x / terrain.size + 0.5f) * terrain.subdivisions));
    const auto z = static_cast<std::uint32_t>(std::round(saturate(local_z / terrain.size + 0.5f) * terrain.subdivisions));
    return normal_at(terrain, x, z);
}

render::mesh_data make_terrain_chunk_mesh(const terrain_component& terrain, std::uint32_t chunk_x, std::uint32_t chunk_z)
{
    render::mesh_data mesh;
    if (!terrain_heightfield_valid(terrain)) return mesh;
    const auto start_x = chunk_x * terrain.chunk_quads;
    const auto start_z = chunk_z * terrain.chunk_quads;
    if (start_x >= terrain.subdivisions || start_z >= terrain.subdivisions) return mesh;
    const auto quads_x = std::min(terrain.chunk_quads, terrain.subdivisions - start_x);
    const auto quads_z = std::min(terrain.chunk_quads, terrain.subdivisions - start_z);
    const auto width = quads_x + 1u;
    const float spacing = terrain.size / terrain.subdivisions;
    const float half = terrain.size * 0.5f;
    mesh.name = "Terrain chunk " + std::to_string(chunk_x) + "," + std::to_string(chunk_z);
    mesh.usage = render::mesh_usage::dynamic_per_frame;
    mesh.vertices.reserve(static_cast<std::size_t>(width) * (quads_z + 1u));
    mesh.indices.reserve(static_cast<std::size_t>(quads_x) * quads_z * 6u);
    for (std::uint32_t z = 0; z <= quads_z; ++z)
    {
        for (std::uint32_t x = 0; x <= quads_x; ++x)
        {
            const auto sx = start_x + x;
            const auto sz = start_z + z;
            const auto normal = normal_at(terrain, sx, sz);
            const auto weights = terrain.layer_weights[sample_index(terrain, sx, sz)];
            render::mesh_vertex vertex{};
            vertex.position[0] = -half + sx * spacing;
            vertex.position[1] = height_at(terrain, sx, sz);
            vertex.position[2] = -half + sz * spacing;
            vertex.normal[0] = normal[0]; vertex.normal[1] = normal[1]; vertex.normal[2] = normal[2];
            vertex.texcoord[0] = vertex.position[0]; vertex.texcoord[1] = vertex.position[2];
            for (std::size_t layer = 0; layer < 4; ++layer)
                vertex.color[layer] = static_cast<float>(weights[layer]) / 255.0f;
            mesh.vertices.push_back(vertex);
        }
    }
    for (std::uint32_t z = 0; z < quads_z; ++z)
    {
        for (std::uint32_t x = 0; x < quads_x; ++x)
        {
            const auto a = z * width + x;
            const auto b = a + 1u;
            const auto d = a + width;
            const auto c = d + 1u;
            mesh.indices.insert(mesh.indices.end(), { a, b, c, a, c, d });
        }
    }
    return mesh;
}

terrain_dirty_region apply_terrain_brush(
    terrain_component& terrain,
    const math::vector3f& local_center,
    const terrain_brush_settings& settings,
    float delta_seconds)
{
    terrain_dirty_region dirty{};
    if (!terrain_heightfield_valid(terrain) || settings.radius <= 0.0f || settings.strength <= 0.0f) return dirty;
    const float spacing = terrain.size / terrain.subdivisions;
    const float center_x = (local_center[0] / terrain.size + 0.5f) * terrain.subdivisions;
    const float center_z = (local_center[2] / terrain.size + 0.5f) * terrain.subdivisions;
    const float radius_samples = settings.radius / spacing;
    const auto min_x = static_cast<std::uint32_t>(std::max(0.0f, std::floor(center_x - radius_samples)));
    const auto min_z = static_cast<std::uint32_t>(std::max(0.0f, std::floor(center_z - radius_samples)));
    const auto max_x = static_cast<std::uint32_t>(std::min<float>(terrain.subdivisions, std::ceil(center_x + radius_samples)));
    const auto max_z = static_cast<std::uint32_t>(std::min<float>(terrain.subdivisions, std::ceil(center_z + radius_samples)));
    const auto original_heights = terrain.heights;
    bool changed{};
    for (std::uint32_t z = min_z; z <= max_z; ++z)
    {
        for (std::uint32_t x = min_x; x <= max_x; ++x)
        {
            const float dx = (static_cast<float>(x) - center_x) * spacing;
            const float dz = (static_cast<float>(z) - center_z) * spacing;
            const float distance = std::sqrt(dx * dx + dz * dz);
            if (distance > settings.radius) continue;
            const float t = 1.0f - distance / settings.radius;
            const float smooth_falloff = t * t * (3.0f - 2.0f * t);
            const float falloff = std::lerp(t, smooth_falloff, saturate(settings.falloff));
            const float amount = settings.strength * falloff * std::max(delta_seconds, 0.0f);
            const auto index = sample_index(terrain, x, z);
            if (settings.tool == terrain_brush_tool::sculpt)
                terrain.heights[index] += (settings.invert ? -1.0f : 1.0f) * amount * 12.0f;
            else if (settings.tool == terrain_brush_tool::flatten)
                terrain.heights[index] = std::lerp(terrain.heights[index], settings.flatten_height, saturate(amount * 8.0f));
            else if (settings.tool == terrain_brush_tool::smooth)
            {
                float total{}; float samples{};
                for (int oz = -1; oz <= 1; ++oz)
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        const auto sx = static_cast<std::uint32_t>(std::clamp<int>(static_cast<int>(x) + ox, 0, terrain.subdivisions));
                        const auto sz = static_cast<std::uint32_t>(std::clamp<int>(static_cast<int>(z) + oz, 0, terrain.subdivisions));
                        total += original_heights[sample_index(terrain, sx, sz)]; ++samples;
                    }
                terrain.heights[index] = std::lerp(terrain.heights[index], total / samples, saturate(amount * 10.0f));
            }
            else
            {
                const auto active = std::min<std::uint32_t>(settings.active_layer, 3u);
                std::array<float, 4> weights{};
                for (std::size_t layer = 0; layer < 4; ++layer)
                    weights[layer] = static_cast<float>(terrain.layer_weights[index][layer]) / 255.0f;
                weights[active] = saturate(weights[active] + (settings.invert ? -amount : amount) * 4.0f);
                const float remaining = 1.0f - weights[active];
                float other_total{};
                for (std::size_t layer = 0; layer < 4; ++layer) if (layer != active) other_total += weights[layer];
                for (std::size_t layer = 0; layer < 4; ++layer)
                    if (layer != active) weights[layer] = other_total > 0.0001f ? weights[layer] / other_total * remaining : remaining / 3.0f;
                terrain.layer_weights[index] = normalized_weights(weights);
            }
            changed = true;
        }
    }
    if (changed)
    {
        ++terrain.content_revision;
        dirty = { min_x > 0 ? min_x - 1u : 0u, min_z > 0 ? min_z - 1u : 0u,
            std::min(max_x + 1u, terrain.subdivisions), std::min(max_z + 1u, terrain.subdivisions), true };
    }
    return dirty;
}

terrain_raycast_hit raycast_terrain(
    const terrain_component& terrain,
    const math::vector3f& local_origin,
    const math::vector3f& local_direction) noexcept
{
    terrain_raycast_hit result{};
    if (!terrain_heightfield_valid(terrain)) return result;
    const float half = terrain.size * 0.5f;
    const float spacing = terrain.size / terrain.subdivisions;
    float nearest = std::numeric_limits<float>::max();
    for (std::uint32_t z = 0; z < terrain.subdivisions; ++z)
    {
        for (std::uint32_t x = 0; x < terrain.subdivisions; ++x)
        {
            const math::vector3f a{ -half + x * spacing, height_at(terrain, x, z), -half + z * spacing };
            const math::vector3f b{ a[0] + spacing, height_at(terrain, x + 1u, z), a[2] };
            const math::vector3f c{ a[0] + spacing, height_at(terrain, x + 1u, z + 1u), a[2] + spacing };
            const math::vector3f d{ a[0], height_at(terrain, x, z + 1u), a[2] + spacing };
            float distance{};
            if ((intersect_triangle(local_origin, local_direction, a, b, c, distance) ||
                intersect_triangle(local_origin, local_direction, a, c, d, distance)) && distance < nearest)
                nearest = distance;
        }
    }
    if (nearest < std::numeric_limits<float>::max())
    {
        result.hit = true;
        result.distance = nearest;
        result.position = math::add(local_origin, math::mul(local_direction, nearest));
        result.normal = sample_terrain_normal(terrain, result.position[0], result.position[2]);
    }
    return result;
}

}
