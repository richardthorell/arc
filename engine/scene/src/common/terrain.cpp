#include <arc/scene/terrain.h>

#include <arc/render/renderer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace arc::scene
{
namespace
{

float saturate(float value) noexcept
{
    return std::clamp(value, 0.0f, 1.0f);
}
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

float random_signed(std::int32_t x, std::int32_t z, std::uint64_t seed = 1u) noexcept
{
    const auto seed_x = static_cast<std::uint32_t>(seed);
    const auto seed_z = static_cast<std::uint32_t>(seed >> 32u);
    return static_cast<float>(hash(static_cast<std::uint32_t>(x) ^ seed_x, static_cast<std::uint32_t>(z) ^ seed_z) &
                              0xffffu) /
               32767.5f -
           1.0f;
}

float value_noise(float x, float z, std::uint64_t seed = 1u) noexcept
{
    const auto ix = static_cast<std::int32_t>(std::floor(x));
    const auto iz = static_cast<std::int32_t>(std::floor(z));
    const float fx = x - static_cast<float>(ix);
    const float fz = z - static_cast<float>(iz);
    const float sx = fx * fx * (3.0f - 2.0f * fx);
    const float sz = fz * fz * (3.0f - 2.0f * fz);
    const float a = std::lerp(random_signed(ix, iz, seed), random_signed(ix + 1, iz, seed), sx);
    const float b = std::lerp(random_signed(ix, iz + 1, seed), random_signed(ix + 1, iz + 1, seed), sx);
    return std::lerp(a, b, sz);
}

float fbm(float x, float z, std::uint32_t octaves, std::uint64_t seed = 1u) noexcept
{
    float result{};
    float amplitude{0.5f};
    for (std::uint32_t octave = 0; octave < octaves; ++octave)
    {
        result += value_noise(x, z, seed + octave * 0x9e3779b97f4a7c15ull) * amplitude;
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
    return math::normalize(math::vector3f{left - right, spacing * 2.0f, down - up});
}

std::array<std::uint8_t, 4> normalized_weights(const std::array<float, 4>& input) noexcept
{
    float total{};
    for (const float value : input)
        total += std::max(value, 0.0f);
    std::array<std::uint8_t, 4> result{};
    if (total <= 0.00001f)
    {
        result[0] = 255;
        return result;
    }
    std::uint32_t remaining{255u};
    for (std::size_t index = 0; index < 3; ++index)
    {
        const auto encoded = static_cast<std::uint32_t>(
            std::clamp(std::lround(std::max(input[index], 0.0f) / total * 255.0f), 0l, 255l));
        result[index] = static_cast<std::uint8_t>(std::min(encoded, remaining));
        remaining -= result[index];
    }
    result[3] = static_cast<std::uint8_t>(remaining);
    return result;
}

bool intersect_triangle(const math::vector3f& origin, const math::vector3f& direction, const math::vector3f& a,
                        const math::vector3f& b, const math::vector3f& c, float& distance) noexcept
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

bool intersect_bounds(const math::vector3f& origin, const math::vector3f& direction, const geometric::box3f& bounds,
                      float maximum_distance) noexcept
{
    float near_distance{};
    float far_distance = maximum_distance;
    for (std::size_t axis = 0; axis < 3u; ++axis)
    {
        if (std::abs(direction[axis]) < 0.000001f)
        {
            if (origin[axis] < bounds.min[axis] || origin[axis] > bounds.max[axis]) return false;
            continue;
        }
        const float inverse = 1.0f / direction[axis];
        float first = (bounds.min[axis] - origin[axis]) * inverse;
        float second = (bounds.max[axis] - origin[axis]) * inverse;
        if (first > second) std::swap(first, second);
        near_distance = std::max(near_distance, first);
        far_distance = std::min(far_distance, second);
        if (near_distance > far_distance) return false;
    }
    return far_distance >= 0.0f;
}

float sinc(float value) noexcept
{
    if (std::abs(value) < 0.00001f) return 1.0f;
    const float angle = value * math::pi<float>;
    return std::sin(angle) / angle;
}

float lanczos(float value) noexcept
{
    value = std::abs(value);
    return value >= 3.0f ? 0.0f : sinc(value) * sinc(value / 3.0f);
}

std::vector<float> resample_scalar_field(std::span<const float> source, std::uint32_t source_width,
                                         std::uint32_t source_height, std::uint32_t target_width,
                                         std::uint32_t target_height)
{
    if (source_width == target_width && source_height == target_height) return {source.begin(), source.end()};
    std::vector<float> horizontal(static_cast<std::size_t>(target_width) * source_height);
    std::vector<float> result(static_cast<std::size_t>(target_width) * target_height);
    const auto sample_line =
        [](auto sample, std::uint32_t source_count, std::uint32_t target_count, std::uint32_t target_index)
    {
        if (target_count < source_count)
        {
            const double begin = static_cast<double>(target_index) * source_count / target_count;
            const double end = static_cast<double>(target_index + 1u) * source_count / target_count;
            double total{};
            double weight{};
            const auto last = static_cast<std::uint32_t>(std::ceil(end));
            for (auto index = static_cast<std::uint32_t>(std::floor(begin)); index < last; ++index)
            {
                const double overlap = std::max(0.0, std::min(end, static_cast<double>(index + 1u)) -
                                                         std::max(begin, static_cast<double>(index)));
                total += sample(std::min(index, source_count - 1u)) * overlap;
                weight += overlap;
            }
            return static_cast<float>(total / std::max(weight, 0.000001));
        }
        const float position = static_cast<float>(target_index) * static_cast<float>(source_count - 1u) /
                               static_cast<float>(target_count - 1u);
        const auto center = static_cast<int>(std::floor(position));
        float total{};
        float weight{};
        for (int offset = -2; offset <= 3; ++offset)
        {
            const int index = std::clamp(center + offset, 0, static_cast<int>(source_count - 1u));
            const float filter = lanczos(position - static_cast<float>(center + offset));
            total += sample(static_cast<std::uint32_t>(index)) * filter;
            weight += filter;
        }
        return total / std::max(weight, 0.000001f);
    };
    for (std::uint32_t z = 0; z < source_height; ++z)
        for (std::uint32_t x = 0; x < target_width; ++x)
            horizontal[static_cast<std::size_t>(z) * target_width + x] = sample_line(
                [&](std::uint32_t value) { return source[static_cast<std::size_t>(z) * source_width + value]; },
                source_width, target_width, x);
    for (std::uint32_t z = 0; z < target_height; ++z)
        for (std::uint32_t x = 0; x < target_width; ++x)
            result[static_cast<std::size_t>(z) * target_width + x] = sample_line(
                [&](std::uint32_t value) { return horizontal[static_cast<std::size_t>(value) * target_width + x]; },
                source_height, target_height, z);
    return result;
}

std::vector<float> resample_square_field(std::span<const float> source, std::uint32_t source_resolution,
                                         std::uint32_t target_resolution)
{
    return resample_scalar_field(source, source_resolution, source_resolution, target_resolution, target_resolution);
}

} // namespace

bool terrain_heightfield_valid(const terrain_component& terrain) noexcept
{
    if (terrain.subdivisions == 0 || terrain.size <= 0.0f) return false;
    const auto resolution = static_cast<std::size_t>(terrain.subdivisions) + 1u;
    const auto count = resolution * resolution;
    return terrain.heights.size() == count && terrain.layer_weights.size() == count;
}

bool terrain_resolution_supported(std::uint32_t resolution) noexcept
{
    return std::find(supported_terrain_resolutions.begin(), supported_terrain_resolutions.end(), resolution) !=
           supported_terrain_resolutions.end();
}

terrain_memory_estimate estimate_terrain_memory(std::uint32_t resolution) noexcept
{
    if (!terrain_resolution_supported(resolution)) return {};
    const auto samples = static_cast<std::uint64_t>(resolution) * resolution;
    return {.cpu_bytes = samples * (sizeof(float) + 4u),
            .gpu_bytes = samples * (sizeof(float) + 4u),
            .staging_bytes = samples * (sizeof(float) + 4u),
            .history_bytes = samples * (sizeof(std::uint16_t) * 2u + 8u)};
}

terrain_authoring_result validate_terrain_creation(const terrain_creation_descriptor& descriptor) noexcept
{
    if (!std::isfinite(descriptor.size) || descriptor.size < 1.0f || descriptor.size > 262144.0f)
        return {false, "terrain size must be finite and in [1, 262144] metres"};
    if (!std::isfinite(descriptor.minimum_elevation) || !std::isfinite(descriptor.maximum_elevation) ||
        descriptor.maximum_elevation <= descriptor.minimum_elevation)
        return {false, "terrain elevation range must be finite and increasing"};
    if (!terrain_resolution_supported(descriptor.sample_resolution))
        return {false, "terrain resolution must be one of 257, 513, 1025, 2049, or 4097"};
    if (descriptor.patch_quads != 16u && descriptor.patch_quads != 32u && descriptor.patch_quads != 64u)
        return {false, "terrain patch topology must contain 16, 32, or 64 quads"};
    if ((descriptor.sample_resolution - 1u) % descriptor.patch_quads != 0u)
        return {false, "terrain resolution is incompatible with the selected patch topology"};
    return {true, {}};
}

terrain_authoring_result generate_terrain(terrain_component& terrain, const terrain_generation_descriptor& descriptor)
{
    if (!std::isfinite(descriptor.minimum_elevation) || !std::isfinite(descriptor.maximum_elevation) ||
        descriptor.maximum_elevation <= descriptor.minimum_elevation)
        return {false, "terrain generation requires a finite increasing elevation range"};
    const auto resolution = terrain.subdivisions + 1u;
    if (!terrain_resolution_supported(resolution)) return {false, "terrain resolution is unsupported"};
    const auto count = static_cast<std::size_t>(resolution) * resolution;
    terrain.heights.assign(count, descriptor.minimum_elevation);
    terrain.layer_weights.assign(count, {255u, 0u, 0u, 0u});
    terrain.height_scale = descriptor.maximum_elevation - descriptor.minimum_elevation;
    if (descriptor.generator_id == "arc.terrain.flat.v1")
    {
        ++terrain.content_revision;
        return {true, {}};
    }
    if (descriptor.generator_id != "arc.terrain.domain_warped.v1")
        return {false, "terrain generator ID is not registered"};

    float minimum = std::numeric_limits<float>::max();
    float maximum = std::numeric_limits<float>::lowest();
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const float nx = static_cast<float>(x) / terrain.subdivisions * 2.0f - 1.0f;
            const float nz = static_cast<float>(z) / terrain.subdivisions * 2.0f - 1.0f;
            const float warp_x = fbm(nx * 1.7f + 7.0f, nz * 1.7f - 4.0f, 3, descriptor.seed) * 0.18f;
            const float warp_z = fbm(nx * 1.7f - 9.0f, nz * 1.7f + 3.0f, 3, descriptor.seed ^ 0xa5a5a5a5u) * 0.18f;
            const float rear = std::exp(-std::pow((nz + 0.58f) / 0.30f, 2.0f));
            const float mountains = std::pow(
                saturate(fbm((nx + warp_x) * 2.25f, (nz + warp_z) * 2.25f, 6, descriptor.seed) * 0.72f + 0.48f), 1.65f);
            const float ridges = std::pow(1.0f - std::abs(fbm((nx + warp_x) * 4.2f + 13.0f, (nz + warp_z) * 4.2f, 4,
                                                              descriptor.seed ^ 0x55aa55aau)),
                                          3.0f);
            const float basin = std::exp(-((nx - 0.18f) * (nx - 0.18f) / 0.10f + (nz - 0.18f) * (nz - 0.18f) / 0.13f));
            float normalized = 0.035f + mountains * (0.22f + rear * 0.80f) + ridges * rear * 0.16f - basin * 0.16f;
            normalized = saturate(normalized);
            const auto index = sample_index(terrain, x, z);
            terrain.heights[index] = std::lerp(descriptor.minimum_elevation, descriptor.maximum_elevation, normalized);
            minimum = std::min(minimum, terrain.heights[index]);
            maximum = std::max(maximum, terrain.heights[index]);
        }
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const auto index = sample_index(terrain, x, z);
            const float height01 = (terrain.heights[index] - minimum) / std::max(maximum - minimum, 0.001f);
            const float slope = 1.0f - normal_at(terrain, x, z)[1];
            const float rock = saturate(smoothstep(0.10f, 0.38f, slope) + smoothstep(0.62f, 0.92f, height01));
            const float sand = (1.0f - smoothstep(0.03f, 0.12f, height01)) * (1.0f - rock);
            const float dirt = saturate(smoothstep(0.04f, 0.22f, slope) * (1.0f - rock));
            terrain.layer_weights[index] =
                normalized_weights({std::max(0.0f, 1.0f - rock - sand - dirt * 0.55f), dirt, rock, sand});
        }
    ++terrain.content_revision;
    return {true, {}};
}

void generate_terrain_heightfield(terrain_component& terrain)
{
    terrain.subdivisions = std::clamp<std::uint32_t>(terrain.subdivisions, 1u, 4096u);
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
            float height =
                (0.035f + mountains * (0.22f + rear * 0.80f) + ridges * rear * 0.16f - basin * 0.16f - edge * 0.12f) *
                terrain.height_scale;
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
            const float shore =
                1.0f - smoothstep(terrain.height_scale * 0.015f, terrain.height_scale * 0.10f, terrain.heights[index]);
            const float rock = saturate(smoothstep(0.10f, 0.38f, slope) + smoothstep(0.62f, 0.92f, height01));
            const float sand = shore * (1.0f - rock);
            const float dirt =
                saturate(smoothstep(0.04f, 0.22f, slope) * (1.0f - rock) +
                         std::abs(fbm(static_cast<float>(x) * 0.025f, static_cast<float>(z) * 0.025f, 3)) * 0.22f);
            const float grass = std::max(0.0f, 1.0f - rock - sand - dirt * 0.55f);
            terrain.layer_weights[index] = normalized_weights({grass, dirt, rock, sand});
        }
    }
    ++terrain.content_revision;
}

terrain_authoring_result resample_terrain(terrain_component& terrain, const terrain_resample_descriptor& descriptor)
{
    if (!terrain_heightfield_valid(terrain)) return {false, "terrain heightfield is invalid"};
    if (!terrain_resolution_supported(descriptor.sample_resolution)) return {false, "target resolution is unsupported"};
    if (!std::isfinite(descriptor.physical_size) || descriptor.physical_size < 1.0f ||
        descriptor.physical_size > 262144.0f)
        return {false, "target terrain size must be finite and in [1, 262144] metres"};
    const auto source_resolution = terrain.subdivisions + 1u;
    if (source_resolution != descriptor.sample_resolution)
    {
        const auto source_weights = terrain.layer_weights;
        terrain.heights = resample_square_field(terrain.heights, source_resolution, descriptor.sample_resolution);
        std::array<std::vector<float>, 4> channels;
        for (std::size_t channel = 0; channel < channels.size(); ++channel)
        {
            std::vector<float> source(static_cast<std::size_t>(source_resolution) * source_resolution);
            for (std::size_t sample = 0; sample < source.size(); ++sample)
                source[sample] = static_cast<float>(source_weights[sample][channel]) / 255.0f;
            channels[channel] = resample_square_field(source, source_resolution, descriptor.sample_resolution);
        }
        terrain.layer_weights.resize(static_cast<std::size_t>(descriptor.sample_resolution) *
                                     descriptor.sample_resolution);
        for (std::size_t sample = 0; sample < terrain.layer_weights.size(); ++sample)
            terrain.layer_weights[sample] = normalized_weights(
                {channels[0][sample], channels[1][sample], channels[2][sample], channels[3][sample]});
        terrain.subdivisions = descriptor.sample_resolution - 1u;
    }
    terrain.size = descriptor.physical_size;
    ++terrain.content_revision;
    return {true, {}};
}

terrain_authoring_result import_terrain_heightmap(terrain_component& terrain, const terrain_heightmap& heightmap,
                                                  const terrain_heightmap_import_settings& settings)
{
    if (heightmap.width < 2u || heightmap.height < 2u ||
        heightmap.samples.size() != static_cast<std::size_t>(heightmap.width) * heightmap.height)
        return {false, "heightmap dimensions and sample payload do not agree"};
    if (!terrain_resolution_supported(settings.target_resolution)) return {false, "target resolution is unsupported"};
    if (!std::isfinite(settings.minimum_elevation) || !std::isfinite(settings.maximum_elevation) ||
        settings.maximum_elevation <= settings.minimum_elevation)
        return {false, "heightmap import elevation range must be finite and increasing"};

    const auto [minimum, maximum] = std::minmax_element(heightmap.samples.begin(), heightmap.samples.end());
    const float source_minimum = settings.normalize_source_range ? static_cast<float>(*minimum) : 0.0f;
    const float source_maximum = settings.normalize_source_range ? static_cast<float>(*maximum) : 65535.0f;
    const float source_range = std::max(source_maximum - source_minimum, 1.0f);
    std::vector<float> source(static_cast<std::size_t>(heightmap.width) * heightmap.height);
    for (std::uint32_t z = 0; z < heightmap.height; ++z)
        for (std::uint32_t x = 0; x < heightmap.width; ++x)
        {
            const auto source_x = settings.flip_x ? heightmap.width - 1u - x : x;
            const auto source_z = settings.flip_z ? heightmap.height - 1u - z : z;
            const float normalized =
                (static_cast<float>(
                     heightmap.samples[static_cast<std::size_t>(source_z) * heightmap.width + source_x]) -
                 source_minimum) /
                source_range;
            source[static_cast<std::size_t>(z) * heightmap.width + x] =
                std::lerp(settings.minimum_elevation, settings.maximum_elevation, saturate(normalized));
        }
    terrain.heights = resample_scalar_field(source, heightmap.width, heightmap.height, settings.target_resolution,
                                            settings.target_resolution);
    terrain.layer_weights.assign(terrain.heights.size(), {255u, 0u, 0u, 0u});
    terrain.subdivisions = settings.target_resolution - 1u;
    terrain.size = settings.physical_size;
    terrain.height_scale = settings.maximum_elevation - settings.minimum_elevation;
    ++terrain.content_revision;
    return {true, {}};
}

terrain_heightmap export_terrain_heightmap(const terrain_component& terrain,
                                           const terrain_heightmap_export_settings& settings)
{
    terrain_heightmap result;
    if (!terrain_heightfield_valid(terrain) || !std::isfinite(settings.minimum_elevation) ||
        !std::isfinite(settings.maximum_elevation) || settings.maximum_elevation <= settings.minimum_elevation)
        return result;
    result.width = terrain.subdivisions + 1u;
    result.height = result.width;
    result.encoded_minimum_elevation = settings.minimum_elevation;
    result.encoded_maximum_elevation = settings.maximum_elevation;
    result.samples.resize(terrain.heights.size());
    const float inverse_range = 1.0f / (settings.maximum_elevation - settings.minimum_elevation);
    for (std::size_t sample = 0; sample < terrain.heights.size(); ++sample)
        result.samples[sample] = static_cast<std::uint16_t>(
            std::lround(saturate((terrain.heights[sample] - settings.minimum_elevation) * inverse_range) * 65535.0f));
    return result;
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
    if (!terrain_heightfield_valid(terrain)) return {0.0f, 1.0f, 0.0f};
    const auto x =
        static_cast<std::uint32_t>(std::round(saturate(local_x / terrain.size + 0.5f) * terrain.subdivisions));
    const auto z =
        static_cast<std::uint32_t>(std::round(saturate(local_z / terrain.size + 0.5f) * terrain.subdivisions));
    return normal_at(terrain, x, z);
}

terrain_dirty_region apply_terrain_brush(terrain_component& terrain, const math::vector3f& local_center,
                                         const terrain_brush_settings& settings, float delta_seconds)
{
    terrain_dirty_region dirty{};
    if (!terrain_heightfield_valid(terrain) || settings.radius <= 0.0f || settings.strength <= 0.0f) return dirty;
    const float spacing = terrain.size / terrain.subdivisions;
    const float center_x = (local_center[0] / terrain.size + 0.5f) * terrain.subdivisions;
    const float center_z = (local_center[2] / terrain.size + 0.5f) * terrain.subdivisions;
    const float radius_samples = settings.radius / spacing;
    const auto min_x = static_cast<std::uint32_t>(std::max(0.0f, std::floor(center_x - radius_samples)));
    const auto min_z = static_cast<std::uint32_t>(std::max(0.0f, std::floor(center_z - radius_samples)));
    const auto max_x = static_cast<std::uint32_t>(
        std::min(static_cast<float>(terrain.subdivisions), std::ceil(center_x + radius_samples)));
    const auto max_z = static_cast<std::uint32_t>(
        std::min(static_cast<float>(terrain.subdivisions), std::ceil(center_z + radius_samples)));
    const auto scratch_min_x = min_x > 0u ? min_x - 1u : 0u;
    const auto scratch_min_z = min_z > 0u ? min_z - 1u : 0u;
    const auto scratch_max_x = std::min(max_x + 1u, terrain.subdivisions);
    const auto scratch_max_z = std::min(max_z + 1u, terrain.subdivisions);
    const auto scratch_width = scratch_max_x - scratch_min_x + 1u;
    std::vector<float> original_heights;
    if (settings.tool == terrain_brush_tool::smooth)
    {
        original_heights.reserve(static_cast<std::size_t>(scratch_width) * (scratch_max_z - scratch_min_z + 1u));
        for (std::uint32_t z = scratch_min_z; z <= scratch_max_z; ++z)
        {
            const auto begin =
                terrain.heights.begin() + static_cast<std::ptrdiff_t>(sample_index(terrain, scratch_min_x, z));
            original_heights.insert(original_heights.end(), begin, begin + scratch_width);
        }
    }
    const auto scratch_height = [&](std::uint32_t x, std::uint32_t z)
    { return original_heights[static_cast<std::size_t>(z - scratch_min_z) * scratch_width + (x - scratch_min_x)]; };
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
                terrain.heights[index] =
                    std::lerp(terrain.heights[index], settings.flatten_height, saturate(amount * 8.0f));
            else if (settings.tool == terrain_brush_tool::smooth)
            {
                float total{};
                float samples{};
                for (int oz = -1; oz <= 1; ++oz)
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        const auto sx = static_cast<std::uint32_t>(
                            std::clamp<int>(static_cast<int>(x) + ox, 0, terrain.subdivisions));
                        const auto sz = static_cast<std::uint32_t>(
                            std::clamp<int>(static_cast<int>(z) + oz, 0, terrain.subdivisions));
                        total += scratch_height(sx, sz);
                        ++samples;
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
                for (std::size_t layer = 0; layer < 4; ++layer)
                    if (layer != active) other_total += weights[layer];
                for (std::size_t layer = 0; layer < 4; ++layer)
                    if (layer != active)
                        weights[layer] =
                            other_total > 0.0001f ? weights[layer] / other_total * remaining : remaining / 3.0f;
                terrain.layer_weights[index] = normalized_weights(weights);
            }
            changed = true;
        }
    }
    if (changed)
    {
        ++terrain.content_revision;
        dirty = {min_x > 0 ? min_x - 1u : 0u,
                 min_z > 0 ? min_z - 1u : 0u,
                 std::min(max_x + 1u, terrain.subdivisions),
                 std::min(max_z + 1u, terrain.subdivisions),
                 true,
                 settings.tool != terrain_brush_tool::paint,
                 settings.tool == terrain_brush_tool::paint};
    }
    return dirty;
}

terrain_raycast_hit raycast_terrain(const terrain_component& terrain, const math::vector3f& local_origin,
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
            const math::vector3f a{-half + x * spacing, height_at(terrain, x, z), -half + z * spacing};
            const math::vector3f b{a[0] + spacing, height_at(terrain, x + 1u, z), a[2]};
            const math::vector3f c{a[0] + spacing, height_at(terrain, x + 1u, z + 1u), a[2] + spacing};
            const math::vector3f d{a[0], height_at(terrain, x, z + 1u), a[2] + spacing};
            float distance{};
            if ((intersect_triangle(local_origin, local_direction, a, b, c, distance) ||
                 intersect_triangle(local_origin, local_direction, a, c, d, distance)) &&
                distance < nearest)
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

terrain_raycast_hit raycast_terrain(const terrain_component& terrain, const render::terrain_hierarchy& hierarchy,
                                    const math::vector3f& local_origin, const math::vector3f& local_direction) noexcept
{
    terrain_raycast_hit result{};
    if (!terrain_heightfield_valid(terrain) || hierarchy.root == render::invalid_terrain_node ||
        hierarchy.root >= hierarchy.nodes.size())
        return result;
    float closest = std::numeric_limits<float>::max();
    std::vector<std::uint32_t> stack{hierarchy.root};
    const float spacing = terrain.size / terrain.subdivisions;
    const float half = terrain.size * 0.5f;
    while (!stack.empty())
    {
        const auto index = stack.back();
        stack.pop_back();
        if (index >= hierarchy.nodes.size()) continue;
        const auto& node = hierarchy.nodes[index];
        if (!intersect_bounds(local_origin, local_direction, node.local_bounds, closest)) continue;
        if (!node.leaf())
        {
            for (auto child = node.children.rbegin(); child != node.children.rend(); ++child)
                if (*child != render::invalid_terrain_node) stack.push_back(*child);
            continue;
        }
        for (std::uint32_t z = node.samples.min_z; z < node.samples.max_z; ++z)
            for (std::uint32_t x = node.samples.min_x; x < node.samples.max_x; ++x)
            {
                const auto position = [&](std::uint32_t sx, std::uint32_t sz)
                {
                    return math::vector3f{-half + static_cast<float>(sx) * spacing, height_at(terrain, sx, sz),
                                          -half + static_cast<float>(sz) * spacing};
                };
                const auto a = position(x, z);
                const auto b = position(x + 1u, z);
                const auto c = position(x + 1u, z + 1u);
                const auto d = position(x, z + 1u);
                float distance{};
                math::vector3f normal{};
                if (intersect_triangle(local_origin, local_direction, a, b, c, distance))
                    normal = math::normalize(math::cross(math::sub(b, a), math::sub(c, a)));
                else if (intersect_triangle(local_origin, local_direction, a, c, d, distance))
                    normal = math::normalize(math::cross(math::sub(c, a), math::sub(d, a)));
                else
                    continue;
                if (distance < closest)
                {
                    closest = distance;
                    result = {.position = math::add(local_origin, math::mul(local_direction, distance)),
                              .normal = normal,
                              .distance = distance,
                              .hit = true};
                }
            }
    }
    return result;
}

terrain_render_proxy* terrain_render_proxy_cache::find(ecs::entity_guid guid) noexcept
{
    const auto found = proxies_.find(guid);
    return found == proxies_.end() ? nullptr : &found->second;
}

const terrain_render_proxy* terrain_render_proxy_cache::find(ecs::entity_guid guid) const noexcept
{
    const auto found = proxies_.find(guid);
    return found == proxies_.end() ? nullptr : &found->second;
}

bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_component& terrain,
                                             render::renderer& renderer, const terrain_dirty_region* dirty_region)
{
    if (!guid.valid() || !terrain_heightfield_valid(terrain)) return false;
    auto& proxy = proxies_[guid];
    if (!renderer.terrain_alive(proxy.handle))
    {
        render::terrain_resource_descriptor descriptor;
        descriptor.sample_resolution = terrain.subdivisions + 1u;
        descriptor.width = terrain.size;
        descriptor.depth = terrain.size;
        descriptor.heights = terrain.heights;
        descriptor.weights = terrain.layer_weights;
        descriptor.material = terrain.material;
        descriptor.lod = {.patch_quads = terrain.patch_quads,
                          .maximum_hierarchy_depth = terrain.maximum_hierarchy_depth,
                          .geometric_error_multiplier = terrain.geometric_error_multiplier};
        descriptor.content_revision = terrain.content_revision;
        descriptor.name = "terrain";
        proxy.handle = renderer.create_terrain(std::move(descriptor));
        proxy.synchronized_revision = terrain.content_revision;
        proxy.material = terrain.material;
        return proxy.handle.valid();
    }
    if (proxy.synchronized_revision == terrain.content_revision && proxy.material == terrain.material) return true;
    if (!dirty_region || !dirty_region->valid)
    {
        if (proxy.synchronized_revision != terrain.content_revision)
        {
            arc::diagnostics::warn("scene.terrain",
                                   "Terrain content revision changed without a dirty region; performing a full "
                                   "resource resynchronization");
            (void)renderer.destroy_terrain(proxy.handle);
            render::terrain_resource_descriptor descriptor;
            descriptor.sample_resolution = terrain.subdivisions + 1u;
            descriptor.width = terrain.size;
            descriptor.depth = terrain.size;
            descriptor.heights = terrain.heights;
            descriptor.weights = terrain.layer_weights;
            descriptor.material = terrain.material;
            descriptor.lod = {.patch_quads = terrain.patch_quads,
                              .maximum_hierarchy_depth = terrain.maximum_hierarchy_depth,
                              .geometric_error_multiplier = terrain.geometric_error_multiplier};
            descriptor.content_revision = terrain.content_revision;
            descriptor.name = "terrain";
            proxy.handle = renderer.create_terrain(std::move(descriptor));
            proxy.synchronized_revision = terrain.content_revision;
            proxy.material = terrain.material;
            return proxy.handle.valid();
        }
        const bool updated = renderer.update_terrain(proxy.handle, terrain.material,
                                                     {.patch_quads = terrain.patch_quads,
                                                      .maximum_hierarchy_depth = terrain.maximum_hierarchy_depth,
                                                      .geometric_error_multiplier = terrain.geometric_error_multiplier},
                                                     terrain.content_revision);
        if (updated)
        {
            proxy.synchronized_revision = terrain.content_revision;
            proxy.material = terrain.material;
        }
        return updated;
    }

    const render::terrain_sample_region region{dirty_region->min_x, dirty_region->min_z, dirty_region->max_x,
                                               dirty_region->max_z};
    const auto width = region.width();
    const auto height = region.height();
    bool updated = true;
    if (dirty_region->heights_changed)
    {
        render::terrain_height_region_update request{
            .region = region, .row_stride = width, .content_revision = terrain.content_revision};
        request.values.reserve(static_cast<std::size_t>(width) * height);
        for (std::uint32_t z = region.min_z; z <= region.max_z; ++z)
        {
            const auto begin =
                terrain.heights.begin() +
                static_cast<std::ptrdiff_t>(static_cast<std::size_t>(z) * (terrain.subdivisions + 1u) + region.min_x);
            request.values.insert(request.values.end(), begin, begin + width);
        }
        updated = renderer.update_terrain_heights(proxy.handle, std::move(request));
    }
    if (updated && dirty_region->weights_changed)
    {
        render::terrain_weight_region_update request{
            .region = region, .row_stride = width, .content_revision = terrain.content_revision};
        request.values.reserve(static_cast<std::size_t>(width) * height);
        for (std::uint32_t z = region.min_z; z <= region.max_z; ++z)
        {
            const auto begin =
                terrain.layer_weights.begin() +
                static_cast<std::ptrdiff_t>(static_cast<std::size_t>(z) * (terrain.subdivisions + 1u) + region.min_x);
            request.values.insert(request.values.end(), begin, begin + width);
        }
        updated = renderer.update_terrain_weights(proxy.handle, std::move(request));
    }
    if (updated)
    {
        proxy.synchronized_revision = terrain.content_revision;
        proxy.material = terrain.material;
    }
    return updated;
}

bool terrain_render_proxy_cache::erase(ecs::entity_guid guid, render::renderer& renderer)
{
    const auto found = proxies_.find(guid);
    if (found == proxies_.end()) return false;
    if (renderer.terrain_alive(found->second.handle)) (void)renderer.destroy_terrain(found->second.handle);
    proxies_.erase(found);
    return true;
}

void terrain_render_proxy_cache::release_missing(std::span<const ecs::entity_guid> active, render::renderer& renderer)
{
    for (auto found = proxies_.begin(); found != proxies_.end();)
    {
        if (std::find(active.begin(), active.end(), found->first) != active.end())
        {
            ++found;
            continue;
        }
        if (renderer.terrain_alive(found->second.handle)) (void)renderer.destroy_terrain(found->second.handle);
        found = proxies_.erase(found);
    }
}

void terrain_render_proxy_cache::clear(render::renderer& renderer)
{
    for (const auto& [guid, proxy] : proxies_)
    {
        (void)guid;
        if (renderer.terrain_alive(proxy.handle)) (void)renderer.destroy_terrain(proxy.handle);
    }
    proxies_.clear();
}

} // namespace arc::scene
