#include <arc/scene/terrain_render_regions.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace arc::scene
{
namespace
{

constexpr float normal_epsilon = 1.0e-12f;

class stable_hash64
{
public:
    void byte(std::uint8_t value) noexcept
    {
        value_ ^= value;
        value_ *= 1099511628211ull;
    }

    void u32(std::uint32_t value) noexcept
    {
        for (std::uint32_t shift = 0; shift < 32u; shift += 8u)
            byte(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }

    void u64(std::uint64_t value) noexcept
    {
        for (std::uint32_t shift = 0; shift < 64u; shift += 8u)
            byte(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }

    void f32(float value) noexcept
    {
        u32(std::bit_cast<std::uint32_t>(value));
    }

    void f64(double value) noexcept
    {
        u64(std::bit_cast<std::uint64_t>(value));
    }

    [[nodiscard]] std::uint64_t value() const noexcept
    {
        return value_ == 0u ? 1u : value_;
    }

private:
    std::uint64_t value_{14695981039346656037ull};
};

void append_region_identity(stable_hash64& hash, terrain_region_id id, const terrain_world_bounds& bounds) noexcept
{
    hash.u64(static_cast<std::uint64_t>(id.x));
    hash.u64(static_cast<std::uint64_t>(id.z));
    hash.f64(bounds.min_x);
    hash.f64(bounds.min_z);
    hash.f64(bounds.max_x);
    hash.f64(bounds.max_z);
}

std::uint32_t region_quad_span(float extent, std::uint32_t total_quads, double target_size) noexcept
{
    const double spacing = static_cast<double>(extent) / static_cast<double>(total_quads);
    if (!std::isfinite(spacing) || spacing <= 0.0) return total_quads;
    const double desired = std::clamp(std::floor(target_size / spacing), 1.0, static_cast<double>(total_quads));
    return static_cast<std::uint32_t>(desired);
}

std::vector<math::vector3f> source_normals(const terrain_surface_ir& surface)
{
    const auto canonical = canonicalize_terrain_surface_geometry(surface);
    if (!canonical) return {};

    std::vector<math::vector3f> normals(canonical->positions.size());
    for (std::size_t index = 0; index + 2u < canonical->indices.size(); index += 3u)
    {
        const auto i0 = canonical->indices[index + 0u];
        const auto i1 = canonical->indices[index + 1u];
        const auto i2 = canonical->indices[index + 2u];
        const auto edge0 = math::sub(canonical->positions[i1], canonical->positions[i0]);
        const auto edge1 = math::sub(canonical->positions[i2], canonical->positions[i0]);
        const auto face = math::cross(edge0, edge1);
        if (math::length_squared(face) <= normal_epsilon) continue;
        normals[i0] = math::add(normals[i0], face);
        normals[i1] = math::add(normals[i1], face);
        normals[i2] = math::add(normals[i2], face);
    }
    for (auto& normal : normals)
        normal =
            math::length_squared(normal) > normal_epsilon ? math::normalize(normal) : math::vector3f{0.0f, 1.0f, 0.0f};
    return normals;
}

std::uint64_t geometry_fingerprint(terrain_region_id id, const terrain_evaluated_surface& surface,
                                   std::span<const math::vector3f> vertex_normals = {}) noexcept
{
    stable_hash64 hash;
    append_region_identity(hash, id, surface.local_bounds);
    if (const auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&surface.geometry))
    {
        hash.byte(0u);
        hash.u32(heightfield->sample_width);
        hash.u32(heightfield->sample_height);
        hash.f32(heightfield->width);
        hash.f32(heightfield->depth);
        for (const auto height : heightfield->heights)
            hash.f32(height);
    }
    else
    {
        hash.byte(1u);
        const auto& mesh = std::get<terrain_evaluated_mesh>(surface.geometry);
        hash.u64(mesh.positions.size());
        hash.u64(mesh.indices.size());
        for (const auto& position : mesh.positions)
        {
            hash.f32(position[0]);
            hash.f32(position[1]);
            hash.f32(position[2]);
        }
        for (const auto index : mesh.indices)
            hash.u32(index);
    }
    hash.u64(vertex_normals.size());
    for (const auto& normal : vertex_normals)
    {
        hash.f32(normal[0]);
        hash.f32(normal[1]);
        hash.f32(normal[2]);
    }
    return hash.value();
}

std::uint64_t attribute_fingerprint(terrain_region_id id, const terrain_evaluated_surface& surface) noexcept
{
    stable_hash64 hash;
    append_region_identity(hash, id, surface.local_bounds);
    if (const auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&surface.geometry))
    {
        hash.u32(heightfield->sample_width);
        hash.u32(heightfield->sample_height);
        for (const auto& weights : heightfield->material_weights)
            for (const auto weight : weights)
                hash.byte(weight);
    }
    else
    {
        hash.byte(0u);
    }
    return hash.value();
}

} // namespace

std::vector<terrain_render_region> build_terrain_render_regions(const terrain_surface_ir& surface,
                                                                double target_region_size)
{
    if (!validate_terrain_surface_ir(surface) || !std::isfinite(target_region_size) || target_region_size <= 0.0)
        return {};

    if (!std::holds_alternative<terrain_surface_heightfield_ir>(surface.geometry))
    {
        auto owned = copy_terrain_surface_ir(surface);
        if (!owned) return {};
        terrain_render_region region;
        region.id = {};
        region.surface = std::move(*owned);
        region.geometry_fingerprint = geometry_fingerprint(region.id, region.surface);
        region.attribute_fingerprint = attribute_fingerprint(region.id, region.surface);
        return {std::move(region)};
    }

    const auto& source = std::get<terrain_surface_heightfield_ir>(surface.geometry);
    const auto normals = source_normals(surface);
    if (normals.size() != source.heights.size()) return {};

    const auto total_quads_x = source.sample_width - 1u;
    const auto total_quads_z = source.sample_height - 1u;
    const auto quads_per_region_x = region_quad_span(source.width, total_quads_x, target_region_size);
    const auto quads_per_region_z = region_quad_span(source.depth, total_quads_z, target_region_size);
    const auto region_count_x = (total_quads_x + quads_per_region_x - 1u) / quads_per_region_x;
    const auto region_count_z = (total_quads_z + quads_per_region_z - 1u) / quads_per_region_z;

    std::vector<terrain_render_region> result;
    result.reserve(static_cast<std::size_t>(region_count_x) * region_count_z);

    const double full_extent_x = surface.local_bounds.max_x - surface.local_bounds.min_x;
    const double full_extent_z = surface.local_bounds.max_z - surface.local_bounds.min_z;
    for (std::uint32_t region_z = 0; region_z < region_count_z; ++region_z)
        for (std::uint32_t region_x = 0; region_x < region_count_x; ++region_x)
        {
            const auto start_x = region_x * quads_per_region_x;
            const auto start_z = region_z * quads_per_region_z;
            const auto end_x = std::min(start_x + quads_per_region_x, total_quads_x);
            const auto end_z = std::min(start_z + quads_per_region_z, total_quads_z);
            const auto sample_width = end_x - start_x + 1u;
            const auto sample_height = end_z - start_z + 1u;

            terrain_render_region region;
            region.id = {static_cast<std::int64_t>(region_x), static_cast<std::int64_t>(region_z)};
            region.surface.schema_version = surface.schema_version;
            region.surface.source_revision = surface.source_revision;
            region.surface.local_bounds.min_x =
                surface.local_bounds.min_x + full_extent_x * static_cast<double>(start_x) / total_quads_x;
            region.surface.local_bounds.max_x =
                surface.local_bounds.min_x + full_extent_x * static_cast<double>(end_x) / total_quads_x;
            region.surface.local_bounds.min_z =
                surface.local_bounds.min_z + full_extent_z * static_cast<double>(start_z) / total_quads_z;
            region.surface.local_bounds.max_z =
                surface.local_bounds.min_z + full_extent_z * static_cast<double>(end_z) / total_quads_z;

            terrain_evaluated_heightfield heightfield;
            heightfield.sample_width = sample_width;
            heightfield.sample_height = sample_height;
            heightfield.width =
                static_cast<float>(region.surface.local_bounds.max_x - region.surface.local_bounds.min_x);
            heightfield.depth =
                static_cast<float>(region.surface.local_bounds.max_z - region.surface.local_bounds.min_z);
            heightfield.heights.reserve(static_cast<std::size_t>(sample_width) * sample_height);
            heightfield.material_weights.reserve(static_cast<std::size_t>(sample_width) * sample_height);
            region.vertex_normals.reserve(static_cast<std::size_t>(sample_width) * sample_height);

            float minimum_height = std::numeric_limits<float>::max();
            float maximum_height = std::numeric_limits<float>::lowest();
            for (std::uint32_t z = start_z; z <= end_z; ++z)
                for (std::uint32_t x = start_x; x <= end_x; ++x)
                {
                    const auto source_index = static_cast<std::size_t>(z) * source.sample_width + x;
                    const auto height = source.heights[source_index];
                    heightfield.heights.push_back(height);
                    heightfield.material_weights.push_back(source.material_weights[source_index]);
                    region.vertex_normals.push_back(normals[source_index]);
                    minimum_height = std::min(minimum_height, height);
                    maximum_height = std::max(maximum_height, height);
                }

            region.surface.local_bounds.min_y = minimum_height;
            region.surface.local_bounds.max_y = maximum_height;
            region.surface.geometry = std::move(heightfield);
            region.geometry_fingerprint = geometry_fingerprint(region.id, region.surface, region.vertex_normals);
            region.attribute_fingerprint = attribute_fingerprint(region.id, region.surface);
            result.push_back(std::move(region));
        }
    return result;
}

std::uint64_t terrain_render_region_instance_id(terrain_region_id id) noexcept
{
    stable_hash64 hash;
    hash.u64(static_cast<std::uint64_t>(id.x));
    hash.u64(static_cast<std::uint64_t>(id.z));
    return hash.value();
}

} // namespace arc::scene
