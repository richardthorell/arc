#include <arc/scene/terrain_surface_ir.h>

#include <arc/scene/terrain.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace arc::scene
{
namespace
{

class stable_hash64
{
public:
    void append_byte(std::uint8_t value) noexcept
    {
        value_ ^= value;
        value_ *= 1099511628211ull;
    }

    void append_u32(std::uint32_t value) noexcept
    {
        for (std::uint32_t shift = 0; shift < 32u; shift += 8u)
            append_byte(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }

    void append_u64(std::uint64_t value) noexcept
    {
        for (std::uint32_t shift = 0; shift < 64u; shift += 8u)
            append_byte(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }

    void append_float(float value) noexcept
    {
        append_u32(std::bit_cast<std::uint32_t>(value));
    }

    void append_double(double value) noexcept
    {
        append_u64(std::bit_cast<std::uint64_t>(value));
    }

    [[nodiscard]] std::uint64_t value() const noexcept
    {
        return value_;
    }

private:
    std::uint64_t value_{14695981039346656037ull};
};

void append_bounds(stable_hash64& hash, const terrain_world_bounds& bounds) noexcept
{
    hash.append_double(bounds.min_x);
    hash.append_double(bounds.min_y);
    hash.append_double(bounds.min_z);
    hash.append_double(bounds.max_x);
    hash.append_double(bounds.max_y);
    hash.append_double(bounds.max_z);
}

} // namespace

terrain_surface_ir terrain_evaluated_surface::view() const noexcept
{
    terrain_surface_ir result;
    result.schema_version = schema_version;
    result.source_revision = source_revision;
    result.local_bounds = local_bounds;
    if (const auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&geometry))
    {
        result.geometry = terrain_surface_heightfield_ir{
            .sample_width = heightfield->sample_width,
            .sample_height = heightfield->sample_height,
            .width = heightfield->width,
            .depth = heightfield->depth,
            .heights = heightfield->heights,
            .material_weights = heightfield->material_weights,
        };
    }
    else
    {
        const auto& mesh = std::get<terrain_evaluated_mesh>(geometry);
        result.geometry = terrain_surface_mesh_ir{.positions = mesh.positions, .indices = mesh.indices};
    }
    return result;
}

bool validate_terrain_surface_ir(const terrain_surface_ir& surface) noexcept
{
    if (surface.schema_version != terrain_surface_ir::current_schema_version || !surface.local_bounds.valid())
        return false;

    if (const auto* heightfield = std::get_if<terrain_surface_heightfield_ir>(&surface.geometry))
    {
        if (heightfield->sample_width < 2u || heightfield->sample_height < 2u || !std::isfinite(heightfield->width) ||
            !std::isfinite(heightfield->depth) || heightfield->width <= 0.0f || heightfield->depth <= 0.0f)
            return false;
        const auto sample_count = static_cast<std::size_t>(heightfield->sample_width) * heightfield->sample_height;
        if (heightfield->heights.size() != sample_count || heightfield->material_weights.size() != sample_count)
            return false;
        return std::all_of(heightfield->heights.begin(), heightfield->heights.end(),
                           [](float value) { return std::isfinite(value); });
    }

    const auto* mesh = std::get_if<terrain_surface_mesh_ir>(&surface.geometry);
    if (!mesh || mesh->positions.empty() || mesh->indices.empty() || mesh->indices.size() % 3u != 0u) return false;
    if (!std::all_of(mesh->positions.begin(), mesh->positions.end(), [](const math::vector3f& value)
                     { return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]); }))
        return false;
    return std::all_of(mesh->indices.begin(), mesh->indices.end(),
                       [&](std::uint32_t index) { return index < mesh->positions.size(); });
}

std::optional<terrain_triangle_geometry>
canonicalize_terrain_surface_geometry(const terrain_surface_ir& surface)
{
    if (!validate_terrain_surface_ir(surface)) return std::nullopt;

    terrain_triangle_geometry result;
    if (const auto* mesh = std::get_if<terrain_surface_mesh_ir>(&surface.geometry))
    {
        if (mesh->positions.size() > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
        result.positions.assign(mesh->positions.begin(), mesh->positions.end());
        result.indices.assign(mesh->indices.begin(), mesh->indices.end());
        return result;
    }

    const auto& heightfield = std::get<terrain_surface_heightfield_ir>(surface.geometry);
    const auto sample_count = static_cast<std::size_t>(heightfield.sample_width) * heightfield.sample_height;
    if (sample_count > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;

    const auto quad_count = static_cast<std::size_t>(heightfield.sample_width - 1u) *
                            static_cast<std::size_t>(heightfield.sample_height - 1u);
    if (quad_count > std::numeric_limits<std::size_t>::max() / 6u) return std::nullopt;

    result.positions.reserve(sample_count);
    result.indices.reserve(quad_count * 6u);

    const float half_width = heightfield.width * 0.5f;
    const float half_depth = heightfield.depth * 0.5f;
    const float x_denominator = static_cast<float>(heightfield.sample_width - 1u);
    const float z_denominator = static_cast<float>(heightfield.sample_height - 1u);

    for (std::uint32_t z = 0; z < heightfield.sample_height; ++z)
        for (std::uint32_t x = 0; x < heightfield.sample_width; ++x)
        {
            const auto index = static_cast<std::size_t>(z) * heightfield.sample_width + x;
            const float local_x = -half_width + heightfield.width * static_cast<float>(x) / x_denominator;
            const float local_z = -half_depth + heightfield.depth * static_cast<float>(z) / z_denominator;
            result.positions.push_back({local_x, heightfield.heights[index], local_z});
        }

    for (std::uint32_t z = 0; z + 1u < heightfield.sample_height; ++z)
        for (std::uint32_t x = 0; x + 1u < heightfield.sample_width; ++x)
        {
            const auto top_left = z * heightfield.sample_width + x;
            const auto top_right = top_left + 1u;
            const auto bottom_left = top_left + heightfield.sample_width;
            const auto bottom_right = bottom_left + 1u;

            // Counter-clockwise winding when viewed from +Y, matching the engine's conventional mesh convention.
            result.indices.push_back(top_left);
            result.indices.push_back(bottom_left);
            result.indices.push_back(top_right);
            result.indices.push_back(top_right);
            result.indices.push_back(bottom_left);
            result.indices.push_back(bottom_right);
        }

    return result;
}

std::optional<terrain_evaluated_surface> copy_terrain_surface_ir(const terrain_surface_ir& surface)
{
    if (!validate_terrain_surface_ir(surface)) return std::nullopt;

    terrain_evaluated_surface result;
    result.schema_version = surface.schema_version;
    result.source_revision = surface.source_revision;
    result.local_bounds = surface.local_bounds;
    if (const auto* heightfield = std::get_if<terrain_surface_heightfield_ir>(&surface.geometry))
    {
        terrain_evaluated_heightfield owned;
        owned.sample_width = heightfield->sample_width;
        owned.sample_height = heightfield->sample_height;
        owned.width = heightfield->width;
        owned.depth = heightfield->depth;
        owned.heights.assign(heightfield->heights.begin(), heightfield->heights.end());
        owned.material_weights.assign(heightfield->material_weights.begin(), heightfield->material_weights.end());
        result.geometry = std::move(owned);
    }
    else
    {
        const auto& mesh = std::get<terrain_surface_mesh_ir>(surface.geometry);
        terrain_evaluated_mesh owned;
        owned.positions.assign(mesh.positions.begin(), mesh.positions.end());
        owned.indices.assign(mesh.indices.begin(), mesh.indices.end());
        result.geometry = std::move(owned);
    }
    return result;
}

std::uint64_t terrain_surface_fingerprint(const terrain_surface_ir& surface) noexcept
{
    if (!validate_terrain_surface_ir(surface)) return 0;

    stable_hash64 hash;
    hash.append_u32(surface.schema_version);
    hash.append_u64(surface.source_revision);
    append_bounds(hash, surface.local_bounds);

    if (const auto* heightfield = std::get_if<terrain_surface_heightfield_ir>(&surface.geometry))
    {
        hash.append_byte(0u);
        hash.append_u32(heightfield->sample_width);
        hash.append_u32(heightfield->sample_height);
        hash.append_float(heightfield->width);
        hash.append_float(heightfield->depth);
        for (const auto height : heightfield->heights)
            hash.append_float(height);
        for (const auto& weights : heightfield->material_weights)
            for (const auto weight : weights)
                hash.append_byte(weight);
    }
    else
    {
        hash.append_byte(1u);
        const auto& mesh = std::get<terrain_surface_mesh_ir>(surface.geometry);
        hash.append_u64(mesh.positions.size());
        hash.append_u64(mesh.indices.size());
        for (const auto& position : mesh.positions)
        {
            hash.append_float(position[0]);
            hash.append_float(position[1]);
            hash.append_float(position[2]);
        }
        for (const auto index : mesh.indices)
            hash.append_u32(index);
    }
    return hash.value();
}

std::optional<terrain_surface_ir> make_legacy_terrain_surface_ir(const terrain_component& terrain) noexcept
{
    if (!terrain_heightfield_valid(terrain)) return std::nullopt;

    const auto [minimum, maximum] = std::minmax_element(terrain.heights.begin(), terrain.heights.end());
    const auto half_size = static_cast<double>(terrain.size) * 0.5;
    terrain_surface_ir result;
    result.source_revision = terrain.content_revision;
    result.local_bounds = {-half_size, static_cast<double>(*minimum), -half_size,
                           half_size,  static_cast<double>(*maximum), half_size};
    result.geometry = terrain_surface_heightfield_ir{
        .sample_width = terrain.subdivisions + 1u,
        .sample_height = terrain.subdivisions + 1u,
        .width = terrain.size,
        .depth = terrain.size,
        .heights = terrain.heights,
        .material_weights = terrain.layer_weights,
    };
    return validate_terrain_surface_ir(result) ? std::optional<terrain_surface_ir>{result} : std::nullopt;
}

} // namespace arc::scene
