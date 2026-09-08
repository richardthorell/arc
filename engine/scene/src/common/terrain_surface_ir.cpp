#include <arc/scene/terrain_surface_ir.h>

#include <arc/scene/terrain.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace arc::scene
{

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
