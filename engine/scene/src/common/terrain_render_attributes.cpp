#include <arc/scene/terrain_render_attributes.h>

#include <variant>

namespace arc::scene
{

std::optional<terrain_render_attributes> build_terrain_render_attributes(const terrain_surface_ir& surface)
{
    if (!validate_terrain_surface_ir(surface)) return std::nullopt;

    terrain_render_attributes result;
    if (const auto* heightfield = std::get_if<terrain_surface_heightfield_ir>(&surface.geometry))
    {
        result.width = heightfield->sample_width;
        result.height = heightfield->sample_height;
        if (!heightfield->material_weights.empty())
        {
            result.material_weights.assign(heightfield->material_weights.begin(), heightfield->material_weights.end());
            result.default_layer_only = false;
        }
        else
        {
            result.material_weights.assign(static_cast<std::size_t>(result.width) * result.height,
                                           std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
        }
        return result;
    }

    // Arbitrary mesh SurfaceIR does not expose authored terrain weights yet. Keep the fallback explicit and
    // deterministic instead of encoding layer semantics into mesh vertex color.
    return result;
}

} // namespace arc::scene
