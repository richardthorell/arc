#include <arc/render/water_material.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace arc::render
{

material_descriptor make_water_material(const water::water_appearance_settings& appearance, std::string name)
{
    const auto scattering_color = [&](std::size_t channel)
    { return std::clamp(appearance.scattering[channel] * 3.5f + 0.025f, 0.0f, 1.0f); };
    const auto transmittance = [&](std::size_t channel)
    { return std::clamp(std::exp(-appearance.absorption[channel] * 3.0f), 0.02f, 1.0f); };
    const float strongest_absorption =
        std::max({appearance.absorption[0], appearance.absorption[1], appearance.absorption[2], 0.001f});

    material_descriptor material;
    material.name = std::move(name);
    material.shading_model = material_shading_model::transmission;
    material.render_path = material_render_path::clustered_forward;
    material.deferred_compatible = false;
    material.base_color = {scattering_color(0), scattering_color(1), scattering_color(2), 0.72f};
    material.roughness = std::clamp(appearance.roughness, 0.0f, 1.0f);
    material.alpha_mode = material_alpha_mode::blend;
    material.double_sided = true;
    material.clear_coat_factor = 0.85f;
    material.clear_coat_roughness = std::min(material.roughness, 0.10f);
    material.transmission_factor = std::clamp(0.30f + appearance.refraction_strength * 2.0f, 0.30f, 0.80f);
    material.index_of_refraction = 1.333f;
    material.thickness_factor = 8.0f;
    material.attenuation_color = {transmittance(0), transmittance(1), transmittance(2)};
    material.attenuation_distance = std::clamp(1.0f / strongest_absorption, 1.0f, 1000.0f);
    return material;
}

} // namespace arc::render
