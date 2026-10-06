#include <arc/render/water_material.h>

#include <algorithm>
#include <cmath>
#include <ranges>
#include <string_view>
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

material_descriptor apply_water_material_appearance(material_descriptor material,
                                                    const water::water_appearance_settings& appearance,
                                                    std::string name)
{
    const auto resolved = make_water_material(appearance, std::move(name));

    material.name = resolved.name;
    material.domain = resolved.domain;
    material.shading_model = resolved.shading_model;
    material.render_path = resolved.render_path;
    material.deferred_compatible = resolved.deferred_compatible;
    material.base_color = resolved.base_color;
    material.metallic = resolved.metallic;
    material.roughness = resolved.roughness;
    material.alpha_mode = resolved.alpha_mode;
    material.double_sided = resolved.double_sided;
    material.cast_shadows = false;
    material.clear_coat_factor = resolved.clear_coat_factor;
    material.clear_coat_roughness = resolved.clear_coat_roughness;
    material.transmission_factor = resolved.transmission_factor;
    material.index_of_refraction = resolved.index_of_refraction;
    material.thickness_factor = resolved.thickness_factor;
    material.attenuation_color = resolved.attenuation_color;
    material.attenuation_distance = resolved.attenuation_distance;

    if (!material.runtime_program) return material;

    material_definition_descriptor definition{
        .material = material,
        .parameter_layout = material.runtime_program->parameters,
    };
    material_instance_descriptor instance{
        .parent = material_handle{1},
        .name = material.name,
    };

    const auto append_if_present = [&](std::string_view stable_id, std::string_view display_name,
                                       shader_parameter_type type, material_parameter_value value)
    {
        const auto id = make_shader_parameter_id(stable_id);
        const auto parameter = std::ranges::find(definition.parameter_layout, id, &shader_parameter_descriptor::id);
        if (parameter == definition.parameter_layout.end() || parameter->type != type) return;
        instance.overrides.push_back({.id = id, .name = std::string(display_name), .value = std::move(value)});
    };

    append_if_present("base-color", "Base Color", shader_parameter_type::float4,
                      math::vector4f{resolved.base_color[0], resolved.base_color[1], resolved.base_color[2], 1.0f});
    append_if_present("roughness", "Roughness", shader_parameter_type::float32, resolved.roughness);
    append_if_present("clear-coat", "Clear Coat", shader_parameter_type::float32, resolved.clear_coat_factor);
    append_if_present("transmission", "Transmission", shader_parameter_type::float32, resolved.transmission_factor);
    append_if_present("opacity", "Opacity", shader_parameter_type::float32, resolved.base_color[3]);
    append_if_present("ior", "Index of Refraction", shader_parameter_type::float32, resolved.index_of_refraction);
    append_if_present("thickness", "Thickness", shader_parameter_type::float32, resolved.thickness_factor);
    append_if_present("attenuation-color", "Attenuation Color", shader_parameter_type::float4,
                      math::vector4f{resolved.attenuation_color[0], resolved.attenuation_color[1],
                                     resolved.attenuation_color[2], 1.0f});
    append_if_present("attenuation-distance", "Attenuation Distance", shader_parameter_type::float32,
                      resolved.attenuation_distance);

    if (instance.overrides.empty()) return material;
    auto applied = resolve_material_instance(definition, instance);
    return applied ? std::move(applied).value() : material;
}

} // namespace arc::render
