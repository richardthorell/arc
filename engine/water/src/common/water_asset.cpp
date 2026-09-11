#include <arc/water/water_asset.h>

#include <cmath>

namespace arc::water
{
namespace
{

bool finite_vector(const math::vector2f& value) noexcept
{
    return std::isfinite(value[0]) && std::isfinite(value[1]);
}

bool finite_non_negative_vector(const math::vector3f& value) noexcept
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]) && value[0] >= 0.0f &&
           value[1] >= 0.0f && value[2] >= 0.0f;
}

} // namespace

water_preset_validation_result validate_water_preset(const water_preset& preset)
{
    water_preset_validation_result result;
    const auto add = [&](water_preset_validation_code code, const char* message)
    { result.issues.push_back({code, message}); };

    if (preset.schema_version != water_preset::current_schema_version)
        add(water_preset_validation_code::unsupported_schema, "Unsupported Water preset schema version");
    if (preset.name.empty()) add(water_preset_validation_code::missing_name, "Water preset name must not be empty");
    if (preset.body_type > water_body_type::river)
        add(water_preset_validation_code::invalid_body_type, "Water body type is invalid");

    const auto& simulation = preset.settings.simulation;
    const float direction_length_squared = simulation.wind_direction[0] * simulation.wind_direction[0] +
                                           simulation.wind_direction[1] * simulation.wind_direction[1];
    if (!std::isfinite(simulation.wind_speed) || simulation.wind_speed < 0.0f ||
        !finite_vector(simulation.wind_direction) || direction_length_squared <= 0.0f ||
        !std::isfinite(simulation.fetch_length) || simulation.fetch_length <= 0.0f ||
        !std::isfinite(simulation.wave_amplitude) || simulation.wave_amplitude < 0.0f ||
        !std::isfinite(simulation.choppiness) || simulation.choppiness < 0.0f)
        add(water_preset_validation_code::invalid_simulation, "Water simulation values are invalid");

    const auto& foam = preset.settings.foam;
    if (!std::isfinite(foam.threshold) || foam.threshold < 0.0f || foam.threshold > 1.0f ||
        !std::isfinite(foam.decay) || foam.decay < 0.0f)
        add(water_preset_validation_code::invalid_foam, "Water foam values are invalid");

    const auto& appearance = preset.settings.appearance;
    if (!finite_non_negative_vector(appearance.absorption) || !finite_non_negative_vector(appearance.scattering) ||
        !std::isfinite(appearance.roughness) || appearance.roughness < 0.0f || appearance.roughness > 1.0f ||
        !std::isfinite(appearance.refraction_strength) || appearance.refraction_strength < 0.0f ||
        appearance.refraction_strength > 1.0f)
        add(water_preset_validation_code::invalid_appearance, "Water appearance values are invalid");

    if (preset.settings.quality > water_quality::ultra)
        add(water_preset_validation_code::invalid_quality, "Water quality is invalid");
    return result;
}

} // namespace arc::water
