#include <arc/scene/water_asset_bridge.h>

#include <arc/water/water_asset.h>

namespace arc::scene
{

bool apply_water_preset(water_component& component, const water::water_preset& preset)
{
    if (!water::validate_water_preset(preset).valid() || component.type != preset.body_type) return false;

    const auto inherits = [&](water_preset_override field)
    { return (component.preset_overrides & water_preset_override_mask(field)) == 0u; };

    if (inherits(water_preset_override::wind_speed))
        component.settings.simulation.wind_speed = preset.settings.simulation.wind_speed;
    if (inherits(water_preset_override::wind_direction))
        component.settings.simulation.wind_direction = preset.settings.simulation.wind_direction;
    if (inherits(water_preset_override::fetch_length))
        component.settings.simulation.fetch_length = preset.settings.simulation.fetch_length;
    if (inherits(water_preset_override::wave_amplitude))
        component.settings.simulation.wave_amplitude = preset.settings.simulation.wave_amplitude;
    if (inherits(water_preset_override::choppiness))
        component.settings.simulation.choppiness = preset.settings.simulation.choppiness;
    component.settings.simulation.seed = preset.settings.simulation.seed;

    if (inherits(water_preset_override::foam_enabled))
        component.settings.foam.enabled = preset.settings.foam.enabled;
    if (inherits(water_preset_override::foam_threshold))
        component.settings.foam.threshold = preset.settings.foam.threshold;
    if (inherits(water_preset_override::foam_decay))
        component.settings.foam.decay = preset.settings.foam.decay;

    if (inherits(water_preset_override::absorption))
        component.settings.appearance.absorption = preset.settings.appearance.absorption;
    if (inherits(water_preset_override::scattering))
        component.settings.appearance.scattering = preset.settings.appearance.scattering;
    if (inherits(water_preset_override::roughness))
        component.settings.appearance.roughness = preset.settings.appearance.roughness;
    if (inherits(water_preset_override::refraction_strength))
        component.settings.appearance.refraction_strength = preset.settings.appearance.refraction_strength;
    if (inherits(water_preset_override::quality)) component.settings.quality = preset.settings.quality;
    return true;
}

water_preset_binding_result refresh_water_preset_binding(water_component& component, assets::asset_manager& manager)
{
    water_preset_binding_result result;
    if (!component.preset.guid.valid() && component.preset.path_hint.empty())
    {
        result.succeeded = true;
        result.message = "Water uses component-authored settings";
        return result;
    }

    auto reference = component.preset;
    reference.expected_type = assets::asset_types::water_preset;
    if (!reference.guid.valid() && !reference.path_hint.empty())
        reference = manager.resolve(reference.path_hint, assets::asset_types::water_preset);
    if (!reference.guid.valid())
    {
        result.message = "Water preset reference could not be resolved";
        return result;
    }

    auto pending = manager.load<water::water_preset>({.reference = reference,
                                                      .priority = assets::asset_streaming_priority::high,
                                                      .residency = assets::asset_residency::cpu,
                                                      .allow_fallback = false});
    auto loaded = pending.get();
    if (!loaded)
    {
        result.message = loaded.error.message.empty() ? "Water preset could not be loaded" : loaded.error.message;
        return result;
    }

    const auto* preset = loaded.asset.get();
    if (!preset)
    {
        result.message = "Water preset payload has the wrong runtime type";
        return result;
    }

    component.preset.guid = loaded.asset.requested_guid();
    component.preset.expected_type = assets::asset_types::water_preset;
    if (const auto snapshot = manager.find(loaded.asset.resolved_guid()))
        component.preset.path_hint = assets::normalize_asset_path(snapshot->source_path);
    if (!apply_water_preset(component, *preset))
    {
        result.message = component.type == preset->body_type ? "Water preset payload failed validation"
                                                               : "Water preset body type does not match the Water Body";
        return result;
    }

    result.succeeded = true;
    result.bound = true;
    result.message = "Water preset loaded";
    return result;
}

} // namespace arc::scene
