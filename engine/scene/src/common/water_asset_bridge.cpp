#include <arc/scene/water_asset_bridge.h>

#include <arc/water/water_asset.h>

namespace arc::scene
{

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
    component.type = preset->body_type;
    component.settings = preset->settings;

    result.succeeded = true;
    result.bound = true;
    result.message = "Water preset loaded";
    return result;
}

} // namespace arc::scene
