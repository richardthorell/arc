#pragma once

#include <arc/assets/assets.h>
#include <arc/scene/components.h>
#include <arc/water/water_asset.h>

#include <string>

namespace arc::scene
{

/** @brief Result of resolving and applying one Water preset to a scene component. */
struct [[nodiscard]] water_preset_binding_result
{
    bool succeeded{};
    bool bound{};
    std::string message;
};

/**
 * @brief Validate and atomically apply preset-owned state to a Water component.
 *
 * Body type, shape, placement, material, visibility, feature toggles, query settings, and priority remain
 * instance-owned. The preset supplies simulation, foam, appearance, and quality values unless a field is overridden.
 */
[[nodiscard]] bool apply_water_preset(water_component& component, const water::water_preset& preset);

/**
 * @brief Resolve, load, and apply the preset referenced by a Water component.
 *
 * Body type/shape and feature toggles remain authored per component. The preset supplies simulation, foam,
 * appearance, and quality defaults while preserving explicit component overrides.
 */
[[nodiscard]] water_preset_binding_result refresh_water_preset_binding(water_component& component,
                                                                       assets::asset_manager& manager);

} // namespace arc::scene
