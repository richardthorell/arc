#pragma once

#include <arc/assets/assets.h>
#include <arc/scene/components.h>

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
 * @brief Resolve, load, and apply the preset referenced by a Water component.
 *
 * Body placement and feature toggles remain authored per component. The preset owns body type, simulation, foam,
 * appearance, and quality settings.
 */
[[nodiscard]] water_preset_binding_result refresh_water_preset_binding(water_component& component,
                                                                        assets::asset_manager& manager);

} // namespace arc::scene
