#pragma once

#include <arc/assets/assets.h>
#include <arc/scene/components.h>

#include <cstdint>
#include <string>

namespace arc::scene
{

struct [[nodiscard]] irradiance_probe_volume_binding_result
{
    bool succeeded{};
    bool bound{};
    std::uint64_t generation{};
    std::string message;
};

/** Register the canonical .arcprobe baked irradiance-volume importer. */
bool register_irradiance_probe_volume_importer(assets::asset_manager& manager);

/** Resolve/load an authored baked irradiance-volume component against the asset registry. */
[[nodiscard]] irradiance_probe_volume_binding_result
refresh_irradiance_probe_volume_binding(irradiance_probe_volume_component& component,
                                        assets::asset_manager& manager);

} // namespace arc::scene
