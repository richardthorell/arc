#pragma once

#include <arc/render/material.h>
#include <arc/water/water_types.h>

#include <string>

namespace arc::render
{

/**
 * @brief Build the backend-neutral W0 optical material from resolved Water appearance settings.
 *
 * This uses the existing clustered-forward transmission path so every backend gets Fresnel, refracted environment
 * lighting, reflected IBL, and Beer-Lambert attenuation without embedding Water policy in a backend.
 */
[[nodiscard]] material_descriptor make_water_material(const water::water_appearance_settings& appearance,
                                                       std::string name = "Water");

} // namespace arc::render
