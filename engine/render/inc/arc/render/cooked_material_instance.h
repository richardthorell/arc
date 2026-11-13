#pragma once

/** @file
 * @brief Decode value-only cooked Material Instances without authoring tools.
 * Function Slot specializations are rejected until their shader/texture package
 * contract is available to the packaged runtime.
 */

#include <arc/assets/assets.h>
#include <arc/core/result.h>
#include <arc/render/cooked_material.h>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace arc::render
{
struct cooked_material_instance_values
{
    assets::asset_guid parent{};
    cooked_material_metadata parent_material;
    std::string name;
    std::vector<material_parameter_override> overrides;
};

/** Decode ARC_MATERIAL_INSTANCE_1 and validate its embedded parent Material ABI. */
[[nodiscard]] core::result<cooked_material_instance_values, std::string>
decode_cooked_material_instance_values(std::span<const std::byte> bytes);

/** Apply only validated scalar/vector overrides to the descriptor.
 * Texture/resource overrides and Function Slot specializations fail closed.
 */
[[nodiscard]] core::result<material_descriptor, std::string>
apply_cooked_material_instance_values(material_descriptor parent, const cooked_material_instance_values& instance);
} // namespace arc::render
