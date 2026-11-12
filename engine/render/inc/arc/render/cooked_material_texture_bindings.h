#pragma once

/** @file
 * @brief Cooked Material ABI texture slot-to-asset mapping, independent of tools.
 */
#include <arc/assets/assets.h>
#include <arc/core/result.h>
#include <arc/render/material.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace arc::render
{
struct cooked_material_texture_binding
{
    material_runtime_texture_binding binding;
    assets::asset_guid asset{};
};

struct cooked_material_texture_manifest
{
    shader_package_id material{};
    std::vector<cooked_material_texture_binding> textures;
};

/** Stable little-endian ARC_MATERIAL_TEXTURES_1 format. All slots must be unique,
 * sequential and dimension indices contiguous per texture type. Unbound slots
 * carry a null GUID so the Vulkan fallback texture can be used.
 */
[[nodiscard]] core::result<std::vector<std::byte>, std::string>
encode_cooked_material_texture_manifest(const cooked_material_texture_manifest& manifest);
[[nodiscard]] core::result<cooked_material_texture_manifest, std::string>
decode_cooked_material_texture_manifest(std::span<const std::byte> bytes);

/** Validate all referenced GUIDs against GPU texture residency and material
 * parameter layout. No mutation on failure.
 */
[[nodiscard]] core::result<material_descriptor, std::string>
bind_cooked_material_textures(material_descriptor material, const cooked_material_texture_manifest& manifest,
                              std::span<const std::pair<assets::asset_guid, texture_handle>> resident);

} // namespace arc::render
