#pragma once
/** Decode validated conventional .arctex mips for renderer-independent staging. */
#include <arc/core/result.h>
#include <arc/render/material.h>
#include <cstddef>
#include <span>
#include <string>

namespace arc::render
{
/** Requires a resident/streamed conventional texture artifact; virtual tiles
 * are a separate streaming path. Verifies each mip's integrity before publish.
 * The texture's compression family is preserved for backend capability checks.
 */
[[nodiscard]] core::result<texture_data, std::string> decode_cooked_texture_data(std::span<const std::byte> bytes);
} // namespace arc::render
