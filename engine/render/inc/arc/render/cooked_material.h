#pragma once

/** Runtime-only decoding of ARC_MATERIAL_4 cooked package metadata.
 * Does not import authored assets or compile shaders.
 */
#include <arc/core/result.h>
#include <arc/render/material_pass.h>
#include <arc/render/shader.h>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace arc::render
{
struct cooked_material_metadata
{
    material_compiled_program compiled;
    std::vector<shader_parameter_descriptor> parameters;
    std::string canonical_document_json;
};

/** Decode the existing cooked material-v4 envelope, rejecting invalid ABI,
 * pass bindings, parameter ranges, ambiguous IDs and truncated/trailing bytes.
 * Shader GPU realization belongs to the renderer hydration phase.
 */
[[nodiscard]] core::result<cooked_material_metadata, std::string>
decode_cooked_material_metadata(std::span<const std::byte> bytes);
} // namespace arc::render
