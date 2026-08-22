#pragma once

/**
 * @file arc/render_tools/material_pass_codegen.h
 * @brief Engine-owned render-pass composition for generated Material ABI evaluators.
 */

#include <arc/render/material_pass.h>
#include <arc/render_tools/material_graph.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc::render::tools
{

/** @brief Version of the engine material/pass Slang composition layer. */
inline constexpr std::uint32_t material_pass_codegen_version = 1;

/** @brief Generated pass shader source and its stable runtime identity. */
struct material_pass_shader_source
{
    material_pass pass{material_pass::forward};
    shader_permutation_id permutation{};
    std::string entry_point{"main"};
    std::string source;
    std::unordered_map<std::uint32_t, std::string> generated_line_nodes;
    std::vector<shader_parameter_descriptor> parameters;
    std::vector<shader_diagnostic> diagnostics;
};

using material_pass_codegen_result = core::result<material_pass_shader_source, shader_compile_error>;

/**
 * @brief Compose one validated material evaluator into an engine-owned render-pass fragment shader.
 *
 * Material graphs only implement the Material ABI evaluator. This function owns pass semantics such as
 * alpha clipping, G-buffer packing, motion-vector output, object IDs, and selection output. The same
 * composition contract is intentionally reusable by handwritten Material Shaders in Stage 9.
 */
[[nodiscard]] material_pass_codegen_result generate_material_pass_slang(
    const material_graph_compilation& compilation, const material_descriptor& material, material_pass pass,
    std::uint8_t debug_view = 0, bool wireframe = false);

} // namespace arc::render::tools
