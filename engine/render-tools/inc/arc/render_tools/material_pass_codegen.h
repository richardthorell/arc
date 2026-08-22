#pragma once

/**
 * @file arc/render_tools/material_pass_codegen.h
 * @brief Engine-owned render-pass composition for Material ABI evaluators.
 */

#include <arc/render/material_pass.h>
#include <arc/render_tools/material_graph.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace arc::render::tools
{

/** @brief Version of the engine material/pass Slang composition layer. */
inline constexpr std::uint32_t material_pass_codegen_version = 2;
/** @brief Version of the handwritten Material Shader evaluator contract. */
inline constexpr std::uint32_t custom_material_shader_version = 1;

/**
 * @brief Material ABI evaluator source independent of how it was authored.
 *
 * Graph materials and handwritten Material Shaders converge on this representation before
 * engine-owned pass composition. The source must provide `arc_evaluate_material(ArcSurfaceInput)`
 * and must not provide a render-pass entry point.
 */
struct material_evaluator_source
{
    std::string source;
    std::unordered_map<std::uint32_t, std::string> generated_line_nodes;
    std::vector<shader_parameter_descriptor> parameters;
    std::vector<shader_diagnostic> diagnostics;
    bool handwritten{};
};

using material_evaluator_result = core::result<material_evaluator_source, shader_compile_error>;

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

/** @brief Convert validated native Material IR into a pass-independent Material ABI evaluator. */
[[nodiscard]] material_evaluator_result make_graph_material_evaluator(const material_graph_compilation& compilation);

/**
 * @brief Wrap handwritten Slang as a Material ABI evaluator.
 *
 * ARC supplies the Material ABI declarations. Authored source only implements
 * `ArcSurfaceData arc_evaluate_material(ArcSurfaceInput input)` and may declare its own parameters
 * and resources. Full render-pass entry points remain engine-owned.
 */
[[nodiscard]] material_evaluator_result make_custom_material_evaluator(std::string_view source,
                                                                       std::string_view source_path = {});

/** @brief Compose a pass-independent evaluator into one engine-owned render-pass fragment shader. */
[[nodiscard]] material_pass_codegen_result generate_material_pass_slang(const material_evaluator_source& evaluator,
                                                                        const material_descriptor& material,
                                                                        material_pass pass, std::uint8_t debug_view = 0,
                                                                        bool wireframe = false);

/**
 * @brief Compatibility overload for graph-generated materials.
 *
 * This routes through @ref make_graph_material_evaluator and the same generic pass composer used by
 * handwritten Material Shaders.
 */
[[nodiscard]] material_pass_codegen_result generate_material_pass_slang(const material_graph_compilation& compilation,
                                                                        const material_descriptor& material,
                                                                        material_pass pass, std::uint8_t debug_view = 0,
                                                                        bool wireframe = false);

} // namespace arc::render::tools
