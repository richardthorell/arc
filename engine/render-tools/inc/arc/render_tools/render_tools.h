#pragma once

/**
 * @file arc/render_tools/render_tools.h
 * @brief Tools-only shader compilation services shared by the editor and cooker.
 */

#include <arc/render/shader.h>
#include <arc/render_tools/material_asset.h>
#include <arc/render_tools/material_graph.h>
#include <arc/render_tools/material_pass_codegen.h>
#include <arc/render_tools/texture_cooker.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace arc::render::tools
{

/** @brief Exact Slang release accepted by ARC's production shader pipeline. */
inline constexpr std::string_view pinned_slang_version = "2026.14.1";

/** @brief Generated shader source returned to editor/tool callers after native Material IR compilation. */
using material_graph_lowering_result = material_shader_codegen_result;

/**
 * @brief Compile authored graph JSON through the native Material IR pipeline and generate Slang.
 *
 * This remains a thin tools-facing bridge for callers that need generated source in one step. There
 * is no separate legacy graph compiler behind it: all validation and normalization are performed by
 * `compile_material_graph_json`, followed by the current Material IR code generator.
 */
[[nodiscard]] inline material_graph_lowering_result lower_material_graph_json(std::string_view graph_json)
{
    auto compilation = compile_material_graph_json(graph_json);
    if (!compilation) return material_graph_lowering_result::failure(compilation.error());
    return generate_material_slang(compilation.value());
}

/** @brief Configuration for the tools-only Slang command-line adapter. */
struct slang_compiler_config
{
    std::filesystem::path executable;
    bool require_pinned_version{true};
};

/**
 * @brief Slang compiler adapter used by editor and cooker processes.
 *
 * The adapter launches `slangc` directly without shell interpretation. It is
 * intentionally provided by `arc-render-tools`; runtime and Shipping targets
 * depend only on compiled `shader_package` data.
 */
class slang_shader_compiler final : public shader_compiler
{
public:
    explicit slang_shader_compiler(slang_compiler_config config = {});

    [[nodiscard]] shader_compile_result compile(const shader_compile_request& request) override;
    [[nodiscard]] std::string_view fingerprint() const noexcept override;
    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] const std::filesystem::path& executable() const noexcept;

private:
    slang_compiler_config config_;
    std::string fingerprint_;
    bool available_{};
};

} // namespace arc::render::tools
