#pragma once

/**
 * @file arc/render_tools/render_tools.h
 * @brief Tools-only shader compilation services shared by the editor and cooker.
 */

#include <arc/render/shader.h>
#include <arc/render_tools/material_asset.h>
#include <arc/render_tools/material_graph.h>
#include <arc/render_tools/material_pass_codegen.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace arc::render::tools
{

/** @brief Exact Slang release accepted by ARC's production shader pipeline. */
inline constexpr std::string_view pinned_slang_version = "2026.14.1";

/** @brief Configuration for the tools-only Slang command-line adapter. */
struct slang_compiler_config
{
    std::filesystem::path executable;
    bool require_pinned_version{true};
};

/** @brief Compatibility name for Material IR generated shader source. */
using material_graph_lowering = material_shader_source;
using material_graph_lowering_result = material_shader_codegen_result;

/**
 * @brief Validate and deterministically lower authored ARC material graph JSON to Slang.
 *
 * This compatibility entry point is now strictly `JSON -> Material IR -> Slang`. It remains for
 * editor/tool callers while the native compiler API is adopted directly; there is no independent
 * JSON-to-shader implementation behind it.
 */
[[nodiscard]] material_graph_lowering_result lower_material_graph_json(std::string_view graph_json);

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
