#pragma once

/**
 * @file arc/render_tools/render_tools.h
 * @brief Tools-only shader compilation services shared by the editor and cooker.
 */

#include <arc/render/shader.h>
#include <arc/render_tools/material_graph.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

/** @brief Native material-graph lowering output consumed by Slang compilation. */
struct material_graph_lowering
{
    std::string source;
    std::unordered_map<std::uint32_t, std::string> generated_line_nodes;
    std::vector<shader_parameter_descriptor> parameters;
    std::vector<shader_diagnostic> diagnostics;
};

using material_graph_lowering_result = core::result<material_graph_lowering, shader_compile_error>;

/**
 * @brief Validate and deterministically lower an ARC material graph to Slang.
 *
 * The JSON representation remains private to tools. Stable graph node and
 * parameter IDs are copied into generated source maps and reflection metadata.
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
