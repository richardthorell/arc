#pragma once

#include <arc/render/material.h>
#include <arc/render_tools/material_graph.h>

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace arc::editor
{

/**
 * @brief Editor-only realization of compiled Material Graph defaults into the renderer preview descriptor.
 *
 * The Material Graph is always validated and normalized by ARC's native Material IR compiler first. This helper
 * evaluates only graph expressions that are statically known from authored/default values so the native preview can
 * use the same source asset without pretending the asset-manager source payload is a renderer handle. Dynamic outputs
 * remain at their Material ABI defaults until runtime compiled-pass binding supports them.
 */
struct material_preview_descriptor_result
{
    render::material_descriptor material;
    std::vector<std::string> texture_sources;
    std::vector<std::string> diagnostics;
    std::string message;
    bool succeeded{};
};

/** @brief Realize one authored v4 material document through native Material IR for preview use. */
[[nodiscard]] material_preview_descriptor_result realize_material_preview_descriptor(std::string_view source,
                                                                                     std::string_view name = {});

/** @brief Realize a Material with reusable functions and compile-time Function Slot selections. */
[[nodiscard]] material_preview_descriptor_result
realize_material_preview_descriptor(
    std::string_view source, std::string_view name,
    std::span<const render::tools::material_function_source> functions,
    std::span<const render::tools::material_function_slot_override> slot_overrides);

/** @brief Read and realize one graph-authored material file for the native Material Preview surface. */
[[nodiscard]] material_preview_descriptor_result
load_material_preview_descriptor(const std::filesystem::path& source_path);

} // namespace arc::editor
