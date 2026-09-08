#pragma once

#include <arc/scene/terrain_asset.h>
#include <arc/scene/terrain_surface_ir.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace arc::scene
{

/** @brief Read-only heightfield source supplied to the terrain evaluator by an asset/source provider. */
struct terrain_heightfield_source_view
{
    std::uint32_t sample_width{};
    std::uint32_t sample_height{};
    float width{};
    float depth{};
    std::span<const float> heights;
    std::span<const std::array<std::uint8_t, 4>> material_weights;
    std::uint64_t source_revision{};
};

enum class terrain_evaluation_diagnostic_severity : std::uint8_t
{
    warning,
    error
};

enum class terrain_evaluation_diagnostic_code : std::uint8_t
{
    invalid_asset,
    missing_source_data,
    unsupported_source,
    unsupported_modifier,
    modifier_failed,
    invalid_surface
};

struct terrain_evaluation_diagnostic
{
    terrain_evaluation_diagnostic_severity severity{terrain_evaluation_diagnostic_severity::error};
    terrain_evaluation_diagnostic_code code{terrain_evaluation_diagnostic_code::invalid_asset};
    terrain_stable_id subject{};
    std::string message;
};

/** @brief Region-local request. Heightfield samples are resolved outside the evaluator so authoring is asset-system
 * agnostic. Registered source evaluators may capture additional asset/source providers without changing this contract.
 */
struct terrain_evaluation_request
{
    terrain_region_id region{};
    std::optional<terrain_heightfield_source_view> heightfield_source;
};

/** @brief Owning result of one deterministic terrain-region evaluation. */
struct [[nodiscard]] terrain_evaluation_result
{
    bool succeeded{};
    terrain_region_id region{};
    double world_origin_x{};
    double world_origin_y{};
    double world_origin_z{};
    terrain_build_region_snapshot build_snapshot;
    terrain_evaluated_surface surface;
    std::uint64_t content_fingerprint{};
    std::vector<terrain_evaluation_diagnostic> diagnostics;
};

using terrain_source_evaluation_fn = std::function<std::optional<terrain_evaluated_surface>(
    const terrain_asset&, const terrain_evaluation_request&, const terrain_build_region_snapshot&, std::string&)>;
using terrain_modifier_evaluation_fn =
    std::function<bool(const terrain_modifier_descriptor&, terrain_evaluated_surface&, std::string&)>;

/**
 * @brief Renderer-independent evaluator for unified TerrainAsset sources and ordered non-destructive modifiers.
 *
 * Source implementations are registered by source kind and may capture the asset-system provider they need. Modifier
 * implementations are registered by stable type ID. Missing source implementations and unknown enabled geometry
 * modifiers fail evaluation with diagnostics rather than being silently ignored.
 */
class terrain_evaluator
{
public:
    [[nodiscard]] bool register_source(terrain_source_kind kind, terrain_source_evaluation_fn evaluator);
    [[nodiscard]] bool register_modifier(std::string type_id, terrain_modifier_evaluation_fn evaluator);
    [[nodiscard]] terrain_evaluation_result evaluate(const terrain_asset& asset,
                                                     const terrain_evaluation_request& request) const;

private:
    std::unordered_map<terrain_source_kind, terrain_source_evaluation_fn> sources_;
    std::unordered_map<std::string, terrain_modifier_evaluation_fn> modifiers_;
};

/** @brief Construct the built-in evaluator with Flat, resolved Heightfield, and `arc.height-offset` implementations. */
[[nodiscard]] terrain_evaluator make_default_terrain_evaluator();

} // namespace arc::scene
