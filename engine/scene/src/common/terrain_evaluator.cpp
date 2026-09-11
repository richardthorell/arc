#include <arc/scene/terrain_evaluator.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace arc::scene
{
namespace
{

bool identity_rotation(const math::quatf& rotation) noexcept
{
    constexpr float epsilon = 1.0e-6f;
    return std::abs(rotation.x()) <= epsilon && std::abs(rotation.y()) <= epsilon &&
           std::abs(rotation.z()) <= epsilon && std::abs(rotation.w() - 1.0f) <= epsilon;
}

bool overlaps_xz(const terrain_world_bounds& lhs, const terrain_world_bounds& rhs) noexcept
{
    return lhs.min_x <= rhs.max_x && lhs.max_x >= rhs.min_x && lhs.min_z <= rhs.max_z && lhs.max_z >= rhs.min_z;
}

void add_diagnostic(terrain_evaluation_result& result, terrain_evaluation_diagnostic_severity severity,
                    terrain_evaluation_diagnostic_code code, std::string message, terrain_stable_id subject = {})
{
    result.diagnostics.push_back({severity, code, subject, std::move(message)});
}

void update_vertical_bounds(terrain_evaluated_surface& surface) noexcept
{
    if (auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&surface.geometry))
    {
        if (heightfield->heights.empty()) return;
        const auto [minimum, maximum] = std::minmax_element(heightfield->heights.begin(), heightfield->heights.end());
        surface.local_bounds.min_y = *minimum;
        surface.local_bounds.max_y = *maximum;
        return;
    }

    auto& mesh = std::get<terrain_evaluated_mesh>(surface.geometry);
    if (mesh.positions.empty()) return;
    float minimum = mesh.positions.front()[1];
    float maximum = minimum;
    for (const auto& position : mesh.positions)
    {
        minimum = std::min(minimum, position[1]);
        maximum = std::max(maximum, position[1]);
    }
    surface.local_bounds.min_y = minimum;
    surface.local_bounds.max_y = maximum;
}

bool apply_height_offset(const terrain_modifier_descriptor& modifier, const terrain_build_region_snapshot&,
                         terrain_evaluated_surface& surface, std::string& error)
{
    const auto parameters = nlohmann::json::parse(modifier.canonical_parameters, nullptr, false);
    if (parameters.is_discarded() || !parameters.is_object() || !parameters.contains("offset") ||
        !parameters["offset"].is_number())
    {
        error = "arc.height-offset requires a numeric `offset` parameter";
        return false;
    }

    const auto offset = parameters["offset"].get<float>();
    if (!std::isfinite(offset))
    {
        error = "arc.height-offset requires a finite offset";
        return false;
    }

    if (auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&surface.geometry))
    {
        for (auto& height : heightfield->heights)
            height += offset;
    }
    else
    {
        auto& mesh = std::get<terrain_evaluated_mesh>(surface.geometry);
        for (auto& position : mesh.positions)
            position[1] += offset;
    }
    update_vertical_bounds(surface);
    return true;
}

bool apply_sculpt_layer(const terrain_modifier_descriptor& modifier, const terrain_build_region_snapshot& snapshot,
                        terrain_evaluated_surface& surface, std::string& error)
{
    const auto* payload = find_terrain_modifier_payload(modifier, snapshot.target);
    if (!payload) return true;
    if (!std::holds_alternative<terrain_sculpt_region_payload>(payload->data))
    {
        error = "Sculpt Layer contains an incompatible sparse region payload";
        return false;
    }

    auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&surface.geometry);
    if (!heightfield || heightfield->sample_width < 2u || heightfield->sample_height < 2u)
    {
        error = "Sculpt Layer currently requires heightfield-backed TerrainSurfaceIR";
        return false;
    }

    const auto& samples = std::get<terrain_sculpt_region_payload>(payload->data).samples;
    const auto coordinate = static_cast<std::uint64_t>(terrain_modifier_sample_coordinate_max);
    for (const auto& sample : samples)
    {
        const auto x = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(sample.x) * (heightfield->sample_width - 1u) + coordinate / 2u) / coordinate);
        const auto z = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(sample.z) * (heightfield->sample_height - 1u) + coordinate / 2u) / coordinate);
        const auto index = static_cast<std::size_t>(z) * heightfield->sample_width + x;
        const auto next = heightfield->heights[index] + sample.delta;
        if (!std::isfinite(next))
        {
            error = "Sculpt Layer produced a non-finite height";
            return false;
        }
        heightfield->heights[index] = next;
    }
    update_vertical_bounds(surface);
    return true;
}

std::optional<terrain_evaluated_surface> evaluate_flat_source(const terrain_asset& asset,
                                                              const terrain_build_region_snapshot& snapshot)
{
    if (!identity_rotation(asset.source.transform.rotation)) return std::nullopt;

    const auto region_width = snapshot.authoring_bounds.max_x - snapshot.authoring_bounds.min_x;
    const auto region_depth = snapshot.authoring_bounds.max_z - snapshot.authoring_bounds.min_z;
    const auto width = static_cast<float>(region_width) * std::abs(asset.source.transform.scale[0]);
    const auto depth = static_cast<float>(region_depth) * std::abs(asset.source.transform.scale[2]);
    const auto height = asset.source.transform.translation[1];
    if (!std::isfinite(width) || !std::isfinite(depth) || width <= 0.0f || depth <= 0.0f) return std::nullopt;

    terrain_evaluated_heightfield heightfield;
    heightfield.sample_width = 2u;
    heightfield.sample_height = 2u;
    heightfield.width = width;
    heightfield.depth = depth;
    heightfield.heights.assign(4u, height);
    heightfield.material_weights.assign(4u, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});

    const auto half_width = static_cast<double>(width) * 0.5;
    const auto half_depth = static_cast<double>(depth) * 0.5;
    terrain_evaluated_surface surface;
    surface.source_revision = asset.authoring_revision;
    surface.local_bounds = {-half_width, height, -half_depth, half_width, height, half_depth};
    surface.geometry = std::move(heightfield);
    return surface;
}

std::optional<terrain_evaluated_surface> evaluate_heightfield_source(const terrain_asset& asset,
                                                                     const terrain_heightfield_source_view& source)
{
    if (!identity_rotation(asset.source.transform.rotation) || source.sample_width < 2u || source.sample_height < 2u ||
        !std::isfinite(source.width) || !std::isfinite(source.depth) || source.width <= 0.0f || source.depth <= 0.0f)
        return std::nullopt;

    const auto sample_count = static_cast<std::size_t>(source.sample_width) * source.sample_height;
    if (source.heights.size() != sample_count || source.material_weights.size() != sample_count) return std::nullopt;

    terrain_evaluated_heightfield heightfield;
    heightfield.sample_width = source.sample_width;
    heightfield.sample_height = source.sample_height;
    heightfield.width = source.width * std::abs(asset.source.transform.scale[0]);
    heightfield.depth = source.depth * std::abs(asset.source.transform.scale[2]);
    heightfield.heights.reserve(source.heights.size());
    for (const auto source_height : source.heights)
    {
        const auto height = source_height * asset.source.transform.scale[1] + asset.source.transform.translation[1];
        if (!std::isfinite(height)) return std::nullopt;
        heightfield.heights.push_back(height);
    }
    heightfield.material_weights.assign(source.material_weights.begin(), source.material_weights.end());

    const auto [minimum, maximum] = std::minmax_element(heightfield.heights.begin(), heightfield.heights.end());
    const auto half_width = static_cast<double>(heightfield.width) * 0.5;
    const auto half_depth = static_cast<double>(heightfield.depth) * 0.5;
    terrain_evaluated_surface surface;
    surface.source_revision = source.source_revision == 0u ? asset.authoring_revision : source.source_revision;
    surface.local_bounds = {-half_width, static_cast<double>(*minimum), -half_depth,
                            half_width,  static_cast<double>(*maximum), half_depth};
    surface.geometry = std::move(heightfield);
    return surface;
}

} // namespace

bool terrain_evaluator::register_modifier(std::string type_id, terrain_modifier_evaluation_fn evaluator)
{
    if (type_id.empty() || !evaluator) return false;
    return modifiers_.emplace(std::move(type_id), std::move(evaluator)).second;
}

terrain_evaluation_result terrain_evaluator::evaluate(const terrain_asset& asset,
                                                      const terrain_evaluation_request& request) const
{
    terrain_evaluation_result result;
    result.region = request.region;
    result.build_snapshot = make_terrain_build_region_snapshot(asset, request.region);
    result.world_origin_x =
        (result.build_snapshot.authoring_bounds.min_x + result.build_snapshot.authoring_bounds.max_x) * 0.5 +
        asset.source.transform.translation[0];
    result.world_origin_y = asset.coordinates.origin_y;
    result.world_origin_z =
        (result.build_snapshot.authoring_bounds.min_z + result.build_snapshot.authoring_bounds.max_z) * 0.5 +
        asset.source.transform.translation[2];

    const auto validation = validate_terrain_asset(asset);
    if (!validation.valid())
    {
        add_diagnostic(result, terrain_evaluation_diagnostic_severity::error,
                       terrain_evaluation_diagnostic_code::invalid_asset,
                       "TerrainAsset failed validation before evaluation", asset.source.id);
        return result;
    }

    std::optional<terrain_evaluated_surface> surface;
    switch (asset.source.kind)
    {
        case terrain_source_kind::flat:
            surface = evaluate_flat_source(asset, result.build_snapshot);
            break;
        case terrain_source_kind::heightfield:
            if (!request.heightfield_source)
            {
                add_diagnostic(result, terrain_evaluation_diagnostic_severity::error,
                               terrain_evaluation_diagnostic_code::missing_source_data,
                               "Heightfield terrain evaluation requires resolved source samples", asset.source.id);
                return result;
            }
            surface = evaluate_heightfield_source(asset, *request.heightfield_source);
            break;
        case terrain_source_kind::mesh:
        case terrain_source_kind::procedural:
            add_diagnostic(
                result, terrain_evaluation_diagnostic_severity::error,
                terrain_evaluation_diagnostic_code::unsupported_source,
                "Terrain source uses the unified evaluator contract but its source provider is not implemented yet",
                asset.source.id);
            return result;
    }

    if (!surface || !validate_terrain_surface_ir(surface->view()))
    {
        add_diagnostic(result, terrain_evaluation_diagnostic_severity::error,
                       terrain_evaluation_diagnostic_code::invalid_surface,
                       "Terrain source did not produce a valid TerrainSurfaceIR", asset.source.id);
        return result;
    }

    for (const auto& modifier : asset.modifiers)
    {
        if (!modifier.enabled) continue;
        const auto geometry_domains = terrain_domain::geometry | terrain_domain::topology;
        if ((modifier.domains & geometry_domains) == terrain_domain::none) continue;
        if (modifier.affected_bounds && !overlaps_xz(*modifier.affected_bounds, result.build_snapshot.authoring_bounds))
            continue;

        const auto found = modifiers_.find(modifier.type_id);
        if (found == modifiers_.end())
        {
            add_diagnostic(result, terrain_evaluation_diagnostic_severity::error,
                           terrain_evaluation_diagnostic_code::unsupported_modifier,
                           "No terrain evaluator is registered for modifier type `" + modifier.type_id + "`",
                           modifier.id);
            return result;
        }

        std::string error;
        if (!found->second(modifier, result.build_snapshot, *surface, error))
        {
            add_diagnostic(result, terrain_evaluation_diagnostic_severity::error,
                           terrain_evaluation_diagnostic_code::modifier_failed,
                           error.empty() ? "Terrain modifier evaluation failed" : std::move(error), modifier.id);
            return result;
        }
        if (!validate_terrain_surface_ir(surface->view()))
        {
            add_diagnostic(result, terrain_evaluation_diagnostic_severity::error,
                           terrain_evaluation_diagnostic_code::invalid_surface,
                           "Terrain modifier produced an invalid TerrainSurfaceIR", modifier.id);
            return result;
        }
    }

    result.surface = std::move(*surface);
    result.content_fingerprint = terrain_surface_fingerprint(result.surface.view());
    result.succeeded = result.content_fingerprint != 0u;
    if (!result.succeeded)
        add_diagnostic(result, terrain_evaluation_diagnostic_severity::error,
                       terrain_evaluation_diagnostic_code::invalid_surface,
                       "Terrain surface fingerprint could not be generated", asset.source.id);
    return result;
}

terrain_evaluator make_default_terrain_evaluator()
{
    terrain_evaluator result;
    (void)result.register_modifier("arc.height-offset", apply_height_offset);
    (void)result.register_modifier(std::string(terrain_builtin_modifier_types::sculpt_layer), apply_sculpt_layer);
    return result;
}

} // namespace arc::scene
