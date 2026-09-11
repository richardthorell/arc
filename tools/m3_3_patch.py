from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1))


def insert_before(path: str, marker: str, content: str) -> None:
    replace_once(path, marker, content + marker)


# -----------------------------------------------------------------------------
# Engine authoring contract: normalized region-local coordinates + additive
# sculpt edits. Persistent sample coordinates are resolution independent.
# -----------------------------------------------------------------------------
path = "engine/scene/inc/arc/scene/terrain_asset.h"
replace_once(path, "#include <optional>\n", "#include <optional>\n#include <span>\n")
replace_once(
    path,
    "[[nodiscard]] terrain_world_bounds expand_terrain_bounds(terrain_world_bounds bounds, double amount) noexcept;\n",
    """inline constexpr std::uint32_t terrain_modifier_sample_coordinate_max = 65535u;

/** @brief Resolution-independent sample address inside one stable authoring region. */
struct terrain_modifier_sample_location
{
    terrain_region_id region{};
    std::uint32_t x{};
    std::uint32_t z{};
};

/** @brief Quantize a world-space XZ position into a stable region-local modifier address. */
[[nodiscard]] terrain_modifier_sample_location
terrain_modifier_sample_at(const terrain_coordinate_system& coordinates, const terrain_partition_settings& partition,
                           double world_x, double world_z) noexcept;

[[nodiscard]] terrain_world_bounds expand_terrain_bounds(terrain_world_bounds bounds, double amount) noexcept;
""",
)
replace_once(
    path,
    """struct terrain_paint_sample_delta
{
""",
    """/** @brief One additive sparse sculpt edit routed to a stable authoring region. */
struct terrain_sculpt_sample_edit
{
    terrain_region_id region{};
    terrain_sculpt_sample_delta sample{};
};

struct terrain_paint_sample_delta
{
""",
)
replace_once(
    path,
    """/** Replace one region's sparse paint payload and dirty only attributes for that authoring region. */
""",
    """/**
 * Add sparse deltas into one sculpt modifier using stable region-local coordinates.
 * Duplicate sample edits are folded, zero results are removed, and every changed region shares one authoring revision.
 */
[[nodiscard]] terrain_dirty_update accumulate_terrain_sculpt_samples(
    terrain_asset& asset, terrain_stable_id modifier, std::span<const terrain_sculpt_sample_edit> edits);

/** Replace one region's sparse paint payload and dirty only attributes for that authoring region. */
""",
)

path = "engine/scene/src/common/terrain_asset.cpp"
insert_before(
    path,
    "terrain_world_bounds expand_terrain_bounds(terrain_world_bounds bounds, double amount) noexcept\n",
    """terrain_modifier_sample_location terrain_modifier_sample_at(const terrain_coordinate_system& coordinates,
                                                                    const terrain_partition_settings& partition,
                                                                    double world_x, double world_z) noexcept
{
    if (!finite(partition.authoring_region_size) || partition.authoring_region_size <= 0.0 || !finite(world_x) ||
        !finite(world_z) || !finite(coordinates.origin_x) || !finite(coordinates.origin_z))
        return {};

    const auto region = terrain_region_at(coordinates, partition, world_x, world_z);
    const auto bounds = terrain_region_bounds(coordinates, partition, region);
    const auto quantize = [](double value, double minimum, double maximum)
    {
        const auto span = maximum - minimum;
        if (!finite(span) || span <= 0.0) return 0u;
        const auto normalized = std::clamp((value - minimum) / span, 0.0, 1.0);
        return static_cast<std::uint32_t>(
            std::llround(normalized * static_cast<double>(terrain_modifier_sample_coordinate_max)));
    };
    return {region, quantize(world_x, bounds.min_x, bounds.max_x), quantize(world_z, bounds.min_z, bounds.max_z)};
}

""",
)

path = "engine/scene/src/common/terrain_edit_layers.cpp"
insert_before(
    path,
    "terrain_dirty_update set_terrain_paint_region_samples(terrain_asset& asset, terrain_stable_id modifier_id,\n",
    """terrain_dirty_update accumulate_terrain_sculpt_samples(terrain_asset& asset, terrain_stable_id modifier_id,
                                                               std::span<const terrain_sculpt_sample_edit> edits)
{
    auto* modifier = find_terrain_modifier(asset, modifier_id);
    if (!modifier || modifier->type_id != terrain_builtin_modifier_types::sculpt_layer || edits.empty()) return {};

    std::vector<terrain_sculpt_sample_edit> pending;
    pending.reserve(edits.size());
    for (const auto& edit : edits)
    {
        if (!std::isfinite(edit.sample.delta) || edit.sample.delta == 0.0f ||
            edit.sample.x > terrain_modifier_sample_coordinate_max ||
            edit.sample.z > terrain_modifier_sample_coordinate_max)
            return {};
        pending.push_back(edit);
    }

    const auto edit_less = [](const terrain_sculpt_sample_edit& lhs, const terrain_sculpt_sample_edit& rhs)
    {
        if (lhs.region.z != rhs.region.z) return lhs.region.z < rhs.region.z;
        if (lhs.region.x != rhs.region.x) return lhs.region.x < rhs.region.x;
        return sample_less(lhs.sample, rhs.sample);
    };
    std::sort(pending.begin(), pending.end(), edit_less);

    std::vector<terrain_sculpt_sample_edit> compact;
    compact.reserve(pending.size());
    for (const auto& edit : pending)
    {
        if (!compact.empty() && compact.back().region == edit.region &&
            compact.back().sample.x == edit.sample.x && compact.back().sample.z == edit.sample.z)
        {
            const auto combined = compact.back().sample.delta + edit.sample.delta;
            if (!std::isfinite(combined)) return {};
            compact.back().sample.delta = combined;
        }
        else
        {
            compact.push_back(edit);
        }
    }
    compact.erase(std::remove_if(compact.begin(), compact.end(),
                                 [](const auto& edit) { return edit.sample.delta == 0.0f; }),
                  compact.end());
    if (compact.empty()) return {};

    std::vector<terrain_region_id> changed_regions;
    for (std::size_t begin = 0; begin < compact.size();)
    {
        const auto region = compact[begin].region;
        auto end = begin + 1u;
        while (end < compact.size() && compact[end].region == region)
            ++end;

        std::vector<terrain_sculpt_sample_delta> samples;
        if (const auto* existing = find_terrain_modifier_payload(*modifier, region);
            existing && std::holds_alternative<terrain_sculpt_region_payload>(existing->data))
            samples = std::get<terrain_sculpt_region_payload>(existing->data).samples;
        sort_samples(samples);

        bool changed{};
        for (auto edit = begin; edit < end; ++edit)
        {
            const auto found = std::lower_bound(samples.begin(), samples.end(), compact[edit].sample,
                                                [](const auto& lhs, const auto& rhs) { return sample_less(lhs, rhs); });
            if (found != samples.end() && found->x == compact[edit].sample.x && found->z == compact[edit].sample.z)
            {
                const auto combined = found->delta + compact[edit].sample.delta;
                if (!std::isfinite(combined)) return {};
                if (combined == found->delta) continue;
                if (combined == 0.0f)
                    samples.erase(found);
                else
                    found->delta = combined;
                changed = true;
            }
            else
            {
                samples.insert(found, compact[edit].sample);
                changed = true;
            }
        }

        if (changed)
        {
            auto& payload = ensure_payload<terrain_sculpt_region_payload>(*modifier, region);
            payload.data = terrain_sculpt_region_payload{std::move(samples)};
            erase_empty_payload(*modifier, region);
            changed_regions.push_back(region);
        }
        begin = end;
    }

    if (changed_regions.empty()) return {};
    recompute_affected_bounds(asset, *modifier);
    if (asset.authoring_revision != std::numeric_limits<std::uint64_t>::max()) ++asset.authoring_revision;

    terrain_dirty_update result;
    result.revision = asset.authoring_revision;
    result.regions = std::move(changed_regions);
    for (const auto region : result.regions)
    {
        auto& record = ensure_terrain_region(asset, region);
        record.dirty_revision = result.revision;
        record.dirty_domains |= terrain_domain::geometry;
    }
    return result;
}

""",
)
replace_once(
    path,
    """            for (const auto& sample : samples)
                if (!std::isfinite(sample.delta) || sample.delta == 0.0f) return false;
""",
    """            for (const auto& sample : samples)
                if (!std::isfinite(sample.delta) || sample.delta == 0.0f ||
                    sample.x > terrain_modifier_sample_coordinate_max ||
                    sample.z > terrain_modifier_sample_coordinate_max)
                    return false;
""",
)
replace_once(
    path,
    """            if (std::any_of(samples.begin(), samples.end(), zero_paint_delta)) return false;
""",
    """            if (std::any_of(samples.begin(), samples.end(), [](const auto& sample)
                            { return zero_paint_delta(sample) || sample.x > terrain_modifier_sample_coordinate_max ||
                                     sample.z > terrain_modifier_sample_coordinate_max; }))
                return false;
""",
)

# -----------------------------------------------------------------------------
# Evaluation: sculpt payloads become a real built-in geometry modifier.
# -----------------------------------------------------------------------------
path = "engine/scene/inc/arc/scene/terrain_evaluator.h"
replace_once(
    path,
    """using terrain_modifier_evaluation_fn =
    std::function<bool(const terrain_modifier_descriptor&, terrain_evaluated_surface&, std::string&)>;
""",
    """using terrain_modifier_evaluation_fn = std::function<bool(const terrain_modifier_descriptor&,
                                                                  const terrain_build_region_snapshot&,
                                                                  terrain_evaluated_surface&, std::string&)>;
""",
)
replace_once(
    path,
    """/** @brief Construct the built-in evaluator with Flat, resolved Heightfield, and `arc.height-offset` implementations. */
""",
    """/** @brief Construct the built-in evaluator with Flat, resolved Heightfield, Sculpt Layer, and height-offset support. */
""",
)

path = "engine/scene/src/common/terrain_evaluator.cpp"
replace_once(
    path,
    """bool apply_height_offset(const terrain_modifier_descriptor& modifier, terrain_evaluated_surface& surface,
                         std::string& error)
""",
    """bool apply_height_offset(const terrain_modifier_descriptor& modifier, const terrain_build_region_snapshot&,
                         terrain_evaluated_surface& surface, std::string& error)
""",
)
insert_before(
    path,
    "std::optional<terrain_evaluated_surface> evaluate_flat_source(const terrain_asset& asset,\n",
    """bool apply_sculpt_layer(const terrain_modifier_descriptor& modifier, const terrain_build_region_snapshot& snapshot,
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

""",
)
replace_once(
    path,
    """        if (!found->second(modifier, *surface, error))
""",
    """        if (!found->second(modifier, result.build_snapshot, *surface, error))
""",
)
replace_once(
    path,
    """    (void)result.register_modifier("arc.height-offset", apply_height_offset);
""",
    """    (void)result.register_modifier("arc.height-offset", apply_height_offset);
    (void)result.register_modifier(std::string(terrain_builtin_modifier_types::sculpt_layer), apply_sculpt_layer);
""",
)

# -----------------------------------------------------------------------------
# Runtime compatibility surface: report per-sample geometry deltas so the
# editor can persist exactly what the legacy preview changed.
# -----------------------------------------------------------------------------
path = "engine/scene/inc/arc/scene/terrain.h"
replace_once(
    path,
    """struct terrain_dirty_region
{
""",
    """struct terrain_sculpt_brush_delta
{
    std::uint32_t x{};
    std::uint32_t z{};
    float delta{};
};

struct terrain_dirty_region
{
""",
)
replace_once(
    path,
    """terrain_dirty_region apply_terrain_brush(terrain_component& terrain, const math::vector3f& local_center,
                                         const terrain_brush_settings& settings, float delta_seconds = 1.0f / 60.0f);
""",
    """terrain_dirty_region apply_terrain_brush(terrain_component& terrain, const math::vector3f& local_center,
                                         const terrain_brush_settings& settings, float delta_seconds = 1.0f / 60.0f,
                                         std::vector<terrain_sculpt_brush_delta>* sculpt_deltas = nullptr);
""",
)

path = "engine/scene/src/common/terrain.cpp"
replace_once(
    path,
    """terrain_dirty_region apply_terrain_brush(terrain_component& terrain, const math::vector3f& local_center,
                                         const terrain_brush_settings& settings, float delta_seconds)
""",
    """terrain_dirty_region apply_terrain_brush(terrain_component& terrain, const math::vector3f& local_center,
                                         const terrain_brush_settings& settings, float delta_seconds,
                                         std::vector<terrain_sculpt_brush_delta>* sculpt_deltas)
""",
)
replace_once(
    path,
    """            const auto index = sample_index(terrain, x, z);
            if (settings.tool == terrain_brush_tool::sculpt)
""",
    """            const auto index = sample_index(terrain, x, z);
            const auto previous_height = terrain.heights[index];
            if (settings.tool == terrain_brush_tool::sculpt)
""",
)
replace_once(
    path,
    """                terrain.layer_weights[index] = normalized_weights(weights);
            }
            changed = true;
""",
    """                terrain.layer_weights[index] = normalized_weights(weights);
            }
            if (sculpt_deltas && settings.tool != terrain_brush_tool::paint)
            {
                const auto delta = terrain.heights[index] - previous_height;
                if (delta != 0.0f && std::isfinite(delta)) sculpt_deltas->push_back({x, z, delta});
            }
            changed = true;
""",
)

# -----------------------------------------------------------------------------
# Native editor host: selected stack modifier is host-authoritative. Sculpt,
# Smooth, and Flatten strokes accumulate into one asset-owned transaction and
# persist once on commit. Paint stays on the compatibility path until M3.5.
# -----------------------------------------------------------------------------
path = "editor/native/src/arc_host_base.inc"
insert_before(
    path,
    "struct arc_host::state\n",
    """struct terrain_sculpt_stroke_session
{
    ecs::entity_guid entity{};
    assets::asset_guid asset_guid{};
    scene::terrain_stable_id modifier{};
    std::filesystem::path source_path;
    scene::terrain_asset asset;
    std::vector<scene::terrain_sculpt_sample_edit> edits;
};

""",
)
replace_once(
    path,
    """    scene::terrain_brush_settings terrain_brush;
    bool terrain_flatten_height_captured{};
""",
    """    scene::terrain_brush_settings terrain_brush;
    std::unordered_map<ecs::entity_guid, scene::terrain_stable_id, ecs::entity_guid_hash> terrain_active_modifiers;
    std::optional<terrain_sculpt_stroke_session> terrain_sculpt_stroke;
    bool terrain_flatten_height_captured{};
""",
)
# Clear staged asset edits when an authoring transaction is cancelled.
replace_once(
    path,
    """        state_->terrain_brush_local_position.reset();
        state_->terrain_stroke_previous_position.reset();
        ++state_->scene_revision;
""",
    """        state_->terrain_brush_local_position.reset();
        state_->terrain_stroke_previous_position.reset();
        state_->terrain_sculpt_stroke.reset();
        ++state_->scene_revision;
""",
)
replace_once(
    path,
    """                    state_->terrain_stroke_previous_position.reset();
                    state_->terrain_flatten_height_captured = false;
""",
    """                    state_->terrain_stroke_previous_position.reset();
                    state_->terrain_sculpt_stroke.reset();
                    state_->terrain_flatten_height_captured = false;
""",
)
# Modifier stack selection is editor session state, not authored asset data.
replace_once(
    path,
    """                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain) return fail("Terrain modifier stack requires a terrain entity", entity);

                const auto stack_json = [&](const scene::terrain_asset* asset, bool asset_backed, bool read_only,
""",
    """                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain) return fail("Terrain modifier stack requires a terrain entity", entity);
                const auto entity_guid = entity_guid_of(state_->scene, entity);
                const auto active_modifier = [&](const scene::terrain_asset* asset) -> scene::terrain_stable_id
                {
                    const auto current = state_->terrain_active_modifiers.find(entity_guid);
                    if (asset && current != state_->terrain_active_modifiers.end())
                    {
                        const auto* modifier = scene::find_terrain_modifier(*asset, current->second);
                        if (modifier) return modifier->id;
                    }
                    if (asset)
                    {
                        for (auto it = asset->modifiers.rbegin(); it != asset->modifiers.rend(); ++it)
                            if (it->enabled && it->type_id == scene::terrain_builtin_modifier_types::sculpt_layer)
                            {
                                state_->terrain_active_modifiers[entity_guid] = it->id;
                                return it->id;
                            }
                    }
                    state_->terrain_active_modifiers.erase(entity_guid);
                    return {};
                };

                const auto stack_json = [&](const scene::terrain_asset* asset, bool asset_backed, bool read_only,
""",
)
replace_once(
    path,
    """                    json["authoringRevision"] = asset ? asset->authoring_revision : 0u;
                    json["modifiers"] = nlohmann::json::array();
""",
    """                    json["authoringRevision"] = asset ? asset->authoring_revision : 0u;
                    const auto active = active_modifier(asset);
                    json["activeModifier"] = active.valid() ? scene::to_string(active) : std::string{};
                    json["modifiers"] = nlohmann::json::array();
""",
)
replace_once(
    path,
    """                if (payload.operation == "inspect")
                    return success(stack_json(&authored, true, read_only, source_reference.generic_string()));
                if (read_only) return fail("Terrain asset is read only", entity);

                const auto bump_revision = [&]
                {
                    if (authored.authoring_revision != std::numeric_limits<std::uint64_t>::max())
                        ++authored.authoring_revision;
                };
                const auto parsed_id = scene::parse_terrain_stable_id(payload.modifier);
                const auto find_modifier = [&]() -> scene::terrain_modifier_descriptor*
                { return parsed_id ? scene::find_terrain_modifier(authored, *parsed_id) : nullptr; };

                if (payload.operation == "add_sculpt")
                    scene::add_terrain_sculpt_layer(authored);
                else if (payload.operation == "add_paint")
                    scene::add_terrain_paint_layer(authored);
""",
    """                if (payload.operation == "inspect")
                    return success(stack_json(&authored, true, read_only, source_reference.generic_string()));

                const auto parsed_id = scene::parse_terrain_stable_id(payload.modifier);
                const auto find_modifier = [&]() -> scene::terrain_modifier_descriptor*
                { return parsed_id ? scene::find_terrain_modifier(authored, *parsed_id) : nullptr; };
                if (payload.operation == "select")
                {
                    const auto* modifier = find_modifier();
                    if (!modifier) return fail("Terrain modifier was not found", entity);
                    state_->terrain_active_modifiers[entity_guid] = modifier->id;
                    return success(stack_json(&authored, true, read_only, source_reference.generic_string()));
                }
                if (read_only) return fail("Terrain asset is read only", entity);

                const auto bump_revision = [&]
                {
                    if (authored.authoring_revision != std::numeric_limits<std::uint64_t>::max())
                        ++authored.authoring_revision;
                };

                if (payload.operation == "add_sculpt")
                {
                    auto& added = scene::add_terrain_sculpt_layer(authored);
                    state_->terrain_active_modifiers[entity_guid] = added.id;
                }
                else if (payload.operation == "add_paint")
                {
                    auto& added = scene::add_terrain_paint_layer(authored);
                    state_->terrain_active_modifiers[entity_guid] = added.id;
                }
""",
)
replace_once(
    path,
    """                    if (found == authored.modifiers.end()) return fail("Terrain modifier was not found", entity);
                    authored.modifiers.erase(found);
                    bump_revision();
                }
                else
""",
    """                    if (found == authored.modifiers.end()) return fail("Terrain modifier was not found", entity);
                    authored.modifiers.erase(found);
                    if (const auto active = state_->terrain_active_modifiers.find(entity_guid);
                        active != state_->terrain_active_modifiers.end() && active->second == *parsed_id)
                        state_->terrain_active_modifiers.erase(active);
                    bump_revision();
                }
                else
""",
)

# Initialize an asset-backed sculpt transaction after the first valid hit.
replace_once(
    path,
    """                state_->terrain_brush_local_position = hit.position;
                if (payload.phase == host_edit_phase::commit)
                {
                    state_->terrain_stroke_previous_position.reset();
                    push_event(state_->events, state_->event_sequence, host_event_type::terrain_stroke_committed,
                               "Terrain stroke committed", entity,
                               "{\\\"revision\\\":" + std::to_string(terrain->content_revision) + '}');
                    return success("{\\\"hit\\\":true,\\\"revision\\\":" + std::to_string(terrain->content_revision) + '}');
                }
                state_->terrain_brush.invert = payload.invert;
                if (payload.phase == host_edit_phase::begin)
                {
                    state_->terrain_flatten_height_captured = true;
                    state_->terrain_brush.flatten_height = hit.position[1];
                    state_->terrain_stroke_previous_position = hit.position;
                }
""",
    """                state_->terrain_brush_local_position = hit.position;
                const bool asset_backed_sculpt = state_->terrain_brush.tool != scene::terrain_brush_tool::paint &&
                                                 (terrain->asset.guid.valid() || !terrain->asset.path_hint.empty());
                if (payload.phase == host_edit_phase::begin)
                {
                    state_->terrain_sculpt_stroke.reset();
                    if (asset_backed_sculpt)
                    {
                        if (!state_->asset_registry) return fail("Terrain asset registry is unavailable", entity);
                        auto reference = terrain->asset;
                        reference.expected_type = assets::asset_types::terrain;
                        if (!reference.guid.valid() && !reference.path_hint.empty())
                            reference = state_->asset_registry->resolve(reference.path_hint, assets::asset_types::terrain);
                        if (!reference.guid.valid()) return fail("Terrain asset reference could not be resolved", entity);
                        auto pending = state_->asset_registry->load<scene::terrain_asset>(
                            {.reference = reference,
                             .priority = assets::asset_streaming_priority::high,
                             .residency = assets::asset_residency::cpu,
                             .allow_fallback = false});
                        auto loaded = pending.get();
                        if (!loaded || !loaded.asset.get())
                            return fail(loaded.error.message.empty() ? "Terrain asset could not be loaded"
                                                                     : loaded.error.message,
                                        entity);
                        auto authored = *loaded.asset.get();
                        const auto entity_guid = entity_guid_of(state_->scene, entity);
                        scene::terrain_stable_id target{};
                        if (const auto active = state_->terrain_active_modifiers.find(entity_guid);
                            active != state_->terrain_active_modifiers.end())
                        {
                            const auto* modifier = scene::find_terrain_modifier(authored, active->second);
                            if (modifier && modifier->enabled &&
                                modifier->type_id == scene::terrain_builtin_modifier_types::sculpt_layer)
                                target = modifier->id;
                        }
                        if (!target.valid())
                            for (auto it = authored.modifiers.rbegin(); it != authored.modifiers.rend(); ++it)
                                if (it->enabled && it->type_id == scene::terrain_builtin_modifier_types::sculpt_layer)
                                {
                                    target = it->id;
                                    state_->terrain_active_modifiers[entity_guid] = target;
                                    break;
                                }
                        if (!target.valid())
                            return fail("Asset-backed sculpting requires an enabled Sculpt Layer", entity);

                        const auto asset_snapshot = state_->asset_registry->find(loaded.asset.resolved_guid());
                        if (asset_snapshot && asset_snapshot->read_only)
                            return fail("Terrain asset is read only", entity);
                        const auto source_reference = !terrain->asset.path_hint.empty()
                                                          ? std::filesystem::path(terrain->asset.path_hint)
                                                      : asset_snapshot ? asset_snapshot->source_path
                                                                       : std::filesystem::path{};
                        const auto resolved = resolve_editor_asset(state_->assets, state_->asset_registry.get(),
                                                                   state_->project.root, source_reference);
                        if (!resolved || resolved->read_only)
                            return fail("Terrain asset source is not writable", entity);
                        state_->terrain_sculpt_stroke = terrain_sculpt_stroke_session{
                            .entity = entity_guid,
                            .asset_guid = loaded.asset.requested_guid(),
                            .modifier = target,
                            .source_path = resolved->path,
                            .asset = std::move(authored)};
                    }
                    state_->terrain_flatten_height_captured = true;
                    state_->terrain_brush.flatten_height = hit.position[1];
                    state_->terrain_stroke_previous_position = hit.position;
                }

                if (payload.phase == host_edit_phase::commit)
                {
                    state_->terrain_stroke_previous_position.reset();
                    if (asset_backed_sculpt)
                    {
                        if (!state_->terrain_sculpt_stroke ||
                            state_->terrain_sculpt_stroke->entity != entity_guid_of(state_->scene, entity))
                            return fail("Terrain sculpt transaction is missing", entity);
                        auto session = std::move(*state_->terrain_sculpt_stroke);
                        state_->terrain_sculpt_stroke.reset();
                        const auto update = scene::accumulate_terrain_sculpt_samples(
                            session.asset, session.modifier, session.edits);
                        if (update.revision != 0u)
                        {
                            const auto encoded = scene::write_terrain_asset_json(session.asset, true);
                            if (!encoded) return fail("Terrain asset could not be serialized", entity);
                            std::ofstream output(session.source_path, std::ios::binary | std::ios::trunc);
                            if (!output) return fail("Terrain asset source could not be opened for writing", entity);
                            output.write(encoded.value().data(), static_cast<std::streamsize>(encoded.value().size()));
                            if (!output) return fail("Terrain asset source could not be written", entity);
                            terrain->asset_authoring_revision = session.asset.authoring_revision;
                            if (state_->asset_registry)
                            {
                                state_->asset_registry->mark_stale(session.asset_guid, "Terrain sculpt layer changed");
                                [[maybe_unused]] const auto reimport = state_->asset_registry->reimport(
                                    session.asset_guid, assets::asset_streaming_priority::high);
                            }
                            push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                                       "Terrain sculpt layer changed", entity);
                        }
                    }
                    push_event(state_->events, state_->event_sequence, host_event_type::terrain_stroke_committed,
                               "Terrain stroke committed", entity,
                               "{\\\"revision\\\":" + std::to_string(terrain->content_revision) + '}');
                    return success("{\\\"hit\\\":true,\\\"revision\\\":" + std::to_string(terrain->content_revision) + '}');
                }
                state_->terrain_brush.invert = payload.invert;
""",
)
# Capture geometry deltas during the compatibility-preview brush pass and map
# them into stable authoring-region addresses.
replace_once(
    path,
    """                const auto stamp_seconds = payload.elapsed_seconds / static_cast<float>(stamp_count);
                for (std::uint32_t stamp_index = 1; stamp_index <= stamp_count; ++stamp_index)
                {
                    const auto alpha = static_cast<float>(stamp_index) / static_cast<float>(stamp_count);
                    merge_dirty(scene::apply_terrain_brush(*terrain, previous + offset * alpha, state_->terrain_brush,
                                                           stamp_seconds));
                }
                state_->terrain_stroke_previous_position = hit.position;
""",
    """                const auto stamp_seconds = payload.elapsed_seconds / static_cast<float>(stamp_count);
                std::vector<scene::terrain_sculpt_brush_delta> sculpt_deltas;
                for (std::uint32_t stamp_index = 1; stamp_index <= stamp_count; ++stamp_index)
                {
                    const auto alpha = static_cast<float>(stamp_index) / static_cast<float>(stamp_count);
                    merge_dirty(scene::apply_terrain_brush(
                        *terrain, previous + offset * alpha, state_->terrain_brush, stamp_seconds,
                        asset_backed_sculpt ? &sculpt_deltas : nullptr));
                }
                if (asset_backed_sculpt && state_->terrain_sculpt_stroke)
                {
                    const auto spacing = terrain->size / static_cast<float>(terrain->subdivisions);
                    const auto half = terrain->size * 0.5f;
                    auto& session = *state_->terrain_sculpt_stroke;
                    session.edits.reserve(session.edits.size() + sculpt_deltas.size());
                    for (const auto& delta : sculpt_deltas)
                    {
                        const math::vector3f local{-half + static_cast<float>(delta.x) * spacing, 0.0f,
                                                   -half + static_cast<float>(delta.z) * spacing};
                        const auto world = math::transform_point(terrain_world, local);
                        const auto address = scene::terrain_modifier_sample_at(
                            session.asset.coordinates, session.asset.partition, static_cast<double>(world[0]),
                            static_cast<double>(world[2]));
                        session.edits.push_back({address.region, {address.x, address.z, delta.delta}});
                    }
                }
                state_->terrain_stroke_previous_position = hit.position;
""",
)

# -----------------------------------------------------------------------------
# Terrain Stack selection now tells the native host which stable layer receives
# future sculpt strokes.
# -----------------------------------------------------------------------------
path = "editor/src/renderer/src/terrain/TerrainStackPanel.tsx"
replace_once(
    path,
    """  authoringRevision: number;
  modifiers: TerrainModifierSnapshot[];
""",
    """  authoringRevision: number;
  activeModifier: string;
  modifiers: TerrainModifierSnapshot[];
""",
)
replace_once(
    path,
    """    setSelectedId(stack.modifiers.at(-1)?.id ?? '');
""",
    """    const active = stack.modifiers.find((modifier) => modifier.id === stack.activeModifier);
    setSelectedId(active?.id ?? stack.modifiers.at(-1)?.id ?? '');
""",
)
replace_once(
    path,
    """  const mutate = async (operation: string, extra: Record<string, unknown> = {}) => {
""",
    """  const selectModifier = async (modifier: TerrainModifierSnapshot) => {
    setSelectedId(modifier.id);
    const next = await execute('select', { modifier: modifier.id });
    if (!next) return;
    setSelectedId(modifier.id);
  };

  const mutate = async (operation: string, extra: Record<string, unknown> = {}) => {
""",
)
replace_once(
    path,
    """                  onClick={() => setSelectedId(modifier.id)}
""",
    """                  onClick={() => void selectModifier(modifier)}
""",
)

path = "editor/src/renderer/src/terrain/TerrainStackPanel.test.tsx"
replace_once(
    path,
    """  authoringRevision: 4,
  modifiers: [sculpt, paint],
""",
    """  authoringRevision: 4,
  activeModifier: sculpt.id,
  modifiers: [sculpt, paint],
""",
)
replace_once(
    path,
    """  it('uses modifier stable ids when toggling visibility', async () => {
""",
    """  it('uses modifier stable ids when selecting the sculpt target', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true, payload: { ...snapshot, activeModifier: paint.id } });
    render(<TerrainStackPanel command={command} entity={entity} />);

    await screen.findByText('Ground');
    await userEvent.click(screen.getByRole('option', { name: /Ground/ }));

    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('terrain.modifierStack', {
        entity,
        operation: 'select',
        modifier: paint.id,
      }),
    );
  });

  it('uses modifier stable ids when toggling visibility', async () => {
""",
)

# -----------------------------------------------------------------------------
# Focused engine tests for M3.3 authoring semantics.
# -----------------------------------------------------------------------------
Path("engine/scene/tests/terrain_m3_sculpt_stroke_tests.cpp").write_text(
    r'''#include <arc/scene/terrain.h>
#include <arc/scene/terrain_asset.h>
#include <arc/scene/terrain_evaluator.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

namespace
{

arc::scene::terrain_asset make_asset()
{
    arc::scene::terrain_asset asset;
    asset.source.id = arc::scene::generate_terrain_stable_id();
    asset.source.kind = arc::scene::terrain_source_kind::flat;
    asset.coordinates = {.origin_x = 0.0, .origin_y = 0.0, .origin_z = 0.0, .meters_per_unit = 1.0};
    asset.partition = {.authoring_region_size = 100.0, .dependency_halo = 4.0};
    return asset;
}

} // namespace

TEST_CASE("M3.3 sculpt edits fold into stable sparse region payloads with one revision")
{
    using namespace arc::scene;
    auto asset = make_asset();
    const auto layer = add_terrain_sculpt_layer(asset, "Detail").id;
    const auto start_revision = asset.authoring_revision;
    const auto left = terrain_modifier_sample_at(asset.coordinates, asset.partition, -25.0, 10.0);
    const auto right = terrain_modifier_sample_at(asset.coordinates, asset.partition, 125.0, 10.0);

    const std::array edits{
        terrain_sculpt_sample_edit{left.region, {left.x, left.z, 1.25f}},
        terrain_sculpt_sample_edit{left.region, {left.x, left.z, -0.25f}},
        terrain_sculpt_sample_edit{right.region, {right.x, right.z, 2.0f}},
    };
    const auto update = accumulate_terrain_sculpt_samples(asset, layer, edits);

    CHECK(update.revision == start_revision + 1u);
    REQUIRE(update.regions.size() == 2u);
    const auto* modifier = find_terrain_modifier(asset, layer);
    REQUIRE(modifier != nullptr);
    REQUIRE(modifier->region_payloads.size() == 2u);
    const auto* left_payload = find_terrain_modifier_payload(*modifier, left.region);
    REQUIRE(left_payload != nullptr);
    const auto& left_samples = std::get<terrain_sculpt_region_payload>(left_payload->data).samples;
    REQUIRE(left_samples.size() == 1u);
    CHECK(left_samples.front().delta == 1.0f);
    CHECK(asset.regions.size() == 2u);
    CHECK(validate_terrain_asset(asset).valid());
}

TEST_CASE("M3.3 sculpt edits remove sparse samples when accumulated delta returns to zero")
{
    using namespace arc::scene;
    auto asset = make_asset();
    const auto layer = add_terrain_sculpt_layer(asset).id;
    const auto address = terrain_modifier_sample_at(asset.coordinates, asset.partition, 10.0, 20.0);
    REQUIRE(accumulate_terrain_sculpt_samples(
                asset, layer, std::array{terrain_sculpt_sample_edit{address.region, {address.x, address.z, 2.0f}}})
                .revision != 0u);
    REQUIRE(accumulate_terrain_sculpt_samples(
                asset, layer, std::array{terrain_sculpt_sample_edit{address.region, {address.x, address.z, -2.0f}}})
                .revision != 0u);

    const auto* modifier = find_terrain_modifier(asset, layer);
    REQUIRE(modifier != nullptr);
    CHECK(find_terrain_modifier_payload(*modifier, address.region) == nullptr);
    CHECK_FALSE(modifier->affected_bounds.has_value());
}

TEST_CASE("M3.3 compatibility brush reports exact per-sample geometry deltas")
{
    using namespace arc::scene;
    terrain_component terrain;
    terrain.size = 4.0f;
    terrain.subdivisions = 4u;
    terrain.heights.assign(25u, 0.0f);
    terrain.layer_weights.assign(25u, {255u, 0u, 0u, 0u});
    std::vector<terrain_sculpt_brush_delta> deltas;
    terrain_brush_settings brush;
    brush.tool = terrain_brush_tool::sculpt;
    brush.radius = 0.25f;
    brush.strength = 0.5f;

    const auto dirty = apply_terrain_brush(terrain, {0.0f, 0.0f, 0.0f}, brush, 1.0f, &deltas);
    REQUIRE(dirty.valid);
    REQUIRE(deltas.size() == 1u);
    CHECK(deltas.front().x == 2u);
    CHECK(deltas.front().z == 2u);
    CHECK(deltas.front().delta == 6.0f);
}

TEST_CASE("M3.3 default evaluator applies sculpt payload for the requested authoring region")
{
    using namespace arc::scene;
    auto asset = make_asset();
    asset.source.kind = terrain_source_kind::heightfield;
    asset.source.asset.guid = arc::assets::generate_asset_guid();
    const auto layer = add_terrain_sculpt_layer(asset).id;
    const auto center = terrain_modifier_sample_coordinate_max / 2u;
    REQUIRE(set_terrain_sculpt_region_samples(asset, layer, {0, 0}, {{center, center, 2.5f}}).revision != 0u);

    std::array<float, 9> heights{};
    std::array<std::array<std::uint8_t, 4>, 9> weights{};
    for (auto& weight : weights)
        weight = {255u, 0u, 0u, 0u};
    terrain_evaluation_request request;
    request.region = {0, 0};
    request.heightfield_source = terrain_heightfield_source_view{.sample_width = 3u,
                                                                  .sample_height = 3u,
                                                                  .width = 100.0f,
                                                                  .depth = 100.0f,
                                                                  .heights = heights,
                                                                  .material_weights = weights,
                                                                  .source_revision = 1u};

    const auto evaluator = make_default_terrain_evaluator();
    const auto result = evaluator.evaluate(asset, request);
    REQUIRE(result.succeeded);
    const auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&result.surface.geometry);
    REQUIRE(heightfield != nullptr);
    REQUIRE(heightfield->heights.size() == 9u);
    CHECK(heightfield->heights[4] == 2.5f);
}
'''
)
