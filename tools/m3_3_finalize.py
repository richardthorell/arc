from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1))


path = "engine/scene/src/common/terrain_edit_layers.cpp"
replace_once(
    path,
    """    samples.erase(std::remove_if(samples.begin(), samples.end(), [](const auto& sample)
                                 { return !std::isfinite(sample.delta) || sample.delta == 0.0f; }),
                  samples.end());
""",
    """    samples.erase(std::remove_if(samples.begin(), samples.end(), [](const auto& sample)
                                 { return !std::isfinite(sample.delta) || sample.delta == 0.0f ||
                                          sample.x > terrain_modifier_sample_coordinate_max ||
                                          sample.z > terrain_modifier_sample_coordinate_max; }),
                  samples.end());
""",
)

path = "editor/native/src/arc_host_base.inc"
replace_once(
    path,
    """                const auto hit = scene::raycast_terrain(*terrain, local_origin, local_direction);
                if (!hit.hit)
                {
                    state_->terrain_brush_local_position.reset();
                    state_->terrain_stroke_previous_position.reset();
                    return success("{\\\"hit\\\":false}");
                }
                state_->terrain_brush_local_position = hit.position;
                const bool asset_backed_sculpt = state_->terrain_brush.tool != scene::terrain_brush_tool::paint &&
                                                 (terrain->asset.guid.valid() || !terrain->asset.path_hint.empty());
""",
    """                const auto hit = scene::raycast_terrain(*terrain, local_origin, local_direction);
                const bool asset_backed_sculpt = state_->terrain_brush.tool != scene::terrain_brush_tool::paint &&
                                                 (terrain->asset.guid.valid() || !terrain->asset.path_hint.empty());
                if (!hit.hit && payload.phase != host_edit_phase::commit)
                {
                    state_->terrain_brush_local_position.reset();
                    state_->terrain_stroke_previous_position.reset();
                    return success("{\\\"hit\\\":false}");
                }
                if (hit.hit) state_->terrain_brush_local_position = hit.position;
""",
)
replace_once(
    path,
    """                            std::ofstream output(session.source_path, std::ios::binary | std::ios::trunc);
                            if (!output) return fail("Terrain asset source could not be opened for writing", entity);
                            output.write(encoded.value().data(), static_cast<std::streamsize>(encoded.value().size()));
                            if (!output) return fail("Terrain asset source could not be written", entity);
                            terrain->asset_authoring_revision = session.asset.authoring_revision;
""",
    """                            {
                                std::ofstream output(session.source_path, std::ios::binary | std::ios::trunc);
                                if (!output) return fail("Terrain asset source could not be opened for writing", entity);
                                output.write(encoded.value().data(),
                                             static_cast<std::streamsize>(encoded.value().size()));
                                if (!output) return fail("Terrain asset source could not be written", entity);
                            }
                            terrain->asset_authoring_revision = session.asset.authoring_revision;
""",
)
replace_once(
    path,
    """                    return success("{\\\"hit\\\":true,\\\"revision\\\":" + std::to_string(terrain->content_revision) + '}');
""",
    """                    return success("{\\\"hit\\\":" + std::string(hit.hit ? "true" : "false") +
                                   ",\\\"revision\\\":" + std::to_string(terrain->content_revision) + '}');
""",
)
replace_once(
    path,
    """                        const math::vector3f local{-half + static_cast<float>(delta.x) * spacing, 0.0f,
                                                   -half + static_cast<float>(delta.z) * spacing};
""",
    """                        const auto sample_index = static_cast<std::size_t>(delta.z) * (terrain->subdivisions + 1u) +
                                                  delta.x;
                        const auto sample_height = sample_index < terrain->heights.size() ? terrain->heights[sample_index]
                                                                                           : 0.0f;
                        const math::vector3f local{-half + static_cast<float>(delta.x) * spacing, sample_height,
                                                   -half + static_cast<float>(delta.z) * spacing};
""",
)
