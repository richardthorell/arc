from pathlib import Path

# 1) Replace the O(N^2) heightfield raycast with regular-grid DDA traversal.
terrain_cpp = Path("engine/scene/src/common/terrain.cpp")
text = terrain_cpp.read_text()
begin = text.index("terrain_raycast_hit raycast_terrain(const terrain_component& terrain, const math::vector3f& local_origin,\n")
end = text.index("\nterrain_raycast_hit raycast_terrain(const terrain_component& terrain, const render::terrain_hierarchy& hierarchy,", begin)
new_raycast = r'''terrain_raycast_hit raycast_terrain(const terrain_component& terrain, const math::vector3f& local_origin,
                                    const math::vector3f& local_direction) noexcept
{
    terrain_raycast_hit result{};
    if (!terrain_heightfield_valid(terrain)) return result;

    constexpr float epsilon = 1.0e-6f;
    const float half = terrain.size * 0.5f;
    const float spacing = terrain.size / terrain.subdivisions;
    float enter = 0.0f;
    float exit = std::numeric_limits<float>::max();
    const auto clip_axis = [&](float origin, float direction)
    {
        if (std::abs(direction) <= epsilon) return origin >= -half && origin <= half;
        float first = (-half - origin) / direction;
        float second = (half - origin) / direction;
        if (first > second) std::swap(first, second);
        enter = std::max(enter, first);
        exit = std::min(exit, second);
        return enter <= exit;
    };
    if (!clip_axis(local_origin[0], local_direction[0]) || !clip_axis(local_origin[2], local_direction[2]) ||
        exit < 0.0f)
        return result;
    enter = std::max(enter, 0.0f);
    if (enter > exit) return result;

    const auto start = math::add(local_origin, math::mul(local_direction, enter));
    const auto initial_cell = [&](float value, float direction)
    {
        const float coordinate = (value + half) / spacing;
        int cell = static_cast<int>(std::floor(coordinate));
        cell = std::clamp(cell, 0, static_cast<int>(terrain.subdivisions) - 1);
        const float boundary = std::round(coordinate);
        if (direction < -epsilon && std::abs(coordinate - boundary) <= epsilon && cell > 0) --cell;
        return cell;
    };
    int x = initial_cell(start[0], local_direction[0]);
    int z = initial_cell(start[2], local_direction[2]);
    const int step_x = local_direction[0] > epsilon ? 1 : local_direction[0] < -epsilon ? -1 : 0;
    const int step_z = local_direction[2] > epsilon ? 1 : local_direction[2] < -epsilon ? -1 : 0;
    const float infinity = std::numeric_limits<float>::infinity();
    const auto first_crossing = [&](int cell, int step, float origin, float direction)
    {
        if (step == 0) return infinity;
        const float boundary = -half + static_cast<float>(step > 0 ? cell + 1 : cell) * spacing;
        return (boundary - origin) / direction;
    };
    float next_x = first_crossing(x, step_x, local_origin[0], local_direction[0]);
    float next_z = first_crossing(z, step_z, local_origin[2], local_direction[2]);
    const float delta_x = step_x == 0 ? infinity : spacing / std::abs(local_direction[0]);
    const float delta_z = step_z == 0 ? infinity : spacing / std::abs(local_direction[2]);
    float nearest = std::numeric_limits<float>::max();

    while (x >= 0 && z >= 0 && x < static_cast<int>(terrain.subdivisions) &&
           z < static_cast<int>(terrain.subdivisions))
    {
        const auto sx = static_cast<std::uint32_t>(x);
        const auto sz = static_cast<std::uint32_t>(z);
        const math::vector3f a{-half + static_cast<float>(sx) * spacing, height_at(terrain, sx, sz),
                               -half + static_cast<float>(sz) * spacing};
        const math::vector3f b{a[0] + spacing, height_at(terrain, sx + 1u, sz), a[2]};
        const math::vector3f c{a[0] + spacing, height_at(terrain, sx + 1u, sz + 1u), a[2] + spacing};
        const math::vector3f d{a[0], height_at(terrain, sx, sz + 1u), a[2] + spacing};
        float distance{};
        if ((intersect_triangle(local_origin, local_direction, a, b, c, distance) ||
             intersect_triangle(local_origin, local_direction, a, c, d, distance)) &&
            distance + epsilon >= enter && distance <= exit + epsilon && distance < nearest)
            nearest = distance;

        const float crossing = std::min(next_x, next_z);
        if (nearest <= crossing + epsilon || crossing > exit) break;
        const float previous_x = next_x;
        const float previous_z = next_z;
        if (previous_x <= previous_z + epsilon)
        {
            x += step_x;
            next_x += delta_x;
        }
        if (previous_z <= previous_x + epsilon)
        {
            z += step_z;
            next_z += delta_z;
        }
    }

    if (nearest < std::numeric_limits<float>::max())
    {
        result.hit = true;
        result.distance = nearest;
        result.position = math::add(local_origin, math::mul(local_direction, nearest));
        result.normal = sample_terrain_normal(terrain, result.position[0], result.position[2]);
    }
    return result;
}
'''
terrain_cpp.write_text(text[:begin] + new_raycast + text[end:])

# 2 + 4) Make dirty-region compatibility publication local and preserve asset-owned handles.
proxy_cpp = Path("engine/scene/src/common/terrain_geometry_proxy.cpp")
text = proxy_cpp.read_text()
old_destroy = r'''void destroy_unreused_old_resources(terrain_render_proxy& previous,
                                    const std::vector<terrain_render_region_proxy>& replacement,
                                    render::renderer& renderer)
{
    for (auto& region : previous.regions)
    {
        if (!geometry_reused(replacement, region.geometry) &&
            (region.geometry.conventional.valid() || region.geometry.virtualized.valid()))
            (void)renderer.destroy_geometry_resource(region.geometry);
        if (!texture_reused(replacement, region.surface_attribute_texture) &&
            renderer.texture_alive(region.surface_attribute_texture))
            renderer.destroy_texture(region.surface_attribute_texture);
        region.geometry = render::geometry_resource_handle{};
        region.surface_attribute_texture = {};
    }
}
'''
new_destroy = r'''void destroy_unreused_old_resources(terrain_render_proxy& previous,
                                    const std::vector<terrain_render_region_proxy>& replacement,
                                    render::renderer& renderer, const terrain_render_proxy* protected_proxy = nullptr)
{
    for (auto& region : previous.regions)
    {
        const bool geometry_protected = protected_proxy && geometry_reused(protected_proxy->regions, region.geometry);
        const bool attributes_protected =
            protected_proxy && texture_reused(protected_proxy->regions, region.surface_attribute_texture);
        if (!geometry_reused(replacement, region.geometry) && !geometry_protected &&
            (region.geometry.conventional.valid() || region.geometry.virtualized.valid()))
            (void)renderer.destroy_geometry_resource(region.geometry);
        if (!texture_reused(replacement, region.surface_attribute_texture) && !attributes_protected &&
            renderer.texture_alive(region.surface_attribute_texture))
            renderer.destroy_texture(region.surface_attribute_texture);
        region.geometry = render::geometry_resource_handle{};
        region.surface_attribute_texture = {};
    }
}
'''
if old_destroy not in text:
    raise SystemExit("terrain proxy destruction anchor changed")
text = text.replace(old_destroy, new_destroy, 1)

namespace_anchor = r'''void cleanup_staged_resources(const terrain_render_proxy& previous, std::vector<terrain_render_region_proxy>& staged,
                              render::renderer& renderer)
{
    for (auto& region : staged)
    {
        const bool geometry_owned_by_previous = geometry_reused(previous.regions, region.geometry);
        const bool attributes_owned_by_previous = texture_reused(previous.regions, region.surface_attribute_texture);
        if (!geometry_owned_by_previous &&
            (region.geometry.conventional.valid() || region.geometry.virtualized.valid()))
            (void)renderer.destroy_geometry_resource(region.geometry);
        if (!attributes_owned_by_previous && renderer.texture_alive(region.surface_attribute_texture))
            renderer.destroy_texture(region.surface_attribute_texture);
        region.geometry = render::geometry_resource_handle{};
        region.surface_attribute_texture = {};
    }
}

} // namespace
'''
namespace_replacement = r'''void cleanup_staged_resources(const terrain_render_proxy& previous, std::vector<terrain_render_region_proxy>& staged,
                              render::renderer& renderer)
{
    for (auto& region : staged)
    {
        const bool geometry_owned_by_previous = geometry_reused(previous.regions, region.geometry);
        const bool attributes_owned_by_previous = texture_reused(previous.regions, region.surface_attribute_texture);
        if (!geometry_owned_by_previous &&
            (region.geometry.conventional.valid() || region.geometry.virtualized.valid()))
            (void)renderer.destroy_geometry_resource(region.geometry);
        if (!attributes_owned_by_previous && renderer.texture_alive(region.surface_attribute_texture))
            renderer.destroy_texture(region.surface_attribute_texture);
        region.geometry = render::geometry_resource_handle{};
        region.surface_attribute_texture = {};
    }
}

bool render_region_overlaps_dirty(const terrain_render_region& region, const terrain_surface_ir& source,
                                  const terrain_dirty_region* dirty) noexcept
{
    if (!dirty || !dirty->valid) return true;
    const auto* heightfield = std::get_if<terrain_surface_heightfield_ir>(&source.geometry);
    if (!heightfield || heightfield->sample_width < 2u || heightfield->sample_height < 2u) return true;
    const auto last_x = heightfield->sample_width - 1u;
    const auto last_z = heightfield->sample_height - 1u;
    const auto min_x = std::min(dirty->min_x, last_x);
    const auto max_x = std::min(dirty->max_x, last_x);
    const auto min_z = std::min(dirty->min_z, last_z);
    const auto max_z = std::min(dirty->max_z, last_z);
    const double spacing_x = (source.local_bounds.max_x - source.local_bounds.min_x) / last_x;
    const double spacing_z = (source.local_bounds.max_z - source.local_bounds.min_z) / last_z;
    const double dirty_min_x = source.local_bounds.min_x + static_cast<double>(min_x) * spacing_x;
    const double dirty_max_x = source.local_bounds.min_x + static_cast<double>(max_x) * spacing_x;
    const double dirty_min_z = source.local_bounds.min_z + static_cast<double>(min_z) * spacing_z;
    const double dirty_max_z = source.local_bounds.min_z + static_cast<double>(max_z) * spacing_z;
    const auto& bounds = region.surface.local_bounds;
    return bounds.max_x >= dirty_min_x && bounds.min_x <= dirty_max_x && bounds.max_z >= dirty_min_z &&
           bounds.min_z <= dirty_max_z;
}

} // namespace
'''
if namespace_anchor not in text:
    raise SystemExit("terrain proxy namespace anchor changed")
text = text.replace(namespace_anchor, namespace_replacement, 1)

sync_begin = text.index("bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_surface_ir& surface,\n")
sync_end = text.index("\nbool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_component& terrain,", sync_begin)
new_sync = r'''bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_surface_ir& surface,
                                             const terrain_component& terrain, render::renderer& renderer,
                                             const terrain_dirty_region* dirty_region)
{
    if (!guid.valid() || !validate_terrain_surface_ir(surface)) return false;

    auto& proxy = proxies_[guid];
    if (proxy.synchronized_revision == surface.source_revision && proxy_resources_alive(proxy, renderer))
    {
        proxy.material = terrain.material;
        return true;
    }

    auto source_regions = build_terrain_render_regions(surface);
    if (source_regions.empty()) return false;
    if (dirty_region && dirty_region->valid)
        std::erase_if(source_regions,
                      [&](const auto& region) { return !render_region_overlaps_dirty(region, surface, dirty_region); });
    if (source_regions.empty())
    {
        proxy.synchronized_revision = surface.source_revision;
        proxy.material = terrain.material;
        return true;
    }

    const auto retained = retained_.find(guid);
    const terrain_render_proxy* protected_proxy = retained == retained_.end() ? nullptr : &retained->second;
    const bool partial = dirty_region && dirty_region->valid && !proxy.regions.empty();
    std::vector<terrain_render_region_proxy> staged = partial ? proxy.regions : std::vector<terrain_render_region_proxy>{};
    staged.reserve(std::max(staged.size(), source_regions.size()));
    for (auto& source_region : source_regions)
    {
        const auto* previous = find_region(proxy, source_region.id);
        terrain_render_region_proxy next;
        next.id = source_region.id;
        next.local_bounds = terrain_local_bounds(source_region.surface.local_bounds);
        next.geometry_fingerprint = source_region.geometry_fingerprint;
        next.attribute_fingerprint = source_region.attribute_fingerprint;

        if (previous && previous->geometry_fingerprint == source_region.geometry_fingerprint &&
            geometry_alive(*previous, renderer))
        {
            next.geometry = previous->geometry;
        }
        else
        {
            const auto view = source_region.surface.view();
            auto artifact = source_region.vertex_normals.empty()
                                ? build_terrain_render_geometry(view)
                                : build_terrain_render_region_geometry(view, source_region.vertex_normals);
            if (!artifact)
            {
                cleanup_staged_resources(proxy, staged, renderer);
                return false;
            }
            next.geometry = renderer.create_geometry_resource(
                std::move(*artifact), terrain_geometry_generation(source_region.geometry_fingerprint));
            if (!next.geometry.valid() || !renderer.mesh_alive(next.geometry.conventional))
            {
                if (next.geometry.conventional.valid() || next.geometry.virtualized.valid())
                    (void)renderer.destroy_geometry_resource(next.geometry);
                cleanup_staged_resources(proxy, staged, renderer);
                return false;
            }
        }

        const auto view = source_region.surface.view();
        auto attributes = build_terrain_render_attributes(view);
        if (!attributes)
        {
            if (!previous || next.geometry != previous->geometry)
                (void)renderer.destroy_geometry_resource(next.geometry);
            cleanup_staged_resources(proxy, staged, renderer);
            return false;
        }

        const bool attributes_protected =
            previous && protected_proxy && texture_reused(protected_proxy->regions, previous->surface_attribute_texture);
        if (previous && previous->attribute_fingerprint == source_region.attribute_fingerprint &&
            attributes_alive(*previous, renderer))
        {
            next.surface_attribute_texture = previous->surface_attribute_texture;
        }
        else if (previous && !attributes_protected && attributes_alive(*previous, renderer) &&
                 renderer.update_texture(previous->surface_attribute_texture, make_attribute_texture_data(*attributes)))
        {
            next.surface_attribute_texture = previous->surface_attribute_texture;
        }
        else
        {
            next.surface_attribute_texture = create_attribute_texture(*attributes, renderer);
            if (!next.surface_attribute_texture.valid())
            {
                if (!previous || next.geometry != previous->geometry)
                    (void)renderer.destroy_geometry_resource(next.geometry);
                cleanup_staged_resources(proxy, staged, renderer);
                return false;
            }
        }

        const auto existing =
            std::find_if(staged.begin(), staged.end(), [&](const auto& region) { return region.id == next.id; });
        if (existing == staged.end())
            staged.push_back(next);
        else
            *existing = next;
    }

    destroy_unreused_old_resources(proxy, staged, renderer, protected_proxy);
    proxy.regions = std::move(staged);
    proxy.synchronized_revision = surface.source_revision;
    proxy.material = terrain.material;
    return true;
}
'''
text = text[:sync_begin] + new_sync + text[sync_end:]

old_wrapper = r'''bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_component& terrain,
                                             render::renderer& renderer, const terrain_dirty_region* dirty_region)
{
    if (auto* proxy = find(guid); proxy && proxy->asset_owned)
    {
        if (!dirty_region)
        {
            proxy->material = terrain.material;
            return true;
        }
        retained_[guid] = std::move(*proxy);
        *proxy = {};
    }
    const auto surface = make_legacy_terrain_surface_ir(terrain);
    return surface && synchronize(guid, *surface, terrain, renderer, dirty_region);
}
'''
new_wrapper = r'''bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_component& terrain,
                                             render::renderer& renderer, const terrain_dirty_region* dirty_region)
{
    if (auto* proxy = find(guid); proxy && proxy->asset_owned)
    {
        if (!dirty_region || !dirty_region->valid)
        {
            proxy->material = terrain.material;
            return true;
        }
        retained_[guid] = std::move(*proxy);
        *proxy = retained_[guid];
        proxy->asset_owned = false;
    }
    const auto surface = make_legacy_terrain_surface_ir(terrain);
    return surface && synchronize(guid, *surface, terrain, renderer, dirty_region);
}
'''
if old_wrapper not in text:
    raise SystemExit("terrain proxy compatibility wrapper anchor changed")
text = text.replace(old_wrapper, new_wrapper, 1)

old_publish_tail = r'''    const auto generation = previous.generation + 1u;
    destroy_unreused_old_resources(previous, staged, renderer);
    if (&previous != &visible) destroy_proxy(visible, renderer);
    visible.regions = std::move(staged);
'''
new_publish_tail = r'''    const auto generation = previous.generation + 1u;
    if (&previous != &visible)
    {
        cleanup_staged_resources(previous, visible.regions, renderer);
        visible.regions.clear();
    }
    destroy_unreused_old_resources(previous, staged, renderer);
    visible.regions = std::move(staged);
'''
if old_publish_tail not in text:
    raise SystemExit("terrain publish transition anchor changed")
text = text.replace(old_publish_tail, new_publish_tail, 1)

erase_begin = text.index("bool terrain_render_proxy_cache::erase(ecs::entity_guid guid, render::renderer& renderer)\n")
release_begin = text.index("\nvoid terrain_render_proxy_cache::release_missing", erase_begin)
new_erase = r'''bool terrain_render_proxy_cache::erase(ecs::entity_guid guid, render::renderer& renderer)
{
    const auto retained = retained_.find(guid);
    const auto found = proxies_.find(guid);
    if (retained != retained_.end())
    {
        if (found != proxies_.end())
        {
            cleanup_staged_resources(retained->second, found->second.regions, renderer);
            proxies_.erase(found);
        }
        destroy_proxy(retained->second, renderer);
        retained_.erase(retained);
        return true;
    }
    if (found == proxies_.end()) return false;
    destroy_proxy(found->second, renderer);
    proxies_.erase(found);
    return true;
}
'''
text = text[:erase_begin] + new_erase + text[release_begin:]

release_begin = text.index("void terrain_render_proxy_cache::release_missing(std::span<const ecs::entity_guid> active, render::renderer& renderer)\n")
clear_begin = text.index("\nvoid terrain_render_proxy_cache::clear", release_begin)
new_release = r'''void terrain_render_proxy_cache::release_missing(std::span<const ecs::entity_guid> active, render::renderer& renderer)
{
    for (auto found = retained_.begin(); found != retained_.end();)
    {
        if (std::find(active.begin(), active.end(), found->first) != active.end())
        {
            ++found;
            continue;
        }
        if (const auto visible = proxies_.find(found->first); visible != proxies_.end())
        {
            cleanup_staged_resources(found->second, visible->second.regions, renderer);
            proxies_.erase(visible);
        }
        destroy_proxy(found->second, renderer);
        found = retained_.erase(found);
    }
    for (auto found = proxies_.begin(); found != proxies_.end();)
    {
        if (std::find(active.begin(), active.end(), found->first) != active.end())
        {
            ++found;
            continue;
        }
        destroy_proxy(found->second, renderer);
        found = proxies_.erase(found);
    }
}
'''
text = text[:release_begin] + new_release + text[clear_begin:]

clear_begin = text.index("void terrain_render_proxy_cache::clear(render::renderer& renderer)\n")
clear_end = text.index("\n} // namespace arc::scene", clear_begin)
new_clear = r'''void terrain_render_proxy_cache::clear(render::renderer& renderer)
{
    for (auto& [guid, retained] : retained_)
    {
        if (const auto visible = proxies_.find(guid); visible != proxies_.end())
        {
            cleanup_staged_resources(retained, visible->second.regions, renderer);
            proxies_.erase(visible);
        }
        destroy_proxy(retained, renderer);
    }
    retained_.clear();
    for (auto& [guid, proxy] : proxies_)
    {
        (void)guid;
        destroy_proxy(proxy, renderer);
    }
    proxies_.clear();
}
'''
text = text[:clear_begin] + new_clear + text[clear_end:]
proxy_cpp.write_text(text)

# 3) Queue compatibility preview dirtiness and publish it once per viewport frame.
host = Path("editor/native/src/arc_host_base.inc")
text = host.read_text()
pump_anchor = "    void pump_terrain_rebuilds()\n    {\n"
if pump_anchor not in text:
    raise SystemExit("terrain rebuild pump anchor changed")
preview_methods = r'''    void queue_terrain_preview(ecs::entity_guid guid, const scene::terrain_dirty_region& dirty)
    {
        if (!guid.valid() || !dirty.valid) return;
        auto [found, inserted] = terrain_preview_dirty.try_emplace(guid, dirty);
        if (inserted) return;
        auto& pending = found->second;
        pending.min_x = std::min(pending.min_x, dirty.min_x);
        pending.min_z = std::min(pending.min_z, dirty.min_z);
        pending.max_x = std::max(pending.max_x, dirty.max_x);
        pending.max_z = std::max(pending.max_z, dirty.max_z);
        pending.valid = true;
        pending.heights_changed = pending.heights_changed || dirty.heights_changed;
        pending.weights_changed = pending.weights_changed || dirty.weights_changed;
    }

    void pump_terrain_previews()
    {
        if (!preview_stopped || terrain_rebuilds_suspended || terrain_preview_dirty.empty()) return;
        auto pending = std::move(terrain_preview_dirty);
        terrain_preview_dirty.clear();
        for (const auto& [guid, dirty] : pending)
        {
            const auto entity = find_entity_by_guid(scene, guid);
            if (!scene.scene.alive(entity)) continue;
            if (!synchronize_terrain_render_resource(scene, *renderer, entity, &dirty))
                diagnostics::warn("editor.terrain", "Terrain compatibility preview update failed");
        }
    }

'''
text = text.replace(pump_anchor, preview_methods + pump_anchor, 1)
member_anchor = "    std::unordered_map<ecs::entity_guid, terrain_rebuild_session, ecs::entity_guid_hash> terrain_rebuilds;\n"
if member_anchor not in text:
    raise SystemExit("terrain rebuild member anchor changed")
text = text.replace(member_anchor, member_anchor +
                    "    std::unordered_map<ecs::entity_guid, scene::terrain_dirty_region, ecs::entity_guid_hash> terrain_preview_dirty;\n", 1)

invalidate_anchor = r'''        if (invalidate_all && found != terrain_rebuilds.end())
        {
            terrain_rebuilds.erase(found);
            found = terrain_rebuilds.end();
        }
'''
invalidate_replacement = r'''        if (invalidate_all && found != terrain_rebuilds.end())
        {
            terrain_preview_dirty.erase(guid);
            terrain_rebuilds.erase(found);
            found = terrain_rebuilds.end();
        }
'''
if invalidate_anchor not in text:
    raise SystemExit("terrain rebuild invalidation anchor changed")
text = text.replace(invalidate_anchor, invalidate_replacement, 1)

stroke_marker = text.index("else if constexpr (std::is_same_v<command_type, host_terrain_stroke_command>)")
begin_marker = text.index("                if (payload.phase == host_edit_phase::begin)\n", stroke_marker)
asset_begin = text.index("                    if (asset_backed_sculpt)\n                    {\n", begin_marker)
asset_end = text.index("                    state_->terrain_flatten_height_captured = true;\n", asset_begin)
new_asset_begin = r'''                    if (asset_backed_sculpt)
                    {
                        if (!state_->asset_registry) return fail("Terrain asset registry is unavailable", entity);
                        const auto entity_guid = entity_guid_of(state_->scene, entity);
                        const auto rebuild = state_->terrain_rebuilds.find(entity_guid);
                        if (rebuild == state_->terrain_rebuilds.end())
                            return fail("Terrain asset is still preparing for sculpting", entity);
                        auto authored = rebuild->second.asset();
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

                        auto reference = terrain->asset;
                        reference.expected_type = assets::asset_types::terrain;
                        if (!reference.guid.valid() && !reference.path_hint.empty())
                            reference =
                                state_->asset_registry->resolve(reference.path_hint, assets::asset_types::terrain);
                        if (!reference.guid.valid())
                            return fail("Terrain asset reference could not be resolved", entity);
                        const auto asset_snapshot = state_->asset_registry->find(reference.guid);
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
                        state_->terrain_sculpt_stroke =
                            terrain_sculpt_stroke_session{.entity = entity_guid,
                                                          .asset_guid = reference.guid,
                                                          .modifier = target,
                                                          .source_path = resolved->path,
                                                          .asset = std::move(authored)};
                    }
'''
text = text[:asset_begin] + new_asset_begin + text[asset_end:]

sync_preview = r'''                state_->terrain_stroke_previous_position = hit.position;
                if (dirty.valid &&
                    !synchronize_terrain_render_resource(state_->scene, *state_->renderer, entity, &dirty))
                    return fail("Terrain runtime chunks could not be updated", entity);
'''
queued_preview = r'''                state_->terrain_stroke_previous_position = hit.position;
                if (dirty.valid) state_->queue_terrain_preview(entity_guid_of(state_->scene, entity), dirty);
'''
if sync_preview not in text:
    raise SystemExit("terrain synchronous preview anchor changed")
text = text.replace(sync_preview, queued_preview, 1)

request_anchor = "    state_->pump_terrain_rebuilds();\n"
if request_anchor not in text:
    raise SystemExit("request viewport terrain pump anchor changed")
text = text.replace(request_anchor, "    state_->pump_terrain_previews();\n" + request_anchor, 1)

text = text.replace("state_->terrain_rebuilds.clear();\n", "state_->terrain_rebuilds.clear();\n    state_->terrain_preview_dirty.clear();\n")
text = text.replace("state_->terrain_rebuilds.erase(guid);\n", "state_->terrain_rebuilds.erase(guid);\n                    state_->terrain_preview_dirty.erase(guid);\n")
erase_entity = "state_->terrain_rebuilds.erase(entity_guid_of(state_->scene, entity));\n"
text = text.replace(erase_entity, erase_entity +
                    "                state_->terrain_preview_dirty.erase(entity_guid_of(state_->scene, entity));\n")
host.write_text(text)

impl = Path("editor/native/src/arc_host_impl.inc")
impl_text = impl.read_text()
impl_text = impl_text.replace("state_->terrain_rebuilds.clear();\n",
                              "state_->terrain_rebuilds.clear();\n                state_->terrain_preview_dirty.clear();\n")
impl.write_text(impl_text)

# Regression: the compatibility preview only replaces the dirty render region.
rebuild_tests = Path("editor/native/tests/terrain_rebuild_tests.cpp")
tests = rebuild_tests.read_text()
old = "    const auto distant = cache.find(guid)->regions.back().geometry;\n    const auto generation = cache.find(guid)->generation;\n"
if old not in tests:
    raise SystemExit("terrain rebuild test setup anchor changed")
tests = tests.replace(old,
                      "    const auto affected = cache.find(guid)->regions.front().geometry;\n"
                      "    const auto distant = cache.find(guid)->regions.back().geometry;\n"
                      "    const auto generation = cache.find(guid)->generation;\n", 1)
old = ("    REQUIRE(cache.synchronize(guid, terrain, renderer, &dirty));\n"
       "    CHECK_FALSE(cache.find(guid)->asset_owned);\n"
       "    CHECK(renderer.mesh_alive(distant.conventional));\n")
if old not in tests:
    raise SystemExit("terrain rebuild test preview anchor changed")
tests = tests.replace(old,
                      "    REQUIRE(cache.synchronize(guid, terrain, renderer, &dirty));\n"
                      "    CHECK_FALSE(cache.find(guid)->asset_owned);\n"
                      "    REQUIRE(cache.find(guid)->regions.size() == 16u);\n"
                      "    CHECK(cache.find(guid)->regions.front().geometry != affected);\n"
                      "    CHECK(cache.find(guid)->regions.back().geometry == distant);\n"
                      "    CHECK(renderer.mesh_alive(distant.conventional));\n", 1)
rebuild_tests.write_text(tests)

raycast_test = Path("engine/scene/tests/terrain_interaction_performance_tests.cpp")
raycast_test.write_text(r'''#include <arc/scene/terrain.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("M3.4 terrain raycast traverses the regular heightfield grid")
{
    using namespace arc;
    scene::terrain_component terrain;
    terrain.size = 1024.0f;
    terrain.subdivisions = 1024u;
    const auto resolution = static_cast<std::size_t>(terrain.subdivisions) + 1u;
    const auto count = resolution * resolution;
    terrain.heights.assign(count, 0.0f);
    terrain.layer_weights.assign(count, {255u, 0u, 0u, 0u});

    const auto vertical = scene::raycast_terrain(terrain, {0.0f, 100.0f, 0.0f}, {0.0f, -1.0f, 0.0f});
    REQUIRE(vertical.hit);
    CHECK(vertical.distance == Catch::Approx(100.0f));
    CHECK(vertical.position[1] == Catch::Approx(0.0f));

    const auto direction = math::normalize(math::vector3f{1.0f, -0.01f, 0.0f});
    const auto grazing = scene::raycast_terrain(terrain, {-600.0f, 10.0f, 0.0f}, direction);
    REQUIRE(grazing.hit);
    CHECK(grazing.position[0] == Catch::Approx(400.0f).margin(0.1f));
    CHECK(grazing.position[1] == Catch::Approx(0.0f).margin(0.01f));
}
''')

docs = Path("docs/m3-4-region-rebuilds.md")
docs.write_text(docs.read_text() +
                "\nInteractive sculpting now uses regular-grid ray traversal, never waits for an asset load on pointer down, "
                "coalesces compatibility preview dirtiness to one viewport publication, and recompiles only render regions "
                "overlapping the dirty sample rectangle. The M3.4 asset-owned rebuild remains the authoritative post-stroke result.\n")

host_text = host.read_text()
stroke = host_text[host_text.index("host_terrain_stroke_command"):host_text.index("host_terrain_hover_command")]
if "pending.get()" in stroke:
    raise SystemExit("terrain stroke still blocks on a pending asset load")
