#include <arc/scene/terrain.h>

#include <arc/render/renderer.h>
#include <arc/scene/terrain_render_attributes.h>
#include <arc/scene/terrain_render_geometry.h>
#include <arc/scene/terrain_render_regions.h>
#include <arc/scene/terrain_region_build.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace arc::scene
{
namespace
{

std::uint32_t terrain_geometry_generation(std::uint64_t fingerprint) noexcept
{
    const auto folded = static_cast<std::uint32_t>(fingerprint) ^ static_cast<std::uint32_t>(fingerprint >> 32u);
    return folded == 0u ? 1u : folded;
}

geometric::box3f terrain_local_bounds(const terrain_world_bounds& bounds) noexcept
{
    return geometric::box3f{geometric::point3f{static_cast<float>(bounds.min_x), static_cast<float>(bounds.min_y),
                                               static_cast<float>(bounds.min_z)},
                            geometric::point3f{static_cast<float>(bounds.max_x), static_cast<float>(bounds.max_y),
                                               static_cast<float>(bounds.max_z)}};
}

bool geometry_alive(const terrain_render_region_proxy& proxy, const render::renderer& renderer)
{
    return proxy.geometry.valid() && renderer.mesh_alive(proxy.geometry.conventional);
}

bool attributes_alive(const terrain_render_region_proxy& proxy, const render::renderer& renderer)
{
    return proxy.surface_attribute_texture.valid() && renderer.texture_alive(proxy.surface_attribute_texture);
}

render::texture_data make_attribute_texture_data(const terrain_render_attributes& attributes)
{
    render::texture_data data;
    data.name = "terrain-material-weights";
    data.width = attributes.width;
    data.height = attributes.height;
    data.format = render::texture_format::rgba8_unorm;
    data.color_space = render::texture_color_space::linear;
    data.mip_levels = 1u;
    data.pixels.resize(attributes.material_weights.size() * sizeof(attributes.material_weights.front()));
    if (!data.pixels.empty()) std::memcpy(data.pixels.data(), attributes.material_weights.data(), data.pixels.size());
    return data;
}

render::texture_handle create_attribute_texture(const terrain_render_attributes& attributes, render::renderer& renderer)
{
    auto texture = renderer.create_texture(make_attribute_texture_data(attributes));
    if (!texture.valid() || !renderer.texture_alive(texture))
    {
        if (texture.valid()) renderer.destroy_texture(texture);
        return {};
    }
    return texture;
}

void destroy_region(terrain_render_region_proxy& proxy, render::renderer& renderer)
{
    if (proxy.geometry.conventional.valid() || proxy.geometry.virtualized.valid())
        (void)renderer.destroy_geometry_resource(proxy.geometry);
    proxy.geometry = render::geometry_resource_handle{};
    if (renderer.texture_alive(proxy.surface_attribute_texture))
        renderer.destroy_texture(proxy.surface_attribute_texture);
    proxy.surface_attribute_texture = {};
}

void destroy_proxy(terrain_render_proxy& proxy, render::renderer& renderer)
{
    for (auto& region : proxy.regions)
        destroy_region(region, renderer);
    proxy.regions.clear();
}

const terrain_render_region_proxy* find_region(const terrain_render_proxy& proxy, terrain_region_id id) noexcept
{
    const auto found = std::find_if(proxy.regions.begin(), proxy.regions.end(),
                                    [id](const terrain_render_region_proxy& value) { return value.id == id; });
    return found == proxy.regions.end() ? nullptr : &*found;
}

bool proxy_resources_alive(const terrain_render_proxy& proxy, const render::renderer& renderer)
{
    return !proxy.regions.empty() &&
           std::all_of(proxy.regions.begin(), proxy.regions.end(), [&](const terrain_render_region_proxy& region)
                       { return geometry_alive(region, renderer) && attributes_alive(region, renderer); });
}

bool geometry_reused(const std::vector<terrain_render_region_proxy>& regions, render::geometry_resource_handle geometry)
{
    return std::any_of(regions.begin(), regions.end(),
                       [&](const terrain_render_region_proxy& candidate) { return candidate.geometry == geometry; });
}

bool texture_reused(const std::vector<terrain_render_region_proxy>& regions, render::texture_handle texture)
{
    return std::any_of(regions.begin(), regions.end(), [&](const terrain_render_region_proxy& candidate)
                       { return candidate.surface_attribute_texture == texture; });
}

void destroy_unreused_old_resources(terrain_render_proxy& previous,
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

void cleanup_staged_resources(const terrain_render_proxy& previous, std::vector<terrain_render_region_proxy>& staged,
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

bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_surface_ir& surface,
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
    std::vector<terrain_render_region_proxy> staged =
        partial ? proxy.regions : std::vector<terrain_render_region_proxy>{};
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

        const bool attributes_protected = previous && protected_proxy &&
                                          texture_reused(protected_proxy->regions, previous->surface_attribute_texture);
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

bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_component& terrain,
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

bool terrain_render_proxy_cache::preview_attributes(ecs::entity_guid guid, const terrain_component& terrain,
                                                    render::renderer& renderer,
                                                    const terrain_dirty_region& dirty_region)
{
    auto* proxy = find(guid);
    if (!proxy || !proxy->asset_owned || !dirty_region.valid || !dirty_region.weights_changed ||
        !terrain_heightfield_valid(terrain) || !proxy_resources_alive(*proxy, renderer))
        return false;

    const auto resolution = terrain.subdivisions + 1u;
    const auto spacing = terrain.size / static_cast<float>(terrain.subdivisions);
    const auto half = terrain.size * 0.5f;
    const auto dirty_min_x = -half + static_cast<float>(dirty_region.min_x) * spacing;
    const auto dirty_max_x = -half + static_cast<float>(dirty_region.max_x) * spacing;
    const auto dirty_min_z = -half + static_cast<float>(dirty_region.min_z) * spacing;
    const auto dirty_max_z = -half + static_cast<float>(dirty_region.max_z) * spacing;
    const auto sample_at = [&](float coordinate)
    {
        return static_cast<std::uint32_t>(std::clamp<std::int64_t>(std::lround((coordinate + half) / spacing), 0,
                                                                   static_cast<std::int64_t>(terrain.subdivisions)));
    };

    struct staged_attribute_page
    {
        std::size_t region{};
        render::texture_handle texture{};
    };
    std::vector<staged_attribute_page> staged;
    for (std::size_t region_index = 0; region_index < proxy->regions.size(); ++region_index)
    {
        const auto& region = proxy->regions[region_index];
        if (region.local_bounds.max[0] < dirty_min_x || region.local_bounds.min[0] > dirty_max_x ||
            region.local_bounds.max[2] < dirty_min_z || region.local_bounds.min[2] > dirty_max_z)
            continue;

        const auto x0 = sample_at(region.local_bounds.min[0]);
        const auto x1 = sample_at(region.local_bounds.max[0]);
        const auto z0 = sample_at(region.local_bounds.min[2]);
        const auto z1 = sample_at(region.local_bounds.max[2]);
        terrain_render_attributes attributes;
        attributes.width = x1 - x0 + 1u;
        attributes.height = z1 - z0 + 1u;
        attributes.default_layer_only = false;
        attributes.material_weights.clear();
        attributes.material_weights.reserve(static_cast<std::size_t>(attributes.width) * attributes.height);
        for (std::uint32_t z = z0; z <= z1; ++z)
            for (std::uint32_t x = x0; x <= x1; ++x)
                attributes.material_weights.push_back(
                    terrain.layer_weights[static_cast<std::size_t>(z) * resolution + x]);

        const auto texture = create_attribute_texture(attributes, renderer);
        if (!texture.valid())
        {
            for (const auto& page : staged)
                if (renderer.texture_alive(page.texture)) renderer.destroy_texture(page.texture);
            return false;
        }
        staged.push_back({region_index, texture});
    }
    if (staged.empty()) return false;

    for (const auto& page : staged)
    {
        auto& region = proxy->regions[page.region];
        const auto previous = region.surface_attribute_texture;
        region.surface_attribute_texture = page.texture;
        region.attribute_fingerprint = 0u;
        if (renderer.texture_alive(previous)) renderer.destroy_texture(previous);
    }
    proxy->synchronized_revision = terrain.content_revision;
    return true;
}

bool terrain_render_proxy_cache::publish(ecs::entity_guid guid, terrain_region_build_batch& batch,
                                         const terrain_component& terrain, render::renderer& renderer)
{
    if (!guid.valid() || !batch.succeeded || batch.stale || batch.regions.empty()) return false;
    auto& visible = proxies_[guid];
    const auto retained = retained_.find(guid);
    terrain_render_proxy empty;
    auto& previous = retained != retained_.end() ? retained->second : visible.asset_owned ? visible : empty;
    auto staged = batch.replace_all_regions ? std::vector<terrain_render_region_proxy>{} : previous.regions;
    for (auto& build : batch.regions)
    {
        const auto* old = find_region(previous, build.evaluation.region);
        const auto geometry_domains = terrain_domain::geometry | terrain_domain::topology;
        const bool rebuild_geometry = (build.domains & geometry_domains) != terrain_domain::none;
        const bool rebuild_attributes =
            rebuild_geometry || (build.domains & terrain_domain::attributes) != terrain_domain::none;
        if ((rebuild_geometry && !build.geometry) || (rebuild_attributes && !build.attributes) ||
            (!rebuild_geometry && (!old || !geometry_alive(*old, renderer))) ||
            (!rebuild_attributes && (!old || !attributes_alive(*old, renderer))))
        {
            cleanup_staged_resources(previous, staged, renderer);
            return false;
        }
        terrain_render_region_proxy next;
        next.id = build.evaluation.region;
        next.local_bounds = terrain_local_bounds(build.evaluation.surface.local_bounds);
        next.geometry_fingerprint = rebuild_geometry ? build.evaluation.content_fingerprint : old->geometry_fingerprint;
        next.attribute_fingerprint =
            rebuild_attributes ? build.evaluation.content_fingerprint : old->attribute_fingerprint;
        if (rebuild_geometry)
            next.geometry = renderer.create_geometry_resource(
                *build.geometry, terrain_geometry_generation(build.evaluation.build_snapshot.target_dirty_revision));
        else
            next.geometry = old->geometry;
        // Uploads are immutable. Never update a texture referenced by the visible generation in place.
        next.surface_attribute_texture =
            rebuild_attributes ? create_attribute_texture(*build.attributes, renderer) : old->surface_attribute_texture;
        if (!geometry_alive(next, renderer) || !attributes_alive(next, renderer))
        {
            if (rebuild_geometry && (next.geometry.conventional.valid() || next.geometry.virtualized.valid()))
                (void)renderer.destroy_geometry_resource(next.geometry);
            if (rebuild_attributes && renderer.texture_alive(next.surface_attribute_texture))
                renderer.destroy_texture(next.surface_attribute_texture);
            cleanup_staged_resources(previous, staged, renderer);
            return false;
        }
        const auto staged_old =
            std::find_if(staged.begin(), staged.end(), [&](const auto& region) { return region.id == next.id; });
        if (staged_old == staged.end())
            staged.push_back(next);
        else
            *staged_old = next;
    }
    const auto generation = previous.generation + 1u;
    if (&previous != &visible)
    {
        cleanup_staged_resources(previous, visible.regions, renderer);
        visible.regions.clear();
    }
    destroy_unreused_old_resources(previous, staged, renderer);
    visible.regions = std::move(staged);
    visible.synchronized_revision = terrain.content_revision;
    visible.material = terrain.material;
    visible.asset_owned = true;
    visible.generation = generation;
    if (retained != retained_.end()) retained_.erase(retained);
    return true;
}

bool terrain_render_proxy_cache::erase(ecs::entity_guid guid, render::renderer& renderer)
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

void terrain_render_proxy_cache::release_missing(std::span<const ecs::entity_guid> active, render::renderer& renderer)
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

void terrain_render_proxy_cache::clear(render::renderer& renderer)
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

} // namespace arc::scene
