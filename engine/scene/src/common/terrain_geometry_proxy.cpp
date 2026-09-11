#include <arc/scene/terrain.h>

#include <arc/render/renderer.h>
#include <arc/scene/terrain_render_attributes.h>
#include <arc/scene/terrain_render_geometry.h>
#include <arc/scene/terrain_render_regions.h>
#include <arc/scene/terrain_region_build.h>

#include <algorithm>
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

} // namespace

bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_surface_ir& surface,
                                             const terrain_component& terrain, render::renderer& renderer,
                                             const terrain_dirty_region* dirty_region)
{
    (void)dirty_region;
    if (!guid.valid() || !validate_terrain_surface_ir(surface)) return false;

    auto& proxy = proxies_[guid];
    if (proxy.synchronized_revision == surface.source_revision && proxy_resources_alive(proxy, renderer))
    {
        proxy.material = terrain.material;
        return true;
    }

    auto source_regions = build_terrain_render_regions(surface);
    if (source_regions.empty()) return false;

    std::vector<terrain_render_region_proxy> staged;
    staged.reserve(source_regions.size());
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

        if (previous && previous->attribute_fingerprint == source_region.attribute_fingerprint &&
            attributes_alive(*previous, renderer))
        {
            next.surface_attribute_texture = previous->surface_attribute_texture;
        }
        else if (previous && attributes_alive(*previous, renderer) &&
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
        staged.push_back(next);
    }

    destroy_unreused_old_resources(proxy, staged, renderer);
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

bool terrain_render_proxy_cache::publish(ecs::entity_guid guid, terrain_region_build_batch& batch,
                                         const terrain_component& terrain, render::renderer& renderer)
{
    if (!guid.valid() || !batch.succeeded || batch.stale || batch.regions.empty()) return false;
    auto& visible = proxies_[guid];
    const auto retained = retained_.find(guid);
    terrain_render_proxy empty;
    auto& previous = retained != retained_.end() ? retained->second : visible.asset_owned ? visible : empty;
    auto staged = previous.regions;
    for (auto& build : batch.regions)
    {
        if (!build.geometry || !build.attributes)
        {
            cleanup_staged_resources(previous, staged, renderer);
            return false;
        }
        terrain_render_region_proxy next;
        next.id = build.evaluation.region;
        next.local_bounds = terrain_local_bounds(build.evaluation.surface.local_bounds);
        next.geometry_fingerprint = build.evaluation.content_fingerprint;
        next.attribute_fingerprint = build.evaluation.content_fingerprint;
        // Uploads are immutable. Never update a texture referenced by the visible generation in place.
        next.geometry = renderer.create_geometry_resource(
            *build.geometry, terrain_geometry_generation(build.evaluation.build_snapshot.target_dirty_revision));
        next.surface_attribute_texture = create_attribute_texture(*build.attributes, renderer);
        if (!geometry_alive(next, renderer) || !attributes_alive(next, renderer))
        {
            destroy_region(next, renderer);
            cleanup_staged_resources(previous, staged, renderer);
            return false;
        }
        const auto old =
            std::find_if(staged.begin(), staged.end(), [&](const auto& region) { return region.id == next.id; });
        if (old == staged.end())
            staged.push_back(next);
        else
            *old = next;
    }
    const auto generation = previous.generation + 1u;
    destroy_unreused_old_resources(previous, staged, renderer);
    if (&previous != &visible) destroy_proxy(visible, renderer);
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
    if (const auto retained = retained_.find(guid); retained != retained_.end())
    {
        destroy_proxy(retained->second, renderer);
        retained_.erase(retained);
    }
    const auto found = proxies_.find(guid);
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
            ++found;
        else
        {
            destroy_proxy(found->second, renderer);
            found = retained_.erase(found);
        }
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
    for (auto& [guid, proxy] : retained_)
    {
        (void)guid;
        destroy_proxy(proxy, renderer);
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
