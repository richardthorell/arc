#include <arc/scene/terrain.h>

#include <arc/render/renderer.h>
#include <arc/scene/terrain_render_geometry.h>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace arc::scene
{
namespace
{

std::uint32_t terrain_geometry_generation(std::uint64_t revision) noexcept
{
    const auto folded = static_cast<std::uint32_t>(revision) ^ static_cast<std::uint32_t>(revision >> 32u);
    return folded == 0u ? 1u : folded;
}

bool geometry_alive(const terrain_render_proxy& proxy, const render::renderer& renderer)
{
    return proxy.geometry.valid() && renderer.mesh_alive(proxy.geometry.conventional);
}

void destroy_proxy_geometry(terrain_render_proxy& proxy, render::renderer& renderer)
{
    if (proxy.geometry.conventional.valid() || proxy.geometry.virtualized.valid())
        (void)renderer.destroy_geometry_resource(proxy.geometry);
    proxy.geometry = render::geometry_resource_handle{};

    if (renderer.terrain_alive(proxy.handle)) (void)renderer.destroy_terrain(proxy.handle);
    proxy.handle = {};
}

} // namespace

bool terrain_render_proxy_cache::synchronize_geometry(ecs::entity_guid guid, const terrain_surface_ir& surface,
                                                      const terrain_component& terrain, render::renderer& renderer,
                                                      const terrain_dirty_region* dirty_region)
{
    (void)dirty_region;
    if (!guid.valid() || !validate_terrain_surface_ir(surface)) return false;

    auto& proxy = proxies_[guid];
    if (geometry_alive(proxy, renderer) && proxy.synchronized_revision == surface.source_revision)
    {
        proxy.local_bounds = surface.local_bounds;
        proxy.material = terrain.material;
        return true;
    }

    auto artifact = build_terrain_render_geometry(surface);
    if (!artifact) return false;

    auto replacement =
        renderer.create_geometry_resource(std::move(*artifact), terrain_geometry_generation(surface.source_revision));
    if (!replacement.valid() || !renderer.mesh_alive(replacement.conventional))
    {
        (void)renderer.destroy_geometry_resource(replacement);
        return false;
    }

    destroy_proxy_geometry(proxy, renderer);
    proxy.geometry = replacement;
    proxy.local_bounds = surface.local_bounds;
    proxy.synchronized_revision = surface.source_revision;
    proxy.material = terrain.material;
    return true;
}

bool terrain_render_proxy_cache::synchronize_geometry(ecs::entity_guid guid, const terrain_component& terrain,
                                                      render::renderer& renderer,
                                                      const terrain_dirty_region* dirty_region)
{
    const auto surface = make_legacy_terrain_surface_ir(terrain);
    return surface && synchronize_geometry(guid, *surface, terrain, renderer, dirty_region);
}

bool terrain_render_proxy_cache::erase_geometry(ecs::entity_guid guid, render::renderer& renderer)
{
    const auto found = proxies_.find(guid);
    if (found == proxies_.end()) return false;
    destroy_proxy_geometry(found->second, renderer);
    proxies_.erase(found);
    return true;
}

void terrain_render_proxy_cache::release_missing_geometry(std::span<const ecs::entity_guid> active,
                                                          render::renderer& renderer)
{
    for (auto found = proxies_.begin(); found != proxies_.end();)
    {
        if (std::find(active.begin(), active.end(), found->first) != active.end())
        {
            ++found;
            continue;
        }
        destroy_proxy_geometry(found->second, renderer);
        found = proxies_.erase(found);
    }
}

void terrain_render_proxy_cache::clear_geometry(render::renderer& renderer)
{
    for (auto& [guid, proxy] : proxies_)
    {
        (void)guid;
        destroy_proxy_geometry(proxy, renderer);
    }
    proxies_.clear();
}

} // namespace arc::scene
