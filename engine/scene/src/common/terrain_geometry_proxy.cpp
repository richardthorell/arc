#include <arc/scene/terrain.h>

#include <arc/render/renderer.h>
#include <arc/scene/terrain_render_attributes.h>
#include <arc/scene/terrain_render_geometry.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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

geometric::box3f terrain_local_bounds(const terrain_world_bounds& bounds) noexcept
{
    return geometric::box3f{
        geometric::point3f{static_cast<float>(bounds.min_x), static_cast<float>(bounds.min_y),
                           static_cast<float>(bounds.min_z)},
        geometric::point3f{static_cast<float>(bounds.max_x), static_cast<float>(bounds.max_y),
                           static_cast<float>(bounds.max_z)}};
}

bool geometry_alive(const terrain_render_proxy& proxy, const render::renderer& renderer)
{
    return proxy.geometry.valid() && renderer.mesh_alive(proxy.geometry.conventional);
}

bool attributes_alive(const terrain_render_proxy& proxy, const render::renderer& renderer)
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

void destroy_proxy_geometry(terrain_render_proxy& proxy, render::renderer& renderer)
{
    if (proxy.geometry.conventional.valid() || proxy.geometry.virtualized.valid())
        (void)renderer.destroy_geometry_resource(proxy.geometry);
    proxy.geometry = render::geometry_resource_handle{};

    if (renderer.texture_alive(proxy.surface_attribute_texture))
        renderer.destroy_texture(proxy.surface_attribute_texture);
    proxy.surface_attribute_texture = {};
}

bool update_attribute_texture(terrain_render_proxy& proxy, const terrain_render_attributes& attributes,
                              render::renderer& renderer)
{
    if (attributes_alive(proxy, renderer) &&
        renderer.update_texture(proxy.surface_attribute_texture, make_attribute_texture_data(attributes)))
        return true;

    auto replacement = create_attribute_texture(attributes, renderer);
    if (!replacement.valid()) return false;

    if (renderer.texture_alive(proxy.surface_attribute_texture))
        renderer.destroy_texture(proxy.surface_attribute_texture);
    proxy.surface_attribute_texture = replacement;
    return true;
}

} // namespace

bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_surface_ir& surface,
                                             const terrain_component& terrain, render::renderer& renderer,
                                             const terrain_dirty_region* dirty_region)
{
    if (!guid.valid() || !validate_terrain_surface_ir(surface)) return false;

    auto& proxy = proxies_[guid];
    const bool has_geometry = geometry_alive(proxy, renderer);
    const bool has_attributes = attributes_alive(proxy, renderer);
    const bool same_revision = proxy.synchronized_revision == surface.source_revision;

    if (has_geometry && has_attributes && same_revision)
    {
        proxy.local_bounds = terrain_local_bounds(surface.local_bounds);
        proxy.material = terrain.material;
        return true;
    }

    const bool weights_only = dirty_region != nullptr && dirty_region->valid && dirty_region->weights_changed &&
                              !dirty_region->heights_changed && has_geometry;
    if (weights_only || (has_geometry && same_revision && !has_attributes))
    {
        auto attributes = build_terrain_render_attributes(surface);
        if (!attributes || !update_attribute_texture(proxy, *attributes, renderer)) return false;

        proxy.local_bounds = terrain_local_bounds(surface.local_bounds);
        proxy.synchronized_revision = surface.source_revision;
        proxy.material = terrain.material;
        return true;
    }

    auto artifact = build_terrain_render_geometry(surface);
    auto attributes = build_terrain_render_attributes(surface);
    if (!artifact || !attributes) return false;

    auto replacement =
        renderer.create_geometry_resource(std::move(*artifact), terrain_geometry_generation(surface.source_revision));
    if (!replacement.valid() || !renderer.mesh_alive(replacement.conventional))
    {
        (void)renderer.destroy_geometry_resource(replacement);
        return false;
    }

    auto replacement_attributes = create_attribute_texture(*attributes, renderer);
    if (!replacement_attributes.valid())
    {
        (void)renderer.destroy_geometry_resource(replacement);
        return false;
    }

    destroy_proxy_geometry(proxy, renderer);
    proxy.geometry = replacement;
    proxy.surface_attribute_texture = replacement_attributes;
    proxy.local_bounds = terrain_local_bounds(surface.local_bounds);
    proxy.synchronized_revision = surface.source_revision;
    proxy.material = terrain.material;
    return true;
}

bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_component& terrain,
                                             render::renderer& renderer, const terrain_dirty_region* dirty_region)
{
    const auto surface = make_legacy_terrain_surface_ir(terrain);
    return surface && synchronize(guid, *surface, terrain, renderer, dirty_region);
}

bool terrain_render_proxy_cache::erase(ecs::entity_guid guid, render::renderer& renderer)
{
    const auto found = proxies_.find(guid);
    if (found == proxies_.end()) return false;
    destroy_proxy_geometry(found->second, renderer);
    proxies_.erase(found);
    return true;
}

void terrain_render_proxy_cache::release_missing(std::span<const ecs::entity_guid> active, render::renderer& renderer)
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

void terrain_render_proxy_cache::clear(render::renderer& renderer)
{
    for (auto& [guid, proxy] : proxies_)
    {
        (void)guid;
        destroy_proxy_geometry(proxy, renderer);
    }
    proxies_.clear();
}

} // namespace arc::scene
