#include <arc/scene/terrain_streaming.h>

#include <arc/render/renderer.h>

#include <algorithm>
#include <utility>

namespace arc::scene
{
namespace
{

const terrain_render_region_proxy* find_proxy_region(const terrain_render_proxy& proxy, terrain_region_id id) noexcept
{
    const auto found = std::find_if(proxy.regions.begin(), proxy.regions.end(),
                                    [id](const auto& region) { return region.id == id; });
    return found == proxy.regions.end() ? nullptr : &*found;
}

std::vector<render::virtual_geometry_artifact_page_range>
page_ranges(const terrain_artifact_reference& artifact)
{
    std::vector<render::virtual_geometry_artifact_page_range> result;
    result.reserve(artifact.pages.size());
    for (const auto& page : artifact.pages)
        result.push_back({.offset = page.offset,
                          .stored_size = page.stored_size,
                          .decoded_size = page.decoded_size,
                          .content_hash = page.content_hash,
                          .root = page.root});
    return result;
}

struct staged_binding
{
    render::virtual_mesh_handle resource{};
    std::uint32_t resource_generation{};
    assets::cooked_artifact_location location;
    std::uint64_t artifact_size{};
    std::vector<render::virtual_geometry_artifact_page_range> pages;
};

} // namespace

terrain_streaming_bind_result terrain_virtual_geometry_streaming_binding::synchronize(
    const terrain_cooked_manifest& manifest, const assets::package_artifact_reader& package,
    const terrain_render_proxy& proxy, const render::renderer& renderer,
    render::filesystem_virtual_geometry_artifact_source& source)
{
    terrain_streaming_bind_result result;
    if (!validate_terrain_cooked_manifest(manifest)) return result;

    std::vector<staged_binding> staged;
    staged.reserve(manifest.regions.size());
    for (const auto& manifest_region : manifest.regions)
    {
        const auto* region = find_proxy_region(proxy, manifest_region.region);
        const auto* artifact = find_terrain_artifact(manifest, manifest_region.region,
                                                     terrain_artifact_kind::render_geometry);
        if (!region || !artifact || !region->geometry.virtualized.valid() ||
            artifact->generation != region->geometry.asset_generation || artifact->pages.empty())
        {
            ++result.skipped_regions;
            continue;
        }

        const auto resource_generation = renderer.virtual_mesh_content_generation(region->geometry.virtualized);
        if (resource_generation == 0u)
        {
            ++result.skipped_regions;
            continue;
        }

        const assets::cooked_artifact_address address{
            manifest.terrain, assets::artifact_schemas::virtual_geometry, artifact->storage_key};
        const auto location = package.locate(address);
        if (!location || location->size < artifact->payload_size)
        {
            ++result.skipped_regions;
            continue;
        }

        staged.push_back({.resource = region->geometry.virtualized,
                          .resource_generation = resource_generation,
                          .location = *location,
                          .artifact_size = artifact->payload_size,
                          .pages = page_ranges(*artifact)});
    }

    if (staged.empty() && !manifest.regions.empty()) return result;

    clear(source);
    resources_.reserve(staged.size());
    for (auto& binding : staged)
    {
        source.register_package_range(binding.resource, binding.resource_generation, std::move(binding.location.path),
                                      binding.location.offset, binding.artifact_size, std::move(binding.pages));
        resources_.push_back(binding.resource);
        ++result.bound_regions;
    }
    result.succeeded = result.bound_regions != 0u || manifest.regions.empty();
    return result;
}

void terrain_virtual_geometry_streaming_binding::clear(
    render::filesystem_virtual_geometry_artifact_source& source) noexcept
{
    for (const auto resource : resources_)
        source.unregister(resource);
    resources_.clear();
}

} // namespace arc::scene
