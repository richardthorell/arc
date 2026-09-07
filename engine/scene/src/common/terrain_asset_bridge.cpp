#include <arc/scene/terrain_asset_bridge.h>

#include <arc/assets/terrain_types.h>
#include <arc/scene/terrain_asset.h>

namespace arc::scene
{

terrain_asset_binding_result refresh_terrain_asset_binding(terrain_component& terrain, assets::asset_manager& manager)
{
    terrain_asset_binding_result result;
    if (!terrain.asset.guid.valid() && terrain.asset.path_hint.empty())
    {
        result.succeeded = true;
        result.message = "Terrain uses the legacy inline compatibility surface";
        return result;
    }

    auto reference = terrain.asset;
    reference.expected_type = assets::asset_types::terrain;
    if (!reference.guid.valid() && !reference.path_hint.empty())
        reference = manager.resolve(reference.path_hint, assets::asset_types::terrain);
    if (!reference.guid.valid())
    {
        result.message = "Terrain asset reference could not be resolved";
        return result;
    }

    auto pending = manager.load<terrain_asset>({.reference = reference,
                                                .priority = assets::asset_streaming_priority::high,
                                                .residency = assets::asset_residency::cpu,
                                                .allow_fallback = false});
    auto loaded = pending.get();
    if (!loaded)
    {
        result.message = loaded.error.message.empty() ? "Terrain asset could not be loaded" : loaded.error.message;
        return result;
    }

    const auto* authored = loaded.asset.get();
    if (!authored)
    {
        result.message = "Terrain asset payload has the wrong runtime type";
        return result;
    }

    terrain.asset.guid = loaded.asset.requested_guid();
    terrain.asset.expected_type = assets::asset_types::terrain;
    if (const auto snapshot = manager.find(loaded.asset.resolved_guid()))
        terrain.asset.path_hint = assets::normalize_asset_path(snapshot->source_path);
    terrain.asset_generation = loaded.asset.generation();
    terrain.asset_authoring_revision = authored->authoring_revision;

    result.succeeded = true;
    result.bound = true;
    result.generation = terrain.asset_generation;
    result.authoring_revision = terrain.asset_authoring_revision;
    result.message = "Terrain asset loaded; legacy heightfield retained as the compatibility surface";
    return result;
}

} // namespace arc::scene
