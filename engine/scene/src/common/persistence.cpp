#include <arc/scene/persistence.h>

#include <arc/ecs/partition.h>
#include <arc/ecs/prefab.h>
#include <arc/scene/components.h>
#include <arc/scene/environment.h>
#include <arc/scene/terrain.h>

#include <array>

namespace arc::scene
{
namespace
{

template <class Component>
bool register_component(persistence::component_persistence_registry& registry,
                        std::initializer_list<std::string> names = {})
{
    return registry.register_component({&ecs::component_metadata<Component>(), std::vector<std::string>(names)});
}

} // namespace

bool register_persistence_components(persistence::component_persistence_registry& registry)
{
    bool result = true;
    result &= register_component<name_component>(registry, {"Name"});
    result &= register_component<transform_component>(registry, {"Transform"});
    result &= register_component<tag_component>(registry, {"Tag"});
    result &= register_component<active_component>(registry, {"Active"});
    result &= register_component<selection_component>(registry, {"Selection"});
    result &= register_component<bounds_component>(registry, {"Bounds"});
    result &= register_component<camera_component>(registry, {"Camera"});
    result &= register_component<mesh_renderer_component>(registry, {"MeshRenderer"});
    result &= register_component<virtual_mesh_renderer_component>(registry, {"VirtualMeshRenderer"});
    result &= register_component<skinned_mesh_renderer_component>(registry, {"SkinnedMeshRenderer"});
    result &= register_component<lod_component>(registry, {"LOD"});
    result &= register_component<instance_group_component>(registry, {"InstanceGroup"});
    result &= register_component<render_layer_component>(registry, {"RenderLayer"});
    result &= register_component<mobility_component>(registry, {"Mobility"});
    result &= register_component<directional_light_component>(registry, {"DirectionalLight"});
    result &= register_component<point_light_component>(registry, {"PointLight"});
    result &= register_component<spot_light_component>(registry, {"SpotLight"});
    result &= register_component<area_light_component>(registry, {"AreaLight"});
    result &= register_component<reflection_probe_component>(registry, {"ReflectionProbe"});
    result &= register_component<irradiance_probe_component>(registry, {"IrradianceProbe"});
    result &= register_component<baked_lighting_component>(registry, {"BakedLighting"});
    result &= register_component<indirect_lighting_component>(registry, {"IndirectLighting"});
    result &= register_component<world_environment_component>(registry, {"WorldEnvironment"});
    result &= register_component<sky_atmosphere_component>(registry, {"SkyAtmosphere"});
    result &= register_component<celestial_sky_component>(registry, {"CelestialSky"});
    result &= register_component<cloud_layers_component>(registry, {"CloudLayers"});
    result &= register_component<environment_lighting_component>(registry, {"EnvironmentLighting"});
    result &= register_component<height_fog_component>(registry, {"HeightFog"});
    result &= register_component<terrain_component>(registry, {"Terrain"});
    result &= register_component<water_component>(registry, {"Water"});
    result &= register_component<vegetation_component>(registry, {"Vegetation"});
    result &= register_component<decal_component>(registry, {"Decal"});
    result &= register_component<ecs::prefab_instance_component>(registry, {"PrefabInstance"});
    result &= register_component<ecs::world_region_component>(registry, {"WorldRegion"});
    return result && registry.freeze();
}

persistence::persistence_status register_persistence_migrations(persistence::schema_migration_registry& registry)
{
    const auto document_upgrade = [](persistence::archive_document&)
    { return persistence::persistence_status::success(); };
    const auto component_upgrade = [](persistence::archive_component_record&)
    { return persistence::persistence_status::success(); };
    if (!registry.register_document(persistence::document_kind::scene, 1, 2, document_upgrade) ||
        !registry.register_document(persistence::document_kind::scene, 2, 3, document_upgrade) ||
        !registry.register_document(persistence::document_kind::prefab, 1, 2, document_upgrade) ||
        !registry.register_component(ecs::component_metadata<terrain_component>().id, 1, 2, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<camera_component>().id, 1, 2, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<mesh_renderer_component>().id, 1, 2, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<mesh_renderer_component>().id, 2, 3, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<mesh_renderer_component>().id, 3, 4, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<vegetation_component>().id, 1, 2, component_upgrade))
    {
        return persistence::persistence_status::failure({.code = persistence::persistence_error_code::migration_invalid,
                                                         .message = "failed to register scene document migrations"});
    }
    const auto register_light = [&](ecs::component_type_id type)
    {
        return registry.register_component(type, 1, 2, component_upgrade) &&
               registry.register_component(type, 2, 3, component_upgrade);
    };
    if (!register_light(ecs::component_metadata<directional_light_component>().id) ||
        !register_light(ecs::component_metadata<point_light_component>().id) ||
        !register_light(ecs::component_metadata<spot_light_component>().id) ||
        !register_light(ecs::component_metadata<area_light_component>().id))
    {
        return persistence::persistence_status::failure({.code = persistence::persistence_error_code::migration_invalid,
                                                         .message = "failed to register scene light migrations"});
    }
    return registry.freeze();
}

} // namespace arc::scene
