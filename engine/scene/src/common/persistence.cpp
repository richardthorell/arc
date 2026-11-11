#include <arc/scene/persistence.h>

#include <arc/ecs/partition.h>
#include <arc/ecs/prefab.h>
#include <arc/scene/components.h>
#include <arc/scene/environment.h>
#include <arc/scene/terrain.h>

#include <array>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

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

persistence::archive_value mesh_bool(bool value)
{
    persistence::archive_value result;
    result.kind = persistence::archive_value_kind::boolean;
    result.boolean = value;
    return result;
}

persistence::archive_value mesh_float(float value)
{
    persistence::archive_value result;
    result.kind = persistence::archive_value_kind::floating_point;
    result.floating_point = value;
    return result;
}

persistence::archive_value mesh_int(std::int32_t value)
{
    persistence::archive_value result;
    result.kind = persistence::archive_value_kind::signed_integer;
    result.signed_integer = value;
    return result;
}

bool read_mesh_bool(const persistence::archive_value& value, bool& result)
{
    if (value.kind != persistence::archive_value_kind::boolean) return false;
    result = value.boolean;
    return true;
}

bool read_mesh_float(const persistence::archive_value& value, float& result)
{
    double number{};
    switch (value.kind)
    {
        case persistence::archive_value_kind::floating_point:
            number = value.floating_point;
            break;
        case persistence::archive_value_kind::signed_integer:
            number = static_cast<double>(value.signed_integer);
            break;
        case persistence::archive_value_kind::unsigned_integer:
            number = static_cast<double>(value.unsigned_integer);
            break;
        default:
            return false;
    }
    if (!std::isfinite(number) || number < -std::numeric_limits<float>::max() ||
        number > std::numeric_limits<float>::max())
        return false;
    result = static_cast<float>(number);
    return true;
}

bool read_mesh_int(const persistence::archive_value& value, std::int32_t& result)
{
    if (value.kind == persistence::archive_value_kind::signed_integer &&
        value.signed_integer >= std::numeric_limits<std::int32_t>::min() &&
        value.signed_integer <= std::numeric_limits<std::int32_t>::max())
    {
        result = static_cast<std::int32_t>(value.signed_integer);
        return true;
    }
    if (value.kind == persistence::archive_value_kind::unsigned_integer &&
        value.unsigned_integer <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    {
        result = static_cast<std::int32_t>(value.unsigned_integer);
        return true;
    }
    return false;
}

template <class Enum> bool read_mesh_enum(const persistence::archive_value& value, Enum& result)
{
    std::int32_t number{};
    if (!read_mesh_int(value, number) || number < 0 || number > 2) return false;
    result = static_cast<Enum>(number);
    return true;
}

void put_mesh_field(persistence::archive_component_record& record, ecs::component_field_id id, std::string_view name,
                    persistence::archive_value value)
{
    record.fields.push_back({id, std::string(name), std::move(value), true});
}

persistence::persistence_status encode_mesh_renderer(const void* component,
                                                     persistence::archive_component_record& record)
{
    const auto& input = *static_cast<const mesh_renderer_component*>(component);
    record.fields.clear();
    // Mesh and material asset GUIDs live in document.dependencies. Their
    // renderer-local handles must never be written to a portable archive.
    put_mesh_field(record, 3u, "visible", mesh_bool(input.visible));
    put_mesh_field(record, 5u, "casts_shadows", mesh_bool(input.casts_shadows));
    put_mesh_field(record, 6u, "receives_shadows", mesh_bool(input.receives_shadows));
    put_mesh_field(record, 7u, "shadow_lod_bias", mesh_float(input.shadow_lod_bias));
    put_mesh_field(record, 8u, "maximum_shadow_distance", mesh_float(input.maximum_shadow_distance));
    put_mesh_field(record, 9u, "representation", mesh_int(static_cast<std::int32_t>(input.representation)));
    put_mesh_field(record, 10u, "affects_indirect_lighting", mesh_bool(input.affects_indirect_lighting));
    put_mesh_field(record, 11u, "surface_card_density_bias", mesh_float(input.surface_card_density_bias));
    put_mesh_field(record, 12u, "distance_field_resolution_bias", mesh_float(input.distance_field_resolution_bias));
    put_mesh_field(record, 13u, "visible_in_hardware_tracing", mesh_bool(input.visible_in_hardware_tracing));
    put_mesh_field(record, 14u, "motion_vectors", mesh_int(static_cast<std::int32_t>(input.motion_vectors)));
    put_mesh_field(record, 15u, "receive_decals", mesh_bool(input.receive_decals));
    put_mesh_field(record, 16u, "occlusion_culling", mesh_bool(input.occlusion_culling));
    put_mesh_field(record, 17u, "bounds_scale", mesh_float(input.bounds_scale));
    put_mesh_field(record, 18u, "minimum_draw_distance", mesh_float(input.minimum_draw_distance));
    put_mesh_field(record, 19u, "maximum_draw_distance", mesh_float(input.maximum_draw_distance));
    put_mesh_field(record, 20u, "forced_lod", mesh_int(input.forced_lod));
    put_mesh_field(record, 21u, "lod_bias", mesh_float(input.lod_bias));
    return persistence::persistence_status::success();
}

persistence::persistence_status decode_mesh_renderer(void* component,
                                                     const persistence::archive_component_record& record)
{
    auto& output = *static_cast<mesh_renderer_component*>(component);
    output.mesh = {};
    output.material = {};
    for (const auto& field : record.fields)
    {
        bool valid = true;
        switch (field.id)
        {
            // Legacy asset slots are resolved from the dependency manifest.
            case 0x0000000000000001ull:
            case 0x0000000000000002ull:
                break;
            case 0x0000000000000003ull:
                valid = read_mesh_bool(field.value, output.visible);
                break;
            case 0x0000000000000005ull:
                valid = read_mesh_bool(field.value, output.casts_shadows);
                break;
            case 0x0000000000000006ull:
                valid = read_mesh_bool(field.value, output.receives_shadows);
                break;
            case 0x0000000000000007ull:
                valid = read_mesh_float(field.value, output.shadow_lod_bias);
                break;
            case 0x0000000000000008ull:
                valid = read_mesh_float(field.value, output.maximum_shadow_distance);
                break;
            case 0x0000000000000009ull:
                valid = read_mesh_enum(field.value, output.representation);
                break;
            case 0x000000000000000aull:
                valid = read_mesh_bool(field.value, output.affects_indirect_lighting);
                break;
            case 0x000000000000000bull:
                valid = read_mesh_float(field.value, output.surface_card_density_bias);
                break;
            case 0x000000000000000cull:
                valid = read_mesh_float(field.value, output.distance_field_resolution_bias);
                break;
            case 0x000000000000000dull:
                valid = read_mesh_bool(field.value, output.visible_in_hardware_tracing);
                break;
            case 0x000000000000000eull:
                valid = read_mesh_enum(field.value, output.motion_vectors);
                break;
            case 0x000000000000000full:
                valid = read_mesh_bool(field.value, output.receive_decals);
                break;
            case 0x0000000000000010ull:
                valid = read_mesh_bool(field.value, output.occlusion_culling);
                break;
            case 0x0000000000000011ull:
                valid = read_mesh_float(field.value, output.bounds_scale);
                break;
            case 0x0000000000000012ull:
                valid = read_mesh_float(field.value, output.minimum_draw_distance);
                break;
            case 0x0000000000000013ull:
                valid = read_mesh_float(field.value, output.maximum_draw_distance);
                break;
            case 0x0000000000000014ull:
                valid = read_mesh_int(field.value, output.forced_lod);
                break;
            case 0x0000000000000015ull:
                valid = read_mesh_float(field.value, output.lod_bias);
                break;
            default:
                valid = false;
                break;
        }
        if (!valid)
        {
            persistence::persistence_error error;
            error.code = persistence::persistence_error_code::decode_failed;
            error.component = record.type;
            error.field = field.id;
            error.message = "Invalid cooked MeshRenderer field: " + field.name;
            return persistence::persistence_status::failure(std::move(error));
        }
    }
    return persistence::persistence_status::success();
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
    persistence::component_persistence_descriptor mesh_codec;
    mesh_codec.component = &ecs::component_metadata<mesh_renderer_component>();
    mesh_codec.legacy_names = {"MeshRenderer"};
    mesh_codec.encode = encode_mesh_renderer;
    mesh_codec.decode = decode_mesh_renderer;
    result &= registry.register_component(std::move(mesh_codec));
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
    result &= register_component<irradiance_probe_volume_component>(registry, {"IrradianceProbeVolume"});
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
    const auto mesh_v4_to_v5 = [](persistence::archive_component_record& component)
    {
        // Editor v4 scene documents use camelCase, while schema v5 uses the
        // reflected snake_case names. Rebind IDs before cooking/binary output.
        constexpr std::pair<std::string_view, std::string_view> aliases[]{
            {"castsShadows", "casts_shadows"},
            {"receivesShadows", "receives_shadows"},
            {"shadowLodBias", "shadow_lod_bias"},
            {"maximumShadowDistance", "maximum_shadow_distance"},
            {"representationPolicy", "representation"},
            {"affectsIndirectLighting", "affects_indirect_lighting"},
            {"surfaceCardDensityBias", "surface_card_density_bias"},
            {"distanceFieldResolutionBias", "distance_field_resolution_bias"},
            {"visibleInHardwareTracing", "visible_in_hardware_tracing"},
            {"motionVectors", "motion_vectors"},
            {"receiveDecals", "receive_decals"},
            {"occlusionCulling", "occlusion_culling"},
            {"boundsScale", "bounds_scale"},
            {"minimumDrawDistance", "minimum_draw_distance"},
            {"maximumDrawDistance", "maximum_draw_distance"},
            {"forcedLod", "forced_lod"},
            {"lodBias", "lod_bias"},
        };
        const auto fields = ecs::component_metadata<mesh_renderer_component>().fields;
        for (auto& field : component.fields)
        {
            if (field.known) continue;
            const auto alias = std::find_if(std::begin(aliases), std::end(aliases),
                                            [&](const auto& entry) { return field.name == entry.first; });
            if (alias == std::end(aliases)) continue;
            const auto metadata = std::find_if(fields.begin(), fields.end(),
                                               [&](const auto& entry) { return entry.name == alias->second; });
            if (metadata == fields.end()) continue;
            field.id = metadata->id;
            field.name = std::string(metadata->name);
            field.known = true;
        }
        return persistence::persistence_status::success();
    };
    if (!registry.register_document(persistence::document_kind::scene, 1, 2, document_upgrade) ||
        !registry.register_document(persistence::document_kind::scene, 2, 3, document_upgrade) ||
        !registry.register_document(persistence::document_kind::prefab, 1, 2, document_upgrade) ||
        !registry.register_component(ecs::component_metadata<terrain_component>().id, 1, 2, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<terrain_component>().id, 2, 3, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<terrain_component>().id, 3, 4, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<camera_component>().id, 1, 2, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<camera_component>().id, 2, 3, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<mesh_renderer_component>().id, 1, 2, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<mesh_renderer_component>().id, 2, 3, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<mesh_renderer_component>().id, 3, 4, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<mesh_renderer_component>().id, 4, 5, mesh_v4_to_v5) ||
        !registry.register_component(ecs::component_metadata<water_component>().id, 1, 2, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<water_component>().id, 2, 3, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<water_component>().id, 3, 4, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<vegetation_component>().id, 1, 2, component_upgrade) ||
        !registry.register_component(ecs::component_metadata<irradiance_probe_volume_component>().id, 1, 2,
                                     component_upgrade))
    {
        return persistence::persistence_status::failure({.code = persistence::persistence_error_code::migration_invalid,
                                                         .message = "failed to register scene document migrations"});
    }
    const auto register_light = [&](ecs::component_type_id type)
    {
        return registry.register_component(type, 1, 2, component_upgrade) &&
               registry.register_component(type, 2, 3, component_upgrade) &&
               registry.register_component(type, 3, 4, component_upgrade) &&
               registry.register_component(type, 4, 5, component_upgrade);
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
