#include <arc/scene/runtime_scene.h>

#include <arc/ecs/hierarchy.h>
#include <arc/scene/components.h>
#include <arc/scene/persistence.h>

#include <algorithm>
#include <exception>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::scene
{
namespace
{

template <class Component>
bool decode_component(ecs::world& destination, ecs::entity entity, const persistence::archive_component_record& record,
                      const persistence::component_persistence_registry& registry, std::string& error)
{
    Component value{};
    const auto status = registry.decode(record.type, &value, record);
    if (!status)
    {
        error = "Cannot decode component '" + record.name + "': " + status.error().message;
        return false;
    }
    // Asset identity is carried by the cooked dependency manifest. Any
    // serialized renderer handles are process-local and must not survive boot.
    if constexpr (std::is_same_v<Component, mesh_renderer_component>)
    {
        value.mesh = {};
        value.material = {};
    }
    else if constexpr (std::is_same_v<Component, virtual_mesh_renderer_component> ||
                       std::is_same_v<Component, skinned_mesh_renderer_component>)
    {
        value.mesh = {};
        value.material = {};
    }
    destination.emplace<Component>(entity, std::move(value));
    return true;
}

bool populate_component(ecs::world& destination, ecs::entity entity,
                        const persistence::archive_component_record& record,
                        const persistence::component_persistence_registry& registry, std::string& error)
{
    if (!record.known)
    {
        error = "Cooked scene has an unregistered component '" + record.name + "'";
        return false;
    }
    for (const auto& field : record.fields)
    {
        if (!field.known)
        {
            error = "Cooked component '" + record.name + "' has an unsupported field '" + field.name + "'";
            return false;
        }
    }
#define ARC_DECODE_SCENE_COMPONENT(Type)                                                                               \
    if (record.type == ecs::component_metadata<Type>().id)                                                             \
    return decode_component<Type>(destination, entity, record, registry, error)

    ARC_DECODE_SCENE_COMPONENT(name_component);
    ARC_DECODE_SCENE_COMPONENT(transform_component);
    ARC_DECODE_SCENE_COMPONENT(tag_component);
    ARC_DECODE_SCENE_COMPONENT(active_component);
    ARC_DECODE_SCENE_COMPONENT(bounds_component);
    ARC_DECODE_SCENE_COMPONENT(camera_component);
    ARC_DECODE_SCENE_COMPONENT(mesh_renderer_component);
    ARC_DECODE_SCENE_COMPONENT(virtual_mesh_renderer_component);
    ARC_DECODE_SCENE_COMPONENT(skinned_mesh_renderer_component);
    ARC_DECODE_SCENE_COMPONENT(lod_component);
    ARC_DECODE_SCENE_COMPONENT(instance_group_component);
    ARC_DECODE_SCENE_COMPONENT(render_layer_component);
    ARC_DECODE_SCENE_COMPONENT(mobility_component);
    ARC_DECODE_SCENE_COMPONENT(directional_light_component);
    ARC_DECODE_SCENE_COMPONENT(point_light_component);
    ARC_DECODE_SCENE_COMPONENT(spot_light_component);
    ARC_DECODE_SCENE_COMPONENT(area_light_component);
    ARC_DECODE_SCENE_COMPONENT(reflection_probe_component);
    ARC_DECODE_SCENE_COMPONENT(irradiance_probe_component);
    ARC_DECODE_SCENE_COMPONENT(irradiance_probe_volume_component);
    ARC_DECODE_SCENE_COMPONENT(baked_lighting_component);
    ARC_DECODE_SCENE_COMPONENT(indirect_lighting_component);
    ARC_DECODE_SCENE_COMPONENT(world_environment_component);
    ARC_DECODE_SCENE_COMPONENT(sky_atmosphere_component);
    ARC_DECODE_SCENE_COMPONENT(celestial_sky_component);
    ARC_DECODE_SCENE_COMPONENT(cloud_layers_component);
    ARC_DECODE_SCENE_COMPONENT(environment_lighting_component);
    ARC_DECODE_SCENE_COMPONENT(height_fog_component);
    ARC_DECODE_SCENE_COMPONENT(terrain_component);
    ARC_DECODE_SCENE_COMPONENT(water_component);
    ARC_DECODE_SCENE_COMPONENT(vegetation_component);
    ARC_DECODE_SCENE_COMPONENT(decal_component);
    ARC_DECODE_SCENE_COMPONENT(ecs::prefab_instance_component);
    ARC_DECODE_SCENE_COMPONENT(ecs::world_region_component);
#undef ARC_DECODE_SCENE_COMPONENT
    error = "Cooked scene component '" + record.name + "' has no runtime ECS factory";
    return false;
}

std::string authored_mesh_subresource(const persistence::archive_entity_record& record)
{
    if (record.extensions.kind != persistence::archive_value_kind::object) return {};
    for (const auto& extension : record.extensions.object)
    {
        if (extension.name != "assetBinding" || extension.value.kind != persistence::archive_value_kind::object)
            continue;
        for (const auto& field : extension.value.object)
            if (field.name == "subresource" && field.value.kind == persistence::archive_value_kind::string)
                return field.value.string;
    }
    return {};
}

} // namespace

std::string materialize_cooked_scene(ecs::world& destination, std::span<const std::byte> bytes,
                                     cooked_scene_mesh_bindings* bindings)
{
    if (bindings) bindings->clear();
    if (destination.live_count() != 0) return "Cooked scene requires an initially empty ECS world";

    persistence::component_persistence_registry components;
    persistence::schema_migration_registry migrations;
    const auto migration_status = register_persistence_migrations(migrations);
    if (!register_persistence_components(components) || !migration_status)
        return "Cannot initialize runtime scene persistence registry";

    const auto archive = persistence::read_tagged_binary(bytes, components, &migrations);
    if (!archive.succeeded()) return "Cannot decode cooked scene: " + archive.error;
    if (!archive.integrity_verified || archive.document.kind != persistence::document_kind::scene ||
        !archive.document.id.valid() || archive.target_identity.empty())
        return "Cooked scene has invalid integrity, identity, kind or cook target";

    std::unordered_map<ecs::entity_guid, ecs::entity, ecs::entity_guid_hash> handles;
    handles.reserve(archive.document.entities.size());
    std::vector<ecs::entity> created;
    created.reserve(archive.document.entities.size());
    cooked_scene_mesh_bindings staged_bindings;
    const auto rollback = [&](std::string message)
    {
        for (auto it = created.rbegin(); it != created.rend(); ++it)
            destination.destroy(*it);
        return message;
    };

    try
    {
        // Create every entity first so parent references may point forward.
        for (const auto& record : archive.document.entities)
        {
            if (!record.id.valid() || handles.contains(record.id))
                return rollback("Cooked scene contains a duplicate or invalid entity GUID");
            const auto entity = destination.create();
            created.push_back(entity);
            handles.emplace(record.id, entity);
            destination.emplace<ecs::persistent_id_component>(entity, record.id);
        }

        for (const auto& record : archive.document.entities)
        {
            const auto entity = handles.at(record.id);
            if (record.region.valid())
                return rollback("Cooked scene requires runtime region activation not yet available");
            for (const auto& component : record.components)
            {
                if (component.type == ecs::component_metadata<ecs::persistent_id_component>().id ||
                    component.type == ecs::component_metadata<ecs::hierarchy_component>().id)
                    return rollback("Cooked scene contains a reserved runtime identity/hierarchy component");
                std::string error;
                if (!populate_component(destination, entity, component, components, error))
                    return rollback("Entity " + ecs::to_string(record.id) + ": " + error);
            }
        }

        // Apply sibling order after all components have been materialized.
        std::vector<const persistence::archive_entity_record*> ordered;
        ordered.reserve(archive.document.entities.size());
        for (const auto& record : archive.document.entities)
            if (record.parent.valid()) ordered.push_back(&record);
        std::sort(ordered.begin(), ordered.end(),
                  [](const auto* a, const auto* b)
                  {
                      if (a->parent != b->parent) return a->parent < b->parent;
                      if (a->sibling_order != b->sibling_order) return a->sibling_order < b->sibling_order;
                      return a->id < b->id;
                  });
        for (const auto* record : ordered)
        {
            const auto parent = handles.find(record->parent);
            if (parent == handles.end() || !ecs::reparent(destination, handles.at(record->id), parent->second))
                return rollback("Cooked scene has missing or cyclic parent for " + ecs::to_string(record->id));
        }

        // GPU handles belong to the renderer generation. Read authored GUIDs
        // from the archive's manifest instead, keyed by the stable entity ID.
        const auto mesh_component_id = ecs::component_metadata<mesh_renderer_component>().id;
        std::unordered_map<ecs::entity_guid, std::size_t, ecs::entity_guid_hash> binding_indices;
        for (const auto& dependency : archive.document.dependencies)
        {
            if (dependency.owner_component != mesh_component_id ||
                (dependency.owner_field != 1u && dependency.owner_field != 2u))
                continue;

            const auto owner = handles.find(dependency.owner_entity);
            if (owner == handles.end() || !destination.has<mesh_renderer_component>(owner->second))
                return rollback("Cooked mesh binding has no owning MeshRenderer entity");
            if (!dependency.reference.guid.valid())
            {
                if (dependency.required) return rollback("Required cooked mesh binding has no asset GUID");
                continue;
            }

            const auto type = dependency.reference.expected_type;
            if (dependency.owner_field == 1u && type.valid() && type != assets::asset_types::unknown &&
                type != assets::asset_types::imported_scene && type != assets::asset_types::static_mesh)
                return rollback("Cooked mesh binding has an incompatible asset type");
            if (dependency.owner_field == 2u && type.valid() && type != assets::asset_types::unknown &&
                type != assets::asset_types::material && type != assets::asset_types::material_instance)
                return rollback("Cooked material binding has an incompatible asset type");

            const auto [it, inserted] = binding_indices.try_emplace(dependency.owner_entity, staged_bindings.size());
            if (inserted) staged_bindings.push_back({.entity = dependency.owner_entity});
            auto& binding = staged_bindings[it->second];
            auto& slot = dependency.owner_field == 1u ? binding.source : binding.material;
            if (slot.guid.valid()) return rollback("Cooked scene contains duplicate mesh/material binding");
            slot = dependency.reference;
        }

        // Preserve the authored model subresource/part from the entity extension.
        for (const auto& record : archive.document.entities)
        {
            const auto found = binding_indices.find(record.id);
            if (found != binding_indices.end())
                staged_bindings[found->second].subresource = authored_mesh_subresource(record);
        }
        std::sort(staged_bindings.begin(), staged_bindings.end(),
                  [](const auto& lhs, const auto& rhs)
                  { return std::tie(lhs.entity.high, lhs.entity.low) < std::tie(rhs.entity.high, rhs.entity.low); });
    }
    catch (const std::exception& error)
    {
        return rollback(std::string("Cooked scene ECS materialization failed: ") + error.what());
    }
    catch (...)
    {
        return rollback("Cooked scene ECS materialization failed with an unknown exception");
    }

    if (bindings) *bindings = std::move(staged_bindings);
    return {};
}

} // namespace arc::scene
