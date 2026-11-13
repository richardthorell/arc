#include <arc/scene/cooked_material_hydration.h>

#include <arc/ecs/hierarchy.h>
#include <arc/render/material_pass.h>
#include <arc/scene/components.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace arc::scene
{
std::string hydrate_cooked_scene_materials(ecs::world& scene, render::renderer& renderer,
                                           std::span<const cooked_material_upload> uploads,
                                           cooked_material_residencies& residency)
{
    if (!residency.empty()) return "Cooked material handles still reference an existing renderer generation";
    if (uploads.empty()) return {};

    std::unordered_map<ecs::entity_guid, ecs::entity, ecs::entity_guid_hash> owners;
    for (const auto entity : scene.entities_snapshot())
    {
        const auto* id = scene.try_get<ecs::persistent_id_component>(entity);
        if (id && !owners.emplace(id->value, entity).second)
            return "Scene contains duplicate persistent entity identities";
    }

    std::unordered_set<ecs::entity_guid, ecs::entity_guid_hash> unique_entities;
    std::unordered_map<assets::asset_guid, std::size_t, assets::asset_guid_hash> unique_materials;
    std::vector<ecs::entity> destinations;
    destinations.reserve(uploads.size());
    for (std::size_t i = 0; i < uploads.size(); ++i)
    {
        const auto& upload = uploads[i];
        if (!upload.entity.valid() || !upload.material.valid() || !unique_entities.insert(upload.entity).second)
            return "Cooked material binding has invalid or duplicate identity";
        const auto owner = owners.find(upload.entity);
        if (owner == owners.end() || !scene.has<mesh_renderer_component>(owner->second))
            return "Cooked material binding has no MeshRenderer owner";
        const auto* component = scene.try_get<mesh_renderer_component>(owner->second);
        if (component->material.valid()) return "MeshRenderer retains a stale material handle";
        if (!upload.descriptor.runtime_program ||
            !render::material_runtime_program_compatible(*upload.descriptor.runtime_program) ||
            upload.descriptor.runtime_program->passes.empty())
            return "Cooked material has no compatible executable runtime program";
        const auto program_owner = upload.program_asset.valid() ? upload.program_asset : upload.material;
        if (upload.descriptor.shader_package.high != program_owner.high ||
            upload.descriptor.shader_package.low != program_owner.low)
            return "Cooked material shader package identity differs from its declared program owner";
        unique_materials.try_emplace(upload.material, i);
        destinations.push_back(owner->second);
    }

    std::unordered_map<assets::asset_guid, render::material_handle, assets::asset_guid_hash> handles;
    for (const auto& [guid, index] : unique_materials)
    {
        auto handle = renderer.create_material(uploads[index].descriptor);
        if (!handle.valid()) return "Renderer failed to allocate a cooked material handle";
        handles.emplace(guid, handle);
    }
    cooked_material_residencies staged;
    staged.reserve(uploads.size());
    for (std::size_t i = 0; i < uploads.size(); ++i)
    {
        const auto handle = handles.at(uploads[i].material);
        scene.get<mesh_renderer_component>(destinations[i]).material = handle;
        staged.push_back({.entity = destinations[i], .handle = handle});
    }
    residency = std::move(staged);
    return {};
}

void unbind_cooked_scene_materials(ecs::world& scene, cooked_material_residencies& residency) noexcept
{
    for (const auto& item : residency)
    {
        if (!scene.alive(item.entity)) continue;
        auto* component = scene.try_get<mesh_renderer_component>(item.entity);
        if (component && component->material == item.handle) component->material = {};
    }
    residency.clear();
}
} // namespace arc::scene
