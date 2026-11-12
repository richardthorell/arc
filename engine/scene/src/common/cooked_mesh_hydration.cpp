#include <arc/scene/cooked_mesh_hydration.h>

#include <arc/ecs/hierarchy.h>
#include <arc/scene/components.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace arc::scene
{
std::string hydrate_cooked_scene_meshes(ecs::world& scene, render::renderer& renderer,
                                        std::span<const cooked_mesh_upload> uploads, cooked_mesh_residencies& residency)
{
    if (!residency.empty()) return "Cooked mesh residency still belongs to an existing renderer generation";
    if (uploads.empty()) return {};

    std::unordered_map<ecs::entity_guid, ecs::entity, ecs::entity_guid_hash> owners;
    for (const auto entity : scene.entities_snapshot())
    {
        const auto* identity = scene.try_get<ecs::persistent_id_component>(entity);
        if (!identity) continue;
        if (!owners.emplace(identity->value, entity).second)
            return "Scene contains duplicate persistent entity identities";
    }

    std::unordered_set<ecs::entity_guid, ecs::entity_guid_hash> unique_uploads;
    std::vector<ecs::entity> destinations;
    destinations.reserve(uploads.size());
    for (const auto& upload : uploads)
    {
        if (!upload.entity.valid() || !unique_uploads.insert(upload.entity).second)
            return "Cooked mesh upload has an invalid or duplicate entity identity";
        const auto found = owners.find(upload.entity);
        if (found == owners.end() || !scene.has<mesh_renderer_component>(found->second))
            return "Cooked mesh upload cannot locate its MeshRenderer entity";
        const auto* component = scene.try_get<mesh_renderer_component>(found->second);
        if (component->mesh.valid()) return "Cooked MeshRenderer still references an older renderer generation";
        if (upload.mesh.vertices.empty() || upload.mesh.indices.empty())
            return "Cooked mesh upload contains no conventional geometry";
        destinations.push_back(found->second);
    }

    cooked_mesh_residencies staged;
    staged.reserve(uploads.size());
    for (std::size_t index = 0; index < uploads.size(); ++index)
    {
        const auto handle = renderer.create_mesh(uploads[index].mesh);
        if (!handle.valid())
        {
            for (const auto& item : staged)
                static_cast<void>(renderer.destroy_mesh(item.mesh));
            return "Renderer failed to allocate cooked conventional mesh";
        }
        staged.push_back({.entity = destinations[index], .mesh = handle});
    }

    for (const auto& item : staged)
    {
        auto& component = scene.get<mesh_renderer_component>(item.entity);
        component.mesh = render::geometry_resource_handle{item.mesh};
        component.representation = render::geometry_representation_policy::conventional;
    }
    residency = std::move(staged);
    return {};
}

void release_cooked_scene_meshes(ecs::world& scene, render::renderer& renderer,
                                 cooked_mesh_residencies& residency) noexcept
{
    for (const auto& item : residency)
    {
        if (scene.alive(item.entity))
        {
            auto* component = scene.try_get<mesh_renderer_component>(item.entity);
            if (component && component->mesh.conventional == item.mesh)
                component->mesh = render::geometry_resource_handle{};
        }
        static_cast<void>(renderer.destroy_mesh(item.mesh));
    }
    residency.clear();
}
} // namespace arc::scene
