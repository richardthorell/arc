#pragma once

#include <arc/ecs/world.h>
#include <arc/scene/components.h>

#include <vector>

namespace arc::scene
{

enum class reparent_transform_policy
{
    preserve_world,
    preserve_local
};

bool is_descendant(const ecs::world& scene, ecs::entity candidate, ecs::entity ancestor) noexcept;
std::vector<ecs::entity> roots(const ecs::world& scene);
std::vector<ecs::entity> children(const ecs::world& scene, ecs::entity parent);
bool reparent(
    ecs::world& scene,
    ecs::entity child,
    ecs::entity parent = {},
    ecs::entity before_sibling = {},
    reparent_transform_policy policy = reparent_transform_policy::preserve_world) noexcept;
bool reorder(ecs::world& scene, ecs::entity child, ecs::entity before_sibling = {}) noexcept;
void detach(ecs::world& scene, ecs::entity child) noexcept;
void mark_transform_subtree_dirty(ecs::world& scene, ecs::entity root) noexcept;
void update_world_transforms(ecs::world& scene) noexcept;
std::vector<ecs::entity> subtree(const ecs::world& scene, ecs::entity root);
bool destroy_subtree(ecs::world& scene, ecs::entity root) noexcept;

} // namespace arc::scene
