#pragma once

#include <arc/scene/components.h>
#include <arc/scene/registry.h>

#include <vector>

namespace arc::scene
{

enum class reparent_transform_policy
{
    preserve_world,
    preserve_local
};

bool is_descendant(const registry& scene, entity candidate, entity ancestor) noexcept;
std::vector<entity> roots(const registry& scene);
std::vector<entity> children(const registry& scene, entity parent);
bool reparent(
    registry& scene,
    entity child,
    entity parent = {},
    entity before_sibling = {},
    reparent_transform_policy policy = reparent_transform_policy::preserve_world) noexcept;
bool reorder(registry& scene, entity child, entity before_sibling = {}) noexcept;
void detach(registry& scene, entity child) noexcept;
void mark_transform_subtree_dirty(registry& scene, entity root) noexcept;
void update_world_transforms(registry& scene) noexcept;
std::vector<entity> subtree(const registry& scene, entity root);
bool destroy_subtree(registry& scene, entity root) noexcept;

} // namespace arc::scene
