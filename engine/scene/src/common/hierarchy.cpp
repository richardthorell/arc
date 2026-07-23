#include <arc/scene/hierarchy.h>
#include <arc/scene/transforms.h>

#include <algorithm>
#include <unordered_set>

namespace arc::scene
{
namespace
{

hierarchy_component& links(registry& scene, entity value)
{
    if (auto* existing = scene.try_get<hierarchy_component>(value))
        return *existing;
    return scene.emplace<hierarchy_component>(value);
}

void rebuild_root_links(registry& scene, const std::vector<entity>& order)
{
    for (std::size_t index = 0; index < order.size(); ++index)
    {
        auto& value = links(scene, order[index]);
        value.parent = {};
        value.previous_sibling = index > 0 ? order[index - 1] : entity{};
        value.next_sibling = index + 1 < order.size() ? order[index + 1] : entity{};
    }
}

void update_subtree(
    registry& scene,
    entity value,
    const math::matrix4f* parent_world,
    std::unordered_set<entity, ecs::entity_hash>& visited) noexcept
{
    if (!scene.alive(value) || !visited.insert(value).second)
        return;
    auto* transform = scene.try_get<transform_component>(value);
    if (transform)
    {
        const auto local = local_matrix(*transform);
        transform->world = parent_world ? math::matmul(*parent_world, local) : local;
        transform->dirty = false;
        if (auto* bounds = scene.try_get<bounds_component>(value))
            bounds->dirty = true;
        parent_world = &transform->world;
    }
    const auto* hierarchy = scene.try_get<hierarchy_component>(value);
    entity child = hierarchy ? hierarchy->first_child : entity{};
    std::unordered_set<entity, ecs::entity_hash> visited_siblings;
    while (scene.alive(child) && visited_siblings.insert(child).second)
    {
        const auto* child_links = scene.try_get<hierarchy_component>(child);
        const entity next = child_links ? child_links->next_sibling : entity{};
        update_subtree(scene, child, parent_world, visited);
        child = next;
    }
}

} // namespace

bool is_descendant(const registry& scene, entity candidate, entity ancestor) noexcept
{
    if (!scene.alive(candidate) || !scene.alive(ancestor))
        return false;
    entity current = candidate;
    std::unordered_set<entity, ecs::entity_hash> visited;
    while (scene.alive(current) && visited.insert(current).second)
    {
        if (current == ancestor)
            return candidate != ancestor;
        const auto* hierarchy = scene.try_get<hierarchy_component>(current);
        current = hierarchy ? hierarchy->parent : entity{};
    }
    return false;
}

std::vector<entity> roots(const registry& scene)
{
    std::vector<entity> candidates;
    std::uint32_t maximum_index{};
    for (const auto value : scene.entities())
    {
        const auto* hierarchy = scene.try_get<hierarchy_component>(value);
        if (!hierarchy || !scene.alive(hierarchy->parent))
        {
            candidates.push_back(value);
            maximum_index = std::max(maximum_index, value.index);
        }
    }

    std::vector<entity> result;
    result.reserve(candidates.size());
    std::vector<bool> candidate_indices(static_cast<std::size_t>(maximum_index) + 1u);
    std::vector<bool> visited_indices(candidate_indices.size());
    for (const auto candidate : candidates)
        candidate_indices[candidate.index] = true;
    const auto is_candidate = [&](entity value) {
        return value.index < candidate_indices.size() && candidate_indices[value.index];
    };
    const auto append_chain = [&](entity head) {
        entity current = head;
        while (scene.alive(current) && is_candidate(current) && !visited_indices[current.index])
        {
            visited_indices[current.index] = true;
            result.push_back(current);
            const auto* hierarchy = scene.try_get<hierarchy_component>(current);
            current = hierarchy ? hierarchy->next_sibling : entity{};
        }
    };

    // Prefer the explicitly linked root chain, then append newly-created/unlinked
    // roots in deterministic registry order until the next hierarchy edit links them.
    for (const auto candidate : candidates)
    {
        const auto* hierarchy = scene.try_get<hierarchy_component>(candidate);
        if (hierarchy && !scene.alive(hierarchy->previous_sibling) && scene.alive(hierarchy->next_sibling))
            append_chain(candidate);
    }
    for (const auto candidate : candidates)
        if (!visited_indices[candidate.index])
            append_chain(candidate);
    return result;
}

std::vector<entity> children(const registry& scene, entity parent)
{
    if (!scene.alive(parent))
        return roots(scene);
    std::vector<entity> result;
    const auto* hierarchy = scene.try_get<hierarchy_component>(parent);
    entity child = hierarchy ? hierarchy->first_child : entity{};
    std::unordered_set<entity, ecs::entity_hash> visited;
    while (scene.alive(child) && visited.insert(child).second)
    {
        result.push_back(child);
        const auto* links = scene.try_get<hierarchy_component>(child);
        child = links ? links->next_sibling : entity{};
    }
    return result;
}

void detach(registry& scene, entity child) noexcept
{
    auto* child_links = scene.try_get<hierarchy_component>(child);
    if (!child_links)
        return;
    const entity parent = child_links->parent;
    if (scene.alive(parent))
    {
        auto& parent_links = links(scene, parent);
        if (parent_links.first_child == child)
            parent_links.first_child = child_links->next_sibling;
        if (parent_links.child_count > 0)
            --parent_links.child_count;
    }
    if (scene.alive(child_links->previous_sibling))
        links(scene, child_links->previous_sibling).next_sibling = child_links->next_sibling;
    if (scene.alive(child_links->next_sibling))
        links(scene, child_links->next_sibling).previous_sibling = child_links->previous_sibling;
    child_links->parent = {};
    child_links->previous_sibling = {};
    child_links->next_sibling = {};
}

bool reparent(registry& scene, entity child, entity parent, entity before_sibling, reparent_transform_policy policy) noexcept
{
    if (!scene.alive(child) || child == parent || (parent.valid() && !scene.alive(parent)) ||
        arc::scene::is_descendant(scene, parent, child))
        return false;
    if (before_sibling.valid())
    {
        const auto* before_links = scene.try_get<hierarchy_component>(before_sibling);
        if (!scene.alive(before_sibling) || before_sibling == child || !before_links || before_links->parent != parent)
            return false;
    }

    math::matrix4f preserved_local = math::identity<float, 4>();
    transform_component preserved_transform;
    bool has_preserved_transform{};
    if (policy == reparent_transform_policy::preserve_world && scene.has<transform_component>(child))
    {
        update_world_transforms(scene);
        preserved_local = scene.get<transform_component>(child).world;
        entity transform_parent = parent;
        while (scene.alive(transform_parent) && !scene.has<transform_component>(transform_parent))
        {
            const auto* hierarchy = scene.try_get<hierarchy_component>(transform_parent);
            transform_parent = hierarchy ? hierarchy->parent : entity{};
        }
        if (const auto* parent_transform = scene.try_get<transform_component>(transform_parent))
        {
            math::matrix4f inverse_parent;
            if (!inverse_affine(parent_transform->world, inverse_parent))
                return false;
            preserved_local = math::matmul(inverse_parent, preserved_local);
        }
        if (!decompose_trs(preserved_local, preserved_transform))
            return false;
        has_preserved_transform = true;
    }

    arc::scene::detach(scene, child);
    auto& child_links = links(scene, child);
    child_links.parent = parent;
    if (scene.alive(parent))
    {
        auto& parent_links = links(scene, parent);
        ++parent_links.child_count;
        if (before_sibling.valid())
        {
            auto& before_links = links(scene, before_sibling);
            child_links.next_sibling = before_sibling;
            child_links.previous_sibling = before_links.previous_sibling;
            if (scene.alive(before_links.previous_sibling))
                links(scene, before_links.previous_sibling).next_sibling = child;
            else
                parent_links.first_child = child;
            before_links.previous_sibling = child;
        }
        else if (!scene.alive(parent_links.first_child))
        {
            parent_links.first_child = child;
        }
        else
        {
            entity last = parent_links.first_child;
            while (scene.alive(links(scene, last).next_sibling))
                last = links(scene, last).next_sibling;
            links(scene, last).next_sibling = child;
            child_links.previous_sibling = last;
        }
    }
    else
    {
        auto order = roots(scene);
        order.erase(std::remove(order.begin(), order.end(), child), order.end());
        const auto before = std::find(order.begin(), order.end(), before_sibling);
        order.insert(before_sibling.valid() && before != order.end() ? before : order.end(), child);
        rebuild_root_links(scene, order);
    }

    if (policy == reparent_transform_policy::preserve_world)
    {
        if (auto* transform = scene.try_get<transform_component>(child); transform && has_preserved_transform)
            *transform = preserved_transform;
    }
    mark_transform_subtree_dirty(scene, child);
    return true;
}

bool reorder(registry& scene, entity child, entity before_sibling) noexcept
{
    if (!scene.alive(child))
        return false;
    const auto* value = scene.try_get<hierarchy_component>(child);
    return reparent(scene, child, value ? value->parent : entity{}, before_sibling, reparent_transform_policy::preserve_local);
}

void mark_transform_subtree_dirty(registry& scene, entity root) noexcept
{
    for (entity value : subtree(scene, root))
    {
        if (auto* transform = scene.try_get<transform_component>(value))
            transform->mark_dirty();
        if (auto* bounds = scene.try_get<bounds_component>(value))
            bounds->dirty = true;
    }
}

void update_world_transforms(registry& scene) noexcept
{
    std::unordered_set<entity, ecs::entity_hash> visited;
    for (const entity value : roots(scene))
        update_subtree(scene, value, nullptr, visited);
}

std::vector<entity> subtree(const registry& scene, entity root)
{
    std::vector<entity> result;
    if (!scene.alive(root))
        return result;
    result.push_back(root);
    std::unordered_set<entity, ecs::entity_hash> visited{ root };
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        const auto nested = arc::scene::children(scene, result[index]);
        for (const entity value : nested)
            if (visited.insert(value).second)
                result.push_back(value);
    }
    return result;
}

bool destroy_subtree(registry& scene, entity root) noexcept
{
    auto values = subtree(scene, root);
    if (values.empty())
        return false;
    arc::scene::detach(scene, root);
    for (auto it = values.rbegin(); it != values.rend(); ++it)
        scene.destroy(*it);
    return true;
}

} // namespace arc::scene
