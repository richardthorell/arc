#pragma once

#include <arc/ecs/identity.h>
#include <arc/ecs/world.h>

namespace arc::ecs
{

struct persistent_id_component
{
    entity_guid value{};
};

/** Intrusive hierarchy links; traversal performs no allocation. */
struct hierarchy_component
{
    entity parent{};
    entity first_child{};
    entity previous_sibling{};
    entity next_sibling{};
    std::uint32_t child_count{};
};

template <>
struct component_traits<persistent_id_component>
{
    static constexpr bool reflected = true;
    static constexpr std::string_view canonical_name = "arc.ecs.persistent_id";
    static constexpr component_type_id id{ 0xa7c0000000000000ull, 0x0000000000000001ull };
    static constexpr std::array<component_field_descriptor, 1> fields{{
        { 1, "value", "GUID", reflected_field_kind::structure,
            reflected_field_flags::serialized }
    }};
    static constexpr component_descriptor descriptor{
        id, canonical_name, "Persistent ID", 1, sizeof(persistent_id_component),
        alignof(persistent_id_component), fields, false, false
    };
};

template <>
struct component_traits<hierarchy_component>
{
    static constexpr bool reflected = true;
    static constexpr std::string_view canonical_name = "arc.ecs.hierarchy";
    static constexpr component_type_id id{ 0xa7c0000000000000ull, 0x0000000000000002ull };
    static constexpr std::array<component_field_descriptor, 5> fields{{
        { 1, "parent", "Parent", reflected_field_kind::entity_reference,
            reflected_field_flags::serialized | reflected_field_flags::prefab_override },
        { 2, "first_child", "First Child", reflected_field_kind::entity_reference,
            reflected_field_flags::transient },
        { 3, "previous_sibling", "Previous Sibling", reflected_field_kind::entity_reference,
            reflected_field_flags::transient },
        { 4, "next_sibling", "Next Sibling", reflected_field_kind::entity_reference,
            reflected_field_flags::transient },
        { 5, "child_count", "Child Count", reflected_field_kind::unsigned_integer,
            reflected_field_flags::transient }
    }};
    static constexpr component_descriptor descriptor{
        id, canonical_name, "Hierarchy", 1, sizeof(hierarchy_component),
        alignof(hierarchy_component), fields, true, false
    };
};

class child_range
{
public:
    class iterator
    {
    public:
        entity operator*() const noexcept { return current_; }
        iterator& operator++() noexcept
        {
            if (current_.valid())
            {
                const auto* links = owner_->try_get<hierarchy_component>(current_);
                current_ = links ? links->next_sibling : entity{};
            }
            return *this;
        }
        friend bool operator==(const iterator&, const iterator&) noexcept = default;

    private:
        iterator(const world* owner, entity current) : owner_(owner), current_(current) {}
        const world* owner_{};
        entity current_{};
        friend class child_range;
    };

    iterator begin() const noexcept { return { owner_, first_ }; }
    iterator end() const noexcept { return { owner_, {} }; }

private:
    child_range(const world& owner, entity first) : owner_(&owner), first_(first) {}
    const world* owner_{};
    entity first_{};
    friend child_range children(const world&, entity) noexcept;
};

inline child_range children(const world& owner, entity parent) noexcept
{
    const auto* links = owner.try_get<hierarchy_component>(parent);
    return { owner, links ? links->first_child : entity{} };
}

inline bool is_descendant(const world& owner, entity candidate, entity ancestor) noexcept
{
    entity current = candidate;
    while (current.valid())
    {
        if (current == ancestor)
            return true;
        const auto* links = owner.try_get<hierarchy_component>(current);
        current = links ? links->parent : entity{};
    }
    return false;
}

inline void detach(world& owner, entity child) noexcept
{
    auto* links = owner.try_get<hierarchy_component>(child);
    if (!links)
        return;
    const entity parent = links->parent;
    const entity previous = links->previous_sibling;
    const entity next = links->next_sibling;
    if (previous.valid())
        owner.get<hierarchy_component>(previous).next_sibling = next;
    else if (parent.valid())
        owner.get<hierarchy_component>(parent).first_child = next;
    if (next.valid())
        owner.get<hierarchy_component>(next).previous_sibling = previous;
    if (parent.valid())
    {
        auto& parent_links = owner.get<hierarchy_component>(parent);
        if (parent_links.child_count)
            --parent_links.child_count;
    }
    links->parent = {};
    links->previous_sibling = {};
    links->next_sibling = {};
}

inline bool reparent(world& owner, entity child, entity parent = {}, entity before = {}) noexcept
{
    if (!owner.alive(child) || (parent.valid() && !owner.alive(parent)) ||
        child == parent || (parent.valid() && is_descendant(owner, parent, child)))
        return false;
    if (before.valid())
    {
        const auto* before_links = owner.try_get<hierarchy_component>(before);
        if (!before_links || before_links->parent != parent || before == child)
            return false;
    }
    if (!owner.has<hierarchy_component>(child))
        owner.emplace<hierarchy_component>(child);
    if (parent.valid() && !owner.has<hierarchy_component>(parent))
        owner.emplace<hierarchy_component>(parent);

    detach(owner, child);
    auto& child_links = owner.get<hierarchy_component>(child);
    child_links.parent = parent;
    if (!parent.valid())
        return true;

    auto& parent_links = owner.get<hierarchy_component>(parent);
    if (!before.valid())
    {
        if (!parent_links.first_child.valid())
            parent_links.first_child = child;
        else
        {
            entity tail = parent_links.first_child;
            while (owner.get<hierarchy_component>(tail).next_sibling.valid())
                tail = owner.get<hierarchy_component>(tail).next_sibling;
            owner.get<hierarchy_component>(tail).next_sibling = child;
            child_links.previous_sibling = tail;
        }
    }
    else
    {
        auto& before_links = owner.get<hierarchy_component>(before);
        child_links.next_sibling = before;
        child_links.previous_sibling = before_links.previous_sibling;
        if (before_links.previous_sibling.valid())
            owner.get<hierarchy_component>(before_links.previous_sibling).next_sibling = child;
        else
            parent_links.first_child = child;
        before_links.previous_sibling = child;
    }
    ++parent_links.child_count;
    return true;
}

} // namespace arc::ecs
