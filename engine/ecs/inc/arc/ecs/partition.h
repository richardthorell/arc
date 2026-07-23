#pragma once

#include <arc/ecs/hierarchy.h>
#include <arc/jobs/jobs.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace arc::ecs
{

struct world_region_id
{
    entity_guid value{};
    constexpr bool valid() const noexcept { return value.valid(); }
    friend constexpr bool operator==(world_region_id, world_region_id) noexcept = default;
};

struct world_region_id_hash
{
    std::size_t operator()(world_region_id value) const noexcept
    {
        return entity_guid_hash{}(value.value);
    }
};

enum class world_region_state : std::uint8_t
{
    unloaded,
    loading,
    loaded,
    unloading,
    failed
};

struct world_region_bounds
{
    float minimum[3]{};
    float maximum[3]{};
};

struct world_region_descriptor
{
    world_region_id id{};
    std::string name;
    std::string content_path;
    world_region_bounds bounds{};
    world_region_state state{ world_region_state::unloaded };
    bool always_loaded{};
};

struct world_region_component
{
    world_region_id region{};
};

template <>
struct component_traits<world_region_component>
{
    static constexpr bool reflected = true;
    static constexpr std::string_view canonical_name = "arc.ecs.world_region";
    static constexpr component_type_id id{ 0xa7c0000000000000ull, 0x0000000000000004ull };
    static constexpr std::array<component_field_descriptor, 1> fields{{
        { 1, "region", "Region", reflected_field_kind::structure,
            reflected_field_flags::serialized | reflected_field_flags::editable }
    }};
    static constexpr component_descriptor descriptor{
        id, canonical_name, "World Region", 1, sizeof(world_region_component),
        alignof(world_region_component), fields, true, false
    };
};

struct region_load_result
{
    world_region_id region{};
    bool succeeded{};
    std::string error;
};

class world_partition_provider
{
public:
    virtual ~world_partition_provider() = default;
    virtual job_future<region_load_result> request_load(
        const world_region_descriptor& region,
        cancellation_token cancellation = {}) = 0;
    virtual job_future<region_load_result> request_unload(
        const world_region_descriptor& region,
        cancellation_token cancellation = {}) = 0;
};

class world_partition
{
public:
    bool add_region(world_region_descriptor descriptor)
    {
        if (!descriptor.id.valid() || regions_.contains(descriptor.id))
            return false;
        regions_.emplace(descriptor.id, std::move(descriptor));
        return true;
    }

    const world_region_descriptor* find(world_region_id id) const noexcept
    {
        const auto found = regions_.find(id);
        return found == regions_.end() ? nullptr : &found->second;
    }

    world_region_descriptor* find(world_region_id id) noexcept
    {
        const auto found = regions_.find(id);
        return found == regions_.end() ? nullptr : &found->second;
    }

    bool assign(world& owner, entity value, world_region_id region)
    {
        const auto* descriptor = find(region);
        if (!owner.alive(value) || !descriptor || descriptor->state != world_region_state::loaded)
            return false;
        if (owner.has<world_region_component>(value))
            owner.get<world_region_component>(value).region = region;
        else
            owner.emplace<world_region_component>(value, region);
        return true;
    }

    bool migrate_subtree(world& owner, entity root, world_region_id region)
    {
        if (!assign(owner, root, region))
            return false;
        for (const entity child : children(owner, root))
            if (!migrate_subtree(owner, child, region))
                return false;
        return true;
    }

    std::vector<world_region_descriptor> snapshot() const
    {
        std::vector<world_region_descriptor> result;
        result.reserve(regions_.size());
        for (const auto& [_, region] : regions_)
            result.push_back(region);
        return result;
    }

private:
    std::unordered_map<world_region_id, world_region_descriptor, world_region_id_hash> regions_;
};

} // namespace arc::ecs
