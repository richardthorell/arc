#include <arc/scene/terrain_region_seams.h>

#include <limits>

namespace arc::scene
{

std::optional<terrain_region_id> terrain_region_neighbor(terrain_region_id region, terrain_region_edge edge) noexcept
{
    switch (edge)
    {
        case terrain_region_edge::negative_x:
            if (region.x == std::numeric_limits<std::int64_t>::min()) return std::nullopt;
            --region.x;
            break;
        case terrain_region_edge::positive_x:
            if (region.x == std::numeric_limits<std::int64_t>::max()) return std::nullopt;
            ++region.x;
            break;
        case terrain_region_edge::negative_z:
            if (region.z == std::numeric_limits<std::int64_t>::min()) return std::nullopt;
            --region.z;
            break;
        case terrain_region_edge::positive_z:
            if (region.z == std::numeric_limits<std::int64_t>::max()) return std::nullopt;
            ++region.z;
            break;
    }
    return region;
}

bool terrain_regions_share_edge(terrain_region_id lhs, terrain_region_id rhs) noexcept
{
    constexpr terrain_region_edge edges[]{terrain_region_edge::negative_x, terrain_region_edge::positive_x,
                                          terrain_region_edge::negative_z, terrain_region_edge::positive_z};
    for (const auto edge : edges)
    {
        const auto neighbor = terrain_region_neighbor(lhs, edge);
        if (neighbor && *neighbor == rhs) return true;
    }
    return false;
}

std::optional<terrain_region_id> terrain_shared_seam_owner(terrain_region_id lhs, terrain_region_id rhs) noexcept
{
    if (!terrain_regions_share_edge(lhs, rhs)) return std::nullopt;
    if (lhs.x < rhs.x || (lhs.x == rhs.x && lhs.z < rhs.z)) return lhs;
    return rhs;
}

} // namespace arc::scene
