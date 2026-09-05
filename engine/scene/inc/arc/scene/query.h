#pragma once

#include <arc/ecs/world.h>
#include <arc/geometric/box.h>
#include <arc/math/math.h>
#include <arc/scene/components.h>

#include <limits>
#include <span>
#include <vector>

namespace arc::render
{
class renderer;
}

namespace arc::scene
{

/** Filters shared by editor/runtime scene geometry queries. */
struct scene_query_filter
{
    ecs::entity ignore_entity{};
    std::span<const ecs::entity> ignored_entities{};
    bool include_inactive{};
    bool include_hidden{};
    bool bounds_fallback{true};
};

/** Nearest scene-query contact in world space. */
struct scene_query_hit
{
    ecs::entity entity{};
    math::vector3f position{};
    math::vector3f normal{};
    float distance{std::numeric_limits<float>::max()};
    bool exact{};

    [[nodiscard]] explicit constexpr operator bool() const noexcept
    {
        return entity.valid();
    }
};

/** Transform local bounds into a conservative world-space AABB. */
[[nodiscard]] geometric::box3f query_world_bounds(const geometric::box3f& local_bounds,
                                                  const transform_component& transform) noexcept;

/**
 * Raycast scene terrain/static meshes and optionally fall back to entity bounds.
 * The renderer may be null when callers only require bounds/terrain queries.
 */
[[nodiscard]] scene_query_hit raycast_scene(const ecs::world& world, const render::renderer* renderer,
                                            const math::vector3f& origin, const math::vector3f& direction,
                                            float max_distance = std::numeric_limits<float>::max(),
                                            const scene_query_filter& filter = {}) noexcept;

/** Return entities whose transformed bounds overlap a world-space AABB. */
[[nodiscard]] std::vector<ecs::entity> overlap_scene_bounds(const ecs::world& world, const geometric::box3f& bounds,
                                                            const scene_query_filter& filter = {});

/**
 * Sweep a world-space AABB along a direction using conservative bounds.
 * This provides a stable query contract before a physics backend supplies shape sweeps.
 */
[[nodiscard]] scene_query_hit sweep_scene_bounds(const ecs::world& world, const geometric::box3f& moving_bounds,
                                                 const math::vector3f& direction, float max_distance,
                                                 const scene_query_filter& filter = {}) noexcept;

} // namespace arc::scene
