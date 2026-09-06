#include <arc/scene/query.h>

#include <arc/render/renderer.h>
#include <arc/scene/terrain.h>
#include <arc/scene/transforms.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace arc::scene
{
namespace
{

bool ignored(const scene_query_filter& filter, ecs::entity entity) noexcept
{
    if (entity == filter.ignore_entity) return true;
    return std::find(filter.ignored_entities.begin(), filter.ignored_entities.end(), entity) !=
           filter.ignored_entities.end();
}

bool queryable(const ecs::world& world, ecs::entity entity, const scene_query_filter& filter) noexcept
{
    if (ignored(filter, entity)) return false;
    if (!filter.include_inactive)
        if (const auto* active = world.try_get<active_component>(entity); active && !active->active) return false;
    if (!filter.include_hidden)
        if (const auto* mesh = world.try_get<mesh_renderer_component>(entity); mesh && !mesh->visible) return false;
    return true;
}

bool intersect_ray_box(const math::vector3f& origin, const math::vector3f& direction, const geometric::box3f& bounds,
                       float& distance) noexcept
{
    float t_min = 0.0f;
    float t_max = std::numeric_limits<float>::max();
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (std::abs(direction[axis]) < 1.0e-6f)
        {
            if (origin[axis] < bounds.min[axis] || origin[axis] > bounds.max[axis]) return false;
            continue;
        }
        float first = (bounds.min[axis] - origin[axis]) / direction[axis];
        float second = (bounds.max[axis] - origin[axis]) / direction[axis];
        if (first > second) std::swap(first, second);
        t_min = std::max(t_min, first);
        t_max = std::min(t_max, second);
        if (t_min > t_max) return false;
    }
    distance = t_min;
    return true;
}

bool intersect_ray_triangle(const math::vector3f& origin, const math::vector3f& direction, const math::vector3f& a,
                            const math::vector3f& b, const math::vector3f& c, float& distance) noexcept
{
    constexpr float epsilon = 1.0e-7f;
    const auto edge1 = math::sub(b, a);
    const auto edge2 = math::sub(c, a);
    const auto p = math::cross(direction, edge2);
    const float determinant = math::dot(edge1, p);
    if (std::abs(determinant) <= epsilon) return false;
    const float inverse_determinant = 1.0f / determinant;
    const auto offset = math::sub(origin, a);
    const float u = math::dot(offset, p) * inverse_determinant;
    if (u < 0.0f || u > 1.0f) return false;
    const auto q = math::cross(offset, edge1);
    const float v = math::dot(direction, q) * inverse_determinant;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float hit = math::dot(edge2, q) * inverse_determinant;
    if (hit < 0.0f) return false;
    distance = hit;
    return true;
}

bool overlaps(const geometric::box3f& first, const geometric::box3f& second) noexcept
{
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (first.max[axis] < second.min[axis] || first.min[axis] > second.max[axis]) return false;
    return true;
}

math::vector3f box_hit_normal(const geometric::box3f& bounds, const math::vector3f& point) noexcept
{
    float nearest = std::numeric_limits<float>::max();
    math::vector3f normal{};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const float minimum_distance = std::abs(point[axis] - bounds.min[axis]);
        if (minimum_distance < nearest)
        {
            nearest = minimum_distance;
            normal = {};
            normal[axis] = -1.0f;
        }
        const float maximum_distance = std::abs(point[axis] - bounds.max[axis]);
        if (maximum_distance < nearest)
        {
            nearest = maximum_distance;
            normal = {};
            normal[axis] = 1.0f;
        }
    }
    return normal;
}

} // namespace

geometric::box3f query_world_bounds(const geometric::box3f& local_bounds, const transform_component& transform) noexcept
{
    const auto matrix = transform.dirty ? local_matrix(transform) : transform.world;
    const std::array<math::vector3f, 8> corners{
        math::vector3f{local_bounds.min[0], local_bounds.min[1], local_bounds.min[2]},
        math::vector3f{local_bounds.max[0], local_bounds.min[1], local_bounds.min[2]},
        math::vector3f{local_bounds.min[0], local_bounds.max[1], local_bounds.min[2]},
        math::vector3f{local_bounds.max[0], local_bounds.max[1], local_bounds.min[2]},
        math::vector3f{local_bounds.min[0], local_bounds.min[1], local_bounds.max[2]},
        math::vector3f{local_bounds.max[0], local_bounds.min[1], local_bounds.max[2]},
        math::vector3f{local_bounds.min[0], local_bounds.max[1], local_bounds.max[2]},
        math::vector3f{local_bounds.max[0], local_bounds.max[1], local_bounds.max[2]}};
    auto minimum = math::transform_point(matrix, corners.front());
    auto maximum = minimum;
    for (std::size_t index = 1; index < corners.size(); ++index)
    {
        const auto point = math::transform_point(matrix, corners[index]);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], point[axis]);
            maximum[axis] = std::max(maximum[axis], point[axis]);
        }
    }
    return geometric::box3f{geometric::point3f{minimum}, geometric::point3f{maximum}};
}

scene_query_hit raycast_scene(const ecs::world& world, const render::renderer* renderer, const math::vector3f& origin,
                              const math::vector3f& direction, float max_distance,
                              const scene_query_filter& filter) noexcept
{
    scene_query_hit result;
    const float length = math::length(direction);
    if (!std::isfinite(length) || length <= 1.0e-6f || !std::isfinite(max_distance) || max_distance < 0.0f)
        return result;
    const auto ray_direction = math::mul(direction, 1.0f / length);

    world.view<transform_component, bounds_component>().each(
        [&](ecs::entity entity, const transform_component& transform, const bounds_component& bounds)
        {
            if (!queryable(world, entity, filter)) return;
            const auto transformed = query_world_bounds(bounds.local_bounds, transform);
            float broad_distance{};
            if (!intersect_ray_box(origin, ray_direction, transformed, broad_distance) ||
                broad_distance > max_distance || broad_distance >= result.distance)
                return;

            scene_query_hit candidate{
                .entity = entity,
                .position = math::add(origin, math::mul(ray_direction, broad_distance)),
                .normal = box_hit_normal(transformed, math::add(origin, math::mul(ray_direction, broad_distance))),
                .distance = broad_distance,
                .exact = false};

            const auto matrix = transform.dirty ? local_matrix(transform) : transform.world;
            math::matrix4f inverse;
            if (!inverse_affine(matrix, inverse)) return;
            const auto local_origin = math::transform_point(inverse, origin);
            const auto local_direction = math::normalize(math::transform_vector(inverse, ray_direction));

            if (const auto* terrain = world.try_get<terrain_component>(entity))
            {
                const auto hit = raycast_terrain(*terrain, local_origin, local_direction);
                if (!hit.hit) return;
                const auto world_position = math::transform_point(matrix, hit.position);
                const float distance = math::dot(math::sub(world_position, origin), ray_direction);
                if (distance < 0.0f || distance > max_distance || distance >= result.distance) return;
                candidate.position = world_position;
                candidate.normal = math::normalize(math::transform_vector(matrix, hit.normal));
                candidate.distance = distance;
                candidate.exact = true;
            }
            else if (renderer)
            {
                const auto* mesh_renderer = world.try_get<mesh_renderer_component>(entity);
                const auto* mesh = mesh_renderer ? renderer->mesh_data_for(mesh_renderer->mesh) : nullptr;
                if (mesh && mesh->indices.size() >= 3u)
                {
                    float nearest_local = std::numeric_limits<float>::max();
                    math::vector3f nearest_normal{};
                    for (std::size_t index = 0; index + 2u < mesh->indices.size(); index += 3u)
                    {
                        const auto ia = mesh->indices[index];
                        const auto ib = mesh->indices[index + 1u];
                        const auto ic = mesh->indices[index + 2u];
                        if (ia >= mesh->vertices.size() || ib >= mesh->vertices.size() || ic >= mesh->vertices.size())
                            continue;
                        const auto position = [](const render::mesh_vertex& vertex)
                        { return math::vector3f{vertex.position[0], vertex.position[1], vertex.position[2]}; };
                        const auto a = position(mesh->vertices[ia]);
                        const auto b = position(mesh->vertices[ib]);
                        const auto c = position(mesh->vertices[ic]);
                        float triangle_distance{};
                        if (!intersect_ray_triangle(local_origin, local_direction, a, b, c, triangle_distance) ||
                            triangle_distance >= nearest_local)
                            continue;
                        nearest_local = triangle_distance;
                        nearest_normal = math::normalize(math::cross(math::sub(b, a), math::sub(c, a)));
                    }
                    if (nearest_local != std::numeric_limits<float>::max())
                    {
                        const auto local_position = math::add(local_origin, math::mul(local_direction, nearest_local));
                        const auto world_position = math::transform_point(matrix, local_position);
                        const float distance = math::dot(math::sub(world_position, origin), ray_direction);
                        if (distance < 0.0f || distance > max_distance || distance >= result.distance) return;
                        candidate.position = world_position;
                        candidate.normal = math::normalize(math::transform_vector(matrix, nearest_normal));
                        candidate.distance = distance;
                        candidate.exact = true;
                    }
                    else if (!filter.bounds_fallback)
                    {
                        return;
                    }
                }
                else if (!filter.bounds_fallback)
                {
                    return;
                }
            }
            else if (!filter.bounds_fallback)
            {
                return;
            }

            if (candidate.distance < result.distance) result = candidate;
        });
    return result;
}

std::vector<ecs::entity> overlap_scene_bounds(const ecs::world& world, const geometric::box3f& bounds,
                                              const scene_query_filter& filter)
{
    std::vector<ecs::entity> result;
    world.view<transform_component, bounds_component>().each(
        [&](ecs::entity entity, const transform_component& transform, const bounds_component& component_bounds)
        {
            if (!queryable(world, entity, filter)) return;
            if (overlaps(bounds, query_world_bounds(component_bounds.local_bounds, transform)))
                result.push_back(entity);
        });
    return result;
}

scene_query_hit sweep_scene_bounds(const ecs::world& world, const geometric::box3f& moving_bounds,
                                   const math::vector3f& direction, float max_distance,
                                   const scene_query_filter& filter) noexcept
{
    scene_query_hit result;
    const float length = math::length(direction);
    if (!std::isfinite(length) || length <= 1.0e-6f || !std::isfinite(max_distance) || max_distance < 0.0f)
        return result;
    const auto ray_direction = math::mul(direction, 1.0f / length);
    const math::vector3f center{(moving_bounds.min[0] + moving_bounds.max[0]) * 0.5f,
                                (moving_bounds.min[1] + moving_bounds.max[1]) * 0.5f,
                                (moving_bounds.min[2] + moving_bounds.max[2]) * 0.5f};
    const math::vector3f half_extent{(moving_bounds.max[0] - moving_bounds.min[0]) * 0.5f,
                                     (moving_bounds.max[1] - moving_bounds.min[1]) * 0.5f,
                                     (moving_bounds.max[2] - moving_bounds.min[2]) * 0.5f};

    world.view<transform_component, bounds_component>().each(
        [&](ecs::entity entity, const transform_component& transform, const bounds_component& component_bounds)
        {
            if (!queryable(world, entity, filter)) return;
            const auto target = query_world_bounds(component_bounds.local_bounds, transform);
            const geometric::box3f expanded{
                geometric::point3f{target.min[0] - half_extent[0], target.min[1] - half_extent[1],
                                   target.min[2] - half_extent[2]},
                geometric::point3f{target.max[0] + half_extent[0], target.max[1] + half_extent[1],
                                   target.max[2] + half_extent[2]}};
            float distance{};
            if (!intersect_ray_box(center, ray_direction, expanded, distance) || distance > max_distance ||
                distance >= result.distance)
                return;
            const auto position = math::add(center, math::mul(ray_direction, distance));
            result = {.entity = entity,
                      .position = position,
                      .normal = box_hit_normal(expanded, position),
                      .distance = distance,
                      .exact = false};
        });
    return result;
}

} // namespace arc::scene
