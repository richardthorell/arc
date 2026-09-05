#include <arc/render/terrain.h>
#include <arc/render/render_world.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace arc::render
{
namespace
{
std::size_t sample_index(std::uint32_t resolution, std::uint32_t x, std::uint32_t z) noexcept
{
    return static_cast<std::size_t>(z) * resolution + x;
}

bool overlaps(const terrain_sample_region& a, const terrain_sample_region& b) noexcept
{
    return a.min_x <= b.max_x && b.min_x <= a.max_x && a.min_z <= b.max_z && b.min_z <= a.max_z;
}

void calculate_node(terrain_hierarchy_node& node, std::span<const float> heights, std::uint32_t resolution, float width,
                    float depth, float inherited_error)
{
    float minimum = std::numeric_limits<float>::max();
    float maximum = std::numeric_limits<float>::lowest();
    float error = inherited_error;
    const auto extent_x = std::max(1u, node.samples.max_x - node.samples.min_x);
    const auto extent_z = std::max(1u, node.samples.max_z - node.samples.min_z);
    const float h00 = heights[sample_index(resolution, node.samples.min_x, node.samples.min_z)];
    const float h10 = heights[sample_index(resolution, node.samples.max_x, node.samples.min_z)];
    const float h01 = heights[sample_index(resolution, node.samples.min_x, node.samples.max_z)];
    const float h11 = heights[sample_index(resolution, node.samples.max_x, node.samples.max_z)];
    for (std::uint32_t z = node.samples.min_z; z <= node.samples.max_z; ++z)
        for (std::uint32_t x = node.samples.min_x; x <= node.samples.max_x; ++x)
        {
            const float value = heights[sample_index(resolution, x, z)];
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            const float tx = static_cast<float>(x - node.samples.min_x) / static_cast<float>(extent_x);
            const float tz = static_cast<float>(z - node.samples.min_z) / static_cast<float>(extent_z);
            const float approximation = std::lerp(std::lerp(h00, h10, tx), std::lerp(h01, h11, tx), tz);
            error = std::max(error, std::abs(value - approximation));
        }
    node.minimum_height = minimum;
    node.maximum_height = maximum;
    node.geometric_error = error;
    const float half_width = width * 0.5f;
    const float half_depth = depth * 0.5f;
    const float subdivisions = static_cast<float>(resolution - 1u);
    const float min_x = -half_width + width * static_cast<float>(node.samples.min_x) / subdivisions;
    const float max_x = -half_width + width * static_cast<float>(node.samples.max_x) / subdivisions;
    const float min_z = -half_depth + depth * static_cast<float>(node.samples.min_z) / subdivisions;
    const float max_z = -half_depth + depth * static_cast<float>(node.samples.max_z) / subdivisions;
    node.local_bounds = {geometric::point3f{min_x, minimum, min_z}, geometric::point3f{max_x, maximum, max_z}};
}

std::uint32_t build_node(terrain_hierarchy& hierarchy, std::span<const float> heights, std::uint32_t resolution,
                         float width, float depth, terrain_sample_region region, std::uint32_t depth_level,
                         const terrain_lod_settings& settings)
{
    const auto index = static_cast<std::uint32_t>(hierarchy.nodes.size());
    hierarchy.nodes.push_back({.samples = region, .depth = depth_level});
    const auto quads_x = region.max_x - region.min_x;
    const auto quads_z = region.max_z - region.min_z;
    const bool depth_limited =
        settings.maximum_hierarchy_depth != 0u && depth_level >= settings.maximum_hierarchy_depth;
    if (!depth_limited && (quads_x > settings.patch_quads || quads_z > settings.patch_quads))
    {
        const auto mid_x = region.min_x + quads_x / 2u;
        const auto mid_z = region.min_z + quads_z / 2u;
        const std::array<terrain_sample_region, 4> regions{{{region.min_x, region.min_z, mid_x, mid_z},
                                                            {mid_x, region.min_z, region.max_x, mid_z},
                                                            {region.min_x, mid_z, mid_x, region.max_z},
                                                            {mid_x, mid_z, region.max_x, region.max_z}}};
        float child_error{};
        for (std::size_t child = 0; child < regions.size(); ++child)
        {
            if (regions[child].min_x == regions[child].max_x || regions[child].min_z == regions[child].max_z) continue;
            const auto child_index =
                build_node(hierarchy, heights, resolution, width, depth, regions[child], depth_level + 1u, settings);
            hierarchy.nodes[index].children[child] = child_index;
            child_error = std::max(child_error, hierarchy.nodes[child_index].geometric_error);
        }
        calculate_node(hierarchy.nodes[index], heights, resolution, width, depth, child_error);
    }
    else
    {
        ++hierarchy.leaf_count;
        calculate_node(hierarchy.nodes[index], heights, resolution, width, depth, 0.0f);
    }
    hierarchy.maximum_depth = std::max(hierarchy.maximum_depth, depth_level);
    return index;
}

geometric::box3f transform_bounds(const geometric::box3f& local, const math::matrix4f& model)
{
    auto corner = [&](std::uint32_t value)
    {
        return math::vector3f{(value & 1u) ? local.max[0] : local.min[0], (value & 2u) ? local.max[1] : local.min[1],
                              (value & 4u) ? local.max[2] : local.min[2]};
    };
    auto point = math::transform_point(model, corner(0));
    geometric::box3f result{geometric::point3f{point}, geometric::point3f{point}};
    for (std::uint32_t i = 1; i < 8; ++i)
        result = geometric::expand(result, geometric::point3f{math::transform_point(model, corner(i))});
    return result;
}

float distance_to_box(const math::vector3f& point, const geometric::box3f& bounds)
{
    const auto nearest = geometric::closest_point(bounds, geometric::point3f{point});
    return std::sqrt(math::length_squared(nearest.as_vector() - point));
}

bool adjacent(const terrain_sample_region& a, const terrain_sample_region& b) noexcept
{
    const bool vertical = (a.max_x == b.min_x || b.max_x == a.min_x) && a.min_z < b.max_z && b.min_z < a.max_z;
    const bool horizontal = (a.max_z == b.min_z || b.max_z == a.min_z) && a.min_x < b.max_x && b.min_x < a.max_x;
    return vertical || horizontal;
}
} // namespace

terrain_hierarchy build_terrain_hierarchy(std::span<const float> heights, std::uint32_t sample_resolution, float width,
                                          float depth, const terrain_lod_settings& input_settings)
{
    terrain_hierarchy result;
    if (sample_resolution < 2u || heights.size() != static_cast<std::size_t>(sample_resolution) * sample_resolution ||
        width <= 0.0f || depth <= 0.0f)
        return result;
    auto settings = input_settings;
    if (settings.patch_quads != 16u && settings.patch_quads != 32u && settings.patch_quads != 64u)
        settings.patch_quads = 32u;
    result.patch_quads = settings.patch_quads;
    result.root = build_node(result, heights, sample_resolution, width, depth,
                             {0u, 0u, sample_resolution - 1u, sample_resolution - 1u}, 0u, settings);
    return result;
}

terrain_gpu_hierarchy make_terrain_gpu_hierarchy(const terrain_hierarchy& hierarchy)
{
    terrain_gpu_hierarchy result{.root = hierarchy.root,
                                 .leaf_count = hierarchy.leaf_count,
                                 .maximum_depth = hierarchy.maximum_depth,
                                 .patch_quads = hierarchy.patch_quads};
    result.nodes.reserve(hierarchy.nodes.size());
    for (const auto& node : hierarchy.nodes)
    {
        gpu_terrain_node_record packed{};
        for (std::uint32_t component = 0; component < 3; ++component)
        {
            packed.bounds_min[component] = node.local_bounds.min[component];
            packed.bounds_max[component] = node.local_bounds.max[component];
        }
        packed.bounds_min[3] = node.minimum_height;
        packed.bounds_max[3] = node.maximum_height;
        packed.samples[0] = node.samples.min_x;
        packed.samples[1] = node.samples.min_z;
        packed.samples[2] = node.samples.max_x;
        packed.samples[3] = node.samples.max_z;
        std::copy(node.children.begin(), node.children.end(), packed.children);
        packed.geometric_error = node.geometric_error;
        packed.depth = node.depth;
        packed.leaf = node.leaf() ? 1u : 0u;
        result.nodes.push_back(packed);
    }
    if (result.root >= result.nodes.size()) result.root = invalid_terrain_node;
    return result;
}

bool update_terrain_hierarchy(terrain_hierarchy& hierarchy, std::span<const float> heights,
                              std::uint32_t sample_resolution, float width, float depth,
                              terrain_sample_region dirty_region, const terrain_lod_settings& settings)
{
    if (hierarchy.root >= hierarchy.nodes.size() || !dirty_region.valid() ||
        heights.size() != static_cast<std::size_t>(sample_resolution) * sample_resolution)
        return false;
    dirty_region.max_x = std::min(dirty_region.max_x, sample_resolution - 1u);
    dirty_region.max_z = std::min(dirty_region.max_z, sample_resolution - 1u);
    for (auto it = hierarchy.nodes.rbegin(); it != hierarchy.nodes.rend(); ++it)
    {
        if (!overlaps(it->samples, dirty_region)) continue;
        float inherited{};
        for (const auto child : it->children)
            if (child != invalid_terrain_node) inherited = std::max(inherited, hierarchy.nodes[child].geometric_error);
        calculate_node(*it, heights, sample_resolution, width, depth, inherited);
    }
    (void)settings;
    return true;
}

terrain_selection_result select_terrain_patches(terrain_handle terrain, const terrain_hierarchy& hierarchy,
                                                const math::matrix4f& model, const render_camera& camera,
                                                float geometry_error_threshold, float terrain_error_bias,
                                                terrain_selection_scratch* scratch)
{
    terrain_selection_result result;
    result.statistics.hierarchy_nodes = static_cast<std::uint32_t>(hierarchy.nodes.size());
    if (hierarchy.root >= hierarchy.nodes.size()) return result;
    const auto frustum = make_view_frustum(camera.view_projection);
    const float sx = math::length(math::vector3f{model(0, 0), model(1, 0), model(2, 0)});
    const float sy = math::length(math::vector3f{model(0, 1), model(1, 1), model(2, 1)});
    const float sz = math::length(math::vector3f{model(0, 2), model(1, 2), model(2, 2)});
    const float scale = std::max({sx, sy, sz, 0.0001f});
    const bool perspective = std::abs(camera.projection(3, 3)) < 0.5f;
    const float projection_scale = std::abs(camera.projection(1, 1)) * 0.5f * camera.render_height;
    std::vector<std::uint32_t> local_stack;
    auto& stack = scratch ? scratch->traversal_stack : local_stack;
    stack.clear();
    stack.push_back(hierarchy.root);
    while (!stack.empty())
    {
        const auto index = stack.back();
        stack.pop_back();
        const auto& node = hierarchy.nodes[index];
        const auto world_bounds = transform_bounds(node.local_bounds, model);
        if (!intersects(frustum, world_bounds))
        {
            ++result.statistics.culled_nodes;
            continue;
        }
        const float distance = std::max(distance_to_box(camera.position, world_bounds), camera.near_plane);
        const float projected =
            node.geometric_error * scale * projection_scale * terrain_error_bias / (perspective ? distance : 1.0f);
        float split_threshold = std::max(geometry_error_threshold, 0.01f);
        if (scratch)
        {
            const bool was_selected = std::find(scratch->previous_nodes.begin(), scratch->previous_nodes.end(),
                                                index) != scratch->previous_nodes.end();
            bool child_was_selected{};
            for (const auto child : node.children)
                child_was_selected |= child != invalid_terrain_node &&
                                      std::find(scratch->previous_nodes.begin(), scratch->previous_nodes.end(),
                                                child) != scratch->previous_nodes.end();
            if (was_selected)
                split_threshold *= 1.1f;
            else if (child_was_selected)
                split_threshold *= 0.9f;
        }
        if (!node.leaf() && projected > split_threshold)
        {
            for (auto it = node.children.rbegin(); it != node.children.rend(); ++it)
                if (*it != invalid_terrain_node) stack.push_back(*it);
        }
        else
            result.patches.push_back({terrain, index, node.samples, node.depth, 0u, projected});
    }

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (std::size_t a = 0; a < result.patches.size() && !changed; ++a)
            for (std::size_t b = a + 1; b < result.patches.size(); ++b)
            {
                if (!adjacent(result.patches[a].samples, result.patches[b].samples)) continue;
                const auto da = result.patches[a].lod;
                const auto db = result.patches[b].lod;
                if (da + 1u >= db && db + 1u >= da) continue;
                const auto coarse = da < db ? a : b;
                const auto node_index = result.patches[coarse].node_index;
                const auto node = hierarchy.nodes[node_index];
                result.patches.erase(result.patches.begin() + static_cast<std::ptrdiff_t>(coarse));
                for (const auto child : node.children)
                    if (child != invalid_terrain_node)
                        result.patches.push_back(
                            {terrain, child, hierarchy.nodes[child].samples, hierarchy.nodes[child].depth, 0u, 0.0f});
                changed = true;
                break;
            }
    }

    for (auto& patch : result.patches)
        for (const auto& neighbor : result.patches)
        {
            if (patch.node_index == neighbor.node_index || neighbor.lod + 1u != patch.lod) continue;
            if (patch.samples.min_x == neighbor.samples.max_x && patch.samples.min_z < neighbor.samples.max_z &&
                neighbor.samples.min_z < patch.samples.max_z)
                patch.stitch_mask |= 1u;
            if (patch.samples.max_x == neighbor.samples.min_x && patch.samples.min_z < neighbor.samples.max_z &&
                neighbor.samples.min_z < patch.samples.max_z)
                patch.stitch_mask |= 2u;
            if (patch.samples.min_z == neighbor.samples.max_z && patch.samples.min_x < neighbor.samples.max_x &&
                neighbor.samples.min_x < patch.samples.max_x)
                patch.stitch_mask |= 4u;
            if (patch.samples.max_z == neighbor.samples.min_z && patch.samples.min_x < neighbor.samples.max_x &&
                neighbor.samples.min_x < patch.samples.max_x)
                patch.stitch_mask |= 8u;
        }
    result.statistics.selected_patches = static_cast<std::uint32_t>(result.patches.size());
    for (const auto& patch : result.patches)
    {
        const auto quads = std::min(hierarchy.patch_quads, patch.samples.max_x - patch.samples.min_x) *
                           std::min(hierarchy.patch_quads, patch.samples.max_z - patch.samples.min_z);
        result.statistics.rendered_triangles += static_cast<std::uint64_t>(quads) * 2u;
        ++result.statistics.patches_per_lod[std::min<std::size_t>(patch.lod, 15u)];
    }
    if (scratch)
    {
        scratch->previous_nodes.clear();
        scratch->previous_nodes.reserve(result.patches.size());
        for (const auto& patch : result.patches)
            scratch->previous_nodes.push_back(patch.node_index);
    }
    return result;
}

bounded_terrain_selection select_terrain_patches_bounded(
    terrain_handle terrain, const terrain_hierarchy& hierarchy, const math::matrix4f& model,
    const render_camera& camera, float geometry_error_threshold, std::uint32_t capacity, float terrain_error_bias,
    terrain_selection_scratch* scratch)
{
    bounded_terrain_selection result;
    result.capacity = capacity;
    result.selection = select_terrain_patches(terrain, hierarchy, model, camera, geometry_error_threshold,
                                              terrain_error_bias, scratch);
    if (result.selection.patches.size() <= capacity) return result;
    result.overflowed = true;
    result.use_conventional_fallback = true;
    result.selection.patches.clear();
    result.selection.statistics.selected_patches = 0u;
    result.selection.statistics.rendered_triangles = 0u;
    result.selection.statistics.patches_per_lod.fill(0u);
    return result;
}

std::vector<std::uint32_t> make_terrain_patch_indices(std::uint32_t patch_quads, std::uint8_t stitch_mask)
{
    if (patch_quads == 0u) return {};
    const auto width = patch_quads + 1u;
    auto remap = [&](std::uint32_t x, std::uint32_t z)
    {
        if ((stitch_mask & 1u) && x == 0u && (z & 1u)) --z;
        if ((stitch_mask & 2u) && x == patch_quads && (z & 1u)) --z;
        if ((stitch_mask & 4u) && z == 0u && (x & 1u)) --x;
        if ((stitch_mask & 8u) && z == patch_quads && (x & 1u)) --x;
        return z * width + x;
    };
    std::vector<std::uint32_t> result;
    result.reserve(static_cast<std::size_t>(patch_quads) * patch_quads * 6u);
    for (std::uint32_t z = 0; z < patch_quads; ++z)
        for (std::uint32_t x = 0; x < patch_quads; ++x)
        {
            const std::array<std::uint32_t, 6> indices{remap(x, z), remap(x + 1u, z),      remap(x + 1u, z + 1u),
                                                       remap(x, z), remap(x + 1u, z + 1u), remap(x, z + 1u)};
            for (std::size_t triangle = 0; triangle < 6; triangle += 3)
                if (indices[triangle] != indices[triangle + 1] && indices[triangle + 1] != indices[triangle + 2] &&
                    indices[triangle] != indices[triangle + 2])
                    result.insert(result.end(), indices.begin() + static_cast<std::ptrdiff_t>(triangle),
                                  indices.begin() + static_cast<std::ptrdiff_t>(triangle + 3));
        }
    return result;
}

} // namespace arc::render
