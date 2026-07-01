#include <arc/render/render_world.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace arc::render
{
namespace
{

frustum_plane normalize_plane(float x, float y, float z, float w)
{
    const float length = std::sqrt(x * x + y * y + z * z);
    if (length <= std::numeric_limits<float>::epsilon())
        return {};
    return {
        .normal = math::vector3f{ x / length, y / length, z / length },
        .distance = w / length
    };
}

float item_depth(const render_world_packet& packet, const render_item& item)
{
    const auto center = geometric::center(item.world_bounds);
    const auto clip = math::transform_point(packet.camera.view_projection, center.as_vector());
    return clip[2];
}

std::uint64_t batch_key(const render_item& item)
{
    return (static_cast<std::uint64_t>(item.material.generation) << 48u) |
        (static_cast<std::uint64_t>(item.material.index & 0xffffu) << 32u) |
        (static_cast<std::uint64_t>(item.mesh.generation) << 16u) |
        static_cast<std::uint64_t>(item.mesh.index & 0xffffu);
}

} // namespace

view_frustum make_view_frustum(const math::matrix4f& m)
{
    view_frustum result{};
    result.planes[0] = normalize_plane(m(3, 0) + m(0, 0), m(3, 1) + m(0, 1), m(3, 2) + m(0, 2), m(3, 3) + m(0, 3));
    result.planes[1] = normalize_plane(m(3, 0) - m(0, 0), m(3, 1) - m(0, 1), m(3, 2) - m(0, 2), m(3, 3) - m(0, 3));
    result.planes[2] = normalize_plane(m(3, 0) + m(1, 0), m(3, 1) + m(1, 1), m(3, 2) + m(1, 2), m(3, 3) + m(1, 3));
    result.planes[3] = normalize_plane(m(3, 0) - m(1, 0), m(3, 1) - m(1, 1), m(3, 2) - m(1, 2), m(3, 3) - m(1, 3));
    result.planes[4] = normalize_plane(m(3, 0) + m(2, 0), m(3, 1) + m(2, 1), m(3, 2) + m(2, 2), m(3, 3) + m(2, 3));
    result.planes[5] = normalize_plane(m(3, 0) - m(2, 0), m(3, 1) - m(2, 1), m(3, 2) - m(2, 2), m(3, 3) - m(2, 3));
    return result;
}

bool intersects(const view_frustum& frustum, const geometric::box3f& bounds)
{
    for (const auto& plane : frustum.planes)
    {
        const math::vector3f positive{
            plane.normal[0] >= 0.0f ? bounds.max[0] : bounds.min[0],
            plane.normal[1] >= 0.0f ? bounds.max[1] : bounds.min[1],
            plane.normal[2] >= 0.0f ? bounds.max[2] : bounds.min[2]
        };
        if (math::dot(plane.normal, positive) + plane.distance < 0.0f)
            return false;
    }
    return true;
}

std::uint64_t make_render_sort_key(scene_render_pass pass, material_handle material, mesh_handle mesh, float depth)
{
    const auto depth_bucket = static_cast<std::uint32_t>(std::clamp(depth, 0.0f, 1.0f) * 65535.0f);
    return (static_cast<std::uint64_t>(pass) << 56u) |
        (static_cast<std::uint64_t>(material.index & 0xfffu) << 44u) |
        (static_cast<std::uint64_t>(material.generation & 0xffu) << 36u) |
        (static_cast<std::uint64_t>(mesh.index & 0xfffu) << 24u) |
        (static_cast<std::uint64_t>(mesh.generation & 0xffu) << 16u) |
        depth_bucket;
}

void prepare_render_world(render_world_packet& packet, const render_world_prepare_options& options)
{
    packet.visible_items.clear();
    packet.instance_batches.clear();
    packet.indirect_draws.clear();
    packet.culled_item_count = 0;

    const auto frustum = make_view_frustum(packet.camera.view_projection);
    for (std::uint32_t index = 0; index < packet.items.size(); ++index)
    {
        auto& item = packet.items[index];
        const bool layer_visible = (item.render_layer_mask & options.render_layer_mask) != 0;
        const bool bounds_visible = !options.enable_frustum_culling || intersects(frustum, item.world_bounds);
        if (!item.visible || !item.mesh.valid() || !layer_visible || !bounds_visible)
        {
            ++packet.culled_item_count;
            continue;
        }

        const auto pass = item.transparent ? scene_render_pass::forward_transparent : scene_render_pass::gbuffer;
        item.sort_key = make_render_sort_key(pass, item.material, item.mesh, item_depth(packet, item));
        packet.visible_items.push_back(index);
    }

    std::sort(packet.visible_items.begin(), packet.visible_items.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
        const auto& left = packet.items[lhs];
        const auto& right = packet.items[rhs];
        if (left.transparent != right.transparent)
            return !left.transparent;
        if (left.transparent)
            return item_depth(packet, left) > item_depth(packet, right);
        return left.sort_key < right.sort_key;
    });

    if (options.enable_instancing)
    {
        std::uint32_t batch_start = 0;
        while (batch_start < packet.visible_items.size())
        {
            const auto first_item_index = packet.visible_items[batch_start];
            const auto& first = packet.items[first_item_index];
            const auto key = batch_key(first);
            std::uint32_t batch_end = batch_start + 1;
            while (batch_end < packet.visible_items.size() && batch_key(packet.items[packet.visible_items[batch_end]]) == key)
                ++batch_end;

            packet.instance_batches.push_back({
                .mesh = first.mesh,
                .material = first.material,
                .pass = first.transparent ? scene_render_pass::forward_transparent : scene_render_pass::gbuffer,
                .first_item = batch_start,
                .item_count = batch_end - batch_start,
                .sort_key = first.sort_key
            });
            batch_start = batch_end;
        }
    }

    if (options.enable_indirect_draws)
    {
        packet.indirect_draws.reserve(packet.instance_batches.empty() ? packet.visible_items.size() : packet.instance_batches.size());
        if (!packet.instance_batches.empty())
        {
            for (const auto& batch : packet.instance_batches)
            {
                packet.indirect_draws.push_back({
                    .index_count = 0,
                    .instance_count = batch.item_count,
                    .first_instance = batch.first_item
                });
            }
        }
        else
        {
            for (std::uint32_t index = 0; index < packet.visible_items.size(); ++index)
                packet.indirect_draws.push_back({ .instance_count = 1, .first_instance = index });
        }
    }
}

render_graph make_scene_draw_graph(std::string_view target_name)
{
    std::string target(target_name);
    if (target.empty())
        target = "viewport";

    render_graph graph;
    graph.add_resource({ .name = target, .kind = render_resource_kind::color_texture, .persistent = true });
    graph.add_resource({ .name = "scene_depth", .kind = render_resource_kind::depth_texture, .persistent = true });
    graph.add_resource({ .name = "scene_color", .kind = render_resource_kind::color_texture, .format = "rgba16f", .persistent = true });
    graph.add_resource({ .name = "gbuffer_albedo", .kind = render_resource_kind::color_texture, .format = "rgba16f" });
    graph.add_resource({ .name = "gbuffer_normal", .kind = render_resource_kind::color_texture, .format = "rgba16f" });
    graph.add_resource({ .name = "gbuffer_material", .kind = render_resource_kind::color_texture, .format = "rgba16f" });
    graph.add_resource({ .name = "gbuffer_motion", .kind = render_resource_kind::color_texture, .format = "rg16f" });
    graph.add_resource({ .name = "gbuffer_object_id", .kind = render_resource_kind::color_texture, .format = "r32ui", .persistent = true });
    graph.add_resource({ .name = "editor_picking", .kind = render_resource_kind::color_texture, .persistent = true });
    graph.add_resource({ .name = "selection_mask", .kind = render_resource_kind::color_texture, .persistent = true });
    graph.add_resource({ .name = "directional_shadow_atlas", .kind = render_resource_kind::depth_texture, .persistent = true });
    graph.add_resource({ .name = "spot_shadow_atlas", .kind = render_resource_kind::depth_texture, .persistent = true });
    graph.add_resource({ .name = "point_shadow_cubemaps", .kind = render_resource_kind::depth_texture, .persistent = true });
    graph.add_resource({ .name = "clustered_light_grid", .kind = render_resource_kind::buffer, .persistent = true });

    graph.add_pass({
        .name = "directional shadow cascades",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .writes = { { .resource = "directional_shadow_atlas", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    graph.add_pass({
        .name = "spot shadow maps",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .writes = { { .resource = "spot_shadow_atlas", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    graph.add_pass({
        .name = "point shadow cubemaps",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .writes = { { .resource = "point_shadow_cubemaps", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    graph.add_pass({
        .name = "sky atmosphere",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::clear,
        .writes = { { .resource = target, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    graph.add_pass({
        .name = "depth prepass",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::depth_prepass,
        .writes = { { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    graph.add_pass({
        .name = "gbuffer pass",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::gbuffer,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment },
            { .resource = "directional_shadow_atlas", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled }
        },
        .writes = {
            { .resource = "gbuffer_albedo", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .resource = "gbuffer_normal", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .resource = "gbuffer_material", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .resource = "gbuffer_motion", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .resource = "gbuffer_object_id", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear }
        }
    });
    graph.add_pass({
        .name = "clustered light culling",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled }
        },
        .writes = {
            { .resource = "clustered_light_grid", .kind = render_resource_kind::buffer, .usage = render_resource_usage::storage, .write = true }
        }
    });
    graph.add_pass({
        .name = "deferred lighting",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::lighting,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled },
            { .resource = "gbuffer_albedo", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .resource = "gbuffer_normal", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .resource = "gbuffer_material", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .resource = "gbuffer_motion", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .resource = "gbuffer_object_id", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .resource = "directional_shadow_atlas", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled },
            { .resource = "spot_shadow_atlas", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled },
            { .resource = "point_shadow_cubemaps", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled },
            { .resource = "clustered_light_grid", .kind = render_resource_kind::buffer, .usage = render_resource_usage::sampled }
        },
        .writes = {
            { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear }
        }
    });
    graph.add_pass({
        .name = "terrain pass",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment },
            { .resource = "directional_shadow_atlas", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled },
            { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment },
            { .resource = "gbuffer_albedo", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled }
        },
        .writes = { { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load } }
    });
    graph.add_pass({
        .name = "vegetation pass",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment },
            { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment }
        },
        .writes = { { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load } }
    });
    graph.add_pass({
        .name = "water/forward transparent pass",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment },
            { .resource = "gbuffer_albedo", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment }
        },
        .writes = { { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load } }
    });
    graph.add_pass({
        .name = "height fog",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled },
            { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled }
        },
        .writes = { { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load } }
    });
    graph.add_pass({
        .name = "editor picking pass",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment },
            { .resource = "gbuffer_object_id", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled }
        },
        .writes = { { .resource = "editor_picking", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    graph.add_pass({
        .name = "selection outline pass",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::custom,
        .reads = {
            { .resource = "scene_depth", .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled },
            { .resource = "editor_picking", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled }
        },
        .writes = {
            { .resource = "selection_mask", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load }
        }
    });
    graph.add_pass({
        .name = "present viewport",
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::present,
        .reads = {
            { .resource = "scene_color", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .resource = "selection_mask", .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled }
        },
        .writes = { { .resource = target, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    return graph;
}

} // namespace arc::render
