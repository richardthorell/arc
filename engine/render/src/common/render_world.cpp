#include <arc/render/render_world.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace arc::render
{
namespace
{

constexpr std::uint32_t maximum_gpu_scene_instances = 1u << 20u;
constexpr std::uint32_t maximum_gpu_draw_bins = 4096u;
constexpr std::uint32_t gpu_scene_instance_stride = 224u;
constexpr std::uint32_t indirect_command_stride = 20u;

frustum_plane normalize_plane(float x, float y, float z, float w)
{
    const float length = std::sqrt(x * x + y * y + z * z);
    if (length <= std::numeric_limits<float>::epsilon()) return {};
    return {.normal = math::vector3f{x / length, y / length, z / length}, .distance = w / length};
}

float item_depth(const render_world_packet& packet, const render_item& item)
{
    const auto center = geometric::center(item.world_bounds);
    const auto clip = math::transform_point(packet.camera.view_projection, center.as_vector());
    return clip[2];
}

float item_depth(const render_world_packet& packet, const virtual_render_item& item)
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
        const math::vector3f positive{plane.normal[0] >= 0.0f ? bounds.max[0] : bounds.min[0],
                                      plane.normal[1] >= 0.0f ? bounds.max[1] : bounds.min[1],
                                      plane.normal[2] >= 0.0f ? bounds.max[2] : bounds.min[2]};
        if (math::dot(plane.normal, positive) + plane.distance < 0.0f) return false;
    }
    return true;
}

std::uint64_t make_render_sort_key(scene_render_pass pass, material_handle material, mesh_handle mesh, float depth)
{
    const auto depth_bucket = static_cast<std::uint32_t>(std::clamp(depth, 0.0f, 1.0f) * 65535.0f);
    return (static_cast<std::uint64_t>(pass) << 56u) | (static_cast<std::uint64_t>(material.index & 0xfffu) << 44u) |
           (static_cast<std::uint64_t>(material.generation & 0xffu) << 36u) |
           (static_cast<std::uint64_t>(mesh.index & 0xfffu) << 24u) |
           (static_cast<std::uint64_t>(mesh.generation & 0xffu) << 16u) | depth_bucket;
}

void prepare_render_world(render_world_packet& packet, const render_world_prepare_options& options)
{
    packet.visible_items.clear();
    packet.visible_virtual_items.clear();
    packet.instance_batches.clear();
    packet.indirect_draws.clear();
    packet.culled_item_count = 0;
    packet.culled_virtual_cluster_count = 0;

    // Keep the reference visibility output even for GPU-driven views. Backends
    // with indirect execution ignore these lists, while limited backends can
    // fall back without re-extracting the scene or changing its representation.
    // The reference output is also used to validate GPU culling in tests and
    // development captures.

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

    for (std::uint32_t index = 0; index < packet.virtual_items.size(); ++index)
    {
        auto& item = packet.virtual_items[index];
        const bool layer_visible = (item.render_layer_mask & options.render_layer_mask) != 0;
        const bool bounds_visible = !options.enable_frustum_culling || intersects(frustum, item.world_bounds);
        if (!item.visible || !item.mesh.valid() || !layer_visible || !bounds_visible)
        {
            ++packet.culled_virtual_cluster_count;
            continue;
        }

        item.sort_key =
            make_render_sort_key(scene_render_pass::gbuffer, item.material, item.mesh, item_depth(packet, item));
        packet.visible_virtual_items.push_back(index);
    }

    std::sort(packet.visible_items.begin(), packet.visible_items.end(),
              [&](std::uint32_t lhs, std::uint32_t rhs)
              {
                  const auto& left = packet.items[lhs];
                  const auto& right = packet.items[rhs];
                  if (left.transparent != right.transparent) return !left.transparent;
                  if (left.transparent) return item_depth(packet, left) > item_depth(packet, right);
                  return left.sort_key < right.sort_key;
              });

    std::sort(packet.visible_virtual_items.begin(), packet.visible_virtual_items.end(),
              [&](std::uint32_t lhs, std::uint32_t rhs)
              {
                  const auto& left = packet.virtual_items[lhs];
                  const auto& right = packet.virtual_items[rhs];
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
            while (batch_end < packet.visible_items.size() &&
                   batch_key(packet.items[packet.visible_items[batch_end]]) == key)
                ++batch_end;

            packet.instance_batches.push_back(
                {.mesh = first.mesh,
                 .material = first.material,
                 .pass = first.transparent ? scene_render_pass::forward_transparent : scene_render_pass::gbuffer,
                 .first_item = batch_start,
                 .item_count = batch_end - batch_start,
                 .sort_key = first.sort_key});
            batch_start = batch_end;
        }
    }

    if (options.enable_indirect_draws)
    {
        packet.indirect_draws.reserve(packet.instance_batches.empty() ? packet.visible_items.size()
                                                                      : packet.instance_batches.size());
        if (!packet.instance_batches.empty())
        {
            for (const auto& batch : packet.instance_batches)
            {
                packet.indirect_draws.push_back(
                    {.index_count = 0, .instance_count = batch.item_count, .first_instance = batch.first_item});
            }
        }
        else
        {
            for (std::uint32_t index = 0; index < packet.visible_items.size(); ++index)
                packet.indirect_draws.push_back({.instance_count = 1, .first_instance = index});
        }
    }
}

render_graph make_scene_draw_graph(std::string_view target_name, render_path path, bool editor_view)
{
    resolved_render_config config{};
    config.path = path;
    config.quality = path == render_path::forward_plus ? render_quality_tier::low : render_quality_tier::medium;
    return make_scene_draw_graph(target_name, config, editor_view);
}

render_graph make_scene_draw_graph(std::string_view target_name, const resolved_render_config& config, bool editor_view)
{
    world_environment_data environment;
    environment.enabled = true;
    environment.sky_visible = true;
    environment.atmosphere.enabled = true;
    return make_scene_draw_graph(target_name, config, editor_view, environment);
}

render_graph make_scene_draw_graph(std::string_view target_name, const resolved_render_config& config, bool editor_view,
                                   const world_environment_data& environment)
{
    std::string target(target_name);
    if (target.empty()) target = "viewport";

    render_graph graph;
    const auto viewport = graph.add_resource({.name = target,
                                              .kind = render_resource_kind::color_texture,
                                              .format = render_format::rgba16_float,
                                              .persistent = true});
    const auto depth = graph.add_resource({.name = "scene_depth",
                                           .kind = render_resource_kind::depth_texture,
                                           .width_scale = config.render_scale,
                                           .height_scale = config.render_scale,
                                           .format = render_format::d32_float,
                                           .persistent = true});
    const auto scene_color = graph.add_resource({.name = "scene_color",
                                                 .kind = render_resource_kind::color_texture,
                                                 .width_scale = config.render_scale,
                                                 .height_scale = config.render_scale,
                                                 .format = render_format::rgba16_float,
                                                 .persistent = true});
    const auto directional_static_shadows =
        graph.add_resource({.name = "directional_static_shadows",
                            .kind = render_resource_kind::depth_texture,
                            .extent = {config.directional_shadow_resolution, config.directional_shadow_resolution, 1},
                            .extent_mode = render_extent_mode::absolute,
                            .format = render_format::d32_float,
                            .array_layers = config.directional_shadow_cascades,
                            .persistent = true});
    const auto directional_dynamic_shadows =
        graph.add_resource({.name = "directional_dynamic_shadows",
                            .kind = render_resource_kind::depth_texture,
                            .extent = {config.directional_shadow_resolution, config.directional_shadow_resolution, 1},
                            .extent_mode = render_extent_mode::absolute,
                            .format = render_format::d32_float,
                            .array_layers = config.directional_shadow_cascades,
                            .persistent = true});
    const auto local_shadow_atlas =
        graph.add_resource({.name = "local_shadow_atlas",
                            .kind = render_resource_kind::depth_texture,
                            .extent = {config.local_shadow_atlas_resolution, config.local_shadow_atlas_resolution, 1},
                            .extent_mode = render_extent_mode::absolute,
                            .format = render_format::d32_float,
                            .persistent = true});

    render_graph_resource_handle gpu_scene_instances{};
    render_graph_resource_handle gpu_visible_instances{};
    render_graph_resource_handle gpu_indirect_commands{};
    render_graph_resource_handle gpu_indirect_count{};
    render_graph_resource_handle depth_pyramid{};
    render_graph_resource_handle virtual_visible_clusters{};
    render_graph_resource_handle virtual_shadow_clusters{};
    render_graph_resource_handle virtual_visibility{};

    const bool needs_depth_pyramid = config.features.hzb_occlusion || config.screen_space_shadows ||
                                     config.features.screen_space_gi || config.features.screen_space_reflections ||
                                     config.features.software_gi || config.features.software_reflections ||
                                     config.features.hardware_gi || config.features.hardware_reflections;
    if (needs_depth_pyramid)
    {
        depth_pyramid = graph.add_resource(
            {.name = "view_depth_pyramid",
             .kind = render_resource_kind::color_texture,
             .width_scale = config.render_scale,
             .height_scale = config.render_scale,
             .format = render_format::rg32_float,
             .mip_levels = 0,
             .persistent_key = "view.depth_hzb",
             .history_length = 2,
             .history_reset = render_history_reset::camera_cut | render_history_reset::resize |
                              render_history_reset::render_scale_change | render_history_reset::world_epoch_change |
                              render_history_reset::projection_change,
             // This generation is consumed by next-frame occlusion even when
             // no current-frame screen-space pass samples it.
             .exported = true});
    }
    render_graph_resource_handle virtual_encoded_depth{};
    const auto compute_queue = config.features.async_compute ? render_queue_type::compute : render_queue_type::graphics;

    if (config.features.gpu_driven_rendering)
    {
        gpu_scene_instances = graph.add_resource(
            {.name = "gpu_scene_instances",
             .kind = render_resource_kind::buffer,
             .byte_size = static_cast<std::uint64_t>(maximum_gpu_scene_instances) * gpu_scene_instance_stride,
             .element_stride = gpu_scene_instance_stride,
             .persistent_key = "gpu_scene.instances",
             .persistent = true});
        const auto candidate_instances = graph.add_resource(
            {.name = "gpu_visibility_candidates",
             .kind = render_resource_kind::buffer,
             .byte_size = static_cast<std::uint64_t>(maximum_gpu_scene_instances) * sizeof(std::uint32_t),
             .element_stride = sizeof(std::uint32_t)});
        gpu_visible_instances = graph.add_resource(
            {.name = "gpu_visible_instances",
             .kind = render_resource_kind::buffer,
             .byte_size = static_cast<std::uint64_t>(maximum_gpu_scene_instances) * sizeof(std::uint32_t),
             .element_stride = sizeof(std::uint32_t)});
        const auto visibility_counters = graph.add_resource({.name = "gpu_visibility_counters",
                                                             .kind = render_resource_kind::buffer,
                                                             .byte_size = 64,
                                                             .element_stride = sizeof(std::uint32_t)});
        const auto draw_bin_counts = graph.add_resource({.name = "gpu_draw_bin_counts",
                                                         .kind = render_resource_kind::buffer,
                                                         .byte_size = maximum_gpu_draw_bins * sizeof(std::uint32_t),
                                                         .element_stride = sizeof(std::uint32_t)});
        const auto draw_bin_offsets = graph.add_resource({.name = "gpu_draw_bin_offsets",
                                                          .kind = render_resource_kind::buffer,
                                                          .byte_size = maximum_gpu_draw_bins * sizeof(std::uint32_t),
                                                          .element_stride = sizeof(std::uint32_t)});
        gpu_indirect_commands = graph.add_resource(
            {.name = "gpu_indirect_commands",
             .kind = render_resource_kind::buffer,
             .byte_size = static_cast<std::uint64_t>(maximum_gpu_draw_bins) * indirect_command_stride,
             .element_stride = indirect_command_stride});
        gpu_indirect_count = graph.add_resource({.name = "gpu_indirect_count",
                                                 .kind = render_resource_kind::buffer,
                                                 .byte_size = sizeof(std::uint32_t),
                                                 .element_stride = sizeof(std::uint32_t)});
        graph.add_pass({.name = "GPU Scene upload",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::gpu_scene_upload,
                        .writes = {{.handle = gpu_scene_instances,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true}}});
        graph.add_pass({.name = "GPU visibility clear",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::gpu_visibility_clear,
                        .writes = {{.handle = visibility_counters,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true},
                                   {.handle = draw_bin_counts,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true},
                                   {.handle = gpu_indirect_count,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true}}});
        graph.add_pass({.name = "GPU frustum and distance culling",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::gpu_frustum_distance_cull,
                        .reads = {{.handle = gpu_scene_instances,
                                   .kind = render_resource_kind::buffer,
                                   .usage = render_resource_usage::storage_buffer}},
                        .writes = {{.handle = candidate_instances,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true},
                                   {.handle = visibility_counters,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true}}});
        if (depth_pyramid.valid())
        {
            graph.add_pass({.name = "GPU HZB occlusion culling",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::gpu_hzb_occlusion_cull,
                            .reads = {{.handle = candidate_instances,
                                       .kind = render_resource_kind::buffer,
                                       .usage = render_resource_usage::storage_buffer},
                                      {.handle = depth_pyramid,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled,
                                       .history = render_history_access::previous}},
                            .writes = {{.handle = gpu_visible_instances,
                                        .kind = render_resource_kind::buffer,
                                        .usage = render_resource_usage::storage_buffer,
                                        .write = true},
                                       {.handle = visibility_counters,
                                        .kind = render_resource_kind::buffer,
                                        .usage = render_resource_usage::storage_buffer,
                                        .write = true}}});
        }
        else
        {
            graph.add_pass({.name = "GPU visibility compaction",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::gpu_visibility_compact,
                            .reads = {{.handle = candidate_instances,
                                       .kind = render_resource_kind::buffer,
                                       .usage = render_resource_usage::storage_buffer}},
                            .writes = {{.handle = gpu_visible_instances,
                                        .kind = render_resource_kind::buffer,
                                        .usage = render_resource_usage::storage_buffer,
                                        .write = true}}});
        }
        graph.add_pass({.name = "GPU draw-bin count",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::gpu_draw_bin_count,
                        .reads = {{.handle = gpu_visible_instances,
                                   .kind = render_resource_kind::buffer,
                                   .usage = render_resource_usage::storage_buffer},
                                  {.handle = gpu_scene_instances,
                                   .kind = render_resource_kind::buffer,
                                   .usage = render_resource_usage::storage_buffer}},
                        .writes = {{.handle = draw_bin_counts,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true}}});
        graph.add_pass({.name = "GPU draw-bin prefix sum",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::gpu_draw_bin_prefix_sum,
                        .reads = {{.handle = draw_bin_counts,
                                   .kind = render_resource_kind::buffer,
                                   .usage = render_resource_usage::storage_buffer}},
                        .writes = {{.handle = draw_bin_offsets,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true}}});
        graph.add_pass({.name = "GPU indirect command generation",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::gpu_indirect_command_generation,
                        .reads = {{.handle = draw_bin_counts,
                                   .kind = render_resource_kind::buffer,
                                   .usage = render_resource_usage::storage_buffer},
                                  {.handle = draw_bin_offsets,
                                   .kind = render_resource_kind::buffer,
                                   .usage = render_resource_usage::storage_buffer},
                                  {.handle = gpu_visible_instances,
                                   .kind = render_resource_kind::buffer,
                                   .usage = render_resource_usage::storage_buffer}},
                        .writes = {{.handle = gpu_indirect_commands,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::indirect_buffer,
                                    .write = true},
                                   {.handle = gpu_indirect_count,
                                    .kind = render_resource_kind::buffer,
                                    .usage = render_resource_usage::storage_buffer,
                                    .write = true}}});

        if (config.features.virtual_geometry)
        {
            const auto virtual_metadata = graph.add_resource({.name = "virtual_geometry_metadata",
                                                              .kind = render_resource_kind::buffer,
                                                              .byte_size = 64ull * 1024ull * 1024ull,
                                                              .element_stride = 16,
                                                              .persistent_key = "virtual_geometry.metadata",
                                                              .imported = true,
                                                              .persistent = true});
            const auto virtual_page_table = graph.add_resource({.name = "virtual_geometry_page_table",
                                                                .kind = render_resource_kind::buffer,
                                                                .byte_size = 16ull * 1024ull * 1024ull,
                                                                .element_stride = sizeof(std::uint32_t),
                                                                .persistent_key = "virtual_geometry.page_table",
                                                                .imported = true,
                                                                .persistent = true});
            virtual_visible_clusters = graph.add_resource(
                {.name = "virtual_geometry_visible_clusters",
                 .kind = render_resource_kind::buffer,
                 .byte_size = static_cast<std::uint64_t>(maximum_gpu_scene_instances) * 64u * sizeof(std::uint32_t),
                 .element_stride = sizeof(std::uint32_t)});
            virtual_shadow_clusters = graph.add_resource(
                {.name = "virtual_geometry_shadow_clusters",
                 .kind = render_resource_kind::buffer,
                 .byte_size = static_cast<std::uint64_t>(maximum_gpu_scene_instances) * 32u * sizeof(std::uint32_t),
                 .element_stride = sizeof(std::uint32_t)});
            const auto virtual_page_requests = graph.add_resource({.name = "virtual_geometry_page_requests",
                                                                   .kind = render_resource_kind::buffer,
                                                                   .byte_size = 4096ull * 16ull,
                                                                   .element_stride = 16});
            const auto virtual_page_request_readback =
                graph.add_resource({.name = "virtual_geometry_page_request_readback",
                                    .kind = render_resource_kind::buffer,
                                    .byte_size = 4096ull * 16ull,
                                    .element_stride = 16,
                                    .persistent_key = "virtual_geometry.page_request_readback",
                                    .exported = true,
                                    .persistent = true});
            const auto virtual_cluster_bins = graph.add_resource({.name = "virtual_geometry_cluster_bins",
                                                                  .kind = render_resource_kind::buffer,
                                                                  .byte_size = 8ull * 1024ull * 1024ull,
                                                                  .element_stride = sizeof(std::uint32_t)});
            virtual_visibility = graph.add_resource({.name = "virtual_geometry_visibility",
                                                     .kind = render_resource_kind::color_texture,
                                                     .width_scale = config.render_scale,
                                                     .height_scale = config.render_scale,
                                                     .format = render_format::r32_uint});
            virtual_encoded_depth = graph.add_resource({.name = "virtual_geometry_encoded_depth",
                                                        .kind = render_resource_kind::color_texture,
                                                        .width_scale = config.render_scale,
                                                        .height_scale = config.render_scale,
                                                        .format = render_format::r32_uint});

            std::vector<render_resource_access> traversal_reads{{.handle = gpu_scene_instances,
                                                                 .kind = render_resource_kind::buffer,
                                                                 .usage = render_resource_usage::storage_buffer},
                                                                {.handle = virtual_metadata,
                                                                 .kind = render_resource_kind::buffer,
                                                                 .usage = render_resource_usage::storage_buffer},
                                                                {.handle = virtual_page_table,
                                                                 .kind = render_resource_kind::buffer,
                                                                 .usage = render_resource_usage::storage_buffer}};
            if (depth_pyramid.valid())
                traversal_reads.push_back({.handle = depth_pyramid,
                                           .kind = render_resource_kind::color_texture,
                                           .usage = render_resource_usage::sampled,
                                           .history = render_history_access::previous});
            graph.add_pass({.name = "virtual geometry hierarchy traversal",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::virtual_geometry_hierarchy_traversal,
                            .reads = std::move(traversal_reads),
                            .writes = {{.handle = virtual_visible_clusters,
                                        .kind = render_resource_kind::buffer,
                                        .usage = render_resource_usage::storage_buffer,
                                        .write = true},
                                       {.handle = virtual_page_requests,
                                        .kind = render_resource_kind::buffer,
                                        .usage = render_resource_usage::storage_buffer,
                                        .write = true}}});
            graph.add_pass({.name = "virtual geometry page requests",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::virtual_geometry_page_requests,
                            .reads = {{.handle = virtual_page_requests,
                                       .kind = render_resource_kind::buffer,
                                       .usage = render_resource_usage::storage_buffer}},
                            .writes = {{.handle = virtual_page_request_readback,
                                        .kind = render_resource_kind::buffer,
                                        .usage = render_resource_usage::transfer_dst,
                                        .write = true}}});
            graph.add_pass({.name = "virtual geometry shadow traversal",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::virtual_geometry_shadow_traversal,
                            .reads = {{.handle = virtual_metadata,
                                       .kind = render_resource_kind::buffer,
                                       .usage = render_resource_usage::storage_buffer},
                                      {.handle = virtual_page_table,
                                       .kind = render_resource_kind::buffer,
                                       .usage = render_resource_usage::storage_buffer}},
                            .writes = {{.handle = virtual_shadow_clusters,
                                        .kind = render_resource_kind::buffer,
                                        .usage = render_resource_usage::storage_buffer,
                                        .write = true}}});

            if (config.features.virtual_geometry_path == virtual_geometry_raster_path::mesh_shader)
            {
                graph.add_pass({.name = "virtual geometry mesh-shader visibility",
                                .kind = render_pass_kind::custom,
                                .builtin = builtin_render_pass::virtual_geometry_mesh_shader_visibility,
                                .reads = {{.handle = virtual_visible_clusters,
                                           .kind = render_resource_kind::buffer,
                                           .usage = render_resource_usage::storage_buffer},
                                          {.handle = virtual_metadata,
                                           .kind = render_resource_kind::buffer,
                                           .usage = render_resource_usage::storage_buffer}},
                                .writes = {{.handle = virtual_visibility,
                                            .kind = render_resource_kind::color_texture,
                                            .usage = render_resource_usage::storage,
                                            .write = true},
                                           {.handle = virtual_encoded_depth,
                                            .kind = render_resource_kind::color_texture,
                                            .usage = render_resource_usage::storage,
                                            .write = true}}});
            }
            else
            {
                graph.add_pass({.name = "virtual geometry cluster binning",
                                .queue = compute_queue,
                                .kind = render_pass_kind::compute,
                                .builtin = builtin_render_pass::virtual_geometry_cluster_binning,
                                .reads = {{.handle = virtual_visible_clusters,
                                           .kind = render_resource_kind::buffer,
                                           .usage = render_resource_usage::storage_buffer}},
                                .writes = {{.handle = virtual_cluster_bins,
                                            .kind = render_resource_kind::buffer,
                                            .usage = render_resource_usage::storage_buffer,
                                            .write = true}}});
                graph.add_pass({.name = "virtual geometry software depth",
                                .queue = compute_queue,
                                .kind = render_pass_kind::compute,
                                .builtin = builtin_render_pass::virtual_geometry_software_depth,
                                .reads = {{.handle = virtual_cluster_bins,
                                           .kind = render_resource_kind::buffer,
                                           .usage = render_resource_usage::storage_buffer},
                                          {.handle = virtual_metadata,
                                           .kind = render_resource_kind::buffer,
                                           .usage = render_resource_usage::storage_buffer}},
                                .writes = {{.handle = virtual_visibility,
                                            .kind = render_resource_kind::color_texture,
                                            .usage = render_resource_usage::storage,
                                            .write = true},
                                           {.handle = virtual_encoded_depth,
                                            .kind = render_resource_kind::color_texture,
                                            .usage = render_resource_usage::storage,
                                            .write = true}}});
            }
        }
    }

    const bool high_quality =
        config.quality == render_quality_tier::high || config.quality == render_quality_tier::ultra;
    const bool low_quality = config.quality == render_quality_tier::low;
    const std::uint32_t radiance_resolution = high_quality ? 512u : low_quality ? 128u : 256u;
    const std::uint32_t irradiance_resolution = high_quality ? 64u : low_quality ? 16u : 32u;
    const std::uint32_t brdf_resolution = low_quality ? 128u : 256u;
    render_graph_resource_handle environment_radiance{};
    render_graph_resource_handle environment_irradiance{};
    render_graph_resource_handle environment_specular{};
    render_graph_resource_handle brdf_lut{};

    render_graph_resource_handle sky_view{};
    if (environment.enabled && environment.sky_visible && environment.atmosphere.enabled &&
        environment.source == sky_source_mode::physical_atmosphere && config.quality != render_quality_tier::low)
    {
        const auto transmittance = graph.add_resource({.name = "atmosphere_transmittance",
                                                       .kind = render_resource_kind::color_texture,
                                                       .extent = {256, 64, 1},
                                                       .extent_mode = render_extent_mode::absolute,
                                                       .format = render_format::rgba16_float,
                                                       .persistent = true});
        const auto multi_scattering = graph.add_resource({.name = "atmosphere_multi_scattering",
                                                          .kind = render_resource_kind::color_texture,
                                                          .extent = {32, 32, 1},
                                                          .extent_mode = render_extent_mode::absolute,
                                                          .format = render_format::rgba16_float,
                                                          .persistent = true});
        sky_view = graph.add_resource({.name = "atmosphere_sky_view",
                                       .kind = render_resource_kind::color_texture,
                                       .extent = {192, 108, 1},
                                       .extent_mode = render_extent_mode::absolute,
                                       .format = render_format::rgba16_float,
                                       .persistent = true});
        graph.add_pass({.name = "atmosphere transmittance",
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::atmosphere_transmittance,
                        .writes = {{.handle = transmittance,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = "atmosphere multi scattering",
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::atmosphere_multi_scattering,
                        .reads = {{.handle = transmittance,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = multi_scattering,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = "atmosphere sky view",
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::atmosphere_sky_view,
                        .reads = {{.handle = transmittance,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = multi_scattering,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = sky_view,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
    }

    render_graph_resource_handle cloud_shadow{};
    if (environment.enabled && environment.clouds.enabled && environment.clouds.cast_shadows &&
        config.quality != render_quality_tier::low)
    {
        cloud_shadow = graph.add_resource({.name = "cloud_shadow",
                                           .kind = render_resource_kind::color_texture,
                                           .extent = {512, 512, 1},
                                           .extent_mode = render_extent_mode::absolute,
                                           .format = render_format::r8_unorm,
                                           .persistent = true});
        graph.add_pass({.name = "cloud shadow",
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::cloud_shadow,
                        .writes = {{.handle = cloud_shadow,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
    }

    const bool generate_ibl = environment.enabled && environment.affect_lighting && environment.lighting.enabled &&
                              environment.lighting.source != environment_lighting_source_mode::constant_color;
    if (generate_ibl)
    {
        environment_radiance = graph.add_resource({.name = "environment_radiance",
                                                   .kind = render_resource_kind::color_texture,
                                                   .extent = {radiance_resolution, radiance_resolution, 1},
                                                   .extent_mode = render_extent_mode::absolute,
                                                   .format = render_format::rgba16_float,
                                                   .mip_levels = high_quality  ? 10u
                                                                 : low_quality ? 8u
                                                                               : 9u,
                                                   .array_layers = 6,
                                                   .persistent = true});
        environment_irradiance = graph.add_resource({.name = "environment_irradiance",
                                                     .kind = render_resource_kind::color_texture,
                                                     .extent = {irradiance_resolution, irradiance_resolution, 1},
                                                     .extent_mode = render_extent_mode::absolute,
                                                     .format = render_format::rgba16_float,
                                                     .array_layers = 6,
                                                     .persistent = true});
        environment_specular = graph.add_resource({.name = "environment_specular",
                                                   .kind = render_resource_kind::color_texture,
                                                   .extent = {radiance_resolution, radiance_resolution, 1},
                                                   .extent_mode = render_extent_mode::absolute,
                                                   .format = render_format::rgba16_float,
                                                   .mip_levels = high_quality  ? 10u
                                                                 : low_quality ? 8u
                                                                               : 9u,
                                                   .array_layers = 6,
                                                   .persistent = true});
        brdf_lut = graph.add_resource({.name = "environment_brdf_lut",
                                       .kind = render_resource_kind::color_texture,
                                       .extent = {brdf_resolution, brdf_resolution, 1},
                                       .extent_mode = render_extent_mode::absolute,
                                       .format = render_format::rg16_float,
                                       .persistent = true});

        std::vector<render_resource_access> conversion_reads;
        if (sky_view.valid())
            conversion_reads.push_back({.handle = sky_view,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::sampled});
        graph.add_pass({.name = "environment radiance conversion",
                        .queue = render_queue_type::compute,
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::environment_equirect_to_cube,
                        .reads = std::move(conversion_reads),
                        .writes = {{.handle = environment_radiance,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = "environment irradiance convolution",
                        .queue = render_queue_type::compute,
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::environment_irradiance,
                        .reads = {{.handle = environment_radiance,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = environment_irradiance,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = "environment specular prefilter",
                        .queue = render_queue_type::compute,
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::environment_specular_prefilter,
                        .reads = {{.handle = environment_radiance,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = environment_specular,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = "BRDF integration",
                        .queue = render_queue_type::compute,
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::brdf_integration,
                        .writes = {{.handle = brdf_lut,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
    }

    std::vector<render_resource_access> shadow_reads;
    if (virtual_shadow_clusters.valid())
        shadow_reads.push_back({.handle = virtual_shadow_clusters,
                                .kind = render_resource_kind::buffer,
                                .usage = render_resource_usage::storage_buffer});
    graph.add_pass({.name = "directional static shadows",
                    .kind = render_pass_kind::custom,
                    .builtin = builtin_render_pass::directional_shadow_static,
                    .reads = shadow_reads,
                    .writes = {{.handle = directional_static_shadows,
                                .kind = render_resource_kind::depth_texture,
                                .usage = render_resource_usage::depth_attachment,
                                .write = true,
                                .load_op = render_load_op::clear}}});
    graph.add_pass({.name = "directional dynamic shadows",
                    .kind = render_pass_kind::custom,
                    .builtin = builtin_render_pass::directional_shadow_dynamic,
                    .reads = shadow_reads,
                    .writes = {{.handle = directional_dynamic_shadows,
                                .kind = render_resource_kind::depth_texture,
                                .usage = render_resource_usage::depth_attachment,
                                .write = true,
                                .load_op = render_load_op::clear}}});
    if (config.max_shadowed_point_lights > 0)
    {
        graph.add_pass({.name = "point light shadows",
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::point_shadow,
                        .writes = {{.handle = local_shadow_atlas,
                                    .kind = render_resource_kind::depth_texture,
                                    .usage = render_resource_usage::depth_attachment,
                                    .write = true,
                                    .load_op = render_load_op::load}}});
    }
    if (config.max_shadowed_spot_lights > 0)
    {
        graph.add_pass({.name = "spot light shadows",
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::spot_shadow,
                        .writes = {{.handle = local_shadow_atlas,
                                    .kind = render_resource_kind::depth_texture,
                                    .usage = render_resource_usage::depth_attachment,
                                    .write = true,
                                    .load_op = render_load_op::load}}});
    }
    std::vector<render_resource_access> sky_reads;
    if (sky_view.valid())
        sky_reads.push_back(
            {.handle = sky_view, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled});
    graph.add_pass({.name = environment.enabled && environment.sky_visible ? "sky composite" : "clear scene color",
                    .kind = render_pass_kind::clear,
                    .builtin = environment.enabled && environment.sky_visible ? builtin_render_pass::sky_composite
                                                                              : builtin_render_pass::none,
                    .reads = std::move(sky_reads),
                    .writes = {{.handle = scene_color,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true,
                                .load_op = render_load_op::clear}}});
    std::vector<render_resource_access> depth_reads;
    if (gpu_indirect_commands.valid())
    {
        depth_reads.push_back({.handle = gpu_indirect_commands,
                               .kind = render_resource_kind::buffer,
                               .usage = render_resource_usage::indirect_buffer});
        depth_reads.push_back({.handle = gpu_indirect_count,
                               .kind = render_resource_kind::buffer,
                               .usage = render_resource_usage::indirect_buffer});
    }
    graph.add_pass({.name = "depth prepass",
                    .kind = render_pass_kind::depth_prepass,
                    .builtin = builtin_render_pass::depth_prepass,
                    .reads = std::move(depth_reads),
                    .writes = {{.handle = depth,
                                .kind = render_resource_kind::depth_texture,
                                .usage = render_resource_usage::depth_attachment,
                                .write = true,
                                .load_op = render_load_op::clear,
                                .clear_depth = 1.0f}}});

    if (virtual_visibility.valid())
    {
        graph.add_pass({.name = "virtual geometry visibility resolve",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::virtual_geometry_visibility_resolve,
                        .reads = {{.handle = virtual_visibility,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = virtual_encoded_depth,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = depth,
                                    .kind = render_resource_kind::depth_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
    }

    if (depth_pyramid.valid())
    {
        graph.add_pass({.name = "visibility depth pyramid",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::depth_pyramid,
                        .reads = {{.handle = depth,
                                   .kind = render_resource_kind::depth_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = depth_pyramid,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
    }

    const auto motion = graph.add_resource({.name = "scene_motion",
                                            .kind = render_resource_kind::color_texture,
                                            .width_scale = config.render_scale,
                                            .height_scale = config.render_scale,
                                            .format = render_format::rg16_float});

    render_graph_resource_handle lighting_albedo{};
    render_graph_resource_handle lighting_normal{};
    render_graph_resource_handle lighting_material{};
    render_graph_resource_handle lighting_emissive{};
    render_graph_resource_handle object_id{};
    if (config.path == render_path::forward_plus)
    {
        std::vector<render_resource_access> forward_reads{{.handle = depth,
                                                           .kind = render_resource_kind::depth_texture,
                                                           .usage = render_resource_usage::depth_attachment},
                                                          {.handle = directional_static_shadows,
                                                           .kind = render_resource_kind::depth_texture,
                                                           .usage = render_resource_usage::sampled},
                                                          {.handle = directional_dynamic_shadows,
                                                           .kind = render_resource_kind::depth_texture,
                                                           .usage = render_resource_usage::sampled},
                                                          {.handle = local_shadow_atlas,
                                                           .kind = render_resource_kind::depth_texture,
                                                           .usage = render_resource_usage::sampled}};
        if (cloud_shadow.valid())
            forward_reads.push_back({.handle = cloud_shadow,
                                     .kind = render_resource_kind::color_texture,
                                     .usage = render_resource_usage::sampled});
        if (environment_irradiance.valid())
        {
            forward_reads.push_back({.handle = environment_irradiance,
                                     .kind = render_resource_kind::color_texture,
                                     .usage = render_resource_usage::sampled});
            forward_reads.push_back({.handle = environment_specular,
                                     .kind = render_resource_kind::color_texture,
                                     .usage = render_resource_usage::sampled});
            forward_reads.push_back({.handle = brdf_lut,
                                     .kind = render_resource_kind::color_texture,
                                     .usage = render_resource_usage::sampled});
        }
        graph.add_pass({.name = "forward opaque",
                        .kind = render_pass_kind::lighting,
                        .builtin = builtin_render_pass::forward_opaque,
                        .reads = std::move(forward_reads),
                        .writes = {{.handle = scene_color,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::color_attachment,
                                    .write = true,
                                    .load_op = render_load_op::load},
                                   {.handle = motion,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::color_attachment,
                                    .write = true,
                                    .load_op = render_load_op::clear}}});
    }
    else
    {
        const auto albedo = graph.add_resource({.name = "gbuffer_albedo",
                                                .kind = render_resource_kind::color_texture,
                                                .width_scale = config.render_scale,
                                                .height_scale = config.render_scale,
                                                .format = render_format::rgba8_srgb});
        const auto normal = graph.add_resource({.name = "gbuffer_normal",
                                                .kind = render_resource_kind::color_texture,
                                                .width_scale = config.render_scale,
                                                .height_scale = config.render_scale,
                                                .format = render_format::rg16_float});
        const auto material = graph.add_resource({.name = "gbuffer_material",
                                                  .kind = render_resource_kind::color_texture,
                                                  .width_scale = config.render_scale,
                                                  .height_scale = config.render_scale,
                                                  .format = render_format::rgba8_unorm});
        const auto emissive = graph.add_resource({.name = "gbuffer_emissive",
                                                  .kind = render_resource_kind::color_texture,
                                                  .width_scale = config.render_scale,
                                                  .height_scale = config.render_scale,
                                                  .format = render_format::rgba16_float});
        lighting_albedo = albedo;
        lighting_normal = normal;
        lighting_material = material;
        lighting_emissive = emissive;
        if (editor_view)
        {
            object_id = graph.add_resource({.name = "gbuffer_object_id",
                                            .kind = render_resource_kind::color_texture,
                                            .width_scale = config.render_scale,
                                            .height_scale = config.render_scale,
                                            .format = render_format::r32_uint,
                                            .persistent = true});
        }

        std::vector<render_resource_access> gbuffer_writes{{.handle = albedo,
                                                            .kind = render_resource_kind::color_texture,
                                                            .usage = render_resource_usage::color_attachment,
                                                            .write = true,
                                                            .load_op = render_load_op::clear},
                                                           {.handle = normal,
                                                            .kind = render_resource_kind::color_texture,
                                                            .usage = render_resource_usage::color_attachment,
                                                            .write = true,
                                                            .load_op = render_load_op::clear},
                                                           {.handle = material,
                                                            .kind = render_resource_kind::color_texture,
                                                            .usage = render_resource_usage::color_attachment,
                                                            .write = true,
                                                            .load_op = render_load_op::clear},
                                                           {.handle = emissive,
                                                            .kind = render_resource_kind::color_texture,
                                                            .usage = render_resource_usage::color_attachment,
                                                            .write = true,
                                                            .load_op = render_load_op::clear},
                                                           {.handle = motion,
                                                            .kind = render_resource_kind::color_texture,
                                                            .usage = render_resource_usage::color_attachment,
                                                            .write = true,
                                                            .load_op = render_load_op::clear}};
        if (object_id.valid())
            gbuffer_writes.push_back({.handle = object_id,
                                      .kind = render_resource_kind::color_texture,
                                      .usage = render_resource_usage::color_attachment,
                                      .write = true,
                                      .load_op = render_load_op::clear});
        graph.add_pass({.name = "gbuffer pass",
                        .kind = render_pass_kind::gbuffer,
                        .builtin = builtin_render_pass::gbuffer,
                        .reads = {{.handle = depth,
                                   .kind = render_resource_kind::depth_texture,
                                   .usage = render_resource_usage::depth_attachment}},
                        .writes = std::move(gbuffer_writes)});

        if (virtual_visibility.valid())
        {
            std::vector<render_resource_access> virtual_material_writes{{.handle = albedo,
                                                                         .kind = render_resource_kind::color_texture,
                                                                         .usage = render_resource_usage::storage,
                                                                         .write = true},
                                                                        {.handle = normal,
                                                                         .kind = render_resource_kind::color_texture,
                                                                         .usage = render_resource_usage::storage,
                                                                         .write = true},
                                                                        {.handle = material,
                                                                         .kind = render_resource_kind::color_texture,
                                                                         .usage = render_resource_usage::storage,
                                                                         .write = true},
                                                                        {.handle = emissive,
                                                                         .kind = render_resource_kind::color_texture,
                                                                         .usage = render_resource_usage::storage,
                                                                         .write = true},
                                                                        {.handle = motion,
                                                                         .kind = render_resource_kind::color_texture,
                                                                         .usage = render_resource_usage::storage,
                                                                         .write = true}};
            if (object_id.valid())
                virtual_material_writes.push_back({.handle = object_id,
                                                   .kind = render_resource_kind::color_texture,
                                                   .usage = render_resource_usage::storage,
                                                   .write = true});
            graph.add_pass({.name = "virtual geometry material resolve",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::virtual_geometry_material_resolve,
                            .reads = {{.handle = virtual_visibility,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = virtual_encoded_depth,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = std::move(virtual_material_writes)});
        }

        render_graph_resource_handle filtered_screen_shadow{};
        if (config.screen_space_shadows)
        {
            const auto screen_shadow_depth_pyramid = depth_pyramid;
            const auto screen_shadow = graph.add_resource({.name = "screen_space_shadow",
                                                           .kind = render_resource_kind::color_texture,
                                                           .width_scale = config.screen_space_shadow_scale,
                                                           .height_scale = config.screen_space_shadow_scale,
                                                           .format = render_format::r8_unorm});
            filtered_screen_shadow = graph.add_resource({.name = "screen_space_shadow_filtered",
                                                         .kind = render_resource_kind::color_texture,
                                                         .width_scale = config.screen_space_shadow_scale,
                                                         .height_scale = config.screen_space_shadow_scale,
                                                         .format = render_format::r8_unorm});
            graph.add_pass({.name = "screen space shadows",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::screen_space_shadow,
                            .reads = {{.handle = screen_shadow_depth_pyramid,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = normal,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = {{.handle = screen_shadow,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
            graph.add_pass({.name = "screen space shadow filter",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::screen_space_shadow_filter,
                            .reads = {{.handle = screen_shadow,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = screen_shadow_depth_pyramid,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = normal,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = {{.handle = filtered_screen_shadow,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
        }

        std::vector<render_resource_access> lighting_reads{
            {.handle = depth, .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled},
            {.handle = albedo, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled},
            {.handle = normal, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled},
            {.handle = material, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled},
            {.handle = emissive, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled},
            {.handle = motion, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled},
            {.handle = directional_static_shadows,
             .kind = render_resource_kind::depth_texture,
             .usage = render_resource_usage::sampled},
            {.handle = directional_dynamic_shadows,
             .kind = render_resource_kind::depth_texture,
             .usage = render_resource_usage::sampled},
            {.handle = local_shadow_atlas,
             .kind = render_resource_kind::depth_texture,
             .usage = render_resource_usage::sampled}};
        if (filtered_screen_shadow.valid())
            lighting_reads.push_back({.handle = filtered_screen_shadow,
                                      .kind = render_resource_kind::color_texture,
                                      .usage = render_resource_usage::sampled});
        if (object_id.valid())
            lighting_reads.push_back({.handle = object_id,
                                      .kind = render_resource_kind::color_texture,
                                      .usage = render_resource_usage::sampled});
        if (cloud_shadow.valid())
            lighting_reads.push_back({.handle = cloud_shadow,
                                      .kind = render_resource_kind::color_texture,
                                      .usage = render_resource_usage::sampled});
        if (environment_irradiance.valid())
        {
            lighting_reads.push_back({.handle = environment_irradiance,
                                      .kind = render_resource_kind::color_texture,
                                      .usage = render_resource_usage::sampled});
            lighting_reads.push_back({.handle = environment_specular,
                                      .kind = render_resource_kind::color_texture,
                                      .usage = render_resource_usage::sampled});
            lighting_reads.push_back({.handle = brdf_lut,
                                      .kind = render_resource_kind::color_texture,
                                      .usage = render_resource_usage::sampled});
        }
        graph.add_pass({.name = "deferred lighting",
                        .kind = render_pass_kind::lighting,
                        .builtin = builtin_render_pass::deferred_lighting,
                        .reads = std::move(lighting_reads),
                        .writes = {{.handle = scene_color,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::color_attachment,
                                    .write = true,
                                    .load_op = render_load_op::load}}});
    }

    const auto authored_indirect_method = environment.indirect_lighting.method;
    const bool permits_screen = authored_indirect_method == indirect_lighting_method::auto_select ||
                                authored_indirect_method == indirect_lighting_method::screen_space ||
                                authored_indirect_method == indirect_lighting_method::software ||
                                authored_indirect_method == indirect_lighting_method::hybrid_hardware;
    const bool permits_software = authored_indirect_method == indirect_lighting_method::auto_select ||
                                  authored_indirect_method == indirect_lighting_method::software ||
                                  authored_indirect_method == indirect_lighting_method::hybrid_hardware;
    const bool permits_hardware = authored_indirect_method == indirect_lighting_method::auto_select ||
                                  authored_indirect_method == indirect_lighting_method::hybrid_hardware;
    const bool execute_screen_gi = permits_screen && config.features.screen_space_gi;
    const bool execute_screen_reflections = permits_screen && config.features.screen_space_reflections;
    const bool execute_software_gi = permits_software && config.features.software_gi;
    const bool execute_software_reflections = permits_software && config.features.software_reflections;
    const bool execute_hardware_gi =
        permits_hardware && environment.indirect_lighting.allow_hardware_ray_tracing && config.features.hardware_gi;
    const bool execute_hardware_reflections = permits_hardware &&
                                              environment.indirect_lighting.allow_hardware_ray_tracing &&
                                              config.features.hardware_reflections;
    const bool indirect_enabled = environment.indirect_lighting.enabled && lighting_normal.valid() &&
                                  (execute_screen_gi || execute_screen_reflections || execute_software_gi ||
                                   execute_software_reflections || execute_hardware_gi || execute_hardware_reflections);
    if (indirect_enabled)
    {
        const auto lighting_hzb = depth_pyramid;

        render_graph_resource_handle surface_material_cache{};
        render_graph_resource_handle surface_radiance_cache{};
        render_graph_resource_handle global_distance_field{};
        render_graph_resource_handle radiance_cache{};
        if (config.features.surface_cache && (execute_software_gi || execute_software_reflections ||
                                              execute_hardware_gi || execute_hardware_reflections))
        {
            surface_material_cache = graph.add_resource({.name = "lighting_surface_material_cache",
                                                         .kind = render_resource_kind::color_texture,
                                                         .extent = {1024, 1024, 1},
                                                         .extent_mode = render_extent_mode::absolute,
                                                         .format = render_format::rgba16_float,
                                                         .persistent_key = "world.lighting.surface_material",
                                                         .imported = true,
                                                         .persistent = true});
            surface_radiance_cache = graph.add_resource({.name = "lighting_surface_radiance_cache",
                                                         .kind = render_resource_kind::color_texture,
                                                         .extent = {1024, 1024, 1},
                                                         .extent_mode = render_extent_mode::absolute,
                                                         .format = render_format::rgba16_float,
                                                         .persistent_key = "world.lighting.surface_radiance",
                                                         .imported = true,
                                                         .persistent = true});
            graph.add_pass({.name = "surface card capture",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::surface_card_capture,
                            .writes = {{.handle = surface_material_cache,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
            graph.add_pass({.name = "surface cache relighting",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::surface_cache_relight,
                            .reads = {{.handle = surface_material_cache,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = lighting_emissive,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = {{.handle = surface_radiance_cache,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
        }
        if (execute_software_gi || execute_software_reflections)
        {
            global_distance_field = graph.add_resource({.name = "lighting_global_distance_field",
                                                        .kind = render_resource_kind::color_texture,
                                                        .extent = {128, 128, 128},
                                                        .extent_mode = render_extent_mode::absolute,
                                                        .format = render_format::r32_float,
                                                        .array_layers = 4,
                                                        .persistent_key = "view.lighting.global_distance_field",
                                                        .imported = true,
                                                        .persistent = true});
            graph.add_pass({.name = "distance field composition",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::distance_field_composition,
                            .writes = {{.handle = global_distance_field,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
        }
        if (config.features.radiance_cache)
        {
            radiance_cache = graph.add_resource({.name = "lighting_radiance_cache",
                                                 .kind = render_resource_kind::color_texture,
                                                 .extent = {64, 64, 64},
                                                 .extent_mode = render_extent_mode::absolute,
                                                 .format = render_format::rgba16_float,
                                                 .array_layers = 3,
                                                 .persistent_key = "view.lighting.radiance_cache",
                                                 .imported = true,
                                                 .persistent = true});
            std::vector<render_resource_access> cache_reads;
            if (surface_radiance_cache.valid())
                cache_reads.push_back({.handle = surface_radiance_cache,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled});
            graph.add_pass({.name = "radiance cache update",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::radiance_cache_update,
                            .reads = std::move(cache_reads),
                            .writes = {{.handle = radiance_cache,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
        }

        const auto make_trace_target = [&](std::string name)
        {
            return graph.add_resource({.name = std::move(name),
                                       .kind = render_resource_kind::color_texture,
                                       .width_scale = config.lighting_trace_scale,
                                       .height_scale = config.lighting_trace_scale,
                                       .format = render_format::rgba16_float});
        };
        auto diffuse_trace = make_trace_target("indirect_diffuse_trace");
        auto reflection_trace = make_trace_target("indirect_reflection_trace");
        if (execute_screen_gi)
            graph.add_pass({.name = "screen space global illumination",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::screen_space_gi,
                            .reads = {{.handle = lighting_hzb,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = lighting_normal,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = scene_color,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = {{.handle = diffuse_trace,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
        if (execute_screen_reflections)
            graph.add_pass({.name = "screen space reflections",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::screen_space_reflections,
                            .reads = {{.handle = lighting_hzb,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = lighting_normal,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = lighting_material,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled},
                                      {.handle = scene_color,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = {{.handle = reflection_trace,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});

        if (execute_software_gi)
        {
            const auto resolved = make_trace_target("software_indirect_diffuse");
            std::vector<render_resource_access> reads{{.handle = global_distance_field,
                                                       .kind = render_resource_kind::color_texture,
                                                       .usage = render_resource_usage::sampled},
                                                      {.handle = surface_radiance_cache,
                                                       .kind = render_resource_kind::color_texture,
                                                       .usage = render_resource_usage::sampled}};
            if (execute_screen_gi)
                reads.push_back({.handle = diffuse_trace,
                                 .kind = render_resource_kind::color_texture,
                                 .usage = render_resource_usage::sampled});
            graph.add_pass({.name = "software global illumination",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::software_gi_trace,
                            .reads = std::move(reads),
                            .writes = {{.handle = resolved,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
            diffuse_trace = resolved;
        }
        if (execute_software_reflections)
        {
            const auto resolved = make_trace_target("software_reflections");
            std::vector<render_resource_access> reads{{.handle = global_distance_field,
                                                       .kind = render_resource_kind::color_texture,
                                                       .usage = render_resource_usage::sampled},
                                                      {.handle = surface_radiance_cache,
                                                       .kind = render_resource_kind::color_texture,
                                                       .usage = render_resource_usage::sampled}};
            if (execute_screen_reflections)
                reads.push_back({.handle = reflection_trace,
                                 .kind = render_resource_kind::color_texture,
                                 .usage = render_resource_usage::sampled});
            graph.add_pass({.name = "software reflections",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::software_reflections,
                            .reads = std::move(reads),
                            .writes = {{.handle = resolved,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
            reflection_trace = resolved;
        }
        if (execute_hardware_gi)
        {
            const auto resolved = make_trace_target("hardware_indirect_diffuse");
            graph.add_pass({.name = "hardware global illumination misses",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::hardware_gi_trace,
                            .reads = {{.handle = diffuse_trace,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = {{.handle = resolved,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
            diffuse_trace = resolved;
        }
        if (execute_hardware_reflections)
        {
            const auto resolved = make_trace_target("hardware_reflection_misses");
            graph.add_pass({.name = "hardware reflection misses",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::hardware_reflections,
                            .reads = {{.handle = reflection_trace,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = {{.handle = resolved,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
            reflection_trace = resolved;
        }

        const auto history_reset = render_history_reset::camera_cut | render_history_reset::resize |
                                   render_history_reset::render_scale_change |
                                   render_history_reset::world_epoch_change | render_history_reset::debug_view_change;
        const auto diffuse_history = graph.add_resource({.name = "indirect_diffuse_history",
                                                         .kind = render_resource_kind::color_texture,
                                                         .width_scale = config.lighting_trace_scale,
                                                         .height_scale = config.lighting_trace_scale,
                                                         .format = render_format::rgba16_float,
                                                         .persistent_key = "view.lighting.indirect_history",
                                                         .history_length = 2,
                                                         .history_reset = history_reset});
        const auto reflection_history = graph.add_resource({.name = "reflection_history",
                                                            .kind = render_resource_kind::color_texture,
                                                            .width_scale = config.lighting_trace_scale,
                                                            .height_scale = config.lighting_trace_scale,
                                                            .format = render_format::rgba16_float,
                                                            .persistent_key = "view.lighting.reflection_history",
                                                            .history_length = 2,
                                                            .history_reset = history_reset});
        graph.add_pass({.name = "indirect lighting temporal reconstruction",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::indirect_lighting_temporal,
                        .reads = {{.handle = diffuse_trace,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = motion,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = depth,
                                   .kind = render_resource_kind::depth_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = diffuse_history,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled,
                                   .history = render_history_access::previous}},
                        .writes = {{.handle = diffuse_history,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = "reflection temporal reconstruction",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::reflection_temporal,
                        .reads = {{.handle = reflection_trace,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = motion,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = reflection_history,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled,
                                   .history = render_history_access::previous}},
                        .writes = {{.handle = reflection_history,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        const auto filtered_lighting = make_trace_target("indirect_lighting_filtered");
        graph.add_pass({.name = "indirect lighting spatial filter",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::lighting_spatial_filter,
                        .reads = {{.handle = diffuse_history,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = reflection_history,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = depth,
                                   .kind = render_resource_kind::depth_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = lighting_normal,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = filtered_lighting,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = "indirect lighting composite",
                        .kind = render_pass_kind::lighting,
                        .builtin = builtin_render_pass::indirect_lighting_composite,
                        .reads = {{.handle = filtered_lighting,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = lighting_albedo,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = scene_color,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::color_attachment,
                                    .write = true,
                                    .load_op = render_load_op::load}}});
    }

    std::vector<render_resource_access> transparent_reads{{.handle = depth,
                                                           .kind = render_resource_kind::depth_texture,
                                                           .usage = render_resource_usage::depth_attachment},
                                                          {.handle = directional_static_shadows,
                                                           .kind = render_resource_kind::depth_texture,
                                                           .usage = render_resource_usage::sampled},
                                                          {.handle = directional_dynamic_shadows,
                                                           .kind = render_resource_kind::depth_texture,
                                                           .usage = render_resource_usage::sampled},
                                                          {.handle = local_shadow_atlas,
                                                           .kind = render_resource_kind::depth_texture,
                                                           .usage = render_resource_usage::sampled}};
    if (cloud_shadow.valid())
        transparent_reads.push_back({.handle = cloud_shadow,
                                     .kind = render_resource_kind::color_texture,
                                     .usage = render_resource_usage::sampled});
    graph.add_pass({.name = "forward transparent",
                    .kind = render_pass_kind::custom,
                    .builtin = builtin_render_pass::forward_transparent,
                    .reads = std::move(transparent_reads),
                    .writes = {{.handle = scene_color,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true,
                                .load_op = render_load_op::load}}});
    if (editor_view)
    {
        graph.add_pass({.name = "debug overlay",
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::debug_overlay,
                        .reads = {{.handle = depth,
                                   .kind = render_resource_kind::depth_texture,
                                   .usage = render_resource_usage::depth_attachment}},
                        .writes = {{.handle = scene_color,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::color_attachment,
                                    .write = true,
                                    .load_op = render_load_op::load}}});
    }

    auto resolved_scene_color = scene_color;
    if (config.features.temporal_antialiasing)
    {
        const auto dilated_motion = graph.add_resource({.name = "temporal_dilated_motion",
                                                        .kind = render_resource_kind::color_texture,
                                                        .width_scale = config.render_scale,
                                                        .height_scale = config.render_scale,
                                                        .format = render_format::rg16_float});
        const auto reactive_mask = graph.add_resource({.name = "temporal_reactive_mask",
                                                       .kind = render_resource_kind::color_texture,
                                                       .width_scale = config.render_scale,
                                                       .height_scale = config.render_scale,
                                                       .format = render_format::r8_unorm});
        const auto disocclusion_mask = graph.add_resource({.name = "temporal_disocclusion_mask",
                                                           .kind = render_resource_kind::color_texture,
                                                           .width_scale = config.render_scale,
                                                           .height_scale = config.render_scale,
                                                           .format = render_format::r8_unorm});
        const auto history_reset = render_history_reset::camera_cut | render_history_reset::resize |
                                   render_history_reset::render_scale_change |
                                   render_history_reset::world_epoch_change | render_history_reset::debug_view_change |
                                   render_history_reset::projection_change;
        const auto color_history = graph.add_resource({.name = "temporal_color_history",
                                                       .kind = render_resource_kind::color_texture,
                                                       .extent_mode = render_extent_mode::relative_to_output,
                                                       .format = render_format::rgba16_float,
                                                       .persistent_key = "view.temporal_color",
                                                       .history_length = 2,
                                                       .history_reset = history_reset});
        const auto depth_history = graph.add_resource({.name = "temporal_depth_history",
                                                       .kind = render_resource_kind::color_texture,
                                                       .extent_mode = render_extent_mode::relative_to_output,
                                                       .format = render_format::r32_float,
                                                       .persistent_key = "view.temporal_depth",
                                                       .history_length = 2,
                                                       .history_reset = history_reset});
        const auto moments_history = graph.add_resource({.name = "temporal_moments_history",
                                                         .kind = render_resource_kind::color_texture,
                                                         .extent_mode = render_extent_mode::relative_to_output,
                                                         .format = render_format::rg16_float,
                                                         .persistent_key = "view.temporal_moments",
                                                         .history_length = 2,
                                                         .history_reset = history_reset});
        const auto confidence_history = graph.add_resource({.name = "temporal_confidence_history",
                                                            .kind = render_resource_kind::color_texture,
                                                            .extent_mode = render_extent_mode::relative_to_output,
                                                            .format = render_format::r8_unorm,
                                                            .persistent_key = "view.temporal_confidence",
                                                            .history_length = 2,
                                                            .history_reset = history_reset});
        graph.add_pass({.name = "temporal velocity dilation",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::velocity_dilation,
                        .reads = {{.handle = motion,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = depth,
                                   .kind = render_resource_kind::depth_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = dilated_motion,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = "temporal reactive mask",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::reactive_mask,
                        .reads = {{.handle = scene_color,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = dilated_motion,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = reactive_mask,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        std::vector<render_resource_access> disocclusion_reads{
            {.handle = depth, .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled},
            {.handle = dilated_motion,
             .kind = render_resource_kind::color_texture,
             .usage = render_resource_usage::sampled},
            {.handle = depth_history,
             .kind = render_resource_kind::color_texture,
             .usage = render_resource_usage::sampled,
             .history = render_history_access::previous}};
        if (lighting_normal.valid())
            disocclusion_reads.push_back({.handle = lighting_normal,
                                          .kind = render_resource_kind::color_texture,
                                          .usage = render_resource_usage::sampled});
        if (object_id.valid())
            disocclusion_reads.push_back({.handle = object_id,
                                          .kind = render_resource_kind::color_texture,
                                          .usage = render_resource_usage::sampled});
        graph.add_pass({.name = "temporal disocclusion",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = builtin_render_pass::disocclusion_mask,
                        .reads = std::move(disocclusion_reads),
                        .writes = {{.handle = disocclusion_mask,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        graph.add_pass({.name = config.features.temporal_upscaling ? "temporal upscale" : "temporal antialiasing",
                        .queue = compute_queue,
                        .kind = render_pass_kind::compute,
                        .builtin = config.features.temporal_upscaling ? builtin_render_pass::temporal_upscale
                                                                      : builtin_render_pass::temporal_antialiasing,
                        .reads = {{.handle = scene_color,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = depth,
                                   .kind = render_resource_kind::depth_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = dilated_motion,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = reactive_mask,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = disocclusion_mask,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled},
                                  {.handle = color_history,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled,
                                   .history = render_history_access::previous},
                                  {.handle = moments_history,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled,
                                   .history = render_history_access::previous},
                                  {.handle = confidence_history,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled,
                                   .history = render_history_access::previous}},
                        .writes = {{.handle = color_history,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true},
                                   {.handle = depth_history,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true},
                                   {.handle = moments_history,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true},
                                   {.handle = confidence_history,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::storage,
                                    .write = true}}});
        resolved_scene_color = color_history;
        if (config.temporal.sharpening > 0.0f)
        {
            const auto sharpened = graph.add_resource({.name = "temporal_sharpened",
                                                       .kind = render_resource_kind::color_texture,
                                                       .extent_mode = render_extent_mode::relative_to_output,
                                                       .format = render_format::rgba16_float});
            graph.add_pass({.name = "temporal sharpen",
                            .queue = compute_queue,
                            .kind = render_pass_kind::compute,
                            .builtin = builtin_render_pass::spatial_sharpen,
                            .reads = {{.handle = color_history,
                                       .kind = render_resource_kind::color_texture,
                                       .usage = render_resource_usage::sampled}},
                            .writes = {{.handle = sharpened,
                                        .kind = render_resource_kind::color_texture,
                                        .usage = render_resource_usage::storage,
                                        .write = true}}});
            resolved_scene_color = sharpened;
        }
    }

    const auto luminance_histogram = graph.add_resource({.name = "luminance_histogram",
                                                         .kind = render_resource_kind::buffer,
                                                         .extent = {low_quality ? 64u : 256u, 1, 1}});
    const auto exposure = graph.add_resource(
        {.name = "view_exposure", .kind = render_resource_kind::buffer, .extent = {1, 1, 1}, .persistent = true});
    graph.add_pass({.name = "luminance histogram",
                    .queue = compute_queue,
                    .kind = render_pass_kind::post_process,
                    .builtin = builtin_render_pass::luminance_histogram,
                    .reads = {{.handle = resolved_scene_color,
                               .kind = render_resource_kind::color_texture,
                               .usage = render_resource_usage::sampled}},
                    .writes = {{.handle = luminance_histogram,
                                .kind = render_resource_kind::buffer,
                                .usage = render_resource_usage::storage_buffer,
                                .write = true}}});
    graph.add_pass({.name = "exposure resolve",
                    .queue = compute_queue,
                    .kind = render_pass_kind::post_process,
                    .builtin = builtin_render_pass::exposure_resolve,
                    .reads = {{.handle = luminance_histogram,
                               .kind = render_resource_kind::buffer,
                               .usage = render_resource_usage::storage_buffer}},
                    .writes = {{.handle = exposure,
                                .kind = render_resource_kind::buffer,
                                .usage = render_resource_usage::storage_buffer,
                                .write = true}}});
    auto presentation_target = viewport;
    if (config.features.fxaa)
        presentation_target = graph.add_resource({.name = "presentation_linear_ldr",
                                                  .kind = render_resource_kind::color_texture,
                                                  .extent_mode = render_extent_mode::relative_to_output,
                                                  .format = render_format::rgba8_unorm});
    graph.add_pass({.name = "output transform",
                    .kind = render_pass_kind::present,
                    .builtin = builtin_render_pass::output_transform,
                    .reads = {{.handle = resolved_scene_color,
                               .kind = render_resource_kind::color_texture,
                               .usage = render_resource_usage::sampled},
                              {.handle = exposure,
                               .kind = render_resource_kind::buffer,
                               .usage = render_resource_usage::storage_buffer}},
                    .writes = {{.handle = presentation_target,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true,
                                .load_op = render_load_op::clear}}});
    if (config.features.fxaa)
    {
        graph.add_pass({.name = "fast approximate antialiasing",
                        .kind = render_pass_kind::post_process,
                        .builtin = builtin_render_pass::fxaa,
                        .reads = {{.handle = presentation_target,
                                   .kind = render_resource_kind::color_texture,
                                   .usage = render_resource_usage::sampled}},
                        .writes = {{.handle = viewport,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::color_attachment,
                                    .write = true,
                                    .load_op = render_load_op::clear}}});
    }
    if (editor_view)
    {
        graph.add_pass({.name = "editor overlay",
                        .kind = render_pass_kind::custom,
                        .builtin = builtin_render_pass::editor_overlay,
                        .writes = {{.handle = viewport,
                                    .kind = render_resource_kind::color_texture,
                                    .usage = render_resource_usage::color_attachment,
                                    .write = true,
                                    .load_op = render_load_op::load}}});
    }
    return graph;
}

} // namespace arc::render
