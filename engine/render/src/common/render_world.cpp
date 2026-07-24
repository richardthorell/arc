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
    packet.visible_virtual_items.clear();
    packet.instance_batches.clear();
    packet.indirect_draws.clear();
    packet.culled_item_count = 0;
    packet.culled_virtual_cluster_count = 0;

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

        item.sort_key = make_render_sort_key(scene_render_pass::gbuffer, item.material, item.mesh, item_depth(packet, item));
        packet.visible_virtual_items.push_back(index);
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

    std::sort(packet.visible_virtual_items.begin(), packet.visible_virtual_items.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
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

render_graph make_scene_draw_graph(std::string_view target_name, render_path path, bool editor_view)
{
    resolved_render_config config{};
    config.path = path;
    config.quality = path == render_path::forward_plus
        ? render_quality_tier::low
        : render_quality_tier::medium;
    return make_scene_draw_graph(target_name, config, editor_view);
}

render_graph make_scene_draw_graph(
    std::string_view target_name,
    const resolved_render_config& config,
    bool editor_view)
{
    world_environment_data environment;
    environment.enabled = true;
    environment.sky_visible = true;
    environment.atmosphere.enabled = true;
    return make_scene_draw_graph(target_name, config, editor_view, environment);
}

render_graph make_scene_draw_graph(
    std::string_view target_name,
    const resolved_render_config& config,
    bool editor_view,
    const world_environment_data& environment)
{
    std::string target(target_name);
    if (target.empty())
        target = "viewport";

    render_graph graph;
    const auto viewport = graph.add_resource({
        .name = target,
        .kind = render_resource_kind::color_texture,
        .format = render_format::rgba16_float,
        .persistent = true });
    const auto depth = graph.add_resource({
        .name = "scene_depth",
        .kind = render_resource_kind::depth_texture,
        .width_scale = config.render_scale,
        .height_scale = config.render_scale,
        .format = render_format::d32_float,
        .persistent = true });
    const auto scene_color = graph.add_resource({
        .name = "scene_color",
        .kind = render_resource_kind::color_texture,
        .width_scale = config.render_scale,
        .height_scale = config.render_scale,
        .format = render_format::rgba16_float,
        .persistent = true });
    const auto shadow_atlas = graph.add_resource({
        .name = "directional_shadow_atlas",
        .kind = render_resource_kind::depth_texture,
        .extent = { config.directional_shadow_resolution, config.directional_shadow_resolution, 1 },
        .extent_mode = render_extent_mode::absolute,
        .format = render_format::d32_float,
        .array_layers = config.directional_shadow_cascades,
        .persistent = true });

    const bool high_quality = config.quality == render_quality_tier::high;
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
        environment.source == sky_source_mode::physical_atmosphere &&
        (config.quality == render_quality_tier::medium || config.quality == render_quality_tier::high))
    {
        const auto transmittance = graph.add_resource({
            .name = "atmosphere_transmittance",
            .kind = render_resource_kind::color_texture,
            .extent = { 256, 64, 1 },
            .extent_mode = render_extent_mode::absolute,
            .format = render_format::rgba16_float,
            .persistent = true });
        const auto multi_scattering = graph.add_resource({
            .name = "atmosphere_multi_scattering",
            .kind = render_resource_kind::color_texture,
            .extent = { 32, 32, 1 },
            .extent_mode = render_extent_mode::absolute,
            .format = render_format::rgba16_float,
            .persistent = true });
        sky_view = graph.add_resource({
            .name = "atmosphere_sky_view",
            .kind = render_resource_kind::color_texture,
            .extent = { 192, 108, 1 },
            .extent_mode = render_extent_mode::absolute,
            .format = render_format::rgba16_float,
            .persistent = true });
        graph.add_pass({
            .name = "atmosphere transmittance",
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::atmosphere_transmittance,
            .writes = { { .handle = transmittance, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::storage, .write = true } }
        });
        graph.add_pass({
            .name = "atmosphere multi scattering",
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::atmosphere_multi_scattering,
            .reads = { { .handle = transmittance, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled } },
            .writes = { { .handle = multi_scattering, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::storage, .write = true } }
        });
        graph.add_pass({
            .name = "atmosphere sky view",
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::atmosphere_sky_view,
            .reads = {
                { .handle = transmittance, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
                { .handle = multi_scattering, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled }
            },
            .writes = { { .handle = sky_view, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::storage, .write = true } }
        });
    }

    render_graph_resource_handle cloud_shadow{};
    if (environment.enabled && environment.clouds.enabled && environment.clouds.cast_shadows &&
        (config.quality == render_quality_tier::medium || config.quality == render_quality_tier::high))
    {
        cloud_shadow = graph.add_resource({
            .name = "cloud_shadow",
            .kind = render_resource_kind::color_texture,
            .extent = { 512, 512, 1 },
            .extent_mode = render_extent_mode::absolute,
            .format = render_format::r8_unorm,
            .persistent = true });
        graph.add_pass({
            .name = "cloud shadow",
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::cloud_shadow,
            .writes = { { .handle = cloud_shadow, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::storage, .write = true } }
        });
    }

    const bool generate_ibl =
        environment.enabled &&
        environment.affect_lighting &&
        environment.lighting.enabled &&
        environment.lighting.source != environment_lighting_source_mode::constant_color;
    if (generate_ibl)
    {
        environment_radiance = graph.add_resource({
            .name = "environment_radiance",
            .kind = render_resource_kind::color_texture,
            .extent = { radiance_resolution, radiance_resolution, 1 },
            .extent_mode = render_extent_mode::absolute,
            .format = render_format::rgba16_float,
            .mip_levels = high_quality ? 10u : low_quality ? 8u : 9u,
            .array_layers = 6,
            .persistent = true });
        environment_irradiance = graph.add_resource({
            .name = "environment_irradiance",
            .kind = render_resource_kind::color_texture,
            .extent = { irradiance_resolution, irradiance_resolution, 1 },
            .extent_mode = render_extent_mode::absolute,
            .format = render_format::rgba16_float,
            .array_layers = 6,
            .persistent = true });
        environment_specular = graph.add_resource({
            .name = "environment_specular",
            .kind = render_resource_kind::color_texture,
            .extent = { radiance_resolution, radiance_resolution, 1 },
            .extent_mode = render_extent_mode::absolute,
            .format = render_format::rgba16_float,
            .mip_levels = high_quality ? 10u : low_quality ? 8u : 9u,
            .array_layers = 6,
            .persistent = true });
        brdf_lut = graph.add_resource({
            .name = "environment_brdf_lut",
            .kind = render_resource_kind::color_texture,
            .extent = { brdf_resolution, brdf_resolution, 1 },
            .extent_mode = render_extent_mode::absolute,
            .format = render_format::rg16_float,
            .persistent = true });

        std::vector<render_resource_access> conversion_reads;
        if (sky_view.valid())
            conversion_reads.push_back({
                .handle = sky_view,
                .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
        graph.add_pass({
            .name = "environment radiance conversion",
            .queue = render_queue_type::compute,
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::environment_equirect_to_cube,
            .reads = std::move(conversion_reads),
            .writes = { { .handle = environment_radiance, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::storage, .write = true } }
        });
        graph.add_pass({
            .name = "environment irradiance convolution",
            .queue = render_queue_type::compute,
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::environment_irradiance,
            .reads = { { .handle = environment_radiance, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled } },
            .writes = { { .handle = environment_irradiance, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::storage, .write = true } }
        });
        graph.add_pass({
            .name = "environment specular prefilter",
            .queue = render_queue_type::compute,
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::environment_specular_prefilter,
            .reads = { { .handle = environment_radiance, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled } },
            .writes = { { .handle = environment_specular, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::storage, .write = true } }
        });
        graph.add_pass({
            .name = "BRDF integration",
            .queue = render_queue_type::compute,
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::brdf_integration,
            .writes = { { .handle = brdf_lut, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::storage, .write = true } }
        });
    }

    graph.add_pass({
        .name = "directional shadow cascades",
        .kind = render_pass_kind::custom,
        .writes = { { .handle = shadow_atlas, .kind = render_resource_kind::depth_texture,
            .usage = render_resource_usage::depth_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    std::vector<render_resource_access> sky_reads;
    if (sky_view.valid())
        sky_reads.push_back({ .handle = sky_view, .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::sampled });
    graph.add_pass({
        .name = environment.enabled && environment.sky_visible ? "sky composite" : "clear scene color",
        .kind = render_pass_kind::clear,
        .builtin = environment.enabled && environment.sky_visible
            ? builtin_render_pass::sky_composite
            : builtin_render_pass::none,
        .reads = std::move(sky_reads),
        .writes = { { .handle = scene_color, .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    graph.add_pass({
        .name = "depth prepass",
        .kind = render_pass_kind::depth_prepass,
        .writes = { { .handle = depth, .kind = render_resource_kind::depth_texture,
            .usage = render_resource_usage::depth_attachment, .write = true, .load_op = render_load_op::clear,
            .clear_depth = 0.0f } }
    });

    if (config.path == render_path::forward_plus)
    {
        std::vector<render_resource_access> forward_reads{
            { .handle = depth, .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment },
            { .handle = shadow_atlas, .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled }
        };
        if (cloud_shadow.valid())
            forward_reads.push_back({ .handle = cloud_shadow, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
        if (environment_irradiance.valid())
        {
            forward_reads.push_back({ .handle = environment_irradiance, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
            forward_reads.push_back({ .handle = environment_specular, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
            forward_reads.push_back({ .handle = brdf_lut, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
        }
        graph.add_pass({
            .name = "forward opaque",
            .kind = render_pass_kind::lighting,
            .reads = std::move(forward_reads),
            .writes = { { .handle = scene_color, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load } }
        });
    }
    else
    {
        const auto albedo = graph.add_resource({
            .name = "gbuffer_albedo", .kind = render_resource_kind::color_texture,
            .width_scale = config.render_scale, .height_scale = config.render_scale,
            .format = render_format::rgba8_srgb });
        const auto normal = graph.add_resource({
            .name = "gbuffer_normal", .kind = render_resource_kind::color_texture,
            .width_scale = config.render_scale, .height_scale = config.render_scale,
            .format = render_format::rg16_float });
        const auto material = graph.add_resource({
            .name = "gbuffer_material", .kind = render_resource_kind::color_texture,
            .width_scale = config.render_scale, .height_scale = config.render_scale,
            .format = render_format::rgba8_unorm });
        const auto emissive = graph.add_resource({
            .name = "gbuffer_emissive", .kind = render_resource_kind::color_texture,
            .width_scale = config.render_scale, .height_scale = config.render_scale,
            .format = render_format::rgba16_float });
        const auto motion = graph.add_resource({
            .name = "gbuffer_motion", .kind = render_resource_kind::color_texture,
            .width_scale = config.render_scale, .height_scale = config.render_scale,
            .format = render_format::rg16_float });
        render_graph_resource_handle object_id{};
        if (editor_view)
        {
            object_id = graph.add_resource({
                .name = "gbuffer_object_id", .kind = render_resource_kind::color_texture,
                .width_scale = config.render_scale, .height_scale = config.render_scale,
                .format = render_format::r32_uint, .persistent = true });
        }

        std::vector<render_resource_access> gbuffer_writes{
            { .handle = albedo, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .handle = normal, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .handle = material, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .handle = emissive, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear },
            { .handle = motion, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear }
        };
        if (object_id.valid())
            gbuffer_writes.push_back({ .handle = object_id, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear });
        graph.add_pass({
            .name = "gbuffer pass",
            .kind = render_pass_kind::gbuffer,
            .reads = { { .handle = depth, .kind = render_resource_kind::depth_texture,
                .usage = render_resource_usage::depth_attachment } },
            .writes = std::move(gbuffer_writes)
        });

        std::vector<render_resource_access> lighting_reads{
            { .handle = depth, .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled },
            { .handle = albedo, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .handle = normal, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .handle = material, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .handle = emissive, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .handle = motion, .kind = render_resource_kind::color_texture, .usage = render_resource_usage::sampled },
            { .handle = shadow_atlas, .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled }
        };
        if (object_id.valid())
            lighting_reads.push_back({ .handle = object_id, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
        if (cloud_shadow.valid())
            lighting_reads.push_back({ .handle = cloud_shadow, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
        if (environment_irradiance.valid())
        {
            lighting_reads.push_back({ .handle = environment_irradiance, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
            lighting_reads.push_back({ .handle = environment_specular, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
            lighting_reads.push_back({ .handle = brdf_lut, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled });
        }
        graph.add_pass({
            .name = "deferred lighting",
            .kind = render_pass_kind::lighting,
            .reads = std::move(lighting_reads),
            .writes = { { .handle = scene_color, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load } }
        });
    }

    std::vector<render_resource_access> transparent_reads{
        { .handle = depth, .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::depth_attachment },
        { .handle = shadow_atlas, .kind = render_resource_kind::depth_texture, .usage = render_resource_usage::sampled }
    };
    if (cloud_shadow.valid())
        transparent_reads.push_back({ .handle = cloud_shadow, .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::sampled });
    graph.add_pass({
        .name = "forward transparent",
        .kind = render_pass_kind::custom,
        .reads = std::move(transparent_reads),
        .writes = { { .handle = scene_color, .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load } }
    });
    if (editor_view)
    {
        graph.add_pass({
            .name = "debug overlay",
            .kind = render_pass_kind::custom,
            .builtin = builtin_render_pass::debug_overlay,
            .reads = { { .handle = depth, .kind = render_resource_kind::depth_texture,
                .usage = render_resource_usage::depth_attachment } },
            .writes = { { .handle = scene_color, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::load } }
        });
    }

    const auto luminance_histogram = graph.add_resource({
        .name = "luminance_histogram",
        .kind = render_resource_kind::buffer,
        .extent = { low_quality ? 64u : 256u, 1, 1 } });
    const auto exposure = graph.add_resource({
        .name = "view_exposure",
        .kind = render_resource_kind::buffer,
        .extent = { 1, 1, 1 },
        .persistent = true });
    graph.add_pass({
        .name = "luminance histogram",
        .queue = render_queue_type::compute,
        .kind = render_pass_kind::post_process,
        .builtin = builtin_render_pass::luminance_histogram,
        .reads = { { .handle = scene_color, .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::sampled } },
        .writes = { { .handle = luminance_histogram, .kind = render_resource_kind::buffer,
            .usage = render_resource_usage::storage_buffer, .write = true } }
    });
    graph.add_pass({
        .name = "exposure resolve",
        .queue = render_queue_type::compute,
        .kind = render_pass_kind::post_process,
        .builtin = builtin_render_pass::exposure_resolve,
        .reads = { { .handle = luminance_histogram, .kind = render_resource_kind::buffer,
            .usage = render_resource_usage::storage_buffer } },
        .writes = { { .handle = exposure, .kind = render_resource_kind::buffer,
            .usage = render_resource_usage::storage_buffer, .write = true } }
    });
    graph.add_pass({
        .name = "output transform",
        .kind = render_pass_kind::present,
        .builtin = builtin_render_pass::output_transform,
        .reads = {
            { .handle = scene_color, .kind = render_resource_kind::color_texture,
                .usage = render_resource_usage::sampled },
            { .handle = exposure, .kind = render_resource_kind::buffer,
                .usage = render_resource_usage::storage_buffer }
        },
        .writes = { { .handle = viewport, .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::color_attachment, .write = true, .load_op = render_load_op::clear } }
    });
    return graph;
}

} // namespace arc::render
