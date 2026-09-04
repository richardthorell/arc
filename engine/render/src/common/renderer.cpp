#include <arc/render/renderer.h>
#include <arc/render/render_world.h>

#include <arc/framework/application.h>
#include <arc/diagnostics/log.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <string>
#include <unordered_set>
#include <utility>

namespace arc::render
{
namespace
{

constexpr std::uint64_t gibibyte = 1024ull * 1024ull * 1024ull;
constexpr std::uint64_t mebibyte = 1024ull * 1024ull;

float halton(std::uint64_t index, std::uint32_t base) noexcept
{
    float result{};
    float fraction = 1.0f;
    while (index > 0)
    {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(index % base);
        index /= base;
    }
    return result;
}

std::uint64_t renderer_resource_key(resource_handle handle) noexcept
{
    return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
}

std::uint64_t texture_content_hash(std::span<const std::byte> bytes) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto value : bytes)
    {
        hash ^= std::to_integer<std::uint8_t>(value);
        hash *= 1099511628211ull;
    }
    return hash;
}

bool valid_streamed_texture_descriptor(const streamed_texture_descriptor& descriptor) noexcept
{
    const auto& texture = descriptor.texture;
    const auto& artifact = descriptor.artifact;
    return descriptor.mode != texture_streaming_mode::resident && descriptor.source != 0 &&
           descriptor.content_generation != 0 && texture.dimension == texture_dimension::texture_2d &&
           texture.depth == 1 && texture.width > 0 && texture.height > 0 && texture.mip_levels > 0 &&
           artifact.schema_version == texture_artifact_schema_version && artifact.width == texture.width &&
           artifact.height == texture.height && artifact.mip_count == texture.mip_levels &&
           artifact.format == texture.format && artifact.mips.size() == artifact.mip_count &&
           artifact.tail_first_mip < artifact.mip_count;
}

gpu_texture_table_record texture_table_record(texture_handle handle, const texture_descriptor& texture,
                                              std::uint32_t mip_window_base, std::uint32_t flags) noexcept
{
    return {.generation = handle.generation,
            .descriptor_index = handle.index,
            .descriptor_generation = handle.generation,
            .flags = flags,
            .mip_window_base = mip_window_base,
            .mip_count = texture.mip_levels > mip_window_base ? texture.mip_levels - mip_window_base : 1u,
            .width = texture.width,
            .height = texture.height};
}

gpu_texture_table_record texture_table_record(texture_handle handle, const texture_data& texture) noexcept
{
    return texture_table_record(handle,
                                {.name = texture.name,
                                 .width = texture.width,
                                 .height = texture.height,
                                 .depth = texture.depth,
                                 .dimension = texture.dimension,
                                 .mip_levels = texture.mip_levels,
                                 .format = texture.format,
                                 .color_space = texture.color_space,
                                 .semantic = texture.semantic},
                                0u, 0u);
}

gpu_material_table_record material_table_record(const material_descriptor& material,
                                                const gpu_resource_tables& tables) noexcept
{
    gpu_material_table_record record{};
    record.generation = material.handle.generation;
    record.flags = static_cast<std::uint32_t>(material.alpha_mode) |
                   (static_cast<std::uint32_t>(material.shading_model) << 4u) |
                   (material.double_sided ? 1u << 8u : 0u);
    for (std::uint32_t component = 0; component < 4u; ++component)
        record.base_color[component] = material.base_color[component];
    record.emissive[0] = material.emissive_factor[0];
    record.emissive[1] = material.emissive_factor[1];
    record.emissive[2] = material.emissive_factor[2];
    record.surface[0] = material.metallic;
    record.surface[1] = material.roughness;
    record.surface[2] = material.alpha_cutoff;
    record.surface[3] = material.normal_scale;
    const std::array textures{material.base_color_texture,
                              material.metallic_roughness_texture,
                              material.normal_texture,
                              material.occlusion_texture,
                              material.emissive_texture,
                              material.clear_coat_texture,
                              material.clear_coat_roughness_texture,
                              material.clear_coat_normal_texture,
                              material.anisotropy_texture,
                              material.subsurface_texture,
                              material.thickness_texture,
                              material.transmission_texture};
    record.texture_indices.fill(resource_handle::invalid_index);
    for (std::size_t index = 0; index < textures.size(); ++index)
        if (const auto reference = tables.find(gpu_resource_table_kind::texture, textures[index]))
        {
            record.texture_indices[index] = reference->index;
            record.texture_generations[index] = reference->generation;
        }
    return record;
}

} // namespace

anti_aliasing_method resolve_anti_aliasing(anti_aliasing_method requested, render_path path, float render_scale,
                                           const render_capabilities& capabilities) noexcept
{
    auto desired = requested;
    if (desired == anti_aliasing_method::auto_select)
    {
        if (path == render_path::forward_plus)
            desired = anti_aliasing_method::fxaa;
        else
            desired = render_scale < 0.999f ? anti_aliasing_method::taau : anti_aliasing_method::taa;
    }

    if (desired == anti_aliasing_method::taau && capabilities.temporal_resolve && capabilities.temporal_upscale)
        return desired;
    if ((desired == anti_aliasing_method::taa || desired == anti_aliasing_method::taau) &&
        capabilities.temporal_resolve)
        return anti_aliasing_method::taa;
    if (desired != anti_aliasing_method::disabled && capabilities.fxaa) return anti_aliasing_method::fxaa;
    return anti_aliasing_method::disabled;
}

resolved_render_config resolve_render_config(const renderer_config& config, const render_capabilities& capabilities)
{
    resolved_render_config result{};
    result.requested_quality = config.quality;
    result.requested_path = config.path;
    result.requested_anti_aliasing = config.anti_aliasing;
    result.temporal = config.temporal;

    if (config.quality == render_quality_tier::auto_select)
    {
        const bool constrained_memory =
            capabilities.integrated_gpu ||
            (capabilities.dedicated_video_memory != 0 && capabilities.dedicated_video_memory < 2ull * gibibyte);
        result.quality = constrained_memory ? render_quality_tier::low : render_quality_tier::medium;
        result.fallback_reasons.push_back(
            constrained_memory ? "auto-selected low quality for an integrated or memory-constrained adapter"
                               : "auto-selected standard quality");
    }
    else if (config.quality == render_quality_tier::ultra &&
             ((capabilities.memory_budget != 0 && capabilities.memory_budget < 12ull * gibibyte) ||
              (capabilities.memory_budget == 0 && capabilities.dedicated_video_memory != 0 &&
               capabilities.dedicated_video_memory < 12ull * gibibyte)))
    {
        result.quality = render_quality_tier::high;
        result.fallback_reasons.push_back(
            "ultra quality requires at least 12 GiB of available GPU memory; using high limits");
    }
    else if (config.quality == render_quality_tier::high &&
             ((capabilities.memory_budget != 0 && capabilities.memory_budget < 6ull * gibibyte) ||
              (capabilities.memory_budget == 0 && capabilities.dedicated_video_memory != 0 &&
               capabilities.dedicated_video_memory < 6ull * gibibyte)))
    {
        result.quality = render_quality_tier::medium;
        result.fallback_reasons.push_back(
            "high shadow quality requires at least 6 GiB of available GPU memory; using standard limits");
    }
    else
    {
        result.quality = config.quality;
    }

    const auto& profile = quality_profile(result.quality);
    result.target_frame_time_ms =
        config.target_frame_time_ms > 0.0f ? config.target_frame_time_ms : profile.target_frame_time_ms;
    if (config.path == render_path::auto_select)
        result.path = profile.default_path;
    else
        result.path = config.path;
    result.anti_aliasing =
        config.force_disable_temporal
            ? (config.anti_aliasing == anti_aliasing_method::disabled
                   ? anti_aliasing_method::disabled
                   : resolve_anti_aliasing(anti_aliasing_method::fxaa, result.path, 1.0f, capabilities))
            : resolve_anti_aliasing(config.anti_aliasing, result.path, 1.0f, capabilities);
    result.minimum_render_scale = config.enable_dynamic_resolution ? profile.minimum_render_scale : 1.0f;
    result.maximum_render_scale = profile.maximum_render_scale;
    result.max_point_lights = profile.max_point_lights;
    result.max_spot_lights = profile.max_spot_lights;
    result.directional_shadow_cascades = profile.directional_shadow_cascades;
    result.directional_shadow_resolution = profile.directional_shadow_resolution;
    result.directional_shadow_distance = profile.directional_shadow_distance;
    result.local_shadow_atlas_resolution = profile.local_shadow_atlas_resolution;
    result.max_shadowed_point_lights = profile.max_shadowed_point_lights;
    result.max_shadowed_spot_lights = profile.max_shadowed_spot_lights;
    result.max_local_shadow_resolution = profile.max_local_shadow_resolution;
    result.screen_space_shadows = profile.screen_space_shadows;
    result.screen_space_shadow_scale = profile.screen_space_shadow_scale;
    result.virtual_shadow_budget_bytes = profile.virtual_shadow_budget_bytes;
    result.virtual_shadow_page_render_budget = profile.virtual_shadow_page_render_budget;
    if (capabilities.memory_budget != 0 && result.virtual_shadow_budget_bytes != 0)
        result.virtual_shadow_budget_bytes =
            std::min(result.virtual_shadow_budget_bytes, capabilities.memory_budget * 8u / 100u);
    result.geometry_error_threshold = profile.geometry_error_threshold;
    result.shadow_resolution_scale = profile.maximum_shadow_resolution_scale;
    result.volumetric_resolution_scale = profile.maximum_volumetric_resolution_scale;
    result.gi_trace_budget = profile.gi_trace_budget;
    result.reflection_ray_budget = profile.reflection_ray_budget;
    result.lighting_trace_scale = profile.lighting_trace_scale;
    result.surface_cache_update_budget = profile.surface_cache_update_budget;
    result.radiance_probe_update_budget = profile.radiance_probe_update_budget;
    result.lighting_scene_gpu_budget_bytes = result.quality == render_quality_tier::ultra  ? 768ull * 1024ull * 1024ull
                                             : result.quality == render_quality_tier::high ? 384ull * 1024ull * 1024ull
                                                                                           : 0ull;
    if (capabilities.memory_budget != 0 && result.lighting_scene_gpu_budget_bytes != 0)
    {
        const std::uint64_t percentage = result.quality == render_quality_tier::ultra ? 12u : 8u;
        result.lighting_scene_gpu_budget_bytes =
            std::min(result.lighting_scene_gpu_budget_bytes, capabilities.memory_budget * percentage / 100u);
    }
    const bool optional_features = !config.force_disable_optional_features;
    const bool texture_streaming = !config.force_disable_texture_streaming && capabilities.texture_mip_streaming;
    const bool virtual_textures = texture_streaming && !config.force_disable_virtual_textures && optional_features &&
                                  capabilities.virtual_texture_feedback && capabilities.virtual_texture_sampling &&
                                  capabilities.compute_shaders && capabilities.storage_buffers &&
                                  capabilities.descriptor_indexing;
    const bool gpu_scene_supported = capabilities.compute_shaders && capabilities.storage_buffers &&
                                     capabilities.shader_draw_parameters && capabilities.draw_indirect &&
                                     capabilities.gpu_scene_indirect;
    const bool gpu_driven =
        optional_features && !config.force_disable_gpu_driven && profile.prefer_gpu_driven && gpu_scene_supported;
    const bool bindless_gpu_scene = gpu_driven && capabilities.gpu_visibility_compaction &&
                                    capabilities.bindless_sampled_images && capabilities.bindless_samplers &&
                                    capabilities.bindless_material_tables && capabilities.bindless_geometry_tables;
    const bool virtual_geometry_quality = result.quality == render_quality_tier::ultra;
    const bool virtual_geometry_common = optional_features && virtual_geometry_quality && gpu_driven &&
                                         capabilities.hzb_occlusion && capabilities.descriptor_indexing &&
                                         capabilities.virtual_geometry_streaming &&
                                         capabilities.bindless_sampled_images && capabilities.bindless_samplers &&
                                         capabilities.bindless_material_tables && capabilities.bindless_geometry_tables;
    const auto virtual_geometry_path =
        virtual_geometry_common && capabilities.virtual_geometry_mesh_shader ? virtual_geometry_raster_path::mesh_shader
        : virtual_geometry_common && capabilities.virtual_geometry_compute   ? virtual_geometry_raster_path::compute
                                                                           : virtual_geometry_raster_path::unavailable;
    const bool virtual_shadow_maps = optional_features && result.quality == render_quality_tier::ultra && gpu_driven &&
                                     capabilities.virtual_shadow_allocation && capabilities.virtual_shadow_feedback &&
                                     capabilities.virtual_shadow_rendering && capabilities.virtual_shadow_sampling &&
                                     capabilities.compute_shaders && capabilities.storage_buffers;
    const bool virtual_shadow_virtual_geometry = virtual_shadow_maps && capabilities.virtual_shadow_virtual_geometry &&
                                                 virtual_geometry_path != virtual_geometry_raster_path::unavailable;
    const bool screen_space_contact_shadows = optional_features && profile.screen_space_shadows &&
                                              capabilities.screen_space_contact_shadows && capabilities.hzb_occlusion &&
                                              capabilities.compute_shaders;
    result.screen_space_shadows = screen_space_contact_shadows;
    const bool screen_space_lighting = optional_features && !config.force_disable_dynamic_gi &&
                                       capabilities.screen_space_indirect_lighting && capabilities.hzb_occlusion &&
                                       capabilities.temporal_resolve && capabilities.storage_images;
    const bool software_lighting = optional_features && !config.force_disable_dynamic_gi &&
                                   capabilities.software_ray_tracing && capabilities.surface_cache &&
                                   capabilities.radiance_cache && capabilities.compute_shaders &&
                                   capabilities.storage_buffers && capabilities.storage_images;
    const bool hardware_lighting = software_lighting && !config.force_disable_hardware_ray_tracing &&
                                   capabilities.ray_tracing && capabilities.hardware_ray_query;
    if (result.quality == render_quality_tier::ultra && hardware_lighting)
        result.indirect_lighting_path = lighting_trace_path::hybrid_hardware;
    else if ((result.quality == render_quality_tier::high || result.quality == render_quality_tier::ultra) &&
             software_lighting)
        result.indirect_lighting_path = lighting_trace_path::software_distance_field;
    else if (result.quality == render_quality_tier::medium && screen_space_lighting)
        result.indirect_lighting_path = lighting_trace_path::screen_space;
    else
        result.indirect_lighting_path = lighting_trace_path::baked_probe;
    gpu_submission_path submission = gpu_submission_path::cpu_direct;
    if (gpu_driven && !config.force_cpu_submission && capabilities.draw_indirect)
        submission = optional_features && capabilities.draw_indirect_count && capabilities.gpu_scene_indirect_count
                         ? gpu_submission_path::indirect_count
                         : gpu_submission_path::indirect;
    result.features = {.dynamic_rendering = optional_features && capabilities.dynamic_rendering,
                       .synchronization2 = optional_features && capabilities.synchronization2,
                       .timeline_semaphores = optional_features && capabilities.timeline_semaphores,
                       .descriptor_indexing = optional_features && capabilities.descriptor_indexing,
                       .descriptor_buffer = optional_features && capabilities.descriptor_buffer,
                       .draw_indirect = capabilities.draw_indirect,
                       .draw_indirect_count = optional_features && capabilities.draw_indirect_count,
                       .gpu_driven_rendering = gpu_driven,
                       .gpu_binding_model = bindless_gpu_scene ? gpu_resource_binding_model::bindless
                                                               : gpu_resource_binding_model::classic,
                       .gpu_visibility_compaction = bindless_gpu_scene,
                       .gpu_transparent_sorting = bindless_gpu_scene && capabilities.gpu_transparent_sorting,
                       .gpu_skinning = bindless_gpu_scene && capabilities.gpu_skinning,
                       .gpu_terrain_traversal = bindless_gpu_scene && capabilities.gpu_terrain_traversal,
                       .hzb_occlusion = gpu_driven && profile.prefer_hzb_occlusion && capabilities.storage_images &&
                                        capabilities.hzb_occlusion,
                       .fxaa = result.anti_aliasing == anti_aliasing_method::fxaa,
                       .temporal_antialiasing = result.anti_aliasing == anti_aliasing_method::taa ||
                                                result.anti_aliasing == anti_aliasing_method::taau,
                       .temporal_upscaling = result.anti_aliasing == anti_aliasing_method::taau,
                       .anti_aliasing = result.anti_aliasing,
                       .async_compute = optional_features && !config.force_disable_async_compute &&
                                        profile.prefer_async_compute && capabilities.dedicated_compute_queue,
                       .virtual_geometry = virtual_geometry_path != virtual_geometry_raster_path::unavailable,
                       .texture_streaming = texture_streaming,
                       .virtual_textures = virtual_textures,
                       .virtual_geometry_path = virtual_geometry_path,
                       .virtual_shadow_maps = virtual_shadow_maps,
                       .virtual_shadow_virtual_geometry = virtual_shadow_virtual_geometry,
                       .screen_space_contact_shadows = screen_space_contact_shadows,
                       .software_ray_tracing = capabilities.software_ray_tracing,
                       .hardware_ray_tracing = capabilities.ray_tracing,
                       .screen_space_gi = screen_space_lighting && result.quality != render_quality_tier::low,
                       .screen_space_reflections = screen_space_lighting && result.quality != render_quality_tier::low,
                       .surface_cache = software_lighting,
                       .radiance_cache = software_lighting,
                       .software_gi = software_lighting,
                       .software_reflections = software_lighting,
                       .hardware_gi = hardware_lighting,
                       .hardware_reflections = hardware_lighting,
                       .sparse_resources = optional_features && capabilities.sparse_resources,
                       .sampler_anisotropy = optional_features && capabilities.sampler_anisotropy,
                       .texture_compression_bc = capabilities.texture_compression_bc,
                       .mesh_shaders = optional_features && capabilities.mesh_shaders,
                       .ray_tracing = capabilities.ray_tracing,
                       .variable_rate_shading = optional_features && capabilities.variable_rate_shading,
                       .submission = submission};

    if (!texture_streaming)
        result.fallback_reasons.push_back(
            "executable mip streaming is unavailable or disabled; streamable textures require resident upload");
    if (!virtual_textures)
        result.fallback_reasons.push_back(
            "virtual-texture feedback or sampling is unavailable or disabled; using conventional mip streaming");

    if (config.force_disable_optional_features)
        result.fallback_reasons.push_back("optional GPU features were disabled by renderer configuration");
    if (!capabilities.dynamic_rendering)
        result.fallback_reasons.push_back("dynamic rendering is unavailable; use the compatibility render-pass path");
    if (!capabilities.synchronization2)
        result.fallback_reasons.push_back("synchronization2 is unavailable; use legacy barriers and submission");
    if (!capabilities.timeline_semaphores)
        result.fallback_reasons.push_back("timeline semaphores are unavailable; use per-frame fences");
    if (!capabilities.descriptor_indexing)
        result.fallback_reasons.push_back("descriptor indexing is unavailable; use classic descriptor sets");
    if (!gpu_scene_supported && profile.prefer_gpu_driven)
        result.fallback_reasons.push_back(
            "GPU-driven rendering requires an executable GPU Scene, compute, storage buffers, shader draw "
            "parameters, and indirect draws; "
            "using CPU visibility");
    else if (gpu_driven && !bindless_gpu_scene)
        result.fallback_reasons.push_back(
            "full GPU multi-draw requires visibility compaction and bindless geometry, material, texture, and "
            "sampler tables; using classic GPU visibility");
    if (result.anti_aliasing != config.anti_aliasing && config.anti_aliasing != anti_aliasing_method::auto_select)
        result.fallback_reasons.push_back("requested anti-aliasing is unavailable; using an executable fallback");
    if (config.force_disable_gpu_driven)
        result.fallback_reasons.push_back("GPU-driven rendering was disabled by renderer configuration");
    if (submission == gpu_submission_path::cpu_direct)
        result.fallback_reasons.push_back("indirect drawing is unavailable or disabled; using CPU draw submission");
    else if (submission == gpu_submission_path::indirect &&
             profile.preferred_submission == gpu_submission_path::indirect_count)
        result.fallback_reasons.push_back(
            "GPU Scene indirect-count submission is unavailable; using fixed indirect draws");
    if (profile.prefer_async_compute && !result.features.async_compute)
        result.fallback_reasons.push_back("dedicated asynchronous compute is unavailable; compute runs on graphics");
    if (virtual_geometry_quality && virtual_geometry_path == virtual_geometry_raster_path::unavailable)
        result.fallback_reasons.push_back(
            "virtual geometry requires executable traversal, HZB, bindless material access, streaming, and a "
            "compute or mesh-shader raster path; using conventional LOD geometry");
    if (result.quality == render_quality_tier::ultra && !virtual_shadow_maps)
        result.fallback_reasons.push_back(
            "virtual shadow maps require executable allocation, feedback, rendering, sampling, compute, and storage "
            "paths; using cascaded and local-atlas shadows");
    if (profile.screen_space_shadows && !screen_space_contact_shadows)
        result.fallback_reasons.push_back(
            "screen-space contact shadows require executable HZB and compute paths; using mapped shadows only");
    if (result.quality == render_quality_tier::medium && !screen_space_lighting)
        result.fallback_reasons.push_back(
            "screen-space GI/reflections require executable HZB, temporal resolve, and storage images; using probes");
    if ((result.quality == render_quality_tier::high || result.quality == render_quality_tier::ultra) &&
        !software_lighting)
        result.fallback_reasons.push_back(
            "software dynamic lighting requires executable distance-field tracing, surface cache, and radiance cache; "
            "using baked lighting and probes");
    if (result.quality == render_quality_tier::ultra && software_lighting && !hardware_lighting)
        result.fallback_reasons.push_back(
            "hardware ray queries are unavailable or disabled; Ultra indirect lighting uses the software trace path");

    return result;
}

void frame_budget_controller::reset(const render_quality_profile& profile, float target_frame_time_ms) noexcept
{
    target_frame_time_ms_ = std::max(1.0f, target_frame_time_ms);
    minimum_scale_ = std::clamp(profile.minimum_render_scale, 0.25f, 1.0f);
    maximum_scale_ = std::clamp(profile.maximum_render_scale, minimum_scale_, 1.0f);
    minimum_geometry_error_ = std::max(0.01f, profile.minimum_geometry_error_threshold);
    maximum_geometry_error_ = std::max(minimum_geometry_error_, profile.maximum_geometry_error_threshold);
    minimum_shadow_scale_ = std::clamp(profile.minimum_shadow_resolution_scale, 0.25f, 1.0f);
    maximum_shadow_scale_ = std::clamp(profile.maximum_shadow_resolution_scale, minimum_shadow_scale_, 1.0f);
    minimum_volumetric_scale_ = std::clamp(profile.minimum_volumetric_resolution_scale, 0.25f, 1.0f);
    maximum_volumetric_scale_ =
        std::clamp(profile.maximum_volumetric_resolution_scale, minimum_volumetric_scale_, 1.0f);
    maximum_gi_trace_budget_ = profile.gi_trace_budget;
    maximum_reflection_ray_budget_ = profile.reflection_ray_budget;
    maximum_lighting_trace_scale_ = profile.lighting_trace_scale;
    maximum_surface_cache_update_budget_ = profile.surface_cache_update_budget;
    maximum_radiance_probe_update_budget_ = profile.radiance_probe_update_budget;
    settings_ = {.render_scale = maximum_scale_,
                 .geometry_error_threshold =
                     std::clamp(profile.geometry_error_threshold, minimum_geometry_error_, maximum_geometry_error_),
                 .shadow_resolution_scale = maximum_shadow_scale_,
                 .volumetric_resolution_scale = maximum_volumetric_scale_,
                 .gi_trace_budget = maximum_gi_trace_budget_,
                 .reflection_ray_budget = maximum_reflection_ray_budget_,
                 .lighting_trace_scale = maximum_lighting_trace_scale_,
                 .surface_cache_update_budget = maximum_surface_cache_update_budget_,
                 .radiance_probe_update_budget = maximum_radiance_probe_update_budget_};
    last_change_ = frame_budget_change::none;
    smoothed_frame_time_ms_ = target_frame_time_ms_;
    over_budget_frames_ = 0;
    under_budget_frames_ = 0;
}

const frame_budget_settings& frame_budget_controller::update(float gpu_frame_time_ms) noexcept
{
    last_change_ = frame_budget_change::none;
    if (!(gpu_frame_time_ms > 0.0f) || !std::isfinite(gpu_frame_time_ms)) return settings_;

    smoothed_frame_time_ms_ += (gpu_frame_time_ms - smoothed_frame_time_ms_) * dynamic_resolution_smoothing;
    if (smoothed_frame_time_ms_ > target_frame_time_ms_ * dynamic_resolution_over_budget_ratio)
    {
        ++over_budget_frames_;
        under_budget_frames_ = 0;
        if (over_budget_frames_ >= dynamic_resolution_over_budget_frames)
        {
            if (settings_.radiance_probe_update_budget > maximum_radiance_probe_update_budget_ / 4u)
            {
                settings_.radiance_probe_update_budget =
                    std::max(maximum_radiance_probe_update_budget_ / 4u, settings_.radiance_probe_update_budget / 2u);
                last_change_ = frame_budget_change::radiance_probe_updates;
            }
            else if (settings_.surface_cache_update_budget > maximum_surface_cache_update_budget_ / 4u)
            {
                settings_.surface_cache_update_budget =
                    std::max(maximum_surface_cache_update_budget_ / 4u, settings_.surface_cache_update_budget / 2u);
                last_change_ = frame_budget_change::surface_cache_updates;
            }
            else if (settings_.reflection_ray_budget > 0)
            {
                --settings_.reflection_ray_budget;
                last_change_ = frame_budget_change::reflection_rays;
            }
            else if (settings_.gi_trace_budget > 0)
            {
                --settings_.gi_trace_budget;
                last_change_ = frame_budget_change::gi_traces;
            }
            else if (settings_.lighting_trace_scale > maximum_lighting_trace_scale_ * 0.5f)
            {
                settings_.lighting_trace_scale =
                    std::max(maximum_lighting_trace_scale_ * 0.5f, settings_.lighting_trace_scale - 0.25f);
                last_change_ = frame_budget_change::lighting_trace_resolution;
            }
            else if (settings_.volumetric_resolution_scale > minimum_volumetric_scale_)
            {
                settings_.volumetric_resolution_scale =
                    std::max(minimum_volumetric_scale_, settings_.volumetric_resolution_scale - 0.125f);
                last_change_ = frame_budget_change::volumetric_resolution;
            }
            else if (settings_.shadow_resolution_scale > minimum_shadow_scale_)
            {
                settings_.shadow_resolution_scale =
                    std::max(minimum_shadow_scale_, settings_.shadow_resolution_scale - 0.125f);
                last_change_ = frame_budget_change::shadow_resolution;
            }
            else if (settings_.geometry_error_threshold < maximum_geometry_error_)
            {
                settings_.geometry_error_threshold =
                    std::min(maximum_geometry_error_, settings_.geometry_error_threshold + 0.25f);
                last_change_ = frame_budget_change::geometry_error;
            }
            else if (settings_.render_scale > minimum_scale_)
            {
                settings_.render_scale =
                    std::max(minimum_scale_, settings_.render_scale - dynamic_resolution_scale_step);
                last_change_ = frame_budget_change::render_scale;
            }
            over_budget_frames_ = 0;
        }
    }
    else if (smoothed_frame_time_ms_ < target_frame_time_ms_ * dynamic_resolution_under_budget_ratio)
    {
        ++under_budget_frames_;
        over_budget_frames_ = 0;
        if (under_budget_frames_ >= dynamic_resolution_under_budget_frames)
        {
            if (settings_.render_scale < maximum_scale_)
            {
                settings_.render_scale =
                    std::min(maximum_scale_, settings_.render_scale + dynamic_resolution_scale_step);
                last_change_ = frame_budget_change::render_scale;
            }
            else if (settings_.geometry_error_threshold > minimum_geometry_error_)
            {
                settings_.geometry_error_threshold =
                    std::max(minimum_geometry_error_, settings_.geometry_error_threshold - 0.25f);
                last_change_ = frame_budget_change::geometry_error;
            }
            else if (settings_.shadow_resolution_scale < maximum_shadow_scale_)
            {
                settings_.shadow_resolution_scale =
                    std::min(maximum_shadow_scale_, settings_.shadow_resolution_scale + 0.125f);
                last_change_ = frame_budget_change::shadow_resolution;
            }
            else if (settings_.volumetric_resolution_scale < maximum_volumetric_scale_)
            {
                settings_.volumetric_resolution_scale =
                    std::min(maximum_volumetric_scale_, settings_.volumetric_resolution_scale + 0.125f);
                last_change_ = frame_budget_change::volumetric_resolution;
            }
            else if (settings_.lighting_trace_scale < maximum_lighting_trace_scale_)
            {
                settings_.lighting_trace_scale =
                    std::min(maximum_lighting_trace_scale_, settings_.lighting_trace_scale + 0.25f);
                last_change_ = frame_budget_change::lighting_trace_resolution;
            }
            else if (settings_.gi_trace_budget < maximum_gi_trace_budget_)
            {
                ++settings_.gi_trace_budget;
                last_change_ = frame_budget_change::gi_traces;
            }
            else if (settings_.reflection_ray_budget < maximum_reflection_ray_budget_)
            {
                ++settings_.reflection_ray_budget;
                last_change_ = frame_budget_change::reflection_rays;
            }
            else if (settings_.surface_cache_update_budget < maximum_surface_cache_update_budget_)
            {
                settings_.surface_cache_update_budget = std::min(
                    maximum_surface_cache_update_budget_, std::max(1u, settings_.surface_cache_update_budget * 2u));
                last_change_ = frame_budget_change::surface_cache_updates;
            }
            else if (settings_.radiance_probe_update_budget < maximum_radiance_probe_update_budget_)
            {
                settings_.radiance_probe_update_budget = std::min(
                    maximum_radiance_probe_update_budget_, std::max(1u, settings_.radiance_probe_update_budget * 2u));
                last_change_ = frame_budget_change::radiance_probe_updates;
            }
            under_budget_frames_ = 0;
        }
    }
    else
    {
        over_budget_frames_ = 0;
        under_budget_frames_ = 0;
    }
    return settings_;
}

const frame_budget_settings& frame_budget_controller::settings() const noexcept
{
    return settings_;
}

frame_budget_change frame_budget_controller::last_change() const noexcept
{
    return last_change_;
}

float frame_budget_controller::smoothed_frame_time_ms() const noexcept
{
    return smoothed_frame_time_ms_;
}

void render_backend::resize_viewport(std::uint32_t, std::uint32_t) {}

void render_backend::configure(const resolved_render_config&) {}

surface_frame_result render_backend::present_surface_frame(std::uint32_t, std::uint32_t)
{
    return surface_frame_result::failure({.code = surface_frame_error_code::unsupported,
                                          .message = "surface presentation is not supported by this backend"});
}

render_viewport_texture render_backend::viewport_texture() const noexcept
{
    return {};
}

render_backend_frame_profile render_backend::last_frame_profile() const
{
    return {};
}

texture_feedback_readback render_backend::take_texture_feedback()
{
    return {};
}

std::vector<texture_stream_upload_result> render_backend::take_texture_stream_upload_results()
{
    return {};
}

void render_backend::request_object_pick(render_object_pick_request) {}

render_object_pick_result render_backend::last_object_pick() const
{
    return {};
}

void render_backend::request_frame_capture(const render_frame_capture_request&) {}

render_frame_capture_result render_backend::last_frame_capture() const
{
    return {};
}

void execute_render_graph(const compiled_render_graph& graph, command_encoder& encoder)
{
    const auto execute_pass = [&](std::uint32_t pass_index)
    {
        for (const auto& transition : graph.transitions)
        {
            if (transition.after_pass == pass_index) encoder.resource_barrier(transition);
        }

        const auto& pass = graph.passes[pass_index];
        encoder.begin_pass(pass);
        render_pass_context context(graph, pass_index, encoder);
        if (pass.record) pass.record(context);
        encoder.end_pass();
    };

    if (graph.submissions.empty())
    {
        for (std::uint32_t pass_index = 0; pass_index < graph.passes.size(); ++pass_index)
            execute_pass(pass_index);
        return;
    }

    for (const auto& submission : graph.submissions)
    {
        encoder.begin_submission(submission);
        for (const auto pass_index : submission.passes)
            execute_pass(pass_index);
        encoder.end_submission(submission);
    }
}

renderer::renderer(renderer_config config) : config_(config) {}

void renderer::set_backend(std::unique_ptr<render_backend> backend)
{
    gpu_scene_.reset();
    temporal_views_.clear();
    backend_ = std::move(backend);
    if (backend_)
    {
        resolved_config_ = resolve_render_config(config_, backend_->capabilities());
        frame_budget_.reset(quality_profile(resolved_config_.quality), resolved_config_.target_frame_time_ms);
        const auto& budget = frame_budget_.settings();
        resolved_config_.render_scale = budget.render_scale;
        resolved_config_.geometry_error_threshold = budget.geometry_error_threshold;
        resolved_config_.shadow_resolution_scale = budget.shadow_resolution_scale;
        resolved_config_.volumetric_resolution_scale = budget.volumetric_resolution_scale;
        resolved_config_.gi_trace_budget = budget.gi_trace_budget;
        resolved_config_.reflection_ray_budget = budget.reflection_ray_budget;
        resolved_config_.lighting_trace_scale = budget.lighting_trace_scale;
        resolved_config_.surface_cache_update_budget = budget.surface_cache_update_budget;
        resolved_config_.radiance_probe_update_budget = budget.radiance_probe_update_budget;
        virtual_geometry_residency_config residency;
        if (resolved_config_.quality == render_quality_tier::ultra)
        {
            residency.gpu_budget_bytes = 1024ull * 1024ull * 1024ull;
            residency.compressed_cpu_budget_bytes = 512ull * 1024ull * 1024ull;
        }
        if (const auto device_budget = backend_->capabilities().memory_budget; device_budget != 0)
            residency.gpu_budget_bytes = std::min(residency.gpu_budget_bytes, device_budget / 10u);
        virtual_geometry_residency_.configure(residency);
        const std::uint64_t texture_quality_budget =
            resolved_config_.quality == render_quality_tier::low      ? 256ull * mebibyte
            : resolved_config_.quality == render_quality_tier::medium ? 512ull * mebibyte
            : resolved_config_.quality == render_quality_tier::high   ? 1024ull * mebibyte
                                                                      : 2048ull * mebibyte;
        const auto adapter_memory = backend_->capabilities().memory_budget != 0
                                        ? backend_->capabilities().memory_budget
                                        : backend_->capabilities().dedicated_video_memory;
        std::uint64_t texture_gpu_budget =
            config_.texture_gpu_budget_bytes != 0 ? config_.texture_gpu_budget_bytes : texture_quality_budget;
        if (adapter_memory != 0)
            texture_gpu_budget =
                std::min(texture_gpu_budget, std::max(std::uint64_t{128} * mebibyte, adapter_memory / 4u));
        const auto quality_upload_budget = resolved_config_.quality == render_quality_tier::low     ? 32ull * mebibyte
                                           : resolved_config_.quality == render_quality_tier::ultra ? 128ull * mebibyte
                                                                                                    : 64ull * mebibyte;
        texture_residency_.configure(
            {.gpu_budget_bytes = texture_gpu_budget,
             .cpu_cache_budget_bytes = config_.texture_cpu_cache_budget_bytes != 0
                                           ? config_.texture_cpu_cache_budget_bytes
                                           : std::min(std::uint64_t{256} * mebibyte, texture_gpu_budget / 4u),
             .upload_budget_per_frame = config_.texture_upload_budget_per_frame != 0
                                            ? config_.texture_upload_budget_per_frame
                                            : quality_upload_budget,
             .maximum_requests_per_frame = 2048,
             .protected_frame_count = 30});
        lighting_scene_config lighting_config;
        lighting_config.gpu_budget_bytes = resolved_config_.lighting_scene_gpu_budget_bytes;
        lighting_config.compressed_cpu_budget_bytes = resolved_config_.lighting_scene_gpu_budget_bytes / 3u;
        lighting_config.maximum_surface_updates_per_frame = resolved_config_.surface_cache_update_budget;
        lighting_config.maximum_radiance_probe_updates_per_frame = resolved_config_.radiance_probe_update_budget;
        lighting_scene_.configure(lighting_config);
        backend_->configure(resolved_config_);
    }
}

render_backend* renderer::backend() noexcept
{
    return backend_.get();
}

const render_backend* renderer::backend() const noexcept
{
    return backend_.get();
}

const renderer_config& renderer::config() const noexcept
{
    return config_;
}

const resolved_render_config& renderer::resolved_config() const noexcept
{
    return resolved_config_;
}

gpu_resource_tables& renderer::gpu_resources() noexcept
{
    return gpu_resources_;
}

const gpu_resource_tables& renderer::gpu_resources() const noexcept
{
    return gpu_resources_;
}

anti_aliasing_method renderer::resolve_view_anti_aliasing(anti_aliasing_method requested) const noexcept
{
    if (!backend_) return anti_aliasing_method::disabled;
    return resolve_anti_aliasing(requested, resolved_config_.path, resolved_config_.render_scale,
                                 backend_->capabilities());
}

render_frame_queue& renderer::frame_queue() noexcept
{
    return frame_queue_;
}

mesh_handle renderer::create_mesh(mesh_data mesh)
{
    const mesh_handle handle = mesh_handles_.allocate();
    auto shared_mesh = std::make_shared<mesh_data>(std::move(mesh));
    mesh_data_[renderer_resource_key(handle)] = shared_mesh;

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.mesh_upload(handle, shared_mesh, shared_mesh->name);
    auto table_update = gpu_resources_.publish_geometry(
        handle, std::as_bytes(std::span{shared_mesh->vertices}), sizeof(mesh_vertex),
        std::as_bytes(std::span{shared_mesh->indices}), sizeof(std::uint32_t), resource_frame_index_);
    if (!table_update.updates.empty())
        writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    if (shared_mesh->usage == mesh_usage::static_gpu && !shared_mesh->vertices.empty() && !shared_mesh->indices.empty())
    {
        auto built = build_lighting_geometry(*shared_mesh);
        const auto lighting_handle = lighting_geometry_handles_.allocate();
        auto lighting = std::make_shared<lighting_geometry_descriptor>(std::move(built.geometry));
        mesh_lighting_geometry_[renderer_resource_key(handle)] = lighting_handle;
        lighting_geometry_data_[renderer_resource_key(lighting_handle)] = lighting;
        buffer.push({.payload = lighting_geometry_upload_event{
                         .handle = lighting_handle, .geometry = std::move(lighting), .label = shared_mesh->name}});
    }
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_mesh_vertices(mesh_handle handle, std::vector<mesh_vertex> vertices)
{
    if (!mesh_handles_.alive(handle)) return false;
    const auto found = mesh_data_.find(renderer_resource_key(handle));
    if (found == mesh_data_.end() || !found->second || found->second->vertices.size() != vertices.size()) return false;
    auto replacement = std::make_shared<mesh_data>(*found->second);
    replacement->vertices = std::move(vertices);
    found->second = replacement;
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.mesh_upload(handle, replacement, replacement->name);
    auto table_update = gpu_resources_.publish_geometry(
        handle, std::as_bytes(std::span{replacement->vertices}), sizeof(mesh_vertex),
        std::as_bytes(std::span{replacement->indices}), sizeof(std::uint32_t), resource_frame_index_);
    if (!table_update.updates.empty())
        writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    const auto mesh_key = renderer_resource_key(handle);
    const auto lighting_entry = mesh_lighting_geometry_.find(mesh_key);
    if (replacement->usage == mesh_usage::static_gpu && lighting_entry != mesh_lighting_geometry_.end())
    {
        auto built = build_lighting_geometry(*replacement);
        auto lighting = std::make_shared<lighting_geometry_descriptor>(std::move(built.geometry));
        auto& generation = lighting->generation;
        const auto previous = lighting_geometry_data_.find(renderer_resource_key(lighting_entry->second));
        generation = previous == lighting_geometry_data_.end() ? 1u : previous->second->generation + 1u;
        lighting_geometry_data_[renderer_resource_key(lighting_entry->second)] = lighting;
        buffer.push({.payload = lighting_geometry_upload_event{.handle = lighting_entry->second,
                                                               .geometry = std::move(lighting),
                                                               .label = replacement->name}});
    }
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::destroy_mesh(mesh_handle handle)
{
    if (!mesh_handles_.release(handle)) return false;
    const auto mesh_key = renderer_resource_key(handle);
    mesh_data_.erase(mesh_key);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.mesh_destroy(handle);
    auto table_update =
        gpu_resources_.tombstone(gpu_resource_table_kind::geometry, handle, resource_frame_index_);
    if (!table_update.updates.empty())
        writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    if (const auto found = mesh_lighting_geometry_.find(mesh_key); found != mesh_lighting_geometry_.end())
    {
        const auto lighting_handle = found->second;
        mesh_lighting_geometry_.erase(found);
        lighting_geometry_data_.erase(renderer_resource_key(lighting_handle));
        (void)lighting_geometry_handles_.release(lighting_handle);
        buffer.push({.payload = lighting_geometry_destroy_event{.handle = lighting_handle}});
    }
    frame_queue_.submit(std::move(buffer));
    return true;
}

virtual_mesh_handle renderer::create_virtual_mesh(virtual_mesh_data mesh)
{
    const virtual_mesh_handle handle = virtual_mesh_handles_.allocate();
    auto shared_mesh = std::make_shared<virtual_mesh_data>(std::move(mesh));
    const auto key = renderer_resource_key(handle);
    virtual_mesh_data_[key] = shared_mesh;
    virtual_mesh_content_generations_[key] = 1;
    virtual_geometry_residency_.register_resource(handle, *shared_mesh, 1);

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.virtual_mesh_upload(handle, shared_mesh, 1, "virtual mesh");
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_virtual_mesh(virtual_mesh_handle handle, virtual_mesh_data mesh)
{
    if (!virtual_mesh_handles_.alive(handle)) return false;
    auto shared_mesh = std::make_shared<virtual_mesh_data>(std::move(mesh));
    const auto key = renderer_resource_key(handle);
    virtual_mesh_data_[key] = shared_mesh;
    auto& content_generation = virtual_mesh_content_generations_[key];
    ++content_generation;
    if (content_generation == 0) content_generation = 1;
    virtual_geometry_residency_.register_resource(handle, *shared_mesh, content_generation);

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.virtual_mesh_upload(handle, shared_mesh, content_generation, "virtual mesh update");
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::destroy_virtual_mesh(virtual_mesh_handle handle)
{
    if (!virtual_mesh_handles_.release(handle)) return false;
    virtual_geometry_residency_.unregister_resource(handle);
    const auto key = renderer_resource_key(handle);
    virtual_mesh_data_.erase(key);
    virtual_mesh_content_generations_.erase(key);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.virtual_mesh_destroy(handle);
    frame_queue_.submit(std::move(buffer));
    return true;
}

surface_frame_result render_backend::create_viewport_output(const viewport_output_descriptor&)
{
    return surface_frame_result::failure(
        {.code = surface_frame_error_code::unsupported, .message = "viewport outputs are unsupported"});
}

surface_frame_result render_backend::resize_viewport_output(std::string_view, std::uint32_t, std::uint32_t)
{
    return surface_frame_result::failure(
        {.code = surface_frame_error_code::unsupported, .message = "viewport output resize is unsupported"});
}

surface_frame_result render_backend::present_viewport_output(std::string_view)
{
    return surface_frame_result::failure(
        {.code = surface_frame_error_code::unsupported, .message = "viewport output presentation is unsupported"});
}

shared_viewport_frame_result render_backend::poll_viewport_output(std::string_view)
{
    return shared_viewport_frame_result::failure(
        {.code = surface_frame_error_code::unsupported, .message = "shared viewport output is unsupported"});
}

void render_backend::release_viewport_frame(std::string_view, std::uint64_t, std::uint64_t) {}

void render_backend::set_viewport_output_visible(std::string_view, bool) {}

void render_backend::destroy_viewport_output(std::string_view) {}

terrain_handle renderer::create_terrain(terrain_resource_descriptor terrain)
{
    if (terrain.sample_resolution < 2u ||
        terrain.heights.size() != static_cast<std::size_t>(terrain.sample_resolution) * terrain.sample_resolution ||
        terrain.weights.size() != terrain.heights.size())
        return {};
    if (terrain.hierarchy.root >= terrain.hierarchy.nodes.size())
        terrain.hierarchy = build_terrain_hierarchy(terrain.heights, terrain.sample_resolution, terrain.width,
                                                    terrain.depth, terrain.lod);
    if (terrain.hierarchy.root >= terrain.hierarchy.nodes.size()) return {};
    terrain.local_bounds = terrain.hierarchy.nodes[terrain.hierarchy.root].local_bounds;
    const terrain_handle handle = terrain_handles_.allocate();
    const auto key = renderer_resource_key(handle);
    auto shared = std::make_shared<terrain_resource_descriptor>(std::move(terrain));
    terrain_data_[key] = shared;
    terrain_snapshots_[key] = {.handle = handle,
                               .sample_resolution = shared->sample_resolution,
                               .hierarchy_nodes = static_cast<std::uint32_t>(shared->hierarchy.nodes.size()),
                               .hierarchy_leaves = shared->hierarchy.leaf_count,
                               .local_bounds = shared->local_bounds,
                               .height_bytes = shared->heights.size() * sizeof(float),
                               .weight_bytes = shared->weights.size() * sizeof(shared->weights[0]),
                               .uploaded_height_bytes = shared->heights.size() * sizeof(float),
                               .uploaded_weight_bytes = shared->weights.size() * sizeof(shared->weights[0]),
                               .content_revision = shared->content_revision,
                               .valid = true};
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.terrain_upload(handle, shared, shared->name);
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_terrain(terrain_handle handle, material_handle material, terrain_lod_settings settings,
                              std::uint64_t content_revision)
{
    if (!terrain_handles_.alive(handle)) return false;
    const auto key = renderer_resource_key(handle);
    const auto found = terrain_data_.find(key);
    if (found == terrain_data_.end()) return false;
    auto replacement = std::make_shared<terrain_resource_descriptor>(*found->second);
    replacement->material = material;
    replacement->lod = settings;
    replacement->content_revision = content_revision;
    replacement->hierarchy = build_terrain_hierarchy(replacement->heights, replacement->sample_resolution,
                                                     replacement->width, replacement->depth, replacement->lod);
    replacement->local_bounds = replacement->hierarchy.nodes[replacement->hierarchy.root].local_bounds;
    terrain_data_[key] = replacement;
    auto& snapshot = terrain_snapshots_[key];
    snapshot.hierarchy_nodes = static_cast<std::uint32_t>(replacement->hierarchy.nodes.size());
    snapshot.hierarchy_leaves = replacement->hierarchy.leaf_count;
    snapshot.local_bounds = replacement->local_bounds;
    snapshot.content_revision = content_revision;
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.terrain_upload(handle, replacement, replacement->name);
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::update_terrain_heights(terrain_handle handle, terrain_height_region_update update)
{
    if (!terrain_handles_.alive(handle) || !update.region.valid()) return false;
    const auto key = renderer_resource_key(handle);
    const auto found = terrain_data_.find(key);
    if (found == terrain_data_.end()) return false;
    auto replacement = std::make_shared<terrain_resource_descriptor>(*found->second);
    if (update.region.max_x >= replacement->sample_resolution ||
        update.region.max_z >= replacement->sample_resolution || update.row_stride < update.region.width() ||
        update.values.size() < static_cast<std::size_t>(update.row_stride) * update.region.height())
        return false;
    for (std::uint32_t z = 0; z < update.region.height(); ++z)
        std::copy_n(update.values.begin() + static_cast<std::ptrdiff_t>(z * update.row_stride), update.region.width(),
                    replacement->heights.begin() +
                        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(update.region.min_z + z) *
                                                        replacement->sample_resolution +
                                                    update.region.min_x));
    update_terrain_hierarchy(replacement->hierarchy, replacement->heights, replacement->sample_resolution,
                             replacement->width, replacement->depth, update.region, replacement->lod);
    replacement->local_bounds = replacement->hierarchy.nodes[replacement->hierarchy.root].local_bounds;
    replacement->content_revision = update.content_revision;
    terrain_data_[key] = replacement;
    auto& snapshot = terrain_snapshots_[key];
    snapshot.local_bounds = replacement->local_bounds;
    snapshot.content_revision = update.content_revision;
    snapshot.uploaded_height_bytes +=
        static_cast<std::uint64_t>(update.region.width()) * update.region.height() * sizeof(float);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.terrain_height_update(handle, std::make_shared<terrain_height_region_update>(std::move(update)));
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::update_terrain_weights(terrain_handle handle, terrain_weight_region_update update)
{
    if (!terrain_handles_.alive(handle) || !update.region.valid()) return false;
    const auto key = renderer_resource_key(handle);
    const auto found = terrain_data_.find(key);
    if (found == terrain_data_.end()) return false;
    auto replacement = std::make_shared<terrain_resource_descriptor>(*found->second);
    if (update.region.max_x >= replacement->sample_resolution ||
        update.region.max_z >= replacement->sample_resolution || update.row_stride < update.region.width() ||
        update.values.size() < static_cast<std::size_t>(update.row_stride) * update.region.height())
        return false;
    for (std::uint32_t z = 0; z < update.region.height(); ++z)
        std::copy_n(update.values.begin() + static_cast<std::ptrdiff_t>(z * update.row_stride), update.region.width(),
                    replacement->weights.begin() +
                        static_cast<std::ptrdiff_t>(static_cast<std::size_t>(update.region.min_z + z) *
                                                        replacement->sample_resolution +
                                                    update.region.min_x));
    replacement->content_revision = update.content_revision;
    terrain_data_[key] = replacement;
    auto& snapshot = terrain_snapshots_[key];
    snapshot.content_revision = update.content_revision;
    snapshot.uploaded_weight_bytes +=
        static_cast<std::uint64_t>(update.region.width()) * update.region.height() * sizeof(replacement->weights[0]);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.terrain_weight_update(handle, std::make_shared<terrain_weight_region_update>(std::move(update)));
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::destroy_terrain(terrain_handle handle)
{
    if (!terrain_handles_.release(handle)) return false;
    const auto key = renderer_resource_key(handle);
    terrain_data_.erase(key);
    terrain_snapshots_.erase(key);
    terrain_selection_scratch_.erase(key);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.terrain_destroy(handle);
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::terrain_alive(terrain_handle handle) const noexcept
{
    return terrain_handles_.alive(handle);
}

const terrain_resource_descriptor* renderer::terrain_data_for(terrain_handle handle) const noexcept
{
    if (!terrain_handles_.alive(handle)) return nullptr;
    const auto found = terrain_data_.find(renderer_resource_key(handle));
    return found == terrain_data_.end() ? nullptr : found->second.get();
}

terrain_resource_snapshot renderer::terrain_snapshot(terrain_handle handle) const noexcept
{
    const auto found = terrain_snapshots_.find(renderer_resource_key(handle));
    return found == terrain_snapshots_.end() ? terrain_resource_snapshot{} : found->second;
}

geometry_resource_handle renderer::create_geometry_resource(virtual_mesh_data geometry, std::uint32_t asset_generation)
{
    geometry_resource_handle result;
    result.asset_generation = asset_generation;
    const auto lod_count = std::min<std::size_t>(geometry.conventional_lods.size(), result.conventional_lods.size());
    for (std::size_t index = 0; index < lod_count; ++index)
    {
        mesh_data lod;
        lod.name = "cooked geometry LOD " + std::to_string(index);
        lod.vertices = geometry.conventional_lods[index].vertices;
        lod.indices = geometry.conventional_lods[index].indices;
        const auto handle = create_mesh(std::move(lod));
        result.conventional_lods[index] = handle;
        result.conventional_lod_errors[index] = geometry.conventional_lods[index].geometric_error;
        if (index == 0) result.conventional = handle;
    }
    result.conventional_lod_count = static_cast<std::uint8_t>(lod_count);
    if (!geometry.clusters.empty() && !geometry.pages.empty())
        result.virtualized = create_virtual_mesh(std::move(geometry));
    return result;
}

bool renderer::destroy_geometry_resource(const geometry_resource_handle& geometry)
{
    bool changed{};
    for (std::size_t index = 0; index < geometry.conventional_lod_count && index < geometry.conventional_lods.size();
         ++index)
    {
        const auto handle = geometry.conventional_lods[index];
        if (!handle.valid()) continue;
        bool duplicate{};
        for (std::size_t previous = 0; previous < index; ++previous)
            duplicate = duplicate || geometry.conventional_lods[previous] == handle;
        if (!duplicate) changed = destroy_mesh(handle) || changed;
    }
    if (geometry.virtualized.valid()) changed = destroy_virtual_mesh(geometry.virtualized) || changed;
    return changed;
}

texture_handle renderer::create_texture(texture_data texture)
{
    const texture_handle handle = texture_handles_.allocate();
    auto shared_texture = std::make_shared<texture_data>(std::move(texture));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.texture_upload(handle, shared_texture, shared_texture->name);
    auto table_update =
        gpu_resources_.publish_texture(handle, texture_table_record(handle, *shared_texture), resource_frame_index_);
    writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_texture(texture_handle handle, texture_data texture)
{
    if (!texture_handles_.alive(handle)) return false;

    auto shared_texture = std::make_shared<texture_data>(std::move(texture));
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.texture_upload(handle, shared_texture, shared_texture->name);
    auto table_update =
        gpu_resources_.publish_texture(handle, texture_table_record(handle, *shared_texture), resource_frame_index_);
    writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    frame_queue_.submit(std::move(buffer));
    return true;
}

texture_handle renderer::create_streamed_texture(streamed_texture_descriptor descriptor)
{
    if (!valid_streamed_texture_descriptor(descriptor)) return {};
    const texture_handle handle = texture_handles_.allocate();
    auto shared_descriptor = std::make_shared<streamed_texture_descriptor>(std::move(descriptor));
    streamed_texture_data_[renderer_resource_key(handle)] = shared_descriptor;
    texture_residency_.register_resource(handle, *shared_descriptor);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.texture_stream_register(handle, shared_descriptor, shared_descriptor->texture.name);
    const auto flags = 1u | (shared_descriptor->mode == texture_streaming_mode::virtual_tiles ? 2u : 0u);
    auto table_update = gpu_resources_.publish_texture(
        handle,
        texture_table_record(handle, shared_descriptor->texture, shared_descriptor->artifact.tail_first_mip, flags),
        resource_frame_index_);
    writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_streamed_texture(texture_handle handle, streamed_texture_descriptor descriptor)
{
    if (!texture_handles_.alive(handle) || !valid_streamed_texture_descriptor(descriptor)) return false;
    const auto key = renderer_resource_key(handle);
    const auto previous = streamed_texture_data_.find(key);
    if (previous == streamed_texture_data_.end()) return false;
    descriptor.content_generation = std::max(descriptor.content_generation, previous->second->content_generation + 1u);
    auto shared_descriptor = std::make_shared<streamed_texture_descriptor>(std::move(descriptor));
    previous->second = shared_descriptor;
    texture_residency_.register_resource(handle, *shared_descriptor);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.texture_stream_register(handle, shared_descriptor, shared_descriptor->texture.name);
    const auto flags = 1u | (shared_descriptor->mode == texture_streaming_mode::virtual_tiles ? 2u : 0u);
    auto table_update = gpu_resources_.publish_texture(
        handle,
        texture_table_record(handle, shared_descriptor->texture, shared_descriptor->artifact.tail_first_mip, flags),
        resource_frame_index_);
    writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::destroy_texture(texture_handle handle)
{
    if (!texture_handles_.release(handle)) return false;
    texture_residency_.unregister_resource(handle);
    streamed_texture_data_.erase(renderer_resource_key(handle));
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.texture_destroy(handle);
    auto table_update = gpu_resources_.tombstone(gpu_resource_table_kind::texture, handle, resource_frame_index_);
    if (!table_update.updates.empty())
        writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    frame_queue_.submit(std::move(buffer));
    return true;
}

texture_residency_manager& renderer::texture_residency() noexcept
{
    return texture_residency_;
}

const texture_residency_manager& renderer::texture_residency() const noexcept
{
    return texture_residency_;
}

void renderer::submit_texture_feedback(const texture_feedback_readback& feedback)
{
    texture_residency_.request(feedback.mips, feedback.tiles);
    texture_residency_.note_feedback_overflow(feedback.overflow);
}

std::vector<texture_stream_load> renderer::take_texture_stream_loads(std::uint32_t maximum_requests)
{
    auto result = texture_residency_.take_load_requests(maximum_requests);
    for (const auto& load : result)
        texture_residency_.mark_loading(load);
    return result;
}

bool renderer::publish_texture_subresource(texture_stream_upload upload)
{
    if (!upload.resource.valid() || !upload.bytes || upload.bytes->empty()) return false;
    const auto resource = streamed_texture_data_.find(renderer_resource_key(upload.resource));
    if (resource == streamed_texture_data_.end() || resource->second->content_generation != upload.content_generation)
        return false;
    const auto& artifact = resource->second->artifact;
    std::uint32_t expected_size{};
    std::uint64_t expected_hash{};
    if (upload.kind == texture_subresource_kind::mip)
    {
        if (upload.mip >= artifact.mips.size()) return false;
        expected_size = artifact.mips[upload.mip].stored_size;
        expected_hash = artifact.mips[upload.mip].content_hash;
    }
    else
    {
        const auto found =
            std::find_if(artifact.tiles.begin(), artifact.tiles.end(), [&](const texture_artifact_tile_range& range)
                         { return range.mip == upload.mip && range.x == upload.x && range.y == upload.y; });
        if (found == artifact.tiles.end()) return false;
        expected_size = found->stored_size;
        expected_hash = found->content_hash;
    }
    if (upload.bytes->size() != expected_size || texture_content_hash(*upload.bytes) != expected_hash) return false;
    upload.stored_bytes = expected_size;
    texture_residency_.mark_uploading(upload);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.texture_stream_upload(std::move(upload));
    frame_queue_.submit(std::move(buffer));
    return true;
}

void renderer::fail_texture_subresource(const texture_stream_load& load)
{
    texture_residency_.fail(load);
}

material_handle renderer::create_material(material_descriptor material)
{
    const material_handle handle = material_handles_.allocate();
    material.handle = handle;
    material.render_path = resolve_material_render_path(material);
    auto shared_material = std::make_shared<material_descriptor>(std::move(material));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.material_upload(handle, shared_material, shared_material->name);
    auto table_update = gpu_resources_.publish_material(
        handle, material_table_record(*shared_material, gpu_resources_), resource_frame_index_);
    writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_material(material_handle handle, material_descriptor material)
{
    if (!material_handles_.alive(handle)) return false;

    material.handle = handle;
    material.render_path = resolve_material_render_path(material);
    auto shared_material = std::make_shared<material_descriptor>(std::move(material));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.material_upload(handle, shared_material, shared_material->name);
    auto table_update = gpu_resources_.publish_material(
        handle, material_table_record(*shared_material, gpu_resources_), resource_frame_index_);
    writer.gpu_resource_table_update(std::make_shared<gpu_table_update_batch>(std::move(table_update)));
    frame_queue_.submit(std::move(buffer));
    return true;
}

environment_handle renderer::create_environment(environment_descriptor environment)
{
    const environment_handle handle = environment_handles_.allocate();
    environment.handle = handle;
    auto shared_environment = std::make_shared<environment_descriptor>(std::move(environment));

    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.environment_upload(handle, shared_environment, shared_environment->name);
    frame_queue_.submit(std::move(buffer));
    return handle;
}

bool renderer::update_environment(environment_handle handle, environment_descriptor environment)
{
    if (!environment_handles_.alive(handle)) return false;
    environment.handle = handle;
    auto shared_environment = std::make_shared<environment_descriptor>(std::move(environment));
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.environment_upload(handle, shared_environment, shared_environment->name);
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::destroy_environment(environment_handle handle)
{
    if (!environment_handles_.release(handle)) return false;
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.environment_destroy(handle);
    frame_queue_.submit(std::move(buffer));
    return true;
}

bool renderer::mesh_alive(mesh_handle handle) const
{
    return mesh_handles_.alive(handle);
}

const mesh_data* renderer::mesh_data_for(mesh_handle handle) const
{
    if (!mesh_handles_.alive(handle)) return nullptr;
    const auto found = mesh_data_.find(renderer_resource_key(handle));
    return found == mesh_data_.end() ? nullptr : found->second.get();
}

bool renderer::virtual_mesh_alive(virtual_mesh_handle handle) const
{
    return virtual_mesh_handles_.alive(handle);
}

const virtual_mesh_data* renderer::virtual_mesh_data_for(virtual_mesh_handle handle) const
{
    if (!virtual_mesh_handles_.alive(handle)) return nullptr;
    const auto found = virtual_mesh_data_.find(renderer_resource_key(handle));
    return found == virtual_mesh_data_.end() ? nullptr : found->second.get();
}

lighting_geometry_handle renderer::lighting_geometry_for(mesh_handle handle) const noexcept
{
    if (!mesh_handles_.alive(handle)) return {};
    const auto found = mesh_lighting_geometry_.find(renderer_resource_key(handle));
    return found == mesh_lighting_geometry_.end() ? lighting_geometry_handle{} : found->second;
}

const lighting_geometry_descriptor* renderer::lighting_geometry_data_for(lighting_geometry_handle handle) const noexcept
{
    if (!lighting_geometry_handles_.alive(handle)) return nullptr;
    const auto found = lighting_geometry_data_.find(renderer_resource_key(handle));
    return found == lighting_geometry_data_.end() ? nullptr : found->second.get();
}

std::uint32_t renderer::virtual_mesh_content_generation(virtual_mesh_handle handle) const noexcept
{
    if (!virtual_mesh_handles_.alive(handle)) return 0;
    const auto found = virtual_mesh_content_generations_.find(renderer_resource_key(handle));
    return found == virtual_mesh_content_generations_.end() ? 0u : found->second;
}

virtual_geometry_residency_manager& renderer::virtual_geometry_residency() noexcept
{
    return virtual_geometry_residency_;
}

const virtual_geometry_residency_manager& renderer::virtual_geometry_residency() const noexcept
{
    return virtual_geometry_residency_;
}

void renderer::submit_virtual_geometry_feedback(const virtual_geometry_feedback_readback& feedback)
{
    virtual_geometry_residency_.request_gpu(feedback.page_requests);
    for (std::uint32_t index = 0; index < feedback.overflow.fallback_instance_count; ++index)
        virtual_geometry_residency_.note_parent_fallback();
}

std::vector<virtual_geometry_page_load> renderer::take_virtual_geometry_page_loads()
{
    auto result = virtual_geometry_residency_.take_load_requests();
    for (const auto& load : result)
        virtual_geometry_residency_.mark_loading(load.resource, load.resource_generation, load.page_index);
    return result;
}

bool renderer::publish_virtual_geometry_page(virtual_geometry_page_upload upload)
{
    if (!upload.resource.valid() || !upload.decoded_bytes || upload.decoded_bytes->empty()) return false;
    const auto key = renderer_resource_key(upload.resource);
    const auto generation = virtual_mesh_content_generations_.find(key);
    if (generation == virtual_mesh_content_generations_.end() || generation->second != upload.resource_generation)
        return false;
    const auto geometry = virtual_mesh_data_.find(key);
    if (geometry == virtual_mesh_data_.end() || upload.page_index >= geometry->second->pages.size() ||
        upload.decoded_bytes->size() != geometry->second->pages[upload.page_index].uncompressed_size)
        return false;

    virtual_geometry_residency_.publish(upload.resource, upload.resource_generation, upload.page_index,
                                        static_cast<std::uint32_t>(upload.decoded_bytes->size()),
                                        upload.compressed_cpu_bytes);
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    writer.virtual_geometry_page_upload(std::move(upload));
    frame_queue_.submit(std::move(buffer));
    return true;
}

void renderer::fail_virtual_geometry_page(virtual_mesh_handle resource, std::uint32_t generation,
                                          std::uint32_t page_index)
{
    virtual_geometry_residency_.fail(resource, generation, page_index);
}

lighting_scene& renderer::indirect_lighting_scene() noexcept
{
    return lighting_scene_;
}

const lighting_scene& renderer::indirect_lighting_scene() const noexcept
{
    return lighting_scene_;
}

bool renderer::texture_alive(texture_handle handle) const
{
    return texture_handles_.alive(handle);
}

bool renderer::material_alive(material_handle handle) const
{
    return material_handles_.alive(handle);
}

bool renderer::environment_alive(environment_handle handle) const
{
    return environment_handles_.alive(handle);
}

void renderer::resize_viewport(std::uint32_t width, std::uint32_t height)
{
    viewport_width_ = width;
    viewport_height_ = height;
    if (backend_) backend_->resize_viewport(width, height);
}

render_viewport_texture renderer::viewport_texture() const noexcept
{
    if (!backend_) return {};
    return backend_->viewport_texture();
}

render_backend_frame_profile renderer::last_frame_profile() const
{
    if (!backend_) return {};
    auto result = backend_->last_frame_profile();
    const auto residency = virtual_geometry_residency_.snapshot();
    result.virtual_geometry.enabled = resolved_config_.features.virtual_geometry;
    result.virtual_geometry.raster_path = resolved_config_.features.virtual_geometry_path;
    result.virtual_geometry.requested_pages = residency.requested_pages;
    result.virtual_geometry.failed_pages = residency.failed_pages;
    result.virtual_geometry.evicted_pages = residency.evictions;
    result.virtual_geometry.stale_page_requests = residency.stale_requests;
    result.virtual_geometry.parent_fallbacks = residency.parent_fallbacks;
    result.virtual_geometry.resident_bytes = residency.gpu_resident_bytes;
    result.virtual_geometry.residency_budget_bytes = residency.gpu_budget_bytes;
    if (!result.virtual_geometry.enabled && result.virtual_geometry.fallback_reason.empty())
        result.virtual_geometry.fallback_reason =
            "virtual geometry is unavailable for the resolved renderer configuration; using conventional LODs";
    const auto textures = texture_residency_.snapshot();
    result.texture_streaming = {.gpu_budget_bytes = textures.gpu_budget_bytes,
                                .gpu_resident_bytes = textures.gpu_resident_bytes,
                                .cpu_budget_bytes = textures.cpu_cache_budget_bytes,
                                .cpu_cached_bytes = textures.cpu_cached_bytes,
                                .upload_budget_bytes = textures.upload_budget_per_frame,
                                .uploaded_bytes = textures.uploaded_bytes,
                                .streamed_textures = textures.streamed_mip_resources,
                                .virtual_textures = textures.virtual_texture_resources,
                                .resident_mips = textures.resident_mips,
                                .resident_pages = textures.resident_tiles,
                                .requested_mips = textures.requested_mips,
                                .requested_pages = textures.requested_tiles,
                                .failed_mips = textures.failed_mips,
                                .failed_pages = textures.failed_tiles,
                                .requested_subresources = textures.requested_subresources,
                                .failed_subresources = textures.failed_subresources,
                                .evictions = textures.evictions,
                                .deduplicated_requests = textures.deduplicated_requests,
                                .stale_requests = textures.stale_requests,
                                .feedback_overflow = textures.feedback_overflow,
                                .parent_fallbacks = textures.parent_fallbacks,
                                .io_read_bytes = texture_streaming_io_.read_bytes,
                                .io_failed_bytes = texture_streaming_io_.failed_bytes,
                                .io_completed_reads = texture_streaming_io_.completed_reads,
                                .io_failed_reads = texture_streaming_io_.failed_reads,
                                .io_in_flight_reads = texture_streaming_io_.in_flight_reads,
                                .io_latency_milliseconds = texture_streaming_io_.total_read_latency_milliseconds,
                                .upload_latency_milliseconds = textures.upload_latency_milliseconds,
                                .cache_hit_rate = textures.cache_hit_rate,
                                .over_budget = textures.over_budget,
                                .fallback_reason = !resolved_config_.features.texture_streaming
                                                       ? "texture streaming unavailable or disabled"
                                                   : !resolved_config_.features.virtual_textures
                                                       ? "virtual textures use conventional mip streaming"
                                                       : std::string{}};
    const auto lighting = lighting_scene_.snapshot();
    result.indirect_lighting = {.enabled = resolved_config_.indirect_lighting_path != lighting_trace_path::baked_probe,
                                .trace_path = resolved_config_.indirect_lighting_path,
                                .trace_scale = resolved_config_.lighting_trace_scale,
                                .gi_rays = resolved_config_.gi_trace_budget,
                                .reflection_rays = resolved_config_.reflection_ray_budget,
                                .surface_cards = lighting.surface_cards,
                                .resident_surface_pages = lighting.resident_surface_pages,
                                .resident_distance_field_pages = lighting.resident_distance_field_pages,
                                .dirty_regions = lighting.dirty_regions,
                                .surface_updates = lighting.surface_updates,
                                .radiance_probe_updates = lighting.radiance_probe_updates,
                                .resident_bytes = lighting.gpu_resident_bytes,
                                .budget_bytes = lighting.gpu_budget_bytes,
                                .fallback_reason =
                                    resolved_config_.indirect_lighting_path == lighting_trace_path::baked_probe
                                        ? "dynamic indirect lighting is unavailable; using lightmaps, probes, and IBL"
                                        : std::string{}};
    return result;
}

void renderer::report_texture_streaming_io(texture_streaming_io_snapshot snapshot) noexcept
{
    texture_streaming_io_ = snapshot;
}

void renderer::request_object_pick(std::uint64_t request_id, std::uint32_t x, std::uint32_t y)
{
    if (backend_) backend_->request_object_pick({.request_id = request_id, .x = x, .y = y});
}

render_object_pick_result renderer::last_object_pick() const
{
    if (!backend_) return {};
    return backend_->last_object_pick();
}

void renderer::request_frame_capture(const render_frame_capture_request& request)
{
    if (backend_) backend_->request_frame_capture(request);
}

render_frame_capture_result renderer::last_frame_capture() const
{
    if (!backend_) return {};
    return backend_->last_frame_capture();
}

render_submit_result renderer::render_frame(std::uint64_t frame_index, const render_graph& graph)
{
    resource_frame_index_ = frame_index;
    virtual_geometry_residency_.begin_frame(frame_index);
    texture_residency_.begin_frame(frame_index);
    if (!backend_)
        return render_submit_result::failure(
            {render_submit_error_code::backend_unavailable, "no render backend attached"});

    submit_texture_feedback(backend_->take_texture_feedback());
    for (const auto& result : backend_->take_texture_stream_upload_results())
        texture_residency_.complete(result);
    auto evictions = texture_residency_.take_evictions();
    if (!evictions.empty())
    {
        render_event_buffer buffer;
        render_event_writer writer(buffer);
        for (auto& eviction : evictions)
            writer.texture_stream_evict(eviction);
        frame_queue_.submit(std::move(buffer));
    }
    auto packet = frame_queue_.commit(frame_index);

    const bool temporal_graph = graph.find_resource("temporal_color_history") != nullptr;
    render_graph_compile_options graph_options{.frame_index = frame_index,
                                               .compute_queue_available = resolved_config_.features.async_compute,
                                               .transfer_queue_available = backend_->capabilities().transfer_queue};
    bool graph_view_initialized{};
    std::vector<std::shared_ptr<const gpu_scene_update_batch>> gpu_scene_updates;
    std::vector<std::shared_ptr<const lighting_scene_update_batch>> lighting_scene_updates;
    for (auto& event : packet.events)
    {
        auto* world = std::get_if<render_world_event>(&event.payload);
        if (!world || !world->packet) continue;
        auto prepared = std::make_shared<render_world_packet>(*world->packet);
        auto& previous = temporal_views_[prepared->render_view_id];
        const bool extent_changed =
            previous.width != prepared->camera.output_width || previous.height != prepared->camera.output_height;
        const bool epoch_changed = previous.world_epoch != prepared->world_epoch;
        const auto camera_delta = math::sub(prepared->camera.position, previous.position);
        const bool teleported = math::length_squared(camera_delta) > 100.0f;
        const bool rotated = previous.valid && math::dot(prepared->camera.forward, previous.forward) < 0.5f;
        prepared->camera.camera_cut = !previous.valid || extent_changed || epoch_changed || teleported || rotated;
        prepared->camera.history_valid = !prepared->camera.camera_cut;
        prepared->camera.previous_view_projection =
            prepared->camera.history_valid ? previous.view_projection : prepared->camera.view_projection;

        if (!graph_view_initialized)
        {
            graph_options.view_id = prepared->render_view_id;
            graph_options.output_extent = {prepared->camera.output_width, prepared->camera.output_height, 1u};
            graph_options.render_extent = {prepared->camera.render_width, prepared->camera.render_height, 1u};
            graph_options.world_epoch = prepared->world_epoch;
            if (prepared->camera.camera_cut)
                graph_options.temporal_reset = graph_options.temporal_reset | render_history_reset::camera_cut;
            if (extent_changed)
                graph_options.temporal_reset = graph_options.temporal_reset | render_history_reset::resize;
            if (epoch_changed)
                graph_options.temporal_reset = graph_options.temporal_reset | render_history_reset::world_epoch_change;
            graph_view_initialized = true;
        }

        if (temporal_graph && prepared->camera.render_width > 0 && prepared->camera.render_height > 0)
        {
            const auto sample_count = std::max<std::uint32_t>(config_.temporal.jitter_sample_count, 1u);
            const auto sample = frame_index % sample_count + 1u;
            prepared->camera.jitter = {(halton(sample, 2u) - 0.5f) / static_cast<float>(prepared->camera.render_width),
                                       (halton(sample, 3u) - 0.5f) /
                                           static_cast<float>(prepared->camera.render_height)};
            prepared->camera.projection(0, 2) -= prepared->camera.jitter[0] * 2.0f;
            prepared->camera.projection(1, 2) -= prepared->camera.jitter[1] * 2.0f;
            prepared->camera.view_projection = math::matmul(prepared->camera.projection, prepared->camera.view);
            if (!math::try_inverse(prepared->camera.view_projection, prepared->camera.inverse_view_projection))
                prepared->camera.camera_cut = true;
        }

        prepared->visible_terrain_patches.clear();
        prepared->terrain_statistics = {};
        render_camera terrain_camera = prepared->camera;
        terrain_camera.render_width = std::max(
            1u, static_cast<std::uint32_t>(std::round(terrain_camera.output_width * resolved_config_.render_scale)));
        terrain_camera.render_height = std::max(
            1u, static_cast<std::uint32_t>(std::round(terrain_camera.output_height * resolved_config_.render_scale)));
        if (!resolved_config_.features.gpu_terrain_traversal)
        {
            for (std::uint32_t terrain_index = 0; terrain_index < prepared->terrains.size(); ++terrain_index)
            {
                const auto& submission = prepared->terrains[terrain_index];
                if ((submission.render_layer_mask & 0xffffffffu) == 0u) continue;
                const auto* resource = terrain_data_for(submission.terrain);
                if (!resource) continue;
                auto selection = select_terrain_patches(
                    submission.terrain, resource->hierarchy, submission.model, terrain_camera,
                    resolved_config_.geometry_error_threshold, resource->lod.geometric_error_multiplier,
                    &terrain_selection_scratch_[renderer_resource_key(submission.terrain)]);
                prepared->terrain_statistics.hierarchy_nodes += selection.statistics.hierarchy_nodes;
                prepared->terrain_statistics.selected_patches += selection.statistics.selected_patches;
                prepared->terrain_statistics.culled_nodes += selection.statistics.culled_nodes;
                prepared->terrain_statistics.rendered_triangles += selection.statistics.rendered_triangles;
                for (std::size_t lod = 0; lod < prepared->terrain_statistics.patches_per_lod.size(); ++lod)
                    prepared->terrain_statistics.patches_per_lod[lod] += selection.statistics.patches_per_lod[lod];
                prepared->visible_terrain_patches.reserve(prepared->visible_terrain_patches.size() +
                                                          selection.patches.size());
                for (const auto& patch : selection.patches)
                    prepared->visible_terrain_patches.push_back({.terrain = patch.terrain,
                                                                 .terrain_index = terrain_index,
                                                                 .hierarchy_node = patch.node_index,
                                                                 .sample_min_x = patch.samples.min_x,
                                                                 .sample_min_z = patch.samples.min_z,
                                                                 .sample_max_x = patch.samples.max_x,
                                                                 .sample_max_z = patch.samples.max_z,
                                                                 .lod = patch.lod,
                                                                 .stitch_mask = patch.stitch_mask,
                                                                 .projected_error = patch.projected_error});
            }
        }

        previous = {.view_projection = prepared->camera.view_projection,
                    .position = prepared->camera.position,
                    .forward = prepared->camera.forward,
                    .world_epoch = prepared->world_epoch,
                    .width = prepared->camera.output_width,
                    .height = prepared->camera.output_height,
                    .valid = true};
        if (resolved_config_.features.gpu_driven_rendering)
            gpu_scene_updates.push_back(
                std::make_shared<gpu_scene_update_batch>(gpu_scene_.synchronize(*prepared, frame_index)));
        if (resolved_config_.features.surface_cache || resolved_config_.features.software_gi ||
            resolved_config_.features.software_reflections || resolved_config_.features.hardware_gi ||
            resolved_config_.features.hardware_reflections)
        {
            std::vector<lighting_scene_instance> instances;
            instances.reserve(prepared->items.size());
            std::uint32_t surface_card_count{};
            std::uint32_t surface_page_count{};
            std::uint32_t distance_field_page_count{};
            std::uint64_t resident_lighting_bytes{};
            std::unordered_set<std::uint64_t> counted_geometry;
            for (const auto& item : prepared->items)
            {
                if (!item.visible || item.transparent || item.skin_matrices.valid() || !item.affects_indirect_lighting)
                    continue;
                const auto geometry = lighting_geometry_for(item.mesh);
                const auto* lighting_geometry = lighting_geometry_data_for(geometry);
                if (!geometry.valid() || !lighting_geometry) continue;
                if (counted_geometry.insert(renderer_resource_key(geometry)).second)
                {
                    surface_card_count += static_cast<std::uint32_t>(lighting_geometry->cards.size());
                    const auto& field = lighting_geometry->distance_field;
                    distance_field_page_count += static_cast<std::uint32_t>(field.page_offsets.size());
                    surface_page_count +=
                        std::max(1u, static_cast<std::uint32_t>((lighting_geometry->cards.size() + 63u) / 64u));
                    resident_lighting_bytes += field.pages.size();
                }
                std::uint64_t transform_revision = 1469598103934665603ull;
                for (std::size_t row = 0; row < 4; ++row)
                    for (std::size_t column = 0; column < 4; ++column)
                    {
                        const auto bits = std::bit_cast<std::uint32_t>(item.model(row, column));
                        transform_revision ^= bits;
                        transform_revision *= 1099511628211ull;
                    }
                const auto object_key = (static_cast<std::uint64_t>(item.object_id.generation) << 32u) |
                                        static_cast<std::uint64_t>(item.object_id.index);
                instances.push_back(
                    {.stable_id = object_key ^ (renderer_resource_key(item.mesh) * 0x9e3779b97f4a7c15ull),
                     .geometry = geometry,
                     .material = item.material,
                     .model = item.model,
                     .world_bounds = item.world_bounds,
                     .transform_revision = transform_revision,
                     .material_revision = renderer_resource_key(item.material),
                     .geometry_generation = lighting_geometry->generation,
                     .card_density_bias = item.surface_card_density_bias,
                     .distance_field_resolution_bias = item.distance_field_resolution_bias,
                     .static_object = item.mobility == render_mobility::static_object,
                     .affects_indirect_lighting = item.affects_indirect_lighting,
                     .visible_in_hardware_tracing = item.visible_in_hardware_tracing});
            }
            lighting_scene_updates.push_back(std::make_shared<lighting_scene_update_batch>(lighting_scene_.synchronize(
                prepared->gpu_scene_world_id, prepared->world_epoch, frame_index, instances)));
            lighting_scene_.update_residency_statistics(surface_card_count, surface_page_count,
                                                        distance_field_page_count, resident_lighting_bytes);
        }
        world->packet = std::move(prepared);
    }

    if (!gpu_scene_updates.empty())
    {
        packet.events.reserve(packet.events.size() + gpu_scene_updates.size());
        for (auto& update : gpu_scene_updates)
            packet.events.push_back({.payload = gpu_scene_update_event{.batch = std::move(update)}});
    }
    if (!lighting_scene_updates.empty())
    {
        packet.events.reserve(packet.events.size() + lighting_scene_updates.size());
        for (auto& update : lighting_scene_updates)
            packet.events.push_back({.payload = lighting_scene_update_event{.batch = std::move(update)}});
    }

    if (config_.enable_dynamic_resolution)
    {
        float gpu_frame_time_ms{};
        for (const auto& timing : backend_->last_frame_profile().pass_timings)
            gpu_frame_time_ms += static_cast<float>(timing.milliseconds);
        if (gpu_frame_time_ms > 0.0f)
        {
            const auto previous = frame_budget_.settings();
            const auto& budget = frame_budget_.update(gpu_frame_time_ms);
            resolved_config_.render_scale = budget.render_scale;
            const auto effective_aa = config_.force_disable_temporal
                                          ? resolve_anti_aliasing(anti_aliasing_method::fxaa, resolved_config_.path,
                                                                  budget.render_scale, backend_->capabilities())
                                          : resolve_anti_aliasing(config_.anti_aliasing, resolved_config_.path,
                                                                  budget.render_scale, backend_->capabilities());
            resolved_config_.anti_aliasing = effective_aa;
            resolved_config_.features.anti_aliasing = effective_aa;
            resolved_config_.features.fxaa = effective_aa == anti_aliasing_method::fxaa;
            resolved_config_.features.temporal_antialiasing =
                effective_aa == anti_aliasing_method::taa || effective_aa == anti_aliasing_method::taau;
            resolved_config_.features.temporal_upscaling = effective_aa == anti_aliasing_method::taau;
            resolved_config_.geometry_error_threshold = budget.geometry_error_threshold;
            resolved_config_.shadow_resolution_scale = budget.shadow_resolution_scale;
            resolved_config_.volumetric_resolution_scale = budget.volumetric_resolution_scale;
            resolved_config_.gi_trace_budget = budget.gi_trace_budget;
            resolved_config_.reflection_ray_budget = budget.reflection_ray_budget;
            resolved_config_.lighting_trace_scale = budget.lighting_trace_scale;
            resolved_config_.surface_cache_update_budget = budget.surface_cache_update_budget;
            resolved_config_.radiance_probe_update_budget = budget.radiance_probe_update_budget;
            const auto& profile = quality_profile(resolved_config_.quality);
            const auto scaled_shadow_dimension = [&](std::uint32_t base, std::uint32_t minimum)
            {
                const auto scaled =
                    static_cast<std::uint32_t>(std::round(static_cast<float>(base) * budget.shadow_resolution_scale));
                return std::max(minimum, (scaled / 128u) * 128u);
            };
            resolved_config_.directional_shadow_resolution =
                scaled_shadow_dimension(profile.directional_shadow_resolution, 512u);
            resolved_config_.local_shadow_atlas_resolution =
                scaled_shadow_dimension(profile.local_shadow_atlas_resolution, 1024u);
            resolved_config_.max_local_shadow_resolution =
                scaled_shadow_dimension(profile.max_local_shadow_resolution, 256u);
            if (frame_budget_.last_change() != frame_budget_change::none ||
                budget.render_scale != previous.render_scale)
                backend_->configure(resolved_config_);
        }
    }

    for (const auto& event : packet.events)
    {
        if (const auto* resize = std::get_if<viewport_resize_event>(&event.payload))
            resize_viewport(resize->width, resize->height);
    }

    auto compilation = graph.compile(graph_options);
    if (!compilation)
    {
        const auto& error = compilation.error();
        std::string message = error.message;
        if (!error.pass.empty()) message += " [pass: " + error.pass + ']';
        if (!error.resource.empty()) message += " [resource: " + error.resource + ']';
        return render_submit_result::failure({render_submit_error_code::invalid_render_graph, std::move(message)});
    }
    auto compiled = std::move(compilation).value();

    return backend_->submit(packet, compiled);
}

renderer_module::renderer_module(renderer_config config)
    : renderer_(config), graph_(make_clear_present_graph("viewport"))
{
}

renderer& renderer_module::service() noexcept
{
    return renderer_;
}

std::string_view renderer_module::name() const
{
    return "renderer";
}

void renderer_module::on_start(framework::module_context&)
{
    arc::diagnostics::info("render", "Renderer module started");
}

void renderer_module::on_update(framework::module_context&, const framework::frame_time& time)
{
    if (!renderer_.backend())
    {
        if (!missing_backend_reported_)
        {
            arc::diagnostics::debug("render", "no render backend attached");
            missing_backend_reported_ = true;
        }
        return;
    }

    const auto result = renderer_.render_frame(time.frame_index, graph_);
    if (!result && !result.error().message.empty()) arc::diagnostics::debug("render", result.error().message);
}

void renderer_module::on_shutdown(framework::module_context&)
{
    arc::diagnostics::info("render", "Renderer module shutdown");
}

} // namespace arc::render
