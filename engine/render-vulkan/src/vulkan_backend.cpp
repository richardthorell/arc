#include <arc/render/vulkan/vulkan_backend.h>

#include <arc/diagnostics/log.h>
#include <arc/render/lighting.h>
#include <arc/render/render_world.h>
#include <arc/render/resources.h>

#include "builtin_shaders.h"
#include "vulkan_pick_utils.h"
#include "vulkan_sky_constants.h"
#include "vulkan_swapchain.h"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(_WIN32) && defined(ARC_EDITOR) && ARC_EDITOR
#define ARC_VULKAN_SHARED_VIEWPORT 1
#else
#define ARC_VULKAN_SHARED_VIEWPORT 0
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#endif

namespace arc::render::vulkan
{
namespace
{

constexpr std::uint32_t material_shadow_binding = 5u;
constexpr std::uint32_t material_shadow_data_binding = 6u;
constexpr std::uint32_t terrain_normal_binding = 7u;
constexpr std::uint32_t terrain_surface_binding = 11u;
constexpr std::uint32_t material_light_data_binding = 15u;
constexpr std::uint32_t material_parameters_binding = 16u;
constexpr std::uint32_t material_local_shadow_binding = 17u;
constexpr std::uint32_t material_binding_count = 18u;
constexpr std::uint32_t material_descriptor_set_capacity = 12288u;
constexpr std::uint32_t directional_shadow_layer_count = directional_shadow_cascade_count * 2u;
constexpr VkDeviceSize upload_staging_capacity = 64u * 1024u * 1024u;
constexpr std::array<std::uint32_t, 15> material_image_bindings{0u,
                                                                1u,
                                                                2u,
                                                                3u,
                                                                4u,
                                                                material_shadow_binding,
                                                                terrain_normal_binding + 0u,
                                                                terrain_normal_binding + 1u,
                                                                terrain_normal_binding + 2u,
                                                                terrain_normal_binding + 3u,
                                                                terrain_surface_binding + 0u,
                                                                terrain_surface_binding + 1u,
                                                                terrain_surface_binding + 2u,
                                                                terrain_surface_binding + 3u,
                                                                material_local_shadow_binding};

const char* vk_result_name(VkResult result) noexcept
{
    switch (result)
    {
        case VK_SUCCESS:
            return "VK_SUCCESS";
        case VK_NOT_READY:
            return "VK_NOT_READY";
        case VK_TIMEOUT:
            return "VK_TIMEOUT";
        case VK_EVENT_SET:
            return "VK_EVENT_SET";
        case VK_EVENT_RESET:
            return "VK_EVENT_RESET";
        case VK_INCOMPLETE:
            return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:
            return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:
            return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:
            return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
            return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:
            return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_SURFACE_LOST_KHR:
            return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
            return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR:
            return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR:
            return "VK_ERROR_OUT_OF_DATE_KHR";
        default:
            return "VK_ERROR_UNKNOWN";
    }
}

std::string describe_vk_result(VkResult result)
{
    return std::string{vk_result_name(result)} + " (" + std::to_string(static_cast<std::int32_t>(result)) + ")";
}

void cmd_begin_rendering(VkCommandBuffer command_buffer, const VkRenderingInfo* rendering)
{
    if (vkCmdBeginRendering != nullptr)
        vkCmdBeginRendering(command_buffer, rendering);
    else
        vkCmdBeginRenderingKHR(command_buffer, rendering);
}

void cmd_end_rendering(VkCommandBuffer command_buffer)
{
    if (vkCmdEndRendering != nullptr)
        vkCmdEndRendering(command_buffer);
    else
        vkCmdEndRenderingKHR(command_buffer);
}

std::uint64_t resource_key(resource_handle handle) noexcept
{
    return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
}

VkDeviceSize buffer_size(std::size_t count, std::size_t stride) noexcept
{
    return static_cast<VkDeviceSize>(count * stride);
}

math::vector3f matrix_translation(const math::matrix4f& matrix) noexcept
{
    return {matrix(0, 3), matrix(1, 3), matrix(2, 3)};
}

math::matrix4f look_at_rh(const math::vector3f& eye, const math::vector3f& target, const math::vector3f& up) noexcept
{
    const auto z = math::normalize(math::sub(eye, target), 0.0f);
    auto x = math::normalize(math::cross(up, z), 0.0f);
    if (math::length_squared(x) < 0.0001f) x = math::vector3f{1.0f, 0.0f, 0.0f};
    const auto y = math::cross(z, x);

    math::matrix4f result = math::identity<float, 4>();
    result(0, 0) = x[0];
    result(0, 1) = x[1];
    result(0, 2) = x[2];
    result(0, 3) = -math::dot(x, eye);
    result(1, 0) = y[0];
    result(1, 1) = y[1];
    result(1, 2) = y[2];
    result(1, 3) = -math::dot(y, eye);
    result(2, 0) = z[0];
    result(2, 1) = z[1];
    result(2, 2) = z[2];
    result(2, 3) = -math::dot(z, eye);
    return result;
}

math::matrix4f perspective_rh_zo(float vertical_fov, float near_plane, float far_plane) noexcept
{
    near_plane = std::max(near_plane, 0.001f);
    far_plane = std::max(far_plane, near_plane + 0.001f);
    const float tangent = std::tan(std::clamp(vertical_fov, 0.01f, math::pi<float> - 0.01f) * 0.5f);
    const float focal = 1.0f / std::max(tangent, 0.001f);
    math::matrix4f result{};
    result(0, 0) = focal;
    result(1, 1) = focal;
    result(2, 2) = far_plane / (near_plane - far_plane);
    result(2, 3) = (far_plane * near_plane) / (near_plane - far_plane);
    result(3, 2) = -1.0f;
    return result;
}

struct mesh_push_constants
{
    float model_view_projection[16]{};
    float model[16]{};
    float base_color[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float light_direction_intensity[4]{0.35f, -0.85f, -0.40f, 1.0f};
    float light_color[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float camera_position[4]{};
    float visualization[4]{};
    float fog_color_density[4]{};
    float fog_params[4]{};
    float material_params[4]{1.0f, 1.0f, 1.0f, 0.0f};
};
static_assert(sizeof(mesh_push_constants) == 256);

struct alignas(16) gpu_scene_transform_record
{
    float model[16]{};
    float previous_model[16]{};
};
static_assert(sizeof(gpu_scene_transform_record) == 128);

struct alignas(16) gpu_scene_visibility_record
{
    float bounds_min[4]{};
    float bounds_max[4]{};
    std::uint32_t geometry[4]{};
    std::uint32_t material_flags[4]{};
    std::uint32_t draw_metadata[4]{};
    float distance_error[4]{};
};
static_assert(sizeof(gpu_scene_visibility_record) == 96);

struct packed_gpu_scene_instance
{
    gpu_scene_transform_record transform;
    gpu_scene_visibility_record visibility;
};

struct alignas(16) gpu_visibility_push_constants
{
    float view_projection[16]{};
    float camera_position_and_error[4]{};
    std::uint32_t instance_capacity{};
    std::uint32_t render_layer_mask{~0u};
    std::uint32_t camera_cut{};
    std::uint32_t reserved{};
    float hzb_parameters[4]{};
};
static_assert(sizeof(gpu_visibility_push_constants) == 112);

inline constexpr VkDeviceSize indexed_indirect_command_stride = sizeof(VkDrawIndexedIndirectCommand);

struct material_uniform_data
{
    float emissive_factor[4]{0.0f, 0.0f, 0.0f, 1.0f};
    float material_lobes[4]{};
    float volume_params[4]{};
    float subsurface_color_factor[4]{1.0f, 0.35f, 0.2f, 0.0f};
    float attenuation_color[4]{1.0f, 1.0f, 1.0f, 0.0f};
};
static_assert(sizeof(material_uniform_data) == 80);

struct deferred_push_constants
{
    float inverse_view_projection[16]{};
    float camera_position[4]{};
    float light_direction_intensity[4]{0.35f, -0.85f, -0.40f, 1.0f};
    float light_color[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float ambient_visualization[4]{0.18f, 0.18f, 0.18f, 0.0f};
};
static_assert(sizeof(deferred_push_constants) == 128);

struct output_transform_push_constants
{
    float exposure_output[4]{1.0f, 0.0f, 0.0f, 0.0f};
    float post_process[4]{};
};

struct histogram_push_constants
{
    float log_luminance_extent[4]{-12.0f, 16.0f, 1.0f, 1.0f};
};

struct exposure_resolve_push_constants
{
    float log_range_percentiles[4]{-12.0f, 16.0f, 0.005f, 0.98f};
    float limits_speeds[4]{-8.0f, 20.0f, 3.0f, 1.0f};
    float timing_mode[4]{1.0f / 60.0f, 1.0f, 10.0f, 0.0f};
};

inline constexpr VkDeviceSize exposure_histogram_bytes = sizeof(std::uint32_t) * 256u;
inline constexpr VkDeviceSize exposure_buffer_bytes = exposure_histogram_bytes + sizeof(std::uint32_t) * 4u;

struct shadow_uniform_data
{
    float light_view_projection[directional_shadow_cascade_count][16]{};
    float cascade_splits[4]{};
    float params[4]{};
    float cascade_texel_size[4]{};
    float cascade_blend_starts[4]{};
    float configuration[4]{};
};

struct gpu_scope_record
{
    std::string name;
    std::uint32_t begin_query{};
    std::uint32_t end_query{};
};

struct graph_image
{
    VkImage image{};
    VmaAllocation allocation{};
    VkImageView view{};
    std::vector<VkImageView> mip_views;
    VkFormat format{};
    VkImageAspectFlags aspect{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t mip_levels{1};
};

struct hzb_reduce_push_constants
{
    std::int32_t destination_width{};
    std::int32_t destination_height{};
    std::int32_t source_width{};
    std::int32_t source_height{};
    std::int32_t source_mip{-1};
};

struct temporal_mask_push_constants
{
    std::int32_t width{};
    std::int32_t height{};
    std::uint32_t history_valid{};
    float disocclusion_threshold{0.01f};
    float reactive_response{1.0f};
};

struct velocity_dilation_push_constants
{
    std::int32_t width{};
    std::int32_t height{};
};

struct temporal_resolve_push_constants
{
    std::int32_t output_width{};
    std::int32_t output_height{};
    float input_width{};
    float input_height{};
    std::uint32_t history_valid{};
    float history_weight{0.9f};
};

struct sharpen_push_constants
{
    std::int32_t output_width{};
    std::int32_t output_height{};
    float strength{0.2f};
    float clamp_strength{0.25f};
};

class vulkan_render_backend final : public render_backend
{
#if ARC_VULKAN_SHARED_VIEWPORT
    struct shared_viewport_slot
    {
        VkImage image{};
        VkDeviceMemory memory{};
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        HANDLE shared_handle{};
        VkCommandPool command_pool{};
        VkCommandBuffer command_buffer{};
        VkFence fence{};
        shared_viewport_frame_state state{shared_viewport_frame_state::available};
        std::uint64_t frame_id{};
        bool initialized{};
    };

    struct shared_viewport_output
    {
        std::string id;
        std::uint64_t generation{};
        std::uint64_t next_frame_id{1};
        std::uint64_t dropped_frames{};
        std::uint32_t width{1};
        std::uint32_t height{1};
        std::uint32_t pending_width{};
        std::uint32_t pending_height{};
        bool visible{true};
        bool destroy_pending{};
        std::array<shared_viewport_slot, 3> slots;
    };
#endif

public:
    vulkan_render_backend(VkInstance instance, VkSurfaceKHR surface, VkPhysicalDevice physical_device, VkDevice device,
                          VkQueue queue, VmaAllocator allocator, std::uint32_t graphics_queue_family,
                          render_capabilities capabilities, viewport_output_type viewport_output)
        : instance_(instance), surface_(surface), physical_device_(physical_device), device_(device), queue_(queue),
          allocator_(allocator), graphics_queue_family_(graphics_queue_family), capabilities_(capabilities),
          configured_viewport_output_(viewport_output)
    {
        if (configured_viewport_output_ == viewport_output_type::shared_texture)
        {
#if ARC_VULKAN_SHARED_VIEWPORT
            viewport_format_ = VK_FORMAT_B8G8R8A8_UNORM;
#else
            arc::diagnostics::warn("render.vulkan",
                                   "shared viewport output is only available in Windows editor builds");
#endif
        }
        create_support_objects();
#if ARC_VULKAN_SHARED_VIEWPORT
        query_shared_viewport_support();
#endif
    }

    ~vulkan_render_backend() override
    {
        shutdown_surface_presenter();
        if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
        destroy_temporal_resources();
        destroy_hzb_resources();
        destroy_mesh_pipeline();
        destroy_shadow_resources();
        destroy_local_shadow_resources();
        destroy_white_texture();
        destroy_buffer(pick_readback_buffer_);
        destroy_buffer(capture_readback_buffer_);
        deferred_releases_.collect(std::numeric_limits<std::uint64_t>::max());
        for (auto& buffer : shadow_uniform_buffers_)
            destroy_buffer(buffer);
        for (auto& buffer : debug_overlay_buffers_)
            destroy_buffer(buffer.vertices);
        destroy_buffer(light_buffer_);
        destroy_buffer(exposure_buffer_);
        destroy_buffer(gpu_scene_visibility_buffer_);
        destroy_buffer(gpu_scene_transform_buffer_);
        destroy_gpu_visibility_resources();
#if ARC_VULKAN_SHARED_VIEWPORT
        destroy_all_shared_viewports();
#endif
        destroy_meshes();
        destroy_support_objects();
        if (allocator_ != VK_NULL_HANDLE) vmaDestroyAllocator(allocator_);
        if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
        if (surface_ != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    }

    render_backend_type type() const noexcept override
    {
        return render_backend_type::vulkan;
    }

    const render_capabilities& capabilities() const noexcept override
    {
        return capabilities_;
    }

    void configure(const resolved_render_config& config) override
    {
        const float previous_scale = resolved_config_.render_scale;
        const std::uint32_t previous_local_shadow_atlas = resolved_config_.local_shadow_atlas_resolution;
        resolved_config_ = config;
        if (!local_shadow_allocator_ || previous_local_shadow_atlas != config.local_shadow_atlas_resolution)
        {
            local_shadow_allocator_ =
                std::make_unique<shadow_atlas_allocator>(config.local_shadow_atlas_resolution, 128u, 2u);
            local_shadow_static_signatures_.clear();
        }
        if (config.features.timeline_semaphores && upload_timeline_ == VK_NULL_HANDLE)
        {
            VkSemaphoreTypeCreateInfo timeline_type{};
            timeline_type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
            timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            timeline_type.initialValue = 0;
            VkSemaphoreCreateInfo semaphore{};
            semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            semaphore.pNext = &timeline_type;
            if (vkCreateSemaphore(device_, &semaphore, nullptr, &upload_timeline_) != VK_SUCCESS)
            {
                upload_timeline_ = VK_NULL_HANDLE;
                arc::diagnostics::warn("render.vulkan",
                                       "timeline upload completion is unavailable; using the fence fallback");
            }
        }
        upload_timeline_enabled_ = config.features.timeline_semaphores && upload_timeline_ != VK_NULL_HANDLE;
        last_profile_.configuration = config;
        if (native_swapchain_initialized_ && previous_scale != config.render_scale && output_viewport_width_ > 0 &&
            output_viewport_height_ > 0)
        {
            ensure_viewport(scaled_dimension(output_viewport_width_), scaled_dimension(output_viewport_height_));
        }
    }

    render_submit_result submit(const render_frame_packet& packet, const compiled_render_graph& graph) override
    {
        last_profile_.frame_index = packet.frame_index;
        last_profile_.gpu_scene = {};
        last_profile_.temporal = {};
        temporal_output_view_ = VK_NULL_HANDLE;
        last_profile_.terrain = {};
        upload_frame_ = packet.frame_index;
        upload_batch_failed_ = false;
        frame_draws_.clear();
        frame_virtual_draws_.clear();
        frame_terrain_draws_.clear();
        frame_shadow_draws_.clear();
        frame_virtual_shadow_draws_.clear();
        frame_directional_lights_.clear();
        frame_point_lights_.clear();
        frame_spot_lights_.clear();
        frame_area_lights_.clear();
        frame_debug_overlay_lines_.clear();
        frame_debug_overlay_triangles_.clear();
        frame_environment_ = {};
        pending_debug_markers_.clear();
        for (const auto& event : packet.events)
        {
            if (const auto* upload = std::get_if<mesh_upload_event>(&event.payload))
            {
                upload_mesh(*upload);
                ++shadow_resource_revision_;
            }
            else if (const auto* destroy = std::get_if<mesh_destroy_event>(&event.payload))
            {
                retire_mesh(destroy->handle);
                ++shadow_resource_revision_;
            }
            else if (const auto* virtual_upload = std::get_if<virtual_mesh_upload_event>(&event.payload))
            {
                upload_virtual_mesh(*virtual_upload);
                ++shadow_resource_revision_;
            }
            else if (const auto* virtual_destroy = std::get_if<virtual_mesh_destroy_event>(&event.payload))
            {
                retire_virtual_mesh(virtual_destroy->handle);
                ++shadow_resource_revision_;
            }
            else if (const auto* terrain = std::get_if<terrain_upload_event>(&event.payload))
            {
                upload_terrain(*terrain);
                ++shadow_resource_revision_;
            }
            else if (const auto* height_update = std::get_if<terrain_height_update_event>(&event.payload))
            {
                update_terrain_heights(*height_update);
                ++shadow_resource_revision_;
            }
            else if (const auto* weight_update = std::get_if<terrain_weight_update_event>(&event.payload))
                update_terrain_weights(*weight_update);
            else if (const auto* terrain_destroy = std::get_if<terrain_destroy_event>(&event.payload))
            {
                retire_terrain(terrain_destroy->handle);
                ++shadow_resource_revision_;
            }
            else if (const auto* texture = std::get_if<texture_upload_event>(&event.payload))
                upload_texture(*texture);
            else if (const auto* material = std::get_if<material_upload_event>(&event.payload))
            {
                upload_material(*material);
                ++shadow_resource_revision_;
            }
            else if (const auto* environment = std::get_if<environment_upload_event>(&event.payload))
                upload_environment(*environment);
            else if (const auto* destroyed_environment = std::get_if<environment_destroy_event>(&event.payload))
            {
                environments_.erase(resource_key(destroyed_environment->handle));
                if (active_environment_ == destroyed_environment->handle) active_environment_ = {};
            }
            else if (const auto* draw = std::get_if<draw_mesh_event>(&event.payload))
            {
                frame_draws_.push_back(*draw);
                frame_shadow_draws_.push_back(*draw);
            }
            else if (const auto* light = std::get_if<directional_light_event>(&event.payload))
                frame_directional_lights_.push_back(*light);
            else if (const auto* point_light = std::get_if<point_light_event>(&event.payload))
                frame_point_lights_.push_back(*point_light);
            else if (const auto* spot_light = std::get_if<spot_light_event>(&event.payload))
                frame_spot_lights_.push_back(*spot_light);
            else if (const auto* area_light = std::get_if<area_light_event>(&event.payload))
                frame_area_lights_.push_back(*area_light);
            else if (const auto* gpu_scene_update = std::get_if<gpu_scene_update_event>(&event.payload))
                apply_gpu_scene_update(*gpu_scene_update);
            else if (const auto* world = std::get_if<render_world_event>(&event.payload))
                append_render_world(*world);
            else if (const auto* marker = std::get_if<debug_marker_event>(&event.payload))
                pending_debug_markers_.push_back(marker->label);
        }
        if (!flush_upload_batch()) upload_batch_failed_ = true;

        last_profile_.graph = graph;
        last_profile_.summary.clear();
        last_profile_.summary.reserve(64);
        last_profile_.summary += std::to_string(graph.passes.size());
        last_profile_.summary += " graph pass(es), ";
        last_profile_.summary += std::to_string(packet.events.size());
        last_profile_.summary += " render event(s)";

        const environment_descriptor* lighting_environment = active_environment();
        if (frame_environment_.lighting.environment.valid())
        {
            const auto found = environments_.find(resource_key(frame_environment_.lighting.environment));
            if (found != environments_.end()) lighting_environment = &found->second.data;
        }
        auto point_lights_for_tier = frame_point_lights_;
        const bool low_area_fallback = resolved_config_.quality == render_quality_tier::low;
        if (low_area_fallback)
        {
            for (const auto& area : frame_area_lights_)
            {
                const float width = std::max(area.width, 0.001f);
                const float height = area.shape == area_light_shape::disk ? width : std::max(area.height, 0.001f);
                const float surface_area =
                    area.shape == area_light_shape::disk ? math::pi<float> * 0.25f * width * width : width * height;
                const float lumens =
                    area.intensity_unit == light_intensity_unit::nit
                        ? area.intensity * math::pi<float> * surface_area * (area.two_sided ? 2.0f : 1.0f)
                        : area.intensity;
                point_lights_for_tier.push_back(
                    {.position = area.position,
                     .color = area.color,
                     .intensity = lumens,
                     .range = std::clamp(std::sqrt(std::max(lumens, 0.0f)) * 0.5f, 5.0f, 50.0f),
                     .enabled = area.enabled,
                     .use_color_temperature = area.use_color_temperature,
                     .temperature_kelvin = area.temperature_kelvin,
                     .intensity_unit = area.intensity_unit == light_intensity_unit::unitless
                                           ? light_intensity_unit::unitless
                                           : light_intensity_unit::lumen,
                     .label = area.label + " (low-tier point fallback)"});
            }
        }
        frame_lighting_ = pack_scene_lighting(
            frame_directional_lights_, point_lights_for_tier, frame_spot_lights_,
            frame_environment_.affect_lighting && frame_environment_.lighting.enabled ? lighting_environment : nullptr,
            resolved_config_.max_point_lights, resolved_config_.max_spot_lights,
            low_area_fallback ? empty_area_lights_ : frame_area_lights_);
        if (frame_environment_.affect_lighting && frame_environment_.lighting.enabled)
        {
            math::vector3f ambient = frame_environment_.lighting.constant_color;
            if (frame_environment_.lighting.source == environment_lighting_source_mode::follow_sky)
            {
                ambient = frame_environment_.source == sky_source_mode::solid_color
                              ? frame_environment_.solid_color
                              : math::vector3f{frame_environment_.atmosphere.tint[0] * 0.28f,
                                               frame_environment_.atmosphere.tint[1] * 0.28f,
                                               frame_environment_.atmosphere.tint[2] * 0.28f};
            }
            if (frame_environment_.lighting.source != environment_lighting_source_mode::hdri || !lighting_environment)
            {
                frame_lighting_.ambient_color_intensity = {ambient[0], ambient[1], ambient[2],
                                                           frame_environment_.lighting.diffuse_intensity};
            }
        }
        update_environment_profile(lighting_environment);
        last_profile_.clustered_lights = make_clustered_light_profile();
        update_shadow_profile(packet.frame_index);
        last_profile_.temporal = {.enabled = resolved_config_.features.temporal_antialiasing,
                                  .upscaling = resolved_config_.features.temporal_upscaling,
                                  .history_valid = frame_camera_.history_valid,
                                  .camera_cut = frame_camera_.camera_cut,
                                  .jitter = frame_camera_.jitter,
                                  .reset_reason =
                                      frame_camera_.camera_cut ? "camera cut, resize, or world change" : ""};
        update_light_buffer();
        warn_about_skipped_lights(frame_lighting_);

        std::ostringstream message;
        message << "vulkan accepted frame " << packet.frame_index << " with " << packet.events.size()
                << " event(s) and " << graph.passes.size() << " pass(es)";
        if (upload_batch_failed_)
        {
            message << "; one or more resource upload batches failed";
            return render_submit_result::failure({render_submit_error_code::backend_failure, message.str()});
        }
        return render_submit_result::success();
    }

    void resize_viewport(std::uint32_t width, std::uint32_t height) override
    {
        output_viewport_width_ = width;
        output_viewport_height_ = height;
        if (native_swapchain_initialized_ && width > 0 && height > 0)
            ensure_viewport(scaled_dimension(width), scaled_dimension(height));
    }

    render_viewport_texture viewport_texture() const noexcept override
    {
        // Native editor presentation owns the surface directly. The old opaque
        // legacy texture handle is intentionally no longer exposed.
        return {};
    }

    render_backend_frame_profile last_frame_profile() const override
    {
        return last_profile_;
    }

    void request_object_pick(render_object_pick_request request) override
    {
        pending_pick_request_ = request;
    }

    render_object_pick_result last_object_pick() const override
    {
        return last_pick_result_;
    }

    void request_frame_capture(render_frame_capture_request request) override
    {
        if (request.capture_id == 0) return;
        pending_capture_request_ = std::move(request);
    }

    render_frame_capture_result last_frame_capture() const override
    {
        return last_capture_result_;
    }

    surface_frame_result present_surface_frame(std::uint32_t width, std::uint32_t height) override
    {
        std::string message;
        if (render_native_viewport_frame(width, height, message)) return surface_frame_result::success();

        surface_frame_error_code code = surface_frame_error_code::backend_failure;
        if (device_lost_)
            code = surface_frame_error_code::device_lost;
        else if (surface_ == VK_NULL_HANDLE)
            code = surface_frame_error_code::unavailable;
        else if (message.find("out of date") != std::string::npos || message.find("suboptimal") != std::string::npos)
            code = surface_frame_error_code::out_of_date;
        return surface_frame_result::failure({code, std::move(message)});
    }

    bool render_native_viewport_frame(std::uint32_t width, std::uint32_t height, std::string& message)
    {
        message.clear();
        if (device_lost_)
        {
            message = "native viewport device is lost; backend recreation required";
            return false;
        }
        if (surface_ == VK_NULL_HANDLE)
        {
            message = "Vulkan backend was created without a presentation surface";
            return false;
        }
        if (width == 0 || height == 0) return true;

        output_viewport_width_ = width;
        output_viewport_height_ = height;

        if (!swapchain_.valid() || swapchain_rebuild_ || swapchain_.extent.width != width ||
            swapchain_.extent.height != height)
        {
            VkBool32 present_supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, graphics_queue_family_, surface_,
                                                 &present_supported);
            if (present_supported != VK_TRUE)
            {
                message = "Vulkan queue does not support the native viewport surface";
                return false;
            }

            const std::array<VkFormat, 4> formats{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
                                                  VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
            const VkFormat previous_format = viewport_format_;
            if (!swapchain_.create_or_resize(physical_device_, device_, surface_, graphics_queue_family_, width, height,
                                             min_image_count_, VK_IMAGE_USAGE_TRANSFER_DST_BIT, formats,
                                             VK_PRESENT_MODE_FIFO_KHR, message))
                return false;

            viewport_format_ = swapchain_.surface_format.format;
            native_swapchain_initialized_ = true;
            swapchain_rebuild_ = false;
            if (previous_format != viewport_format_ && viewport_image_ != VK_NULL_HANDLE)
            {
                destroy_mesh_pipeline();
                destroy_viewport();
            }
        }

        ensure_viewport(scaled_dimension(width), scaled_dimension(height));
        if (!swapchain_.valid() || swapchain_.semaphore_index >= swapchain_.semaphores.size())
        {
            message = "native viewport swapchain has no usable frame resources";
            return false;
        }

        const auto& sync = swapchain_.semaphores[swapchain_.semaphore_index];
        const VkSemaphore image_acquired_semaphore = sync.image_acquired;
        const VkSemaphore render_complete_semaphore = sync.render_complete;
        VkResult result = vkAcquireNextImageKHR(device_, swapchain_.handle, UINT64_MAX, image_acquired_semaphore,
                                                VK_NULL_HANDLE, &swapchain_.frame_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            swapchain_rebuild_ = true;
            return true;
        }
        if (result == VK_ERROR_SURFACE_LOST_KHR)
        {
            message = "native viewport surface lost (" + describe_vk_result(result) + "); backend recreation required";
            return false;
        }
        if (result == VK_ERROR_DEVICE_LOST)
        {
            device_lost_ = true;
            message = "native viewport device lost while acquiring the swapchain image (" + describe_vk_result(result) +
                      "); backend recreation required";
            return false;
        }
        if (result != VK_SUCCESS)
        {
            swapchain_rebuild_ = true;
            message = "failed to acquire native viewport swapchain image: " + describe_vk_result(result);
            return false;
        }
        if (swapchain_.frame_index >= swapchain_.frames.size())
        {
            message = "Vulkan returned a swapchain image index outside ARC's frame resources";
            return false;
        }
        active_frame_index_ = swapchain_.frame_index;

        auto* frame = &swapchain_.frames[swapchain_.frame_index];
        vkWaitForFences(device_, 1, &frame->fence, VK_TRUE, UINT64_MAX);
        collect_timestamp_results();
        collect_object_pick_result();
        collect_frame_capture_result();
        retire_completed_resources();

        // Frame-dependent resources may wait on every swapchain fence. Keep
        // the acquired fence signaled until preparation has completed.
        prepare_frame_gpu_resources();

        vkResetFences(device_, 1, &frame->fence);
        vkResetCommandPool(device_, frame->command_pool, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(frame->command_buffer, &begin_info);

        begin_debug_label(frame->command_buffer, "ARC native viewport frame", {0.16f, 0.45f, 1.0f, 1.0f});
        reset_timestamp_queries(frame->command_buffer);

        execute_compiled_graph(frame->command_buffer);

        transition_viewport(frame->command_buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkImageMemoryBarrier swapchain_to_transfer{};
        swapchain_to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        swapchain_to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        swapchain_to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapchain_to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapchain_to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapchain_to_transfer.image = frame->backbuffer;
        swapchain_to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        swapchain_to_transfer.subresourceRange.levelCount = 1;
        swapchain_to_transfer.subresourceRange.layerCount = 1;
        swapchain_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(frame->command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &swapchain_to_transfer);

        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = {static_cast<std::int32_t>(viewport_width_), static_cast<std::int32_t>(viewport_height_),
                              1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1] = {static_cast<std::int32_t>(swapchain_.extent.width),
                              static_cast<std::int32_t>(swapchain_.extent.height), 1};
        vkCmdBlitImage(frame->command_buffer, viewport_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, frame->backbuffer,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        VkImageMemoryBarrier swapchain_to_present = swapchain_to_transfer;
        swapchain_to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapchain_to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        swapchain_to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapchain_to_present.dstAccessMask = 0;
        vkCmdPipelineBarrier(frame->command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapchain_to_present);

        end_debug_label(frame->command_buffer);
        vkEndCommandBuffer(frame->command_buffer);

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &image_acquired_semaphore;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame->command_buffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &render_complete_semaphore;
        result = vkQueueSubmit(queue_, 1, &submit, frame->fence);
        if (result != VK_SUCCESS)
        {
            device_lost_ = result == VK_ERROR_DEVICE_LOST;
            message = "failed to submit native viewport frame: " + describe_vk_result(result);
            return false;
        }

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &render_complete_semaphore;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_.handle;
        present.pImageIndices = &swapchain_.frame_index;
        result = vkQueuePresentKHR(queue_, &present);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            swapchain_rebuild_ = true;
            return true;
        }
        if (result != VK_SUCCESS)
        {
            if (result == VK_ERROR_SURFACE_LOST_KHR)
                message = "native viewport surface lost while presenting (" + describe_vk_result(result) +
                          "); backend recreation required";
            else
            {
                device_lost_ = result == VK_ERROR_DEVICE_LOST;
                message = "failed to present native viewport frame: " + describe_vk_result(result);
            }
            return false;
        }

        swapchain_.semaphore_index =
            (swapchain_.semaphore_index + 1u) % static_cast<std::uint32_t>(swapchain_.semaphores.size());
        last_completed_frame_ = last_profile_.frame_index;
        return true;
    }

    void shutdown_surface_presenter() noexcept
    {
        if (!native_swapchain_initialized_ && !swapchain_.valid()) return;
        if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
        destroy_viewport();
        swapchain_.destroy(device_);
        native_swapchain_initialized_ = false;
        swapchain_rebuild_ = false;
    }

private:
    struct vulkan_context
    {
        VkInstance instance{};
        VkPhysicalDevice physical_device{};
        VkDevice device{};
        VkQueue graphics_queue{};
        std::uint32_t graphics_queue_family{};
        render_capabilities capabilities{};
    };

    std::uint32_t scaled_dimension(std::uint32_t value) const noexcept
    {
        return std::max(
            1u, static_cast<std::uint32_t>(std::round(static_cast<float>(value) * resolved_config_.render_scale)));
    }

    void wait_for_in_flight_frames() const
    {
        std::vector<VkFence> fences;
        fences.reserve(swapchain_.frames.size());
        for (const auto& frame : swapchain_.frames)
            if (frame.fence != VK_NULL_HANDLE) fences.push_back(frame.fence);
        if (!fences.empty())
            vkWaitForFences(device_, static_cast<std::uint32_t>(fences.size()), fences.data(), VK_TRUE, UINT64_MAX);
#if ARC_VULKAN_SHARED_VIEWPORT
        for (const auto& [_, output] : shared_viewports_)
            for (const auto& slot : output.slots)
                if (slot.fence != VK_NULL_HANDLE && slot.state == shared_viewport_frame_state::rendering)
                    vkWaitForFences(device_, 1, &slot.fence, VK_TRUE, UINT64_MAX);
#endif
    }

#if ARC_VULKAN_SHARED_VIEWPORT
    void query_shared_viewport_support()
    {
        shared_viewport_supported_ = false;
        get_memory_win32_handle_properties_ = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandlePropertiesKHR"));
        if (get_memory_win32_handle_properties_ == nullptr)
        {
            shared_viewport_failure_ = "VK_KHR_external_memory_win32 is unavailable";
            return;
        }
        VkPhysicalDeviceExternalImageFormatInfo external{};
        external.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
        external.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
        VkPhysicalDeviceImageFormatInfo2 image{};
        image.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
        image.pNext = &external;
        image.format = VK_FORMAT_B8G8R8A8_UNORM;
        image.type = VK_IMAGE_TYPE_2D;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VkExternalImageFormatProperties external_properties{};
        external_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
        VkImageFormatProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
        properties.pNext = &external_properties;
        const auto result = vkGetPhysicalDeviceImageFormatProperties2(physical_device_, &image, &properties);
        const auto features = external_properties.externalMemoryProperties.externalMemoryFeatures;
        const auto compatible = external_properties.externalMemoryProperties.compatibleHandleTypes;
        if (result != VK_SUCCESS || (features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0 ||
            (compatible & VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT) == 0)
        {
            std::ostringstream diagnostic;
            diagnostic << "selected Vulkan adapter cannot import BGRA8 D3D11-compatible textures (query="
                       << describe_vk_result(result) << ", features=0x" << std::hex << features << ", compatible=0x"
                       << compatible << ')';
            shared_viewport_failure_ = std::move(diagnostic).str();
            return;
        }
        if (!create_shared_d3d_device()) return;
        shared_viewport_supported_ = true;
        shared_viewport_failure_.clear();
    }

    bool create_shared_d3d_device()
    {
        VkPhysicalDeviceIDProperties vulkan_id{};
        vulkan_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &vulkan_id;
        vkGetPhysicalDeviceProperties2(physical_device_, &properties);
        if (vulkan_id.deviceLUIDValid != VK_TRUE)
        {
            shared_viewport_failure_ = "selected Vulkan adapter does not expose a Windows adapter LUID";
            return false;
        }

        LUID vulkan_luid{};
        static_assert(sizeof(vulkan_luid) == VK_LUID_SIZE);
        std::memcpy(&vulkan_luid, vulkan_id.deviceLUID, sizeof(vulkan_luid));
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        {
            shared_viewport_failure_ = "failed to create the DXGI factory for shared viewport textures";
            return false;
        }

        Microsoft::WRL::ComPtr<IDXGIAdapter1> selected_adapter;
        for (UINT index = 0;; ++index)
        {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 descriptor{};
            if (SUCCEEDED(candidate->GetDesc1(&descriptor)) &&
                descriptor.AdapterLuid.HighPart == vulkan_luid.HighPart &&
                descriptor.AdapterLuid.LowPart == vulkan_luid.LowPart)
            {
                selected_adapter = std::move(candidate);
                break;
            }
        }
        if (!selected_adapter)
        {
            shared_viewport_failure_ = "could not match the Vulkan adapter to a DXGI adapter";
            return false;
        }

        constexpr std::array<D3D_FEATURE_LEVEL, 3> levels{D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_11_1,
                                                          D3D_FEATURE_LEVEL_11_0};
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
        D3D_FEATURE_LEVEL selected_level{};
        const auto result = D3D11CreateDevice(
            selected_adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels.data(),
            static_cast<UINT>(levels.size()), D3D11_SDK_VERSION, &shared_d3d_device_, &selected_level, &context);
        if (FAILED(result))
        {
            std::ostringstream diagnostic;
            diagnostic << "failed to create the D3D11 interoperability device (HRESULT=0x" << std::hex
                       << static_cast<std::uint32_t>(result) << ')';
            shared_viewport_failure_ = std::move(diagnostic).str();
            return false;
        }
        return true;
    }

    std::uint32_t shared_memory_type(std::uint32_t type_bits) const noexcept
    {
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &properties);
        for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index)
            if ((type_bits & (1u << index)) != 0 &&
                (properties.memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
                return index;
        for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index)
            if ((type_bits & (1u << index)) != 0) return index;
        return UINT32_MAX;
    }

    bool create_shared_output_slots(shared_viewport_output& output)
    {
        shared_viewport_failure_.clear();
        const auto fail_hresult = [this](std::string_view operation, HRESULT result)
        {
            std::ostringstream diagnostic;
            diagnostic << operation << " (HRESULT=0x" << std::hex << static_cast<std::uint32_t>(result) << ')';
            shared_viewport_failure_ = std::move(diagnostic).str();
            return false;
        };
        const auto fail_vk = [this](std::string_view operation, VkResult result)
        {
            std::ostringstream diagnostic;
            diagnostic << operation << " (VkResult=" << static_cast<std::int32_t>(result) << ')';
            shared_viewport_failure_ = std::move(diagnostic).str();
            return false;
        };

        for (auto& slot : output.slots)
        {
            D3D11_TEXTURE2D_DESC texture{};
            texture.Width = output.width;
            texture.Height = output.height;
            texture.MipLevels = 1;
            texture.ArraySize = 1;
            texture.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            texture.SampleDesc.Count = 1;
            texture.Usage = D3D11_USAGE_DEFAULT;
            texture.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
            texture.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
            const auto create_texture_result = shared_d3d_device_->CreateTexture2D(&texture, nullptr, &slot.texture);
            if (FAILED(create_texture_result))
                return fail_hresult("D3D11 CreateTexture2D for shared viewport texture failed", create_texture_result);
            Microsoft::WRL::ComPtr<IDXGIResource1> resource;
            const auto resource_result = slot.texture.As(&resource);
            if (FAILED(resource_result))
                return fail_hresult("D3D11 shared texture QueryInterface<IDXGIResource1> failed", resource_result);
            const auto shared_handle_result = resource->CreateSharedHandle(
                nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &slot.shared_handle);
            if (FAILED(shared_handle_result))
                return fail_hresult("IDXGIResource1::CreateSharedHandle for shared viewport texture failed",
                                    shared_handle_result);

            VkExternalMemoryImageCreateInfo external_image{};
            external_image.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
            external_image.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
            VkImageCreateInfo image{};
            image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image.pNext = &external_image;
            image.imageType = VK_IMAGE_TYPE_2D;
            image.format = VK_FORMAT_B8G8R8A8_UNORM;
            image.extent = {output.width, output.height, 1};
            image.mipLevels = 1;
            image.arrayLayers = 1;
            image.samples = VK_SAMPLE_COUNT_1_BIT;
            image.tiling = VK_IMAGE_TILING_OPTIMAL;
            image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            const auto create_image_result = vkCreateImage(device_, &image, nullptr, &slot.image);
            if (create_image_result != VK_SUCCESS)
                return fail_vk("vkCreateImage for imported D3D11 viewport texture failed", create_image_result);

            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device_, slot.image, &requirements);
            VkMemoryWin32HandlePropertiesKHR handle_properties{};
            handle_properties.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR;
            const auto handle_properties_result = get_memory_win32_handle_properties_(
                device_, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT, slot.shared_handle, &handle_properties);
            if (handle_properties_result != VK_SUCCESS)
                return fail_vk("vkGetMemoryWin32HandlePropertiesKHR for D3D11 viewport texture failed",
                               handle_properties_result);
            const auto memory_type = shared_memory_type(requirements.memoryTypeBits & handle_properties.memoryTypeBits);
            if (memory_type == UINT32_MAX)
            {
                std::ostringstream diagnostic;
                diagnostic << "no compatible Vulkan memory type for imported D3D11 viewport texture (imageTypes=0x"
                           << std::hex << requirements.memoryTypeBits << ", handleTypes=0x"
                           << handle_properties.memoryTypeBits << ')';
                shared_viewport_failure_ = std::move(diagnostic).str();
                return false;
            }
            VkImportMemoryWin32HandleInfoKHR import_memory{};
            import_memory.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
            import_memory.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
            import_memory.handle = slot.shared_handle;
            VkMemoryDedicatedAllocateInfo dedicated{};
            dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
            dedicated.pNext = &import_memory;
            dedicated.image = slot.image;
            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.pNext = &dedicated;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = memory_type;
            const auto allocate_result = vkAllocateMemory(device_, &allocation, nullptr, &slot.memory);
            if (allocate_result != VK_SUCCESS)
                return fail_vk("vkAllocateMemory for imported D3D11 viewport texture failed", allocate_result);
            const auto bind_result = vkBindImageMemory(device_, slot.image, slot.memory, 0);
            if (bind_result != VK_SUCCESS)
                return fail_vk("vkBindImageMemory for imported D3D11 viewport texture failed", bind_result);

            VkCommandPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = graphics_queue_family_;
            const auto command_pool_result = vkCreateCommandPool(device_, &pool, nullptr, &slot.command_pool);
            if (command_pool_result != VK_SUCCESS)
                return fail_vk("vkCreateCommandPool for shared viewport frame failed", command_pool_result);
            VkCommandBufferAllocateInfo command{};
            command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            command.commandPool = slot.command_pool;
            command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command.commandBufferCount = 1;
            const auto command_buffer_result = vkAllocateCommandBuffers(device_, &command, &slot.command_buffer);
            if (command_buffer_result != VK_SUCCESS)
                return fail_vk("vkAllocateCommandBuffers for shared viewport frame failed", command_buffer_result);
            VkFenceCreateInfo fence{};
            fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            const auto fence_result = vkCreateFence(device_, &fence, nullptr, &slot.fence);
            if (fence_result != VK_SUCCESS)
                return fail_vk("vkCreateFence for shared viewport frame failed", fence_result);
            slot.state = shared_viewport_frame_state::available;
        }
        return true;
    }

    void poll_shared_output_fences(shared_viewport_output& output)
    {
        for (auto& slot : output.slots)
            if (slot.state == shared_viewport_frame_state::rendering &&
                vkGetFenceStatus(device_, slot.fence) == VK_SUCCESS)
            {
                slot.state = shared_viewport_frame_state::ready;
                last_completed_frame_ = std::max(last_completed_frame_, slot.frame_id);
            }
    }

    void wait_for_shared_output(shared_viewport_output& output)
    {
        for (auto& slot : output.slots)
            if (slot.fence != VK_NULL_HANDLE && slot.state == shared_viewport_frame_state::rendering)
            {
                vkWaitForFences(device_, 1, &slot.fence, VK_TRUE, UINT64_MAX);
                slot.state = shared_viewport_frame_state::ready;
            }
    }

    surface_frame_result render_shared_viewport_frame(shared_viewport_output& output, shared_viewport_slot& slot)
    {
        output_viewport_width_ = output.width;
        output_viewport_height_ = output.height;
        const auto slot_index = static_cast<std::uint32_t>(&slot - output.slots.data());
        active_frame_index_ = slot_index;
        ensure_viewport(scaled_dimension(output.width), scaled_dimension(output.height));
        if (viewport_image_ == VK_NULL_HANDLE)
            return surface_frame_result::failure({.code = surface_frame_error_code::backend_failure,
                                                  .message = "viewport render target is unavailable"});
        collect_timestamp_results();
        collect_object_pick_result();
        collect_frame_capture_result();
        retire_completed_resources();
        prepare_frame_gpu_resources();
        vkResetFences(device_, 1, &slot.fence);
        vkResetCommandPool(device_, slot.command_pool, 0);
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(slot.command_buffer, &begin) != VK_SUCCESS)
            return surface_frame_result::failure({.code = surface_frame_error_code::backend_failure,
                                                  .message = "failed to begin shared viewport frame"});

        begin_debug_label(slot.command_buffer, "ARC shared viewport frame", {0.16f, 0.75f, 0.65f, 1.0f});
        reset_timestamp_queries(slot.command_buffer);
        execute_compiled_graph(slot.command_buffer);
        transition_viewport(slot.command_buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkImageMemoryBarrier destination{};
        destination.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        destination.oldLayout = slot.initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        destination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        destination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        destination.dstQueueFamilyIndex = graphics_queue_family_;
        destination.image = slot.image;
        destination.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        destination.subresourceRange.levelCount = 1;
        destination.subresourceRange.layerCount = 1;
        destination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &destination);
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = {static_cast<std::int32_t>(viewport_width_), static_cast<std::int32_t>(viewport_height_),
                              1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1] = {static_cast<std::int32_t>(output.width), static_cast<std::int32_t>(output.height), 1};
        vkCmdBlitImage(slot.command_buffer, viewport_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        destination.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        destination.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        destination.srcQueueFamilyIndex = graphics_queue_family_;
        destination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        destination.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        destination.dstAccessMask = 0;
        vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &destination);
        end_debug_label(slot.command_buffer);
        if (vkEndCommandBuffer(slot.command_buffer) != VK_SUCCESS)
            return surface_frame_result::failure({.code = surface_frame_error_code::backend_failure,
                                                  .message = "failed to record shared viewport frame"});
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &slot.command_buffer;
        const auto result = vkQueueSubmit(queue_, 1, &submit, slot.fence);
        if (result != VK_SUCCESS)
            return surface_frame_result::failure(
                {.code = result == VK_ERROR_DEVICE_LOST ? surface_frame_error_code::device_lost
                                                        : surface_frame_error_code::backend_failure,
                 .message = "failed to submit shared viewport frame: " + describe_vk_result(result)});
        slot.frame_id = output.next_frame_id++;
        slot.state = shared_viewport_frame_state::rendering;
        slot.initialized = true;
        return surface_frame_result::success();
    }

    void retire_shared_output(shared_viewport_output& output, bool preserve_identity) noexcept
    {
        for (auto& slot : output.slots)
        {
            if (slot.fence != VK_NULL_HANDLE) vkDestroyFence(device_, slot.fence, nullptr);
            if (slot.command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device_, slot.command_pool, nullptr);
            if (slot.image != VK_NULL_HANDLE) vkDestroyImage(device_, slot.image, nullptr);
            if (slot.memory != VK_NULL_HANDLE) vkFreeMemory(device_, slot.memory, nullptr);
            if (slot.shared_handle != nullptr) CloseHandle(slot.shared_handle);
            slot.texture.Reset();
            slot = {};
        }
        if (!preserve_identity) output = {};
    }

    void destroy_all_shared_viewports() noexcept
    {
        for (auto& [_, output] : shared_viewports_)
        {
            wait_for_shared_output(output);
            retire_shared_output(output, false);
        }
        shared_viewports_.clear();
    }
#endif // ARC_VULKAN_SHARED_VIEWPORT

    struct vulkan_command_context
    {
        VkCommandPool graphics_pool{};
        VkCommandBuffer graphics_buffer{};
        VkFence fence{};
    };

    struct gpu_buffer
    {
        VkBuffer buffer{};
        VmaAllocation allocation{};
    };

    struct debug_overlay_vertex
    {
        math::vector3f position{};
        math::vector4f color{};
    };

    struct debug_overlay_frame_buffer
    {
        gpu_buffer vertices;
        VkDeviceSize capacity{};
        std::uint32_t tested_line_offset{};
        std::uint32_t tested_line_count{};
        std::uint32_t tested_triangle_offset{};
        std::uint32_t tested_triangle_count{};
        std::uint32_t output_line_offset{};
        std::uint32_t output_line_count{};
        std::uint32_t output_triangle_offset{};
        std::uint32_t output_triangle_count{};
    };

    struct gpu_mesh
    {
        gpu_buffer vertices;
        std::vector<gpu_buffer> dynamic_vertices;
        gpu_buffer indices;
        std::vector<mesh_vertex> pending_vertices;
        std::vector<std::uint64_t> uploaded_revisions;
        std::uint64_t vertex_revision{};
        std::uint32_t index_count{};
        bool dynamic{};
    };

    struct gpu_virtual_mesh
    {
        gpu_buffer vertices;
        gpu_buffer indices;
        std::vector<virtual_mesh_cluster> clusters;
        std::uint32_t index_count{};
    };

    struct alignas(16) terrain_resource_uniform
    {
        std::uint32_t sample_resolution{};
        std::uint32_t patch_quads{};
        std::uint32_t reserved[2]{};
        float width{};
        float depth{};
        float padding[2]{};
    };

    struct gpu_terrain
    {
        gpu_buffer heights;
        gpu_buffer weights;
        gpu_buffer parameters;
        VkDescriptorSet descriptor_set{};
        std::uint32_t sample_resolution{};
        std::uint32_t patch_quads{32};
    };

    struct terrain_topology
    {
        gpu_buffer indices;
        std::uint32_t index_count{};
    };

    struct terrain_patch_draw
    {
        terrain_render_data terrain;
        terrain_patch_render_data patch;
        math::matrix4f view_projection{math::identity<float, 4>()};
        math::matrix4f previous_view_projection{math::identity<float, 4>()};
        render_mode mode{render_mode::shaded};
        mesh_visualization_mode visualization{mesh_visualization_mode::standard};
    };

    struct virtual_cluster_draw
    {
        draw_mesh_event draw;
        virtual_mesh_handle mesh{};
        std::uint32_t cluster_index{};
    };

    struct gpu_texture
    {
        texture_data data;
        VkImage image{};
        VmaAllocation allocation{};
        VkImageView view{};
        VkSampler sampler{};
        VkFormat format{};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::uint32_t mip_count{1};
    };

    struct gpu_environment
    {
        environment_descriptor data;
    };

    struct gpu_material
    {
        material_descriptor data;
        std::vector<gpu_buffer> parameter_buffers;
        std::vector<VkDescriptorSet> descriptor_sets;
    };

    struct folded_light_constants
    {
        math::vector3f direction{0.35f, -0.85f, -0.40f};
        math::vector3f color = math::vector3f::one;
        float intensity{1.0f};
    };

    struct vulkan_shadow_atlas
    {
        VkImage image{};
        VmaAllocation allocation{};
        VkImageView array_view{};
        std::array<VkImageView, directional_shadow_layer_count> cascade_views{};
        VkSampler sampler{};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::uint32_t resolution{};
    };

    struct vulkan_local_shadow_atlas
    {
        VkImage image{};
        VmaAllocation allocation{};
        VkImageView view{};
        VkSampler sampler{};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::uint32_t resolution{};
    };

    struct active_local_shadow
    {
        shadow_light_kind kind{shadow_light_kind::spot};
        shadow_atlas_allocation allocation{};
        math::vector3f position{};
        math::vector3f direction{0.0f, -1.0f, 0.0f};
        float range{1.0f};
        float outer_angle{math::pi<float> * 0.25f};
        shadow_settings settings{};
        render_mobility mobility{render_mobility::movable};
        bool redraw{true};
    };

    struct vulkan_shadow_cache
    {
        directional_shadow_cache_key last_directional_key{};
        bool has_directional_key{};
        std::uint64_t static_signature{};
        bool static_layers_valid{};
    };

    struct object_pick_readback
    {
        render_object_pick_request request{};
        std::uint64_t frame_index{};
        std::uint32_t frame_slot{};
        std::unordered_map<std::uint32_t, render_object_id> objects;
        bool active{};
    };

    struct frame_capture_readback
    {
        render_frame_capture_request request{};
        std::uint64_t frame_index{};
        std::uint32_t frame_slot{};
        render_capture_camera_state camera{};
        std::vector<render_capture_image> images;
        std::vector<VkDeviceSize> offsets;
        std::vector<render_capture_object> objects;
        std::vector<std::string> diagnostics;
        VkDeviceSize byte_size{};
        bool active{};
    };

    static math::vector4f cluster_debug_color(std::uint32_t cluster_index) noexcept
    {
        const std::uint32_t hash = cluster_index * 747796405u + 2891336453u;
        const float r = static_cast<float>((hash >> 0u) & 0xffu) / 255.0f;
        const float g = static_cast<float>((hash >> 8u) & 0xffu) / 255.0f;
        const float b = static_cast<float>((hash >> 16u) & 0xffu) / 255.0f;
        return {0.25f + r * 0.75f, 0.25f + g * 0.75f, 0.25f + b * 0.75f, 1.0f};
    }

    void append_render_world(const render_world_event& event)
    {
        if (!event.packet) return;

        const auto& packet = *event.packet;
        const auto make_draw = [&](const render_item& item, bool selected_for_overlay)
        {
            return draw_mesh_event{.gpu_scene_instance = item.gpu_scene_instance,
                                   .mesh = item.mesh,
                                   .material = item.material,
                                   .model = item.model,
                                   .previous_model = item.previous_model,
                                   .view_projection = packet.camera.view_projection,
                                   .previous_view_projection = packet.camera.previous_view_projection,
                                   .world_bounds = item.world_bounds,
                                   .mode = packet.mode,
                                   .visualization = packet.visualization,
                                   .object_id = item.object_id,
                                   .selected = selected_for_overlay,
                                   .casts_shadows = item.casts_shadows,
                                   .receives_shadows = item.receives_shadows,
                                   .mobility = item.mobility,
                                   .shadow_lod_bias = item.shadow_lod_bias,
                                   .maximum_shadow_distance = item.maximum_shadow_distance,
                                   .base_color_tint = item.base_color_tint,
                                   .wire_color = math::vector4f{1.0f, 0.48f, 0.04f, 1.0f},
                                   .label = item.label};
        };
        const auto make_virtual_draw = [&](const virtual_render_item& item, bool selected_for_overlay)
        {
            auto tint = item.base_color_tint;
            auto material = item.material;
            if (packet.visualization == mesh_visualization_mode::cluster_debug)
            {
                tint = cluster_debug_color(item.root_node);
                material = {};
            }
            const auto visualization = packet.visualization == mesh_visualization_mode::cluster_debug
                                           ? mesh_visualization_mode::albedo
                                           : packet.visualization;
            return virtual_cluster_draw{
                .draw = draw_mesh_event{.gpu_scene_instance = item.gpu_scene_instance,
                                        .mesh = item.mesh,
                                        .material = material,
                                        .model = item.model,
                                        .previous_model = item.previous_model,
                                        .view_projection = packet.camera.view_projection,
                                        .previous_view_projection = packet.camera.previous_view_projection,
                                        .world_bounds = item.world_bounds,
                                        .mode = packet.mode,
                                        .visualization = visualization,
                                        .object_id = item.object_id,
                                        .selected = selected_for_overlay,
                                        .casts_shadows = item.casts_shadows,
                                        .receives_shadows = item.receives_shadows,
                                        .mobility = item.mobility,
                                        .shadow_lod_bias = item.shadow_lod_bias,
                                        .maximum_shadow_distance = item.maximum_shadow_distance,
                                        .base_color_tint = tint,
                                        .wire_color = math::vector4f{1.0f, 0.48f, 0.04f, 1.0f},
                                        .label = item.label},
                .mesh = item.mesh,
                .cluster_index = item.root_node};
        };

        frame_directional_lights_.insert(frame_directional_lights_.end(), packet.directional_lights.begin(),
                                         packet.directional_lights.end());
        frame_point_lights_.insert(frame_point_lights_.end(), packet.point_lights.begin(), packet.point_lights.end());
        frame_spot_lights_.insert(frame_spot_lights_.end(), packet.spot_lights.begin(), packet.spot_lights.end());
        frame_area_lights_.insert(frame_area_lights_.end(), packet.area_lights.begin(), packet.area_lights.end());

        if (resolved_config_.features.gpu_driven_rendering)
        {
            for (const auto& item : packet.items)
            {
                if (!item.visible || !item.mesh.valid()) continue;
                frame_draws_.push_back(
                    make_draw(item, packet.overlay == editor_overlay_mode::all_wireframe ||
                                        (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
            }
        }
        else
            for (const auto index : packet.visible_items)
            {
                if (index >= packet.items.size()) continue;
                const auto& item = packet.items[index];
                frame_draws_.push_back(
                    make_draw(item, packet.overlay == editor_overlay_mode::all_wireframe ||
                                        (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
            }

        if (resolved_config_.features.gpu_driven_rendering)
        {
            for (const auto& item : packet.virtual_items)
            {
                if (!item.visible || !item.mesh.valid()) continue;
                frame_virtual_draws_.push_back(make_virtual_draw(
                    item, packet.overlay == editor_overlay_mode::all_wireframe ||
                              (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
            }
        }
        else
            for (const auto index : packet.visible_virtual_items)
            {
                if (index >= packet.virtual_items.size()) continue;
                const auto& item = packet.virtual_items[index];
                frame_virtual_draws_.push_back(make_virtual_draw(
                    item, packet.overlay == editor_overlay_mode::all_wireframe ||
                              (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
            }

        for (const auto& item : packet.items)
        {
            if (!item.visible || !item.casts_shadows || !item.mesh.valid()) continue;
            frame_shadow_draws_.push_back(make_draw(item, item.selected));
        }
        for (const auto& item : packet.virtual_items)
        {
            if (!item.visible || !item.casts_shadows || !item.mesh.valid()) continue;
            frame_virtual_shadow_draws_.push_back(make_virtual_draw(item, item.selected));
        }

        for (const auto& patch : packet.visible_terrain_patches)
        {
            if (patch.terrain_index >= packet.terrains.size()) continue;
            const auto& terrain = packet.terrains[patch.terrain_index];
            if (!terrain.terrain.valid()) continue;
            frame_terrain_draws_.push_back({terrain, patch, packet.camera.view_projection,
                                            packet.camera.previous_view_projection, packet.mode, packet.visualization});
        }

        if (!frame_camera_valid_ ||
            math::length_squared(math::sub(packet.camera.position, frame_camera_.position)) > 100.0f)
            exposure_needs_reset_ = true;
        frame_camera_ = packet.camera;
        frame_camera_valid_ = true;
        frame_environment_ = packet.environment;
        frame_shadows_enabled_ = packet.shadows_enabled;
        last_profile_.virtual_geometry.enabled = resolved_config_.features.virtual_geometry;
        last_profile_.virtual_geometry.raster_path = resolved_config_.features.virtual_geometry_path;
        if (!resolved_config_.features.virtual_geometry)
            last_profile_.virtual_geometry.fallback_reason =
                "Vulkan virtual-geometry traversal, streaming, and visibility rasterization are unavailable; "
                "using conventional LOD geometry";
        frame_debug_overlay_lines_.insert(frame_debug_overlay_lines_.end(), packet.debug_overlay.lines.begin(),
                                          packet.debug_overlay.lines.end());
        frame_debug_overlay_triangles_.insert(frame_debug_overlay_triangles_.end(),
                                              packet.debug_overlay.triangles.begin(),
                                              packet.debug_overlay.triangles.end());
        last_profile_.terrain.hierarchy_nodes += packet.terrain_statistics.hierarchy_nodes;
        last_profile_.terrain.selected_patches += packet.terrain_statistics.selected_patches;
        last_profile_.terrain.culled_nodes += packet.terrain_statistics.culled_nodes;
        last_profile_.terrain.rendered_triangles += packet.terrain_statistics.rendered_triangles;
        for (std::size_t lod = 0; lod < last_profile_.terrain.patches_per_lod.size(); ++lod)
            last_profile_.terrain.patches_per_lod[lod] += packet.terrain_statistics.patches_per_lod[lod];
        if (resolved_config_.features.gpu_driven_rendering)
        {
            auto& profile = last_profile_.gpu_scene;
            profile.enabled = true;
            profile.hzb_occlusion = resolved_config_.features.hzb_occlusion;
            profile.history_valid = packet.camera.history_valid;
            profile.visible_instances =
                static_cast<std::uint32_t>(packet.visible_items.size() + packet.visible_virtual_items.size());
            profile.frustum_rejected =
                static_cast<std::uint32_t>(packet.culled_item_count + packet.culled_virtual_cluster_count);
            profile.indirect_commands = static_cast<std::uint32_t>(packet.items.size() + packet.virtual_items.size());
            if (resolved_config_.features.gpu_binding_model == gpu_resource_binding_model::classic)
                profile.cpu_submissions += static_cast<std::uint32_t>(
                    packet.items.size() + packet.virtual_items.size() + packet.visible_terrain_patches.size());
        }
    }

    void create_support_objects()
    {
        VkPipelineCacheCreateInfo pipeline_cache_info{};
        pipeline_cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (vkCreatePipelineCache(device_, &pipeline_cache_info, nullptr, &vk_pipeline_cache_) != VK_SUCCESS)
            vk_pipeline_cache_ = VK_NULL_HANDLE;

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device_, &properties);
        timestamp_period_ = properties.limits.timestampPeriod;
        max_push_constant_bytes_ = properties.limits.maxPushConstantsSize;

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, families.data());
        timestamps_supported_ =
            graphics_queue_family_ < families.size() && families[graphics_queue_family_].timestampValidBits > 0;

        if (timestamps_supported_)
        {
            VkQueryPoolCreateInfo query_pool{};
            query_pool.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            query_pool.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query_pool.queryCount = max_timestamp_queries_;
            if (vkCreateQueryPool(device_, &query_pool, nullptr, &timestamp_query_pool_) != VK_SUCCESS)
            {
                timestamp_query_pool_ = VK_NULL_HANDLE;
                timestamps_supported_ = false;
            }
        }

        (void)descriptor_slots_.allocate(descriptor_resource_type::sampled_image);

        if (!create_buffer(upload_staging_capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                           upload_staging_) ||
            vmaMapMemory(allocator_, upload_staging_.allocation, &upload_staging_mapped_) != VK_SUCCESS)
        {
            destroy_buffer(upload_staging_);
            upload_staging_mapped_ = nullptr;
            arc::diagnostics::error("render.vulkan", "failed to create the persistent upload staging buffer");
            return;
        }

        upload_arena_ = std::make_unique<gpu_upload_arena>(std::span<std::byte>(
            static_cast<std::byte*>(upload_staging_mapped_), static_cast<std::size_t>(upload_staging_capacity)));

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = graphics_queue_family_;
        if (vkCreateCommandPool(device_, &pool_info, nullptr, &upload_command_pool_) != VK_SUCCESS)
        {
            destroy_upload_objects();
            arc::diagnostics::error("render.vulkan", "failed to create the persistent upload command pool");
            return;
        }

        VkCommandBufferAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool = upload_command_pool_;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &allocate, &upload_command_buffer_) != VK_SUCCESS)
        {
            destroy_upload_objects();
            arc::diagnostics::error("render.vulkan", "failed to allocate the persistent upload command buffer");
            return;
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device_, &fence_info, nullptr, &upload_fence_) != VK_SUCCESS)
        {
            destroy_upload_objects();
            arc::diagnostics::error("render.vulkan", "failed to create the persistent upload fence");
        }
    }

    void destroy_support_objects() noexcept
    {
        deferred_releases_.collect(UINT64_MAX);
        destroy_upload_objects();
        if (timestamp_query_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(device_, timestamp_query_pool_, nullptr);
            timestamp_query_pool_ = VK_NULL_HANDLE;
        }
        if (vk_pipeline_cache_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineCache(device_, vk_pipeline_cache_, nullptr);
            vk_pipeline_cache_ = VK_NULL_HANDLE;
        }
    }

    void retire_completed_resources()
    {
        deferred_releases_.collect(last_completed_frame_);
        frame_arena_.reset();
    }

    void begin_debug_label(VkCommandBuffer command_buffer, std::string_view name,
                           const std::array<float, 4>& color) const
    {
        if (vkCmdBeginDebugUtilsLabelEXT == nullptr || name.empty()) return;

        VkDebugUtilsLabelEXT label{};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name.data();
        std::copy(color.begin(), color.end(), label.color);
        vkCmdBeginDebugUtilsLabelEXT(command_buffer, &label);
    }

    void insert_debug_label(VkCommandBuffer command_buffer, std::string_view name,
                            const std::array<float, 4>& color) const
    {
        if (vkCmdInsertDebugUtilsLabelEXT == nullptr || name.empty()) return;

        VkDebugUtilsLabelEXT label{};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name.data();
        std::copy(color.begin(), color.end(), label.color);
        vkCmdInsertDebugUtilsLabelEXT(command_buffer, &label);
    }

    void end_debug_label(VkCommandBuffer command_buffer) const
    {
        if (vkCmdEndDebugUtilsLabelEXT != nullptr) vkCmdEndDebugUtilsLabelEXT(command_buffer);
    }

    void reset_timestamp_queries(VkCommandBuffer command_buffer)
    {
        next_timestamp_query_ = 0;
        timestamp_scopes_.clear();
        if (timestamp_query_pool_ != VK_NULL_HANDLE)
            vkCmdResetQueryPool(command_buffer, timestamp_query_pool_, 0, max_timestamp_queries_);
    }

    std::uint32_t begin_gpu_scope(VkCommandBuffer command_buffer, std::string_view name)
    {
        begin_debug_label(command_buffer, name, {0.10f, 0.55f, 1.0f, 1.0f});
        if (timestamp_query_pool_ == VK_NULL_HANDLE || next_timestamp_query_ + 1 >= max_timestamp_queries_)
            return UINT32_MAX;

        const std::uint32_t begin_query = next_timestamp_query_++;
        const std::uint32_t end_query = next_timestamp_query_++;
        timestamp_scopes_.push_back({.name = std::string(name), .begin_query = begin_query, .end_query = end_query});
        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_query_pool_, begin_query);
        return end_query;
    }

    void end_gpu_scope(VkCommandBuffer command_buffer, std::uint32_t end_query)
    {
        if (timestamp_query_pool_ != VK_NULL_HANDLE && end_query != UINT32_MAX)
            vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamp_query_pool_, end_query);
        end_debug_label(command_buffer);
    }

    void collect_timestamp_results()
    {
        if (timestamp_query_pool_ == VK_NULL_HANDLE || timestamp_scopes_.empty()) return;

        std::array<std::uint64_t, max_timestamp_queries_> values{};
        const VkResult result =
            vkGetQueryPoolResults(device_, timestamp_query_pool_, 0, max_timestamp_queries_, sizeof(values),
                                  values.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS) return;

        last_profile_.pass_timings.clear();
        last_profile_.pass_timings.reserve(timestamp_scopes_.size());
        for (const auto& scope : timestamp_scopes_)
        {
            if (scope.end_query >= values.size() || values[scope.end_query] < values[scope.begin_query]) continue;
            const auto ticks = values[scope.end_query] - values[scope.begin_query];
            last_profile_.pass_timings.push_back(
                {.name = scope.name,
                 .milliseconds = static_cast<double>(ticks) * static_cast<double>(timestamp_period_) / 1'000'000.0});
        }
    }

    void collect_object_pick_result()
    {
        if (!in_flight_pick_.active || pick_readback_buffer_.buffer == VK_NULL_HANDLE) return;
        if (in_flight_pick_.frame_slot != active_frame_index_)
        {
            if (in_flight_pick_.frame_slot >= swapchain_.frames.size()) return;
            const auto submitting_fence = swapchain_.frames[in_flight_pick_.frame_slot].fence;
            if (submitting_fence == VK_NULL_HANDLE || vkGetFenceStatus(device_, submitting_fence) != VK_SUCCESS) return;
        }

        void* mapped{};
        if (vmaMapMemory(allocator_, pick_readback_buffer_.allocation, &mapped) != VK_SUCCESS) return;

        vmaInvalidateAllocation(allocator_, pick_readback_buffer_.allocation, 0, sizeof(std::uint32_t));
        std::uint32_t encoded_id{};
        std::memcpy(&encoded_id, mapped, sizeof(encoded_id));
        vmaUnmapMemory(allocator_, pick_readback_buffer_.allocation);

        last_pick_result_ = {.request_id = in_flight_pick_.request.request_id,
                             .available = true,
                             .hit = false,
                             .object = {},
                             .x = in_flight_pick_.request.x,
                             .y = in_flight_pick_.request.y,
                             .frame_index = in_flight_pick_.frame_index};

        if (encoded_id != 0)
        {
            if (const auto found = in_flight_pick_.objects.find(encoded_id); found != in_flight_pick_.objects.end())
            {
                last_pick_result_.hit = true;
                last_pick_result_.object = found->second;
            }
        }

        in_flight_pick_ = {};
    }

    void collect_frame_capture_result()
    {
        if (!in_flight_capture_.active || capture_readback_buffer_.buffer == VK_NULL_HANDLE) return;
        if (in_flight_capture_.frame_slot != active_frame_index_)
        {
            if (in_flight_capture_.frame_slot >= swapchain_.frames.size()) return;
            const auto submitting_fence = swapchain_.frames[in_flight_capture_.frame_slot].fence;
            if (submitting_fence == VK_NULL_HANDLE || vkGetFenceStatus(device_, submitting_fence) != VK_SUCCESS) return;
        }

        void* mapped{};
        if (vmaMapMemory(allocator_, capture_readback_buffer_.allocation, &mapped) != VK_SUCCESS) return;
        vmaInvalidateAllocation(allocator_, capture_readback_buffer_.allocation, 0, in_flight_capture_.byte_size);
        const auto* bytes = static_cast<const std::byte*>(mapped);
        for (std::size_t index = 0; index < in_flight_capture_.images.size(); ++index)
        {
            auto& image = in_flight_capture_.images[index];
            std::memcpy(image.data.data(), bytes + in_flight_capture_.offsets[index], image.data.size());
        }
        vmaUnmapMemory(allocator_, capture_readback_buffer_.allocation);

        last_capture_result_ = {.capture_id = in_flight_capture_.request.capture_id,
                                .frame_index = in_flight_capture_.frame_index,
                                .available = true,
                                .succeeded = !in_flight_capture_.images.empty(),
                                .camera = in_flight_capture_.camera,
                                .images = std::move(in_flight_capture_.images),
                                .objects = std::move(in_flight_capture_.objects),
                                .diagnostics = std::move(in_flight_capture_.diagnostics)};
        if (last_capture_result_.images.empty())
            last_capture_result_.diagnostics.emplace_back("none of the requested capture channels are supported");
        in_flight_capture_ = {};
    }

    clustered_light_grid_profile make_clustered_light_profile() const noexcept
    {
        clustered_light_grid_profile profile{};
        const std::uint32_t width = std::max(1u, viewport_width_);
        const std::uint32_t height = std::max(1u, viewport_height_);
        profile.tiles_x = (width + profile.tile_size_pixels - 1u) / profile.tile_size_pixels;
        profile.tiles_y = (height + profile.tile_size_pixels - 1u) / profile.tile_size_pixels;
        profile.cluster_count = profile.tiles_x * profile.tiles_y * profile.depth_slices;
        profile.point_light_references = frame_lighting_.point_count * profile.depth_slices;
        profile.spot_light_references = frame_lighting_.spot_count * profile.depth_slices;
        profile.overflow_count = frame_lighting_.skipped_point_count + frame_lighting_.skipped_spot_count;
        profile.available = true;
        return profile;
    }

    static std::uint64_t light_shadow_key(render_object_id object) noexcept
    {
        return (static_cast<std::uint64_t>(object.generation) << 32u) | static_cast<std::uint64_t>(object.index);
    }

    void update_shadow_profile(std::uint64_t frame_index)
    {
        auto& profile = last_profile_.shadows;
        profile = {};
        active_local_shadows_.clear();
        frame_lighting_.local_shadow_face_count = 0u;
        profile.directional_cascade_count = resolved_config_.directional_shadow_cascades;
        profile.directional_resolution = resolved_config_.directional_shadow_resolution;
        profile.local_atlas_resolution = resolved_config_.local_shadow_atlas_resolution;
        profile.static_cache_hit = last_static_shadow_cache_hit_;
        if (!frame_shadows_enabled_)
        {
            profile.directional_cascade_count = 0;
            profile.local_atlas_resolution = 0;
            profile.fallback_reason = "Shadows disabled by the viewport";
            return;
        }
        const auto detect_moved_static = [&](draw_mesh_event& draw)
        {
            if (draw.mobility != render_mobility::static_object || !draw.object_id.valid()) return;
            std::uint64_t transform_hash = 1469598103934665603ull;
            for (std::size_t index = 0; index < 16u; ++index)
            {
                const float value = draw.model.data()[index];
                transform_hash ^= std::bit_cast<std::uint32_t>(value);
                transform_hash *= 1099511628211ull;
            }
            const auto key = light_shadow_key(draw.object_id);
            const auto previous = static_shadow_transform_hashes_.find(key);
            if (previous != static_shadow_transform_hashes_.end() && previous->second != transform_hash)
            {
                draw.mobility = render_mobility::movable;
                if (reported_moved_static_objects_.insert(key).second)
                    arc::diagnostics::warn(
                        "render.vulkan",
                        "A static shadow caster moved at runtime; treating it as movable while rebuilding caches");
            }
            static_shadow_transform_hashes_[key] = transform_hash;
        };
        for (auto& draw : frame_shadow_draws_)
        {
            detect_moved_static(draw);
            if (!draw.casts_shadows) continue;
            if (draw.mobility == render_mobility::static_object)
                ++profile.static_caster_count;
            else
                ++profile.dynamic_caster_count;
        }
        for (auto& draw : frame_virtual_shadow_draws_)
        {
            detect_moved_static(draw.draw);
            if (!draw.draw.casts_shadows) continue;
            if (draw.draw.mobility == render_mobility::static_object)
                ++profile.static_caster_count;
            else
                ++profile.dynamic_caster_count;
        }
        if (!local_shadow_allocator_) return;

        struct candidate
        {
            shadow_light_kind kind{};
            std::uint64_t key{};
            std::uint32_t resolution{};
            std::uint16_t priority{};
            float score{};
            math::vector3f position{};
            math::vector3f direction{0.0f, -1.0f, 0.0f};
            float range{1.0f};
            float outer_angle{math::pi<float> * 0.25f};
            shadow_settings settings{};
            render_mobility mobility{render_mobility::movable};
            render_object_id object_id{};
        };
        std::vector<candidate> candidates;
        candidates.reserve(frame_point_lights_.size() + frame_spot_lights_.size());
        const auto influence = [&](const math::vector3f& position, float intensity, float range)
        {
            const auto delta = math::sub(position, frame_camera_.position);
            const float distance_squared = std::max(math::length_squared(delta), 1.0f);
            return std::max(intensity, 0.0f) * std::max(range, 0.0f) / distance_squared;
        };
        for (const auto& light : frame_point_lights_)
        {
            if (!light.enabled || !light.casts_shadows || !light.shadow.enabled ||
                resolved_config_.max_shadowed_point_lights == 0u)
                continue;
            candidates.push_back({.kind = shadow_light_kind::point,
                                  .key = light_shadow_key(light.object_id),
                                  .resolution = light.shadow.resolution,
                                  .priority = light.shadow.priority,
                                  .score = static_cast<float>(light.shadow.priority) * 100000.0f +
                                           influence(light.position, light.intensity, light.range),
                                  .position = light.position,
                                  .range = light.range,
                                  .settings = light.shadow,
                                  .mobility = light.mobility,
                                  .object_id = light.object_id});
        }
        for (const auto& light : frame_spot_lights_)
        {
            if (!light.enabled || !light.casts_shadows || !light.shadow.enabled ||
                resolved_config_.max_shadowed_spot_lights == 0u)
                continue;
            candidates.push_back({.kind = shadow_light_kind::spot,
                                  .key = light_shadow_key(light.object_id),
                                  .resolution = light.shadow.resolution,
                                  .priority = light.shadow.priority,
                                  .score = static_cast<float>(light.shadow.priority) * 100000.0f +
                                           influence(light.position, light.intensity, light.range),
                                  .position = light.position,
                                  .direction = light.direction,
                                  .range = light.range,
                                  .outer_angle = light.outer_angle,
                                  .settings = light.shadow,
                                  .mobility = light.mobility,
                                  .object_id = light.object_id});
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const candidate& lhs, const candidate& rhs)
                         {
                             if (lhs.score != rhs.score) return lhs.score > rhs.score;
                             if (lhs.kind != rhs.kind) return lhs.kind < rhs.kind;
                             return lhs.key < rhs.key;
                         });

        std::uint32_t point_count{};
        std::uint32_t spot_count{};
        for (const auto& candidate : candidates)
        {
            auto& count = candidate.kind == shadow_light_kind::point ? point_count : spot_count;
            const std::uint32_t budget = candidate.kind == shadow_light_kind::point
                                             ? resolved_config_.max_shadowed_point_lights
                                             : resolved_config_.max_shadowed_spot_lights;
            if (count >= budget) continue;
            const auto allocation = local_shadow_allocator_->allocate(
                {.kind = candidate.kind,
                 .light_key = candidate.key,
                 .requested_resolution = std::min(candidate.resolution, resolved_config_.max_local_shadow_resolution),
                 .minimum_resolution = 128u,
                 .priority = candidate.priority,
                 .frame_index = frame_index});
            if (allocation)
            {
                ++count;
                bool redraw = true;
                if (candidate.mobility == render_mobility::static_object &&
                    candidate.settings.cache_mode != shadow_cache_mode::always_update)
                {
                    std::uint64_t signature = 1469598103934665603ull;
                    const auto hash = [&](const void* bytes, std::size_t count)
                    {
                        const auto* data = static_cast<const std::byte*>(bytes);
                        for (std::size_t index = 0; index < count; ++index)
                        {
                            signature ^= std::to_integer<unsigned char>(data[index]);
                            signature *= 1099511628211ull;
                        }
                    };
                    hash(&candidate.kind, sizeof(candidate.kind));
                    hash(&candidate.position, sizeof(candidate.position));
                    hash(&candidate.direction, sizeof(candidate.direction));
                    hash(&candidate.range, sizeof(candidate.range));
                    hash(&candidate.outer_angle, sizeof(candidate.outer_angle));
                    hash(&candidate.settings.resolution, sizeof(candidate.settings.resolution));
                    hash(&candidate.settings.priority, sizeof(candidate.settings.priority));
                    hash(&candidate.settings.strength, sizeof(candidate.settings.strength));
                    hash(&candidate.settings.bias, sizeof(candidate.settings.bias));
                    hash(&candidate.settings.normal_bias, sizeof(candidate.settings.normal_bias));
                    hash(&candidate.settings.filter, sizeof(candidate.settings.filter));
                    hash(&candidate.settings.cache_mode, sizeof(candidate.settings.cache_mode));
                    hash(&allocation->handle, sizeof(allocation->handle));
                    hash(&shadow_resource_revision_, sizeof(shadow_resource_revision_));
                    for (const auto& draw : frame_shadow_draws_)
                    {
                        if (!draw.casts_shadows || draw.mobility != render_mobility::static_object) continue;
                        hash(&draw.object_id, sizeof(draw.object_id));
                        hash(draw.model.data(), sizeof(float) * 16u);
                        hash(&draw.mesh, sizeof(draw.mesh));
                        hash(&draw.material, sizeof(draw.material));
                    }
                    for (const auto& draw : frame_virtual_shadow_draws_)
                    {
                        if (!draw.draw.casts_shadows || draw.draw.mobility != render_mobility::static_object) continue;
                        hash(&draw.draw.object_id, sizeof(draw.draw.object_id));
                        hash(draw.draw.model.data(), sizeof(float) * 16u);
                        hash(&draw.mesh, sizeof(draw.mesh));
                        hash(&draw.cluster_index, sizeof(draw.cluster_index));
                    }
                    const auto cached = local_shadow_static_signatures_.find(candidate.key);
                    redraw = cached == local_shadow_static_signatures_.end() || cached->second != signature;
                    local_shadow_static_signatures_[candidate.key] = signature;
                    if (redraw)
                        ++profile.local_cache_misses;
                    else
                        ++profile.local_cache_hits;
                }
                active_local_shadows_.push_back({.kind = candidate.kind,
                                                 .allocation = *allocation,
                                                 .position = candidate.position,
                                                 .direction = candidate.direction,
                                                 .range = candidate.range,
                                                 .outer_angle = candidate.outer_angle,
                                                 .settings = candidate.settings,
                                                 .mobility = candidate.mobility,
                                                 .redraw = redraw});
                const std::uint32_t first_face = frame_lighting_.local_shadow_face_count;
                const std::uint32_t available = max_local_shadow_faces - first_face;
                const std::uint32_t face_count = std::min(allocation->face_count, available);
                const float inverse_atlas =
                    1.0f / static_cast<float>(std::max(resolved_config_.local_shadow_atlas_resolution, 1u));
                static constexpr std::array<math::vector3f, point_shadow_face_count> point_directions{
                    math::vector3f{1.0f, 0.0f, 0.0f}, math::vector3f{-1.0f, 0.0f, 0.0f},
                    math::vector3f{0.0f, 1.0f, 0.0f}, math::vector3f{0.0f, -1.0f, 0.0f},
                    math::vector3f{0.0f, 0.0f, 1.0f}, math::vector3f{0.0f, 0.0f, -1.0f}};
                static constexpr std::array<math::vector3f, point_shadow_face_count> point_ups{
                    math::vector3f{0.0f, -1.0f, 0.0f}, math::vector3f{0.0f, -1.0f, 0.0f},
                    math::vector3f{0.0f, 0.0f, 1.0f},  math::vector3f{0.0f, 0.0f, -1.0f},
                    math::vector3f{0.0f, -1.0f, 0.0f}, math::vector3f{0.0f, -1.0f, 0.0f}};
                const float near_plane = std::clamp(candidate.range * 0.002f, 0.02f, 0.25f);
                const auto projection = perspective_rh_zo(candidate.kind == shadow_light_kind::point
                                                              ? math::pi<float> * 0.5f
                                                              : std::max(candidate.outer_angle * 2.0f, 0.02f),
                                                          near_plane, std::max(candidate.range, near_plane + 0.01f));
                for (std::uint32_t face = 0; face < face_count; ++face)
                {
                    const auto direction = candidate.kind == shadow_light_kind::point
                                               ? point_directions[face]
                                               : math::normalize(candidate.direction, 0.0f);
                    const auto up = candidate.kind == shadow_light_kind::point
                                        ? point_ups[face]
                                        : (std::abs(direction[1]) > 0.98f ? math::vector3f{0.0f, 0.0f, 1.0f}
                                                                          : math::vector3f{0.0f, 1.0f, 0.0f});
                    auto& packed_face = frame_lighting_.local_shadow_faces[first_face + face];
                    packed_face.light_view_projection = math::matmul(
                        projection, look_at_rh(candidate.position, math::add(candidate.position, direction), up));
                    const auto& rect = allocation->faces[face];
                    packed_face.atlas_rect = {static_cast<float>(rect.content_x()) * inverse_atlas,
                                              static_cast<float>(rect.content_y()) * inverse_atlas,
                                              static_cast<float>(rect.content_size()) * inverse_atlas,
                                              static_cast<float>(rect.content_size()) * inverse_atlas};
                    packed_face.parameters = {inverse_atlas, std::max(candidate.settings.bias, 0.0f), near_plane,
                                              candidate.range};
                }
                frame_lighting_.local_shadow_face_count += face_count;
                const auto patch_light = [&](auto& packed_lights, std::uint32_t packed_count)
                {
                    for (std::uint32_t index = 0; index < packed_count; ++index)
                    {
                        auto& packed = packed_lights[index];
                        if (static_cast<std::uint32_t>(packed.object_id_shadow[0]) != candidate.object_id.index ||
                            static_cast<std::uint32_t>(packed.object_id_shadow[1]) != candidate.object_id.generation)
                            continue;
                        packed.shadow_parameters = {static_cast<float>(first_face), static_cast<float>(face_count),
                                                    std::clamp(candidate.settings.strength, 0.0f, 1.0f),
                                                    std::max(candidate.settings.normal_bias, 0.0f)};
                        break;
                    }
                };
                if (candidate.kind == shadow_light_kind::point)
                    patch_light(frame_lighting_.point_lights, frame_lighting_.point_count);
                else
                    patch_light(frame_lighting_.spot_lights, frame_lighting_.spot_count);
            }
            else
                profile.fallback_reason = "local shadow atlas exhausted; affected lights render unshadowed";
        }
        profile.shadowed_point_lights = point_count;
        profile.shadowed_spot_lights = spot_count;
        const auto statistics = local_shadow_allocator_->statistics();
        profile.local_allocation_count = statistics.allocation_count;
        profile.local_occupied_texels = statistics.occupied_texels;
        profile.local_eviction_count = statistics.eviction_count;
        profile.local_resolution_reductions = statistics.resolution_reduction_count;
        profile.screen_space_shadows = false;
        if (resolved_config_.screen_space_shadows && profile.fallback_reason.empty())
            profile.fallback_reason =
                "screen-space shadow passes selected but unavailable in the Vulkan compatibility path";
    }

    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage, gpu_buffer& out)
    {
        VkBufferCreateInfo buffer{};
        buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer.size = size;
        buffer.usage = usage;
        buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = memory_usage;
        return vmaCreateBuffer(allocator_, &buffer, &allocation, &out.buffer, &out.allocation, nullptr) == VK_SUCCESS;
    }

    bool submit_upload_commands(VkCommandBuffer command_buffer)
    {
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence{};
        if (vkCreateFence(device_, &fence_info, nullptr, &fence) != VK_SUCCESS) return false;

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        const VkResult submit_result = vkQueueSubmit(queue_, 1, &submit, fence);
        if (submit_result == VK_SUCCESS) vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device_, fence, nullptr);
        return submit_result == VK_SUCCESS;
    }

    void destroy_upload_objects() noexcept
    {
        upload_arena_.reset();
        if (upload_fence_ != VK_NULL_HANDLE)
        {
            vkDestroyFence(device_, upload_fence_, nullptr);
            upload_fence_ = VK_NULL_HANDLE;
        }
        if (upload_timeline_ != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device_, upload_timeline_, nullptr);
            upload_timeline_ = VK_NULL_HANDLE;
            upload_timeline_value_ = 0;
        }
        upload_timeline_enabled_ = false;
        if (upload_command_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device_, upload_command_pool_, nullptr);
            upload_command_pool_ = VK_NULL_HANDLE;
            upload_command_buffer_ = VK_NULL_HANDLE;
        }
        if (upload_staging_mapped_ != nullptr && upload_staging_.allocation != VK_NULL_HANDLE)
        {
            vmaUnmapMemory(allocator_, upload_staging_.allocation);
            upload_staging_mapped_ = nullptr;
        }
        destroy_buffer(upload_staging_);
        upload_batch_active_ = false;
        upload_batch_has_work_ = false;
    }

    bool begin_upload_batch()
    {
        if (upload_batch_active_) return true;
        if (!upload_arena_ || upload_command_pool_ == VK_NULL_HANDLE || upload_command_buffer_ == VK_NULL_HANDLE ||
            upload_fence_ == VK_NULL_HANDLE)
        {
            return false;
        }

        upload_arena_->retire_completed(std::numeric_limits<std::uint64_t>::max());
        upload_arena_->begin_frame(upload_frame_);
        if (vkResetCommandPool(device_, upload_command_pool_, 0) != VK_SUCCESS) return false;

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(upload_command_buffer_, &begin) != VK_SUCCESS) return false;
        upload_batch_active_ = true;
        upload_batch_has_work_ = false;
        return true;
    }

    upload_allocation reserve_upload(VkDeviceSize size, std::size_t alignment)
    {
        if (!begin_upload_batch()) return {};

        auto allocation = upload_arena_->try_allocate(static_cast<std::size_t>(size), alignment);
        if (allocation) return allocation;

        if (!flush_upload_batch() || !begin_upload_batch()) return {};
        return upload_arena_->try_allocate(static_cast<std::size_t>(size), alignment);
    }

    bool flush_upload_batch()
    {
        if (!upload_batch_active_) return true;
        if (vkEndCommandBuffer(upload_command_buffer_) != VK_SUCCESS)
        {
            upload_batch_active_ = false;
            return false;
        }
        upload_batch_active_ = false;
        if (!upload_batch_has_work_) return true;

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &upload_command_buffer_;
        VkTimelineSemaphoreSubmitInfo timeline_submit{};
        std::uint64_t signal_value{};
        VkFence completion_fence = upload_fence_;
        if (upload_timeline_enabled_)
        {
            signal_value = ++upload_timeline_value_;
            timeline_submit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_submit.signalSemaphoreValueCount = 1;
            timeline_submit.pSignalSemaphoreValues = &signal_value;
            submit.pNext = &timeline_submit;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &upload_timeline_;
            completion_fence = VK_NULL_HANDLE;
        }
        else
        {
            vkResetFences(device_, 1, &upload_fence_);
        }

        const VkResult submit_result = vkQueueSubmit(queue_, 1, &submit, completion_fence);
        if (submit_result != VK_SUCCESS) return false;
        VkResult wait_result{};
        if (upload_timeline_enabled_)
        {
            VkSemaphoreWaitInfo wait{};
            wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wait.semaphoreCount = 1;
            wait.pSemaphores = &upload_timeline_;
            wait.pValues = &signal_value;
            wait_result = vkWaitSemaphores(device_, &wait, UINT64_MAX);
        }
        else
        {
            wait_result = vkWaitForFences(device_, 1, &upload_fence_, VK_TRUE, UINT64_MAX);
        }
        if (wait_result == VK_SUCCESS) upload_arena_->retire_completed(upload_frame_);
        upload_batch_has_work_ = false;
        return wait_result == VK_SUCCESS;
    }

    void destroy_buffer(gpu_buffer& value) noexcept
    {
        if (value.buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator_, value.buffer, value.allocation);
            value.buffer = VK_NULL_HANDLE;
            value.allocation = VK_NULL_HANDLE;
        }
    }

    bool ensure_pick_readback_buffer()
    {
        if (pick_readback_buffer_.buffer != VK_NULL_HANDLE) return true;
        return create_buffer(sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                             pick_readback_buffer_);
    }

    bool ensure_capture_readback_buffer(VkDeviceSize required_size)
    {
        if (capture_readback_buffer_.buffer != VK_NULL_HANDLE && capture_readback_capacity_ >= required_size)
            return true;
        if (in_flight_capture_.active) return false;
        destroy_buffer(capture_readback_buffer_);
        capture_readback_capacity_ = 0;
        if (!create_buffer(required_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                           capture_readback_buffer_))
            return false;
        capture_readback_capacity_ = required_size;
        return true;
    }

    static bool capture_channel_requested(const render_frame_capture_request& request, render_capture_channel channel)
    {
        return std::ranges::find(request.channels, channel) != request.channels.end();
    }

    static VkDeviceSize align_capture_offset(VkDeviceSize value) noexcept
    {
        constexpr VkDeviceSize alignment = 256;
        return (value + alignment - 1u) & ~(alignment - 1u);
    }

    static std::optional<std::pair<render_capture_format, std::uint32_t>> capture_format_for(VkFormat format)
    {
        switch (format)
        {
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
                return std::pair{render_capture_format::rgba8_unorm, 4u};
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_B8G8R8A8_SRGB:
                return std::pair{render_capture_format::bgra8_unorm, 4u};
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return std::pair{render_capture_format::rgba16_float, 8u};
            case VK_FORMAT_R32_SFLOAT:
            case VK_FORMAT_D32_SFLOAT:
                return std::pair{render_capture_format::r32_float, 4u};
            case VK_FORMAT_R32_UINT:
                return std::pair{render_capture_format::r32_uint, 4u};
            default:
                return std::nullopt;
        }
    }

    void record_frame_capture(VkCommandBuffer command_buffer)
    {
        if (!pending_capture_request_ || in_flight_capture_.active) return;

        frame_capture_readback readback{};
        readback.request = std::move(*pending_capture_request_);
        pending_capture_request_.reset();
        readback.frame_index = last_profile_.frame_index;
        readback.frame_slot = active_frame_index_;
        readback.camera = {.view_projection = frame_camera_.view_projection,
                           .inverse_view_projection = frame_camera_.inverse_view_projection,
                           .projection = frame_camera_.projection,
                           .position = frame_camera_.position,
                           .forward = frame_camera_.forward,
                           .up = frame_camera_.up,
                           .near_plane = frame_camera_.near_plane,
                           .far_plane = frame_camera_.far_plane,
                           .render_width = frame_camera_.render_width,
                           .render_height = frame_camera_.render_height,
                           .output_width = frame_camera_.output_width,
                           .output_height = frame_camera_.output_height};

        const auto append_image = [&](render_capture_channel channel, VkFormat format, std::uint32_t width,
                                      std::uint32_t height) -> bool
        {
            const auto capture_format = capture_format_for(format);
            if (!capture_format || width == 0 || height == 0) return false;
            const VkDeviceSize offset = align_capture_offset(readback.byte_size);
            const VkDeviceSize byte_size = static_cast<VkDeviceSize>(width) * height * capture_format->second;
            render_capture_image image{};
            image.channel = channel;
            image.format = capture_format->first;
            image.width = width;
            image.height = height;
            image.data.resize(static_cast<std::size_t>(byte_size));
            readback.images.push_back(std::move(image));
            readback.offsets.push_back(offset);
            readback.byte_size = offset + byte_size;
            return true;
        };

        if (capture_channel_requested(readback.request, render_capture_channel::output_color))
            append_image(render_capture_channel::output_color, viewport_format_, viewport_width_, viewport_height_);
        if (capture_channel_requested(readback.request, render_capture_channel::scene_color))
            append_image(render_capture_channel::scene_color, scene_color_.format, scene_color_.width,
                         scene_color_.height);
        if (capture_channel_requested(readback.request, render_capture_channel::linear_depth))
            append_image(render_capture_channel::linear_depth, depth_format_, viewport_width_, viewport_height_);
        if (capture_channel_requested(readback.request, render_capture_channel::object_id))
        {
            if (resolved_config_.path == render_path::deferred)
                append_image(render_capture_channel::object_id, gbuffer_object_id_.format, gbuffer_object_id_.width,
                             gbuffer_object_id_.height);
            else
                readback.diagnostics.emplace_back("ObjectID capture is unavailable in the active forward+ path");
        }
        if (capture_channel_requested(readback.request, render_capture_channel::world_normal))
        {
            if (resolved_config_.path == render_path::deferred)
                append_image(render_capture_channel::world_normal, gbuffer_normal_.format, gbuffer_normal_.width,
                             gbuffer_normal_.height);
            else
                readback.diagnostics.emplace_back("World-normal capture is unavailable in the active forward+ path");
        }
        const auto append_deferred_channel =
            [&](render_capture_channel channel, const graph_image& image, std::string_view label)
        {
            if (!capture_channel_requested(readback.request, channel)) return;
            if (resolved_config_.path == render_path::deferred)
                append_image(channel, image.format, image.width, image.height);
            else
                readback.diagnostics.emplace_back(std::string(label) +
                                                  " capture is unavailable in the active forward+ path");
        };
        append_deferred_channel(render_capture_channel::base_color, gbuffer_albedo_, "Base-color");
        append_deferred_channel(render_capture_channel::material_properties, gbuffer_material_, "Material-properties");
        append_deferred_channel(render_capture_channel::emissive, gbuffer_emissive_, "Emissive");
        const std::array unsupported_lighting_channels{
            render_capture_channel::indirect_diffuse, render_capture_channel::reflections,
            render_capture_channel::trace_source, render_capture_channel::mesh_distance_field,
            render_capture_channel::temporal_confidence};
        for (const auto channel : unsupported_lighting_channels)
            if (capture_channel_requested(readback.request, channel))
                readback.diagnostics.emplace_back(
                    "The requested dynamic-lighting debug channel is unavailable in the active Vulkan path");

        if (readback.images.empty() || !ensure_capture_readback_buffer(readback.byte_size))
        {
            if (readback.diagnostics.empty())
                readback.diagnostics.emplace_back(
                    "capture readback allocation failed or no requested channel is supported");
            last_capture_result_ = {.capture_id = readback.request.capture_id,
                                    .frame_index = readback.frame_index,
                                    .available = true,
                                    .succeeded = false,
                                    .camera = readback.camera,
                                    .diagnostics = std::move(readback.diagnostics)};
            return;
        }

        for (const auto& draw : frame_draws_)
        {
            if (draw.object_id.valid()) readback.objects.push_back({draw.object_id.index + 1u, draw.object_id});
        }
        for (const auto& draw : frame_virtual_draws_)
        {
            if (draw.draw.object_id.valid())
                readback.objects.push_back({draw.draw.object_id.index + 1u, draw.draw.object_id});
        }

        for (std::size_t index = 0; index < readback.images.size(); ++index)
        {
            const auto& image = readback.images[index];
            VkImage source{};
            VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
            switch (image.channel)
            {
                case render_capture_channel::output_color:
                    transition_viewport(command_buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                    source = viewport_image_;
                    break;
                case render_capture_channel::scene_color:
                    transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                    source = scene_color_.image;
                    break;
                case render_capture_channel::linear_depth:
                    transition_depth(command_buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                    source = viewport_depth_image_;
                    aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                    break;
                case render_capture_channel::object_id:
                    transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                    source = gbuffer_object_id_.image;
                    break;
                case render_capture_channel::world_normal:
                    transition_graph_image(command_buffer, gbuffer_normal_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                    source = gbuffer_normal_.image;
                    break;
                case render_capture_channel::base_color:
                    transition_graph_image(command_buffer, gbuffer_albedo_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                    source = gbuffer_albedo_.image;
                    break;
                case render_capture_channel::material_properties:
                    transition_graph_image(command_buffer, gbuffer_material_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                    source = gbuffer_material_.image;
                    break;
                case render_capture_channel::emissive:
                    transition_graph_image(command_buffer, gbuffer_emissive_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                    source = gbuffer_emissive_.image;
                    break;
                case render_capture_channel::indirect_diffuse:
                case render_capture_channel::reflections:
                case render_capture_channel::trace_source:
                case render_capture_channel::mesh_distance_field:
                case render_capture_channel::temporal_confidence:
                    break;
            }
            if (source == VK_NULL_HANDLE) continue;
            VkBufferImageCopy region{};
            region.bufferOffset = readback.offsets[index];
            region.imageSubresource.aspectMask = aspect;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {image.width, image.height, 1};
            vkCmdCopyImageToBuffer(command_buffer, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   capture_readback_buffer_.buffer, 1, &region);
        }
        readback.active = true;
        in_flight_capture_ = std::move(readback);
    }

    void destroy_texture(gpu_texture& value) noexcept
    {
        if (value.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device_, value.sampler, nullptr);
            value.sampler = VK_NULL_HANDLE;
        }
        if (value.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, value.view, nullptr);
            value.view = VK_NULL_HANDLE;
        }
        if (value.image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator_, value.image, value.allocation);
            value.image = VK_NULL_HANDLE;
            value.allocation = VK_NULL_HANDLE;
        }
        value.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void destroy_meshes() noexcept
    {
        for (auto& [_, mesh] : meshes_)
        {
            destroy_buffer(mesh.vertices);
            for (auto& vertices : mesh.dynamic_vertices)
                destroy_buffer(vertices);
            destroy_buffer(mesh.indices);
        }
        meshes_.clear();
        for (auto& [_, mesh] : virtual_meshes_)
        {
            destroy_buffer(mesh.vertices);
            destroy_buffer(mesh.indices);
        }
        virtual_meshes_.clear();
        for (auto& [_, terrain] : terrains_)
            destroy_terrain_buffers(terrain);
        terrains_.clear();
        for (auto& [_, topology] : terrain_topologies_)
            destroy_buffer(topology.indices);
        terrain_topologies_.clear();
        for (auto& [_, texture] : textures_)
            destroy_texture(texture);
        textures_.clear();
        for (auto& [_, material] : materials_)
        {
            for (auto& parameters : material.parameter_buffers)
                destroy_buffer(parameters);
        }
        materials_.clear();
        environments_.clear();
        if (terrain_descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, terrain_descriptor_pool_, nullptr);
            terrain_descriptor_pool_ = VK_NULL_HANDLE;
        }
        if (terrain_descriptor_set_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, terrain_descriptor_set_layout_, nullptr);
            terrain_descriptor_set_layout_ = VK_NULL_HANDLE;
        }
    }

    std::optional<VkFormat> vulkan_texture_format(texture_format format) const noexcept
    {
        switch (format)
        {
            case texture_format::rgba8_unorm:
                return VK_FORMAT_R8G8B8A8_UNORM;
            case texture_format::rgba8_srgb:
                return VK_FORMAT_R8G8B8A8_SRGB;
            case texture_format::rgba16f:
                return VK_FORMAT_R16G16B16A16_SFLOAT;
            case texture_format::rgba32f:
                return VK_FORMAT_R32G32B32A32_SFLOAT;
            case texture_format::bc1_rgba_unorm:
                return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            case texture_format::bc1_rgba_srgb:
                return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
            case texture_format::bc2_rgba_unorm:
                return VK_FORMAT_BC2_UNORM_BLOCK;
            case texture_format::bc2_rgba_srgb:
                return VK_FORMAT_BC2_SRGB_BLOCK;
            case texture_format::bc3_rgba_unorm:
                return VK_FORMAT_BC3_UNORM_BLOCK;
            case texture_format::bc3_rgba_srgb:
                return VK_FORMAT_BC3_SRGB_BLOCK;
            case texture_format::bc4_r_unorm:
                return VK_FORMAT_BC4_UNORM_BLOCK;
            case texture_format::bc5_rg_unorm:
                return VK_FORMAT_BC5_UNORM_BLOCK;
            case texture_format::bc6h_rgb_ufloat:
                return VK_FORMAT_BC6H_UFLOAT_BLOCK;
            case texture_format::bc7_rgba_unorm:
                return VK_FORMAT_BC7_UNORM_BLOCK;
            case texture_format::bc7_rgba_srgb:
                return VK_FORMAT_BC7_SRGB_BLOCK;
        }
        return std::nullopt;
    }

    bool texture_format_supported(VkFormat format) const noexcept
    {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device_, format, &properties);
        constexpr VkFormatFeatureFlags required =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        return (properties.optimalTilingFeatures & required) == required;
    }

    bool upload_buffer(const void* source, VkDeviceSize size, VkBufferUsageFlags usage, gpu_buffer& destination)
    {
        if (size == 0) return false;

        const auto staging = reserve_upload(size, 16u);
        if (!staging) return false;
        std::memcpy(staging.bytes.data(), source, static_cast<std::size_t>(size));
        vmaFlushAllocation(allocator_, upload_staging_.allocation, static_cast<VkDeviceSize>(staging.offset), size);

        if (!create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, destination))
            return false;

        VkBufferCopy copy{};
        copy.srcOffset = static_cast<VkDeviceSize>(staging.offset);
        copy.size = size;
        vkCmdCopyBuffer(upload_command_buffer_, upload_staging_.buffer, destination.buffer, 1, &copy);
        upload_batch_has_work_ = true;
        return true;
    }

    bool upload_texture_image(const texture_data& data, gpu_texture& destination)
    {
        const auto format = vulkan_texture_format(data.format);
        if (!format || !texture_format_supported(*format)) return false;

        const bool encoded = data.has_encoded_mips();
        const bool pixels = data.has_pixels();
        if (!encoded && !pixels) return false;

        const auto& upload_bytes = encoded ? data.encoded : data.pixels;
        if (upload_bytes.empty()) return false;

        const auto staging = reserve_upload(upload_bytes.size(), 16u);
        if (!staging) return false;
        std::memcpy(staging.bytes.data(), upload_bytes.data(), upload_bytes.size());
        vmaFlushAllocation(allocator_, upload_staging_.allocation, static_cast<VkDeviceSize>(staging.offset),
                           static_cast<VkDeviceSize>(upload_bytes.size()));

        const bool has_mip_payload = !data.mips.empty();
        const std::uint32_t mip_count = has_mip_payload ? static_cast<std::uint32_t>(data.mips.size()) : 1u;

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = data.dimension == texture_dimension::texture_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        if (data.dimension == texture_dimension::cube) image.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        image.format = *format;
        image.extent = {data.width, data.height, data.dimension == texture_dimension::texture_3d ? data.depth : 1u};
        image.mipLevels = mip_count;
        image.arrayLayers = data.dimension == texture_dimension::texture_3d
                                ? 1u
                                : std::max(1u, data.dimension == texture_dimension::cube ? 6u : data.array_layers);
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &destination.image, &destination.allocation, nullptr) !=
            VK_SUCCESS)
            return false;

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = destination.image;
        view.viewType = data.dimension == texture_dimension::cube         ? VK_IMAGE_VIEW_TYPE_CUBE
                        : data.dimension == texture_dimension::texture_3d ? VK_IMAGE_VIEW_TYPE_3D
                        : data.array_layers > 1                           ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                                          : VK_IMAGE_VIEW_TYPE_2D;
        view.format = *format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = mip_count;
        view.subresourceRange.layerCount = image.arrayLayers;
        if (vkCreateImageView(device_, &view, nullptr, &destination.view) != VK_SUCCESS)
        {
            destroy_texture(destination);
            return false;
        }

        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = mip_count > 1 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.maxLod = static_cast<float>(mip_count);
        if (resolved_config_.features.sampler_anisotropy)
        {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physical_device_, &properties);
            sampler.anisotropyEnable = VK_TRUE;
            sampler.maxAnisotropy = std::min(8.0f, properties.limits.maxSamplerAnisotropy);
        }
        if (vkCreateSampler(device_, &sampler, nullptr, &destination.sampler) != VK_SUCCESS)
        {
            destroy_texture(destination);
            return false;
        }

        VkImageMemoryBarrier to_copy{};
        to_copy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_copy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.image = destination.image;
        to_copy.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_copy.subresourceRange.levelCount = mip_count;
        to_copy.subresourceRange.layerCount = image.arrayLayers;
        vkCmdPipelineBarrier(upload_command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_copy);

        std::vector<VkBufferImageCopy> regions;
        if (has_mip_payload)
        {
            regions.reserve(data.mips.size());
            for (std::uint32_t mip = 0; mip < data.mips.size(); ++mip)
            {
                const auto& source_mip = data.mips[mip];
                VkBufferImageCopy copy{};
                copy.bufferOffset =
                    static_cast<VkDeviceSize>(staging.offset) + static_cast<VkDeviceSize>(source_mip.offset);
                copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copy.imageSubresource.mipLevel = mip;
                copy.imageSubresource.layerCount = image.arrayLayers;
                copy.imageExtent = {source_mip.width, source_mip.height,
                                    data.dimension == texture_dimension::texture_3d ? std::max(1u, data.depth >> mip)
                                                                                    : 1u};
                regions.push_back(copy);
            }
        }
        else
        {
            VkBufferImageCopy copy{};
            copy.bufferOffset = static_cast<VkDeviceSize>(staging.offset);
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.layerCount = image.arrayLayers;
            copy.imageExtent = {data.width, data.height,
                                data.dimension == texture_dimension::texture_3d ? data.depth : 1u};
            regions.push_back(copy);
        }

        vkCmdCopyBufferToImage(upload_command_buffer_, upload_staging_.buffer, destination.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()),
                               regions.data());

        VkImageMemoryBarrier to_shader = to_copy;
        to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(upload_command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_shader);
        upload_batch_has_work_ = true;
        destination.format = *format;
        destination.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        destination.mip_count = mip_count;
        return true;
    }

    void upload_mesh(const mesh_upload_event& event)
    {
        if (!event.mesh || event.mesh->vertices.empty() || event.mesh->indices.empty()) return;

        const VkDeviceSize vertex_size = buffer_size(event.mesh->vertices.size(), sizeof(mesh_vertex));
        const VkDeviceSize index_size = buffer_size(event.mesh->indices.size(), sizeof(std::uint32_t));
        const std::uint64_t key = resource_key(event.handle);
        if (auto found = meshes_.find(key); found != meshes_.end() && found->second.dynamic &&
                                            event.mesh->usage == mesh_usage::dynamic_per_frame &&
                                            found->second.index_count == event.mesh->indices.size() &&
                                            (!found->second.pending_vertices.empty() &&
                                             found->second.pending_vertices.size() == event.mesh->vertices.size()))
        {
            found->second.pending_vertices = event.mesh->vertices;
            ++found->second.vertex_revision;
            return;
        }

        gpu_mesh mesh;
        mesh.dynamic = event.mesh->usage == mesh_usage::dynamic_per_frame;
        const bool vertices_ready =
            mesh.dynamic
                ? [&]
        {
            const auto count = frame_resource_count();
            mesh.dynamic_vertices.resize(count);
            mesh.uploaded_revisions.assign(count, 0u);
            mesh.pending_vertices = event.mesh->vertices;
            mesh.vertex_revision = 1u;
            for (auto& vertices : mesh.dynamic_vertices)
            {
                if (!create_buffer(vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                                   vertices))
                    return false;
            }
            return true;
        }()
                : upload_buffer(event.mesh->vertices.data(), vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices);
        if (!vertices_ready ||
            !upload_buffer(event.mesh->indices.data(), index_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices))
        {
            destroy_buffer(mesh.vertices);
            for (auto& vertices : mesh.dynamic_vertices)
                destroy_buffer(vertices);
            destroy_buffer(mesh.indices);
            arc::diagnostics::error("render.vulkan", "Failed to upload mesh '" + event.label + "'");
            return;
        }

        mesh.index_count = static_cast<std::uint32_t>(event.mesh->indices.size());
        if (auto found = meshes_.find(key); found != meshes_.end())
        {
            auto replaced = std::move(found->second);
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                     [this, replaced]() mutable
                                     {
                                         destroy_buffer(replaced.vertices);
                                         for (auto& vertices : replaced.dynamic_vertices)
                                             destroy_buffer(vertices);
                                         destroy_buffer(replaced.indices);
                                     });
        }
        meshes_[key] = std::move(mesh);
    }

    void retire_mesh(mesh_handle handle)
    {
        const auto found = meshes_.find(resource_key(handle));
        if (found == meshes_.end()) return;
        auto retired = std::move(found->second);
        meshes_.erase(found);
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, retired]() mutable
                                 {
                                     destroy_buffer(retired.vertices);
                                     for (auto& vertices : retired.dynamic_vertices)
                                         destroy_buffer(vertices);
                                     destroy_buffer(retired.indices);
                                 });
    }

    surface_frame_result create_viewport_output(const viewport_output_descriptor& descriptor) override
    {
        if (descriptor.type == viewport_output_type::native_window) return surface_frame_result::success();
#if ARC_VULKAN_SHARED_VIEWPORT
        if (!shared_viewport_supported_)
            return surface_frame_result::failure({.code = surface_frame_error_code::unsupported,
                                                  .message = shared_viewport_failure_.empty()
                                                                 ? "Vulkan shared textures are unsupported"
                                                                 : shared_viewport_failure_});
        if (auto existing = shared_viewports_.find(descriptor.id); existing != shared_viewports_.end())
        {
            existing->second.visible = descriptor.visible;
            existing->second.destroy_pending = false;
            return resize_viewport_output(descriptor.id, descriptor.width, descriptor.height);
        }
        auto& output = shared_viewports_[descriptor.id];
        output.id = descriptor.id;
        output.width = std::max(1u, descriptor.width);
        output.height = std::max(1u, descriptor.height);
        output.visible = descriptor.visible;
        output.generation = ++shared_viewport_generations_[descriptor.id];
        if (!create_shared_output_slots(output))
        {
            retire_shared_output(output, false);
            shared_viewports_.erase(descriptor.id);
            return surface_frame_result::failure({.code = surface_frame_error_code::backend_failure,
                                                  .message = shared_viewport_failure_.empty()
                                                                 ? "failed to create Vulkan shared viewport frame pool"
                                                                 : shared_viewport_failure_});
        }
        return surface_frame_result::success();
#else
        (void)descriptor;
        return surface_frame_result::failure(
            {.code = surface_frame_error_code::unsupported,
             .message = "shared viewport textures are not implemented on this platform"});
#endif
    }

    surface_frame_result resize_viewport_output(std::string_view viewport_id, std::uint32_t width,
                                                std::uint32_t height) override
    {
#if ARC_VULKAN_SHARED_VIEWPORT
        auto found = shared_viewports_.find(std::string(viewport_id));
        if (found == shared_viewports_.end())
            return surface_frame_result::failure(
                {.code = surface_frame_error_code::unavailable, .message = "shared viewport is not created"});
        auto& output = found->second;
        width = std::max(1u, width);
        height = std::max(1u, height);
        if (output.width == width && output.height == height) return surface_frame_result::success();
        if (std::ranges::any_of(output.slots, [](const auto& slot)
                                { return slot.state == shared_viewport_frame_state::consumer_owned; }))
        {
            output.pending_width = width;
            output.pending_height = height;
            return surface_frame_result::success();
        }
        wait_for_shared_output(output);
        retire_shared_output(output, true);
        output.width = width;
        output.height = height;
        ++output.generation;
        shared_viewport_generations_[output.id] = output.generation;
        if (!create_shared_output_slots(output))
            return surface_frame_result::failure({.code = surface_frame_error_code::backend_failure,
                                                  .message = shared_viewport_failure_.empty()
                                                                 ? "failed to resize Vulkan shared viewport frame pool"
                                                                 : shared_viewport_failure_});
        return surface_frame_result::success();
#else
        (void)viewport_id;
        (void)width;
        (void)height;
        return surface_frame_result::failure(
            {.code = surface_frame_error_code::unsupported, .message = "shared viewport textures are unsupported"});
#endif
    }

    surface_frame_result present_viewport_output(std::string_view viewport_id) override
    {
#if ARC_VULKAN_SHARED_VIEWPORT
        auto found = shared_viewports_.find(std::string(viewport_id));
        if (found == shared_viewports_.end())
            return surface_frame_result::failure(
                {.code = surface_frame_error_code::unavailable, .message = "shared viewport is not created"});
        auto& output = found->second;
        if (!output.visible) return surface_frame_result::success();
        if (output.pending_width != 0 &&
            std::ranges::none_of(output.slots, [](const auto& slot)
                                 { return slot.state == shared_viewport_frame_state::consumer_owned; }))
        {
            const auto resize = resize_viewport_output(output.id, output.pending_width, output.pending_height);
            output.pending_width = 0;
            output.pending_height = 0;
            if (!resize) return resize;
        }
        poll_shared_output_fences(output);
        if (std::ranges::any_of(output.slots,
                                [](const auto& slot) { return slot.state == shared_viewport_frame_state::rendering; }))
        {
            ++output.dropped_frames;
            return surface_frame_result::success();
        }
        auto available = std::ranges::find_if(output.slots, [](const auto& slot)
                                              { return slot.state == shared_viewport_frame_state::available; });
        if (available == output.slots.end())
        {
            ++output.dropped_frames;
            return surface_frame_result::success();
        }
        return render_shared_viewport_frame(output, *available);
#else
        (void)viewport_id;
        return surface_frame_result::failure(
            {.code = surface_frame_error_code::unsupported, .message = "shared viewport textures are unsupported"});
#endif
    }

    shared_viewport_frame_result poll_viewport_output(std::string_view viewport_id) override
    {
#if ARC_VULKAN_SHARED_VIEWPORT
        auto found = shared_viewports_.find(std::string(viewport_id));
        if (found == shared_viewports_.end()) return shared_viewport_frame_result::success(std::nullopt);
        auto& output = found->second;
        poll_shared_output_fences(output);
        auto ready =
            std::ranges::max_element(output.slots, {}, [](const auto& slot)
                                     { return slot.state == shared_viewport_frame_state::ready ? slot.frame_id : 0u; });
        if (ready == output.slots.end() || ready->state != shared_viewport_frame_state::ready)
            return shared_viewport_frame_result::success(std::nullopt);
        ready->state = shared_viewport_frame_state::consumer_owned;
        return shared_viewport_frame_result::success(
            shared_viewport_frame{.viewport_id = output.id,
                                  .frame_id = ready->frame_id,
                                  .generation = output.generation,
                                  .width = output.width,
                                  .height = output.height,
                                  .format = viewport_pixel_format::bgra8_unorm,
                                  .texture = {.type = external_gpu_handle_type::win32_nt_handle,
                                              .payload = reinterpret_cast<std::uint64_t>(ready->shared_handle)},
                                  .synchronization = {
                                      .producer_complete = true,
                                      .value = ready->frame_id
                                  }});
#else
        (void)viewport_id;
        return shared_viewport_frame_result::failure(
            {.code = surface_frame_error_code::unsupported, .message = "shared viewport textures are unsupported"});
#endif
    }

    void release_viewport_frame(std::string_view viewport_id, std::uint64_t generation, std::uint64_t frame_id) override
    {
#if ARC_VULKAN_SHARED_VIEWPORT
        auto found = shared_viewports_.find(std::string(viewport_id));
        if (found == shared_viewports_.end() || found->second.generation != generation) return;
        auto& output = found->second;
        const auto slot =
            std::ranges::find_if(output.slots, [&](const auto& candidate) { return candidate.frame_id == frame_id; });
        if (slot != output.slots.end() && slot->state == shared_viewport_frame_state::consumer_owned)
            slot->state = shared_viewport_frame_state::available;
        if (output.destroy_pending &&
            std::ranges::none_of(output.slots, [](const auto& candidate)
                                 { return candidate.state == shared_viewport_frame_state::consumer_owned; }))
        {
            wait_for_shared_output(output);
            retire_shared_output(output, false);
            shared_viewports_.erase(found);
        }
#else
        (void)viewport_id;
        (void)generation;
        (void)frame_id;
#endif
    }

    void set_viewport_output_visible(std::string_view viewport_id, bool visible) override
    {
#if ARC_VULKAN_SHARED_VIEWPORT
        if (auto found = shared_viewports_.find(std::string(viewport_id)); found != shared_viewports_.end())
            found->second.visible = visible;
#else
        (void)viewport_id;
        (void)visible;
#endif
    }

    void destroy_viewport_output(std::string_view viewport_id) override
    {
#if ARC_VULKAN_SHARED_VIEWPORT
        const auto found = shared_viewports_.find(std::string(viewport_id));
        if (found == shared_viewports_.end()) return;
        found->second.visible = false;
        if (std::ranges::any_of(found->second.slots, [](const auto& slot)
                                { return slot.state == shared_viewport_frame_state::consumer_owned; }))
        {
            found->second.destroy_pending = true;
            return;
        }
        wait_for_shared_output(found->second);
        retire_shared_output(found->second, false);
        shared_viewports_.erase(found);
#else
        (void)viewport_id;
#endif
    }

    bool ensure_terrain_descriptors()
    {
        if (terrain_descriptor_set_layout_ != VK_NULL_HANDLE && terrain_descriptor_pool_ != VK_NULL_HANDLE) return true;
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        bindings[0] = {0u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bindings[1] = {1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bindings[2] = {2u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &terrain_descriptor_set_layout_) != VK_SUCCESS)
            return false;

        constexpr std::uint32_t capacity = 2048u;
        const std::array<VkDescriptorPoolSize, 2> sizes{
            {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, capacity * 2u}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, capacity}}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = capacity;
        pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        pool.pPoolSizes = sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &terrain_descriptor_pool_) != VK_SUCCESS)
        {
            vkDestroyDescriptorSetLayout(device_, terrain_descriptor_set_layout_, nullptr);
            terrain_descriptor_set_layout_ = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }

    bool allocate_terrain_descriptor(gpu_terrain& terrain)
    {
        if (!ensure_terrain_descriptors()) return false;
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = terrain_descriptor_pool_;
        allocate.descriptorSetCount = 1u;
        allocate.pSetLayouts = &terrain_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &terrain.descriptor_set) != VK_SUCCESS) return false;
        const VkDescriptorBufferInfo heights{terrain.heights.buffer, 0u, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo weights{terrain.weights.buffer, 0u, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo parameters{terrain.parameters.buffer, 0u, sizeof(terrain_resource_uniform)};
        std::array<VkWriteDescriptorSet, 3> writes{};
        const std::array<const VkDescriptorBufferInfo*, 3> infos{&heights, &weights, &parameters};
        const std::array<VkDescriptorType, 3> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        for (std::size_t index = 0; index < writes.size(); ++index)
        {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = terrain.descriptor_set;
            writes[index].dstBinding = static_cast<std::uint32_t>(index);
            writes[index].descriptorCount = 1u;
            writes[index].descriptorType = types[index];
            writes[index].pBufferInfo = infos[index];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        return true;
    }

    void destroy_terrain_buffers(gpu_terrain& terrain) noexcept
    {
        destroy_buffer(terrain.heights);
        destroy_buffer(terrain.weights);
        destroy_buffer(terrain.parameters);
    }

    void upload_terrain(const terrain_upload_event& event)
    {
        if (!event.terrain || event.terrain->heights.empty() || event.terrain->weights.empty()) return;
        gpu_terrain terrain;
        terrain.sample_resolution = event.terrain->sample_resolution;
        terrain.patch_quads = event.terrain->lod.patch_quads;
        const terrain_resource_uniform parameters{event.terrain->sample_resolution,
                                                  event.terrain->lod.patch_quads,
                                                  {},
                                                  event.terrain->width,
                                                  event.terrain->depth,
                                                  {}};
        const auto height_bytes = buffer_size(event.terrain->heights.size(), sizeof(float));
        const auto weight_bytes = buffer_size(event.terrain->weights.size(), sizeof(event.terrain->weights[0]));
        if (!ensure_terrain_topologies(terrain.patch_quads) ||
            !upload_buffer(event.terrain->heights.data(), height_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           terrain.heights) ||
            !upload_buffer(event.terrain->weights.data(), weight_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           terrain.weights) ||
            !upload_buffer(&parameters, sizeof(parameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, terrain.parameters) ||
            !allocate_terrain_descriptor(terrain))
        {
            destroy_terrain_buffers(terrain);
            arc::diagnostics::error("render.vulkan", "Failed to upload terrain '" + event.label + "'");
            return;
        }
        const auto key = resource_key(event.handle);
        if (auto found = terrains_.find(key); found != terrains_.end())
        {
            auto replaced = std::move(found->second);
            const auto descriptor = replaced.descriptor_set;
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                     [this, replaced, descriptor]() mutable
                                     {
                                         destroy_terrain_buffers(replaced);
                                         if (descriptor != VK_NULL_HANDLE && terrain_descriptor_pool_ != VK_NULL_HANDLE)
                                             vkFreeDescriptorSets(device_, terrain_descriptor_pool_, 1u, &descriptor);
                                     });
        }
        terrains_[key] = std::move(terrain);
        last_profile_.terrain.height_bytes += static_cast<std::uint64_t>(height_bytes);
        last_profile_.terrain.weight_bytes += static_cast<std::uint64_t>(weight_bytes);
        last_profile_.terrain.uploaded_height_bytes += static_cast<std::uint64_t>(height_bytes);
        last_profile_.terrain.uploaded_weight_bytes += static_cast<std::uint64_t>(weight_bytes);
    }

    bool ensure_terrain_topologies(std::uint32_t patch_quads)
    {
        for (std::uint8_t mask = 0u; mask < 16u; ++mask)
        {
            const auto key = (patch_quads << 8u) | mask;
            if (terrain_topologies_.contains(key)) continue;
            const auto indices = make_terrain_patch_indices(patch_quads, mask);
            terrain_topology topology;
            if (indices.empty() || !upload_buffer(indices.data(), buffer_size(indices.size(), sizeof(std::uint32_t)),
                                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, topology.indices))
                return false;
            topology.index_count = static_cast<std::uint32_t>(indices.size());
            terrain_topologies_.emplace(key, std::move(topology));
        }
        return true;
    }

    template <typename T>
    bool update_terrain_rows(VkBuffer destination, std::uint32_t destination_resolution,
                             const terrain_sample_region& region, std::uint32_t row_stride,
                             const std::vector<T>& values)
    {
        if (destination == VK_NULL_HANDLE || row_stride < region.width() ||
            values.size() < static_cast<std::size_t>(row_stride) * region.height())
            return false;
        const auto byte_size = buffer_size(values.size(), sizeof(T));
        const auto staging = reserve_upload(byte_size, alignof(T));
        if (!staging) return false;
        std::memcpy(staging.bytes.data(), values.data(), static_cast<std::size_t>(byte_size));
        vmaFlushAllocation(allocator_, upload_staging_.allocation, static_cast<VkDeviceSize>(staging.offset),
                           byte_size);
        std::vector<VkBufferCopy> copies(region.height());
        for (std::uint32_t row = 0; row < region.height(); ++row)
            copies[row] = {
                .srcOffset =
                    static_cast<VkDeviceSize>(staging.offset) + static_cast<VkDeviceSize>(row) * row_stride * sizeof(T),
                .dstOffset =
                    (static_cast<VkDeviceSize>(region.min_z + row) * destination_resolution + region.min_x) * sizeof(T),
                .size = static_cast<VkDeviceSize>(region.width()) * sizeof(T)};
        vkCmdCopyBuffer(upload_command_buffer_, upload_staging_.buffer, destination,
                        static_cast<std::uint32_t>(copies.size()), copies.data());
        upload_batch_has_work_ = true;
        return true;
    }

    void update_terrain_heights(const terrain_height_update_event& event)
    {
        if (!event.update) return;
        const auto found = terrains_.find(resource_key(event.handle));
        if (found == terrains_.end()) return;
        if (!update_terrain_rows(found->second.heights.buffer, found->second.sample_resolution, event.update->region,
                                 event.update->row_stride, event.update->values))
            arc::diagnostics::warn("render.vulkan", "Failed to upload a terrain height region");
        else
            last_profile_.terrain.uploaded_height_bytes += static_cast<std::uint64_t>(event.update->region.width()) *
                                                           event.update->region.height() * sizeof(float);
    }

    void update_terrain_weights(const terrain_weight_update_event& event)
    {
        if (!event.update) return;
        const auto found = terrains_.find(resource_key(event.handle));
        if (found == terrains_.end()) return;
        if (!update_terrain_rows(found->second.weights.buffer, found->second.sample_resolution, event.update->region,
                                 event.update->row_stride, event.update->values))
            arc::diagnostics::warn("render.vulkan", "Failed to upload a terrain weight region");
        else
            last_profile_.terrain.uploaded_weight_bytes += static_cast<std::uint64_t>(event.update->region.width()) *
                                                           event.update->region.height() *
                                                           sizeof(event.update->values[0]);
    }

    void retire_terrain(terrain_handle handle)
    {
        const auto found = terrains_.find(resource_key(handle));
        if (found == terrains_.end()) return;
        auto retired = std::move(found->second);
        terrains_.erase(found);
        const auto descriptor = retired.descriptor_set;
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, retired, descriptor]() mutable
                                 {
                                     destroy_terrain_buffers(retired);
                                     if (descriptor != VK_NULL_HANDLE && terrain_descriptor_pool_ != VK_NULL_HANDLE)
                                         vkFreeDescriptorSets(device_, terrain_descriptor_pool_, 1u, &descriptor);
                                 });
    }

    VkBuffer mesh_vertex_buffer(const gpu_mesh& mesh) const noexcept
    {
        if (!mesh.dynamic) return mesh.vertices.buffer;
        const auto slot = current_frame_slot();
        return slot < mesh.dynamic_vertices.size() ? mesh.dynamic_vertices[slot].buffer : VK_NULL_HANDLE;
    }

    void update_dynamic_mesh_vertices()
    {
        const auto slot = current_frame_slot();
        const auto frame_count = frame_resource_count();
        for (auto& [_, mesh] : meshes_)
        {
            if (mesh.dynamic && mesh.dynamic_vertices.size() != frame_count)
            {
                wait_for_in_flight_frames();
                for (auto& vertices : mesh.dynamic_vertices)
                    destroy_buffer(vertices);
                mesh.dynamic_vertices.assign(frame_count, {});
                mesh.uploaded_revisions.assign(frame_count, 0u);
                const auto bytes = buffer_size(mesh.pending_vertices.size(), sizeof(mesh_vertex));
                for (auto& vertices : mesh.dynamic_vertices)
                {
                    if (!create_buffer(bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, vertices))
                        arc::diagnostics::error("render.vulkan", "Failed to resize per-frame dynamic mesh buffers");
                }
            }
            if (!mesh.dynamic || slot >= mesh.dynamic_vertices.size() || slot >= mesh.uploaded_revisions.size() ||
                mesh.uploaded_revisions[slot] == mesh.vertex_revision)
                continue;
            auto& target = mesh.dynamic_vertices[slot];
            void* mapped{};
            if (target.buffer == VK_NULL_HANDLE || vmaMapMemory(allocator_, target.allocation, &mapped) != VK_SUCCESS)
                continue;
            const auto bytes = buffer_size(mesh.pending_vertices.size(), sizeof(mesh_vertex));
            std::memcpy(mapped, mesh.pending_vertices.data(), static_cast<std::size_t>(bytes));
            vmaFlushAllocation(allocator_, target.allocation, 0, bytes);
            vmaUnmapMemory(allocator_, target.allocation);
            mesh.uploaded_revisions[slot] = mesh.vertex_revision;
        }
    }

    void upload_virtual_mesh(const virtual_mesh_upload_event& event)
    {
        if (!event.mesh || event.mesh->vertices.empty() || event.mesh->indices.empty() || event.mesh->clusters.empty())
            return;

        gpu_virtual_mesh mesh;
        const VkDeviceSize vertex_size = buffer_size(event.mesh->vertices.size(), sizeof(mesh_vertex));
        const VkDeviceSize index_size = buffer_size(event.mesh->indices.size(), sizeof(std::uint32_t));
        if (!upload_buffer(event.mesh->vertices.data(), vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           mesh.vertices) ||
            !upload_buffer(event.mesh->indices.data(), index_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices))
        {
            destroy_buffer(mesh.vertices);
            destroy_buffer(mesh.indices);
            arc::diagnostics::error("render.vulkan", "Failed to upload virtual mesh '" + event.label + "'");
            return;
        }

        mesh.index_count = static_cast<std::uint32_t>(event.mesh->indices.size());
        mesh.clusters = event.mesh->clusters;
        const std::uint64_t key = resource_key(event.handle);
        if (auto found = virtual_meshes_.find(key); found != virtual_meshes_.end())
        {
            auto retired = std::move(found->second);
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                     [this, retired]() mutable
                                     {
                                         destroy_buffer(retired.vertices);
                                         destroy_buffer(retired.indices);
                                     });
        }
        virtual_meshes_[key] = std::move(mesh);
    }

    void retire_virtual_mesh(virtual_mesh_handle handle)
    {
        const auto found = virtual_meshes_.find(resource_key(handle));
        if (found == virtual_meshes_.end()) return;
        auto retired = std::move(found->second);
        virtual_meshes_.erase(found);
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, retired]() mutable
                                 {
                                     destroy_buffer(retired.vertices);
                                     destroy_buffer(retired.indices);
                                 });
    }

    void upload_texture(const texture_upload_event& event)
    {
        if (!event.texture) return;

        gpu_texture texture{.data = *event.texture};
        const bool uploaded = upload_texture_image(*event.texture, texture);
        if (!uploaded && event.texture->dds && event.texture->compressed)
        {
            arc::diagnostics::warn(
                "render.vulkan",
                "DDS texture '" + event.label +
                    "' uses a compressed format unsupported by this Vulkan device; using fallback descriptors");
        }
        else if (!uploaded && !event.texture->has_pixels() && !event.texture->encoded.empty())
        {
            arc::diagnostics::debug("render.vulkan", "Texture '" + event.label +
                                                         "' kept as encoded data until image decoding is available");
        }

        const std::uint64_t key = resource_key(event.handle);
        if (auto found = textures_.find(key); found != textures_.end()) destroy_texture(found->second);
        textures_[key] = std::move(texture);
    }

    void upload_material(const material_upload_event& event)
    {
        if (!event.material) return;

        auto& material = materials_[resource_key(event.handle)];
        material.data = *event.material;
    }

    void upload_environment(const environment_upload_event& event)
    {
        if (!event.environment) return;

        auto environment = *event.environment;
        if (!environment.prefiltered)
        {
            environment.diffuse_irradiance = environment.fallback_color;
            environment.diffuse_intensity = environment.intensity;
        }
        environments_[resource_key(event.handle)] = gpu_environment{.data = std::move(environment)};
        active_environment_ = event.handle;
    }

    const environment_descriptor* active_environment() const noexcept
    {
        const auto found = environments_.find(resource_key(active_environment_));
        return found == environments_.end() ? nullptr : &found->second.data;
    }

    void update_environment_profile(const environment_descriptor* lighting_environment)
    {
        auto& profile = last_profile_.environment;
        profile = {};
        profile.enabled = frame_environment_.enabled;
        profile.sky_visible = frame_environment_.enabled && frame_environment_.sky_visible;
        profile.affects_lighting = frame_environment_.affect_lighting && frame_environment_.lighting.enabled;
        switch (frame_environment_.source)
        {
            case sky_source_mode::physical_atmosphere:
                profile.source = "Physical atmosphere";
                break;
            case sky_source_mode::hdri:
                profile.source = "HDRI";
                break;
            case sky_source_mode::solid_color:
                profile.source = "Solid color";
                break;
        }

        if (!profile.enabled)
        {
            profile.quality_path = "Disabled";
            profile.atmosphere_lut_state = "Not required";
        }
        else if (frame_environment_.source == sky_source_mode::physical_atmosphere)
        {
            profile.quality_path = resolved_config_.quality == render_quality_tier::low
                                       ? "Analytic low-tier"
                                       : "Analytic compatibility fallback";
            profile.atmosphere_lut_state = resolved_config_.quality == render_quality_tier::low
                                               ? "Not required by low tier"
                                               : "Graph scheduled; Vulkan execution pending";
        }
        else
        {
            profile.quality_path = "Texture/constant composite";
            profile.atmosphere_lut_state = "Not required";
        }

        if (!profile.affects_lighting)
            profile.environment_lighting_state = "Disabled";
        else if (lighting_environment && lighting_environment->prefiltered)
            profile.environment_lighting_state = "Prefiltered environment";
        else
            profile.environment_lighting_state = "Diffuse fallback";

        // The graph owns the future standard-tier cloud shadow pass, but the
        // current Vulkan executor does not allocate or sample that texture yet.
        profile.cloud_shadow_resolution = 0;
        profile.fallback_reason = frame_environment_.fallback_reason;
        if (frame_environment_.source == sky_source_mode::hdri &&
            (!frame_environment_.hdri_texture.valid() ||
             textures_.find(resource_key(frame_environment_.hdri_texture)) == textures_.end()))
        {
            profile.fallback_reason = "HDRI texture is unavailable; using the visible fallback color";
        }
        else if (frame_environment_.source == sky_source_mode::physical_atmosphere &&
                 resolved_config_.quality != render_quality_tier::low && profile.fallback_reason.empty())
        {
            profile.fallback_reason = "Atmosphere LUT execution is not available in Vulkan yet; using the analytic sky";
        }
    }

    packed_gpu_scene_instance pack_gpu_scene_instance(const gpu_scene_instance& source) const
    {
        packed_gpu_scene_instance result{};
        std::copy(source.model.data(), source.model.data() + 16, result.transform.model);
        std::copy(source.previous_model.data(), source.previous_model.data() + 16, result.transform.previous_model);
        for (std::uint32_t component = 0; component < 3; ++component)
        {
            result.visibility.bounds_min[component] = source.world_bounds.min[component];
            result.visibility.bounds_max[component] = source.world_bounds.max[component];
        }
        resource_handle geometry{};
        switch (source.geometry_kind)
        {
            case gpu_scene_geometry_kind::mesh:
            case gpu_scene_geometry_kind::skinned_mesh:
                geometry = source.mesh;
                break;
            case gpu_scene_geometry_kind::terrain:
                geometry = source.terrain;
                break;
            case gpu_scene_geometry_kind::virtual_mesh:
                geometry = source.virtual_mesh;
                break;
        }
        result.visibility.geometry[0] = geometry.index;
        result.visibility.geometry[1] = geometry.generation;
        result.visibility.geometry[2] = source.submesh_or_cluster;
        result.visibility.geometry[3] = static_cast<std::uint32_t>(source.geometry_kind);
        result.visibility.material_flags[0] = source.material.index;
        result.visibility.material_flags[1] = source.material.generation;
        result.visibility.material_flags[2] = source.render_layer_mask;
        result.visibility.material_flags[3] = static_cast<std::uint32_t>(source.flags);
        if (source.geometry_kind == gpu_scene_geometry_kind::mesh ||
            source.geometry_kind == gpu_scene_geometry_kind::skinned_mesh)
        {
            const auto found = meshes_.find(resource_key(source.mesh));
            if (found != meshes_.end()) result.visibility.draw_metadata[0] = found->second.index_count;
        }
        else if (source.geometry_kind == gpu_scene_geometry_kind::virtual_mesh)
        {
            const auto found = virtual_meshes_.find(resource_key(source.virtual_mesh));
            if (found != virtual_meshes_.end() && source.submesh_or_cluster < found->second.clusters.size())
            {
                const auto& cluster = found->second.clusters[source.submesh_or_cluster];
                result.visibility.draw_metadata[0] = cluster.index_count;
                result.visibility.draw_metadata[1] = cluster.first_index;
            }
        }
        result.visibility.distance_error[0] = source.maximum_draw_distance;
        result.visibility.distance_error[1] = source.geometry_error_scale;
        return result;
    }

    bool ensure_gpu_scene_buffer(std::uint32_t required_capacity)
    {
        if (required_capacity <= gpu_scene_capacity_ && gpu_scene_visibility_buffer_.buffer != VK_NULL_HANDLE &&
            gpu_scene_transform_buffer_.buffer != VK_NULL_HANDLE)
            return true;
        const std::uint32_t new_capacity = std::max(256u, std::bit_ceil(std::max(required_capacity, 1u)));
        gpu_buffer replacement_visibility{};
        gpu_buffer replacement_transforms{};
        const auto visibility_bytes = static_cast<VkDeviceSize>(new_capacity) * sizeof(gpu_scene_visibility_record);
        const auto transform_bytes = static_cast<VkDeviceSize>(new_capacity) * sizeof(gpu_scene_transform_record);
        if (!create_buffer(visibility_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_CPU_TO_GPU, replacement_visibility) ||
            !create_buffer(transform_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_CPU_TO_GPU, replacement_transforms))
        {
            destroy_buffer(replacement_visibility);
            destroy_buffer(replacement_transforms);
            return false;
        }

        auto retired_visibility = gpu_scene_visibility_buffer_;
        auto retired_transforms = gpu_scene_transform_buffer_;
        const auto retired_capacity = gpu_scene_capacity_;
        gpu_scene_visibility_buffer_ = replacement_visibility;
        gpu_scene_transform_buffer_ = replacement_transforms;
        gpu_scene_capacity_ = new_capacity;
        gpu_scene_visibility_mirror_.resize(new_capacity);
        gpu_scene_transform_mirror_.resize(new_capacity);
        gpu_visibility_descriptors_dirty_ = true;

        const auto upload_mirror = [&](gpu_buffer& destination, const auto& mirror) -> bool
        {
            void* mapped{};
            if (vmaMapMemory(allocator_, destination.allocation, &mapped) != VK_SUCCESS) return false;
            const auto byte_count =
                static_cast<VkDeviceSize>(mirror.size()) * sizeof(typename std::decay_t<decltype(mirror)>::value_type);
            std::memcpy(mapped, mirror.data(), static_cast<std::size_t>(byte_count));
            vmaFlushAllocation(allocator_, destination.allocation, 0, byte_count);
            vmaUnmapMemory(allocator_, destination.allocation);
            return true;
        };
        if (!upload_mirror(gpu_scene_visibility_buffer_, gpu_scene_visibility_mirror_) ||
            !upload_mirror(gpu_scene_transform_buffer_, gpu_scene_transform_mirror_))
        {
            auto failed_visibility = gpu_scene_visibility_buffer_;
            auto failed_transforms = gpu_scene_transform_buffer_;
            gpu_scene_visibility_buffer_ = retired_visibility;
            gpu_scene_transform_buffer_ = retired_transforms;
            gpu_scene_capacity_ = retired_capacity;
            destroy_buffer(failed_visibility);
            destroy_buffer(failed_transforms);
            return false;
        }
        if (retired_visibility.buffer != VK_NULL_HANDLE || retired_transforms.buffer != VK_NULL_HANDLE)
        {
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                     [this, retired_visibility, retired_transforms]() mutable
                                     {
                                         destroy_buffer(retired_visibility);
                                         destroy_buffer(retired_transforms);
                                     });
        }
        return true;
    }

    void apply_gpu_scene_update(const gpu_scene_update_event& event)
    {
        if (!event.batch) return;
        const auto& batch = *event.batch;
        auto& profile = last_profile_.gpu_scene;
        profile.enabled = true;
        profile.hzb_occlusion = resolved_config_.features.hzb_occlusion;
        profile.submission = resolved_config_.features.submission;
        profile.binding_model = resolved_config_.features.gpu_binding_model;
        profile.capacity = batch.capacity;
        profile.active_instances = batch.active_instance_count;
        profile.geometry_table_entries =
            static_cast<std::uint32_t>(meshes_.size() + virtual_meshes_.size() + terrains_.size());
        profile.material_table_entries = static_cast<std::uint32_t>(materials_.size());
        profile.texture_table_entries = static_cast<std::uint32_t>(textures_.size());
        profile.uploaded_ranges += static_cast<std::uint32_t>(batch.dirty_ranges.size());
        if (!ensure_gpu_scene_buffer(batch.capacity))
        {
            profile.fallback_reason = "GPU Scene buffer allocation failed; using CPU draw submission";
            return;
        }

        bool reset{};
        for (const auto& update : batch.updates)
        {
            if (update.kind == gpu_scene_update_kind::reset)
            {
                reset = true;
            }
            else if (update.handle.index < gpu_scene_visibility_mirror_.size())
            {
                if (update.kind == gpu_scene_update_kind::upsert)
                {
                    const auto packed = pack_gpu_scene_instance(update.instance);
                    gpu_scene_visibility_mirror_[update.handle.index] = packed.visibility;
                    gpu_scene_transform_mirror_[update.handle.index] = packed.transform;
                    ++profile.uploaded_instances;
                }
                else
                {
                    gpu_scene_visibility_mirror_[update.handle.index] = {};
                    gpu_scene_transform_mirror_[update.handle.index] = {};
                    ++profile.destroyed_instances;
                }
            }
        }

        if (batch.dirty_ranges.empty()) return;
        void* mapped_visibility{};
        void* mapped_transforms{};
        const auto visibility_map_result =
            vmaMapMemory(allocator_, gpu_scene_visibility_buffer_.allocation, &mapped_visibility);
        const auto transform_map_result =
            vmaMapMemory(allocator_, gpu_scene_transform_buffer_.allocation, &mapped_transforms);
        if (visibility_map_result != VK_SUCCESS || transform_map_result != VK_SUCCESS)
        {
            if (visibility_map_result == VK_SUCCESS)
                vmaUnmapMemory(allocator_, gpu_scene_visibility_buffer_.allocation);
            if (transform_map_result == VK_SUCCESS) vmaUnmapMemory(allocator_, gpu_scene_transform_buffer_.allocation);
            profile.fallback_reason = "GPU Scene buffer mapping failed; retaining the previous generation";
            return;
        }
        for (const auto& range : batch.dirty_ranges)
        {
            if (range.count == 0u || range.end() > gpu_scene_visibility_mirror_.size()) continue;
            const auto visibility_offset = static_cast<VkDeviceSize>(range.first) * sizeof(gpu_scene_visibility_record);
            const auto visibility_bytes = static_cast<VkDeviceSize>(range.count) * sizeof(gpu_scene_visibility_record);
            const auto transform_offset = static_cast<VkDeviceSize>(range.first) * sizeof(gpu_scene_transform_record);
            const auto transform_bytes = static_cast<VkDeviceSize>(range.count) * sizeof(gpu_scene_transform_record);
            std::memcpy(static_cast<std::byte*>(mapped_visibility) + visibility_offset,
                        gpu_scene_visibility_mirror_.data() + range.first, static_cast<std::size_t>(visibility_bytes));
            std::memcpy(static_cast<std::byte*>(mapped_transforms) + transform_offset,
                        gpu_scene_transform_mirror_.data() + range.first, static_cast<std::size_t>(transform_bytes));
            vmaFlushAllocation(allocator_, gpu_scene_visibility_buffer_.allocation, visibility_offset,
                               visibility_bytes);
            vmaFlushAllocation(allocator_, gpu_scene_transform_buffer_.allocation, transform_offset, transform_bytes);
            profile.uploaded_bytes += visibility_bytes + transform_bytes;
        }
        vmaUnmapMemory(allocator_, gpu_scene_visibility_buffer_.allocation);
        vmaUnmapMemory(allocator_, gpu_scene_transform_buffer_.allocation);
        if (reset) profile.history_valid = false;
    }

    void destroy_gpu_visibility_resources()
    {
        destroy_buffer(gpu_visibility_commands_);
        destroy_buffer(gpu_visibility_counters_);
        if (gpu_visibility_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, gpu_visibility_pipeline_, nullptr);
        if (gpu_visibility_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, gpu_visibility_pipeline_layout_, nullptr);
        if (gpu_visibility_descriptor_pool_ != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device_, gpu_visibility_descriptor_pool_, nullptr);
        if (gpu_visibility_descriptor_set_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, gpu_visibility_descriptor_set_layout_, nullptr);
        gpu_visibility_pipeline_ = VK_NULL_HANDLE;
        gpu_visibility_pipeline_layout_ = VK_NULL_HANDLE;
        gpu_visibility_descriptor_pool_ = VK_NULL_HANDLE;
        gpu_visibility_descriptor_set_layout_ = VK_NULL_HANDLE;
        gpu_visibility_descriptor_set_ = VK_NULL_HANDLE;
        gpu_visibility_capacity_ = 0;
        gpu_visibility_active_ = false;
    }

    bool ensure_gpu_visibility_resources()
    {
        if (!resolved_config_.features.gpu_driven_rendering || gpu_scene_capacity_ == 0 ||
            gpu_scene_visibility_buffer_.buffer == VK_NULL_HANDLE)
            return false;
        const bool hzb_resources_available =
            resolved_config_.features.hzb_occlusion && ensure_hzb_resources(viewport_width_, viewport_height_);

        if (gpu_visibility_capacity_ < gpu_scene_capacity_ || gpu_visibility_commands_.buffer == VK_NULL_HANDLE)
        {
            const auto capacity = std::max(256u, std::bit_ceil(gpu_scene_capacity_));
            gpu_buffer commands{};
            gpu_buffer counters{};
            if (!create_buffer(static_cast<VkDeviceSize>(capacity) * indexed_indirect_command_stride,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VMA_MEMORY_USAGE_GPU_ONLY, commands) ||
                !create_buffer(16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VMA_MEMORY_USAGE_GPU_ONLY, counters))
            {
                destroy_buffer(commands);
                destroy_buffer(counters);
                return false;
            }
            auto retired_commands = gpu_visibility_commands_;
            auto retired_counters = gpu_visibility_counters_;
            gpu_visibility_commands_ = commands;
            gpu_visibility_counters_ = counters;
            gpu_visibility_capacity_ = capacity;
            gpu_visibility_descriptors_dirty_ = true;
            if (retired_commands.buffer != VK_NULL_HANDLE || retired_counters.buffer != VK_NULL_HANDLE)
            {
                deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                         [this, retired_commands, retired_counters]() mutable
                                         {
                                             destroy_buffer(retired_commands);
                                             destroy_buffer(retired_counters);
                                         });
            }
        }

        if (gpu_visibility_descriptor_set_layout_ == VK_NULL_HANDLE)
        {
            std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
            for (std::uint32_t index = 0; index < 3; ++index)
            {
                bindings[index].binding = index;
                bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[index].descriptorCount = 1;
                bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo layout{};
            layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
            layout.pBindings = bindings.data();
            if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &gpu_visibility_descriptor_set_layout_) !=
                VK_SUCCESS)
                return false;

            gpu_visibility_descriptors_dirty_ = true;
        }

        if (gpu_visibility_descriptors_dirty_)
        {
            VkDescriptorPool replacement_pool{};
            VkDescriptorSet replacement_set{};
            const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
                                        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}};
            VkDescriptorPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool.maxSets = 1;
            pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
            pool.pPoolSizes = pool_sizes.data();
            if (vkCreateDescriptorPool(device_, &pool, nullptr, &replacement_pool) != VK_SUCCESS) return false;
            VkDescriptorSetAllocateInfo allocate{};
            allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool = replacement_pool;
            allocate.descriptorSetCount = 1;
            allocate.pSetLayouts = &gpu_visibility_descriptor_set_layout_;
            if (vkAllocateDescriptorSets(device_, &allocate, &replacement_set) != VK_SUCCESS)
            {
                vkDestroyDescriptorPool(device_, replacement_pool, nullptr);
                return false;
            }
            std::array<VkDescriptorBufferInfo, 3> buffers{
                VkDescriptorBufferInfo{gpu_scene_visibility_buffer_.buffer, 0, VK_WHOLE_SIZE},
                VkDescriptorBufferInfo{gpu_visibility_commands_.buffer, 0, VK_WHOLE_SIZE},
                VkDescriptorBufferInfo{gpu_visibility_counters_.buffer, 0, VK_WHOLE_SIZE}};
            std::array<VkWriteDescriptorSet, 3> writes{};
            for (std::uint32_t index = 0; index < writes.size(); ++index)
            {
                writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[index].dstSet = replacement_set;
                writes[index].dstBinding = index;
                writes[index].descriptorCount = 1;
                writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[index].pBufferInfo = &buffers[index];
            }
            vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            const VkDescriptorImageInfo fallback_hzb{white_sampler_, white_view_,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkDescriptorImageInfo, 2> hzb_images{fallback_hzb, fallback_hzb};
            if (hzb_resources_available)
                for (std::size_t index = 0; index < hzb_images.size(); ++index)
                    hzb_images[index] = {hzb_sampler_, hzb_history_[index].view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet hzb_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            hzb_write.dstSet = replacement_set;
            hzb_write.dstBinding = 3;
            hzb_write.descriptorCount = static_cast<std::uint32_t>(hzb_images.size());
            hzb_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            hzb_write.pImageInfo = hzb_images.data();
            vkUpdateDescriptorSets(device_, 1, &hzb_write, 0, nullptr);
            const auto retired_pool = gpu_visibility_descriptor_pool_;
            gpu_visibility_descriptor_pool_ = replacement_pool;
            gpu_visibility_descriptor_set_ = replacement_set;
            if (retired_pool != VK_NULL_HANDLE)
                deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(), [this, retired_pool]()
                                         { vkDestroyDescriptorPool(device_, retired_pool, nullptr); });
            gpu_visibility_descriptors_dirty_ = false;
        }

        if (gpu_visibility_pipeline_ == VK_NULL_HANDLE)
        {
            const auto shader = create_shader_module(builtin::gpu_visibility_indirect_comp_spv,
                                                     std::size(builtin::gpu_visibility_indirect_comp_spv));
            if (shader == VK_NULL_HANDLE) return false;
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpu_visibility_push_constants)};
            VkPipelineLayoutCreateInfo layout{};
            layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout.setLayoutCount = 1;
            layout.pSetLayouts = &gpu_visibility_descriptor_set_layout_;
            layout.pushConstantRangeCount = 1;
            layout.pPushConstantRanges = &push;
            if (vkCreatePipelineLayout(device_, &layout, nullptr, &gpu_visibility_pipeline_layout_) != VK_SUCCESS)
            {
                vkDestroyShaderModule(device_, shader, nullptr);
                return false;
            }
            VkComputePipelineCreateInfo pipeline{};
            pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipeline.stage.module = shader;
            pipeline.stage.pName = "main";
            pipeline.layout = gpu_visibility_pipeline_layout_;
            const auto result =
                vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &gpu_visibility_pipeline_);
            vkDestroyShaderModule(device_, shader, nullptr);
            if (result != VK_SUCCESS) return false;
        }
        return true;
    }

    void dispatch_gpu_visibility(VkCommandBuffer command_buffer)
    {
        gpu_visibility_active_ = false;
        if (!ensure_gpu_visibility_resources())
        {
            if (resolved_config_.features.gpu_driven_rendering)
                last_profile_.gpu_scene.fallback_reason =
                    "GPU visibility resources are unavailable; using direct candidate submission";
            return;
        }

        vkCmdFillBuffer(command_buffer, gpu_visibility_counters_.buffer, 0, VK_WHOLE_SIZE, 0u);
        VkBufferMemoryBarrier input_barrier{};
        input_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        input_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        input_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        input_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        input_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        input_barrier.buffer = gpu_scene_visibility_buffer_.buffer;
        input_barrier.size = VK_WHOLE_SIZE;
        VkBufferMemoryBarrier counter_barrier = input_barrier;
        counter_barrier.buffer = gpu_visibility_counters_.buffer;
        std::array<VkBufferMemoryBarrier, 2> input_barriers{input_barrier, counter_barrier};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             static_cast<std::uint32_t>(input_barriers.size()), input_barriers.data(), 0, nullptr);

        gpu_visibility_push_constants constants{};
        std::copy(frame_camera_.view_projection.data(), frame_camera_.view_projection.data() + 16,
                  constants.view_projection);
        constants.camera_position_and_error[0] = frame_camera_.position[0];
        constants.camera_position_and_error[1] = frame_camera_.position[1];
        constants.camera_position_and_error[2] = frame_camera_.position[2];
        constants.camera_position_and_error[3] = resolved_config_.geometry_error_threshold;
        constants.instance_capacity = gpu_scene_capacity_;
        constants.camera_cut = frame_camera_.camera_cut ? 1u : 0u;
        const bool hzb_available = resolved_config_.features.hzb_occlusion &&
                                   ensure_hzb_resources(viewport_width_, viewport_height_) && hzb_history_valid_ &&
                                   !frame_camera_.camera_cut;
        if (hzb_available)
        {
            const auto previous_generation = static_cast<std::uint32_t>(
                (last_profile_.frame_index + hzb_history_.size() - 1u) % hzb_history_.size());
            auto& previous_hzb = hzb_history_[previous_generation];
            transition_graph_image(command_buffer, previous_hzb, VK_IMAGE_LAYOUT_GENERAL);
            constants.reserved = previous_generation;
        }
        constants.hzb_parameters[0] = static_cast<float>(viewport_width_);
        constants.hzb_parameters[1] = static_cast<float>(viewport_height_);
        constants.hzb_parameters[2] = static_cast<float>(hzb_mip_count_);
        constants.hzb_parameters[3] = hzb_available ? 1.0f : 0.0f;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_visibility_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_visibility_pipeline_layout_, 0, 1,
                                &gpu_visibility_descriptor_set_, 0, nullptr);
        vkCmdPushConstants(command_buffer, gpu_visibility_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, (gpu_scene_capacity_ + 63u) / 64u, 1u, 1u);

        VkBufferMemoryBarrier output_barrier{};
        output_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.buffer = gpu_visibility_commands_.buffer;
        output_barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             0, 0, nullptr, 1, &output_barrier, 0, nullptr);
        gpu_visibility_active_ = true;
        last_profile_.gpu_scene.enabled = true;
        last_profile_.gpu_scene.submission = gpu_submission_path::indirect;
    }

    bool draw_gpu_visibility_command(VkCommandBuffer command_buffer, gpu_scene_instance_handle handle) const
    {
        if (!gpu_visibility_active_ || !handle.valid() || handle.index >= gpu_visibility_capacity_) return false;
        vkCmdDrawIndexedIndirect(command_buffer, gpu_visibility_commands_.buffer,
                                 static_cast<VkDeviceSize>(handle.index) * indexed_indirect_command_stride, 1u,
                                 static_cast<std::uint32_t>(indexed_indirect_command_stride));
        return true;
    }

    void update_light_buffer()
    {
        if (light_buffer_.buffer == VK_NULL_HANDLE)
        {
            if (!create_buffer(sizeof(scene_lighting_data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_TO_GPU, light_buffer_))
            {
                arc::diagnostics::warn("render.vulkan", "Failed to allocate scene light buffer");
                return;
            }
        }

        void* mapped{};
        if (vmaMapMemory(allocator_, light_buffer_.allocation, &mapped) != VK_SUCCESS) return;
        std::memcpy(mapped, &frame_lighting_, sizeof(frame_lighting_));
        vmaFlushAllocation(allocator_, light_buffer_.allocation, 0, sizeof(frame_lighting_));
        vmaUnmapMemory(allocator_, light_buffer_.allocation);
    }

    void warn_about_skipped_lights(const scene_lighting_data& lighting)
    {
        if (lighting.skipped_directional_count > 0)
            arc::diagnostics::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_directional_count) +
                                                        " directional light(s) over the v1 cap");
        if (lighting.skipped_point_count > 0)
            arc::diagnostics::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_point_count) +
                                                        " point light(s) over the v1 cap");
        if (lighting.skipped_spot_count > 0)
            arc::diagnostics::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_spot_count) +
                                                        " spot light(s) over the v1 cap");
    }

    static math::vector3f vector_sub(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
    {
        return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
    }

    static math::vector3f vector_mul(const math::vector3f& value, float scale) noexcept
    {
        return {value[0] * scale, value[1] * scale, value[2] * scale};
    }

    static math::vector3f vector_add(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
    {
        return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
    }

    static float vector_dot(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
    {
        return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
    }

    static math::vector3f vector_normalize(const math::vector3f& value) noexcept
    {
        const float length_sq = std::max(vector_dot(value, value), 0.000001f);
        const float inv_length = 1.0f / std::sqrt(length_sq);
        return vector_mul(value, inv_length);
    }

    folded_light_constants fold_lighting_for_draw(const draw_mesh_event& draw) const noexcept
    {
        const math::vector3f origin{draw.model(0, 3), draw.model(1, 3), draw.model(2, 3)};
        math::vector3f color{frame_lighting_.ambient_color_intensity[0] * frame_lighting_.ambient_color_intensity[3],
                             frame_lighting_.ambient_color_intensity[1] * frame_lighting_.ambient_color_intensity[3],
                             frame_lighting_.ambient_color_intensity[2] * frame_lighting_.ambient_color_intensity[3]};
        math::vector3f weighted_direction{};
        float total_weight{};

        for (std::uint32_t index = 0; index < frame_lighting_.directional_count; ++index)
        {
            const auto& light = frame_lighting_.directional_lights[index];
            const float contribution = std::max(light.direction_intensity[3], 0.0f);
            color = vector_add(
                color, vector_mul({light.color_flags[0], light.color_flags[1], light.color_flags[2]}, contribution));
            weighted_direction = vector_add(
                weighted_direction,
                vector_mul({light.direction_intensity[0], light.direction_intensity[1], light.direction_intensity[2]},
                           contribution));
            total_weight += contribution;
        }

        for (std::uint32_t index = 0; index < frame_lighting_.point_count; ++index)
        {
            const auto& light = frame_lighting_.point_lights[index];
            const math::vector3f position{light.position_range[0], light.position_range[1], light.position_range[2]};
            const float range = std::max(light.position_range[3], 0.001f);
            const math::vector3f to_light = vector_sub(position, origin);
            const float distance_sq = std::max(vector_dot(to_light, to_light), 0.000001f);
            const float attenuation = std::max(0.0f, 1.0f - std::sqrt(distance_sq) / range);
            const float contribution = light.color_intensity[3] * attenuation * attenuation;
            color = vector_add(
                color, vector_mul({light.color_intensity[0], light.color_intensity[1], light.color_intensity[2]},
                                  contribution));
            weighted_direction =
                vector_add(weighted_direction, vector_mul(vector_mul(vector_normalize(to_light), -1.0f), contribution));
            total_weight += contribution;
        }

        for (std::uint32_t index = 0; index < frame_lighting_.spot_count; ++index)
        {
            const auto& light = frame_lighting_.spot_lights[index];
            const math::vector3f position{light.position_range[0], light.position_range[1], light.position_range[2]};
            const float range = std::max(light.position_range[3], 0.001f);
            const math::vector3f to_light = vector_sub(position, origin);
            const float attenuation = std::max(0.0f, 1.0f - std::sqrt(vector_dot(to_light, to_light)) / range);
            const math::vector3f light_forward = vector_normalize(
                {light.direction_inner_angle[0], light.direction_inner_angle[1], light.direction_inner_angle[2]});
            const float cone_cos = vector_dot(vector_mul(vector_normalize(to_light), -1.0f), light_forward);
            const float inner = std::cos(light.direction_inner_angle[3]);
            const float outer = std::cos(light.params[0]);
            const float cone = outer == inner ? 1.0f : std::clamp((cone_cos - outer) / (inner - outer), 0.0f, 1.0f);
            const float contribution = light.color_intensity[3] * attenuation * attenuation * cone;
            color = vector_add(
                color, vector_mul({light.color_intensity[0], light.color_intensity[1], light.color_intensity[2]},
                                  contribution));
            weighted_direction =
                vector_add(weighted_direction, vector_mul(vector_mul(vector_normalize(to_light), -1.0f), contribution));
            total_weight += contribution;
        }

        folded_light_constants folded;
        folded.color = color;
        folded.intensity = 1.0f;
        folded.direction = total_weight > 0.0001f ? vector_normalize(weighted_direction) : folded.direction;
        return folded;
    }

    material_alpha_mode material_alpha_mode_for(const draw_mesh_event& draw) const noexcept
    {
        if (const auto material = materials_.find(resource_key(draw.material)); material != materials_.end())
            return material->second.data.alpha_mode;
        return material_alpha_mode::opaque;
    }

    bool texture_ready(texture_handle handle) const noexcept
    {
        if (!handle.valid()) return false;
        const auto found = textures_.find(resource_key(handle));
        return found != textures_.end() && found->second.view != VK_NULL_HANDLE &&
               found->second.sampler != VK_NULL_HANDLE;
    }

    bool material_is_terrain(const draw_mesh_event& draw) const noexcept
    {
        const auto material = materials_.find(resource_key(draw.material));
        return material != materials_.end() && material->second.data.domain == material_domain::terrain;
    }

    bool material_requires_forward(const draw_mesh_event& draw) const noexcept
    {
        const auto material = materials_.find(resource_key(draw.material));
        if (material == materials_.end()) return false;
        return material->second.data.render_path == material_render_path::clustered_forward;
    }

    mesh_push_constants build_mesh_constants(const draw_mesh_event& draw) const
    {
        const math::matrix4f mvp = math::matmul(draw.view_projection, draw.model);
        mesh_push_constants constants{};
        std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
        std::copy(draw.model.data(), draw.model.data() + 16, constants.model);
        const auto folded_light = fold_lighting_for_draw(draw);
        constants.light_direction_intensity[0] = folded_light.direction[0];
        constants.light_direction_intensity[1] = folded_light.direction[1];
        constants.light_direction_intensity[2] = folded_light.direction[2];
        constants.light_direction_intensity[3] = folded_light.intensity;
        constants.light_color[0] = folded_light.color[0];
        constants.light_color[1] = folded_light.color[1];
        constants.light_color[2] = folded_light.color[2];
        constants.camera_position[0] = frame_camera_.position[0];
        constants.camera_position[1] = frame_camera_.position[1];
        constants.camera_position[2] = frame_camera_.position[2];
        constants.camera_position[3] =
            frame_environment_.enabled ? std::max(frame_environment_.atmosphere.exposure, 0.001f) : 1.0f;
        constants.fog_params[3] = draw.object_id.valid() ? static_cast<float>(draw.object_id.index + 1u) : 0.0f;

        if (frame_environment_.fog.enabled)
        {
            constants.fog_color_density[0] = frame_environment_.fog.color[0];
            constants.fog_color_density[1] = frame_environment_.fog.color[1];
            constants.fog_color_density[2] = frame_environment_.fog.color[2];
            constants.fog_color_density[3] = std::max(0.0f, frame_environment_.fog.density);
            constants.fog_params[0] = std::max(0.0f, frame_environment_.fog.start_distance);
            constants.fog_params[1] = std::max(0.0f, frame_environment_.fog.height_falloff);
            constants.fog_params[2] = std::clamp(frame_environment_.fog.max_opacity, 0.0f, 1.0f);
        }

        if (const auto material = materials_.find(resource_key(draw.material)); material != materials_.end())
        {
            const auto& desc = material->second.data;
            constants.base_color[0] = desc.base_color[0] * draw.base_color_tint[0];
            constants.base_color[1] = desc.base_color[1] * draw.base_color_tint[1];
            constants.base_color[2] = desc.base_color[2] * draw.base_color_tint[2];
            constants.base_color[3] = desc.base_color[3] * draw.base_color_tint[3];
            constants.visualization[1] = desc.metallic;
            constants.visualization[2] = desc.roughness;
            constants.visualization[3] = desc.alpha_cutoff;
            if (desc.domain == material_domain::terrain)
            {
                constants.base_color[0] = draw.base_color_tint[0];
                constants.base_color[1] = draw.base_color_tint[1];
                constants.base_color[2] = draw.base_color_tint[2];
                constants.base_color[3] = draw.base_color_tint[3];
                constants.material_params[0] = desc.terrain_layers[0].world_scale;
                constants.material_params[1] = desc.terrain_layers[1].world_scale;
                constants.material_params[2] = desc.terrain_layers[2].world_scale;
                constants.material_params[3] = desc.terrain_layers[3].world_scale;
                constants.light_color[3] = (texture_ready(desc.terrain_layers[0].base_color_texture) ? 1.0f : 0.0f) +
                                           (texture_ready(desc.terrain_layers[1].base_color_texture) ? 2.0f : 0.0f) +
                                           (texture_ready(desc.terrain_layers[2].base_color_texture) ? 4.0f : 0.0f) +
                                           (texture_ready(desc.terrain_layers[3].base_color_texture) ? 8.0f : 0.0f);
                constants.visualization[1] = (texture_ready(desc.terrain_layers[0].normal_texture) ? 1.0f : 0.0f) +
                                             (texture_ready(desc.terrain_layers[1].normal_texture) ? 2.0f : 0.0f) +
                                             (texture_ready(desc.terrain_layers[2].normal_texture) ? 4.0f : 0.0f) +
                                             (texture_ready(desc.terrain_layers[3].normal_texture) ? 8.0f : 0.0f);
                constants.visualization[2] =
                    (texture_ready(desc.terrain_layers[0].packed_surface_texture) ? 1.0f : 0.0f) +
                    (texture_ready(desc.terrain_layers[1].packed_surface_texture) ? 2.0f : 0.0f) +
                    (texture_ready(desc.terrain_layers[2].packed_surface_texture) ? 4.0f : 0.0f) +
                    (texture_ready(desc.terrain_layers[3].packed_surface_texture) ? 8.0f : 0.0f);
            }
            else
            {
                constants.material_params[0] = desc.normal_scale;
                constants.material_params[1] = desc.occlusion_strength;
                constants.material_params[2] = desc.emissive_strength;
                constants.material_params[3] = static_cast<float>(desc.alpha_mode);
                constants.light_color[3] = (texture_ready(desc.base_color_texture) ? 1.0f : 0.0f) +
                                           (texture_ready(desc.metallic_roughness_texture) ? 2.0f : 0.0f) +
                                           (texture_ready(desc.normal_texture) ? 4.0f : 0.0f) +
                                           (texture_ready(desc.occlusion_texture) ? 8.0f : 0.0f) +
                                           (texture_ready(desc.emissive_texture) ? 16.0f : 0.0f);
            }
        }
        else
        {
            constants.base_color[0] = draw.base_color_tint[0];
            constants.base_color[1] = draw.base_color_tint[1];
            constants.base_color[2] = draw.base_color_tint[2];
            constants.base_color[3] = draw.base_color_tint[3];
        }
        constants.visualization[0] = static_cast<float>(draw.visualization);
        return constants;
    }

    draw_mesh_event terrain_mesh_draw(const terrain_patch_draw& draw) const
    {
        return {.material = draw.terrain.material,
                .model = draw.terrain.model,
                .previous_model = draw.terrain.previous_model,
                .view_projection = draw.view_projection,
                .previous_view_projection = draw.previous_view_projection,
                .world_bounds = draw.terrain.world_bounds,
                .mode = draw.mode,
                .visualization = draw.visualization,
                .object_id = draw.terrain.object_id,
                .selected = draw.terrain.selected,
                .casts_shadows = draw.terrain.cast_shadows,
                .receives_shadows = draw.terrain.receive_shadows,
                .shadow_lod_bias = draw.terrain.shadow_lod_bias,
                .maximum_shadow_distance = draw.terrain.maximum_shadow_distance,
                .label = draw.terrain.label};
    }

    void draw_terrain_patch(VkCommandBuffer command_buffer, const terrain_patch_draw& draw, VkPipeline pipeline,
                            bool write_motion)
    {
        if (pipeline == VK_NULL_HANDLE || terrain_pipeline_layout_ == VK_NULL_HANDLE) return;
        const auto terrain = terrains_.find(resource_key(draw.terrain.terrain));
        if (terrain == terrains_.end()) return;
        const auto topology_key = (terrain->second.patch_quads << 8u) | draw.patch.stitch_mask;
        const auto topology = terrain_topologies_.find(topology_key);
        if (topology == terrain_topologies_.end()) return;
        auto mesh_draw = terrain_mesh_draw(draw);
        auto constants = build_mesh_constants(mesh_draw);
        constants.base_color[0] = static_cast<float>(draw.patch.sample_min_x);
        constants.base_color[1] = static_cast<float>(draw.patch.sample_min_z);
        constants.base_color[2] = static_cast<float>(draw.patch.sample_max_x);
        constants.base_color[3] = static_cast<float>(draw.patch.sample_max_z);
        if (write_motion)
        {
            const auto previous_mvp = math::matmul(draw.previous_view_projection, draw.terrain.previous_model);
            const auto* values = previous_mvp.data();
            std::copy(values, values + 4, constants.light_direction_intensity);
            std::copy(values + 4, values + 7, constants.light_color);
            constants.camera_position[0] = values[7];
            std::copy(values + 8, values + 11, constants.camera_position + 1);
            constants.fog_color_density[0] = values[11];
            std::copy(values + 12, values + 15, constants.fog_color_density + 1);
            constants.fog_params[0] = values[15];
        }
        const std::array descriptor_sets{material_descriptor_set_for(mesh_draw), terrain->second.descriptor_set};
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain_pipeline_layout_, 0u,
                                static_cast<std::uint32_t>(descriptor_sets.size()), descriptor_sets.data(), 0u,
                                nullptr);
        vkCmdPushConstants(command_buffer, terrain_pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0u, sizeof(constants),
                           &constants);
        vkCmdBindIndexBuffer(command_buffer, topology->second.indices.buffer, 0u, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, topology->second.index_count, 1u, 0u, 0, 0u);
    }

    VkDescriptorSet material_descriptor_set_for(const draw_mesh_event& draw) const noexcept
    {
        if (const auto material = materials_.find(resource_key(draw.material)); material != materials_.end())
        {
            const auto slot = current_frame_slot();
            if (slot < material->second.descriptor_sets.size() &&
                material->second.descriptor_sets[slot] != VK_NULL_HANDLE)
                return material->second.descriptor_sets[slot];
        }
        const auto slot = current_frame_slot();
        return slot < white_descriptor_sets_.size() ? white_descriptor_sets_[slot] : VK_NULL_HANDLE;
    }

    void destroy_mesh_pipeline() noexcept
    {
        const auto destroy_debug_pipeline = [&](VkPipeline& pipeline)
        {
            if (pipeline == VK_NULL_HANDLE) return;
            vkDestroyPipeline(device_, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        };
        destroy_debug_pipeline(debug_overlay_line_pipeline_);
        destroy_debug_pipeline(debug_overlay_triangle_pipeline_);
        destroy_debug_pipeline(debug_overlay_output_line_pipeline_);
        destroy_debug_pipeline(debug_overlay_output_triangle_pipeline_);
        if (debug_overlay_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, debug_overlay_pipeline_layout_, nullptr);
            debug_overlay_pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (deferred_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, deferred_pipeline_, nullptr);
            deferred_pipeline_ = VK_NULL_HANDLE;
        }
        if (deferred_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, deferred_pipeline_layout_, nullptr);
            deferred_pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (output_transform_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, output_transform_pipeline_, nullptr);
            output_transform_pipeline_ = VK_NULL_HANDLE;
        }
        if (output_transform_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, output_transform_pipeline_layout_, nullptr);
            output_transform_pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (luminance_histogram_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, luminance_histogram_pipeline_, nullptr);
            luminance_histogram_pipeline_ = VK_NULL_HANDLE;
        }
        if (luminance_histogram_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, luminance_histogram_pipeline_layout_, nullptr);
            luminance_histogram_pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (exposure_resolve_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, exposure_resolve_pipeline_, nullptr);
            exposure_resolve_pipeline_ = VK_NULL_HANDLE;
        }
        if (exposure_resolve_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, exposure_resolve_pipeline_layout_, nullptr);
            exposure_resolve_pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (output_transform_descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, output_transform_descriptor_pool_, nullptr);
            output_transform_descriptor_pool_ = VK_NULL_HANDLE;
            output_transform_descriptor_set_ = VK_NULL_HANDLE;
        }
        if (output_transform_descriptor_set_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, output_transform_descriptor_set_layout_, nullptr);
            output_transform_descriptor_set_layout_ = VK_NULL_HANDLE;
        }
        if (gbuffer_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, gbuffer_pipeline_, nullptr);
            gbuffer_pipeline_ = VK_NULL_HANDLE;
        }
        if (terrain_gbuffer_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, terrain_gbuffer_pipeline_, nullptr);
            terrain_gbuffer_pipeline_ = VK_NULL_HANDLE;
        }
        if (gbuffer_descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, gbuffer_descriptor_pool_, nullptr);
            gbuffer_descriptor_pool_ = VK_NULL_HANDLE;
            gbuffer_descriptor_set_ = VK_NULL_HANDLE;
        }
        if (gbuffer_sampler_ != VK_NULL_HANDLE)
        {
            vkDestroySampler(device_, gbuffer_sampler_, nullptr);
            gbuffer_sampler_ = VK_NULL_HANDLE;
        }
        if (gbuffer_descriptor_set_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, gbuffer_descriptor_set_layout_, nullptr);
            gbuffer_descriptor_set_layout_ = VK_NULL_HANDLE;
        }
        if (shadow_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, shadow_pipeline_, nullptr);
            shadow_pipeline_ = VK_NULL_HANDLE;
        }
        if (terrain_shadow_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, terrain_shadow_pipeline_, nullptr);
            terrain_shadow_pipeline_ = VK_NULL_HANDLE;
        }
        if (shadow_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, shadow_pipeline_layout_, nullptr);
            shadow_pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, mesh_wire_pipeline_, nullptr);
            mesh_wire_pipeline_ = VK_NULL_HANDLE;
        }
        if (mesh_transparent_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, mesh_transparent_pipeline_, nullptr);
            mesh_transparent_pipeline_ = VK_NULL_HANDLE;
        }
        if (terrain_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, terrain_pipeline_, nullptr);
            terrain_pipeline_ = VK_NULL_HANDLE;
        }
        if (mesh_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, mesh_pipeline_, nullptr);
            mesh_pipeline_ = VK_NULL_HANDLE;
        }
        if (mesh_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, mesh_pipeline_layout_, nullptr);
            mesh_pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (terrain_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, terrain_pipeline_layout_, nullptr);
            terrain_pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (sky_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, sky_pipeline_, nullptr);
            sky_pipeline_ = VK_NULL_HANDLE;
        }
        if (sky_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, sky_pipeline_layout_, nullptr);
            sky_pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    void destroy_white_texture() noexcept
    {
        for (auto& parameters : white_material_parameter_buffers_)
            destroy_buffer(parameters);
        white_material_parameter_buffers_.clear();
        if (white_descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, white_descriptor_pool_, nullptr);
            white_descriptor_pool_ = VK_NULL_HANDLE;
            white_descriptor_sets_.clear();
            sky_descriptor_sets_.clear();
            for (auto& [_, material] : materials_)
                material.descriptor_sets.clear();
        }
        if (white_descriptor_set_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, white_descriptor_set_layout_, nullptr);
            white_descriptor_set_layout_ = VK_NULL_HANDLE;
        }
        if (white_sampler_ != VK_NULL_HANDLE)
        {
            vkDestroySampler(device_, white_sampler_, nullptr);
            white_sampler_ = VK_NULL_HANDLE;
        }
        if (white_view_ != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, white_view_, nullptr);
            white_view_ = VK_NULL_HANDLE;
        }
        if (white_image_ != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator_, white_image_, white_allocation_);
            white_image_ = VK_NULL_HANDLE;
            white_allocation_ = VK_NULL_HANDLE;
        }
    }

    VkShaderModule create_shader_module(const std::uint32_t* code, std::size_t word_count)
    {
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = word_count * sizeof(std::uint32_t);
        info.pCode = code;

        VkShaderModule module{};
        if (vkCreateShaderModule(device_, &info, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
        return module;
    }

    void destroy_shadow_resources() noexcept
    {
        for (auto& view : shadow_atlas_.cascade_views)
        {
            if (view != VK_NULL_HANDLE)
            {
                vkDestroyImageView(device_, view, nullptr);
                view = VK_NULL_HANDLE;
            }
        }
        if (shadow_atlas_.array_view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, shadow_atlas_.array_view, nullptr);
            shadow_atlas_.array_view = VK_NULL_HANDLE;
        }
        if (shadow_atlas_.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device_, shadow_atlas_.sampler, nullptr);
            shadow_atlas_.sampler = VK_NULL_HANDLE;
        }
        if (shadow_atlas_.image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator_, shadow_atlas_.image, shadow_atlas_.allocation);
            shadow_atlas_.image = VK_NULL_HANDLE;
            shadow_atlas_.allocation = VK_NULL_HANDLE;
        }
        shadow_atlas_.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        shadow_atlas_.resolution = 0;
        shadow_cache_.static_layers_valid = false;
    }

    void destroy_local_shadow_resources() noexcept
    {
        if (local_shadow_atlas_.sampler != VK_NULL_HANDLE)
            vkDestroySampler(device_, local_shadow_atlas_.sampler, nullptr);
        if (local_shadow_atlas_.view != VK_NULL_HANDLE) vkDestroyImageView(device_, local_shadow_atlas_.view, nullptr);
        if (local_shadow_atlas_.image != VK_NULL_HANDLE)
            vmaDestroyImage(allocator_, local_shadow_atlas_.image, local_shadow_atlas_.allocation);
        local_shadow_atlas_ = {};
    }

    bool ensure_local_shadow_resources()
    {
        const std::uint32_t resolution =
            active_local_shadows_.empty() ? 1u : std::max(resolved_config_.local_shadow_atlas_resolution, 128u);
        if (local_shadow_atlas_.image != VK_NULL_HANDLE && local_shadow_atlas_.resolution == resolution) return true;

        wait_for_in_flight_frames();
        destroy_local_shadow_resources();
        for (auto& shadow : active_local_shadows_)
            shadow.redraw = true;

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = depth_format_;
        image.extent = {resolution, resolution, 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &local_shadow_atlas_.image, &local_shadow_atlas_.allocation,
                           nullptr) != VK_SUCCESS)
            return false;

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = local_shadow_atlas_.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = depth_format_;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device_, &view, nullptr, &local_shadow_atlas_.view) != VK_SUCCESS)
        {
            destroy_local_shadow_resources();
            return false;
        }

        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sampler.compareEnable = VK_TRUE;
        sampler.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        if (vkCreateSampler(device_, &sampler, nullptr, &local_shadow_atlas_.sampler) != VK_SUCCESS)
        {
            destroy_local_shadow_resources();
            return false;
        }
        local_shadow_atlas_.resolution = resolution;
        local_shadow_atlas_.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    }

    std::uint32_t frame_resource_count() const noexcept
    {
        return std::max(1u, swapchain_.image_count());
    }

    std::uint32_t current_frame_slot() const noexcept
    {
        return active_frame_index_ % frame_resource_count();
    }

    bool ensure_shadow_uniform_buffers()
    {
        const auto count = frame_resource_count();
        if (shadow_uniform_buffers_.size() == count)
        {
            bool ready = true;
            for (const auto& buffer : shadow_uniform_buffers_)
                ready = ready && buffer.buffer != VK_NULL_HANDLE;
            if (ready) return true;
        }

        wait_for_in_flight_frames();
        for (auto& buffer : shadow_uniform_buffers_)
            destroy_buffer(buffer);
        shadow_uniform_buffers_.clear();
        shadow_uniform_buffers_.resize(count);

        for (auto& buffer : shadow_uniform_buffers_)
        {
            if (!create_buffer(sizeof(shadow_uniform_data), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_TO_GPU, buffer))
                return false;
        }
        return true;
    }

    gpu_buffer* current_shadow_uniform_buffer() noexcept
    {
        const auto slot = current_frame_slot();
        if (slot >= shadow_uniform_buffers_.size()) return nullptr;
        return &shadow_uniform_buffers_[slot];
    }

    bool update_debug_overlay_buffer()
    {
        const auto count = frame_resource_count();
        if (debug_overlay_buffers_.size() != count)
        {
            wait_for_in_flight_frames();
            for (auto& buffer : debug_overlay_buffers_)
                destroy_buffer(buffer.vertices);
            debug_overlay_buffers_.clear();
            debug_overlay_buffers_.resize(count);
        }
        auto& target = debug_overlay_buffers_[current_frame_slot()];
        std::vector<debug_overlay_vertex> vertices;
        vertices.reserve(frame_debug_overlay_lines_.size() * 2u + frame_debug_overlay_triangles_.size() * 3u);
        const auto append_lines = [&](debug_overlay_depth_mode mode)
        {
            for (const auto& line : frame_debug_overlay_lines_)
            {
                if (line.depth != mode) continue;
                vertices.push_back({line.start, line.color});
                vertices.push_back({line.end, line.color});
            }
        };
        const auto append_triangles = [&](debug_overlay_depth_mode mode)
        {
            for (const auto& triangle : frame_debug_overlay_triangles_)
            {
                if (triangle.depth != mode) continue;
                vertices.push_back({triangle.first, triangle.color});
                vertices.push_back({triangle.second, triangle.color});
                vertices.push_back({triangle.third, triangle.color});
            }
        };
        const auto append_range =
            [&](auto&& append, debug_overlay_depth_mode mode, std::uint32_t& offset, std::uint32_t& count)
        {
            offset = static_cast<std::uint32_t>(vertices.size());
            append(mode);
            count = static_cast<std::uint32_t>(vertices.size()) - offset;
        };
        append_range(append_lines, debug_overlay_depth_mode::tested, target.tested_line_offset,
                     target.tested_line_count);
        append_range(append_triangles, debug_overlay_depth_mode::tested, target.tested_triangle_offset,
                     target.tested_triangle_count);
        append_range(append_lines, debug_overlay_depth_mode::always, target.output_line_offset,
                     target.output_line_count);
        append_range(append_triangles, debug_overlay_depth_mode::always, target.output_triangle_offset,
                     target.output_triangle_count);
        if (vertices.empty()) return true;
        const VkDeviceSize bytes = vertices.size() * sizeof(debug_overlay_vertex);
        if (target.capacity < bytes)
        {
            destroy_buffer(target.vertices);
            target.capacity = std::max<VkDeviceSize>(4096u, std::bit_ceil(static_cast<std::uint64_t>(bytes)));
            if (!create_buffer(target.capacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                               target.vertices))
            {
                target.capacity = 0;
                return false;
            }
        }
        void* mapped{};
        if (vmaMapMemory(allocator_, target.vertices.allocation, &mapped) != VK_SUCCESS) return false;
        std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(bytes));
        vmaFlushAllocation(allocator_, target.vertices.allocation, 0, bytes);
        vmaUnmapMemory(allocator_, target.vertices.allocation);
        return true;
    }

    const gpu_buffer* shadow_uniform_buffer_for_slot(std::uint32_t slot) const noexcept
    {
        if (slot >= shadow_uniform_buffers_.size()) return nullptr;
        return &shadow_uniform_buffers_[slot];
    }

    bool ensure_shadow_resources(const shadow_settings& settings)
    {
        const std::uint32_t resolution = std::clamp(settings.resolution, 256u, 8192u);
        if (shadow_atlas_.image != VK_NULL_HANDLE && shadow_atlas_.resolution == resolution) return true;

        wait_for_in_flight_frames();
        destroy_shadow_resources();

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = depth_format_;
        image.extent = {resolution, resolution, 1};
        image.mipLevels = 1;
        image.arrayLayers = directional_shadow_layer_count;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &shadow_atlas_.image, &shadow_atlas_.allocation, nullptr) !=
            VK_SUCCESS)
        {
            arc::diagnostics::warn("render.vulkan", "Failed to allocate directional shadow atlas");
            return false;
        }

        VkImageViewCreateInfo array_view{};
        array_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        array_view.image = shadow_atlas_.image;
        array_view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        array_view.format = depth_format_;
        array_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        array_view.subresourceRange.levelCount = 1;
        array_view.subresourceRange.layerCount = directional_shadow_layer_count;
        if (vkCreateImageView(device_, &array_view, nullptr, &shadow_atlas_.array_view) != VK_SUCCESS)
        {
            destroy_shadow_resources();
            return false;
        }

        for (std::uint32_t layer = 0; layer < directional_shadow_layer_count; ++layer)
        {
            VkImageViewCreateInfo layer_view = array_view;
            layer_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            layer_view.subresourceRange.baseArrayLayer = layer;
            layer_view.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device_, &layer_view, nullptr, &shadow_atlas_.cascade_views[layer]) != VK_SUCCESS)
            {
                destroy_shadow_resources();
                return false;
            }
        }

        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sampler.compareEnable = VK_TRUE;
        sampler.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        if (vkCreateSampler(device_, &sampler, nullptr, &shadow_atlas_.sampler) != VK_SUCCESS)
        {
            destroy_shadow_resources();
            return false;
        }

        shadow_atlas_.resolution = resolution;
        shadow_atlas_.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    }

    material_uniform_data build_material_parameters(const material_descriptor* material) const noexcept
    {
        material_uniform_data parameters{};
        if (material == nullptr || material->domain == material_domain::terrain) return parameters;

        parameters.emissive_factor[0] = material->emissive_factor[0];
        parameters.emissive_factor[1] = material->emissive_factor[1];
        parameters.emissive_factor[2] = material->emissive_factor[2];
        parameters.emissive_factor[3] = material->emissive_luminance_nits > 0.0f
                                            ? material->emissive_luminance_nits / 100.0f
                                            : material->emissive_strength;
        parameters.material_lobes[0] = material->clear_coat_factor;
        parameters.material_lobes[1] = material->clear_coat_roughness;
        parameters.material_lobes[2] = material->anisotropy_factor;
        parameters.material_lobes[3] = material->transmission_factor;
        parameters.volume_params[0] = static_cast<float>(material->shading_model);
        parameters.volume_params[1] = material->index_of_refraction;
        parameters.volume_params[2] = material->thickness_factor;
        parameters.volume_params[3] = material->attenuation_distance;
        parameters.subsurface_color_factor[0] = material->subsurface_color[0];
        parameters.subsurface_color_factor[1] = material->subsurface_color[1];
        parameters.subsurface_color_factor[2] = material->subsurface_color[2];
        parameters.subsurface_color_factor[3] = material->subsurface_factor;
        parameters.attenuation_color[0] = material->attenuation_color[0];
        parameters.attenuation_color[1] = material->attenuation_color[1];
        parameters.attenuation_color[2] = material->attenuation_color[2];
        parameters.attenuation_color[3] = (texture_ready(material->clear_coat_texture) ? 1.0f : 0.0f) +
                                          (texture_ready(material->clear_coat_roughness_texture) ? 2.0f : 0.0f) +
                                          (texture_ready(material->clear_coat_normal_texture) ? 4.0f : 0.0f) +
                                          (texture_ready(material->anisotropy_texture) ? 8.0f : 0.0f) +
                                          (texture_ready(material->subsurface_texture) ? 16.0f : 0.0f) +
                                          (texture_ready(material->thickness_texture) ? 32.0f : 0.0f) +
                                          (texture_ready(material->transmission_texture) ? 64.0f : 0.0f);
        return parameters;
    }

    bool update_material_parameter_buffer(gpu_buffer& buffer, const material_descriptor* material)
    {
        if (buffer.buffer == VK_NULL_HANDLE &&
            !create_buffer(sizeof(material_uniform_data), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                           VMA_MEMORY_USAGE_CPU_TO_GPU, buffer))
            return false;

        const auto parameters = build_material_parameters(material);
        void* mapped{};
        if (vmaMapMemory(allocator_, buffer.allocation, &mapped) != VK_SUCCESS) return false;
        std::memcpy(mapped, &parameters, sizeof(parameters));
        vmaFlushAllocation(allocator_, buffer.allocation, 0, sizeof(parameters));
        vmaUnmapMemory(allocator_, buffer.allocation);
        return true;
    }

    bool ensure_material_parameter_buffers(std::vector<gpu_buffer>& buffers, const material_descriptor* material)
    {
        const auto count = frame_resource_count();
        if (buffers.size() != count)
        {
            for (auto& buffer : buffers)
                destroy_buffer(buffer);
            buffers.assign(count, {});
        }
        for (auto& buffer : buffers)
        {
            if (buffer.buffer == VK_NULL_HANDLE && !update_material_parameter_buffer(buffer, material)) return false;
        }
        return true;
    }

    bool ensure_material_descriptor_sets(gpu_material& material)
    {
        if (!ensure_material_parameter_buffers(material.parameter_buffers, &material.data)) return false;
        const auto count = frame_resource_count();
        if (material.descriptor_sets.size() != count) material.descriptor_sets.assign(count, VK_NULL_HANDLE);
        for (auto& set : material.descriptor_sets)
        {
            if (set != VK_NULL_HANDLE) continue;
            set = allocate_material_descriptor_set();
            if (set == VK_NULL_HANDLE) return false;
        }
        return true;
    }

    bool ensure_white_descriptor_sets()
    {
        if (!ensure_material_parameter_buffers(white_material_parameter_buffers_, nullptr)) return false;
        const auto count = frame_resource_count();
        if (white_descriptor_sets_.size() != count) white_descriptor_sets_.assign(count, VK_NULL_HANDLE);
        for (auto& set : white_descriptor_sets_)
        {
            if (set != VK_NULL_HANDLE) continue;
            set = allocate_material_descriptor_set();
            if (set == VK_NULL_HANDLE) return false;
        }
        return true;
    }

    bool ensure_sky_descriptor_sets()
    {
        if (!ensure_white_texture()) return false;
        const auto count = frame_resource_count();
        if (sky_descriptor_sets_.size() != count) sky_descriptor_sets_.assign(count, VK_NULL_HANDLE);
        for (auto& set : sky_descriptor_sets_)
        {
            if (set == VK_NULL_HANDLE) set = allocate_material_descriptor_set();
            if (set == VK_NULL_HANDLE) return false;
        }
        return true;
    }

    VkDescriptorSet update_current_sky_descriptor_set()
    {
        if (!ensure_sky_descriptor_sets()) return VK_NULL_HANDLE;
        const auto slot = current_frame_slot();
        if (slot >= sky_descriptor_sets_.size()) return VK_NULL_HANDLE;
        const auto set = sky_descriptor_sets_[slot];
        update_material_descriptor_set(
            set, nullptr,
            slot < white_material_parameter_buffers_.size() ? &white_material_parameter_buffers_[slot] : nullptr, slot);

        VkSampler sampler = white_sampler_;
        VkImageView view = white_view_;
        if (frame_environment_.source == sky_source_mode::hdri && frame_environment_.hdri_texture.valid())
        {
            const auto found = textures_.find(resource_key(frame_environment_.hdri_texture));
            if (found != textures_.end() && found->second.view != VK_NULL_HANDLE &&
                found->second.sampler != VK_NULL_HANDLE)
            {
                sampler = found->second.sampler;
                view = found->second.view;
            }
            else
            {
                frame_environment_.fallback_reason = "HDRI texture is unavailable; using the visible fallback color";
            }
        }
        VkDescriptorImageInfo image{};
        image.sampler = sampler;
        image.imageView = view;
        image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        return set;
    }

    void update_material_descriptor_set(VkDescriptorSet descriptor_set, const material_descriptor* material,
                                        const gpu_buffer* material_parameters, std::uint32_t frame_slot)
    {
        const auto* shadow_buffer_resource = shadow_uniform_buffer_for_slot(frame_slot);
        if (descriptor_set == VK_NULL_HANDLE || white_view_ == VK_NULL_HANDLE ||
            shadow_atlas_.array_view == VK_NULL_HANDLE || local_shadow_atlas_.view == VK_NULL_HANDLE ||
            shadow_buffer_resource == nullptr || shadow_buffer_resource->buffer == VK_NULL_HANDLE ||
            material_parameters == nullptr || material_parameters->buffer == VK_NULL_HANDLE ||
            light_buffer_.buffer == VK_NULL_HANDLE)
            return;

        const auto resolve_texture = [&](texture_handle handle, VkSampler& sampler, VkImageView& view,
                                         std::optional<texture_semantic> expected = std::nullopt)
        {
            sampler = white_sampler_;
            view = white_view_;
            if (!handle.valid()) return;
            if (const auto found = textures_.find(resource_key(handle)); found != textures_.end())
            {
                if (expected && !valid_texture_color_space(*expected, found->second.data.color_space))
                {
                    const auto diagnostic_key = resource_key(handle) ^ (static_cast<std::uint64_t>(*expected) << 56u);
                    if (texture_semantic_diagnostics_.insert(diagnostic_key).second)
                    {
                        arc::diagnostics::warn("render.vulkan", "Texture '" + found->second.data.name +
                                                                    "' has a color space incompatible with its "
                                                                    "material slot; binding the explicit fallback");
                    }
                    return;
                }
                if (found->second.view != VK_NULL_HANDLE && found->second.sampler != VK_NULL_HANDLE)
                {
                    sampler = found->second.sampler;
                    view = found->second.view;
                }
            }
        };

        std::array<VkDescriptorImageInfo, material_image_bindings.size()> image_infos{};
        VkSampler sampler{};
        VkImageView view{};
        resolve_texture({}, sampler, view);
        for (auto& image : image_infos)
            image = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        if (material != nullptr && material->domain == material_domain::terrain)
        {
            for (std::size_t layer = 0; layer < material->terrain_layers.size(); ++layer)
            {
                resolve_texture(material->terrain_layers[layer].base_color_texture, sampler, view,
                                texture_semantic::base_color);
                image_infos[layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                resolve_texture(material->terrain_layers[layer].normal_texture, sampler, view,
                                texture_semantic::normal);
                image_infos[6u + layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                resolve_texture(material->terrain_layers[layer].packed_surface_texture, sampler, view,
                                texture_semantic::metallic_roughness);
                image_infos[10u + layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
        }
        else
        {
            resolve_texture(material ? material->base_color_texture : texture_handle{}, sampler, view,
                            texture_semantic::base_color);
            image_infos[0] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->metallic_roughness_texture : texture_handle{}, sampler, view,
                            texture_semantic::metallic_roughness);
            image_infos[1] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->normal_texture : texture_handle{}, sampler, view,
                            texture_semantic::normal);
            image_infos[2] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->occlusion_texture : texture_handle{}, sampler, view,
                            texture_semantic::occlusion);
            image_infos[3] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->emissive_texture : texture_handle{}, sampler, view,
                            texture_semantic::emissive);
            image_infos[4] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->clear_coat_texture : texture_handle{}, sampler, view,
                            texture_semantic::clear_coat);
            image_infos[6] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->clear_coat_roughness_texture : texture_handle{}, sampler, view,
                            texture_semantic::clear_coat);
            image_infos[7] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->clear_coat_normal_texture : texture_handle{}, sampler, view,
                            texture_semantic::normal);
            image_infos[8] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->anisotropy_texture : texture_handle{}, sampler, view,
                            texture_semantic::anisotropy);
            image_infos[9] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->subsurface_texture : texture_handle{}, sampler, view,
                            texture_semantic::thickness);
            image_infos[10] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->thickness_texture : texture_handle{}, sampler, view,
                            texture_semantic::thickness);
            image_infos[11] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material ? material->transmission_texture : texture_handle{}, sampler, view,
                            texture_semantic::transmission);
            image_infos[12] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }

        image_infos[5].sampler = shadow_atlas_.sampler;
        image_infos[5].imageView = shadow_atlas_.array_view;
        image_infos[5].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        image_infos.back().sampler = local_shadow_atlas_.sampler;
        image_infos.back().imageView = local_shadow_atlas_.view;
        image_infos.back().imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorBufferInfo shadow_buffer{};
        shadow_buffer.buffer = shadow_buffer_resource->buffer;
        shadow_buffer.offset = 0;
        shadow_buffer.range = sizeof(shadow_uniform_data);

        std::array<VkWriteDescriptorSet, material_image_bindings.size() + 3u> writes{};
        for (std::uint32_t image_index = 0; image_index < image_infos.size(); ++image_index)
        {
            writes[image_index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[image_index].dstSet = descriptor_set;
            writes[image_index].dstBinding = material_image_bindings[image_index];
            writes[image_index].descriptorCount = 1;
            writes[image_index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[image_index].pImageInfo = &image_infos[image_index];
        }
        auto& shadow_write = writes[material_image_bindings.size()];
        shadow_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadow_write.dstSet = descriptor_set;
        shadow_write.dstBinding = material_shadow_data_binding;
        shadow_write.descriptorCount = 1;
        shadow_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        shadow_write.pBufferInfo = &shadow_buffer;
        VkDescriptorBufferInfo light_buffer{};
        light_buffer.buffer = light_buffer_.buffer;
        light_buffer.range = sizeof(scene_lighting_data);
        auto& light_write = writes[material_image_bindings.size() + 1u];
        light_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        light_write.dstSet = descriptor_set;
        light_write.dstBinding = material_light_data_binding;
        light_write.descriptorCount = 1;
        light_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        light_write.pBufferInfo = &light_buffer;
        VkDescriptorBufferInfo parameter_buffer{};
        parameter_buffer.buffer = material_parameters->buffer;
        parameter_buffer.range = sizeof(material_uniform_data);
        auto& parameter_write = writes.back();
        parameter_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        parameter_write.dstSet = descriptor_set;
        parameter_write.dstBinding = material_parameters_binding;
        parameter_write.descriptorCount = 1;
        parameter_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        parameter_write.pBufferInfo = &parameter_buffer;
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void update_material_descriptor_sets(gpu_material& material)
    {
        if (!ensure_material_descriptor_sets(material)) return;
        for (std::uint32_t frame_slot = 0; frame_slot < material.descriptor_sets.size(); ++frame_slot)
        {
            if (!update_material_parameter_buffer(material.parameter_buffers[frame_slot], &material.data)) continue;
            update_material_descriptor_set(material.descriptor_sets[frame_slot], &material.data,
                                           &material.parameter_buffers[frame_slot], frame_slot);
        }
    }

    void update_white_descriptor_sets()
    {
        if (!ensure_white_descriptor_sets()) return;
        for (std::uint32_t frame_slot = 0; frame_slot < white_descriptor_sets_.size(); ++frame_slot)
            update_material_descriptor_set(white_descriptor_sets_[frame_slot], nullptr,
                                           &white_material_parameter_buffers_[frame_slot], frame_slot);
    }

    void update_all_material_descriptor_sets()
    {
        update_white_descriptor_sets();
        for (auto& [_, material] : materials_)
            update_material_descriptor_sets(material);
    }

    void update_current_material_descriptor_sets()
    {
        const auto frame_slot = current_frame_slot();
        if (ensure_white_descriptor_sets() && frame_slot < white_descriptor_sets_.size())
            update_material_descriptor_set(white_descriptor_sets_[frame_slot], nullptr,
                                           &white_material_parameter_buffers_[frame_slot], frame_slot);
        for (auto& [_, material] : materials_)
        {
            if (ensure_material_descriptor_sets(material) && frame_slot < material.descriptor_sets.size())
            {
                if (!update_material_parameter_buffer(material.parameter_buffers[frame_slot], &material.data)) continue;
                update_material_descriptor_set(material.descriptor_sets[frame_slot], &material.data,
                                               &material.parameter_buffers[frame_slot], frame_slot);
            }
        }
    }

    VkDescriptorSet allocate_material_descriptor_set()
    {
        if (white_descriptor_pool_ == VK_NULL_HANDLE || white_descriptor_set_layout_ == VK_NULL_HANDLE)
            return VK_NULL_HANDLE;

        VkDescriptorSet set{};
        VkDescriptorSetAllocateInfo descriptor_allocate{};
        descriptor_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptor_allocate.descriptorPool = white_descriptor_pool_;
        descriptor_allocate.descriptorSetCount = 1;
        descriptor_allocate.pSetLayouts = &white_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &descriptor_allocate, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
        return set;
    }

    bool ensure_white_texture()
    {
        const auto shadow_resolution = shadow_atlas_.resolution == 0 ? 2048u : shadow_atlas_.resolution;
        if (!ensure_shadow_uniform_buffers() ||
            !ensure_shadow_resources({.enabled = false, .resolution = shadow_resolution}) ||
            !ensure_local_shadow_resources())
            return false;

        if (white_descriptor_set_layout_ != VK_NULL_HANDLE && white_descriptor_pool_ != VK_NULL_HANDLE &&
            white_view_ != VK_NULL_HANDLE && white_sampler_ != VK_NULL_HANDLE)
        {
            return ensure_white_descriptor_sets();
        }

        std::array<VkDescriptorSetLayoutBinding, material_binding_count> bindings{};
        for (std::uint32_t binding_index = 0; binding_index < material_binding_count; ++binding_index)
        {
            bindings[binding_index].binding = binding_index;
            bindings[binding_index].descriptorType =
                binding_index == material_shadow_data_binding || binding_index == material_parameters_binding
                    ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                : binding_index == material_light_data_binding ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                               : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[binding_index].descriptorCount = 1;
            bindings[binding_index].stageFlags = binding_index == material_shadow_data_binding ||
                                                         binding_index == material_light_data_binding ||
                                                         binding_index == material_parameters_binding
                                                     ? VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                                     : VK_SHADER_STAGE_FRAGMENT_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &white_descriptor_set_layout_) != VK_SUCCESS)
            return false;

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = VK_FORMAT_R8G8B8A8_UNORM;
        image.extent = {1, 1, 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &white_image_, &white_allocation_, nullptr) != VK_SUCCESS)
            return false;

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = white_image_;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R8G8B8A8_UNORM;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device_, &view, nullptr, &white_view_) != VK_SUCCESS) return false;

        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &sampler, nullptr, &white_sampler_) != VK_SUCCESS) return false;

        const std::uint32_t white = 0xffffffffu;
        gpu_buffer staging;
        if (!create_buffer(sizeof(white), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, staging))
            return false;
        void* mapped{};
        vmaMapMemory(allocator_, staging.allocation, &mapped);
        std::memcpy(mapped, &white, sizeof(white));
        vmaUnmapMemory(allocator_, staging.allocation);

        VkCommandPool pool{};
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = graphics_queue_family_;
        vkCreateCommandPool(device_, &pool_info, nullptr, &pool);
        VkCommandBuffer command_buffer{};
        VkCommandBufferAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool = pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &allocate, &command_buffer);
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin);

        VkImageMemoryBarrier to_copy{};
        to_copy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_copy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.image = white_image_;
        to_copy.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_copy.subresourceRange.levelCount = 1;
        to_copy.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &to_copy);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {1, 1, 1};
        vkCmdCopyBufferToImage(command_buffer, staging.buffer, white_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copy);

        VkImageMemoryBarrier to_shader = to_copy;
        to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &to_shader);
        vkEndCommandBuffer(command_buffer);

        submit_upload_commands(command_buffer);
        vkDestroyCommandPool(device_, pool, nullptr);
        destroy_buffer(staging);

        std::array<VkDescriptorPoolSize, 3> pool_sizes{};
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_sizes[0].descriptorCount =
            static_cast<std::uint32_t>(material_image_bindings.size()) * material_descriptor_set_capacity;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_sizes[1].descriptorCount = material_descriptor_set_capacity * 2u;
        pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[2].descriptorCount = material_descriptor_set_capacity;
        VkDescriptorPoolCreateInfo descriptor_pool{};
        descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptor_pool.maxSets = material_descriptor_set_capacity;
        descriptor_pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        descriptor_pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &descriptor_pool, nullptr, &white_descriptor_pool_) != VK_SUCCESS)
            return false;

        update_all_material_descriptor_sets();
        return true;
    }

    bool ensure_mesh_pipeline()
    {
        if (mesh_pipeline_ != VK_NULL_HANDLE) return true;
        if (max_push_constant_bytes_ < sizeof(mesh_push_constants))
        {
            if (!push_constant_limit_warning_reported_)
            {
                arc::diagnostics::error(
                    "render.vulkan", "The selected adapter exposes only " + std::to_string(max_push_constant_bytes_) +
                                         " push-constant bytes; ARC's raster mesh path currently requires " +
                                         std::to_string(sizeof(mesh_push_constants)));
                push_constant_limit_warning_reported_ = true;
            }
            return false;
        }
        if (!ensure_white_texture() || !ensure_terrain_descriptors()) return false;

        VkShaderModule vert =
            create_shader_module(builtin::default_phong_vert_spv, std::size(builtin::default_phong_vert_spv));
        VkShaderModule frag =
            create_shader_module(builtin::default_phong_frag_spv, std::size(builtin::default_phong_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(mesh_push_constants);

        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &white_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &mesh_pipeline_layout_) != VK_SUCCESS) return false;
        const std::array terrain_set_layouts{white_descriptor_set_layout_, terrain_descriptor_set_layout_};
        layout.setLayoutCount = static_cast<std::uint32_t>(terrain_set_layouts.size());
        layout.pSetLayouts = terrain_set_layouts.data();
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &terrain_pipeline_layout_) != VK_SUCCESS) return false;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(mesh_vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 5> attributes{};
        attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)};
        attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)};
        attributes[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)};
        attributes[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)};
        attributes[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)};

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding;
        vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
        vertex_input.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState color_attachment{};
        color_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = 1;
        color_blend.pAttachments = &color_attachment;

        const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &scene_color_format_;
        rendering.depthAttachmentFormat = depth_format_;

        VkGraphicsPipelineCreateInfo pipeline{};
        pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline.pNext = &rendering;
        pipeline.stageCount = 2;
        pipeline.pStages = stages;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input_assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &depth;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = mesh_pipeline_layout_;
        pipeline.renderPass = VK_NULL_HANDLE;

        const VkResult result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_pipeline_);
        if (result == VK_SUCCESS)
        {
            depth.depthWriteEnable = VK_FALSE;
            color_attachment.blendEnable = VK_TRUE;
            color_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            color_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            color_attachment.colorBlendOp = VK_BLEND_OP_ADD;
            color_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            color_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            color_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
            const VkResult blend_result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr,
                                                                    &mesh_transparent_pipeline_);
            if (blend_result != VK_SUCCESS)
                arc::diagnostics::warn(
                    "render.vulkan",
                    "Vulkan transparent mesh pipeline creation failed; blended materials will render opaque");
            color_attachment = {};
            color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            depth.depthWriteEnable = VK_TRUE;
        }
        if (result == VK_SUCCESS && capabilities_.fill_mode_non_solid)
        {
            raster.polygonMode = VK_POLYGON_MODE_LINE;
            depth.depthWriteEnable = VK_FALSE;
            const VkResult wire_result =
                vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_wire_pipeline_);
            if (wire_result != VK_SUCCESS)
                arc::diagnostics::warn("render.vulkan",
                                       "Vulkan wireframe pipeline creation failed; shaded rendering will continue");
        }
        else if (result == VK_SUCCESS && !capabilities_.fill_mode_non_solid && !wireframe_warning_reported_)
        {
            arc::diagnostics::warn("render.vulkan",
                                   "Vulkan device does not support fillModeNonSolid; wireframe rendering is disabled");
            wireframe_warning_reported_ = true;
        }

        if (result == VK_SUCCESS)
        {
            VkShaderModule terrain_vert = create_shader_module(builtin::terrain_patch_forward_vert_spv,
                                                               std::size(builtin::terrain_patch_forward_vert_spv));
            VkShaderModule terrain_frag =
                create_shader_module(builtin::terrain_forward_frag_spv, std::size(builtin::terrain_forward_frag_spv));
            if (terrain_vert != VK_NULL_HANDLE && terrain_frag != VK_NULL_HANDLE)
            {
                stages[0].module = terrain_vert;
                stages[1].module = terrain_frag;
                VkPipelineVertexInputStateCreateInfo terrain_vertex_input{
                    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
                pipeline.pVertexInputState = &terrain_vertex_input;
                pipeline.layout = terrain_pipeline_layout_;
                raster.polygonMode = VK_POLYGON_MODE_FILL;
                depth.depthWriteEnable = VK_TRUE;
                color_attachment = {};
                color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                if (vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &terrain_pipeline_) !=
                    VK_SUCCESS)
                {
                    terrain_pipeline_ = VK_NULL_HANDLE;
                    arc::diagnostics::warn(
                        "render.vulkan", "Vulkan terrain forward pipeline creation failed; using the surface fallback");
                }
                vkDestroyShaderModule(device_, terrain_frag, nullptr);
            }
            else
            {
                arc::diagnostics::warn("render.vulkan",
                                       "Vulkan terrain shader module creation failed; using the surface fallback");
            }
            if (terrain_vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_vert, nullptr);
        }
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return result == VK_SUCCESS;
    }

    bool ensure_debug_overlay_pipeline()
    {
        if (debug_overlay_line_pipeline_ != VK_NULL_HANDLE && debug_overlay_triangle_pipeline_ != VK_NULL_HANDLE &&
            debug_overlay_output_line_pipeline_ != VK_NULL_HANDLE &&
            debug_overlay_output_triangle_pipeline_ != VK_NULL_HANDLE)
            return true;
        VkShaderModule vert =
            create_shader_module(builtin::debug_overlay_vert_spv, std::size(builtin::debug_overlay_vert_spv));
        VkShaderModule frag =
            create_shader_module(builtin::debug_overlay_frag_spv, std::size(builtin::debug_overlay_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
        {
            if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
            if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
            return false;
        }

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push.size = sizeof(float) * 16u;
        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &debug_overlay_pipeline_layout_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
            return false;
        }

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_VERTEX_BIT,
                     vert,
                     "main",
                     nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_FRAGMENT_BIT,
                     frag,
                     "main",
                     nullptr};
        VkVertexInputBindingDescription binding{0, sizeof(debug_overlay_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        const std::array<VkVertexInputAttributeDescription, 2> attributes{
            VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                              offsetof(debug_overlay_vertex, position)},
            VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
                                              offsetof(debug_overlay_vertex, color)}};
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding;
        vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
        vertex_input.pVertexAttributeDescriptions = attributes.data();
        VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_FALSE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        blend.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        color_blend.attachmentCount = 1;
        color_blend.pAttachments = &blend;
        const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        const auto create_pipeline =
            [&](VkPrimitiveTopology topology, VkFormat color_format, bool depth_test, VkPipeline& destination)
        {
            input.topology = topology;
            depth.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &color_format;
            rendering.depthAttachmentFormat = depth_test ? depth_format_ : VK_FORMAT_UNDEFINED;
            VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipeline.pNext = &rendering;
            pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
            pipeline.pStages = stages.data();
            pipeline.pVertexInputState = &vertex_input;
            pipeline.pInputAssemblyState = &input;
            pipeline.pViewportState = &viewport;
            pipeline.pRasterizationState = &raster;
            pipeline.pMultisampleState = &multisample;
            pipeline.pDepthStencilState = &depth;
            pipeline.pColorBlendState = &color_blend;
            pipeline.pDynamicState = &dynamic;
            pipeline.layout = debug_overlay_pipeline_layout_;
            return vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &destination);
        };

        const auto tested_line_result =
            create_pipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, scene_color_format_, true, debug_overlay_line_pipeline_);
        const auto tested_triangle_result =
            tested_line_result == VK_SUCCESS ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, scene_color_format_,
                                                               true, debug_overlay_triangle_pipeline_)
                                             : VK_ERROR_INITIALIZATION_FAILED;
        const auto output_line_result = tested_triangle_result == VK_SUCCESS
                                            ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, viewport_format_, false,
                                                              debug_overlay_output_line_pipeline_)
                                            : VK_ERROR_INITIALIZATION_FAILED;
        const auto output_triangle_result = output_line_result == VK_SUCCESS
                                                ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, viewport_format_,
                                                                  false, debug_overlay_output_triangle_pipeline_)
                                                : VK_ERROR_INITIALIZATION_FAILED;
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        if (tested_line_result != VK_SUCCESS || tested_triangle_result != VK_SUCCESS ||
            output_line_result != VK_SUCCESS || output_triangle_result != VK_SUCCESS)
        {
            arc::diagnostics::warn("render.vulkan", "Vulkan debug-overlay pipeline creation failed");
            const auto destroy_pipeline = [&](VkPipeline& pipeline)
            {
                if (pipeline == VK_NULL_HANDLE) return;
                vkDestroyPipeline(device_, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            };
            destroy_pipeline(debug_overlay_line_pipeline_);
            destroy_pipeline(debug_overlay_triangle_pipeline_);
            destroy_pipeline(debug_overlay_output_line_pipeline_);
            destroy_pipeline(debug_overlay_output_triangle_pipeline_);
            vkDestroyPipelineLayout(device_, debug_overlay_pipeline_layout_, nullptr);
            debug_overlay_pipeline_layout_ = VK_NULL_HANDLE;
        }
        return tested_line_result == VK_SUCCESS && tested_triangle_result == VK_SUCCESS &&
               output_line_result == VK_SUCCESS && output_triangle_result == VK_SUCCESS;
    }

    bool ensure_gbuffer_pipeline()
    {
        if (gbuffer_pipeline_ != VK_NULL_HANDLE) return true;
        if (!ensure_mesh_pipeline()) return false;

        VkShaderModule vert = create_shader_module(builtin::gbuffer_vert_spv, std::size(builtin::gbuffer_vert_spv));
        VkShaderModule frag = create_shader_module(builtin::gbuffer_frag_spv, std::size(builtin::gbuffer_frag_spv));
        VkShaderModule terrain_vert = create_shader_module(builtin::terrain_patch_gbuffer_vert_spv,
                                                           std::size(builtin::terrain_patch_gbuffer_vert_spv));
        VkShaderModule terrain_frag =
            create_shader_module(builtin::terrain_gbuffer_frag_spv, std::size(builtin::terrain_gbuffer_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
        {
            if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
            if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
            if (terrain_frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_frag, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(mesh_vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 5> attributes{};
        attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)};
        attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)};
        attributes[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)};
        attributes[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)};
        attributes[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)};

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding;
        vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
        vertex_input.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_FALSE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        std::array<VkPipelineColorBlendAttachmentState, 6> attachments{};
        for (auto& attachment : attachments)
        {
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                        VK_COLOR_COMPONENT_A_BIT;
        }
        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        color_blend.pAttachments = attachments.data();

        const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        const std::array<VkFormat, 6> color_formats{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_FORMAT_R16G16_SFLOAT,       VK_FORMAT_R32_UINT};
        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size());
        rendering.pColorAttachmentFormats = color_formats.data();
        rendering.depthAttachmentFormat = depth_format_;

        VkGraphicsPipelineCreateInfo pipeline{};
        pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline.pNext = &rendering;
        pipeline.stageCount = 2;
        pipeline.pStages = stages;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input_assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &depth;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = mesh_pipeline_layout_;
        pipeline.renderPass = VK_NULL_HANDLE;

        const VkResult result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &gbuffer_pipeline_);
        if (result == VK_SUCCESS && terrain_vert != VK_NULL_HANDLE && terrain_frag != VK_NULL_HANDLE)
        {
            VkPipelineVertexInputStateCreateInfo terrain_vertex_input{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            stages[0].module = terrain_vert;
            stages[1].module = terrain_frag;
            pipeline.pVertexInputState = &terrain_vertex_input;
            pipeline.layout = terrain_pipeline_layout_;
            if (vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr,
                                          &terrain_gbuffer_pipeline_) != VK_SUCCESS)
            {
                terrain_gbuffer_pipeline_ = VK_NULL_HANDLE;
                arc::diagnostics::warn("render.vulkan",
                                       "Vulkan terrain G-buffer pipeline creation failed; using the surface fallback");
            }
        }
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        if (terrain_vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_vert, nullptr);
        if (terrain_frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_frag, nullptr);
        if (result != VK_SUCCESS)
            arc::diagnostics::warn("render.vulkan",
                                   "Vulkan G-buffer pipeline creation failed; falling back to forward rendering");
        return result == VK_SUCCESS;
    }

    bool ensure_gbuffer_descriptor_set()
    {
        if (gbuffer_descriptor_set_ != VK_NULL_HANDLE) return true;
        if (!ensure_white_texture() || gbuffer_albedo_.view == VK_NULL_HANDLE ||
            gbuffer_normal_.view == VK_NULL_HANDLE || gbuffer_material_.view == VK_NULL_HANDLE ||
            gbuffer_emissive_.view == VK_NULL_HANDLE || gbuffer_motion_.view == VK_NULL_HANDLE ||
            gbuffer_object_id_.view == VK_NULL_HANDLE || viewport_depth_view_ == VK_NULL_HANDLE ||
            shadow_atlas_.array_view == VK_NULL_HANDLE || shadow_atlas_.sampler == VK_NULL_HANDLE ||
            local_shadow_atlas_.view == VK_NULL_HANDLE || local_shadow_atlas_.sampler == VK_NULL_HANDLE ||
            current_shadow_uniform_buffer() == nullptr)
            return false;

        if (gbuffer_sampler_ == VK_NULL_HANDLE)
        {
            VkSamplerCreateInfo sampler{};
            sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler.magFilter = VK_FILTER_NEAREST;
            sampler.minFilter = VK_FILTER_NEAREST;
            sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (vkCreateSampler(device_, &sampler, nullptr, &gbuffer_sampler_) != VK_SUCCESS) return false;
        }

        if (gbuffer_descriptor_set_layout_ == VK_NULL_HANDLE)
        {
            std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
            for (std::uint32_t index = 0; index < 7; ++index)
            {
                bindings[index].binding = index;
                bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bindings[index].descriptorCount = 1;
                bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            bindings[7].binding = 7;
            bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[7].descriptorCount = 1;
            bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[8].binding = 8;
            bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[8].descriptorCount = 1;
            bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[9].binding = 9;
            bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[9].descriptorCount = 1;
            bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[10].binding = 10;
            bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[10].descriptorCount = 1;
            bindings[10].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[11].binding = 11;
            bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[11].descriptorCount = 1;
            bindings[11].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo layout{};
            layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
            layout.pBindings = bindings.data();
            if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &gbuffer_descriptor_set_layout_) != VK_SUCCESS)
                return false;
        }

        if (gbuffer_descriptor_pool_ == VK_NULL_HANDLE)
        {
            std::array<VkDescriptorPoolSize, 3> pool_sizes{
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
            VkDescriptorPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool.maxSets = 1;
            pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
            pool.pPoolSizes = pool_sizes.data();
            if (vkCreateDescriptorPool(device_, &pool, nullptr, &gbuffer_descriptor_pool_) != VK_SUCCESS) return false;
        }

        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = gbuffer_descriptor_pool_;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &gbuffer_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &gbuffer_descriptor_set_) != VK_SUCCESS) return false;

        update_gbuffer_descriptor_set();
        return true;
    }

    void update_gbuffer_descriptor_set()
    {
        if (gbuffer_descriptor_set_ == VK_NULL_HANDLE || light_buffer_.buffer == VK_NULL_HANDLE) return;

        std::array<VkDescriptorImageInfo, 10> images{};
        const VkSampler sampler = gbuffer_sampler_ != VK_NULL_HANDLE ? gbuffer_sampler_ : white_sampler_;
        images[0] = {sampler, gbuffer_albedo_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[1] = {sampler, gbuffer_normal_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[2] = {sampler, gbuffer_material_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[3] = {sampler, gbuffer_emissive_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[4] = {sampler, gbuffer_object_id_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[5] = {sampler, gbuffer_motion_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[6] = {sampler, viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[7] = {white_sampler_, white_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[8] = {shadow_atlas_.sampler, shadow_atlas_.array_view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        images[9] = {local_shadow_atlas_.sampler, local_shadow_atlas_.view,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        if (const auto* environment = active_environment())
        {
            if (const auto found = textures_.find(resource_key(environment->equirectangular_texture));
                found != textures_.end() && found->second.view != VK_NULL_HANDLE)
            {
                images[7] = {found->second.sampler, found->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
        }

        std::array<VkWriteDescriptorSet, 10> writes{};
        for (std::uint32_t index = 0; index < 7; ++index)
        {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = gbuffer_descriptor_set_;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[index].pImageInfo = &images[index];
        }
        writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[7].dstSet = gbuffer_descriptor_set_;
        writes[7].dstBinding = 8;
        writes[7].descriptorCount = 1;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[7].pImageInfo = &images[7];
        writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[8].dstSet = gbuffer_descriptor_set_;
        writes[8].dstBinding = 9;
        writes[8].descriptorCount = 1;
        writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[8].pImageInfo = &images[8];
        writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[9].dstSet = gbuffer_descriptor_set_;
        writes[9].dstBinding = 11;
        writes[9].descriptorCount = 1;
        writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[9].pImageInfo = &images[9];
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        VkDescriptorBufferInfo lights{light_buffer_.buffer, 0, sizeof(scene_lighting_data)};
        VkWriteDescriptorSet light_write{};
        light_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        light_write.dstSet = gbuffer_descriptor_set_;
        light_write.dstBinding = 7;
        light_write.descriptorCount = 1;
        light_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        light_write.pBufferInfo = &lights;
        const auto* shadow_buffer = current_shadow_uniform_buffer();
        if (shadow_buffer == nullptr) return;
        VkDescriptorBufferInfo shadow{shadow_buffer->buffer, 0, sizeof(shadow_uniform_data)};
        VkWriteDescriptorSet shadow_write{};
        shadow_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadow_write.dstSet = gbuffer_descriptor_set_;
        shadow_write.dstBinding = 10;
        shadow_write.descriptorCount = 1;
        shadow_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        shadow_write.pBufferInfo = &shadow;
        const std::array buffer_writes{light_write, shadow_write};
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(buffer_writes.size()), buffer_writes.data(), 0,
                               nullptr);
    }

    bool ensure_deferred_pipeline()
    {
        if (deferred_pipeline_ != VK_NULL_HANDLE) return true;
        if (!ensure_gbuffer_descriptor_set()) return false;

        VkShaderModule vert =
            create_shader_module(builtin::deferred_lighting_vert_spv, std::size(builtin::deferred_lighting_vert_spv));
        VkShaderModule frag =
            create_shader_module(builtin::deferred_lighting_frag_spv, std::size(builtin::deferred_lighting_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(deferred_push_constants);

        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &gbuffer_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &deferred_pipeline_layout_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState color_attachment{};
        color_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = 1;
        color_blend.pAttachments = &color_attachment;
        const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &scene_color_format_;

        VkGraphicsPipelineCreateInfo pipeline{};
        pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline.pNext = &rendering;
        pipeline.stageCount = 2;
        pipeline.pStages = stages;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input_assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = deferred_pipeline_layout_;
        pipeline.renderPass = VK_NULL_HANDLE;

        const VkResult result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &deferred_pipeline_);
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        if (result != VK_SUCCESS)
            arc::diagnostics::warn(
                "render.vulkan",
                "Vulkan deferred lighting pipeline creation failed; falling back to forward rendering");
        return result == VK_SUCCESS;
    }

    bool ensure_output_transform_pipeline()
    {
        if (exposure_buffer_.buffer == VK_NULL_HANDLE &&
            !create_buffer(exposure_buffer_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_GPU_ONLY, exposure_buffer_))
            return false;

        if (output_transform_pipeline_ != VK_NULL_HANDLE)
        {
            VkDescriptorImageInfo image{
                viewport_sampler_, temporal_output_view_ != VK_NULL_HANDLE ? temporal_output_view_ : scene_color_.view,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = output_transform_descriptor_set_;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &image;
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
            return true;
        }
        if (scene_color_.view == VK_NULL_HANDLE || viewport_sampler_ == VK_NULL_HANDLE) return false;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo descriptor_layout{};
        descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        descriptor_layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr,
                                        &output_transform_descriptor_set_layout_) != VK_SUCCESS)
            return false;

        std::array<VkDescriptorPoolSize, 2> pool_sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 1;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &output_transform_descriptor_pool_) != VK_SUCCESS)
            return false;
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = output_transform_descriptor_pool_;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &output_transform_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &output_transform_descriptor_set_) != VK_SUCCESS) return false;

        VkDescriptorImageInfo image{viewport_sampler_,
                                    temporal_output_view_ != VK_NULL_HANDLE ? temporal_output_view_ : scene_color_.view,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorBufferInfo exposure_buffer_info{exposure_buffer_.buffer, 0, exposure_buffer_bytes};
        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = output_transform_descriptor_set_;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &image;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = output_transform_descriptor_set_;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &exposure_buffer_info;
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

        const auto vert =
            create_shader_module(builtin::deferred_lighting_vert_spv, std::size(builtin::deferred_lighting_vert_spv));
        const auto frag =
            create_shader_module(builtin::output_transform_frag_spv, std::size(builtin::output_transform_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push.size = sizeof(output_transform_push_constants);
        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &output_transform_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &output_transform_pipeline_layout_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
            return false;
        }

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo input_assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState color_attachment{};
        color_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        color_blend.attachmentCount = 1;
        color_blend.pAttachments = &color_attachment;
        const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &viewport_format_;
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline.pNext = &rendering;
        pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
        pipeline.pStages = stages.data();
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input_assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = output_transform_pipeline_layout_;
        const auto result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &output_transform_pipeline_);
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        if (result != VK_SUCCESS)
            arc::diagnostics::warn("render.vulkan", "Vulkan output-transform pipeline creation failed");
        return result == VK_SUCCESS;
    }

    bool ensure_exposure_pipelines()
    {
        if (luminance_histogram_pipeline_ != VK_NULL_HANDLE && exposure_resolve_pipeline_ != VK_NULL_HANDLE)
            return true;
        if (!ensure_output_transform_pipeline()) return false;

        const auto create_compute = [&](const std::uint32_t* words, std::size_t word_count, std::uint32_t push_size,
                                        VkPipelineLayout& pipeline_layout, VkPipeline& pipeline)
        {
            const auto shader = create_shader_module(words, word_count);
            if (shader == VK_NULL_HANDLE) return false;
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.size = push_size;
            VkPipelineLayoutCreateInfo layout{};
            layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout.setLayoutCount = 1;
            layout.pSetLayouts = &output_transform_descriptor_set_layout_;
            layout.pushConstantRangeCount = 1;
            layout.pPushConstantRanges = &push;
            if (vkCreatePipelineLayout(device_, &layout, nullptr, &pipeline_layout) != VK_SUCCESS)
            {
                vkDestroyShaderModule(device_, shader, nullptr);
                return false;
            }
            VkComputePipelineCreateInfo create{};
            create.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            create.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            create.stage.module = shader;
            create.stage.pName = "main";
            create.layout = pipeline_layout;
            const auto result = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &create, nullptr, &pipeline);
            vkDestroyShaderModule(device_, shader, nullptr);
            return result == VK_SUCCESS;
        };

        if (!create_compute(builtin::luminance_histogram_comp_spv, std::size(builtin::luminance_histogram_comp_spv),
                            sizeof(histogram_push_constants), luminance_histogram_pipeline_layout_,
                            luminance_histogram_pipeline_))
            return false;
        if (!create_compute(builtin::exposure_resolve_comp_spv, std::size(builtin::exposure_resolve_comp_spv),
                            sizeof(exposure_resolve_push_constants), exposure_resolve_pipeline_layout_,
                            exposure_resolve_pipeline_))
            return false;
        return true;
    }

    void dispatch_exposure(VkCommandBuffer command_buffer)
    {
        if (!ensure_exposure_pipelines()) return;

        // The exposure state is persistent and shared between frames.
        // Serialize the transfer clear after the previous frame's resolve and
        // output-transform read before reusing the buffer.
        VkBufferMemoryBarrier reuse_barrier{};
        reuse_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        reuse_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        reuse_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        reuse_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        reuse_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        reuse_barrier.buffer = exposure_buffer_.buffer;
        reuse_barrier.size = exposure_buffer_bytes;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &reuse_barrier, 0, nullptr);

        const VkDeviceSize clear_size = exposure_needs_reset_ ? exposure_buffer_bytes : exposure_histogram_bytes;
        vkCmdFillBuffer(command_buffer, exposure_buffer_.buffer, 0, clear_size, 0u);
        VkBufferMemoryBarrier clear_barrier{};
        clear_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        clear_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        clear_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        clear_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clear_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clear_barrier.buffer = exposure_buffer_.buffer;
        clear_barrier.size = exposure_buffer_bytes;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 1, &clear_barrier, 0, nullptr);

        histogram_push_constants histogram{};
        histogram.log_luminance_extent[2] = static_cast<float>(viewport_width_);
        histogram.log_luminance_extent[3] = static_cast<float>(viewport_height_);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_histogram_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_histogram_pipeline_layout_, 0,
                                1, &output_transform_descriptor_set_, 0, nullptr);
        vkCmdPushConstants(command_buffer, luminance_histogram_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(histogram), &histogram);
        const std::uint32_t sample_width = (viewport_width_ + 3u) / 4u;
        const std::uint32_t sample_height = (viewport_height_ + 3u) / 4u;
        vkCmdDispatch(command_buffer, (sample_width + 15u) / 16u, (sample_height + 15u) / 16u, 1u);

        VkBufferMemoryBarrier histogram_barrier = clear_barrier;
        histogram_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        histogram_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &histogram_barrier, 0, nullptr);

        exposure_resolve_push_constants resolve{};
        resolve.limits_speeds[0] = frame_camera_.exposure.minimum_ev100;
        resolve.limits_speeds[1] = frame_camera_.exposure.maximum_ev100;
        resolve.limits_speeds[2] = frame_camera_.exposure.brighten_speed;
        resolve.limits_speeds[3] = frame_camera_.exposure.darken_speed;
        resolve.timing_mode[1] = frame_camera_.exposure.mode == exposure_mode::automatic ? 1.0f : 0.0f;
        resolve.timing_mode[2] = frame_camera_.exposure.manual_ev100;
        resolve.timing_mode[3] = exposure_needs_reset_ ? 1.0f : 0.0f;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_resolve_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_resolve_pipeline_layout_, 0, 1,
                                &output_transform_descriptor_set_, 0, nullptr);
        vkCmdPushConstants(command_buffer, exposure_resolve_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(resolve), &resolve);
        vkCmdDispatch(command_buffer, 1u, 1u, 1u);

        VkBufferMemoryBarrier output_barrier = histogram_barrier;
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 1, &output_barrier, 0, nullptr);
        exposure_needs_reset_ = false;
    }

    bool ensure_sky_pipeline()
    {
        if (sky_pipeline_ != VK_NULL_HANDLE) return true;
        if (!ensure_white_texture()) return false;

        VkShaderModule vert =
            create_shader_module(builtin::sky_atmosphere_vert_spv, std::size(builtin::sky_atmosphere_vert_spv));
        VkShaderModule frag =
            create_shader_module(builtin::sky_atmosphere_frag_spv, std::size(builtin::sky_atmosphere_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(detail::sky_push_constants);

        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &white_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &sky_pipeline_layout_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = VK_FALSE;
        depth.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState color_attachment{};
        color_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = 1;
        color_blend.pAttachments = &color_attachment;

        const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &scene_color_format_;
        rendering.depthAttachmentFormat = depth_format_;

        VkGraphicsPipelineCreateInfo pipeline{};
        pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline.pNext = &rendering;
        pipeline.stageCount = 2;
        pipeline.pStages = stages;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input_assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &depth;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = sky_pipeline_layout_;
        pipeline.renderPass = VK_NULL_HANDLE;

        const VkResult result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &sky_pipeline_);
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return result == VK_SUCCESS;
    }

    void destroy_graph_image(graph_image& image) noexcept
    {
        for (const auto view : image.mip_views)
            if (view != VK_NULL_HANDLE) vkDestroyImageView(device_, view, nullptr);
        image.mip_views.clear();
        if (image.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, image.view, nullptr);
            image.view = VK_NULL_HANDLE;
        }
        if (image.image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator_, image.image, image.allocation);
            image.image = VK_NULL_HANDLE;
            image.allocation = VK_NULL_HANDLE;
        }
        image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        image.width = 0;
        image.height = 0;
        image.mip_levels = 1;
    }

    bool ensure_graph_image(graph_image& target, std::uint32_t width, std::uint32_t height, VkFormat format,
                            VkImageUsageFlags usage, VkImageAspectFlags aspect, std::uint32_t mip_levels = 1)
    {
        if (target.image != VK_NULL_HANDLE && target.format == format && target.aspect == aspect &&
            target.width == width && target.height == height && target.mip_levels == mip_levels)
            return true;

        destroy_graph_image(target);

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = format;
        image.extent = {width, height, 1};
        image.mipLevels = mip_levels;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = usage;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &target.image, &target.allocation, nullptr) != VK_SUCCESS)
            return false;

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = target.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange.aspectMask = aspect;
        view.subresourceRange.levelCount = mip_levels;
        view.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device_, &view, nullptr, &target.view) != VK_SUCCESS)
        {
            destroy_graph_image(target);
            return false;
        }

        if (mip_levels > 1)
        {
            target.mip_views.resize(mip_levels);
            view.subresourceRange.levelCount = 1;
            for (std::uint32_t mip = 0; mip < mip_levels; ++mip)
            {
                view.subresourceRange.baseMipLevel = mip;
                if (vkCreateImageView(device_, &view, nullptr, &target.mip_views[mip]) != VK_SUCCESS)
                {
                    destroy_graph_image(target);
                    return false;
                }
            }
        }

        target.format = format;
        target.aspect = aspect;
        target.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        target.width = width;
        target.height = height;
        target.mip_levels = mip_levels;
        return true;
    }

    void transition_graph_image(VkCommandBuffer command_buffer, graph_image& image, VkImageLayout new_layout)
    {
        if (image.image == VK_NULL_HANDLE || image.layout == new_layout) return;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = image.layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image;
        barrier.subresourceRange.aspectMask = image.aspect;
        barrier.subresourceRange.levelCount = image.mip_levels;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        if (image.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (image.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (image.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            // Graph images are consumed by both raster passes and compute
            // post-processing (notably the luminance histogram). Restricting
            // this dependency to fragment shaders lets the next frame start
            // overwriting scene_color while the histogram still reads it.
            src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }
        else if (image.layout == VK_IMAGE_LAYOUT_GENERAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }

        if (new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_GENERAL)
        {
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        image.layout = new_layout;
    }

    bool ensure_deferred_targets(std::uint32_t width, std::uint32_t height)
    {
        const VkImageUsageFlags sampled_color_usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        const bool ok = ensure_graph_image(scene_color_, width, height, scene_color_format_, sampled_color_usage,
                                           VK_IMAGE_ASPECT_COLOR_BIT) &&
                        ensure_graph_image(gbuffer_albedo_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                           sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
                        ensure_graph_image(gbuffer_normal_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                           sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
                        ensure_graph_image(gbuffer_material_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                           sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
                        ensure_graph_image(gbuffer_emissive_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                           sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
                        ensure_graph_image(gbuffer_motion_, width, height, VK_FORMAT_R16G16_SFLOAT, sampled_color_usage,
                                           VK_IMAGE_ASPECT_COLOR_BIT) &&
                        ensure_graph_image(gbuffer_object_id_, width, height, VK_FORMAT_R32_UINT, sampled_color_usage,
                                           VK_IMAGE_ASPECT_COLOR_BIT) &&
                        ensure_graph_image(selection_mask_, width, height, VK_FORMAT_R8_UNORM, sampled_color_usage,
                                           VK_IMAGE_ASPECT_COLOR_BIT);
        if (ok) update_gbuffer_descriptor_set();
        return ok;
    }

    void destroy_hzb_resources() noexcept
    {
        for (auto& image : hzb_history_)
            destroy_graph_image(image);
        hzb_descriptor_sets_.clear();
        if (hzb_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, hzb_pipeline_, nullptr);
        if (hzb_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, hzb_pipeline_layout_, nullptr);
        if (hzb_descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, hzb_descriptor_pool_, nullptr);
        if (hzb_descriptor_set_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, hzb_descriptor_set_layout_, nullptr);
        if (hzb_sampler_ != VK_NULL_HANDLE) vkDestroySampler(device_, hzb_sampler_, nullptr);
        hzb_pipeline_ = VK_NULL_HANDLE;
        hzb_pipeline_layout_ = VK_NULL_HANDLE;
        hzb_descriptor_pool_ = VK_NULL_HANDLE;
        hzb_descriptor_set_layout_ = VK_NULL_HANDLE;
        hzb_sampler_ = VK_NULL_HANDLE;
        hzb_mip_count_ = 0;
        hzb_history_valid_ = false;
    }

    bool ensure_hzb_resources(std::uint32_t width, std::uint32_t height)
    {
        if (!capabilities_.hzb_occlusion || viewport_depth_view_ == VK_NULL_HANDLE) return false;
        const auto mip_count = hzb_mip_count(width, height);
        const bool extent_changed =
            hzb_mip_count_ != mip_count || hzb_history_[0].width != width || hzb_history_[0].height != height;

        if (hzb_sampler_ == VK_NULL_HANDLE)
        {
            VkSamplerCreateInfo sampler{};
            sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler.magFilter = VK_FILTER_NEAREST;
            sampler.minFilter = VK_FILTER_NEAREST;
            sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.maxLod = static_cast<float>(mip_count);
            if (vkCreateSampler(device_, &sampler, nullptr, &hzb_sampler_) != VK_SUCCESS) return false;
        }

        if (hzb_descriptor_set_layout_ == VK_NULL_HANDLE)
        {
            const std::array bindings{VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                      VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                                                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
            VkDescriptorSetLayoutCreateInfo layout{};
            layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
            layout.pBindings = bindings.data();
            if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &hzb_descriptor_set_layout_) != VK_SUCCESS)
                return false;
        }

        if (hzb_pipeline_ == VK_NULL_HANDLE)
        {
            const auto shader =
                create_shader_module(builtin::depth_pyramid_comp_spv, std::size(builtin::depth_pyramid_comp_spv));
            if (shader == VK_NULL_HANDLE) return false;
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hzb_reduce_push_constants)};
            VkPipelineLayoutCreateInfo layout{};
            layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout.setLayoutCount = 1;
            layout.pSetLayouts = &hzb_descriptor_set_layout_;
            layout.pushConstantRangeCount = 1;
            layout.pPushConstantRanges = &push;
            if (vkCreatePipelineLayout(device_, &layout, nullptr, &hzb_pipeline_layout_) != VK_SUCCESS)
            {
                vkDestroyShaderModule(device_, shader, nullptr);
                return false;
            }
            VkComputePipelineCreateInfo pipeline{};
            pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipeline.stage.module = shader;
            pipeline.stage.pName = "main";
            pipeline.layout = hzb_pipeline_layout_;
            const auto result =
                vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &hzb_pipeline_);
            vkDestroyShaderModule(device_, shader, nullptr);
            if (result != VK_SUCCESS) return false;
        }

        if (!extent_changed && hzb_descriptor_pool_ != VK_NULL_HANDLE) return true;
        for (auto& image : hzb_history_)
            destroy_graph_image(image);
        if (hzb_descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, hzb_descriptor_pool_, nullptr);
            hzb_descriptor_pool_ = VK_NULL_HANDLE;
        }
        hzb_descriptor_sets_.clear();
        const auto usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        for (auto& image : hzb_history_)
            if (!ensure_graph_image(image, width, height, VK_FORMAT_R32G32_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT,
                                    mip_count))
                return false;

        const std::uint32_t set_count = mip_count * static_cast<std::uint32_t>(hzb_history_.size());
        const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, set_count},
                                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, set_count}};
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = set_count;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &hzb_descriptor_pool_) != VK_SUCCESS) return false;
        hzb_descriptor_sets_.resize(set_count);
        std::vector<VkDescriptorSetLayout> layouts(set_count, hzb_descriptor_set_layout_);
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = hzb_descriptor_pool_;
        allocate.descriptorSetCount = set_count;
        allocate.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(device_, &allocate, hzb_descriptor_sets_.data()) != VK_SUCCESS) return false;

        for (std::uint32_t generation = 0; generation < hzb_history_.size(); ++generation)
        {
            auto& image = hzb_history_[generation];
            for (std::uint32_t mip = 0; mip < mip_count; ++mip)
            {
                const auto index = generation * mip_count + mip;
                const VkDescriptorImageInfo source{hzb_sampler_, mip == 0 ? viewport_depth_view_ : image.view,
                                                   mip == 0 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                            : VK_IMAGE_LAYOUT_GENERAL};
                const VkDescriptorImageInfo destination{VK_NULL_HANDLE, image.mip_views[mip], VK_IMAGE_LAYOUT_GENERAL};
                std::array writes{VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
                                  VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
                writes[0].dstSet = hzb_descriptor_sets_[index];
                writes[0].dstBinding = 0;
                writes[0].descriptorCount = 1;
                writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[0].pImageInfo = &source;
                writes[1].dstSet = hzb_descriptor_sets_[index];
                writes[1].dstBinding = 1;
                writes[1].descriptorCount = 1;
                writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[1].pImageInfo = &destination;
                vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            }
        }
        hzb_mip_count_ = mip_count;
        hzb_history_valid_ = false;
        gpu_visibility_descriptors_dirty_ = true;
        return true;
    }

    void dispatch_hzb(VkCommandBuffer command_buffer)
    {
        if (!ensure_hzb_resources(viewport_width_, viewport_height_))
        {
            last_profile_.gpu_scene.fallback_reason = "HZB resources are unavailable; occlusion is disabled";
            return;
        }
        transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % hzb_history_.size());
        auto& image = hzb_history_[generation];
        transition_graph_image(command_buffer, image, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, hzb_pipeline_);
        for (std::uint32_t mip = 0; mip < hzb_mip_count_; ++mip)
        {
            const std::uint32_t destination_width = std::max(1u, viewport_width_ >> mip);
            const std::uint32_t destination_height = std::max(1u, viewport_height_ >> mip);
            const std::uint32_t source_width = mip == 0 ? viewport_width_ : std::max(1u, viewport_width_ >> (mip - 1u));
            const std::uint32_t source_height =
                mip == 0 ? viewport_height_ : std::max(1u, viewport_height_ >> (mip - 1u));
            const hzb_reduce_push_constants constants{
                static_cast<std::int32_t>(destination_width), static_cast<std::int32_t>(destination_height),
                static_cast<std::int32_t>(source_width), static_cast<std::int32_t>(source_height),
                mip == 0 ? -1 : static_cast<std::int32_t>(mip - 1u)};
            const auto set = hzb_descriptor_sets_[generation * hzb_mip_count_ + mip];
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, hzb_pipeline_layout_, 0, 1, &set, 0,
                                    nullptr);
            vkCmdPushConstants(command_buffer, hzb_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants),
                               &constants);
            vkCmdDispatch(command_buffer, (destination_width + 7u) / 8u, (destination_height + 7u) / 8u, 1u);
            if (mip + 1u < hzb_mip_count_)
            {
                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image.image;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.baseMipLevel = mip;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            }
        }
        hzb_history_valid_ = !frame_camera_.camera_cut;
        last_profile_.gpu_scene.history_valid = hzb_history_valid_;
        last_profile_.temporal.hzb_mip_count = hzb_mip_count_;
    }

    void destroy_temporal_resources() noexcept
    {
        for (auto& image : temporal_dilated_motion_)
            destroy_graph_image(image);
        for (auto& image : temporal_reactive_)
            destroy_graph_image(image);
        for (auto& image : temporal_disocclusion_)
            destroy_graph_image(image);
        for (auto& image : temporal_color_history_)
            destroy_graph_image(image);
        for (auto& image : temporal_depth_history_)
            destroy_graph_image(image);
        for (auto& image : temporal_moments_history_)
            destroy_graph_image(image);
        for (auto& image : temporal_confidence_history_)
            destroy_graph_image(image);
        for (auto& image : temporal_sharpened_)
            destroy_graph_image(image);
        if (temporal_descriptor_pool_ != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device_, temporal_descriptor_pool_, nullptr);
        if (temporal_mask_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, temporal_mask_pipeline_, nullptr);
        if (temporal_velocity_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, temporal_velocity_pipeline_, nullptr);
        if (temporal_resolve_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, temporal_resolve_pipeline_, nullptr);
        if (temporal_sharpen_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, temporal_sharpen_pipeline_, nullptr);
        if (temporal_mask_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, temporal_mask_pipeline_layout_, nullptr);
        if (temporal_velocity_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, temporal_velocity_pipeline_layout_, nullptr);
        if (temporal_resolve_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, temporal_resolve_pipeline_layout_, nullptr);
        if (temporal_sharpen_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, temporal_sharpen_pipeline_layout_, nullptr);
        if (temporal_mask_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, temporal_mask_descriptor_layout_, nullptr);
        if (temporal_velocity_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, temporal_velocity_descriptor_layout_, nullptr);
        if (temporal_resolve_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, temporal_resolve_descriptor_layout_, nullptr);
        if (temporal_sharpen_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, temporal_sharpen_descriptor_layout_, nullptr);
        temporal_descriptor_pool_ = VK_NULL_HANDLE;
        temporal_mask_pipeline_ = VK_NULL_HANDLE;
        temporal_velocity_pipeline_ = VK_NULL_HANDLE;
        temporal_resolve_pipeline_ = VK_NULL_HANDLE;
        temporal_sharpen_pipeline_ = VK_NULL_HANDLE;
        temporal_mask_pipeline_layout_ = VK_NULL_HANDLE;
        temporal_velocity_pipeline_layout_ = VK_NULL_HANDLE;
        temporal_resolve_pipeline_layout_ = VK_NULL_HANDLE;
        temporal_sharpen_pipeline_layout_ = VK_NULL_HANDLE;
        temporal_mask_descriptor_layout_ = VK_NULL_HANDLE;
        temporal_velocity_descriptor_layout_ = VK_NULL_HANDLE;
        temporal_resolve_descriptor_layout_ = VK_NULL_HANDLE;
        temporal_sharpen_descriptor_layout_ = VK_NULL_HANDLE;
        temporal_mask_sets_ = {};
        temporal_velocity_sets_ = {};
        temporal_resolve_sets_ = {};
        temporal_sharpen_sets_ = {};
        temporal_input_width_ = temporal_input_height_ = temporal_output_width_ = temporal_output_height_ = 0;
        temporal_history_valid_ = false;
        temporal_resources_initialized_ = false;
        temporal_output_view_ = VK_NULL_HANDLE;
    }

    bool create_temporal_pipeline(const std::uint32_t* code, std::size_t code_words, VkDescriptorSetLayout set_layout,
                                  std::uint32_t push_size, VkPipelineLayout& pipeline_layout, VkPipeline& pipeline)
    {
        const auto shader = create_shader_module(code, code_words);
        if (shader == VK_NULL_HANDLE) return false;
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &set_layout;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &pipeline_layout) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, shader, nullptr);
            return false;
        }
        VkComputePipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = shader;
        info.stage.pName = "main";
        info.layout = pipeline_layout;
        const auto result = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &info, nullptr, &pipeline);
        vkDestroyShaderModule(device_, shader, nullptr);
        return result == VK_SUCCESS;
    }

    bool ensure_temporal_pipelines()
    {
        if (temporal_resolve_pipeline_ != VK_NULL_HANDLE) return true;
        const auto make_layout =
            [&](std::uint32_t sampled_count, std::uint32_t storage_count, VkDescriptorSetLayout& result)
        {
            std::vector<VkDescriptorSetLayoutBinding> bindings(sampled_count + storage_count);
            for (std::uint32_t binding = 0; binding < bindings.size(); ++binding)
            {
                bindings[binding].binding = binding;
                bindings[binding].descriptorType = binding < sampled_count ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                           : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                bindings[binding].descriptorCount = 1;
                bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            info.bindingCount = static_cast<std::uint32_t>(bindings.size());
            info.pBindings = bindings.data();
            return vkCreateDescriptorSetLayout(device_, &info, nullptr, &result) == VK_SUCCESS;
        };
        if (!make_layout(2, 1, temporal_velocity_descriptor_layout_) ||
            !make_layout(4, 2, temporal_mask_descriptor_layout_) ||
            !make_layout(8, 4, temporal_resolve_descriptor_layout_) ||
            !make_layout(1, 1, temporal_sharpen_descriptor_layout_))
            return false;
        return create_temporal_pipeline(builtin::velocity_dilation_comp_spv,
                                        std::size(builtin::velocity_dilation_comp_spv),
                                        temporal_velocity_descriptor_layout_, sizeof(velocity_dilation_push_constants),
                                        temporal_velocity_pipeline_layout_, temporal_velocity_pipeline_) &&
               create_temporal_pipeline(builtin::temporal_masks_comp_spv, std::size(builtin::temporal_masks_comp_spv),
                                        temporal_mask_descriptor_layout_, sizeof(temporal_mask_push_constants),
                                        temporal_mask_pipeline_layout_, temporal_mask_pipeline_) &&
               create_temporal_pipeline(builtin::temporal_resolve_comp_spv,
                                        std::size(builtin::temporal_resolve_comp_spv),
                                        temporal_resolve_descriptor_layout_, sizeof(temporal_resolve_push_constants),
                                        temporal_resolve_pipeline_layout_, temporal_resolve_pipeline_) &&
               create_temporal_pipeline(builtin::spatial_sharpen_comp_spv, std::size(builtin::spatial_sharpen_comp_spv),
                                        temporal_sharpen_descriptor_layout_, sizeof(sharpen_push_constants),
                                        temporal_sharpen_pipeline_layout_, temporal_sharpen_pipeline_);
    }

    bool ensure_temporal_resources(std::uint32_t input_width, std::uint32_t input_height, std::uint32_t output_width,
                                   std::uint32_t output_height)
    {
        if (!capabilities_.temporal_resolve || !ensure_temporal_pipelines()) return false;
        if (temporal_descriptor_pool_ != VK_NULL_HANDLE && temporal_input_width_ == input_width &&
            temporal_input_height_ == input_height && temporal_output_width_ == output_width &&
            temporal_output_height_ == output_height)
            return true;

        for (auto& image : temporal_dilated_motion_)
            destroy_graph_image(image);
        for (auto& image : temporal_reactive_)
            destroy_graph_image(image);
        for (auto& image : temporal_disocclusion_)
            destroy_graph_image(image);
        for (auto& image : temporal_color_history_)
            destroy_graph_image(image);
        for (auto& image : temporal_depth_history_)
            destroy_graph_image(image);
        for (auto& image : temporal_moments_history_)
            destroy_graph_image(image);
        for (auto& image : temporal_confidence_history_)
            destroy_graph_image(image);
        for (auto& image : temporal_sharpened_)
            destroy_graph_image(image);
        if (temporal_descriptor_pool_ != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device_, temporal_descriptor_pool_, nullptr);
        temporal_descriptor_pool_ = VK_NULL_HANDLE;
        const auto usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        for (std::uint32_t generation = 0; generation < 2; ++generation)
        {
            if (!ensure_graph_image(temporal_dilated_motion_[generation], input_width, input_height,
                                    VK_FORMAT_R16G16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
                !ensure_graph_image(temporal_reactive_[generation], input_width, input_height, VK_FORMAT_R8_UNORM,
                                    usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
                !ensure_graph_image(temporal_disocclusion_[generation], input_width, input_height, VK_FORMAT_R8_UNORM,
                                    usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
                !ensure_graph_image(temporal_color_history_[generation], output_width, output_height,
                                    VK_FORMAT_R16G16B16A16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
                !ensure_graph_image(temporal_depth_history_[generation], output_width, output_height,
                                    VK_FORMAT_R32_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
                !ensure_graph_image(temporal_moments_history_[generation], output_width, output_height,
                                    VK_FORMAT_R16G16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
                !ensure_graph_image(temporal_confidence_history_[generation], output_width, output_height,
                                    VK_FORMAT_R8_UNORM, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
                !ensure_graph_image(temporal_sharpened_[generation], output_width, output_height,
                                    VK_FORMAT_R16G16B16A16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT))
                return false;
        }
        const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 30},
                                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16}};
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 8;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &temporal_descriptor_pool_) != VK_SUCCESS) return false;
        const std::array layouts{temporal_velocity_descriptor_layout_, temporal_mask_descriptor_layout_,
                                 temporal_resolve_descriptor_layout_,  temporal_sharpen_descriptor_layout_,
                                 temporal_velocity_descriptor_layout_, temporal_mask_descriptor_layout_,
                                 temporal_resolve_descriptor_layout_,  temporal_sharpen_descriptor_layout_};
        std::array<VkDescriptorSet, 8> sets{};
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = temporal_descriptor_pool_;
        allocate.descriptorSetCount = static_cast<std::uint32_t>(sets.size());
        allocate.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(device_, &allocate, sets.data()) != VK_SUCCESS) return false;
        for (std::uint32_t generation = 0; generation < 2; ++generation)
        {
            temporal_velocity_sets_[generation] = sets[generation * 4];
            temporal_mask_sets_[generation] = sets[generation * 4 + 1];
            temporal_resolve_sets_[generation] = sets[generation * 4 + 2];
            temporal_sharpen_sets_[generation] = sets[generation * 4 + 3];
        }
        temporal_input_width_ = input_width;
        temporal_input_height_ = input_height;
        temporal_output_width_ = output_width;
        temporal_output_height_ = output_height;
        temporal_history_valid_ = false;
        temporal_resources_initialized_ = false;
        return true;
    }

    void update_temporal_descriptors(std::uint32_t generation)
    {
        const auto previous = (generation + 1u) % 2u;
        const auto sampled = [&](VkImageView view, VkImageLayout layout)
        { return VkDescriptorImageInfo{viewport_sampler_, view, layout}; };
        const auto storage = [](VkImageView view)
        { return VkDescriptorImageInfo{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL}; };

        const std::array velocity_images{sampled(gbuffer_motion_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                                         sampled(viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                                         storage(temporal_dilated_motion_[generation].view)};
        const std::array mask_images{
            sampled(scene_color_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            sampled(viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            sampled(temporal_depth_history_[previous].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            sampled(temporal_dilated_motion_[generation].view, VK_IMAGE_LAYOUT_GENERAL),
            storage(temporal_reactive_[generation].view),
            storage(temporal_disocclusion_[generation].view)};
        const std::array resolve_images{
            sampled(scene_color_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            sampled(temporal_color_history_[previous].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            sampled(temporal_dilated_motion_[generation].view, VK_IMAGE_LAYOUT_GENERAL),
            sampled(temporal_reactive_[generation].view, VK_IMAGE_LAYOUT_GENERAL),
            sampled(temporal_disocclusion_[generation].view, VK_IMAGE_LAYOUT_GENERAL),
            sampled(viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            sampled(temporal_moments_history_[previous].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            sampled(temporal_confidence_history_[previous].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            storage(temporal_color_history_[generation].view),
            storage(temporal_depth_history_[generation].view),
            storage(temporal_moments_history_[generation].view),
            storage(temporal_confidence_history_[generation].view)};
        const std::array sharpen_images{
            sampled(temporal_color_history_[generation].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            storage(temporal_sharpened_[generation].view)};
        const auto write_images = [&](VkDescriptorSet set, const auto& images, std::uint32_t sampled_count)
        {
            std::vector<VkWriteDescriptorSet> writes(images.size());
            for (std::uint32_t binding = 0; binding < images.size(); ++binding)
            {
                writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[binding].dstSet = set;
                writes[binding].dstBinding = binding;
                writes[binding].descriptorCount = 1;
                writes[binding].descriptorType = binding < sampled_count ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                         : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[binding].pImageInfo = &images[binding];
            }
            vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        };
        write_images(temporal_velocity_sets_[generation], velocity_images, 2);
        write_images(temporal_mask_sets_[generation], mask_images, 4);
        write_images(temporal_resolve_sets_[generation], resolve_images, 8);
        write_images(temporal_sharpen_sets_[generation], sharpen_images, 1);
    }

    void prepare_temporal_images(VkCommandBuffer command_buffer, std::uint32_t generation)
    {
        const auto previous = (generation + 1u) % 2u;
        transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_depth(command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, temporal_dilated_motion_[generation], VK_IMAGE_LAYOUT_GENERAL);
        transition_graph_image(command_buffer, temporal_color_history_[previous],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, temporal_depth_history_[previous],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, temporal_moments_history_[previous],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, temporal_confidence_history_[previous],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, temporal_reactive_[generation], VK_IMAGE_LAYOUT_GENERAL);
        transition_graph_image(command_buffer, temporal_disocclusion_[generation], VK_IMAGE_LAYOUT_GENERAL);
        transition_graph_image(command_buffer, temporal_color_history_[generation], VK_IMAGE_LAYOUT_GENERAL);
        transition_graph_image(command_buffer, temporal_depth_history_[generation], VK_IMAGE_LAYOUT_GENERAL);
        transition_graph_image(command_buffer, temporal_moments_history_[generation], VK_IMAGE_LAYOUT_GENERAL);
        transition_graph_image(command_buffer, temporal_confidence_history_[generation], VK_IMAGE_LAYOUT_GENERAL);
        transition_graph_image(command_buffer, temporal_sharpened_[generation], VK_IMAGE_LAYOUT_GENERAL);
    }

    void dispatch_velocity_dilation(VkCommandBuffer command_buffer)
    {
        if (!ensure_temporal_resources(viewport_width_, viewport_height_, output_viewport_width_,
                                       output_viewport_height_))
            return;
        const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % 2u);
        prepare_temporal_images(command_buffer, generation);
        update_temporal_descriptors(generation);
        const velocity_dilation_push_constants constants{static_cast<std::int32_t>(viewport_width_),
                                                         static_cast<std::int32_t>(viewport_height_)};
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_velocity_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_velocity_pipeline_layout_, 0,
                                1, &temporal_velocity_sets_[generation], 0, nullptr);
        vkCmdPushConstants(command_buffer, temporal_velocity_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, (viewport_width_ + 7u) / 8u, (viewport_height_ + 7u) / 8u, 1u);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    void dispatch_temporal_masks(VkCommandBuffer command_buffer)
    {
        if (!ensure_temporal_resources(viewport_width_, viewport_height_, output_viewport_width_,
                                       output_viewport_height_))
            return;
        const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % 2u);
        prepare_temporal_images(command_buffer, generation);
        update_temporal_descriptors(generation);
        const temporal_mask_push_constants constants{
            static_cast<std::int32_t>(viewport_width_), static_cast<std::int32_t>(viewport_height_),
            temporal_history_valid_ && frame_camera_.history_valid ? 1u : 0u,
            resolved_config_.temporal.disocclusion_threshold, resolved_config_.temporal.reactive_response};
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_mask_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_mask_pipeline_layout_, 0, 1,
                                &temporal_mask_sets_[generation], 0, nullptr);
        vkCmdPushConstants(command_buffer, temporal_mask_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, (viewport_width_ + 7u) / 8u, (viewport_height_ + 7u) / 8u, 1u);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    void dispatch_temporal_resolve(VkCommandBuffer command_buffer)
    {
        const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % 2u);
        if (temporal_resolve_pipeline_ == VK_NULL_HANDLE || temporal_resolve_sets_[generation] == VK_NULL_HANDLE)
            return;
        const temporal_resolve_push_constants constants{static_cast<std::int32_t>(output_viewport_width_),
                                                        static_cast<std::int32_t>(output_viewport_height_),
                                                        static_cast<float>(viewport_width_),
                                                        static_cast<float>(viewport_height_),
                                                        temporal_history_valid_ && frame_camera_.history_valid ? 1u
                                                                                                               : 0u,
                                                        resolved_config_.temporal.history_weight};
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_resolve_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_resolve_pipeline_layout_, 0, 1,
                                &temporal_resolve_sets_[generation], 0, nullptr);
        vkCmdPushConstants(command_buffer, temporal_resolve_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, (output_viewport_width_ + 7u) / 8u, (output_viewport_height_ + 7u) / 8u, 1u);
        transition_graph_image(command_buffer, temporal_color_history_[generation],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        temporal_output_view_ = temporal_color_history_[generation].view;
        temporal_history_valid_ = true;
        last_profile_.temporal.enabled = true;
        last_profile_.temporal.upscaling =
            viewport_width_ != output_viewport_width_ || viewport_height_ != output_viewport_height_;
        last_profile_.temporal.effective_method =
            last_profile_.temporal.upscaling ? anti_aliasing_method::taau : anti_aliasing_method::taa;
        last_profile_.temporal.history_valid = frame_camera_.history_valid && !frame_camera_.camera_cut;
    }

    void dispatch_temporal_sharpen(VkCommandBuffer command_buffer)
    {
        const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % 2u);
        if (temporal_sharpen_pipeline_ == VK_NULL_HANDLE || temporal_sharpen_sets_[generation] == VK_NULL_HANDLE)
            return;
        transition_graph_image(command_buffer, temporal_sharpened_[generation], VK_IMAGE_LAYOUT_GENERAL);
        const sharpen_push_constants constants{static_cast<std::int32_t>(output_viewport_width_),
                                               static_cast<std::int32_t>(output_viewport_height_),
                                               resolved_config_.temporal.sharpening, 0.25f};
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_sharpen_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_sharpen_pipeline_layout_, 0, 1,
                                &temporal_sharpen_sets_[generation], 0, nullptr);
        vkCmdPushConstants(command_buffer, temporal_sharpen_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, (output_viewport_width_ + 7u) / 8u, (output_viewport_height_ + 7u) / 8u, 1u);
        transition_graph_image(command_buffer, temporal_sharpened_[generation],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        temporal_output_view_ = temporal_sharpened_[generation].view;
    }

    void ensure_viewport(std::uint32_t width, std::uint32_t height)
    {
        width = std::max(1u, width);
        height = std::max(1u, height);
        if (viewport_image_ != VK_NULL_HANDLE && viewport_width_ == width && viewport_height_ == height) return;

        wait_for_in_flight_frames();
        destroy_viewport();

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = viewport_format_;
        image.extent = {width, height, 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &viewport_image_, &viewport_allocation_, nullptr) !=
            VK_SUCCESS)
            return;

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = viewport_image_;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = viewport_format_;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device_, &view, nullptr, &viewport_view_) != VK_SUCCESS)
        {
            destroy_viewport();
            return;
        }

        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &sampler, nullptr, &viewport_sampler_) != VK_SUCCESS)
        {
            destroy_viewport();
            return;
        }

        viewport_width_ = width;
        viewport_height_ = height;
        exposure_needs_reset_ = true;
        viewport_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        ensure_deferred_targets(width, height);

        VkImageCreateInfo depth_image{};
        depth_image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        depth_image.imageType = VK_IMAGE_TYPE_2D;
        depth_image.format = depth_format_;
        depth_image.extent = {width, height, 1};
        depth_image.mipLevels = 1;
        depth_image.arrayLayers = 1;
        depth_image.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_image.tiling = VK_IMAGE_TILING_OPTIMAL;
        depth_image.usage =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo depth_allocation{};
        depth_allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &depth_image, &depth_allocation, &viewport_depth_image_,
                           &viewport_depth_allocation_, nullptr) != VK_SUCCESS)
        {
            destroy_viewport();
            return;
        }

        VkImageViewCreateInfo depth_view{};
        depth_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depth_view.image = viewport_depth_image_;
        depth_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depth_view.format = depth_format_;
        depth_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depth_view.subresourceRange.levelCount = 1;
        depth_view.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device_, &depth_view, nullptr, &viewport_depth_view_) != VK_SUCCESS)
        {
            destroy_viewport();
            return;
        }
        viewport_depth_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void destroy_viewport() noexcept
    {
        if (viewport_sampler_ != VK_NULL_HANDLE)
        {
            vkDestroySampler(device_, viewport_sampler_, nullptr);
            viewport_sampler_ = VK_NULL_HANDLE;
        }
        if (viewport_view_ != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, viewport_view_, nullptr);
            viewport_view_ = VK_NULL_HANDLE;
        }
        if (viewport_image_ != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator_, viewport_image_, viewport_allocation_);
            viewport_image_ = VK_NULL_HANDLE;
            viewport_allocation_ = VK_NULL_HANDLE;
        }
        if (viewport_depth_view_ != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, viewport_depth_view_, nullptr);
            viewport_depth_view_ = VK_NULL_HANDLE;
        }
        if (viewport_depth_image_ != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator_, viewport_depth_image_, viewport_depth_allocation_);
            viewport_depth_image_ = VK_NULL_HANDLE;
            viewport_depth_allocation_ = VK_NULL_HANDLE;
        }
        viewport_width_ = 0;
        viewport_height_ = 0;
        viewport_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        viewport_depth_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        destroy_graph_image(gbuffer_albedo_);
        destroy_graph_image(gbuffer_normal_);
        destroy_graph_image(gbuffer_material_);
        destroy_graph_image(gbuffer_emissive_);
        destroy_graph_image(gbuffer_motion_);
        destroy_graph_image(gbuffer_object_id_);
        destroy_graph_image(selection_mask_);
        destroy_graph_image(scene_color_);
        if (gbuffer_descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, gbuffer_descriptor_pool_, nullptr);
            gbuffer_descriptor_pool_ = VK_NULL_HANDLE;
            gbuffer_descriptor_set_ = VK_NULL_HANDLE;
        }
    }

    void transition_viewport(VkCommandBuffer command_buffer, VkImageLayout new_layout)
    {
        if (viewport_image_ == VK_NULL_HANDLE || viewport_layout_ == new_layout) return;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = viewport_layout_;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = viewport_image_;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (viewport_layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (viewport_layout_ == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (viewport_layout_ == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            if (viewport_layout_ == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            else
                barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src_stage = viewport_layout_ == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                            ? VK_PIPELINE_STAGE_TRANSFER_BIT
                            : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        viewport_layout_ = new_layout;
    }

    void transition_depth(VkCommandBuffer command_buffer, VkImageLayout new_layout)
    {
        if (viewport_depth_image_ == VK_NULL_HANDLE || viewport_depth_layout_ == new_layout) return;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = viewport_depth_layout_;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = viewport_depth_image_;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        VkPipelineStageFlags source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags destination_stage =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        if (viewport_depth_layout_ == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            source_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        }
        else if (viewport_depth_layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            source_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (viewport_depth_layout_ == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else
        {
            barrier.dstAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }
        vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        viewport_depth_layout_ = new_layout;
    }

    const directional_light_event* active_directional_shadow_light() const noexcept
    {
        if (!frame_shadows_enabled_) return nullptr;
        for (const auto& light : frame_directional_lights_)
        {
            if (light.enabled && light.casts_shadows && light.shadow.enabled) return &light;
        }
        return nullptr;
    }

    void execute_compiled_graph(VkCommandBuffer command_buffer)
    {
        bool directional_shadows_executed{};
        bool viewport_executed{};
        bool scene_executed{};
        bool point_shadows_executed{};
        bool spot_shadows_executed{};
        bool gpu_visibility_executed{};
        frame_fxaa_enabled_ = std::any_of(last_profile_.graph.passes.begin(), last_profile_.graph.passes.end(),
                                          [](const auto& pass) { return pass.builtin == builtin_render_pass::fxaa; });

        for (const auto& pass : last_profile_.graph.passes)
        {
            const auto scope = begin_gpu_scope(command_buffer, pass.name.c_str());
            switch (pass.builtin)
            {
                case builtin_render_pass::directional_shadow_static:
                case builtin_render_pass::directional_shadow_dynamic:
                    if (!directional_shadows_executed)
                    {
                        render_shadow_maps(command_buffer);
                        directional_shadows_executed = true;
                    }
                    break;
                case builtin_render_pass::point_shadow:
                    if (!point_shadows_executed)
                    {
                        render_local_shadow_maps(command_buffer, shadow_light_kind::point);
                        point_shadows_executed = true;
                    }
                    break;
                case builtin_render_pass::spot_shadow:
                    if (!spot_shadows_executed)
                    {
                        render_local_shadow_maps(command_buffer, shadow_light_kind::spot);
                        spot_shadows_executed = true;
                    }
                    break;
                case builtin_render_pass::gpu_frustum_distance_cull:
                    if (!gpu_visibility_executed)
                    {
                        dispatch_gpu_visibility(command_buffer);
                        gpu_visibility_executed = true;
                    }
                    break;
                case builtin_render_pass::depth_prepass:
                    if (!scene_executed)
                    {
                        // The current Vulkan raster path records depth and material outputs
                        // together. Starting it at the graph's depth boundary makes the
                        // produced depth available to the following HZB pass.
                        render_viewport(command_buffer, true, false);
                        scene_executed = true;
                    }
                    break;
                case builtin_render_pass::depth_pyramid:
                    dispatch_hzb(command_buffer);
                    break;
                case builtin_render_pass::velocity_dilation:
                    dispatch_velocity_dilation(command_buffer);
                    break;
                case builtin_render_pass::reactive_mask:
                    dispatch_temporal_masks(command_buffer);
                    break;
                case builtin_render_pass::temporal_antialiasing:
                case builtin_render_pass::temporal_upscale:
                    dispatch_temporal_resolve(command_buffer);
                    break;
                case builtin_render_pass::spatial_sharpen:
                    dispatch_temporal_sharpen(command_buffer);
                    break;
                case builtin_render_pass::gbuffer:
                case builtin_render_pass::forward_opaque:
                    if (!scene_executed)
                    {
                        render_viewport(command_buffer, true, false);
                        scene_executed = true;
                    }
                    break;
                case builtin_render_pass::output_transform:
                    // FXAA is folded into output conversion so it can filter
                    // tone-mapped linear color immediately before the single
                    // sRGB conversion. Its graph pass remains the execution
                    // boundary for this fused implementation.
                    if (frame_fxaa_enabled_) break;
                    [[fallthrough]];
                case builtin_render_pass::fxaa:
                    if (!viewport_executed)
                    {
                        if (!scene_executed)
                        {
                            render_viewport(command_buffer, true, false);
                            scene_executed = true;
                        }
                        render_viewport(command_buffer, false, true);
                        viewport_executed = true;
                    }
                    break;
                default:
                    break;
            }
            end_gpu_scope(command_buffer, scope);
        }

        if (!directional_shadows_executed)
            transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        if (!point_shadows_executed && !spot_shadows_executed)
            transition_local_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }

    void prepare_frame_gpu_resources()
    {
        update_dynamic_mesh_vertices();
        const auto* light = active_directional_shadow_light();
        auto settings = light ? light->shadow : shadow_settings{.enabled = false, .resolution = 2048};
        settings.resolution =
            std::min(std::bit_ceil(std::max(settings.resolution, 1u)), resolved_config_.directional_shadow_resolution);
        if (ensure_shadow_uniform_buffers() && ensure_shadow_resources(settings))
        {
            update_shadow_uniform(build_shadow_uniform(light));
            update_gbuffer_descriptor_set();
        }
        if (!ensure_local_shadow_resources() && !active_local_shadows_.empty())
        {
            last_profile_.shadows.fallback_reason =
                "local shadow atlas allocation failed; affected lights render unshadowed";
            active_local_shadows_.clear();
            frame_lighting_.local_shadow_face_count = 0u;
            update_light_buffer();
        }
        else
            update_gbuffer_descriptor_set();

        if (ensure_mesh_pipeline()) update_current_material_descriptor_sets();

        if ((light && !frame_shadow_draws_.empty()) || !active_local_shadows_.empty()) ensure_shadow_pipeline();
        const auto clear_overlay_counts = [&]
        {
            const auto slot = current_frame_slot();
            if (slot < debug_overlay_buffers_.size())
            {
                auto& buffer = debug_overlay_buffers_[slot];
                buffer.tested_line_count = 0;
                buffer.tested_triangle_count = 0;
                buffer.output_line_count = 0;
                buffer.output_triangle_count = 0;
            }
        };
        if (!frame_debug_overlay_lines_.empty() || !frame_debug_overlay_triangles_.empty())
        {
            if (!ensure_debug_overlay_pipeline() || !update_debug_overlay_buffer()) clear_overlay_counts();
        }
        else
            clear_overlay_counts();
    }

    void transition_shadow_atlas(VkCommandBuffer command_buffer, VkImageLayout new_layout)
    {
        if (shadow_atlas_.image == VK_NULL_HANDLE || shadow_atlas_.layout == new_layout) return;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = shadow_atlas_.layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = shadow_atlas_.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = directional_shadow_layer_count;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        if (shadow_atlas_.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (shadow_atlas_.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            src_stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        }

        if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src_stage = shadow_atlas_.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                                          : VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        shadow_atlas_.layout = new_layout;
    }

    void transition_local_shadow_atlas(VkCommandBuffer command_buffer, VkImageLayout new_layout)
    {
        if (local_shadow_atlas_.image == VK_NULL_HANDLE || local_shadow_atlas_.layout == new_layout) return;
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = local_shadow_atlas_.layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = local_shadow_atlas_.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        VkPipelineStageFlags source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags destination_stage =
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        if (local_shadow_atlas_.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            source_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (local_shadow_atlas_.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            source_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        }
        if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
            barrier.dstAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        local_shadow_atlas_.layout = new_layout;
    }

    shadow_uniform_data build_shadow_uniform(const directional_light_event* light) const noexcept
    {
        shadow_uniform_data data{};
        const auto identity = math::identity<float, 4>();
        for (auto& matrix : data.light_view_projection)
            std::copy(identity.data(), identity.data() + 16, matrix);

        if (!light)
        {
            data.params[0] = 0.0f;
            return data;
        }

        auto cascade_settings = light->cascades;
        cascade_settings.cascade_count =
            std::min(cascade_settings.cascade_count, resolved_config_.directional_shadow_cascades);
        cascade_settings.maximum_distance =
            std::min(cascade_settings.maximum_distance, resolved_config_.directional_shadow_distance);
        const auto layout =
            fit_directional_shadow_cascades({.inverse_view_projection = frame_camera_.inverse_view_projection,
                                             .near_plane = frame_camera_.near_plane,
                                             .far_plane = frame_camera_.far_plane},
                                            light->direction, cascade_settings, shadow_atlas_.resolution);

        for (std::uint32_t cascade = 0; cascade < layout.cascade_count; ++cascade)
        {
            const auto& fitted = layout.cascades[cascade];
            std::copy(fitted.light_view_projection.data(), fitted.light_view_projection.data() + 16,
                      data.light_view_projection[cascade]);
            data.cascade_splits[cascade] = fitted.split_depth;
            data.cascade_blend_starts[cascade] = fitted.blend_start_depth;
            data.cascade_texel_size[cascade] = fitted.texel_world_size;
        }

        data.params[0] = std::clamp(light->shadow.strength, 0.0f, 1.0f);
        data.params[1] = std::max(0.0f, light->shadow.bias);
        data.params[2] = std::max(0.0f, light->shadow.normal_bias);
        const auto filter = resolved_config_.quality == render_quality_tier::low
                                ? shadow_filter::pcf_3x3
                                : static_cast<shadow_filter>(std::min(static_cast<unsigned>(light->shadow.filter),
                                                                      static_cast<unsigned>(shadow_filter::pcf_5x5)));
        data.params[3] = static_cast<float>(filter);
        data.configuration[0] = static_cast<float>(layout.cascade_count);
        // Cascade splits are authored in camera view depth, not radial
        // distance. The remaining configuration lanes carry the normalized
        // camera forward vector so every lighting path selects the same
        // frustum slice without expanding the Vulkan 1.2-safe uniform.
        data.configuration[1] = frame_camera_.forward[0];
        data.configuration[2] = frame_camera_.forward[1];
        data.configuration[3] = frame_camera_.forward[2];
        return data;
    }

    void update_shadow_uniform(const shadow_uniform_data& data)
    {
        if (!ensure_shadow_uniform_buffers()) return;
        auto* shadow_buffer = current_shadow_uniform_buffer();
        if (shadow_buffer == nullptr || shadow_buffer->buffer == VK_NULL_HANDLE) return;
        void* mapped{};
        if (vmaMapMemory(allocator_, shadow_buffer->allocation, &mapped) != VK_SUCCESS) return;
        std::memcpy(mapped, &data, sizeof(data));
        vmaFlushAllocation(allocator_, shadow_buffer->allocation, 0, sizeof(data));
        vmaUnmapMemory(allocator_, shadow_buffer->allocation);
    }

    bool ensure_shadow_pipeline()
    {
        if (shadow_pipeline_ != VK_NULL_HANDLE) return true;
        if (max_push_constant_bytes_ < sizeof(mesh_push_constants) || !ensure_mesh_pipeline()) return false;

        VkShaderModule vert =
            create_shader_module(builtin::shadow_depth_vert_spv, std::size(builtin::shadow_depth_vert_spv));
        VkShaderModule frag =
            create_shader_module(builtin::shadow_depth_frag_spv, std::size(builtin::shadow_depth_frag_spv));
        VkShaderModule terrain_vert = create_shader_module(builtin::terrain_patch_shadow_vert_spv,
                                                           std::size(builtin::terrain_patch_shadow_vert_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
        {
            if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
            if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
            return false;
        }

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(mesh_push_constants);

        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &white_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &shadow_pipeline_layout_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
            return false;
        }

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(mesh_vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 2> attributes{
            VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)},
            VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)}};

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding;
        vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
        vertex_input.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        raster.depthBiasEnable = VK_TRUE;
        raster.depthBiasConstantFactor = 1.25f;
        raster.depthBiasSlopeFactor = 1.75f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.depthAttachmentFormat = depth_format_;

        VkGraphicsPipelineCreateInfo pipeline{};
        pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline.pNext = &rendering;
        pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
        pipeline.pStages = stages.data();
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input_assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &depth;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = shadow_pipeline_layout_;
        pipeline.renderPass = VK_NULL_HANDLE;

        const VkResult result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &shadow_pipeline_);
        if (result == VK_SUCCESS && terrain_vert != VK_NULL_HANDLE)
        {
            VkPipelineShaderStageCreateInfo terrain_stage{};
            terrain_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            terrain_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
            terrain_stage.module = terrain_vert;
            terrain_stage.pName = "main";
            VkPipelineVertexInputStateCreateInfo terrain_vertex_input{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            pipeline.stageCount = 1u;
            pipeline.pStages = &terrain_stage;
            pipeline.pVertexInputState = &terrain_vertex_input;
            pipeline.layout = terrain_pipeline_layout_;
            if (vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr,
                                          &terrain_shadow_pipeline_) != VK_SUCCESS)
            {
                terrain_shadow_pipeline_ = VK_NULL_HANDLE;
                arc::diagnostics::warn("render.vulkan",
                                       "Vulkan terrain shadow pipeline creation failed; terrain shadows are disabled");
            }
        }
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        if (terrain_vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_vert, nullptr);
        if (result != VK_SUCCESS)
        {
            arc::diagnostics::warn("render.vulkan",
                                   "Vulkan shadow pipeline creation failed; rendering will continue without shadows");
            return false;
        }
        return true;
    }

    void render_shadow_maps(VkCommandBuffer command_buffer)
    {
        const auto* light = active_directional_shadow_light();
        const shadow_settings settings = light ? light->shadow : shadow_settings{.enabled = false, .resolution = 2048};
        if (shadow_atlas_.image == VK_NULL_HANDLE) return;
        const auto uniform = build_shadow_uniform(light);

        if (!light || shadow_pipeline_ == VK_NULL_HANDLE)
        {
            transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
            return;
        }

        transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        const float resolution = static_cast<float>(shadow_atlas_.resolution);
        VkViewport viewport{};
        viewport.width = resolution;
        viewport.height = resolution;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor{};
        scissor.extent = {shadow_atlas_.resolution, shadow_atlas_.resolution};

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        const auto cascade_count = static_cast<std::uint32_t>(
            std::clamp(uniform.configuration[0], 1.0f, static_cast<float>(directional_shadow_cascade_count)));
        const auto within_shadow_distance = [&](const draw_mesh_event& draw)
        {
            if (draw.maximum_shadow_distance <= 0.0f) return true;
            // Draw events intentionally stay compact and do not duplicate the
            // render world's bounds. The model origin is a conservative,
            // stable distance proxy until shadow draws carry their selected
            // shadow LOD's tight bounds.
            const float x = draw.model(0, 3) - frame_camera_.position[0];
            const float y = draw.model(1, 3) - frame_camera_.position[1];
            const float z = draw.model(2, 3) - frame_camera_.position[2];
            return x * x + y * y + z * z <= draw.maximum_shadow_distance * draw.maximum_shadow_distance;
        };
        const auto is_static_caster = [&](const draw_mesh_event& draw)
        {
            return draw.casts_shadows && within_shadow_distance(draw) &&
                   draw.mobility == render_mobility::static_object;
        };
        const auto is_dynamic_caster = [&](const draw_mesh_event& draw)
        {
            if (!draw.casts_shadows || !within_shadow_distance(draw)) return false;
            if (light->mobility == render_mobility::static_object) return false;
            if (light->mobility == render_mobility::movable) return true;
            return draw.mobility != render_mobility::static_object;
        };
        const auto intersects_cascade = [](const draw_mesh_event& draw, const math::matrix4f& matrix)
        {
            const auto bounds_size = geometric::size(draw.world_bounds);
            if (math::length_squared(bounds_size) <= 1.0e-8f) return true;

            bool outside_left = true;
            bool outside_right = true;
            bool outside_bottom = true;
            bool outside_top = true;
            bool outside_near = true;
            bool outside_far = true;
            for (std::uint32_t corner = 0; corner < 8u; ++corner)
            {
                const float x = (corner & 1u) ? draw.world_bounds.max[0] : draw.world_bounds.min[0];
                const float y = (corner & 2u) ? draw.world_bounds.max[1] : draw.world_bounds.min[1];
                const float z = (corner & 4u) ? draw.world_bounds.max[2] : draw.world_bounds.min[2];
                const float clip_x = matrix(0, 0) * x + matrix(0, 1) * y + matrix(0, 2) * z + matrix(0, 3);
                const float clip_y = matrix(1, 0) * x + matrix(1, 1) * y + matrix(1, 2) * z + matrix(1, 3);
                const float clip_z = matrix(2, 0) * x + matrix(2, 1) * y + matrix(2, 2) * z + matrix(2, 3);
                const float clip_w = matrix(3, 0) * x + matrix(3, 1) * y + matrix(3, 2) * z + matrix(3, 3);
                outside_left &= clip_x < -clip_w;
                outside_right &= clip_x > clip_w;
                outside_bottom &= clip_y < -clip_w;
                outside_top &= clip_y > clip_w;
                outside_near &= clip_z < 0.0f;
                outside_far &= clip_z > clip_w;
            }
            return !(outside_left || outside_right || outside_bottom || outside_top || outside_near || outside_far);
        };

        std::uint64_t static_signature = 1469598103934665603ull;
        const auto hash_bytes = [&](const void* bytes, std::size_t count)
        {
            const auto* data = static_cast<const std::byte*>(bytes);
            for (std::size_t index = 0; index < count; ++index)
            {
                static_signature ^= static_cast<std::uint64_t>(std::to_integer<unsigned char>(data[index]));
                static_signature *= 1099511628211ull;
            }
        };
        hash_bytes(&uniform, sizeof(uniform));
        hash_bytes(&shadow_resource_revision_, sizeof(shadow_resource_revision_));
        for (const auto& draw : frame_shadow_draws_)
        {
            if (!is_static_caster(draw)) continue;
            hash_bytes(&draw.object_id, sizeof(draw.object_id));
            hash_bytes(draw.model.data(), sizeof(float) * 16u);
            hash_bytes(&draw.mesh, sizeof(draw.mesh));
            hash_bytes(&draw.material, sizeof(draw.material));
        }
        for (const auto& draw : frame_virtual_shadow_draws_)
        {
            if (!is_static_caster(draw.draw)) continue;
            hash_bytes(&draw.draw.object_id, sizeof(draw.draw.object_id));
            hash_bytes(draw.draw.model.data(), sizeof(float) * 16u);
            hash_bytes(&draw.mesh, sizeof(draw.mesh));
            hash_bytes(&draw.cluster_index, sizeof(draw.cluster_index));
            hash_bytes(&draw.draw.material, sizeof(draw.draw.material));
        }
        const bool redraw_static =
            !shadow_cache_.static_layers_valid || shadow_cache_.static_signature != static_signature ||
            light->mobility == render_mobility::movable || settings.cache_mode == shadow_cache_mode::always_update;
        last_static_shadow_cache_hit_ = !redraw_static;

        const auto render_layers = [&](std::uint32_t layer_offset, const auto& accepts_draw, bool enabled)
        {
            for (std::uint32_t cascade = 0; cascade < cascade_count; ++cascade)
            {
                VkRenderingAttachmentInfo depth_attachment{};
                depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                depth_attachment.imageView = shadow_atlas_.cascade_views[layer_offset + cascade];
                depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                depth_attachment.clearValue.depthStencil.depth = 1.0f;

                VkRenderingInfo rendering{};
                rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                rendering.renderArea.extent = {shadow_atlas_.resolution, shadow_atlas_.resolution};
                rendering.layerCount = 1;
                rendering.pDepthAttachment = &depth_attachment;
                cmd_begin_rendering(command_buffer, &rendering);

                if (enabled)
                {
                    math::matrix4f cascade_matrix;
                    std::copy(uniform.light_view_projection[cascade], uniform.light_view_projection[cascade] + 16,
                              cascade_matrix.data());
                    for (const auto& draw : frame_shadow_draws_)
                    {
                        if (!accepts_draw(draw) || !intersects_cascade(draw, cascade_matrix)) continue;

                        auto found = meshes_.find(resource_key(draw.mesh));
                        if (found == meshes_.end()) continue;

                        const math::matrix4f mvp = math::matmul(cascade_matrix, draw.model);
                        mesh_push_constants constants = build_mesh_constants(draw);
                        std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                        VkDescriptorSet descriptor_set = material_descriptor_set_for(draw);
                        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                shadow_pipeline_layout_, 0, 1, &descriptor_set, 0, nullptr);
                        vkCmdPushConstants(command_buffer, shadow_pipeline_layout_,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                           sizeof(constants), &constants);

                        const VkDeviceSize offset = 0;
                        const VkBuffer vertex_buffer = mesh_vertex_buffer(found->second);
                        if (vertex_buffer == VK_NULL_HANDLE) continue;
                        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
                        vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                        vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
                    }
                    for (const auto& draw : frame_virtual_shadow_draws_)
                    {
                        if (!accepts_draw(draw.draw) || !intersects_cascade(draw.draw, cascade_matrix)) continue;
                        const auto found = virtual_meshes_.find(resource_key(draw.mesh));
                        if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size())
                            continue;
                        const auto& cluster = found->second.clusters[draw.cluster_index];
                        if (cluster.index_count == 0 ||
                            cluster.first_index + cluster.index_count > found->second.index_count)
                            continue;

                        const math::matrix4f mvp = math::matmul(cascade_matrix, draw.draw.model);
                        mesh_push_constants constants = build_mesh_constants(draw.draw);
                        std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                        VkDescriptorSet descriptor_set = material_descriptor_set_for(draw.draw);
                        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                shadow_pipeline_layout_, 0, 1, &descriptor_set, 0, nullptr);
                        vkCmdPushConstants(command_buffer, shadow_pipeline_layout_,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                           sizeof(constants), &constants);
                        const VkDeviceSize offset = 0;
                        vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                        vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                        vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
                    }
                    if (layer_offset == directional_shadow_cascade_count && terrain_shadow_pipeline_ != VK_NULL_HANDLE)
                    {
                        for (const auto& draw : frame_terrain_draws_)
                        {
                            const auto terrain_draw = terrain_mesh_draw(draw);
                            if (!draw.terrain.cast_shadows || !intersects_cascade(terrain_draw, cascade_matrix))
                                continue;
                            auto shadow_draw = draw;
                            shadow_draw.view_projection = cascade_matrix;
                            draw_terrain_patch(command_buffer, shadow_draw, terrain_shadow_pipeline_, false);
                        }
                    }
                }
                cmd_end_rendering(command_buffer);
            }
        };

        if (redraw_static)
        {
            render_layers(0u, is_static_caster, light->mobility != render_mobility::movable);
            shadow_cache_.static_signature = static_signature;
            shadow_cache_.static_layers_valid = true;
        }
        render_layers(directional_shadow_cascade_count, is_dynamic_caster, true);

        transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        shadow_cache_.last_directional_key = {
            .light_index = 0, .resolution = shadow_atlas_.resolution, .filter = settings.filter};
        shadow_cache_.has_directional_key = true;
    }

    void render_local_shadow_maps(VkCommandBuffer command_buffer, shadow_light_kind requested_kind)
    {
        if (active_local_shadows_.empty() || local_shadow_atlas_.image == VK_NULL_HANDLE ||
            shadow_pipeline_ == VK_NULL_HANDLE)
            return;

        transition_local_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);

        std::uint32_t packed_face_index{};
        for (const auto& shadow : active_local_shadows_)
        {
            if (shadow.kind != requested_kind)
            {
                packed_face_index += shadow.allocation.face_count;
                continue;
            }
            if (!shadow.redraw)
            {
                packed_face_index += shadow.allocation.face_count;
                continue;
            }

            const auto in_light_range = [&](const draw_mesh_event& draw)
            {
                const auto bounds_size = geometric::size(draw.world_bounds);
                math::vector3f center = matrix_translation(draw.model);
                if (math::length_squared(bounds_size) > 1.0e-8f)
                {
                    const auto bounds_center = geometric::center(draw.world_bounds);
                    center = {bounds_center[0], bounds_center[1], bounds_center[2]};
                }
                return math::length_squared(math::sub(center, shadow.position)) <= shadow.range * shadow.range;
            };
            for (std::uint32_t face = 0; face < shadow.allocation.face_count; ++face)
            {
                if (packed_face_index >= frame_lighting_.local_shadow_face_count) break;
                const auto& packed = frame_lighting_.local_shadow_faces[packed_face_index++];
                const auto& rect = shadow.allocation.faces[face];
                VkViewport viewport{};
                viewport.x = static_cast<float>(rect.content_x());
                viewport.y = static_cast<float>(rect.content_y());
                viewport.width = static_cast<float>(rect.content_size());
                viewport.height = static_cast<float>(rect.content_size());
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                VkRect2D scissor{};
                scissor.offset = {static_cast<std::int32_t>(rect.content_x()),
                                  static_cast<std::int32_t>(rect.content_y())};
                scissor.extent = {rect.content_size(), rect.content_size()};
                vkCmdSetViewport(command_buffer, 0, 1, &viewport);
                vkCmdSetScissor(command_buffer, 0, 1, &scissor);

                VkRenderingAttachmentInfo depth_attachment{};
                depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                depth_attachment.imageView = local_shadow_atlas_.view;
                depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                depth_attachment.clearValue.depthStencil.depth = 1.0f;
                VkRenderingInfo rendering{};
                rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                rendering.renderArea.offset = {static_cast<std::int32_t>(rect.x), static_cast<std::int32_t>(rect.y)};
                rendering.renderArea.extent = {rect.size, rect.size};
                rendering.layerCount = 1;
                rendering.pDepthAttachment = &depth_attachment;
                cmd_begin_rendering(command_buffer, &rendering);

                const auto draw_mesh = [&](const draw_mesh_event& draw, const gpu_mesh& mesh)
                {
                    if (!draw.casts_shadows || !in_light_range(draw) ||
                        (shadow.mobility == render_mobility::static_object &&
                         draw.mobility != render_mobility::static_object))
                        return;
                    mesh_push_constants constants = build_mesh_constants(draw);
                    const auto mvp = math::matmul(packed.light_view_projection, draw.model);
                    std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                    VkDescriptorSet descriptor_set = material_descriptor_set_for(draw);
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0,
                                            1, &descriptor_set, 0, nullptr);
                    vkCmdPushConstants(command_buffer, shadow_pipeline_layout_,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                       &constants);
                    const VkDeviceSize offset{};
                    const auto vertex_buffer = mesh_vertex_buffer(mesh);
                    if (vertex_buffer == VK_NULL_HANDLE) return;
                    vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
                    vkCmdBindIndexBuffer(command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
                };
                for (const auto& draw : frame_shadow_draws_)
                {
                    const auto found = meshes_.find(resource_key(draw.mesh));
                    if (found != meshes_.end()) draw_mesh(draw, found->second);
                }
                for (const auto& draw : frame_virtual_shadow_draws_)
                {
                    if (!draw.draw.casts_shadows || !in_light_range(draw.draw) ||
                        (shadow.mobility == render_mobility::static_object &&
                         draw.draw.mobility != render_mobility::static_object))
                        continue;
                    const auto found = virtual_meshes_.find(resource_key(draw.mesh));
                    if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size()) continue;
                    const auto& cluster = found->second.clusters[draw.cluster_index];
                    mesh_push_constants constants = build_mesh_constants(draw.draw);
                    const auto mvp = math::matmul(packed.light_view_projection, draw.draw.model);
                    std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                    VkDescriptorSet descriptor_set = material_descriptor_set_for(draw.draw);
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0,
                                            1, &descriptor_set, 0, nullptr);
                    vkCmdPushConstants(command_buffer, shadow_pipeline_layout_,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                       &constants);
                    const VkDeviceSize offset{};
                    vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                    vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
                }
                if (terrain_shadow_pipeline_ != VK_NULL_HANDLE)
                {
                    for (const auto& draw : frame_terrain_draws_)
                    {
                        const auto terrain_draw = terrain_mesh_draw(draw);
                        if (!draw.terrain.cast_shadows || !in_light_range(terrain_draw)) continue;
                        auto shadow_draw = draw;
                        shadow_draw.view_projection = packed.light_view_projection;
                        draw_terrain_patch(command_buffer, shadow_draw, terrain_shadow_pipeline_, false);
                    }
                }
                cmd_end_rendering(command_buffer);
            }
        }
        transition_local_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }

    void set_viewport_and_scissor(VkCommandBuffer command_buffer) const
    {
        VkViewport viewport{};
        viewport.y = static_cast<float>(viewport_height_);
        viewport.width = static_cast<float>(viewport_width_);
        viewport.height = -static_cast<float>(viewport_height_);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor{};
        scissor.extent = {viewport_width_, viewport_height_};
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
    }

    void draw_debug_overlay(VkCommandBuffer command_buffer, debug_overlay_depth_mode mode)
    {
        const auto slot = current_frame_slot();
        if (slot >= debug_overlay_buffers_.size() || debug_overlay_pipeline_layout_ == VK_NULL_HANDLE) return;
        const auto& buffer = debug_overlay_buffers_[slot];
        if (buffer.vertices.buffer == VK_NULL_HANDLE) return;
        set_viewport_and_scissor(command_buffer);
        vkCmdPushConstants(command_buffer, debug_overlay_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(float) * 16u, frame_camera_.view_projection.data());
        const VkDeviceSize offset{};
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &buffer.vertices.buffer, &offset);
        const auto draw_range = [&](VkPipeline pipeline, std::uint32_t count, std::uint32_t first)
        {
            if (pipeline == VK_NULL_HANDLE || count == 0) return;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdDraw(command_buffer, count, 1, first, 0);
        };
        if (mode == debug_overlay_depth_mode::tested)
        {
            draw_range(debug_overlay_line_pipeline_, buffer.tested_line_count, buffer.tested_line_offset);
            draw_range(debug_overlay_triangle_pipeline_, buffer.tested_triangle_count, buffer.tested_triangle_offset);
        }
        else
        {
            draw_range(debug_overlay_output_line_pipeline_, buffer.output_line_count, buffer.output_line_offset);
            draw_range(debug_overlay_output_triangle_pipeline_, buffer.output_triangle_count,
                       buffer.output_triangle_offset);
        }
    }

    void draw_indexed_mesh(VkCommandBuffer command_buffer, const draw_mesh_event& draw, VkPipelineLayout layout,
                           VkShaderStageFlags stages, bool gpu_culled = false, bool write_motion = false)
    {
        const auto found = meshes_.find(resource_key(draw.mesh));
        if (found == meshes_.end()) return;

        auto constants = build_mesh_constants(draw);
        if (write_motion)
        {
            const auto previous_mvp = math::matmul(draw.previous_view_projection, draw.previous_model);
            const auto* values = previous_mvp.data();
            std::copy(values, values + 4, constants.light_direction_intensity);
            std::copy(values + 4, values + 7, constants.light_color);
            constants.camera_position[0] = values[7];
            std::copy(values + 8, values + 11, constants.camera_position + 1);
            constants.fog_color_density[0] = values[11];
            std::copy(values + 12, values + 15, constants.fog_color_density + 1);
            constants.fog_params[0] = values[15];
        }
        vkCmdPushConstants(command_buffer, layout, stages, 0, sizeof(constants), &constants);
        const VkDeviceSize offset = 0;
        const VkBuffer vertex_buffer = mesh_vertex_buffer(found->second);
        if (vertex_buffer == VK_NULL_HANDLE) return;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
        vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        if (!gpu_culled || !draw_gpu_visibility_command(command_buffer, draw.gpu_scene_instance))
            vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
    }

    void draw_indexed_virtual_cluster(VkCommandBuffer command_buffer, const virtual_cluster_draw& draw,
                                      VkPipelineLayout layout, VkShaderStageFlags stages, bool gpu_culled = false,
                                      bool write_motion = false)
    {
        const auto found = virtual_meshes_.find(resource_key(draw.mesh));
        if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size()) return;

        const auto& cluster = found->second.clusters[draw.cluster_index];
        if (cluster.index_count == 0 || cluster.first_index + cluster.index_count > found->second.index_count) return;

        auto constants = build_mesh_constants(draw.draw);
        if (write_motion)
        {
            const auto previous_mvp = math::matmul(draw.draw.previous_view_projection, draw.draw.previous_model);
            const auto* values = previous_mvp.data();
            std::copy(values, values + 4, constants.light_direction_intensity);
            std::copy(values + 4, values + 7, constants.light_color);
            constants.camera_position[0] = values[7];
            std::copy(values + 8, values + 11, constants.camera_position + 1);
            constants.fog_color_density[0] = values[11];
            std::copy(values + 12, values + 15, constants.fog_color_density + 1);
            constants.fog_params[0] = values[15];
        }
        vkCmdPushConstants(command_buffer, layout, stages, 0, sizeof(constants), &constants);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
        vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        if (!gpu_culled || !draw_gpu_visibility_command(command_buffer, draw.draw.gpu_scene_instance))
            vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
    }

    bool render_deferred_scene(VkCommandBuffer command_buffer)
    {
        if ((frame_draws_.empty() && frame_virtual_draws_.empty() && frame_terrain_draws_.empty()) ||
            !ensure_deferred_targets(viewport_width_, viewport_height_) || !ensure_shadow_pipeline() ||
            !ensure_gbuffer_pipeline() || !ensure_gbuffer_descriptor_set() || !ensure_deferred_pipeline())
            return false;

        bool has_opaque_draws = false;
        for (const auto& draw : frame_draws_)
        {
            if (draw.mode != render_mode::wireframe && material_alpha_mode_for(draw) != material_alpha_mode::blend &&
                !material_requires_forward(draw))
            {
                has_opaque_draws = true;
                break;
            }
        }
        for (const auto& draw : frame_virtual_draws_)
        {
            if (draw.draw.mode != render_mode::wireframe &&
                material_alpha_mode_for(draw.draw) != material_alpha_mode::blend &&
                !material_requires_forward(draw.draw))
            {
                has_opaque_draws = true;
                break;
            }
        }
        has_opaque_draws = has_opaque_draws || !frame_terrain_draws_.empty();
        if (!has_opaque_draws) return false;

        transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        {
            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = viewport_depth_view_;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil.depth = 1.0f;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.extent = {viewport_width_, viewport_height_};
            rendering.layerCount = 1;
            rendering.pDepthAttachment = &depth_attachment;
            cmd_begin_rendering(command_buffer, &rendering);
            set_viewport_and_scissor(command_buffer);
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);

            for (const auto& draw : frame_draws_)
            {
                if (draw.mode == render_mode::wireframe) continue;
                VkDescriptorSet descriptor_set = material_descriptor_set_for(draw);
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0, 1,
                                        &descriptor_set, 0, nullptr);
                draw_indexed_mesh(command_buffer, draw, shadow_pipeline_layout_,
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true);
            }
            for (const auto& draw : frame_virtual_draws_)
            {
                if (draw.draw.mode == render_mode::wireframe) continue;
                VkDescriptorSet descriptor_set = material_descriptor_set_for(draw.draw);
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0, 1,
                                        &descriptor_set, 0, nullptr);
                draw_indexed_virtual_cluster(command_buffer, draw, shadow_pipeline_layout_,
                                             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true);
            }
            if (terrain_shadow_pipeline_ != VK_NULL_HANDLE)
            {
                for (const auto& draw : frame_terrain_draws_)
                    draw_terrain_patch(command_buffer, draw, terrain_shadow_pipeline_, false);
            }
            cmd_end_rendering(command_buffer);
        }

        transition_graph_image(command_buffer, gbuffer_albedo_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_normal_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_material_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_emissive_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        {
            std::array<VkRenderingAttachmentInfo, 6> color_attachments{};
            graph_image* images[6]{&gbuffer_albedo_,   &gbuffer_normal_, &gbuffer_material_,
                                   &gbuffer_emissive_, &gbuffer_motion_, &gbuffer_object_id_};
            for (std::size_t index = 0; index < color_attachments.size(); ++index)
            {
                color_attachments[index].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                color_attachments[index].imageView = images[index]->view;
                color_attachments[index].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                color_attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                color_attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                color_attachments[index].clearValue.color.float32[0] = 0.0f;
                color_attachments[index].clearValue.color.float32[1] = 0.0f;
                color_attachments[index].clearValue.color.float32[2] = 0.0f;
                color_attachments[index].clearValue.color.float32[3] = 0.0f;
            }

            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = viewport_depth_view_;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.extent = {viewport_width_, viewport_height_};
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_attachments.size());
            rendering.pColorAttachments = color_attachments.data();
            rendering.pDepthAttachment = &depth_attachment;
            cmd_begin_rendering(command_buffer, &rendering);
            set_viewport_and_scissor(command_buffer);
            for (const auto& draw : frame_draws_)
            {
                if (draw.mode == render_mode::wireframe || material_alpha_mode_for(draw) == material_alpha_mode::blend)
                    continue;
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  material_is_terrain(draw) && terrain_gbuffer_pipeline_ != VK_NULL_HANDLE
                                      ? terrain_gbuffer_pipeline_
                                      : gbuffer_pipeline_);
                VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw);
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline_layout_, 0, 1,
                                        &material_descriptor_set, 0, nullptr);
                draw_indexed_mesh(command_buffer, draw, mesh_pipeline_layout_,
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true, true);
            }
            for (const auto& draw : frame_virtual_draws_)
            {
                if (draw.draw.mode == render_mode::wireframe ||
                    material_alpha_mode_for(draw.draw) == material_alpha_mode::blend)
                    continue;
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  material_is_terrain(draw.draw) && terrain_gbuffer_pipeline_ != VK_NULL_HANDLE
                                      ? terrain_gbuffer_pipeline_
                                      : gbuffer_pipeline_);
                VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw.draw);
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline_layout_, 0, 1,
                                        &material_descriptor_set, 0, nullptr);
                draw_indexed_virtual_cluster(command_buffer, draw, mesh_pipeline_layout_,
                                             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true, true);
            }
            for (const auto& draw : frame_terrain_draws_)
                draw_terrain_patch(command_buffer, draw, terrain_gbuffer_pipeline_, true);

            cmd_end_rendering(command_buffer);
        }

        transition_graph_image(command_buffer, gbuffer_albedo_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_normal_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_material_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_emissive_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_depth(command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        update_gbuffer_descriptor_set();

        {
            transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            VkRenderingAttachmentInfo color_attachment{};
            color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment.imageView = scene_color_.view;
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.extent = {viewport_width_, viewport_height_};
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color_attachment;
            cmd_begin_rendering(command_buffer, &rendering);
            set_viewport_and_scissor(command_buffer);
            deferred_push_constants constants{};
            std::copy(frame_camera_.inverse_view_projection.data(), frame_camera_.inverse_view_projection.data() + 16,
                      constants.inverse_view_projection);
            constants.camera_position[0] = frame_camera_.position[0];
            constants.camera_position[1] = frame_camera_.position[1];
            constants.camera_position[2] = frame_camera_.position[2];
            constants.light_direction_intensity[0] = 0.35f;
            constants.light_direction_intensity[1] = -0.85f;
            constants.light_direction_intensity[2] = -0.40f;
            constants.light_direction_intensity[3] = frame_shadows_enabled_ ? 1.0f : 0.0f;
            if (!frame_directional_lights_.empty())
            {
                const auto& light = frame_directional_lights_.front();
                constants.light_direction_intensity[0] = light.direction[0];
                constants.light_direction_intensity[1] = light.direction[1];
                constants.light_direction_intensity[2] = light.direction[2];
                constants.light_color[0] = light.color[0];
                constants.light_color[1] = light.color[1];
                constants.light_color[2] = light.color[2];
            }
            constants.ambient_visualization[0] =
                frame_lighting_.ambient_color_intensity[0] * frame_lighting_.ambient_color_intensity[3];
            constants.ambient_visualization[1] =
                frame_lighting_.ambient_color_intensity[1] * frame_lighting_.ambient_color_intensity[3];
            constants.ambient_visualization[2] =
                frame_lighting_.ambient_color_intensity[2] * frame_lighting_.ambient_color_intensity[3];
            if (const auto* environment = active_environment())
            {
                const auto found = textures_.find(resource_key(environment->equirectangular_texture));
                constants.light_color[3] =
                    found != textures_.end() && found->second.view != VK_NULL_HANDLE ? 1.0f : 0.0f;
            }
            constants.ambient_visualization[3] =
                !frame_draws_.empty() ? static_cast<float>(frame_draws_.front().visualization)
                                      : static_cast<float>(frame_virtual_draws_.front().draw.visualization);
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferred_pipeline_);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferred_pipeline_layout_, 0, 1,
                                    &gbuffer_descriptor_set_, 0, nullptr);
            vkCmdPushConstants(command_buffer, deferred_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(constants), &constants);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
            cmd_end_rendering(command_buffer);
        }

        if (pending_pick_request_ && ensure_pick_readback_buffer())
        {
            const auto request = *pending_pick_request_;
            pending_pick_request_.reset();

            if (request.x < output_viewport_width_ && request.y < output_viewport_height_ &&
                gbuffer_object_id_.width > 0 && gbuffer_object_id_.height > 0)
            {
                object_pick_readback readback{};
                readback.request = request;
                readback.frame_index = last_profile_.frame_index;
                readback.frame_slot = active_frame_index_;
                readback.active = true;

                for (const auto& draw : frame_draws_)
                {
                    if (draw.object_id.valid()) readback.objects.emplace(draw.object_id.index + 1u, draw.object_id);
                }
                for (const auto& draw : frame_virtual_draws_)
                {
                    if (draw.draw.object_id.valid())
                        readback.objects.emplace(draw.draw.object_id.index + 1u, draw.draw.object_id);
                }

                transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

                VkBufferImageCopy region{};
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.layerCount = 1;
                const auto render_x = detail::map_output_pixel_to_render_pixel(request.x, output_viewport_width_,
                                                                               gbuffer_object_id_.width);
                const auto render_y = detail::map_output_pixel_to_render_pixel(request.y, output_viewport_height_,
                                                                               gbuffer_object_id_.height);
                region.imageOffset = {static_cast<std::int32_t>(render_x), static_cast<std::int32_t>(render_y), 0};
                region.imageExtent = {1, 1, 1};
                vkCmdCopyImageToBuffer(command_buffer, gbuffer_object_id_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       pick_readback_buffer_.buffer, 1, &region);

                in_flight_pick_ = std::move(readback);
            }
            else
            {
                last_pick_result_ = {.request_id = request.request_id,
                                     .available = true,
                                     .hit = false,
                                     .object = {},
                                     .x = request.x,
                                     .y = request.y,
                                     .frame_index = last_profile_.frame_index};
            }
        }

        return true;
    }

    void render_viewport(VkCommandBuffer command_buffer, bool render_scene, bool render_output)
    {
        if (viewport_image_ == VK_NULL_HANDLE) return;

        if (render_scene)
        {
            transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

            VkRenderingAttachmentInfo color_attachment{};
            color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment.imageView = scene_color_.view;
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            for (std::uint32_t channel = 0; channel < 4; ++channel)
                color_attachment.clearValue.color.float32[channel] = frame_camera_.clear_color[channel];

            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = viewport_depth_view_;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil.depth = 1.0f;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.extent = {viewport_width_, viewport_height_};
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color_attachment;
            rendering.pDepthAttachment = &depth_attachment;
            cmd_begin_rendering(command_buffer, &rendering);

            if (frame_environment_.enabled && frame_environment_.sky_visible && ensure_sky_pipeline())
            {
                VkViewport viewport{};
                viewport.y = static_cast<float>(viewport_height_);
                viewport.width = static_cast<float>(viewport_width_);
                viewport.height = -static_cast<float>(viewport_height_);
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                VkRect2D scissor{};
                scissor.extent = {viewport_width_, viewport_height_};
                vkCmdSetViewport(command_buffer, 0, 1, &viewport);
                vkCmdSetScissor(command_buffer, 0, 1, &scissor);

                math::vector3f sun_direction_override{};
                if (!frame_environment_.celestial.enabled && !frame_directional_lights_.empty())
                    sun_direction_override = frame_directional_lights_.front().direction;
                const auto constants = detail::build_sky_push_constants(
                    frame_environment_, frame_camera_, viewport_width_, viewport_height_,
                    resolved_config_.quality != render_quality_tier::low, sun_direction_override);

                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_);
                const auto sky_descriptor = update_current_sky_descriptor_set();
                if (sky_descriptor != VK_NULL_HANDLE)
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_layout_, 0, 1,
                                            &sky_descriptor, 0, nullptr);
                vkCmdPushConstants(command_buffer, sky_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(constants), &constants);
                vkCmdDraw(command_buffer, 3, 1, 0, 0);
            }

            cmd_end_rendering(command_buffer);
            const bool deferred_rendered =
                resolved_config_.path == render_path::deferred && render_deferred_scene(command_buffer);

            transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            cmd_begin_rendering(command_buffer, &rendering);

            if ((!frame_draws_.empty() || !frame_virtual_draws_.empty()) && mesh_pipeline_ != VK_NULL_HANDLE &&
                !white_descriptor_sets_.empty())
            {
                VkViewport viewport{};
                viewport.y = static_cast<float>(viewport_height_);
                viewport.width = static_cast<float>(viewport_width_);
                viewport.height = -static_cast<float>(viewport_height_);
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                VkRect2D scissor{};
                scissor.extent = {viewport_width_, viewport_height_};
                vkCmdSetViewport(command_buffer, 0, 1, &viewport);
                vkCmdSetScissor(command_buffer, 0, 1, &scissor);

                const auto draw_with_pipeline = [&](const draw_mesh_event& draw, VkPipeline pipeline)
                {
                    if (pipeline == VK_NULL_HANDLE) return;
                    auto found = meshes_.find(resource_key(draw.mesh));
                    if (found == meshes_.end()) return;

                    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                    auto constants = build_mesh_constants(draw);
                    if (pipeline == mesh_wire_pipeline_)
                    {
                        std::copy(draw.wire_color.data(), draw.wire_color.data() + 4, constants.base_color);
                        constants.light_color[3] = 0.0f;
                        constants.visualization[0] = static_cast<float>(mesh_visualization_mode::albedo);
                        constants.fog_color_density[3] = 0.0f;
                        constants.material_params[3] = static_cast<float>(material_alpha_mode::opaque);
                    }
                    VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw);
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline_layout_, 0,
                                            1, &material_descriptor_set, 0, nullptr);
                    vkCmdPushConstants(command_buffer, mesh_pipeline_layout_,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                       &constants);
                    const VkDeviceSize offset = 0;
                    const VkBuffer vertex_buffer = mesh_vertex_buffer(found->second);
                    if (vertex_buffer == VK_NULL_HANDLE) return;
                    vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
                    vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                    if (!draw_gpu_visibility_command(command_buffer, draw.gpu_scene_instance))
                        vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
                };
                const auto draw_virtual_with_pipeline = [&](const virtual_cluster_draw& draw, VkPipeline pipeline)
                {
                    if (pipeline == VK_NULL_HANDLE) return;
                    const auto found = virtual_meshes_.find(resource_key(draw.mesh));
                    if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size()) return;

                    const auto& cluster = found->second.clusters[draw.cluster_index];
                    if (cluster.index_count == 0 ||
                        cluster.first_index + cluster.index_count > found->second.index_count)
                        return;

                    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                    auto constants = build_mesh_constants(draw.draw);
                    if (pipeline == mesh_wire_pipeline_)
                    {
                        std::copy(draw.draw.wire_color.data(), draw.draw.wire_color.data() + 4, constants.base_color);
                        constants.light_color[3] = 0.0f;
                        constants.visualization[0] = static_cast<float>(mesh_visualization_mode::albedo);
                        constants.fog_color_density[3] = 0.0f;
                        constants.material_params[3] = static_cast<float>(material_alpha_mode::opaque);
                    }
                    VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw.draw);
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline_layout_, 0,
                                            1, &material_descriptor_set, 0, nullptr);
                    vkCmdPushConstants(command_buffer, mesh_pipeline_layout_,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                       &constants);
                    const VkDeviceSize offset = 0;
                    vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                    vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                    if (!draw_gpu_visibility_command(command_buffer, draw.draw.gpu_scene_instance))
                        vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
                };

                for (const auto& draw : frame_draws_)
                {
                    if (draw.mode == render_mode::wireframe)
                    {
                        if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
                            draw_with_pipeline(draw, mesh_wire_pipeline_);
                        else
                            draw_with_pipeline(draw, mesh_pipeline_);
                        continue;
                    }

                    if (material_alpha_mode_for(draw) == material_alpha_mode::blend) continue;

                    if (deferred_rendered && !material_requires_forward(draw))
                    {
                        continue;
                    }

                    draw_with_pipeline(draw, material_is_terrain(draw) && terrain_pipeline_ != VK_NULL_HANDLE
                                                 ? terrain_pipeline_
                                                 : mesh_pipeline_);
                }

                for (const auto& draw : frame_virtual_draws_)
                {
                    if (draw.draw.mode == render_mode::wireframe)
                    {
                        if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
                            draw_virtual_with_pipeline(draw, mesh_wire_pipeline_);
                        else
                            draw_virtual_with_pipeline(draw, mesh_pipeline_);
                        continue;
                    }

                    if (material_alpha_mode_for(draw.draw) == material_alpha_mode::blend) continue;

                    if (deferred_rendered && !material_requires_forward(draw.draw)) continue;

                    draw_virtual_with_pipeline(
                        draw, material_is_terrain(draw.draw) && terrain_pipeline_ != VK_NULL_HANDLE ? terrain_pipeline_
                                                                                                    : mesh_pipeline_);
                }

                if (!deferred_rendered)
                    for (const auto& draw : frame_terrain_draws_)
                        draw_terrain_patch(command_buffer, draw, terrain_pipeline_, false);

                std::vector<const draw_mesh_event*> transparent_draws;
                for (const auto& draw : frame_draws_)
                {
                    if (draw.mode == render_mode::wireframe ||
                        material_alpha_mode_for(draw) != material_alpha_mode::blend)
                        continue;
                    transparent_draws.push_back(&draw);
                }
                std::sort(transparent_draws.begin(), transparent_draws.end(),
                          [&](const draw_mesh_event* lhs, const draw_mesh_event* rhs)
                          {
                              const auto lhs_delta = math::sub(matrix_translation(lhs->model), frame_camera_.position);
                              const auto rhs_delta = math::sub(matrix_translation(rhs->model), frame_camera_.position);
                              return math::length_squared(lhs_delta) > math::length_squared(rhs_delta);
                          });
                for (const auto* draw : transparent_draws)
                {
                    draw_with_pipeline(*draw, mesh_transparent_pipeline_ != VK_NULL_HANDLE ? mesh_transparent_pipeline_
                                                                                           : mesh_pipeline_);
                }

                // Selection is an editor overlay, not part of the material path.
                // Draw it after deferred, forward, and transparent geometry so
                // ordinary deferred objects cannot skip their highlight.
                if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
                {
                    for (const auto& draw : frame_draws_)
                        if (draw.selected && draw.mode != render_mode::wireframe && !material_is_terrain(draw))
                            draw_with_pipeline(draw, mesh_wire_pipeline_);
                    for (const auto& draw : frame_virtual_draws_)
                        if (draw.draw.selected && draw.draw.mode != render_mode::wireframe &&
                            !material_is_terrain(draw.draw))
                            draw_virtual_with_pipeline(draw, mesh_wire_pipeline_);
                }
            }

            draw_debug_overlay(command_buffer, debug_overlay_depth_mode::tested);
            cmd_end_rendering(command_buffer);
        }

        if (!render_output) return;
        transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (!ensure_output_transform_pipeline())
        {
            arc::diagnostics::warn("render.vulkan",
                                   "Output transform unavailable; the viewport retains its previous valid image");
            return;
        }
        dispatch_exposure(command_buffer);
        transition_viewport(command_buffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo output_attachment{};
        output_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        output_attachment.imageView = viewport_view_;
        output_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        output_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        output_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo output_rendering{};
        output_rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        output_rendering.renderArea.extent = {viewport_width_, viewport_height_};
        output_rendering.layerCount = 1;
        output_rendering.colorAttachmentCount = 1;
        output_rendering.pColorAttachments = &output_attachment;
        cmd_begin_rendering(command_buffer, &output_rendering);
        set_viewport_and_scissor(command_buffer);
        output_transform_push_constants output_constants{};
        output_constants.exposure_output[0] =
            frame_camera_.exposure.mode == exposure_mode::manual
                ? exposure_multiplier(frame_camera_.exposure.manual_ev100, frame_camera_.exposure.compensation_ev)
                : 1.0f;
        output_constants.exposure_output[1] = frame_camera_.exposure.mode == exposure_mode::automatic ? 1.0f : 0.0f;
        output_constants.exposure_output[2] = frame_camera_.exposure.compensation_ev;
        const auto visualization = !frame_draws_.empty()           ? frame_draws_.front().visualization
                                   : !frame_virtual_draws_.empty() ? frame_virtual_draws_.front().draw.visualization
                                                                   : mesh_visualization_mode::standard;
        output_constants.exposure_output[3] = visualization == mesh_visualization_mode::standard ? 0.0f : 1.0f;
        output_constants.post_process[0] = frame_fxaa_enabled_ ? 1.0f : 0.0f;
        output_constants.post_process[1] = 1.0f / static_cast<float>(std::max(viewport_width_, 1u));
        output_constants.post_process[2] = 1.0f / static_cast<float>(std::max(viewport_height_, 1u));
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, output_transform_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, output_transform_pipeline_layout_, 0,
                                1, &output_transform_descriptor_set_, 0, nullptr);
        vkCmdPushConstants(command_buffer, output_transform_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(output_constants), &output_constants);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        draw_debug_overlay(command_buffer, debug_overlay_depth_mode::always);
        cmd_end_rendering(command_buffer);
        record_frame_capture(command_buffer);
        transition_viewport(command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    VkInstance instance_{};
    VkSurfaceKHR surface_{};
    VkPhysicalDevice physical_device_{};
    VkDevice device_{};
    VkQueue queue_{};
    VmaAllocator allocator_{};
    std::uint32_t graphics_queue_family_{};
    render_capabilities capabilities_{};
    resolved_render_config resolved_config_{};
    vulkan_context context_{};
    vulkan_command_context command_context_{};
    descriptor_slot_pool descriptor_slots_;
    deferred_resource_releaser deferred_releases_;
    frame_allocator frame_arena_{256u * 1024u};
    pipeline_handle_cache pipeline_handles_;
    VkPipelineCache vk_pipeline_cache_{};
    gpu_buffer upload_staging_;
    void* upload_staging_mapped_{};
    std::unique_ptr<gpu_upload_arena> upload_arena_;
    VkCommandPool upload_command_pool_{};
    VkCommandBuffer upload_command_buffer_{};
    VkFence upload_fence_{};
    VkSemaphore upload_timeline_{};
    std::uint64_t upload_timeline_value_{};
    bool upload_timeline_enabled_{};
    std::uint64_t upload_frame_{};
    bool upload_batch_active_{};
    bool upload_batch_has_work_{};
    bool upload_batch_failed_{};
    static constexpr std::uint32_t max_timestamp_queries_{64};
    VkQueryPool timestamp_query_pool_{};
    float timestamp_period_{1.0f};
    std::uint32_t max_push_constant_bytes_{};
    bool push_constant_limit_warning_reported_{};
    bool timestamps_supported_{};
    std::uint32_t next_timestamp_query_{};
    std::vector<gpu_scope_record> timestamp_scopes_;
    render_backend_frame_profile last_profile_;
    std::uint64_t last_completed_frame_{};
    std::optional<render_object_pick_request> pending_pick_request_;
    render_object_pick_result last_pick_result_{};
    gpu_buffer pick_readback_buffer_;
    object_pick_readback in_flight_pick_;
    std::optional<render_frame_capture_request> pending_capture_request_;
    render_frame_capture_result last_capture_result_{};
    gpu_buffer capture_readback_buffer_;
    VkDeviceSize capture_readback_capacity_{};
    frame_capture_readback in_flight_capture_;
    std::vector<std::string> pending_debug_markers_;
    std::unordered_map<std::uint64_t, gpu_mesh> meshes_;
    std::unordered_map<std::uint64_t, gpu_virtual_mesh> virtual_meshes_;
    std::unordered_map<std::uint64_t, gpu_terrain> terrains_;
    std::unordered_map<std::uint32_t, terrain_topology> terrain_topologies_;
    std::unordered_map<std::uint64_t, gpu_texture> textures_;
    std::unordered_map<std::uint64_t, gpu_material> materials_;
    std::unordered_map<std::uint64_t, gpu_environment> environments_;
    std::unordered_set<std::uint64_t> texture_semantic_diagnostics_;
    std::vector<draw_mesh_event> frame_draws_;
    std::vector<virtual_cluster_draw> frame_virtual_draws_;
    std::vector<terrain_patch_draw> frame_terrain_draws_;
    std::vector<draw_mesh_event> frame_shadow_draws_;
    std::vector<virtual_cluster_draw> frame_virtual_shadow_draws_;
    std::vector<directional_light_event> frame_directional_lights_;
    std::vector<point_light_event> frame_point_lights_;
    std::vector<spot_light_event> frame_spot_lights_;
    std::vector<area_light_event> frame_area_lights_;
    const std::vector<area_light_event> empty_area_lights_{};
    std::vector<debug_overlay_line> frame_debug_overlay_lines_;
    std::vector<debug_overlay_triangle> frame_debug_overlay_triangles_;
    scene_lighting_data frame_lighting_;
    world_environment_data frame_environment_;
    render_camera frame_camera_;
    bool frame_camera_valid_{};
    bool frame_shadows_enabled_{true};
    bool frame_fxaa_enabled_{};
    gpu_buffer light_buffer_;
    gpu_buffer gpu_scene_visibility_buffer_;
    gpu_buffer gpu_scene_transform_buffer_;
    std::vector<gpu_scene_visibility_record> gpu_scene_visibility_mirror_;
    std::vector<gpu_scene_transform_record> gpu_scene_transform_mirror_;
    std::uint32_t gpu_scene_capacity_{};
    gpu_buffer gpu_visibility_commands_;
    gpu_buffer gpu_visibility_counters_;
    std::uint32_t gpu_visibility_capacity_{};
    VkDescriptorSetLayout gpu_visibility_descriptor_set_layout_{};
    VkDescriptorPool gpu_visibility_descriptor_pool_{};
    VkDescriptorSet gpu_visibility_descriptor_set_{};
    VkPipelineLayout gpu_visibility_pipeline_layout_{};
    VkPipeline gpu_visibility_pipeline_{};
    bool gpu_visibility_active_{};
    bool gpu_visibility_descriptors_dirty_{true};
    std::vector<gpu_buffer> shadow_uniform_buffers_;
    std::vector<debug_overlay_frame_buffer> debug_overlay_buffers_;
    std::uint32_t active_frame_index_{};
    environment_handle active_environment_;
    vulkan_shadow_atlas shadow_atlas_;
    vulkan_local_shadow_atlas local_shadow_atlas_;
    vulkan_shadow_cache shadow_cache_;
    std::unique_ptr<shadow_atlas_allocator> local_shadow_allocator_;
    std::vector<active_local_shadow> active_local_shadows_;
    std::unordered_map<std::uint64_t, std::uint64_t> local_shadow_static_signatures_;
    std::unordered_map<std::uint64_t, std::uint64_t> static_shadow_transform_hashes_;
    std::unordered_set<std::uint64_t> reported_moved_static_objects_;
    std::uint64_t shadow_resource_revision_{1};
    bool last_static_shadow_cache_hit_{};

    VkDescriptorSetLayout white_descriptor_set_layout_{};
    VkDescriptorPool white_descriptor_pool_{};
    std::vector<VkDescriptorSet> white_descriptor_sets_;
    std::vector<gpu_buffer> white_material_parameter_buffers_;
    std::vector<VkDescriptorSet> sky_descriptor_sets_;
    VkImage white_image_{};
    VmaAllocation white_allocation_{};
    VkImageView white_view_{};
    VkSampler white_sampler_{};
    VkPipelineLayout mesh_pipeline_layout_{};
    VkDescriptorSetLayout terrain_descriptor_set_layout_{};
    VkDescriptorPool terrain_descriptor_pool_{};
    VkPipelineLayout terrain_pipeline_layout_{};
    VkPipeline mesh_pipeline_{};
    VkPipeline mesh_transparent_pipeline_{};
    VkPipeline mesh_wire_pipeline_{};
    VkPipeline terrain_pipeline_{};
    VkPipeline gbuffer_pipeline_{};
    VkPipeline terrain_gbuffer_pipeline_{};
    VkDescriptorSetLayout gbuffer_descriptor_set_layout_{};
    VkDescriptorPool gbuffer_descriptor_pool_{};
    VkDescriptorSet gbuffer_descriptor_set_{};
    VkSampler gbuffer_sampler_{};
    VkPipelineLayout deferred_pipeline_layout_{};
    VkPipeline deferred_pipeline_{};
    VkDescriptorSetLayout output_transform_descriptor_set_layout_{};
    VkDescriptorPool output_transform_descriptor_pool_{};
    VkDescriptorSet output_transform_descriptor_set_{};
    VkPipelineLayout output_transform_pipeline_layout_{};
    VkPipeline output_transform_pipeline_{};
    gpu_buffer exposure_buffer_;
    VkPipelineLayout luminance_histogram_pipeline_layout_{};
    VkPipeline luminance_histogram_pipeline_{};
    VkPipelineLayout exposure_resolve_pipeline_layout_{};
    VkPipeline exposure_resolve_pipeline_{};
    bool exposure_needs_reset_{true};
    VkPipelineLayout sky_pipeline_layout_{};
    VkPipeline sky_pipeline_{};
    VkPipelineLayout shadow_pipeline_layout_{};
    VkPipeline shadow_pipeline_{};
    VkPipeline terrain_shadow_pipeline_{};
    VkPipelineLayout debug_overlay_pipeline_layout_{};
    VkPipeline debug_overlay_line_pipeline_{};
    VkPipeline debug_overlay_triangle_pipeline_{};
    VkPipeline debug_overlay_output_line_pipeline_{};
    VkPipeline debug_overlay_output_triangle_pipeline_{};
    bool wireframe_warning_reported_{};
    viewport_output_type configured_viewport_output_{viewport_output_type::native_window};

#if ARC_VULKAN_SHARED_VIEWPORT
    bool shared_viewport_supported_{};
    std::string shared_viewport_failure_;
    PFN_vkGetMemoryWin32HandlePropertiesKHR get_memory_win32_handle_properties_{};
    Microsoft::WRL::ComPtr<ID3D11Device> shared_d3d_device_;
    std::unordered_map<std::string, std::uint64_t> shared_viewport_generations_;
    std::unordered_map<std::string, shared_viewport_output> shared_viewports_;
#endif

    detail::vulkan_swapchain swapchain_{};
    bool native_swapchain_initialized_{};
    bool swapchain_rebuild_{};
    bool device_lost_{};
    std::uint32_t min_image_count_{2};
    VkFormat viewport_format_{VK_FORMAT_R16G16B16A16_SFLOAT};
    VkFormat scene_color_format_{VK_FORMAT_R16G16B16A16_SFLOAT};
    VkFormat depth_format_{VK_FORMAT_D32_SFLOAT};
    VkImage viewport_image_{};
    VmaAllocation viewport_allocation_{};
    VkImageView viewport_view_{};
    VkSampler viewport_sampler_{};
    VkImageLayout viewport_layout_{VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage viewport_depth_image_{};
    VmaAllocation viewport_depth_allocation_{};
    VkImageView viewport_depth_view_{};
    VkImageLayout viewport_depth_layout_{VK_IMAGE_LAYOUT_UNDEFINED};
    graph_image scene_color_{};
    graph_image gbuffer_albedo_{};
    graph_image gbuffer_normal_{};
    graph_image gbuffer_material_{};
    graph_image gbuffer_emissive_{};
    graph_image gbuffer_motion_{};
    graph_image gbuffer_object_id_{};
    graph_image selection_mask_{};
    std::array<graph_image, 2> hzb_history_{};
    VkSampler hzb_sampler_{};
    VkDescriptorSetLayout hzb_descriptor_set_layout_{};
    VkDescriptorPool hzb_descriptor_pool_{};
    std::vector<VkDescriptorSet> hzb_descriptor_sets_;
    VkPipelineLayout hzb_pipeline_layout_{};
    VkPipeline hzb_pipeline_{};
    std::uint32_t hzb_mip_count_{};
    bool hzb_history_valid_{};
    std::array<graph_image, 2> temporal_dilated_motion_{};
    std::array<graph_image, 2> temporal_reactive_{};
    std::array<graph_image, 2> temporal_disocclusion_{};
    std::array<graph_image, 2> temporal_color_history_{};
    std::array<graph_image, 2> temporal_depth_history_{};
    std::array<graph_image, 2> temporal_moments_history_{};
    std::array<graph_image, 2> temporal_confidence_history_{};
    std::array<graph_image, 2> temporal_sharpened_{};
    VkDescriptorSetLayout temporal_velocity_descriptor_layout_{};
    VkDescriptorSetLayout temporal_mask_descriptor_layout_{};
    VkDescriptorSetLayout temporal_resolve_descriptor_layout_{};
    VkDescriptorSetLayout temporal_sharpen_descriptor_layout_{};
    VkDescriptorPool temporal_descriptor_pool_{};
    std::array<VkDescriptorSet, 2> temporal_velocity_sets_{};
    std::array<VkDescriptorSet, 2> temporal_mask_sets_{};
    std::array<VkDescriptorSet, 2> temporal_resolve_sets_{};
    std::array<VkDescriptorSet, 2> temporal_sharpen_sets_{};
    VkPipelineLayout temporal_velocity_pipeline_layout_{};
    VkPipelineLayout temporal_mask_pipeline_layout_{};
    VkPipelineLayout temporal_resolve_pipeline_layout_{};
    VkPipelineLayout temporal_sharpen_pipeline_layout_{};
    VkPipeline temporal_velocity_pipeline_{};
    VkPipeline temporal_mask_pipeline_{};
    VkPipeline temporal_resolve_pipeline_{};
    VkPipeline temporal_sharpen_pipeline_{};
    std::uint32_t temporal_input_width_{};
    std::uint32_t temporal_input_height_{};
    std::uint32_t temporal_output_width_{};
    std::uint32_t temporal_output_height_{};
    bool temporal_history_valid_{};
    bool temporal_resources_initialized_{};
    VkImageView temporal_output_view_{};
    std::uint32_t viewport_width_{};
    std::uint32_t viewport_height_{};
    std::uint32_t output_viewport_width_{};
    std::uint32_t output_viewport_height_{};
};

bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name)
{
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension)
                       { return std::strcmp(extension.extensionName, name) == 0; });
}

std::vector<const char*> make_c_strings(const std::vector<std::string>& values)
{
    std::vector<const char*> result;
    result.reserve(values.size());
    for (const auto& value : values)
        result.push_back(value.c_str());
    return result;
}

void append_unique_extension(std::vector<std::string>& extensions, const char* name)
{
    if (std::find(extensions.begin(), extensions.end(), name) == extensions.end()) extensions.emplace_back(name);
}

std::uint32_t find_graphics_queue_family(VkPhysicalDevice physical_device, VkSurfaceKHR surface = VK_NULL_HANDLE)
{
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());

    for (std::uint32_t index = 0; index < count; ++index)
    {
        if ((families[index].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) !=
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            continue;

        if (surface != VK_NULL_HANDLE)
        {
            VkBool32 present_supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, index, surface, &present_supported);
            if (present_supported != VK_TRUE) continue;
        }

        return index;
    }

    return UINT32_MAX;
}

bool supports_device_extensions(VkPhysicalDevice physical_device, const std::vector<std::string>& required_extensions)
{
    std::uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data());

    for (const auto& required : required_extensions)
    {
        if (!has_extension(extensions, required.c_str())) return false;
    }

    return true;
}

render_capabilities query_capabilities(VkPhysicalDevice physical_device, VkSurfaceKHR surface)
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);

    std::uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data());

    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering{};
    dynamic_rendering.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    VkPhysicalDeviceSynchronization2Features synchronization2{};
    synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer{};
    descriptor_buffer.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader{};
    mesh_shader.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing{};
    ray_tracing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragment_shading_rate{};
    fragment_shading_rate.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkBaseOutStructure* tail = reinterpret_cast<VkBaseOutStructure*>(&features);
    auto append_feature = [&](auto& feature)
    {
        tail->pNext = reinterpret_cast<VkBaseOutStructure*>(&feature);
        tail = reinterpret_cast<VkBaseOutStructure*>(&feature);
    };
    const bool vulkan12_or_newer = properties.apiVersion >= VK_API_VERSION_1_2;
    const bool vulkan13_or_newer = properties.apiVersion >= VK_API_VERSION_1_3;
    if (vulkan12_or_newer) append_feature(vulkan12);
    if (vulkan13_or_newer || has_extension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
        append_feature(dynamic_rendering);
    if (vulkan13_or_newer || has_extension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME))
        append_feature(synchronization2);
    if (has_extension(extensions, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) append_feature(descriptor_buffer);
    if (has_extension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME)) append_feature(mesh_shader);
    if (has_extension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) append_feature(ray_tracing);
    if (has_extension(extensions, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) append_feature(fragment_shading_rate);
    vkGetPhysicalDeviceFeatures2(physical_device, &features);

    VkPhysicalDeviceDriverProperties driver_properties{};
    driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceDescriptorIndexingProperties descriptor_indexing_properties{};
    descriptor_indexing_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
    if (vulkan12_or_newer)
    {
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &driver_properties;
        driver_properties.pNext = &descriptor_indexing_properties;
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    }

    VkPhysicalDeviceMemoryProperties2 memory_properties{};
    memory_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT memory_budget{};
    memory_budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    const bool has_memory_budget = has_extension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (has_memory_budget) memory_properties.pNext = &memory_budget;
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &memory_properties);

    render_capabilities capabilities{};
    capabilities.backend = render_backend_type::vulkan;
    capabilities.api_major = VK_VERSION_MAJOR(properties.apiVersion);
    capabilities.api_minor = VK_VERSION_MINOR(properties.apiVersion);
    capabilities.adapter_name = properties.deviceName;
    capabilities.driver_name = driver_properties.driverName;
    capabilities.vendor_id = properties.vendorID;
    capabilities.device_id = properties.deviceID;
    capabilities.driver_version = properties.driverVersion;
    capabilities.discrete_gpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    capabilities.integrated_gpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    capabilities.max_texture_dimension_2d = properties.limits.maxImageDimension2D;
    capabilities.max_color_attachments = properties.limits.maxColorAttachments;
    capabilities.max_compute_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;
    for (std::uint32_t heap = 0; heap < memory_properties.memoryProperties.memoryHeapCount; ++heap)
    {
        const auto bytes = memory_properties.memoryProperties.memoryHeaps[heap].size;
        if ((memory_properties.memoryProperties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
        {
            capabilities.dedicated_video_memory += bytes;
            capabilities.memory_budget += has_memory_budget ? memory_budget.heapBudget[heap] : bytes;
            capabilities.memory_usage += has_memory_budget ? memory_budget.heapUsage[heap] : 0;
        }
        else
        {
            capabilities.shared_system_memory += bytes;
        }
    }

    std::uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_count, queues.data());
    for (std::uint32_t index = 0; index < queue_count; ++index)
    {
        capabilities.graphics_queue |= (queues[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        capabilities.compute_queue |= (queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        capabilities.transfer_queue |= (queues[index].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
        capabilities.gpu_timestamps |= queues[index].timestampValidBits > 0;
        if (surface != VK_NULL_HANDLE)
        {
            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, index, surface, &supported);
            capabilities.presentation |= supported == VK_TRUE;
        }
    }
    if (surface == VK_NULL_HANDLE) capabilities.presentation = true;

    capabilities.draw_indirect = properties.limits.maxDrawIndirectCount > 0;
    capabilities.draw_indirect_count =
        vulkan12_or_newer || has_extension(extensions, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
    capabilities.compute_shaders = capabilities.compute_queue;
    capabilities.storage_buffers = properties.limits.maxStorageBufferRange >= 128u * 1024u * 1024u;
    capabilities.storage_images = properties.limits.maxPerStageDescriptorStorageImages > 0;
    capabilities.shader_draw_parameters = properties.apiVersion >= VK_API_VERSION_1_1;
    capabilities.gpu_scene_indirect =
        capabilities.compute_shaders && capabilities.storage_buffers && capabilities.draw_indirect;
    VkFormatProperties hzb_format{};
    vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R32G32_SFLOAT, &hzb_format);
    const auto required_hzb_features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    capabilities.hzb_occlusion = capabilities.compute_shaders && capabilities.storage_images &&
                                 (hzb_format.optimalTilingFeatures & required_hzb_features) == required_hzb_features;
    const auto supports_storage_sampled = [&](VkFormat format)
    {
        VkFormatProperties format_properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &format_properties);
        const auto required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        return (format_properties.optimalTilingFeatures & required) == required;
    };
    capabilities.temporal_resolve =
        capabilities.compute_shaders && capabilities.storage_images &&
        supports_storage_sampled(VK_FORMAT_R16G16B16A16_SFLOAT) && supports_storage_sampled(VK_FORMAT_R16G16_SFLOAT) &&
        supports_storage_sampled(VK_FORMAT_R32_SFLOAT) && supports_storage_sampled(VK_FORMAT_R8_UNORM);
    capabilities.temporal_upscale = capabilities.temporal_resolve;
    // FXAA is implemented as the final linear-LDR stage fused into the output
    // transform and selected by the executable graph's FXAA pass.
    capabilities.fxaa = true;
    capabilities.virtual_geometry_compute = false;
    capabilities.virtual_geometry_mesh_shader = false;
    capabilities.virtual_geometry_streaming = false;
    capabilities.sampler_anisotropy = features.features.samplerAnisotropy == VK_TRUE;
    capabilities.texture_compression_bc = features.features.textureCompressionBC == VK_TRUE;
    capabilities.synchronization2 = synchronization2.synchronization2 == VK_TRUE;
    capabilities.timeline_semaphores = vulkan12.timelineSemaphore == VK_TRUE;
    capabilities.dynamic_rendering = dynamic_rendering.dynamicRendering == VK_TRUE;
    constexpr std::uint32_t minimum_bindless_sampled_images = 4096u;
    constexpr std::uint32_t minimum_bindless_samplers = 256u;
    const bool complete_descriptor_indexing =
        vulkan12.descriptorIndexing == VK_TRUE && vulkan12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
        vulkan12.runtimeDescriptorArray == VK_TRUE && vulkan12.descriptorBindingPartiallyBound == VK_TRUE &&
        vulkan12.descriptorBindingVariableDescriptorCount == VK_TRUE &&
        vulkan12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE &&
        descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindSampledImages >=
            minimum_bindless_sampled_images &&
        descriptor_indexing_properties.maxPerStageDescriptorUpdateAfterBindSampledImages >=
            minimum_bindless_sampled_images &&
        descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindSamplers >= minimum_bindless_samplers;
    capabilities.descriptor_indexing = complete_descriptor_indexing;
    capabilities.bindless_sampled_images = complete_descriptor_indexing;
    capabilities.bindless_samplers = complete_descriptor_indexing;
    // These executable facts remain false until the shared heap/material-table
    // graphics pipelines replace ARC's classic per-resource bindings.
    capabilities.bindless_material_tables = false;
    capabilities.bindless_geometry_tables = false;
    capabilities.gpu_visibility_compaction = false;
    capabilities.gpu_transparent_sorting = false;
    capabilities.gpu_skinning = false;
    capabilities.gpu_terrain_traversal = false;
    capabilities.descriptor_buffer = descriptor_buffer.descriptorBuffer == VK_TRUE;
    capabilities.mesh_shaders = mesh_shader.meshShader == VK_TRUE;
    // Capability facts describe executable ARC paths. Ray-query acceleration structures and
    // their graph execution are enabled together by the lighting backend; a driver extension
    // alone must never select the hybrid path.
    capabilities.screen_space_indirect_lighting = false;
    capabilities.surface_cache = false;
    capabilities.radiance_cache = false;
    capabilities.software_ray_tracing = false;
    capabilities.hardware_ray_query = false;
    capabilities.ray_tracing = false;
    capabilities.sparse_resources = features.features.sparseBinding == VK_TRUE;
    capabilities.variable_rate_shading = fragment_shading_rate.pipelineFragmentShadingRate == VK_TRUE;
    capabilities.fill_mode_non_solid = features.features.fillModeNonSolid == VK_TRUE;
    return capabilities;
}

bool supports_required_attachment_formats(VkPhysicalDevice physical_device)
{
    const auto supports = [&](VkFormat format, VkFormatFeatureFlags features)
    {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        return (properties.optimalTilingFeatures & features) == features;
    };
    return supports(VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
           supports(VK_FORMAT_R16G16_SFLOAT,
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
           supports(VK_FORMAT_R32_UINT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
           supports(VK_FORMAT_D32_SFLOAT,
                    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
}

bool supports_required_features(const render_capabilities& capabilities, VkPhysicalDevice physical_device)
{
    const bool vulkan12 = capabilities.api_major > 1 || (capabilities.api_major == 1 && capabilities.api_minor >= 2);
    return vulkan12 && capabilities.graphics_queue && capabilities.compute_queue && capabilities.presentation &&
           capabilities.dynamic_rendering && capabilities.max_color_attachments >= 5 &&
           supports_required_attachment_formats(physical_device);
}

std::uint64_t adapter_score(const render_capabilities& capabilities)
{
    std::uint64_t score = capabilities.discrete_gpu     ? 1'000'000ull
                          : capabilities.integrated_gpu ? 500'000ull
                                                        : 100'000ull;
    score += std::min<std::uint64_t>(capabilities.memory_budget / (1024ull * 1024ull), 250'000ull);
    score += capabilities.timeline_semaphores ? 10'000ull : 0ull;
    score += capabilities.synchronization2 ? 10'000ull : 0ull;
    score += capabilities.descriptor_indexing ? 5'000ull : 0ull;
    return score;
}

bool instance_extension_available(const char* name)
{
    std::uint32_t extension_count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr) != VK_SUCCESS) return false;

    std::vector<VkExtensionProperties> extensions(extension_count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data()) != VK_SUCCESS)
        return false;

    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension)
                       { return std::strcmp(extension.extensionName, name) == 0; });
}

} // namespace

bool vulkan_loader_available() noexcept
{
    return volkInitialize() == VK_SUCCESS;
}

render_backend_create_result create_vulkan_backend(const vulkan_backend_config& config)
{
    if (volkInitialize() != VK_SUCCESS)
        return render_backend_create_result::failure(
            {render_backend_create_error_code::loader_unavailable, "failed to initialize Vulkan loader"});

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ARC";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "ARC";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    auto requested_instance_extensions = config.instance_extensions;
    if (instance_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME) &&
        std::find(requested_instance_extensions.begin(), requested_instance_extensions.end(),
                  VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == requested_instance_extensions.end())
    {
        requested_instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    const auto instance_extensions = make_c_strings(requested_instance_extensions);

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount = static_cast<std::uint32_t>(instance_extensions.size());
    instance_info.ppEnabledExtensionNames = instance_extensions.data();

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS)
        return render_backend_create_result::failure(
            {render_backend_create_error_code::instance_creation_failed, "failed to create Vulkan instance"});

    volkLoadInstance(instance);

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (config.create_surface)
    {
        if (!config.create_surface(instance, vkGetInstanceProcAddr, &surface, config.surface_user_data) ||
            surface == VK_NULL_HANDLE)
        {
            vkDestroyInstance(instance, nullptr);
            return render_backend_create_result::failure({render_backend_create_error_code::surface_creation_failed,
                                                          "failed to create Vulkan presentation surface"});
        }
    }

    std::uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);
    if (physical_device_count == 0)
    {
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return render_backend_create_result::failure(
            {render_backend_create_error_code::adapter_unavailable, "no Vulkan physical devices found"});
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data());

    VkPhysicalDevice selected_device = VK_NULL_HANDLE;
    render_capabilities selected_capabilities{};
    std::uint32_t graphics_queue_family = UINT32_MAX;
    std::uint64_t selected_score{};
    std::vector<std::string> selected_device_extensions;

    auto required_device_extensions = config.device_extensions;
    if (surface != VK_NULL_HANDLE) append_unique_extension(required_device_extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#if ARC_VULKAN_SHARED_VIEWPORT
    if (config.viewport_output == viewport_output_type::shared_texture)
    {
        append_unique_extension(required_device_extensions, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        append_unique_extension(required_device_extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        append_unique_extension(required_device_extensions, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        append_unique_extension(required_device_extensions, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    }
#endif

    for (std::uint32_t adapter_index = 0; adapter_index < physical_devices.size(); ++adapter_index)
    {
        if (config.adapter_index && *config.adapter_index != adapter_index) continue;

        const auto physical_device = physical_devices[adapter_index];
        const auto capabilities = query_capabilities(physical_device, surface);
        const auto queue_family = find_graphics_queue_family(physical_device, surface);
        auto candidate_extensions = required_device_extensions;
        if (capabilities.api_major == 1 && capabilities.api_minor < 3)
            append_unique_extension(candidate_extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);

        std::string rejection;
        if (queue_family == UINT32_MAX)
            rejection = "no combined graphics/compute queue with required presentation support";
        else if (!supports_required_features(capabilities, physical_device))
            rejection = "missing Vulkan 1.2 baseline, dynamic rendering, limits, or required attachment formats";
        else if (!supports_device_extensions(physical_device, candidate_extensions))
            rejection = "missing required device extensions";

        if (!rejection.empty())
        {
            arc::diagnostics::warn("render.vulkan", "Rejected adapter " + std::to_string(adapter_index) + " (" +
                                                        capabilities.adapter_name + "): " + rejection);
            continue;
        }

        const auto score = adapter_score(capabilities);
        if (selected_device == VK_NULL_HANDLE || score > selected_score)
        {
            selected_device = physical_device;
            selected_capabilities = capabilities;
            graphics_queue_family = queue_family;
            selected_score = score;
            selected_device_extensions = std::move(candidate_extensions);
        }
    }

    if (selected_device == VK_NULL_HANDLE)
    {
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return render_backend_create_result::failure(
            {render_backend_create_error_code::adapter_unavailable,
             "no Vulkan 1.2 graphics/compute device with required attachment formats and dynamic rendering found"});
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = graphics_queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering{};
    dynamic_rendering.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dynamic_rendering.dynamicRendering = VK_TRUE;

    const bool enable_optional_features = !config.force_disable_optional_features;
    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.timelineSemaphore =
        enable_optional_features && selected_capabilities.timeline_semaphores ? VK_TRUE : VK_FALSE;
    const auto enable_descriptor_indexing = enable_optional_features && selected_capabilities.descriptor_indexing;
    vulkan12.descriptorIndexing = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.shaderSampledImageArrayNonUniformIndexing = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.runtimeDescriptorArray = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.descriptorBindingPartiallyBound = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.descriptorBindingVariableDescriptorCount = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.descriptorBindingSampledImageUpdateAfterBind = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceSynchronization2Features synchronization2{};
    synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    synchronization2.synchronization2 =
        enable_optional_features && selected_capabilities.synchronization2 ? VK_TRUE : VK_FALSE;
    dynamic_rendering.pNext = &vulkan12;
    vulkan12.pNext = &synchronization2;

    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.fillModeNonSolid = selected_capabilities.fill_mode_non_solid ? VK_TRUE : VK_FALSE;
    enabled_features.samplerAnisotropy =
        enable_optional_features && selected_capabilities.sampler_anisotropy ? VK_TRUE : VK_FALSE;

    if (synchronization2.synchronization2 == VK_TRUE && selected_capabilities.api_major == 1 &&
        selected_capabilities.api_minor < 3)
    {
        append_unique_extension(selected_device_extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
    const auto device_extension_names = make_c_strings(selected_device_extensions);

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &dynamic_rendering;
    device_info.pEnabledFeatures = &enabled_features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<std::uint32_t>(device_extension_names.size());
    device_info.ppEnabledExtensionNames = device_extension_names.data();

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(selected_device, &device_info, nullptr, &device) != VK_SUCCESS)
    {
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return render_backend_create_result::failure(
            {render_backend_create_error_code::device_creation_failed, "failed to create Vulkan device"});
    }

    volkLoadDevice(device);
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, graphics_queue_family, 0, &queue);

    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.instance = instance;
    allocator_info.physicalDevice = selected_device;
    allocator_info.device = device;
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;

    VmaAllocator allocator = VK_NULL_HANDLE;
    if (vmaCreateAllocator(&allocator_info, &allocator) != VK_SUCCESS)
    {
        vkDestroyDevice(device, nullptr);
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return render_backend_create_result::failure(
            {render_backend_create_error_code::memory_allocator_creation_failed,
             "failed to create Vulkan memory allocator"});
    }

    arc::diagnostics::info(
        "render.vulkan",
        "Selected adapter " + selected_capabilities.adapter_name + " (Vulkan " +
            std::to_string(selected_capabilities.api_major) + "." + std::to_string(selected_capabilities.api_minor) +
            ", " + std::to_string(selected_capabilities.memory_budget / (1024ull * 1024ull)) + " MiB budget)");
    if (config.force_disable_optional_features)
        arc::diagnostics::info("render.vulkan",
                               "Developer compatibility override left all non-required Vulkan features disabled");
    arc::diagnostics::info("render.vulkan", "Created Vulkan backend");
    return render_backend_create_result::success(
        std::make_unique<vulkan_render_backend>(instance, surface, selected_device, device, queue, allocator,
                                                graphics_queue_family, selected_capabilities, config.viewport_output));
}

} // namespace arc::render::vulkan
