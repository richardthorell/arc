#include "vulkan_backend_internal.h"

namespace arc::render::vulkan::backend_detail
{
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

std::uint64_t gpu_scene_key(gpu_scene_instance_handle handle) noexcept
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

vulkan_render_backend::vulkan_render_backend(VkInstance instance, VkSurfaceKHR surface,
                                             VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                                             VmaAllocator allocator, std::uint32_t graphics_queue_family,
                                             render_capabilities capabilities, viewport_output_type viewport_output)
    : instance_(instance), surface_(surface), physical_device_(physical_device), device_(device), queue_(queue),
      allocator_(allocator), graphics_queue_family_(graphics_queue_family), capabilities_(capabilities),
      configured_viewport_output_(viewport_output)
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    max_indirect_draw_count_ = properties.limits.maxDrawIndirectCount;
    if (configured_viewport_output_ == viewport_output_type::shared_texture)
    {
#if ARC_VULKAN_SHARED_VIEWPORT
        viewport_format_ = VK_FORMAT_B8G8R8A8_UNORM;
#else
        arc::diagnostics::warn("render.vulkan", "shared viewport output is only available in Windows editor builds");
#endif
    }
    create_support_objects();
#if ARC_VULKAN_SHARED_VIEWPORT
    query_shared_viewport_support();
#endif
}

vulkan_render_backend::~vulkan_render_backend()
{
    shutdown_surface_presenter();
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    destroy_temporal_resources();
    destroy_hzb_resources();
    destroy_mesh_pipeline();
    destroy_virtual_shadow_resources(virtual_shadow_resources_);
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
    destroy_gpu_resource_tables();
    destroy_gpu_visibility_resources();
    destroy_texture_feedback_resources();
    destroy_virtual_texture_resources();
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

render_backend_type vulkan_render_backend::type() const noexcept
{
    return render_backend_type::vulkan;
}

const render_capabilities& vulkan_render_backend::capabilities() const noexcept
{
    return capabilities_;
}

void vulkan_render_backend::configure(const resolved_render_config& config)
{
    const float previous_scale = resolved_config_.render_scale;
    const std::uint32_t previous_local_shadow_atlas = resolved_config_.local_shadow_atlas_resolution;
    const std::uint64_t previous_virtual_shadow_budget = resolved_config_.virtual_shadow_budget_bytes;
    resolved_config_ = config;
    if (!local_shadow_allocator_ || previous_local_shadow_atlas != config.local_shadow_atlas_resolution)
    {
        local_shadow_allocator_ =
            std::make_unique<shadow_atlas_allocator>(config.local_shadow_atlas_resolution, 128u, 2u);
        local_shadow_static_signatures_.clear();
    }
    if (config.features.virtual_shadow_maps)
    {
        if (!virtual_shadow_cache_ || previous_virtual_shadow_budget != config.virtual_shadow_budget_bytes)
        {
            virtual_shadow_cache_ =
                std::make_unique<virtual_shadow_cache>(config.virtual_shadow_budget_bytes, capabilities_.memory_budget,
                                                       virtual_shadow_depth_format::d16_unorm);
            virtual_shadow_lights_.clear();
            retire_virtual_shadow_resources();
        }
        if (!ensure_virtual_shadow_resources())
        {
            resolved_config_.features.virtual_shadow_maps = false;
            resolved_config_.features.virtual_shadow_virtual_geometry = false;
            resolved_config_.fallback_reasons.emplace_back(
                "Vulkan could not allocate the virtual shadow page pool; using conventional shadows");
        }
    }
    else if (virtual_shadow_resources_.static_image != VK_NULL_HANDLE)
    {
        retire_virtual_shadow_resources();
        virtual_shadow_cache_.reset();
        virtual_shadow_lights_.clear();
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
    last_profile_.configuration = resolved_config_;
    if (native_swapchain_initialized_ && previous_scale != config.render_scale && output_viewport_width_ > 0 &&
        output_viewport_height_ > 0)
    {
        ensure_viewport(scaled_dimension(output_viewport_width_), scaled_dimension(output_viewport_height_));
    }
}

void vulkan_render_backend::resize_viewport(std::uint32_t width, std::uint32_t height)
{
    output_viewport_width_ = width;
    output_viewport_height_ = height;
    if (native_swapchain_initialized_ && width > 0 && height > 0)
        ensure_viewport(scaled_dimension(width), scaled_dimension(height));
}

render_viewport_texture vulkan_render_backend::viewport_texture() const noexcept
{
    // Native editor presentation owns the surface directly. The old opaque
    // legacy texture handle is intentionally no longer exposed.
    return {};
}

render_backend_frame_profile vulkan_render_backend::last_frame_profile() const
{
    return last_profile_;
}

texture_feedback_readback vulkan_render_backend::take_texture_feedback()
{
    auto result = std::move(completed_texture_feedback_);
    completed_texture_feedback_ = {};
    return result;
}

std::vector<texture_stream_upload_result> vulkan_render_backend::take_texture_stream_upload_results()
{
    auto result = std::move(completed_texture_upload_results_);
    completed_texture_upload_results_.clear();
    return result;
}

virtual_geometry_feedback_readback vulkan_render_backend::take_virtual_geometry_feedback()
{
    auto result = std::move(completed_virtual_geometry_feedback_);
    completed_virtual_geometry_feedback_ = {};
    return result;
}

std::vector<virtual_geometry_page_upload_result> vulkan_render_backend::take_virtual_geometry_page_upload_results()
{
    auto result = std::move(completed_virtual_geometry_upload_results_);
    completed_virtual_geometry_upload_results_.clear();
    return result;
}

void vulkan_render_backend::request_object_pick(render_object_pick_request request)
{
    pending_pick_request_ = request;
}

render_object_pick_result vulkan_render_backend::last_object_pick() const
{
    return last_pick_result_;
}

void vulkan_render_backend::request_frame_capture(const render_frame_capture_request& request)
{
    if (request.capture_id == 0) return;
    pending_capture_request_ = request;
}

render_frame_capture_result vulkan_render_backend::last_frame_capture() const
{
    return last_capture_result_;
}

surface_frame_result vulkan_render_backend::present_surface_frame(std::uint32_t width, std::uint32_t height)
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

bool vulkan_render_backend::render_native_viewport_frame(std::uint32_t width, std::uint32_t height,
                                                         std::string& message)
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
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, graphics_queue_family_, surface_, &present_supported);
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
    collect_texture_mip_feedback(swapchain_.frame_index);
    collect_gpu_visibility_feedback(swapchain_.frame_index);
    collect_gpu_terrain_feedback(swapchain_.frame_index);
    collect_virtual_geometry_feedback(swapchain_.frame_index);
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
    vkCmdPipelineBarrier(frame->command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &swapchain_to_transfer);

    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1] = {static_cast<std::int32_t>(viewport_width_), static_cast<std::int32_t>(viewport_height_), 1};
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
    vkCmdPipelineBarrier(frame->command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &swapchain_to_present);

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

void vulkan_render_backend::shutdown_surface_presenter() noexcept
{
    if (!native_swapchain_initialized_ && !swapchain_.valid()) return;
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    destroy_viewport();
    swapchain_.destroy(device_);
    native_swapchain_initialized_ = false;
    swapchain_rebuild_ = false;
}

std::uint32_t vulkan_render_backend::scaled_dimension(std::uint32_t value) const noexcept
{
    return std::max(1u,
                    static_cast<std::uint32_t>(std::round(static_cast<float>(value) * resolved_config_.render_scale)));
}

void vulkan_render_backend::wait_for_in_flight_frames() const
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
void vulkan_render_backend::query_shared_viewport_support()
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
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
bool vulkan_render_backend::create_shared_d3d_device()
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
        if (SUCCEEDED(candidate->GetDesc1(&descriptor)) && descriptor.AdapterLuid.HighPart == vulkan_luid.HighPart &&
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
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
std::uint32_t vulkan_render_backend::shared_memory_type(std::uint32_t type_bits) const noexcept
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
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
bool vulkan_render_backend::create_shared_output_slots(shared_viewport_output& output)
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
        if (fence_result != VK_SUCCESS) return fail_vk("vkCreateFence for shared viewport frame failed", fence_result);
        slot.state = shared_viewport_frame_state::available;
    }
    return true;
}
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
void vulkan_render_backend::poll_shared_output_fences(shared_viewport_output& output)
{
    for (auto& slot : output.slots)
        if (slot.state == shared_viewport_frame_state::rendering && vkGetFenceStatus(device_, slot.fence) == VK_SUCCESS)
        {
            slot.state = shared_viewport_frame_state::ready;
            last_completed_frame_ = std::max(last_completed_frame_, slot.frame_id);
        }
}
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
void vulkan_render_backend::wait_for_shared_output(shared_viewport_output& output)
{
    for (auto& slot : output.slots)
        if (slot.fence != VK_NULL_HANDLE && slot.state == shared_viewport_frame_state::rendering)
        {
            vkWaitForFences(device_, 1, &slot.fence, VK_TRUE, UINT64_MAX);
            slot.state = shared_viewport_frame_state::ready;
        }
}
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
surface_frame_result vulkan_render_backend::render_shared_viewport_frame(shared_viewport_output& output,
                                                                         shared_viewport_slot& slot)
{
    output_viewport_width_ = output.width;
    output_viewport_height_ = output.height;
    const auto slot_index = static_cast<std::uint32_t>(&slot - output.slots.data());
    active_frame_index_ = slot_index;
    ensure_viewport(scaled_dimension(output.width), scaled_dimension(output.height));
    if (viewport_image_ == VK_NULL_HANDLE)
        return surface_frame_result::failure(
            {.code = surface_frame_error_code::backend_failure, .message = "viewport render target is unavailable"});
    collect_texture_mip_feedback(slot_index);
    collect_gpu_visibility_feedback(slot_index);
    collect_gpu_terrain_feedback(slot_index);
    collect_virtual_geometry_feedback(slot_index);
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
        return surface_frame_result::failure(
            {.code = surface_frame_error_code::backend_failure, .message = "failed to begin shared viewport frame"});

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
    vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &destination);
    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1] = {static_cast<std::int32_t>(viewport_width_), static_cast<std::int32_t>(viewport_height_), 1};
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
    vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &destination);
    end_debug_label(slot.command_buffer);
    if (vkEndCommandBuffer(slot.command_buffer) != VK_SUCCESS)
        return surface_frame_result::failure(
            {.code = surface_frame_error_code::backend_failure, .message = "failed to record shared viewport frame"});
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
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
void vulkan_render_backend::retire_shared_output(shared_viewport_output& output, bool preserve_identity) noexcept
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
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
void vulkan_render_backend::destroy_all_shared_viewports() noexcept
{
    for (auto& [_, output] : shared_viewports_)
    {
        wait_for_shared_output(output);
        retire_shared_output(output, false);
    }
    shared_viewports_.clear();
}
#endif

math::vector4f vulkan_render_backend::cluster_debug_color(std::uint32_t cluster_index) noexcept
{
    const std::uint32_t hash = cluster_index * 747796405u + 2891336453u;
    const float r = static_cast<float>((hash >> 0u) & 0xffu) / 255.0f;
    const float g = static_cast<float>((hash >> 8u) & 0xffu) / 255.0f;
    const float b = static_cast<float>((hash >> 16u) & 0xffu) / 255.0f;
    return {0.25f + r * 0.75f, 0.25f + g * 0.75f, 0.25f + b * 0.75f, 1.0f};
}

void vulkan_render_backend::create_support_objects()
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

void vulkan_render_backend::destroy_support_objects() noexcept
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

void vulkan_render_backend::retire_completed_resources()
{
    deferred_releases_.collect(last_completed_frame_);
    collect_texture_feedback_slots();
    frame_arena_.reset();
}

void vulkan_render_backend::begin_debug_label(VkCommandBuffer command_buffer, std::string_view name,
                                              const std::array<float, 4>& color) const
{
    if (vkCmdBeginDebugUtilsLabelEXT == nullptr || name.empty()) return;

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name.data();
    std::copy(color.begin(), color.end(), label.color);
    vkCmdBeginDebugUtilsLabelEXT(command_buffer, &label);
}

void vulkan_render_backend::insert_debug_label(VkCommandBuffer command_buffer, std::string_view name,
                                               const std::array<float, 4>& color) const
{
    if (vkCmdInsertDebugUtilsLabelEXT == nullptr || name.empty()) return;

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name.data();
    std::copy(color.begin(), color.end(), label.color);
    vkCmdInsertDebugUtilsLabelEXT(command_buffer, &label);
}

void vulkan_render_backend::end_debug_label(VkCommandBuffer command_buffer) const
{
    if (vkCmdEndDebugUtilsLabelEXT != nullptr) vkCmdEndDebugUtilsLabelEXT(command_buffer);
}

void vulkan_render_backend::reset_timestamp_queries(VkCommandBuffer command_buffer)
{
    next_timestamp_query_ = 0;
    timestamp_scopes_.clear();
    if (timestamp_query_pool_ != VK_NULL_HANDLE)
        vkCmdResetQueryPool(command_buffer, timestamp_query_pool_, 0, max_timestamp_queries_);
}

std::uint32_t vulkan_render_backend::begin_gpu_scope(VkCommandBuffer command_buffer, std::string_view name)
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

void vulkan_render_backend::end_gpu_scope(VkCommandBuffer command_buffer, std::uint32_t end_query)
{
    if (timestamp_query_pool_ != VK_NULL_HANDLE && end_query != UINT32_MAX)
        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamp_query_pool_, end_query);
    end_debug_label(command_buffer);
}

void vulkan_render_backend::collect_timestamp_results()
{
    if (timestamp_query_pool_ == VK_NULL_HANDLE || timestamp_scopes_.empty()) return;

    std::array<std::uint64_t, max_timestamp_queries_> values{};
    const VkResult result =
        vkGetQueryPoolResults(device_, timestamp_query_pool_, 0, max_timestamp_queries_, sizeof(values), values.data(),
                              sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
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

void vulkan_render_backend::collect_object_pick_result()
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

void vulkan_render_backend::collect_frame_capture_result()
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

bool vulkan_render_backend::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage,
                                          gpu_buffer& out)
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

bool vulkan_render_backend::submit_upload_commands(VkCommandBuffer command_buffer)
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

void vulkan_render_backend::destroy_upload_objects() noexcept
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

bool vulkan_render_backend::begin_upload_batch()
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

upload_allocation vulkan_render_backend::reserve_upload(VkDeviceSize size, std::size_t alignment)
{
    if (!begin_upload_batch()) return {};

    auto allocation = upload_arena_->try_allocate(static_cast<std::size_t>(size), alignment);
    if (allocation) return allocation;

    if (!flush_upload_batch() || !begin_upload_batch()) return {};
    return upload_arena_->try_allocate(static_cast<std::size_t>(size), alignment);
}

bool vulkan_render_backend::flush_upload_batch()
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

void vulkan_render_backend::destroy_buffer(gpu_buffer& value) noexcept
{
    if (value.buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(allocator_, value.buffer, value.allocation);
        value.buffer = VK_NULL_HANDLE;
        value.allocation = VK_NULL_HANDLE;
    }
}

bool vulkan_render_backend::ensure_pick_readback_buffer()
{
    if (pick_readback_buffer_.buffer != VK_NULL_HANDLE) return true;
    return create_buffer(sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                         pick_readback_buffer_);
}

bool vulkan_render_backend::ensure_capture_readback_buffer(VkDeviceSize required_size)
{
    if (capture_readback_buffer_.buffer != VK_NULL_HANDLE && capture_readback_capacity_ >= required_size) return true;
    if (in_flight_capture_.active) return false;
    destroy_buffer(capture_readback_buffer_);
    capture_readback_capacity_ = 0;
    if (!create_buffer(required_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                       capture_readback_buffer_))
        return false;
    capture_readback_capacity_ = required_size;
    return true;
}

bool vulkan_render_backend::capture_channel_requested(const render_frame_capture_request& request,
                                                      render_capture_channel channel)
{
    return std::ranges::find(request.channels, channel) != request.channels.end();
}

VkDeviceSize vulkan_render_backend::align_capture_offset(VkDeviceSize value) noexcept
{
    constexpr VkDeviceSize alignment = 256;
    return (value + alignment - 1u) & ~(alignment - 1u);
}

std::optional<std::pair<render_capture_format, std::uint32_t>>
vulkan_render_backend::capture_format_for(VkFormat format)
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

void vulkan_render_backend::record_frame_capture(VkCommandBuffer command_buffer)
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
        append_image(render_capture_channel::scene_color, scene_color_.format, scene_color_.width, scene_color_.height);
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

void vulkan_render_backend::destroy_texture(gpu_texture& value) noexcept
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

void vulkan_render_backend::destroy_meshes() noexcept
{
    for (auto& [_, mesh] : meshes_)
    {
        destroy_buffer(mesh.vertices);
        for (auto& vertices : mesh.dynamic_vertices)
            destroy_buffer(vertices);
        destroy_buffer(mesh.skin_vertices);
        destroy_buffer(mesh.indices);
    }
    meshes_.clear();
    for (auto& [_, palette] : skin_palettes_)
        destroy_skin_palette_buffers(palette);
    skin_palettes_.clear();
    for (auto& [_, instance] : gpu_skinned_instances_)
        destroy_gpu_skinned_instance(instance);
    gpu_skinned_instances_.clear();
    for (auto& [_, mesh] : virtual_meshes_)
    {
        destroy_buffer(mesh.vertices);
        destroy_buffer(mesh.indices);
    }
    virtual_meshes_.clear();
    for (auto& [_, instance] : gpu_terrain_instances_)
        destroy_gpu_terrain_instance(instance);
    gpu_terrain_instances_.clear();
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
        destroy_material_runtime(material.runtime);
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
    if (gpu_terrain_traversal_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, gpu_terrain_traversal_pipeline_, nullptr);
        gpu_terrain_traversal_pipeline_ = VK_NULL_HANDLE;
    }
    if (gpu_terrain_traversal_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, gpu_terrain_traversal_pipeline_layout_, nullptr);
        gpu_terrain_traversal_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (gpu_terrain_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, gpu_terrain_descriptor_pool_, nullptr);
        gpu_terrain_descriptor_pool_ = VK_NULL_HANDLE;
    }
    if (gpu_terrain_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, gpu_terrain_descriptor_set_layout_, nullptr);
        gpu_terrain_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
}

std::optional<VkFormat> vulkan_render_backend::vulkan_texture_format(texture_format format) const noexcept
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

bool vulkan_render_backend::texture_format_supported(VkFormat format) const noexcept
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device_, format, &properties);
    constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return (properties.optimalTilingFeatures & required) == required;
}

bool vulkan_render_backend::upload_buffer(const void* source, VkDeviceSize size, VkBufferUsageFlags usage,
                                          gpu_buffer& destination)
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

bool vulkan_render_backend::upload_buffer_region(const void* source, VkDeviceSize size, gpu_buffer& destination,
                                                 VkDeviceSize offset)
{
    if (!source || size == 0 || destination.buffer == VK_NULL_HANDLE) return false;
    const auto staging = reserve_upload(size, 16u);
    if (!staging) return false;
    std::memcpy(staging.bytes.data(), source, static_cast<std::size_t>(size));
    vmaFlushAllocation(allocator_, upload_staging_.allocation, static_cast<VkDeviceSize>(staging.offset), size);
    const VkBufferCopy copy{.srcOffset = static_cast<VkDeviceSize>(staging.offset), .dstOffset = offset, .size = size};
    vkCmdCopyBuffer(upload_command_buffer_, upload_staging_.buffer, destination.buffer, 1, &copy);
    upload_batch_has_work_ = true;
    return true;
}

bool vulkan_render_backend::upload_texture_image(const texture_data& data, gpu_texture& destination)
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
    vkCmdPipelineBarrier(upload_command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &to_copy);

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
                                data.dimension == texture_dimension::texture_3d ? std::max(1u, data.depth >> mip) : 1u};
            regions.push_back(copy);
        }
    }
    else
    {
        VkBufferImageCopy copy{};
        copy.bufferOffset = static_cast<VkDeviceSize>(staging.offset);
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = image.arrayLayers;
        copy.imageExtent = {data.width, data.height, data.dimension == texture_dimension::texture_3d ? data.depth : 1u};
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
    vkCmdPipelineBarrier(upload_command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_shader);
    upload_batch_has_work_ = true;
    destination.format = *format;
    destination.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    destination.mip_count = mip_count;
    return true;
}

void vulkan_render_backend::upload_mesh(const mesh_upload_event& event)
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
        found->second.source_vertices = event.mesh->vertices;
        found->second.skin_influences = event.mesh->skin_vertices;
        ++found->second.vertex_revision;
        return;
    }

    gpu_mesh mesh;
    mesh.dynamic = event.mesh->usage == mesh_usage::dynamic_per_frame;
    mesh.source_vertices = event.mesh->vertices;
    mesh.skin_influences = event.mesh->skin_vertices;
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
            if (!create_buffer(vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_TO_GPU, vertices))
                return false;
        }
        return true;
    }()
            : upload_buffer(event.mesh->vertices.data(), vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mesh.vertices);
    const bool skin_ready = event.mesh->skin_vertices.empty() ||
                            upload_buffer(event.mesh->skin_vertices.data(),
                                          buffer_size(event.mesh->skin_vertices.size(), sizeof(mesh_skin_vertex)),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mesh.skin_vertices);
    if (!vertices_ready || !skin_ready ||
        !upload_buffer(event.mesh->indices.data(), index_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices))
    {
        destroy_buffer(mesh.vertices);
        for (auto& vertices : mesh.dynamic_vertices)
            destroy_buffer(vertices);
        destroy_buffer(mesh.skin_vertices);
        destroy_buffer(mesh.indices);
        arc::diagnostics::error("render.vulkan", "Failed to upload mesh '" + event.label + "'");
        return;
    }

    mesh.vertex_count = static_cast<std::uint32_t>(event.mesh->vertices.size());
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
                                     destroy_buffer(replaced.skin_vertices);
                                     destroy_buffer(replaced.indices);
                                 });
    }
    meshes_[key] = std::move(mesh);
}

void vulkan_render_backend::retire_mesh(mesh_handle handle)
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
                                 destroy_buffer(retired.skin_vertices);
                                 destroy_buffer(retired.indices);
                             });
}

void vulkan_render_backend::destroy_skin_palette_buffers(gpu_skin_palette& palette) noexcept
{
    destroy_buffer(palette.current);
    destroy_buffer(palette.previous);
}

void vulkan_render_backend::upload_skin_palette(const skin_palette_upload_event& event)
{
    if (!event.palette || !event.palette->valid()) return;
    gpu_skin_palette palette;
    const auto byte_size = buffer_size(event.palette->current.size(), sizeof(math::matrix4f));
    const auto previous =
        event.palette->previous.empty() ? std::span{event.palette->current} : std::span{event.palette->previous};
    if (!upload_buffer(event.palette->current.data(), byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, palette.current) ||
        !upload_buffer(previous.data(), byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, palette.previous))
    {
        destroy_skin_palette_buffers(palette);
        arc::diagnostics::error("render.vulkan", "Failed to upload skin palette '" + event.label + "'");
        return;
    }
    palette.joint_count = static_cast<std::uint32_t>(event.palette->current.size());
    palette.content_revision = event.palette->content_revision;
    palette.current_matrices = event.palette->current;
    palette.previous_matrices.assign(previous.begin(), previous.end());
    const auto key = resource_key(event.handle);
    if (auto found = skin_palettes_.find(key); found != skin_palettes_.end())
    {
        auto replaced = std::move(found->second);
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, replaced]() mutable { destroy_skin_palette_buffers(replaced); });
    }
    skin_palettes_[key] = std::move(palette);
}

void vulkan_render_backend::retire_skin_palette(buffer_handle handle)
{
    const auto found = skin_palettes_.find(resource_key(handle));
    if (found == skin_palettes_.end()) return;
    auto retired = std::move(found->second);
    skin_palettes_.erase(found);
    deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                             [this, retired]() mutable { destroy_skin_palette_buffers(retired); });
    for (auto instance = gpu_skinned_instances_.begin(); instance != gpu_skinned_instances_.end();)
    {
        if (instance->second.palette != handle)
        {
            ++instance;
            continue;
        }
        destroy_gpu_skinned_instance(instance->second);
        instance = gpu_skinned_instances_.erase(instance);
    }
}

surface_frame_result vulkan_render_backend::create_viewport_output(const viewport_output_descriptor& descriptor)
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
    return surface_frame_result::failure({.code = surface_frame_error_code::unsupported,
                                          .message = "shared viewport textures are not implemented on this platform"});
#endif
}

surface_frame_result vulkan_render_backend::resize_viewport_output(std::string_view viewport_id, std::uint32_t width,
                                                                   std::uint32_t height)
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
    if (std::ranges::any_of(output.slots,
                            [](const auto& slot) { return slot.state == shared_viewport_frame_state::consumer_owned; }))
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

surface_frame_result vulkan_render_backend::present_viewport_output(std::string_view viewport_id)
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

shared_viewport_frame_result vulkan_render_backend::poll_viewport_output(std::string_view viewport_id)
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
                              .synchronization = {.producer_complete = true, .value = ready->frame_id}});
#else
    (void)viewport_id;
    return shared_viewport_frame_result::failure(
        {.code = surface_frame_error_code::unsupported, .message = "shared viewport textures are unsupported"});
#endif
}

void vulkan_render_backend::release_viewport_frame(std::string_view viewport_id, std::uint64_t generation,
                                                   std::uint64_t frame_id)
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

void vulkan_render_backend::set_viewport_output_visible(std::string_view viewport_id, bool visible)
{
#if ARC_VULKAN_SHARED_VIEWPORT
    if (auto found = shared_viewports_.find(std::string(viewport_id)); found != shared_viewports_.end())
        found->second.visible = visible;
#else
    (void)viewport_id;
    (void)visible;
#endif
}

void vulkan_render_backend::destroy_viewport_output(std::string_view viewport_id)
{
#if ARC_VULKAN_SHARED_VIEWPORT
    const auto found = shared_viewports_.find(std::string(viewport_id));
    if (found == shared_viewports_.end()) return;
    found->second.visible = false;
    if (std::ranges::any_of(found->second.slots,
                            [](const auto& slot) { return slot.state == shared_viewport_frame_state::consumer_owned; }))
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

} // namespace arc::render::vulkan::backend_detail
