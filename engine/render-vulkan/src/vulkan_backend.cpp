#include <arc/render/vulkan/vulkan_backend.h>

#include <arc/diagnostics/log.h>
#include <arc/render/lighting.h>
#include <arc/render/render_world.h>
#include <arc/render/resources.h>

#include "builtin_shaders.h"
#include "vulkan_sky_constants.h"

#if ARC_RENDER_VULKAN_ENABLE_IMGUI
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#endif

#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::render::vulkan
{
namespace
{

void log_vk_result(VkResult result)
{
    if (result != VK_SUCCESS)
        arc::error("render.vulkan", "Vulkan call failed");
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
    return { matrix(0, 3), matrix(1, 3), matrix(2, 3) };
}

math::matrix4f look_at_rh(const math::vector3f& eye, const math::vector3f& target, const math::vector3f& up) noexcept
{
    const auto z = math::normalize(math::sub(eye, target), 0.0f);
    auto x = math::normalize(math::cross(up, z), 0.0f);
    if (math::length_squared(x) < 0.0001f)
        x = math::vector3f{ 1.0f, 0.0f, 0.0f };
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

math::matrix4f orthographic_rh_zo(float width, float height, float near_plane, float far_plane) noexcept
{
    width = std::max(width, 0.001f);
    height = std::max(height, 0.001f);
    near_plane = std::max(near_plane, 0.001f);
    far_plane = std::max(far_plane, near_plane + 0.001f);

    math::matrix4f result = math::identity<float, 4>();
    result(0, 0) = 2.0f / width;
    result(1, 1) = 2.0f / height;
    result(2, 2) = 1.0f / (near_plane - far_plane);
    result(2, 3) = near_plane / (near_plane - far_plane);
    return result;
}

bool inverse_matrix4(const math::matrix4f& input, math::matrix4f& output) noexcept
{
    float augmented[4][8]{};
    for (std::uint32_t row = 0; row < 4; ++row)
    {
        for (std::uint32_t col = 0; col < 4; ++col)
            augmented[row][col] = input(row, col);
        augmented[row][4 + row] = 1.0f;
    }

    for (std::uint32_t col = 0; col < 4; ++col)
    {
        std::uint32_t pivot = col;
        for (std::uint32_t row = col + 1; row < 4; ++row)
        {
            if (std::abs(augmented[row][col]) > std::abs(augmented[pivot][col]))
                pivot = row;
        }
        if (std::abs(augmented[pivot][col]) < 0.000001f)
            return false;

        if (pivot != col)
        {
            for (std::uint32_t index = 0; index < 8; ++index)
                std::swap(augmented[pivot][index], augmented[col][index]);
        }

        const float divisor = augmented[col][col];
        for (float& value : augmented[col])
            value /= divisor;

        for (std::uint32_t row = 0; row < 4; ++row)
        {
            if (row == col)
                continue;
            const float factor = augmented[row][col];
            for (std::uint32_t index = 0; index < 8; ++index)
                augmented[row][index] -= factor * augmented[col][index];
        }
    }

    for (std::uint32_t row = 0; row < 4; ++row)
    {
        for (std::uint32_t col = 0; col < 4; ++col)
            output(row, col) = augmented[row][4 + col];
    }
    return true;
}

math::vector3f transform_clip_point(const math::matrix4f& matrix, float x, float y, float z) noexcept
{
    const float clip[4]{ x, y, z, 1.0f };
    float result[4]{};
    for (std::uint32_t row = 0; row < 4; ++row)
    {
        for (std::uint32_t col = 0; col < 4; ++col)
            result[row] += matrix(row, col) * clip[col];
    }
    const float inv_w = std::abs(result[3]) > 0.000001f ? 1.0f / result[3] : 1.0f;
    return { result[0] * inv_w, result[1] * inv_w, result[2] * inv_w };
}

math::vector3f matrix_row3(const math::matrix4f& matrix, std::uint32_t row) noexcept
{
    return { matrix(row, 0), matrix(row, 1), matrix(row, 2) };
}

struct mesh_push_constants
{
    float model_view_projection[16]{};
    float model[16]{};
    float base_color[4]{ 1.0f, 1.0f, 1.0f, 1.0f };
    float light_direction_intensity[4]{ 0.35f, -0.85f, -0.40f, 1.0f };
    float light_color[4]{ 1.0f, 1.0f, 1.0f, 1.0f };
    float camera_position[4]{};
    float visualization[4]{};
    float fog_color_density[4]{};
    float fog_params[4]{};
    float material_params[4]{ 1.0f, 1.0f, 1.0f, 0.0f };
};

struct deferred_push_constants
{
    float light_direction_intensity[4]{ 0.35f, -0.85f, -0.40f, 1.0f };
    float light_color_exposure[4]{ 1.0f, 1.0f, 1.0f, 1.0f };
    float ambient_visualization[4]{ 0.18f, 0.18f, 0.18f, 0.0f };
    float debug_counts[4]{};
};

struct shadow_uniform_data
{
    float light_view_projection[directional_shadow_cascade_count][16]{};
    float cascade_splits[4]{};
    float params[4]{};
    float cascade_texel_size[4]{};
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
    VkFormat format{};
    VkImageAspectFlags aspect{};
    VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
};

class vulkan_render_backend final : public vulkan_backend
{
public:
    vulkan_render_backend(
        VkInstance instance,
        VkSurfaceKHR surface,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue queue,
        VmaAllocator allocator,
        std::uint32_t graphics_queue_family,
        render_capabilities capabilities)
        : instance_(instance)
        , surface_(surface)
        , physical_device_(physical_device)
        , device_(device)
        , queue_(queue)
        , allocator_(allocator)
        , graphics_queue_family_(graphics_queue_family)
        , capabilities_(capabilities)
    {
        create_support_objects();
    }

    ~vulkan_render_backend() override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        shutdown_imgui();
#endif
        if (device_ != VK_NULL_HANDLE)
            vkDeviceWaitIdle(device_);
        destroy_mesh_pipeline();
        destroy_shadow_resources();
        destroy_white_texture();
        destroy_buffer(pick_readback_buffer_);
        for (auto& buffer : shadow_uniform_buffers_)
            destroy_buffer(buffer);
        destroy_buffer(light_buffer_);
        destroy_meshes();
        destroy_support_objects();
        if (allocator_ != VK_NULL_HANDLE)
            vmaDestroyAllocator(allocator_);
        if (device_ != VK_NULL_HANDLE)
            vkDestroyDevice(device_, nullptr);
        if (surface_ != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (instance_ != VK_NULL_HANDLE)
            vkDestroyInstance(instance_, nullptr);
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
        resolved_config_ = config;
        last_profile_.configuration = config;
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        if (imgui_initialized_ && previous_scale != config.render_scale &&
            output_viewport_width_ > 0 && output_viewport_height_ > 0)
        {
            ensure_viewport(
                scaled_dimension(output_viewport_width_),
                scaled_dimension(output_viewport_height_));
        }
#endif
    }

    VkInstance instance() const noexcept override
    {
        return instance_;
    }

    VkPhysicalDevice physical_device() const noexcept override
    {
        return physical_device_;
    }

    VkDevice device() const noexcept override
    {
        return device_;
    }

    std::uint32_t queue_family() const noexcept override
    {
        return graphics_queue_family_;
    }

    VkQueue queue() const noexcept override
    {
        return queue_;
    }

    render_submit_result submit(const render_frame_packet& packet, const compiled_render_graph& graph) override
    {
        frame_draws_.clear();
        frame_virtual_draws_.clear();
        frame_shadow_draws_.clear();
        frame_directional_lights_.clear();
        frame_point_lights_.clear();
        frame_spot_lights_.clear();
        frame_environment_ = {};
        pending_debug_markers_.clear();
        for (const auto& event : packet.events)
        {
            if (const auto* upload = std::get_if<mesh_upload_event>(&event.payload))
                upload_mesh(*upload);
            else if (const auto* upload = std::get_if<virtual_mesh_upload_event>(&event.payload))
                upload_virtual_mesh(*upload);
            else if (const auto* texture = std::get_if<texture_upload_event>(&event.payload))
                upload_texture(*texture);
            else if (const auto* material = std::get_if<material_upload_event>(&event.payload))
                upload_material(*material);
            else if (const auto* environment = std::get_if<environment_upload_event>(&event.payload))
                upload_environment(*environment);
            else if (const auto* environment = std::get_if<environment_destroy_event>(&event.payload))
            {
                environments_.erase(resource_key(environment->handle));
                if (active_environment_ == environment->handle)
                    active_environment_ = {};
            }
            else if (const auto* draw = std::get_if<draw_mesh_event>(&event.payload))
            {
                frame_draws_.push_back(*draw);
                frame_shadow_draws_.push_back(*draw);
            }
            else if (const auto* light = std::get_if<directional_light_event>(&event.payload))
                frame_directional_lights_.push_back(*light);
            else if (const auto* light = std::get_if<point_light_event>(&event.payload))
                frame_point_lights_.push_back(*light);
            else if (const auto* light = std::get_if<spot_light_event>(&event.payload))
                frame_spot_lights_.push_back(*light);
            else if (const auto* world = std::get_if<render_world_event>(&event.payload))
                append_render_world(*world);
            else if (const auto* marker = std::get_if<debug_marker_event>(&event.payload))
                pending_debug_markers_.push_back(marker->label);
        }

        last_profile_.frame_index = packet.frame_index;
        last_profile_.graph = graph;
        last_profile_.summary.clear();
        last_profile_.summary.reserve(64);
        last_profile_.summary += std::to_string(graph.passes.size());
        last_profile_.summary += " graph pass(es), ";
        last_profile_.summary += std::to_string(packet.events.size());
        last_profile_.summary += " render event(s)";

        const environment_desc* lighting_environment = active_environment();
        if (frame_environment_.lighting.environment.valid())
        {
            const auto found = environments_.find(resource_key(frame_environment_.lighting.environment));
            if (found != environments_.end())
                lighting_environment = &found->second.data;
        }
        frame_lighting_ = pack_scene_lighting(
            frame_directional_lights_,
            frame_point_lights_,
            frame_spot_lights_,
            frame_environment_.affect_lighting && frame_environment_.lighting.enabled
                ? lighting_environment
                : nullptr,
            resolved_config_.max_point_lights,
            resolved_config_.max_spot_lights);
        if (frame_environment_.affect_lighting && frame_environment_.lighting.enabled)
        {
            math::vector3f ambient = frame_environment_.lighting.constant_color;
            if (frame_environment_.lighting.source == environment_lighting_source_mode::follow_sky)
            {
                ambient = frame_environment_.source == sky_source_mode::solid_color
                    ? frame_environment_.solid_color
                    : math::vector3f{
                        frame_environment_.atmosphere.tint[0] * 0.28f,
                        frame_environment_.atmosphere.tint[1] * 0.28f,
                        frame_environment_.atmosphere.tint[2] * 0.28f };
            }
            if (frame_environment_.lighting.source != environment_lighting_source_mode::hdri || !lighting_environment)
            {
                frame_lighting_.ambient_color_intensity = {
                    ambient[0], ambient[1], ambient[2], frame_environment_.lighting.diffuse_intensity
                };
            }
        }
        update_environment_profile(lighting_environment);
        last_profile_.clustered_lights = make_clustered_light_profile();
        update_light_buffer();
        warn_about_skipped_lights(frame_lighting_);

        std::ostringstream message;
        message << "vulkan accepted frame " << packet.frame_index << " with "
                << packet.events.size() << " event(s) and " << graph.passes.size() << " pass(es)";
        return { .submitted = true, .message = message.str() };
    }

    void resize_viewport(std::uint32_t width, std::uint32_t height) override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        output_viewport_width_ = width;
        output_viewport_height_ = height;
        if (imgui_initialized_ && width > 0 && height > 0)
            ensure_viewport(scaled_dimension(width), scaled_dimension(height));
#else
        (void)width;
        (void)height;
#endif
    }

    render_viewport_texture viewport_texture() const noexcept override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        return {
            .id = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(viewport_descriptor_)),
            .width = viewport_width_,
            .height = viewport_height_
        };
#else
        return {};
#endif
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

    bool initialize_imgui(std::uint32_t width, std::uint32_t height, std::string& message) override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        if (surface_ == VK_NULL_HANDLE)
        {
            message = "Vulkan backend was created without a presentation surface";
            return false;
        }

        VkBool32 present_supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, graphics_queue_family_, surface_, &present_supported);
        if (present_supported != VK_TRUE)
        {
            message = "Vulkan queue does not support the editor surface";
            return false;
        }

        const std::array<VkFormat, 4> formats{
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_B8G8R8_UNORM,
            VK_FORMAT_R8G8B8_UNORM
        };
        window_.Surface = surface_;
        window_.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
            physical_device_,
            window_.Surface,
            formats.data(),
            formats.size(),
            VK_COLORSPACE_SRGB_NONLINEAR_KHR);

        const VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
        window_.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(physical_device_, window_.Surface, &present_mode, 1);
        ImGui_ImplVulkanH_CreateOrResizeWindow(
            instance_,
            physical_device_,
            device_,
            &window_,
            graphics_queue_family_,
            nullptr,
            static_cast<int>(std::max(1u, width)),
            static_cast<int>(std::max(1u, height)),
            min_image_count_,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

        ImGui_ImplVulkan_InitInfo init_info{};
        init_info.ApiVersion = VK_API_VERSION_1_2;
        init_info.Instance = instance_;
        init_info.PhysicalDevice = physical_device_;
        init_info.Device = device_;
        init_info.QueueFamily = graphics_queue_family_;
        init_info.Queue = queue_;
        init_info.DescriptorPoolSize = 1024;
        init_info.MinImageCount = min_image_count_;
        init_info.ImageCount = window_.ImageCount;
        init_info.PipelineInfoMain.RenderPass = window_.RenderPass;
        init_info.PipelineInfoMain.Subpass = 0;
        init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.CheckVkResultFn = log_vk_result;

        if (!ImGui_ImplVulkan_Init(&init_info))
        {
            message = "failed to initialize ImGui Vulkan backend";
            return false;
        }

        imgui_initialized_ = true;
        output_viewport_width_ = std::max(1u, width / 2);
        output_viewport_height_ = std::max(1u, height / 2);
        ensure_viewport(
            scaled_dimension(output_viewport_width_),
            scaled_dimension(output_viewport_height_));
        message = "initialized Vulkan editor presentation";
        return true;
#else
        (void)width;
        (void)height;
        message = "Vulkan ImGui presentation support is not compiled";
        return false;
#endif
    }

    void new_imgui_frame() override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        if (imgui_initialized_)
            ImGui_ImplVulkan_NewFrame();
#endif
    }

    bool render_imgui_frame(void* draw_data, std::uint32_t width, std::uint32_t height, std::string& message) override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        if (!imgui_initialized_)
        {
            message = "Vulkan ImGui presentation is not initialized";
            return false;
        }
        if (!draw_data || width == 0 || height == 0)
            return true;

        if (swapchain_rebuild_ || window_.Width != static_cast<int>(width) || window_.Height != static_cast<int>(height))
        {
            ImGui_ImplVulkan_SetMinImageCount(min_image_count_);
            ImGui_ImplVulkanH_CreateOrResizeWindow(
                instance_,
                physical_device_,
                device_,
                &window_,
                graphics_queue_family_,
                nullptr,
                static_cast<int>(width),
                static_cast<int>(height),
                min_image_count_,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            window_.FrameIndex = 0;
            swapchain_rebuild_ = false;
        }

        VkSemaphore image_acquired_semaphore = window_.FrameSemaphores[window_.SemaphoreIndex].ImageAcquiredSemaphore;
        VkSemaphore render_complete_semaphore = window_.FrameSemaphores[window_.SemaphoreIndex].RenderCompleteSemaphore;
        VkResult result = vkAcquireNextImageKHR(device_, window_.Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &window_.FrameIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            swapchain_rebuild_ = true;
            return true;
        }
        if (result != VK_SUCCESS)
        {
            message = "failed to acquire Vulkan swapchain image";
            return false;
        }
        active_frame_index_ = window_.FrameIndex;

        ImGui_ImplVulkanH_Frame* frame = &window_.Frames[window_.FrameIndex];
        vkWaitForFences(device_, 1, &frame->Fence, VK_TRUE, UINT64_MAX);
        collect_timestamp_results();
        collect_object_pick_result();
        retire_completed_resources();

        // Resource preparation may need to wait for every swapchain frame
        // before replacing frame-count-dependent buffers or attachments. Keep
        // the acquired frame fence signaled until that work has completed;
        // resetting it first makes wait_for_in_flight_frames() wait forever on
        // a fence that cannot be submitted until this function continues.
        prepare_frame_gpu_resources();

        vkResetFences(device_, 1, &frame->Fence);
        vkResetCommandPool(device_, frame->CommandPool, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(frame->CommandBuffer, &begin_info);

        begin_debug_label(frame->CommandBuffer, "ARC frame", { 0.16f, 0.45f, 1.0f, 1.0f });
        reset_timestamp_queries(frame->CommandBuffer);
        for (const auto& marker : pending_debug_markers_)
            insert_debug_label(frame->CommandBuffer, marker, { 0.25f, 0.75f, 1.0f, 1.0f });

        const auto shadow_scope = begin_gpu_scope(frame->CommandBuffer, "directional shadow cascades");
        render_shadow_maps(frame->CommandBuffer);
        end_gpu_scope(frame->CommandBuffer, shadow_scope);

        const auto viewport_scope = begin_gpu_scope(
            frame->CommandBuffer,
            resolved_config_.path == render_path::forward_plus ? "forward+ viewport raster" : "deferred viewport raster");
        render_viewport(frame->CommandBuffer);
        end_gpu_scope(frame->CommandBuffer, viewport_scope);

        VkClearValue clear_value{};
        clear_value.color.float32[0] = 0.055f;
        clear_value.color.float32[1] = 0.071f;
        clear_value.color.float32[2] = 0.086f;
        clear_value.color.float32[3] = 1.0f;

        const auto imgui_scope = begin_gpu_scope(frame->CommandBuffer, "imgui pass");
        VkRenderPassBeginInfo render_pass{};
        render_pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass.renderPass = window_.RenderPass;
        render_pass.framebuffer = frame->Framebuffer;
        render_pass.renderArea.extent.width = window_.Width;
        render_pass.renderArea.extent.height = window_.Height;
        render_pass.clearValueCount = 1;
        render_pass.pClearValues = &clear_value;
        vkCmdBeginRenderPass(frame->CommandBuffer, &render_pass, VK_SUBPASS_CONTENTS_INLINE);
        ImGui_ImplVulkan_RenderDrawData(static_cast<ImDrawData*>(draw_data), frame->CommandBuffer);
        vkCmdEndRenderPass(frame->CommandBuffer);
        end_gpu_scope(frame->CommandBuffer, imgui_scope);
        end_debug_label(frame->CommandBuffer);
        vkEndCommandBuffer(frame->CommandBuffer);

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &image_acquired_semaphore;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame->CommandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &render_complete_semaphore;
        result = vkQueueSubmit(queue_, 1, &submit, frame->Fence);
        if (result != VK_SUCCESS)
        {
            message = "failed to submit Vulkan editor frame";
            return false;
        }

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &render_complete_semaphore;
        present.swapchainCount = 1;
        present.pSwapchains = &window_.Swapchain;
        present.pImageIndices = &window_.FrameIndex;
        result = vkQueuePresentKHR(queue_, &present);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            swapchain_rebuild_ = true;
            return true;
        }
        if (result != VK_SUCCESS)
        {
            message = "failed to present Vulkan editor frame";
            return false;
        }

        window_.SemaphoreIndex = (window_.SemaphoreIndex + 1) % window_.SemaphoreCount;
        last_completed_frame_ = last_profile_.frame_index;
        return true;
#else
        (void)draw_data;
        (void)width;
        (void)height;
        message = "Vulkan ImGui presentation support is not compiled";
        return false;
#endif
    }

    bool render_native_viewport_frame(std::uint32_t width, std::uint32_t height, std::string& message) override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        if (surface_ == VK_NULL_HANDLE)
        {
            message = "Vulkan backend was created without a presentation surface";
            return false;
        }
        if (width == 0 || height == 0)
            return true;

        if (!native_swapchain_initialized_)
        {
            VkBool32 present_supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, graphics_queue_family_, surface_, &present_supported);
            if (present_supported != VK_TRUE)
            {
                message = "Vulkan queue does not support the native viewport surface";
                return false;
            }

            const std::array<VkFormat, 4> formats{
                VK_FORMAT_B8G8R8A8_UNORM,
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_FORMAT_B8G8R8_UNORM,
                VK_FORMAT_R8G8B8_UNORM
            };
            window_.Surface = surface_;
            window_.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
                physical_device_,
                window_.Surface,
                formats.data(),
                formats.size(),
                VK_COLORSPACE_SRGB_NONLINEAR_KHR);

            const VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
            window_.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(physical_device_, window_.Surface, &present_mode, 1);
            ImGui_ImplVulkanH_CreateOrResizeWindow(
                instance_,
                physical_device_,
                device_,
                &window_,
                graphics_queue_family_,
                nullptr,
                static_cast<int>(width),
                static_cast<int>(height),
                min_image_count_,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            viewport_format_ = window_.SurfaceFormat.format;
            native_swapchain_initialized_ = true;
        }

        if (swapchain_rebuild_ || window_.Width != static_cast<int>(width) || window_.Height != static_cast<int>(height))
        {
            ImGui_ImplVulkanH_CreateOrResizeWindow(
                instance_,
                physical_device_,
                device_,
                &window_,
                graphics_queue_family_,
                nullptr,
                static_cast<int>(width),
                static_cast<int>(height),
                min_image_count_,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            window_.FrameIndex = 0;
            swapchain_rebuild_ = false;
        }

        ensure_viewport(scaled_dimension(width), scaled_dimension(height));

        VkSemaphore image_acquired_semaphore = window_.FrameSemaphores[window_.SemaphoreIndex].ImageAcquiredSemaphore;
        VkSemaphore render_complete_semaphore = window_.FrameSemaphores[window_.SemaphoreIndex].RenderCompleteSemaphore;
        VkResult result = vkAcquireNextImageKHR(device_, window_.Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &window_.FrameIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            swapchain_rebuild_ = true;
            return true;
        }
        if (result != VK_SUCCESS)
        {
            message = "failed to acquire native viewport swapchain image";
            return false;
        }
        active_frame_index_ = window_.FrameIndex;

        ImGui_ImplVulkanH_Frame* frame = &window_.Frames[window_.FrameIndex];
        vkWaitForFences(device_, 1, &frame->Fence, VK_TRUE, UINT64_MAX);
        collect_timestamp_results();
        collect_object_pick_result();
        retire_completed_resources();

        // See the ImGui presentation path above. Frame-dependent resource
        // creation can wait on all swapchain fences and must happen before the
        // current acquired fence is reset for this frame's submission.
        prepare_frame_gpu_resources();

        vkResetFences(device_, 1, &frame->Fence);
        vkResetCommandPool(device_, frame->CommandPool, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(frame->CommandBuffer, &begin_info);

        begin_debug_label(frame->CommandBuffer, "ARC native viewport frame", { 0.16f, 0.45f, 1.0f, 1.0f });
        reset_timestamp_queries(frame->CommandBuffer);

        const auto shadow_scope = begin_gpu_scope(frame->CommandBuffer, "directional shadow cascades");
        render_shadow_maps(frame->CommandBuffer);
        end_gpu_scope(frame->CommandBuffer, shadow_scope);

        const auto viewport_scope = begin_gpu_scope(
            frame->CommandBuffer,
            resolved_config_.path == render_path::forward_plus ? "forward+ viewport raster" : "deferred viewport raster");
        render_viewport(frame->CommandBuffer);
        end_gpu_scope(frame->CommandBuffer, viewport_scope);

        transition_viewport(frame->CommandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkImageMemoryBarrier swapchain_to_transfer{};
        swapchain_to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        swapchain_to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        swapchain_to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapchain_to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapchain_to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapchain_to_transfer.image = frame->Backbuffer;
        swapchain_to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        swapchain_to_transfer.subresourceRange.levelCount = 1;
        swapchain_to_transfer.subresourceRange.layerCount = 1;
        swapchain_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            frame->CommandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &swapchain_to_transfer);

        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = {
            static_cast<std::int32_t>(viewport_width_),
            static_cast<std::int32_t>(viewport_height_),
            1
        };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1] = {
            static_cast<std::int32_t>(width),
            static_cast<std::int32_t>(height),
            1
        };
        vkCmdBlitImage(
            frame->CommandBuffer,
            viewport_image_,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            frame->Backbuffer,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_LINEAR);

        VkImageMemoryBarrier swapchain_to_present = swapchain_to_transfer;
        swapchain_to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapchain_to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        swapchain_to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapchain_to_present.dstAccessMask = 0;
        vkCmdPipelineBarrier(
            frame->CommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &swapchain_to_present);

        end_debug_label(frame->CommandBuffer);
        vkEndCommandBuffer(frame->CommandBuffer);

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &image_acquired_semaphore;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame->CommandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &render_complete_semaphore;
        result = vkQueueSubmit(queue_, 1, &submit, frame->Fence);
        if (result != VK_SUCCESS)
        {
            message = "failed to submit native viewport frame";
            return false;
        }

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &render_complete_semaphore;
        present.swapchainCount = 1;
        present.pSwapchains = &window_.Swapchain;
        present.pImageIndices = &window_.FrameIndex;
        result = vkQueuePresentKHR(queue_, &present);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            swapchain_rebuild_ = true;
            return true;
        }
        if (result != VK_SUCCESS)
        {
            message = "failed to present native viewport frame";
            return false;
        }

        window_.SemaphoreIndex = (window_.SemaphoreIndex + 1) % window_.SemaphoreCount;
        last_completed_frame_ = last_profile_.frame_index;
        return true;
#else
        (void)width;
        (void)height;
        message = "Vulkan native viewport presentation support is not compiled";
        return false;
#endif
    }

    void shutdown_imgui() noexcept override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        if (!imgui_initialized_ && !native_swapchain_initialized_)
            return;

        vkDeviceWaitIdle(device_);
        destroy_viewport();
        if (imgui_initialized_)
            ImGui_ImplVulkan_Shutdown();
        if (window_.Surface != VK_NULL_HANDLE)
            ImGui_ImplVulkanH_DestroyWindow(instance_, device_, &window_, nullptr);
        imgui_initialized_ = false;
        native_swapchain_initialized_ = false;
#endif
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

#if ARC_RENDER_VULKAN_ENABLE_IMGUI
    std::uint32_t scaled_dimension(std::uint32_t value) const noexcept
    {
        return std::max(1u, static_cast<std::uint32_t>(
            std::round(static_cast<float>(value) * resolved_config_.render_scale)));
    }

    void wait_for_in_flight_frames() const
    {
        std::vector<VkFence> fences;
        fences.reserve(window_.ImageCount);
        for (std::uint32_t index = 0; index < window_.ImageCount; ++index)
        {
            const auto fence = window_.Frames[index].Fence;
            if (fence != VK_NULL_HANDLE)
                fences.push_back(fence);
        }
        if (!fences.empty())
            vkWaitForFences(device_, static_cast<std::uint32_t>(fences.size()), fences.data(), VK_TRUE, UINT64_MAX);
    }
#endif

    struct vulkan_command_context
    {
        VkCommandPool graphics_pool{};
        VkCommandBuffer graphics_buffer{};
        VkFence fence{};
    };

    struct vulkan_swapchain_state
    {
        VkSwapchainKHR swapchain{};
        VkFormat format{};
        VkExtent2D extent{};
        bool rebuild_requested{};
    };

    struct gpu_buffer
    {
        VkBuffer buffer{};
        VmaAllocation allocation{};
    };

    struct gpu_mesh
    {
        gpu_buffer vertices;
        gpu_buffer indices;
        std::uint32_t index_count{};
    };

    struct gpu_virtual_mesh
    {
        gpu_buffer vertices;
        gpu_buffer indices;
        std::vector<virtual_mesh_cluster> clusters;
        std::uint32_t index_count{};
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
        VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
        std::uint32_t mip_count{ 1 };
    };

    struct gpu_environment
    {
        environment_desc data;
    };

    struct gpu_material
    {
        material_desc data;
        std::vector<VkDescriptorSet> descriptor_sets;
    };

    struct folded_light_constants
    {
        math::vector3f direction{ 0.35f, -0.85f, -0.40f };
        math::vector3f color{ 1.0f, 1.0f, 1.0f };
        float intensity{ 1.0f };
    };

    struct vulkan_shadow_atlas
    {
        VkImage image{};
        VmaAllocation allocation{};
        VkImageView array_view{};
        std::array<VkImageView, directional_shadow_cascade_count> cascade_views{};
        VkSampler sampler{};
        VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
        std::uint32_t resolution{};
    };

    struct vulkan_shadow_cache
    {
        directional_shadow_cache_key last_directional_key{};
        bool has_directional_key{};
    };

    struct object_pick_readback
    {
        render_object_pick_request request{};
        std::uint64_t frame_index{};
        std::unordered_map<std::uint32_t, render_object_id> objects;
        bool active{};
    };

    static math::vector4f cluster_debug_color(std::uint32_t cluster_index) noexcept
    {
        const std::uint32_t hash = cluster_index * 747796405u + 2891336453u;
        const float r = static_cast<float>((hash >> 0u) & 0xffu) / 255.0f;
        const float g = static_cast<float>((hash >> 8u) & 0xffu) / 255.0f;
        const float b = static_cast<float>((hash >> 16u) & 0xffu) / 255.0f;
        return {
            0.25f + r * 0.75f,
            0.25f + g * 0.75f,
            0.25f + b * 0.75f,
            1.0f
        };
    }

    void append_render_world(const render_world_event& event)
    {
        if (!event.packet)
            return;

        const auto& packet = *event.packet;
        const auto make_draw = [&](const render_item& item, bool selected_for_overlay) {
            return draw_mesh_event{
                .mesh = item.mesh,
                .material = item.material,
                .model = item.model,
                .previous_model = item.previous_model,
                .view_projection = packet.camera.view_projection,
                .previous_view_projection = packet.camera.previous_view_projection,
                .mode = packet.mode,
                .visualization = packet.visualization,
                .object_id = item.object_id,
                .selected = selected_for_overlay,
                .base_color_tint = item.base_color_tint,
                .wire_color = math::vector4f{ 0.28f, 0.62f, 1.0f, 1.0f },
                .label = item.label
            };
        };
        const auto make_virtual_draw = [&](const virtual_render_item& item, bool selected_for_overlay) {
            auto tint = item.base_color_tint;
            auto material = item.material;
            if (packet.visualization == mesh_visualization_mode::cluster_debug)
            {
                tint = cluster_debug_color(item.cluster_index);
                material = {};
            }
            const auto visualization = packet.visualization == mesh_visualization_mode::cluster_debug
                ? mesh_visualization_mode::albedo
                : packet.visualization;
            return virtual_cluster_draw{
                .draw = draw_mesh_event{
                    .mesh = item.mesh,
                    .material = material,
                    .model = item.model,
                    .previous_model = item.previous_model,
                    .view_projection = packet.camera.view_projection,
                    .previous_view_projection = packet.camera.previous_view_projection,
                    .mode = packet.mode,
                    .visualization = visualization,
                    .object_id = item.object_id,
                    .selected = selected_for_overlay,
                    .base_color_tint = tint,
                    .wire_color = math::vector4f{ 0.28f, 0.62f, 1.0f, 1.0f },
                    .label = item.label
                },
                .mesh = item.mesh,
                .cluster_index = item.cluster_index
            };
        };

        frame_directional_lights_.insert(
            frame_directional_lights_.end(),
            packet.directional_lights.begin(),
            packet.directional_lights.end());
        frame_point_lights_.insert(
            frame_point_lights_.end(),
            packet.point_lights.begin(),
            packet.point_lights.end());
        frame_spot_lights_.insert(
            frame_spot_lights_.end(),
            packet.spot_lights.begin(),
            packet.spot_lights.end());

        for (const auto index : packet.visible_items)
        {
            if (index >= packet.items.size())
                continue;
            const auto& item = packet.items[index];
            frame_draws_.push_back(make_draw(
                item,
                packet.overlay == editor_overlay_mode::all_wireframe ||
                    (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
        }

        for (const auto index : packet.visible_virtual_items)
        {
            if (index >= packet.virtual_items.size())
                continue;
            const auto& item = packet.virtual_items[index];
            frame_virtual_draws_.push_back(make_virtual_draw(
                item,
                packet.overlay == editor_overlay_mode::all_wireframe ||
                    (packet.overlay == editor_overlay_mode::selected_wireframe && item.selected)));
        }

        for (const auto& item : packet.items)
        {
            if (!item.visible || !item.casts_shadows || !item.mesh.valid())
                continue;
            frame_shadow_draws_.push_back(make_draw(item, item.selected));
        }

        frame_camera_ = packet.camera;
        frame_environment_ = packet.environment;
        frame_shadows_enabled_ = packet.shadows_enabled;
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

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, families.data());
        timestamps_supported_ =
            graphics_queue_family_ < families.size() &&
            families[graphics_queue_family_].timestampValidBits > 0;

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

        descriptor_slots_.allocate(descriptor_resource_type::sampled_image);
    }

    void destroy_support_objects() noexcept
    {
        deferred_releases_.collect(UINT64_MAX);
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

    void begin_debug_label(VkCommandBuffer command_buffer, std::string_view name, const std::array<float, 4>& color) const
    {
        if (vkCmdBeginDebugUtilsLabelEXT == nullptr || name.empty())
            return;

        VkDebugUtilsLabelEXT label{};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name.data();
        std::copy(color.begin(), color.end(), label.color);
        vkCmdBeginDebugUtilsLabelEXT(command_buffer, &label);
    }

    void insert_debug_label(VkCommandBuffer command_buffer, std::string_view name, const std::array<float, 4>& color) const
    {
        if (vkCmdInsertDebugUtilsLabelEXT == nullptr || name.empty())
            return;

        VkDebugUtilsLabelEXT label{};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name.data();
        std::copy(color.begin(), color.end(), label.color);
        vkCmdInsertDebugUtilsLabelEXT(command_buffer, &label);
    }

    void end_debug_label(VkCommandBuffer command_buffer) const
    {
        if (vkCmdEndDebugUtilsLabelEXT != nullptr)
            vkCmdEndDebugUtilsLabelEXT(command_buffer);
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
        begin_debug_label(command_buffer, name, { 0.10f, 0.55f, 1.0f, 1.0f });
        if (timestamp_query_pool_ == VK_NULL_HANDLE || next_timestamp_query_ + 1 >= max_timestamp_queries_)
            return UINT32_MAX;

        const std::uint32_t begin_query = next_timestamp_query_++;
        const std::uint32_t end_query = next_timestamp_query_++;
        timestamp_scopes_.push_back({
            .name = std::string(name),
            .begin_query = begin_query,
            .end_query = end_query
        });
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
        if (timestamp_query_pool_ == VK_NULL_HANDLE || timestamp_scopes_.empty())
            return;

        std::array<std::uint64_t, max_timestamp_queries_> values{};
        const VkResult result = vkGetQueryPoolResults(
            device_,
            timestamp_query_pool_,
            0,
            max_timestamp_queries_,
            sizeof(values),
            values.data(),
            sizeof(std::uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS)
            return;

        last_profile_.pass_timings.clear();
        last_profile_.pass_timings.reserve(timestamp_scopes_.size());
        for (const auto& scope : timestamp_scopes_)
        {
            if (scope.end_query >= values.size() || values[scope.end_query] < values[scope.begin_query])
                continue;
            const auto ticks = values[scope.end_query] - values[scope.begin_query];
            last_profile_.pass_timings.push_back({
                .name = scope.name,
                .milliseconds = static_cast<double>(ticks) * static_cast<double>(timestamp_period_) / 1'000'000.0
            });
        }
    }

    void collect_object_pick_result()
    {
        if (!in_flight_pick_.active || pick_readback_buffer_.buffer == VK_NULL_HANDLE)
            return;

        void* mapped{};
        if (vmaMapMemory(allocator_, pick_readback_buffer_.allocation, &mapped) != VK_SUCCESS)
            return;

        std::uint32_t encoded_id{};
        std::memcpy(&encoded_id, mapped, sizeof(encoded_id));
        vmaUnmapMemory(allocator_, pick_readback_buffer_.allocation);

        last_pick_result_ = {
            .available = true,
            .hit = false,
            .object = {},
            .x = in_flight_pick_.request.x,
            .y = in_flight_pick_.request.y,
            .frame_index = in_flight_pick_.frame_index
        };

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

    clustered_light_grid_profile make_clustered_light_profile() const noexcept
    {
        clustered_light_grid_profile profile{};
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        const std::uint32_t width = std::max(1u, viewport_width_);
        const std::uint32_t height = std::max(1u, viewport_height_);
#else
        const std::uint32_t width = 1u;
        const std::uint32_t height = 1u;
#endif
        profile.tiles_x = (width + profile.tile_size_pixels - 1u) / profile.tile_size_pixels;
        profile.tiles_y = (height + profile.tile_size_pixels - 1u) / profile.tile_size_pixels;
        profile.cluster_count = profile.tiles_x * profile.tiles_y * profile.depth_slices;
        profile.point_light_references = frame_lighting_.point_count * profile.depth_slices;
        profile.spot_light_references = frame_lighting_.spot_count * profile.depth_slices;
        profile.overflow_count = frame_lighting_.skipped_point_count + frame_lighting_.skipped_spot_count;
        profile.available = true;
        return profile;
    }

    bool create_buffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VmaMemoryUsage memory_usage,
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

    bool submit_upload_commands(VkCommandBuffer command_buffer)
    {
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence{};
        if (vkCreateFence(device_, &fence_info, nullptr, &fence) != VK_SUCCESS)
            return false;

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        const VkResult submit_result = vkQueueSubmit(queue_, 1, &submit, fence);
        if (submit_result == VK_SUCCESS)
            vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device_, fence, nullptr);
        return submit_result == VK_SUCCESS;
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
        if (pick_readback_buffer_.buffer != VK_NULL_HANDLE)
            return true;
        return create_buffer(
            sizeof(std::uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY,
            pick_readback_buffer_);
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
            destroy_buffer(mesh.indices);
        }
        meshes_.clear();
        for (auto& [_, mesh] : virtual_meshes_)
        {
            destroy_buffer(mesh.vertices);
            destroy_buffer(mesh.indices);
        }
        virtual_meshes_.clear();
        for (auto& [_, texture] : textures_)
            destroy_texture(texture);
        textures_.clear();
        materials_.clear();
        environments_.clear();
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
        if (size == 0)
            return false;

        gpu_buffer staging;
        if (!create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, staging))
            return false;

        void* mapped{};
        if (vmaMapMemory(allocator_, staging.allocation, &mapped) != VK_SUCCESS)
        {
            destroy_buffer(staging);
            return false;
        }
        std::memcpy(mapped, source, static_cast<std::size_t>(size));
        vmaUnmapMemory(allocator_, staging.allocation);

        if (!create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, destination))
        {
            destroy_buffer(staging);
            return false;
        }

        VkCommandPool pool{};
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = graphics_queue_family_;
        if (vkCreateCommandPool(device_, &pool_info, nullptr, &pool) != VK_SUCCESS)
        {
            destroy_buffer(destination);
            destroy_buffer(staging);
            return false;
        }

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
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(command_buffer, staging.buffer, destination.buffer, 1, &copy);
        vkEndCommandBuffer(command_buffer);

        const bool submitted = submit_upload_commands(command_buffer);

        vkDestroyCommandPool(device_, pool, nullptr);
        destroy_buffer(staging);
        if (!submitted)
        {
            destroy_buffer(destination);
            return false;
        }
        return true;
    }

    bool upload_texture_image(const texture_data& data, gpu_texture& destination)
    {
        const auto format = vulkan_texture_format(data.format);
        if (!format || !texture_format_supported(*format))
            return false;

        const bool encoded = data.has_encoded_mips();
        const bool pixels = data.has_pixels();
        if (!encoded && !pixels)
            return false;

        const auto& upload_bytes = encoded ? data.encoded : data.pixels;
        if (upload_bytes.empty())
            return false;

        gpu_buffer staging;
        if (!create_buffer(
                upload_bytes.size(),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU,
                staging))
        {
            return false;
        }

        void* mapped{};
        if (vmaMapMemory(allocator_, staging.allocation, &mapped) != VK_SUCCESS)
        {
            destroy_buffer(staging);
            return false;
        }
        std::memcpy(mapped, upload_bytes.data(), upload_bytes.size());
        vmaUnmapMemory(allocator_, staging.allocation);

        const bool has_mip_payload = !data.mips.empty();
        const std::uint32_t mip_count = has_mip_payload
            ? static_cast<std::uint32_t>(data.mips.size())
            : 1u;

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = data.dimension == texture_dimension::texture_3d
            ? VK_IMAGE_TYPE_3D
            : VK_IMAGE_TYPE_2D;
        if (data.dimension == texture_dimension::cube)
            image.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        image.format = *format;
        image.extent = { data.width, data.height, data.dimension == texture_dimension::texture_3d ? data.depth : 1u };
        image.mipLevels = mip_count;
        image.arrayLayers = data.dimension == texture_dimension::texture_3d
            ? 1u
            : std::max(1u, data.dimension == texture_dimension::cube ? 6u : data.array_layers);
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &destination.image, &destination.allocation, nullptr) != VK_SUCCESS)
        {
            destroy_buffer(staging);
            return false;
        }

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = destination.image;
        view.viewType = data.dimension == texture_dimension::cube
            ? VK_IMAGE_VIEW_TYPE_CUBE
            : data.dimension == texture_dimension::texture_3d
                ? VK_IMAGE_VIEW_TYPE_3D
                : data.array_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        view.format = *format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = mip_count;
        view.subresourceRange.layerCount = image.arrayLayers;
        if (vkCreateImageView(device_, &view, nullptr, &destination.view) != VK_SUCCESS)
        {
            destroy_texture(destination);
            destroy_buffer(staging);
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
            destroy_buffer(staging);
            return false;
        }

        VkCommandPool pool{};
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = graphics_queue_family_;
        if (vkCreateCommandPool(device_, &pool_info, nullptr, &pool) != VK_SUCCESS)
        {
            destroy_texture(destination);
            destroy_buffer(staging);
            return false;
        }

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
        to_copy.image = destination.image;
        to_copy.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_copy.subresourceRange.levelCount = mip_count;
        to_copy.subresourceRange.layerCount = image.arrayLayers;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_copy);

        std::vector<VkBufferImageCopy> regions;
        if (has_mip_payload)
        {
            regions.reserve(data.mips.size());
            for (std::uint32_t mip = 0; mip < data.mips.size(); ++mip)
            {
                const auto& source_mip = data.mips[mip];
                VkBufferImageCopy copy{};
                copy.bufferOffset = static_cast<VkDeviceSize>(source_mip.offset);
                copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copy.imageSubresource.mipLevel = mip;
                copy.imageSubresource.layerCount = image.arrayLayers;
                copy.imageExtent = { source_mip.width, source_mip.height,
                    data.dimension == texture_dimension::texture_3d ? std::max(1u, data.depth >> mip) : 1u };
                regions.push_back(copy);
            }
        }
        else
        {
            VkBufferImageCopy copy{};
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.layerCount = image.arrayLayers;
            copy.imageExtent = { data.width, data.height,
                data.dimension == texture_dimension::texture_3d ? data.depth : 1u };
            regions.push_back(copy);
        }

        vkCmdCopyBufferToImage(
            command_buffer,
            staging.buffer,
            destination.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(regions.size()),
            regions.data());

        VkImageMemoryBarrier to_shader = to_copy;
        to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_shader);
        vkEndCommandBuffer(command_buffer);

        const bool submitted = submit_upload_commands(command_buffer);
        vkDestroyCommandPool(device_, pool, nullptr);
        destroy_buffer(staging);
        if (!submitted)
        {
            destroy_texture(destination);
            return false;
        }

        destination.format = *format;
        destination.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        destination.mip_count = mip_count;
        return true;
    }

    void upload_mesh(const mesh_upload_event& event)
    {
        if (!event.mesh || event.mesh->vertices.empty() || event.mesh->indices.empty())
            return;

        gpu_mesh mesh;
        const VkDeviceSize vertex_size = buffer_size(event.mesh->vertices.size(), sizeof(mesh_vertex));
        const VkDeviceSize index_size = buffer_size(event.mesh->indices.size(), sizeof(std::uint32_t));
        if (!upload_buffer(event.mesh->vertices.data(), vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices) ||
            !upload_buffer(event.mesh->indices.data(), index_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices))
        {
            destroy_buffer(mesh.vertices);
            destroy_buffer(mesh.indices);
            arc::error("render.vulkan", "Failed to upload mesh '" + event.label + "'");
            return;
        }

        mesh.index_count = static_cast<std::uint32_t>(event.mesh->indices.size());
        const std::uint64_t key = resource_key(event.handle);
        if (auto found = meshes_.find(key); found != meshes_.end())
        {
            destroy_buffer(found->second.vertices);
            destroy_buffer(found->second.indices);
        }
        meshes_[key] = std::move(mesh);
    }

    void upload_virtual_mesh(const virtual_mesh_upload_event& event)
    {
        if (!event.mesh || event.mesh->vertices.empty() || event.mesh->indices.empty() || event.mesh->clusters.empty())
            return;

        gpu_virtual_mesh mesh;
        const VkDeviceSize vertex_size = buffer_size(event.mesh->vertices.size(), sizeof(mesh_vertex));
        const VkDeviceSize index_size = buffer_size(event.mesh->indices.size(), sizeof(std::uint32_t));
        if (!upload_buffer(event.mesh->vertices.data(), vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices) ||
            !upload_buffer(event.mesh->indices.data(), index_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices))
        {
            destroy_buffer(mesh.vertices);
            destroy_buffer(mesh.indices);
            arc::error("render.vulkan", "Failed to upload virtual mesh '" + event.label + "'");
            return;
        }

        mesh.index_count = static_cast<std::uint32_t>(event.mesh->indices.size());
        mesh.clusters = event.mesh->clusters;
        const std::uint64_t key = resource_key(event.handle);
        if (auto found = virtual_meshes_.find(key); found != virtual_meshes_.end())
        {
            destroy_buffer(found->second.vertices);
            destroy_buffer(found->second.indices);
        }
        virtual_meshes_[key] = std::move(mesh);
    }

    void upload_texture(const texture_upload_event& event)
    {
        if (!event.texture)
            return;

        gpu_texture texture{ .data = *event.texture };
        const bool uploaded = upload_texture_image(*event.texture, texture);
        if (!uploaded && event.texture->dds && event.texture->compressed)
        {
            arc::warn(
                "render.vulkan",
                "DDS texture '" + event.label + "' uses a compressed format unsupported by this Vulkan device; using fallback descriptors");
        }
        else if (!uploaded && !event.texture->has_pixels() && !event.texture->encoded.empty())
        {
            arc::debug("render.vulkan", "Texture '" + event.label + "' kept as encoded data until image decoding is available");
        }

        const std::uint64_t key = resource_key(event.handle);
        if (auto found = textures_.find(key); found != textures_.end())
            destroy_texture(found->second);
        textures_[key] = std::move(texture);
    }

    void upload_material(const material_upload_event& event)
    {
        if (!event.material)
            return;

        auto& material = materials_[resource_key(event.handle)];
        material.data = *event.material;
    }

    void upload_environment(const environment_upload_event& event)
    {
        if (!event.environment)
            return;

        auto environment = *event.environment;
        if (!environment.prefiltered)
        {
            environment.diffuse_irradiance = environment.fallback_color;
            environment.diffuse_intensity = environment.intensity;
        }
        environments_[resource_key(event.handle)] = gpu_environment{ .data = std::move(environment) };
        active_environment_ = event.handle;
    }

    const environment_desc* active_environment() const noexcept
    {
        const auto found = environments_.find(resource_key(active_environment_));
        return found == environments_.end() ? nullptr : &found->second.data;
    }

    void update_environment_profile(const environment_desc* lighting_environment)
    {
        auto& profile = last_profile_.environment;
        profile = {};
        profile.enabled = frame_environment_.enabled;
        profile.sky_visible = frame_environment_.enabled && frame_environment_.sky_visible;
        profile.affects_lighting = frame_environment_.affect_lighting && frame_environment_.lighting.enabled;
        switch (frame_environment_.source)
        {
        case sky_source_mode::physical_atmosphere: profile.source = "Physical atmosphere"; break;
        case sky_source_mode::hdri: profile.source = "HDRI"; break;
        case sky_source_mode::solid_color: profile.source = "Solid color"; break;
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

    void update_light_buffer()
    {
        if (light_buffer_.buffer == VK_NULL_HANDLE)
        {
            if (!create_buffer(
                    sizeof(scene_lighting_data),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU,
                    light_buffer_))
            {
                arc::warn("render.vulkan", "Failed to allocate scene light buffer");
                return;
            }
        }

        void* mapped{};
        if (vmaMapMemory(allocator_, light_buffer_.allocation, &mapped) != VK_SUCCESS)
            return;
        std::memcpy(mapped, &frame_lighting_, sizeof(frame_lighting_));
        vmaUnmapMemory(allocator_, light_buffer_.allocation);
    }

    void warn_about_skipped_lights(const scene_lighting_data& lighting)
    {
        if (lighting.skipped_directional_count > 0)
            arc::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_directional_count) + " directional light(s) over the v1 cap");
        if (lighting.skipped_point_count > 0)
            arc::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_point_count) + " point light(s) over the v1 cap");
        if (lighting.skipped_spot_count > 0)
            arc::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_spot_count) + " spot light(s) over the v1 cap");
    }

    static math::vector3f vector_sub(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
    {
        return { lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2] };
    }

    static math::vector3f vector_mul(const math::vector3f& value, float scale) noexcept
    {
        return { value[0] * scale, value[1] * scale, value[2] * scale };
    }

    static math::vector3f vector_add(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
    {
        return { lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2] };
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
        const math::vector3f origin{ draw.model(0, 3), draw.model(1, 3), draw.model(2, 3) };
        math::vector3f color{
            frame_lighting_.ambient_color_intensity[0] * frame_lighting_.ambient_color_intensity[3],
            frame_lighting_.ambient_color_intensity[1] * frame_lighting_.ambient_color_intensity[3],
            frame_lighting_.ambient_color_intensity[2] * frame_lighting_.ambient_color_intensity[3]
        };
        math::vector3f weighted_direction{};
        float total_weight{};

        for (std::uint32_t index = 0; index < frame_lighting_.directional_count; ++index)
        {
            const auto& light = frame_lighting_.directional_lights[index];
            const float contribution = std::max(light.direction_intensity[3], 0.0f);
            color = vector_add(color, vector_mul({ light.color_flags[0], light.color_flags[1], light.color_flags[2] }, contribution));
            weighted_direction = vector_add(weighted_direction, vector_mul(
                { light.direction_intensity[0], light.direction_intensity[1], light.direction_intensity[2] },
                contribution));
            total_weight += contribution;
        }

        for (std::uint32_t index = 0; index < frame_lighting_.point_count; ++index)
        {
            const auto& light = frame_lighting_.point_lights[index];
            const math::vector3f position{ light.position_range[0], light.position_range[1], light.position_range[2] };
            const float range = std::max(light.position_range[3], 0.001f);
            const math::vector3f to_light = vector_sub(position, origin);
            const float distance_sq = std::max(vector_dot(to_light, to_light), 0.000001f);
            const float attenuation = std::max(0.0f, 1.0f - std::sqrt(distance_sq) / range);
            const float contribution = light.color_intensity[3] * attenuation * attenuation;
            color = vector_add(color, vector_mul({ light.color_intensity[0], light.color_intensity[1], light.color_intensity[2] }, contribution));
            weighted_direction = vector_add(weighted_direction, vector_mul(vector_mul(vector_normalize(to_light), -1.0f), contribution));
            total_weight += contribution;
        }

        for (std::uint32_t index = 0; index < frame_lighting_.spot_count; ++index)
        {
            const auto& light = frame_lighting_.spot_lights[index];
            const math::vector3f position{ light.position_range[0], light.position_range[1], light.position_range[2] };
            const float range = std::max(light.position_range[3], 0.001f);
            const math::vector3f to_light = vector_sub(position, origin);
            const float attenuation = std::max(0.0f, 1.0f - std::sqrt(vector_dot(to_light, to_light)) / range);
            const math::vector3f light_forward = vector_normalize({ light.direction_inner_angle[0], light.direction_inner_angle[1], light.direction_inner_angle[2] });
            const float cone_cos = vector_dot(vector_mul(vector_normalize(to_light), -1.0f), light_forward);
            const float inner = std::cos(light.direction_inner_angle[3]);
            const float outer = std::cos(light.params[0]);
            const float cone = outer == inner ? 1.0f : std::clamp((cone_cos - outer) / (inner - outer), 0.0f, 1.0f);
            const float contribution = light.color_intensity[3] * attenuation * attenuation * cone;
            color = vector_add(color, vector_mul({ light.color_intensity[0], light.color_intensity[1], light.color_intensity[2] }, contribution));
            weighted_direction = vector_add(weighted_direction, vector_mul(vector_mul(vector_normalize(to_light), -1.0f), contribution));
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
        if (!handle.valid())
            return false;
        const auto found = textures_.find(resource_key(handle));
        return found != textures_.end() &&
            found->second.view != VK_NULL_HANDLE &&
            found->second.sampler != VK_NULL_HANDLE;
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
        constants.camera_position[3] = frame_environment_.enabled
            ? std::max(frame_environment_.atmosphere.exposure, 0.001f)
            : 1.0f;
        constants.fog_params[3] = draw.object_id.valid()
            ? static_cast<float>(draw.object_id.index + 1u)
            : 0.0f;

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
            constants.material_params[0] = desc.normal_scale;
            constants.material_params[1] = desc.occlusion_strength;
            constants.material_params[2] = desc.emissive_strength;
            constants.material_params[3] = static_cast<float>(desc.alpha_mode);
            constants.light_color[3] =
                (texture_ready(desc.base_color_texture) ? 1.0f : 0.0f) +
                (texture_ready(desc.metallic_roughness_texture) ? 2.0f : 0.0f) +
                (texture_ready(desc.normal_texture) ? 4.0f : 0.0f) +
                (texture_ready(desc.occlusion_texture) ? 8.0f : 0.0f) +
                (texture_ready(desc.emissive_texture) ? 16.0f : 0.0f);
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

    VkDescriptorSet material_descriptor_set_for(const draw_mesh_event& draw) const noexcept
    {
        if (const auto material = materials_.find(resource_key(draw.material)); material != materials_.end())
        {
            const auto slot = current_frame_slot();
            if (slot < material->second.descriptor_sets.size() && material->second.descriptor_sets[slot] != VK_NULL_HANDLE)
                return material->second.descriptor_sets[slot];
        }
        const auto slot = current_frame_slot();
        return slot < white_descriptor_sets_.size() ? white_descriptor_sets_[slot] : VK_NULL_HANDLE;
    }

    void destroy_mesh_pipeline() noexcept
    {
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
        if (gbuffer_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, gbuffer_pipeline_, nullptr);
            gbuffer_pipeline_ = VK_NULL_HANDLE;
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

#if ARC_RENDER_VULKAN_ENABLE_IMGUI
    VkShaderModule create_shader_module(const std::uint32_t* code, std::size_t word_count)
    {
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = word_count * sizeof(std::uint32_t);
        info.pCode = code;

        VkShaderModule module{};
        if (vkCreateShaderModule(device_, &info, nullptr, &module) != VK_SUCCESS)
            return VK_NULL_HANDLE;
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
    }

    std::uint32_t frame_resource_count() const noexcept
    {
        return std::max(1u, window_.ImageCount);
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
            if (ready)
                return true;
        }

        wait_for_in_flight_frames();
        for (auto& buffer : shadow_uniform_buffers_)
            destroy_buffer(buffer);
        shadow_uniform_buffers_.clear();
        shadow_uniform_buffers_.resize(count);

        for (auto& buffer : shadow_uniform_buffers_)
        {
            if (!create_buffer(
                    sizeof(shadow_uniform_data),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU,
                    buffer))
                return false;
        }
        return true;
    }

    gpu_buffer* current_shadow_uniform_buffer() noexcept
    {
        const auto slot = current_frame_slot();
        if (slot >= shadow_uniform_buffers_.size())
            return nullptr;
        return &shadow_uniform_buffers_[slot];
    }

    const gpu_buffer* shadow_uniform_buffer_for_slot(std::uint32_t slot) const noexcept
    {
        if (slot >= shadow_uniform_buffers_.size())
            return nullptr;
        return &shadow_uniform_buffers_[slot];
    }

    bool ensure_shadow_resources(const shadow_settings& settings)
    {
        const std::uint32_t resolution = std::clamp(settings.resolution, 256u, 8192u);
        if (shadow_atlas_.image != VK_NULL_HANDLE && shadow_atlas_.resolution == resolution)
            return true;

        wait_for_in_flight_frames();
        destroy_shadow_resources();

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = depth_format_;
        image.extent = { resolution, resolution, 1 };
        image.mipLevels = 1;
        image.arrayLayers = directional_shadow_cascade_count;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &shadow_atlas_.image, &shadow_atlas_.allocation, nullptr) != VK_SUCCESS)
        {
            arc::warn("render.vulkan", "Failed to allocate directional shadow atlas");
            return false;
        }

        VkImageViewCreateInfo array_view{};
        array_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        array_view.image = shadow_atlas_.image;
        array_view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        array_view.format = depth_format_;
        array_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        array_view.subresourceRange.levelCount = 1;
        array_view.subresourceRange.layerCount = directional_shadow_cascade_count;
        if (vkCreateImageView(device_, &array_view, nullptr, &shadow_atlas_.array_view) != VK_SUCCESS)
        {
            destroy_shadow_resources();
            return false;
        }

        for (std::uint32_t layer = 0; layer < directional_shadow_cascade_count; ++layer)
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

    bool ensure_material_descriptor_sets(gpu_material& material)
    {
        const auto count = frame_resource_count();
        if (material.descriptor_sets.size() != count)
            material.descriptor_sets.assign(count, VK_NULL_HANDLE);
        for (auto& set : material.descriptor_sets)
        {
            if (set != VK_NULL_HANDLE)
                continue;
            set = allocate_material_descriptor_set();
            if (set == VK_NULL_HANDLE)
                return false;
        }
        return true;
    }

    bool ensure_white_descriptor_sets()
    {
        const auto count = frame_resource_count();
        if (white_descriptor_sets_.size() != count)
            white_descriptor_sets_.assign(count, VK_NULL_HANDLE);
        for (auto& set : white_descriptor_sets_)
        {
            if (set != VK_NULL_HANDLE)
                continue;
            set = allocate_material_descriptor_set();
            if (set == VK_NULL_HANDLE)
                return false;
        }
        return true;
    }

    bool ensure_sky_descriptor_sets()
    {
        if (!ensure_white_texture())
            return false;
        const auto count = frame_resource_count();
        if (sky_descriptor_sets_.size() != count)
            sky_descriptor_sets_.assign(count, VK_NULL_HANDLE);
        for (auto& set : sky_descriptor_sets_)
        {
            if (set == VK_NULL_HANDLE)
                set = allocate_material_descriptor_set();
            if (set == VK_NULL_HANDLE)
                return false;
        }
        return true;
    }

    VkDescriptorSet update_current_sky_descriptor_set()
    {
        if (!ensure_sky_descriptor_sets())
            return VK_NULL_HANDLE;
        const auto slot = current_frame_slot();
        if (slot >= sky_descriptor_sets_.size())
            return VK_NULL_HANDLE;
        const auto set = sky_descriptor_sets_[slot];
        update_material_descriptor_set(set, nullptr, slot);

        VkSampler sampler = white_sampler_;
        VkImageView view = white_view_;
        if (frame_environment_.source == sky_source_mode::hdri && frame_environment_.hdri_texture.valid())
        {
            const auto found = textures_.find(resource_key(frame_environment_.hdri_texture));
            if (found != textures_.end() && found->second.view != VK_NULL_HANDLE && found->second.sampler != VK_NULL_HANDLE)
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

    void update_material_descriptor_set(VkDescriptorSet descriptor_set, const material_desc* material, std::uint32_t frame_slot)
    {
        const auto* shadow_buffer_resource = shadow_uniform_buffer_for_slot(frame_slot);
        if (descriptor_set == VK_NULL_HANDLE ||
            white_view_ == VK_NULL_HANDLE ||
            shadow_atlas_.array_view == VK_NULL_HANDLE ||
            shadow_buffer_resource == nullptr ||
            shadow_buffer_resource->buffer == VK_NULL_HANDLE)
            return;

        const auto resolve_texture = [&](texture_handle handle, VkSampler& sampler, VkImageView& view) {
            sampler = white_sampler_;
            view = white_view_;
            if (!handle.valid())
                return;
            if (const auto found = textures_.find(resource_key(handle)); found != textures_.end())
            {
                if (found->second.view != VK_NULL_HANDLE && found->second.sampler != VK_NULL_HANDLE)
                {
                    sampler = found->second.sampler;
                    view = found->second.view;
                }
            }
        };

        std::array<VkDescriptorImageInfo, 6> image_infos{};
        VkSampler sampler{};
        VkImageView view{};
        resolve_texture(material ? material->base_color_texture : texture_handle{}, sampler, view);
        image_infos[0] = { sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        resolve_texture(material ? material->metallic_roughness_texture : texture_handle{}, sampler, view);
        image_infos[1] = { sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        resolve_texture(material ? material->normal_texture : texture_handle{}, sampler, view);
        image_infos[2] = { sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        resolve_texture(material ? material->occlusion_texture : texture_handle{}, sampler, view);
        image_infos[3] = { sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        resolve_texture(material ? material->emissive_texture : texture_handle{}, sampler, view);
        image_infos[4] = { sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

        image_infos[5].sampler = shadow_atlas_.sampler;
        image_infos[5].imageView = shadow_atlas_.array_view;
        image_infos[5].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorBufferInfo shadow_buffer{};
        shadow_buffer.buffer = shadow_buffer_resource->buffer;
        shadow_buffer.offset = 0;
        shadow_buffer.range = sizeof(shadow_uniform_data);

        std::array<VkWriteDescriptorSet, 7> writes{};
        for (std::uint32_t binding_index = 0; binding_index < image_infos.size(); ++binding_index)
        {
            writes[binding_index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding_index].dstSet = descriptor_set;
            writes[binding_index].dstBinding = binding_index;
            writes[binding_index].descriptorCount = 1;
            writes[binding_index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[binding_index].pImageInfo = &image_infos[binding_index];
        }
        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet = descriptor_set;
        writes[6].dstBinding = 6;
        writes[6].descriptorCount = 1;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[6].pBufferInfo = &shadow_buffer;
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void update_material_descriptor_sets(gpu_material& material)
    {
        if (!ensure_material_descriptor_sets(material))
            return;
        for (std::uint32_t frame_slot = 0; frame_slot < material.descriptor_sets.size(); ++frame_slot)
            update_material_descriptor_set(material.descriptor_sets[frame_slot], &material.data, frame_slot);
    }

    void update_white_descriptor_sets()
    {
        if (!ensure_white_descriptor_sets())
            return;
        for (std::uint32_t frame_slot = 0; frame_slot < white_descriptor_sets_.size(); ++frame_slot)
            update_material_descriptor_set(white_descriptor_sets_[frame_slot], nullptr, frame_slot);
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
            update_material_descriptor_set(white_descriptor_sets_[frame_slot], nullptr, frame_slot);
        for (auto& [_, material] : materials_)
        {
            if (ensure_material_descriptor_sets(material) && frame_slot < material.descriptor_sets.size())
                update_material_descriptor_set(material.descriptor_sets[frame_slot], &material.data, frame_slot);
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
        if (vkAllocateDescriptorSets(device_, &descriptor_allocate, &set) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return set;
    }

    bool ensure_white_texture()
    {
        const auto shadow_resolution = shadow_atlas_.resolution == 0 ? 2048u : shadow_atlas_.resolution;
        if (!ensure_shadow_uniform_buffers() || !ensure_shadow_resources({ .enabled = false, .resolution = shadow_resolution }))
            return false;

        if (white_descriptor_set_layout_ != VK_NULL_HANDLE &&
            white_descriptor_pool_ != VK_NULL_HANDLE &&
            white_view_ != VK_NULL_HANDLE &&
            white_sampler_ != VK_NULL_HANDLE)
        {
            return ensure_white_descriptor_sets();
        }

        std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
        for (std::uint32_t binding_index = 0; binding_index < 6; ++binding_index)
        {
            bindings[binding_index].binding = binding_index;
            bindings[binding_index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[binding_index].descriptorCount = 1;
            bindings[binding_index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        bindings[6].binding = 6;
        bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

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
        image.extent = { 1, 1, 1 };
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
        if (vkCreateImageView(device_, &view, nullptr, &white_view_) != VK_SUCCESS)
            return false;

        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &sampler, nullptr, &white_sampler_) != VK_SUCCESS)
            return false;

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
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_copy);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = { 1, 1, 1 };
        vkCmdCopyBufferToImage(command_buffer, staging.buffer, white_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        VkImageMemoryBarrier to_shader = to_copy;
        to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_shader);
        vkEndCommandBuffer(command_buffer);

        submit_upload_commands(command_buffer);
        vkDestroyCommandPool(device_, pool, nullptr);
        destroy_buffer(staging);

        std::array<VkDescriptorPoolSize, 2> pool_sizes{};
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_sizes[0].descriptorCount = 73728;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_sizes[1].descriptorCount = 12288;
        VkDescriptorPoolCreateInfo descriptor_pool{};
        descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptor_pool.maxSets = 12288;
        descriptor_pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        descriptor_pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &descriptor_pool, nullptr, &white_descriptor_pool_) != VK_SUCCESS)
            return false;

        update_all_material_descriptor_sets();
        return true;
    }

    bool ensure_mesh_pipeline()
    {
        if (mesh_pipeline_ != VK_NULL_HANDLE)
            return true;
        if (!ensure_white_texture())
            return false;

        VkShaderModule vert = create_shader_module(
            builtin::default_phong_vert_spv,
            std::size(builtin::default_phong_vert_spv));
        VkShaderModule frag = create_shader_module(
            builtin::default_phong_frag_spv,
            std::size(builtin::default_phong_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
            return false;

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
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &mesh_pipeline_layout_) != VK_SUCCESS)
            return false;

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
        attributes[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position) };
        attributes[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal) };
        attributes[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord) };
        attributes[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color) };
        attributes[4] = { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent) };

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

        const std::array<VkDynamicState, 2> dynamic_states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &viewport_format_;
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

        const VkResult result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_pipeline_);
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
            const VkResult blend_result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_transparent_pipeline_);
            if (blend_result != VK_SUCCESS)
                arc::warn("render.vulkan", "Vulkan transparent mesh pipeline creation failed; blended materials will render opaque");
            color_attachment = {};
            color_attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            depth.depthWriteEnable = VK_TRUE;
        }
        if (result == VK_SUCCESS && capabilities_.fill_mode_non_solid)
        {
            raster.polygonMode = VK_POLYGON_MODE_LINE;
            depth.depthWriteEnable = VK_FALSE;
            const VkResult wire_result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_wire_pipeline_);
            if (wire_result != VK_SUCCESS)
                arc::warn("render.vulkan", "Vulkan wireframe pipeline creation failed; shaded rendering will continue");
        }
        else if (result == VK_SUCCESS && !capabilities_.fill_mode_non_solid && !wireframe_warning_reported_)
        {
            arc::warn("render.vulkan", "Vulkan device does not support fillModeNonSolid; wireframe rendering is disabled");
            wireframe_warning_reported_ = true;
        }
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return result == VK_SUCCESS;
    }

    bool ensure_gbuffer_pipeline()
    {
        if (gbuffer_pipeline_ != VK_NULL_HANDLE)
            return true;
        if (!ensure_mesh_pipeline())
            return false;

        VkShaderModule vert = create_shader_module(builtin::gbuffer_vert_spv, std::size(builtin::gbuffer_vert_spv));
        VkShaderModule frag = create_shader_module(builtin::gbuffer_frag_spv, std::size(builtin::gbuffer_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
            return false;

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
        attributes[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position) };
        attributes[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal) };
        attributes[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord) };
        attributes[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color) };
        attributes[4] = { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent) };

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

        std::array<VkPipelineColorBlendAttachmentState, 5> attachments{};
        for (auto& attachment : attachments)
        {
            attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        }
        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        color_blend.pAttachments = attachments.data();

        const std::array<VkDynamicState, 2> dynamic_states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        const std::array<VkFormat, 5> color_formats{
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R16G16_SFLOAT,
            VK_FORMAT_R32_UINT
        };
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

        const VkResult result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &gbuffer_pipeline_);
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        if (result != VK_SUCCESS)
            arc::warn("render.vulkan", "Vulkan G-buffer pipeline creation failed; falling back to forward rendering");
        return result == VK_SUCCESS;
    }

    bool ensure_gbuffer_descriptor_set()
    {
        if (gbuffer_descriptor_set_ != VK_NULL_HANDLE)
            return true;
        if (!ensure_white_texture() ||
            gbuffer_albedo_.view == VK_NULL_HANDLE ||
            gbuffer_normal_.view == VK_NULL_HANDLE ||
            gbuffer_material_.view == VK_NULL_HANDLE ||
            gbuffer_motion_.view == VK_NULL_HANDLE ||
            gbuffer_object_id_.view == VK_NULL_HANDLE)
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
            if (vkCreateSampler(device_, &sampler, nullptr, &gbuffer_sampler_) != VK_SUCCESS)
                return false;
        }

        if (gbuffer_descriptor_set_layout_ == VK_NULL_HANDLE)
        {
            std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
            for (std::uint32_t index = 0; index < bindings.size(); ++index)
            {
                bindings[index].binding = index;
                bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bindings[index].descriptorCount = 1;
                bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }

            VkDescriptorSetLayoutCreateInfo layout{};
            layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
            layout.pBindings = bindings.data();
            if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &gbuffer_descriptor_set_layout_) != VK_SUCCESS)
                return false;
        }

        if (gbuffer_descriptor_pool_ == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_size.descriptorCount = 5;
            VkDescriptorPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool.maxSets = 1;
            pool.poolSizeCount = 1;
            pool.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(device_, &pool, nullptr, &gbuffer_descriptor_pool_) != VK_SUCCESS)
                return false;
        }

        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = gbuffer_descriptor_pool_;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &gbuffer_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &gbuffer_descriptor_set_) != VK_SUCCESS)
            return false;

        update_gbuffer_descriptor_set();
        return true;
    }

    void update_gbuffer_descriptor_set()
    {
        if (gbuffer_descriptor_set_ == VK_NULL_HANDLE)
            return;

        std::array<VkDescriptorImageInfo, 5> images{};
        const VkSampler sampler = gbuffer_sampler_ != VK_NULL_HANDLE ? gbuffer_sampler_ : white_sampler_;
        images[0] = { sampler, gbuffer_albedo_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        images[1] = { sampler, gbuffer_normal_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        images[2] = { sampler, gbuffer_material_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        images[3] = { sampler, gbuffer_object_id_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        images[4] = { sampler, gbuffer_motion_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

        std::array<VkWriteDescriptorSet, 5> writes{};
        for (std::uint32_t index = 0; index < writes.size(); ++index)
        {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = gbuffer_descriptor_set_;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[index].pImageInfo = &images[index];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    bool ensure_deferred_pipeline()
    {
        if (deferred_pipeline_ != VK_NULL_HANDLE)
            return true;
        if (!ensure_gbuffer_descriptor_set())
            return false;

        VkShaderModule vert = create_shader_module(builtin::deferred_lighting_vert_spv, std::size(builtin::deferred_lighting_vert_spv));
        VkShaderModule frag = create_shader_module(builtin::deferred_lighting_frag_spv, std::size(builtin::deferred_lighting_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
            return false;

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
        const std::array<VkDynamicState, 2> dynamic_states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &viewport_format_;

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

        const VkResult result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &deferred_pipeline_);
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        if (result != VK_SUCCESS)
            arc::warn("render.vulkan", "Vulkan deferred lighting pipeline creation failed; falling back to forward rendering");
        return result == VK_SUCCESS;
    }

    bool ensure_sky_pipeline()
    {
        if (sky_pipeline_ != VK_NULL_HANDLE)
            return true;
        if (!ensure_white_texture())
            return false;

        VkShaderModule vert = create_shader_module(
            builtin::sky_atmosphere_vert_spv,
            std::size(builtin::sky_atmosphere_vert_spv));
        VkShaderModule frag = create_shader_module(
            builtin::sky_atmosphere_frag_spv,
            std::size(builtin::sky_atmosphere_frag_spv));
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
            return false;

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

        const std::array<VkDynamicState, 2> dynamic_states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &viewport_format_;
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

        const VkResult result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &sky_pipeline_);
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return result == VK_SUCCESS;
    }

    void destroy_graph_image(graph_image& image) noexcept
    {
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
    }

    bool ensure_graph_image(
        graph_image& target,
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format,
        VkImageUsageFlags usage,
        VkImageAspectFlags aspect)
    {
        if (target.image != VK_NULL_HANDLE && target.format == format && target.aspect == aspect)
            return true;

        destroy_graph_image(target);

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = format;
        image.extent = { width, height, 1 };
        image.mipLevels = 1;
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
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device_, &view, nullptr, &target.view) != VK_SUCCESS)
        {
            destroy_graph_image(target);
            return false;
        }

        target.format = format;
        target.aspect = aspect;
        target.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    }

    void transition_graph_image(VkCommandBuffer command_buffer, graph_image& image, VkImageLayout new_layout)
    {
        if (image.image == VK_NULL_HANDLE || image.layout == new_layout)
            return;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = image.layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image;
        barrier.subresourceRange.aspectMask = image.aspect;
        barrier.subresourceRange.levelCount = 1;
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
            src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }

        if (new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        image.layout = new_layout;
    }

    bool ensure_deferred_targets(std::uint32_t width, std::uint32_t height)
    {
        const VkImageUsageFlags sampled_color_usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        const bool ok = ensure_graph_image(gbuffer_albedo_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
            ensure_graph_image(gbuffer_normal_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
            ensure_graph_image(gbuffer_material_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
            ensure_graph_image(gbuffer_motion_, width, height, VK_FORMAT_R16G16_SFLOAT, sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
            ensure_graph_image(gbuffer_object_id_, width, height, VK_FORMAT_R32_UINT, sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
            ensure_graph_image(selection_mask_, width, height, VK_FORMAT_R8_UNORM, sampled_color_usage, VK_IMAGE_ASPECT_COLOR_BIT);
        if (ok)
            update_gbuffer_descriptor_set();
        return ok;
    }

    void ensure_viewport(std::uint32_t width, std::uint32_t height)
    {
        width = std::max(1u, width);
        height = std::max(1u, height);
        if (viewport_image_ != VK_NULL_HANDLE && viewport_width_ == width && viewport_height_ == height)
            return;

        wait_for_in_flight_frames();
        destroy_viewport();

        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = viewport_format_;
        image.extent = { width, height, 1 };
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image, &allocation, &viewport_image_, &viewport_allocation_, nullptr) != VK_SUCCESS)
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

        if (imgui_initialized_)
            viewport_descriptor_ = ImGui_ImplVulkan_AddTexture(viewport_sampler_, viewport_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        viewport_width_ = width;
        viewport_height_ = height;
        viewport_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        ensure_deferred_targets(width, height);

        VkImageCreateInfo depth_image{};
        depth_image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        depth_image.imageType = VK_IMAGE_TYPE_2D;
        depth_image.format = depth_format_;
        depth_image.extent = { width, height, 1 };
        depth_image.mipLevels = 1;
        depth_image.arrayLayers = 1;
        depth_image.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_image.tiling = VK_IMAGE_TILING_OPTIMAL;
        depth_image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        VmaAllocationCreateInfo depth_allocation{};
        depth_allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &depth_image, &depth_allocation, &viewport_depth_image_, &viewport_depth_allocation_, nullptr) != VK_SUCCESS)
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
        if (viewport_descriptor_ != VK_NULL_HANDLE)
        {
            ImGui_ImplVulkan_RemoveTexture(viewport_descriptor_);
            viewport_descriptor_ = VK_NULL_HANDLE;
        }
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
        destroy_graph_image(gbuffer_motion_);
        destroy_graph_image(gbuffer_object_id_);
        destroy_graph_image(selection_mask_);
        if (gbuffer_descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, gbuffer_descriptor_pool_, nullptr);
            gbuffer_descriptor_pool_ = VK_NULL_HANDLE;
            gbuffer_descriptor_set_ = VK_NULL_HANDLE;
        }
    }

    void transition_viewport(VkCommandBuffer command_buffer, VkImageLayout new_layout)
    {
        if (viewport_image_ == VK_NULL_HANDLE || viewport_layout_ == new_layout)
            return;

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

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        viewport_layout_ = new_layout;
    }

    void transition_depth(VkCommandBuffer command_buffer, VkImageLayout new_layout)
    {
        if (viewport_depth_image_ == VK_NULL_HANDLE || viewport_depth_layout_ == new_layout)
            return;

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
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);
        viewport_depth_layout_ = new_layout;
    }

    const directional_light_event* active_directional_shadow_light() const noexcept
    {
        if (!frame_shadows_enabled_)
            return nullptr;
        for (const auto& light : frame_directional_lights_)
        {
            if (light.enabled && light.casts_shadows && light.shadow.enabled)
                return &light;
        }
        return nullptr;
    }

    void prepare_frame_gpu_resources()
    {
        const auto* light = active_directional_shadow_light();
        auto settings = light ? light->shadow : shadow_settings{ .enabled = false, .resolution = 2048 };
        settings.resolution = std::min(settings.resolution, resolved_config_.directional_shadow_resolution);
        if (ensure_shadow_uniform_buffers() && ensure_shadow_resources(settings))
        {
            update_shadow_uniform(build_shadow_uniform(light));
        }

        if (ensure_mesh_pipeline())
            update_current_material_descriptor_sets();

        if (light && !frame_shadow_draws_.empty())
            ensure_shadow_pipeline();
    }

    void transition_shadow_atlas(VkCommandBuffer command_buffer, VkImageLayout new_layout)
    {
        if (shadow_atlas_.image == VK_NULL_HANDLE || shadow_atlas_.layout == new_layout)
            return;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = shadow_atlas_.layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = shadow_atlas_.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = directional_shadow_cascade_count;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
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
            src_stage = shadow_atlas_.layout == VK_IMAGE_LAYOUT_UNDEFINED
                ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                : VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        shadow_atlas_.layout = new_layout;
    }

    shadow_uniform_data build_shadow_uniform(const directional_light_event* light) const noexcept
    {
        shadow_uniform_data data{};
        const auto identity = math::identity<float, 4>();
        for (auto& matrix : data.light_view_projection)
            std::copy(identity.data(), identity.data() + 16, matrix);

        constexpr std::array<float, directional_shadow_cascade_count> cascade_radii{ 6.0f, 14.0f, 32.0f, 64.0f };
        for (std::size_t index = 0; index < cascade_radii.size(); ++index)
            data.cascade_splits[index] = cascade_radii[index];

        if (!light)
        {
            data.params[0] = 0.0f;
            return data;
        }

        auto light_direction = math::normalize(light->direction);
        if (math::length_squared(light_direction) < 0.0001f)
            light_direction = math::vector3f{ 0.35f, -0.85f, -0.40f };
        const auto up = std::abs(math::dot(light_direction, math::vector3f{ 0.0f, 1.0f, 0.0f })) > 0.95f
            ? math::vector3f{ 0.0f, 0.0f, 1.0f }
            : math::vector3f{ 0.0f, 1.0f, 0.0f };

        auto camera_forward = math::normalize(frame_camera_.forward, 0.0f);
        if (math::length_squared(camera_forward) < 0.0001f)
            camera_forward = math::vector3f{ 0.0f, 0.0f, -1.0f };

        for (std::uint32_t cascade = 0; cascade < directional_shadow_cascade_count; ++cascade)
        {
            const float cascade_radius = cascade_radii[cascade];
            math::vector3f center = math::add(frame_camera_.position, math::mul(camera_forward, cascade_radius));

            auto view = look_at_rh(math::sub(center, math::mul(light_direction, cascade_radius * 2.5f)), center, up);
            const float texel_size = (cascade_radius * 2.0f) / static_cast<float>(std::max(1u, shadow_atlas_.resolution));
            if (texel_size > 0.0f)
            {
                const auto light_center = math::transform_point(view, center);
                const float snapped_x = std::round(light_center[0] / texel_size) * texel_size;
                const float snapped_y = std::round(light_center[1] / texel_size) * texel_size;
                const auto right = matrix_row3(view, 0);
                const auto light_up = matrix_row3(view, 1);
                center = math::add(center, math::mul(right, snapped_x - light_center[0]));
                center = math::add(center, math::mul(light_up, snapped_y - light_center[1]));
                view = look_at_rh(math::sub(center, math::mul(light_direction, cascade_radius * 2.5f)), center, up);
            }

            const auto projection = orthographic_rh_zo(cascade_radius * 2.0f, cascade_radius * 2.0f, 0.1f, cascade_radius * 5.0f);
            const auto light_view_projection = math::matmul(projection, view);
            std::copy(light_view_projection.data(), light_view_projection.data() + 16, data.light_view_projection[cascade]);
            data.cascade_texel_size[cascade] = texel_size;
        }

        data.params[0] = std::clamp(light->shadow.strength, 0.0f, 1.0f);
        data.params[1] = std::max(0.0f, light->shadow.bias);
        data.params[2] = std::max(0.0f, light->shadow.normal_bias);
        const auto filter = resolved_config_.quality == render_quality_tier::low
            ? shadow_filter::pcf_3x3
            : static_cast<shadow_filter>(std::min(
                static_cast<unsigned>(light->shadow.filter),
                static_cast<unsigned>(shadow_filter::pcf_5x5)));
        data.params[3] = static_cast<float>(filter);
        return data;
    }

    void update_shadow_uniform(const shadow_uniform_data& data)
    {
        if (!ensure_shadow_uniform_buffers())
            return;
        auto* shadow_buffer = current_shadow_uniform_buffer();
        if (shadow_buffer == nullptr || shadow_buffer->buffer == VK_NULL_HANDLE)
            return;
        void* mapped{};
        if (vmaMapMemory(allocator_, shadow_buffer->allocation, &mapped) != VK_SUCCESS)
            return;
        std::memcpy(mapped, &data, sizeof(data));
        vmaUnmapMemory(allocator_, shadow_buffer->allocation);
    }

    bool ensure_shadow_pipeline()
    {
        if (shadow_pipeline_ != VK_NULL_HANDLE)
            return true;

        VkShaderModule vert = create_shader_module(
            builtin::shadow_depth_vert_spv,
            std::size(builtin::shadow_depth_vert_spv));
        if (vert == VK_NULL_HANDLE)
            return false;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push.offset = 0;
        push.size = sizeof(mesh_push_constants);

        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &shadow_pipeline_layout_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, vert, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = vert;
        stage.pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(mesh_vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputAttributeDescription attribute{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position) };

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding;
        vertex_input.vertexAttributeDescriptionCount = 1;
        vertex_input.pVertexAttributeDescriptions = &attribute;

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

        const std::array<VkDynamicState, 2> dynamic_states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
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
        pipeline.stageCount = 1;
        pipeline.pStages = &stage;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input_assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &depth;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = shadow_pipeline_layout_;
        pipeline.renderPass = VK_NULL_HANDLE;

        const VkResult result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &shadow_pipeline_);
        vkDestroyShaderModule(device_, vert, nullptr);
        if (result != VK_SUCCESS)
        {
            arc::warn("render.vulkan", "Vulkan shadow pipeline creation failed; rendering will continue without shadows");
            return false;
        }
        return true;
    }

    void render_shadow_maps(VkCommandBuffer command_buffer)
    {
        const auto* light = active_directional_shadow_light();
        const shadow_settings settings = light ? light->shadow : shadow_settings{ .enabled = false, .resolution = 2048 };
        if (shadow_atlas_.image == VK_NULL_HANDLE)
            return;
        const auto uniform = build_shadow_uniform(light);

        if (!light || frame_shadow_draws_.empty() || shadow_pipeline_ == VK_NULL_HANDLE)
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
        scissor.extent = { shadow_atlas_.resolution, shadow_atlas_.resolution };

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        for (std::uint32_t cascade = 0; cascade < directional_shadow_cascade_count; ++cascade)
        {
            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = shadow_atlas_.cascade_views[cascade];
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil.depth = 1.0f;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.extent = { shadow_atlas_.resolution, shadow_atlas_.resolution };
            rendering.layerCount = 1;
            rendering.pDepthAttachment = &depth_attachment;
            cmd_begin_rendering(command_buffer, &rendering);

            for (const auto& draw : frame_shadow_draws_)
            {
                if (material_alpha_mode_for(draw) == material_alpha_mode::blend)
                    continue;

                auto found = meshes_.find(resource_key(draw.mesh));
                if (found == meshes_.end())
                    continue;

                const auto cascade_matrix = [&] {
                    math::matrix4f matrix;
                    std::copy(uniform.light_view_projection[cascade], uniform.light_view_projection[cascade] + 16, matrix.data());
                    return matrix;
                }();
                const math::matrix4f mvp = math::matmul(cascade_matrix, draw.model);
                mesh_push_constants constants{};
                std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                vkCmdPushConstants(
                    command_buffer,
                    shadow_pipeline_layout_,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(constants),
                    &constants);

                const VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
            }

            cmd_end_rendering(command_buffer);
        }

        transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        shadow_cache_.last_directional_key = {
            .light_index = 0,
            .resolution = shadow_atlas_.resolution,
            .filter = settings.filter
        };
        shadow_cache_.has_directional_key = true;
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
        scissor.extent = { viewport_width_, viewport_height_ };
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);
    }

    void draw_indexed_mesh(VkCommandBuffer command_buffer, const draw_mesh_event& draw, VkPipelineLayout layout, VkShaderStageFlags stages)
    {
        const auto found = meshes_.find(resource_key(draw.mesh));
        if (found == meshes_.end())
            return;

        const auto constants = build_mesh_constants(draw);
        vkCmdPushConstants(command_buffer, layout, stages, 0, sizeof(constants), &constants);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
        vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
    }

    void draw_indexed_virtual_cluster(
        VkCommandBuffer command_buffer,
        const virtual_cluster_draw& draw,
        VkPipelineLayout layout,
        VkShaderStageFlags stages)
    {
        const auto found = virtual_meshes_.find(resource_key(draw.mesh));
        if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size())
            return;

        const auto& cluster = found->second.clusters[draw.cluster_index];
        if (cluster.index_count == 0 || cluster.first_index + cluster.index_count > found->second.index_count)
            return;

        const auto constants = build_mesh_constants(draw.draw);
        vkCmdPushConstants(command_buffer, layout, stages, 0, sizeof(constants), &constants);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
        vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
    }

    bool render_deferred_scene(VkCommandBuffer command_buffer)
    {
        if ((frame_draws_.empty() && frame_virtual_draws_.empty()) ||
            !ensure_deferred_targets(viewport_width_, viewport_height_) ||
            !ensure_shadow_pipeline() ||
            !ensure_gbuffer_pipeline() ||
            !ensure_gbuffer_descriptor_set() ||
            !ensure_deferred_pipeline())
            return false;

        bool has_opaque_draws = false;
        for (const auto& draw : frame_draws_)
        {
            if (draw.mode != render_mode::wireframe && material_alpha_mode_for(draw) != material_alpha_mode::blend)
            {
                has_opaque_draws = true;
                break;
            }
        }
        for (const auto& draw : frame_virtual_draws_)
        {
            if (draw.draw.mode != render_mode::wireframe && material_alpha_mode_for(draw.draw) != material_alpha_mode::blend)
            {
                has_opaque_draws = true;
                break;
            }
        }
        if (!has_opaque_draws)
            return false;

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
            rendering.renderArea.extent = { viewport_width_, viewport_height_ };
            rendering.layerCount = 1;
            rendering.pDepthAttachment = &depth_attachment;
            cmd_begin_rendering(command_buffer, &rendering);
            set_viewport_and_scissor(command_buffer);
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);

            for (const auto& draw : frame_draws_)
            {
                if (draw.mode == render_mode::wireframe || material_alpha_mode_for(draw) == material_alpha_mode::blend)
                    continue;
                draw_indexed_mesh(command_buffer, draw, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT);
            }
            for (const auto& draw : frame_virtual_draws_)
            {
                if (draw.draw.mode == render_mode::wireframe || material_alpha_mode_for(draw.draw) == material_alpha_mode::blend)
                    continue;
                draw_indexed_virtual_cluster(command_buffer, draw, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT);
            }

            cmd_end_rendering(command_buffer);
        }

        transition_graph_image(command_buffer, gbuffer_albedo_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_normal_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_material_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        {
            std::array<VkRenderingAttachmentInfo, 5> color_attachments{};
            graph_image* images[5]{
                &gbuffer_albedo_,
                &gbuffer_normal_,
                &gbuffer_material_,
                &gbuffer_motion_,
                &gbuffer_object_id_
            };
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
            rendering.renderArea.extent = { viewport_width_, viewport_height_ };
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_attachments.size());
            rendering.pColorAttachments = color_attachments.data();
            rendering.pDepthAttachment = &depth_attachment;
            cmd_begin_rendering(command_buffer, &rendering);
            set_viewport_and_scissor(command_buffer);
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, gbuffer_pipeline_);

            for (const auto& draw : frame_draws_)
            {
                if (draw.mode == render_mode::wireframe || material_alpha_mode_for(draw) == material_alpha_mode::blend)
                    continue;
                VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw);
                vkCmdBindDescriptorSets(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    mesh_pipeline_layout_,
                    0,
                    1,
                    &material_descriptor_set,
                    0,
                    nullptr);
                draw_indexed_mesh(
                    command_buffer,
                    draw,
                    mesh_pipeline_layout_,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            }
            for (const auto& draw : frame_virtual_draws_)
            {
                if (draw.draw.mode == render_mode::wireframe || material_alpha_mode_for(draw.draw) == material_alpha_mode::blend)
                    continue;
                VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw.draw);
                vkCmdBindDescriptorSets(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    mesh_pipeline_layout_,
                    0,
                    1,
                    &material_descriptor_set,
                    0,
                    nullptr);
                draw_indexed_virtual_cluster(
                    command_buffer,
                    draw,
                    mesh_pipeline_layout_,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            }

            cmd_end_rendering(command_buffer);
        }

        transition_graph_image(command_buffer, gbuffer_albedo_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_normal_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_material_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        update_gbuffer_descriptor_set();

        {
            transition_viewport(command_buffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            VkRenderingAttachmentInfo color_attachment{};
            color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment.imageView = viewport_view_;
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.extent = { viewport_width_, viewport_height_ };
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color_attachment;
            cmd_begin_rendering(command_buffer, &rendering);
            set_viewport_and_scissor(command_buffer);
            deferred_push_constants constants{};
            constants.light_direction_intensity[0] = 0.35f;
            constants.light_direction_intensity[1] = -0.85f;
            constants.light_direction_intensity[2] = -0.40f;
            if (!frame_directional_lights_.empty())
            {
                const auto& light = frame_directional_lights_.front();
                constants.light_direction_intensity[0] = light.direction[0];
                constants.light_direction_intensity[1] = light.direction[1];
                constants.light_direction_intensity[2] = light.direction[2];
                constants.light_direction_intensity[3] = light.intensity;
                constants.light_color_exposure[0] = light.color[0];
                constants.light_color_exposure[1] = light.color[1];
                constants.light_color_exposure[2] = light.color[2];
            }
            constants.light_color_exposure[3] = frame_environment_.enabled
                ? std::max(frame_environment_.atmosphere.exposure, 0.001f)
                : 1.0f;
            constants.ambient_visualization[0] = frame_lighting_.ambient_color_intensity[0] * frame_lighting_.ambient_color_intensity[3];
            constants.ambient_visualization[1] = frame_lighting_.ambient_color_intensity[1] * frame_lighting_.ambient_color_intensity[3];
            constants.ambient_visualization[2] = frame_lighting_.ambient_color_intensity[2] * frame_lighting_.ambient_color_intensity[3];
            constants.ambient_visualization[3] = !frame_draws_.empty()
                ? static_cast<float>(frame_draws_.front().visualization)
                : static_cast<float>(frame_virtual_draws_.front().draw.visualization);
            constants.debug_counts[0] = static_cast<float>(frame_lighting_.point_count);
            constants.debug_counts[1] = static_cast<float>(frame_lighting_.spot_count);
            constants.debug_counts[2] = 16.0f;
            constants.debug_counts[3] = static_cast<float>(frame_lighting_.skipped_point_count + frame_lighting_.skipped_spot_count);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferred_pipeline_);
            vkCmdBindDescriptorSets(
                command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                deferred_pipeline_layout_,
                0,
                1,
                &gbuffer_descriptor_set_,
                0,
                nullptr);
            vkCmdPushConstants(
                command_buffer,
                deferred_pipeline_layout_,
                VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(constants),
                &constants);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
            cmd_end_rendering(command_buffer);
        }

        if (pending_pick_request_ && ensure_pick_readback_buffer())
        {
            const auto request = *pending_pick_request_;
            pending_pick_request_.reset();

            if (request.x < viewport_width_ && request.y < viewport_height_)
            {
                object_pick_readback readback{};
                readback.request = request;
                readback.frame_index = last_profile_.frame_index;
                readback.active = true;

                for (const auto& draw : frame_draws_)
                {
                    if (draw.object_id.valid())
                        readback.objects.emplace(draw.object_id.index + 1u, draw.object_id);
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
                region.imageOffset = {
                    static_cast<std::int32_t>(request.x),
                    static_cast<std::int32_t>(request.y),
                    0
                };
                region.imageExtent = { 1, 1, 1 };
                vkCmdCopyImageToBuffer(
                    command_buffer,
                    gbuffer_object_id_.image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    pick_readback_buffer_.buffer,
                    1,
                    &region);

                in_flight_pick_ = std::move(readback);
            }
            else
            {
                last_pick_result_ = {
                    .available = true,
                    .hit = false,
                    .object = {},
                    .x = request.x,
                    .y = request.y,
                    .frame_index = last_profile_.frame_index
                };
            }
        }

        return true;
    }

    void render_viewport(VkCommandBuffer command_buffer)
    {
        if (viewport_image_ == VK_NULL_HANDLE)
            return;

        transition_viewport(command_buffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color_attachment{};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = viewport_view_;
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
        rendering.renderArea.extent = { viewport_width_, viewport_height_ };
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
            scissor.extent = { viewport_width_, viewport_height_ };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);

            math::vector3f sun_direction_override{};
            if (!frame_environment_.celestial.enabled && !frame_directional_lights_.empty())
                sun_direction_override = frame_directional_lights_.front().direction;
            const auto constants = detail::build_sky_push_constants(
                frame_environment_,
                frame_camera_,
                viewport_width_,
                viewport_height_,
                resolved_config_.quality != render_quality_tier::low,
                sun_direction_override);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_);
            const auto sky_descriptor = update_current_sky_descriptor_set();
            if (sky_descriptor != VK_NULL_HANDLE)
                vkCmdBindDescriptorSets(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    sky_pipeline_layout_,
                    0,
                    1,
                    &sky_descriptor,
                    0,
                    nullptr);
            vkCmdPushConstants(
                command_buffer,
                sky_pipeline_layout_,
                VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(constants),
                &constants);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
        }

        cmd_end_rendering(command_buffer);
        const bool deferred_rendered = resolved_config_.path == render_path::deferred &&
            render_deferred_scene(command_buffer);

        transition_viewport(command_buffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        cmd_begin_rendering(command_buffer, &rendering);

        if ((!frame_draws_.empty() || !frame_virtual_draws_.empty()) && mesh_pipeline_ != VK_NULL_HANDLE && !white_descriptor_sets_.empty())
        {
            VkViewport viewport{};
            viewport.y = static_cast<float>(viewport_height_);
            viewport.width = static_cast<float>(viewport_width_);
            viewport.height = -static_cast<float>(viewport_height_);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.extent = { viewport_width_, viewport_height_ };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);

            const auto draw_with_pipeline = [&](const draw_mesh_event& draw, VkPipeline pipeline) {
                if (pipeline == VK_NULL_HANDLE)
                    return;
                auto found = meshes_.find(resource_key(draw.mesh));
                if (found == meshes_.end())
                    return;

                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                const auto constants = build_mesh_constants(draw);
                VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw);
                vkCmdBindDescriptorSets(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    mesh_pipeline_layout_,
                    0,
                    1,
                    &material_descriptor_set,
                    0,
                    nullptr);
                vkCmdPushConstants(
                    command_buffer,
                    mesh_pipeline_layout_,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(constants),
                    &constants);
                const VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
            };
            const auto draw_virtual_with_pipeline = [&](const virtual_cluster_draw& draw, VkPipeline pipeline) {
                if (pipeline == VK_NULL_HANDLE)
                    return;
                const auto found = virtual_meshes_.find(resource_key(draw.mesh));
                if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size())
                    return;

                const auto& cluster = found->second.clusters[draw.cluster_index];
                if (cluster.index_count == 0 || cluster.first_index + cluster.index_count > found->second.index_count)
                    return;

                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                const auto constants = build_mesh_constants(draw.draw);
                VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw.draw);
                vkCmdBindDescriptorSets(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    mesh_pipeline_layout_,
                    0,
                    1,
                    &material_descriptor_set,
                    0,
                    nullptr);
                vkCmdPushConstants(
                    command_buffer,
                    mesh_pipeline_layout_,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(constants),
                    &constants);
                const VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
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

                if (material_alpha_mode_for(draw) == material_alpha_mode::blend)
                    continue;

                if (deferred_rendered)
                {
                    continue;
                }

                draw_with_pipeline(draw, mesh_pipeline_);
                if (draw.selected && mesh_wire_pipeline_ != VK_NULL_HANDLE)
                    draw_with_pipeline(draw, mesh_wire_pipeline_);
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

                if (material_alpha_mode_for(draw.draw) == material_alpha_mode::blend)
                    continue;

                if (deferred_rendered)
                    continue;

                draw_virtual_with_pipeline(draw, mesh_pipeline_);
                if (draw.draw.selected && mesh_wire_pipeline_ != VK_NULL_HANDLE)
                    draw_virtual_with_pipeline(draw, mesh_wire_pipeline_);
            }

            std::vector<const draw_mesh_event*> transparent_draws;
            for (const auto& draw : frame_draws_)
            {
                if (draw.mode == render_mode::wireframe || material_alpha_mode_for(draw) != material_alpha_mode::blend)
                    continue;
                transparent_draws.push_back(&draw);
            }
            std::sort(transparent_draws.begin(), transparent_draws.end(), [&](const draw_mesh_event* lhs, const draw_mesh_event* rhs) {
                const auto lhs_delta = math::sub(matrix_translation(lhs->model), frame_camera_.position);
                const auto rhs_delta = math::sub(matrix_translation(rhs->model), frame_camera_.position);
                return math::length_squared(lhs_delta) > math::length_squared(rhs_delta);
            });
            for (const auto* draw : transparent_draws)
            {
                draw_with_pipeline(
                    *draw,
                    mesh_transparent_pipeline_ != VK_NULL_HANDLE ? mesh_transparent_pipeline_ : mesh_pipeline_);
            }
        }

        cmd_end_rendering(command_buffer);
        transition_viewport(command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
#endif

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
    vulkan_swapchain_state swapchain_state_{};
    vulkan_command_context command_context_{};
    descriptor_slot_pool descriptor_slots_;
    deferred_resource_releaser deferred_releases_;
    frame_allocator frame_arena_{ 256u * 1024u };
    pipeline_handle_cache pipeline_handles_;
    VkPipelineCache vk_pipeline_cache_{};
    static constexpr std::uint32_t max_timestamp_queries_{ 64 };
    VkQueryPool timestamp_query_pool_{};
    float timestamp_period_{ 1.0f };
    bool timestamps_supported_{};
    std::uint32_t next_timestamp_query_{};
    std::vector<gpu_scope_record> timestamp_scopes_;
    render_backend_frame_profile last_profile_;
    std::uint64_t last_completed_frame_{};
    std::optional<render_object_pick_request> pending_pick_request_;
    render_object_pick_result last_pick_result_{};
    gpu_buffer pick_readback_buffer_;
    object_pick_readback in_flight_pick_;
    std::vector<std::string> pending_debug_markers_;
    std::unordered_map<std::uint64_t, gpu_mesh> meshes_;
    std::unordered_map<std::uint64_t, gpu_virtual_mesh> virtual_meshes_;
    std::unordered_map<std::uint64_t, gpu_texture> textures_;
    std::unordered_map<std::uint64_t, gpu_material> materials_;
    std::unordered_map<std::uint64_t, gpu_environment> environments_;
    std::vector<draw_mesh_event> frame_draws_;
    std::vector<virtual_cluster_draw> frame_virtual_draws_;
    std::vector<draw_mesh_event> frame_shadow_draws_;
    std::vector<directional_light_event> frame_directional_lights_;
    std::vector<point_light_event> frame_point_lights_;
    std::vector<spot_light_event> frame_spot_lights_;
    scene_lighting_data frame_lighting_;
    world_environment_data frame_environment_;
    render_camera frame_camera_;
    bool frame_shadows_enabled_{ true };
    gpu_buffer light_buffer_;
    std::vector<gpu_buffer> shadow_uniform_buffers_;
    std::uint32_t active_frame_index_{};
    environment_handle active_environment_;
    vulkan_shadow_atlas shadow_atlas_;
    vulkan_shadow_cache shadow_cache_;

    VkDescriptorSetLayout white_descriptor_set_layout_{};
    VkDescriptorPool white_descriptor_pool_{};
    std::vector<VkDescriptorSet> white_descriptor_sets_;
    std::vector<VkDescriptorSet> sky_descriptor_sets_;
    VkImage white_image_{};
    VmaAllocation white_allocation_{};
    VkImageView white_view_{};
    VkSampler white_sampler_{};
    VkPipelineLayout mesh_pipeline_layout_{};
    VkPipeline mesh_pipeline_{};
    VkPipeline mesh_transparent_pipeline_{};
    VkPipeline mesh_wire_pipeline_{};
    VkPipeline gbuffer_pipeline_{};
    VkDescriptorSetLayout gbuffer_descriptor_set_layout_{};
    VkDescriptorPool gbuffer_descriptor_pool_{};
    VkDescriptorSet gbuffer_descriptor_set_{};
    VkSampler gbuffer_sampler_{};
    VkPipelineLayout deferred_pipeline_layout_{};
    VkPipeline deferred_pipeline_{};
    VkPipelineLayout sky_pipeline_layout_{};
    VkPipeline sky_pipeline_{};
    VkPipelineLayout shadow_pipeline_layout_{};
    VkPipeline shadow_pipeline_{};
    bool wireframe_warning_reported_{};

#if ARC_RENDER_VULKAN_ENABLE_IMGUI
    ImGui_ImplVulkanH_Window window_{};
    bool imgui_initialized_{};
    bool native_swapchain_initialized_{};
    bool swapchain_rebuild_{};
    std::uint32_t min_image_count_{ 2 };
    VkFormat viewport_format_{ VK_FORMAT_R16G16B16A16_SFLOAT };
    VkFormat depth_format_{ VK_FORMAT_D32_SFLOAT };
    VkImage viewport_image_{};
    VmaAllocation viewport_allocation_{};
    VkImageView viewport_view_{};
    VkSampler viewport_sampler_{};
    VkDescriptorSet viewport_descriptor_{};
    VkImageLayout viewport_layout_{ VK_IMAGE_LAYOUT_UNDEFINED };
    VkImage viewport_depth_image_{};
    VmaAllocation viewport_depth_allocation_{};
    VkImageView viewport_depth_view_{};
    VkImageLayout viewport_depth_layout_{ VK_IMAGE_LAYOUT_UNDEFINED };
    graph_image gbuffer_albedo_{};
    graph_image gbuffer_normal_{};
    graph_image gbuffer_material_{};
    graph_image gbuffer_motion_{};
    graph_image gbuffer_object_id_{};
    graph_image selection_mask_{};
    std::uint32_t viewport_width_{};
    std::uint32_t viewport_height_{};
    std::uint32_t output_viewport_width_{};
    std::uint32_t output_viewport_height_{};
#endif
};

bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name)
{
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
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
    if (std::find(extensions.begin(), extensions.end(), name) == extensions.end())
        extensions.emplace_back(name);
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
            if (present_supported != VK_TRUE)
                continue;
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
        if (!has_extension(extensions, required.c_str()))
            return false;
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
    auto append_feature = [&](auto& feature) {
        tail->pNext = reinterpret_cast<VkBaseOutStructure*>(&feature);
        tail = reinterpret_cast<VkBaseOutStructure*>(&feature);
    };
    const bool vulkan12_or_newer = properties.apiVersion >= VK_API_VERSION_1_2;
    const bool vulkan13_or_newer = properties.apiVersion >= VK_API_VERSION_1_3;
    if (vulkan12_or_newer)
        append_feature(vulkan12);
    if (vulkan13_or_newer || has_extension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
        append_feature(dynamic_rendering);
    if (vulkan13_or_newer || has_extension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME))
        append_feature(synchronization2);
    if (has_extension(extensions, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME))
        append_feature(descriptor_buffer);
    if (has_extension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME))
        append_feature(mesh_shader);
    if (has_extension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
        append_feature(ray_tracing);
    if (has_extension(extensions, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME))
        append_feature(fragment_shading_rate);
    vkGetPhysicalDeviceFeatures2(physical_device, &features);

    VkPhysicalDeviceDriverProperties driver_properties{};
    driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    if (vulkan12_or_newer)
    {
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &driver_properties;
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    }

    VkPhysicalDeviceMemoryProperties2 memory_properties{};
    memory_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT memory_budget{};
    memory_budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    const bool has_memory_budget = has_extension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (has_memory_budget)
        memory_properties.pNext = &memory_budget;
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
    if (surface == VK_NULL_HANDLE)
        capabilities.presentation = true;

    capabilities.draw_indirect = properties.limits.maxDrawIndirectCount > 0;
    capabilities.draw_indirect_count = vulkan12_or_newer || has_extension(extensions, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
    capabilities.sampler_anisotropy = features.features.samplerAnisotropy == VK_TRUE;
    capabilities.texture_compression_bc = features.features.textureCompressionBC == VK_TRUE;
    capabilities.synchronization2 = synchronization2.synchronization2 == VK_TRUE;
    capabilities.timeline_semaphores = vulkan12.timelineSemaphore == VK_TRUE;
    capabilities.dynamic_rendering = dynamic_rendering.dynamicRendering == VK_TRUE;
    capabilities.descriptor_indexing = vulkan12.descriptorIndexing == VK_TRUE;
    capabilities.descriptor_buffer = descriptor_buffer.descriptorBuffer == VK_TRUE;
    capabilities.mesh_shaders = mesh_shader.meshShader == VK_TRUE;
    capabilities.ray_tracing = ray_tracing.rayTracingPipeline == VK_TRUE;
    capabilities.variable_rate_shading = fragment_shading_rate.pipelineFragmentShadingRate == VK_TRUE;
    capabilities.fill_mode_non_solid = features.features.fillModeNonSolid == VK_TRUE;
    return capabilities;
}

bool supports_required_attachment_formats(VkPhysicalDevice physical_device)
{
    const auto supports = [&](VkFormat format, VkFormatFeatureFlags features) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        return (properties.optimalTilingFeatures & features) == features;
    };
    return supports(VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
        supports(VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
        supports(VK_FORMAT_R32_UINT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
        supports(VK_FORMAT_D32_SFLOAT, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
}

bool supports_required_features(const render_capabilities& capabilities, VkPhysicalDevice physical_device)
{
    const bool vulkan12 = capabilities.api_major > 1 ||
        (capabilities.api_major == 1 && capabilities.api_minor >= 2);
    return vulkan12 && capabilities.graphics_queue && capabilities.compute_queue &&
        capabilities.presentation && capabilities.dynamic_rendering &&
        capabilities.max_color_attachments >= 5 && supports_required_attachment_formats(physical_device);
}

std::uint64_t adapter_score(const render_capabilities& capabilities)
{
    std::uint64_t score = capabilities.discrete_gpu ? 1'000'000ull : capabilities.integrated_gpu ? 500'000ull : 100'000ull;
    score += std::min<std::uint64_t>(capabilities.memory_budget / (1024ull * 1024ull), 250'000ull);
    score += capabilities.timeline_semaphores ? 10'000ull : 0ull;
    score += capabilities.synchronization2 ? 10'000ull : 0ull;
    score += capabilities.descriptor_indexing ? 5'000ull : 0ull;
    return score;
}

bool instance_extension_available(const char* name)
{
    std::uint32_t extension_count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr) != VK_SUCCESS)
        return false;

    std::vector<VkExtensionProperties> extensions(extension_count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data()) != VK_SUCCESS)
        return false;

    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

} // namespace

bool vulkan_loader_available() noexcept
{
    return volkInitialize() == VK_SUCCESS;
}

render_backend_create_result create_vulkan_backend(const vulkan_backend_config& config)
{
    if (volkInitialize() != VK_SUCCESS)
        return { .message = "failed to initialize Vulkan loader" };

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ARC";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "ARC";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    auto requested_instance_extensions = config.instance_extensions;
    if (instance_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME) &&
        std::find(requested_instance_extensions.begin(), requested_instance_extensions.end(), VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == requested_instance_extensions.end())
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
        return { .message = "failed to create Vulkan instance" };

    volkLoadInstance(instance);

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (config.create_surface)
    {
        if (!config.create_surface(instance, &surface, config.surface_user_data) || surface == VK_NULL_HANDLE)
        {
            vkDestroyInstance(instance, nullptr);
            return { .message = "failed to create Vulkan presentation surface" };
        }
    }

    std::uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);
    if (physical_device_count == 0)
    {
        if (surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return { .message = "no Vulkan physical devices found" };
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data());

    VkPhysicalDevice selected_device = VK_NULL_HANDLE;
    render_capabilities selected_capabilities{};
    std::uint32_t graphics_queue_family = UINT32_MAX;
    std::uint64_t selected_score{};
    std::vector<std::string> selected_device_extensions;

    auto required_device_extensions = config.device_extensions;
    if (surface != VK_NULL_HANDLE)
        append_unique_extension(required_device_extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    for (std::uint32_t adapter_index = 0; adapter_index < physical_devices.size(); ++adapter_index)
    {
        if (config.adapter_index && *config.adapter_index != adapter_index)
            continue;

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
            arc::warn("render.vulkan", "Rejected adapter " + std::to_string(adapter_index) + " (" +
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
        if (surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return { .message = "no Vulkan 1.2 graphics/compute device with required attachment formats and dynamic rendering found" };
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
    vulkan12.timelineSemaphore = enable_optional_features && selected_capabilities.timeline_semaphores ? VK_TRUE : VK_FALSE;
    vulkan12.descriptorIndexing = enable_optional_features && selected_capabilities.descriptor_indexing ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceSynchronization2Features synchronization2{};
    synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    synchronization2.synchronization2 = enable_optional_features && selected_capabilities.synchronization2 ? VK_TRUE : VK_FALSE;
    dynamic_rendering.pNext = &vulkan12;
    vulkan12.pNext = &synchronization2;

    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.fillModeNonSolid = selected_capabilities.fill_mode_non_solid ? VK_TRUE : VK_FALSE;
    enabled_features.samplerAnisotropy =
        enable_optional_features && selected_capabilities.sampler_anisotropy ? VK_TRUE : VK_FALSE;

    if (synchronization2.synchronization2 == VK_TRUE &&
        selected_capabilities.api_major == 1 && selected_capabilities.api_minor < 3)
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
        if (surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return { .message = "failed to create Vulkan device" };
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
        if (surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return { .message = "failed to create Vulkan memory allocator" };
    }

    arc::info("render.vulkan", "Selected adapter " + selected_capabilities.adapter_name + " (Vulkan " +
        std::to_string(selected_capabilities.api_major) + "." + std::to_string(selected_capabilities.api_minor) +
        ", " + std::to_string(selected_capabilities.memory_budget / (1024ull * 1024ull)) + " MiB budget)");
    if (config.force_disable_optional_features)
        arc::info("render.vulkan", "Developer compatibility override left all non-required Vulkan features disabled");
    arc::info("render.vulkan", "Created Vulkan backend");
    return {
        .backend = std::make_unique<vulkan_render_backend>(
            instance,
            surface,
            selected_device,
            device,
            queue,
            allocator,
            graphics_queue_family,
            selected_capabilities),
        .message = "created Vulkan backend"
    };
}

vulkan_backend* as_vulkan_backend(render_backend* backend) noexcept
{
    return dynamic_cast<vulkan_backend*>(backend);
}

} // namespace arc::render::vulkan
