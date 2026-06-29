#include <arc/render/vulkan/vulkan_backend.h>

#include <arc/log.h>

#include "builtin_shaders.h"

#if ARC_RENDER_VULKAN_ENABLE_IMGUI
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#endif

#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
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

std::uint64_t resource_key(resource_handle handle) noexcept
{
    return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
}

VkDeviceSize buffer_size(std::size_t count, std::size_t stride) noexcept
{
    return static_cast<VkDeviceSize>(count * stride);
}

struct mesh_push_constants
{
    float model_view_projection[16]{};
    float model[16]{};
    float base_color[4]{ 1.0f, 1.0f, 1.0f, 1.0f };
    float light_direction_intensity[4]{ 0.35f, -0.85f, -0.40f, 1.0f };
    float light_color[4]{ 1.0f, 0.96f, 0.88f, 1.0f };
    float visualization[4]{};
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
    }

    ~vulkan_render_backend() override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        shutdown_imgui();
#endif
        if (device_ != VK_NULL_HANDLE)
            vkDeviceWaitIdle(device_);
        destroy_mesh_pipeline();
        destroy_white_texture();
        destroy_meshes();
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
        frame_directional_lights_.clear();
        for (const auto& event : packet.events)
        {
            if (const auto* upload = std::get_if<mesh_upload_event>(&event.payload))
                upload_mesh(*upload);
            else if (const auto* draw = std::get_if<draw_mesh_event>(&event.payload))
                frame_draws_.push_back(*draw);
            else if (const auto* light = std::get_if<directional_light_event>(&event.payload))
                frame_directional_lights_.push_back(*light);
        }

        std::ostringstream message;
        message << "vulkan accepted frame " << packet.frame_index << " with "
                << packet.events.size() << " event(s) and " << graph.passes.size() << " pass(es)";
        return { .submitted = true, .message = message.str() };
    }

    void resize_viewport(std::uint32_t width, std::uint32_t height) override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        if (imgui_initialized_ && width > 0 && height > 0)
            ensure_viewport(width, height);
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
        init_info.ApiVersion = VK_API_VERSION_1_3;
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
        ensure_viewport(std::max(1u, width / 2), std::max(1u, height / 2));
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

        ImGui_ImplVulkanH_Frame* frame = &window_.Frames[window_.FrameIndex];
        vkWaitForFences(device_, 1, &frame->Fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &frame->Fence);
        vkResetCommandPool(device_, frame->CommandPool, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(frame->CommandBuffer, &begin_info);

        render_viewport(frame->CommandBuffer);

        VkClearValue clear_value{};
        clear_value.color.float32[0] = 0.055f;
        clear_value.color.float32[1] = 0.071f;
        clear_value.color.float32[2] = 0.086f;
        clear_value.color.float32[3] = 1.0f;

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
        return true;
#else
        (void)draw_data;
        (void)width;
        (void)height;
        message = "Vulkan ImGui presentation support is not compiled";
        return false;
#endif
    }

    void shutdown_imgui() noexcept override
    {
#if ARC_RENDER_VULKAN_ENABLE_IMGUI
        if (!imgui_initialized_)
            return;

        vkDeviceWaitIdle(device_);
        destroy_viewport();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplVulkanH_DestroyWindow(instance_, device_, &window_, nullptr);
        imgui_initialized_ = false;
#endif
    }

private:
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

    void destroy_buffer(gpu_buffer& value) noexcept
    {
        if (value.buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator_, value.buffer, value.allocation);
            value.buffer = VK_NULL_HANDLE;
            value.allocation = VK_NULL_HANDLE;
        }
    }

    void destroy_meshes() noexcept
    {
        for (auto& [_, mesh] : meshes_)
        {
            destroy_buffer(mesh.vertices);
            destroy_buffer(mesh.indices);
        }
        meshes_.clear();
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

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        const VkResult result = vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE);
        if (result == VK_SUCCESS)
            vkQueueWaitIdle(queue_);

        vkDestroyCommandPool(device_, pool, nullptr);
        destroy_buffer(staging);
        if (result != VK_SUCCESS)
        {
            destroy_buffer(destination);
            return false;
        }
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

    void destroy_mesh_pipeline() noexcept
    {
        if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, mesh_wire_pipeline_, nullptr);
            mesh_wire_pipeline_ = VK_NULL_HANDLE;
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
    }

    void destroy_white_texture() noexcept
    {
        if (white_descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, white_descriptor_pool_, nullptr);
            white_descriptor_pool_ = VK_NULL_HANDLE;
            white_descriptor_set_ = VK_NULL_HANDLE;
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

    bool ensure_white_texture()
    {
        if (white_descriptor_set_ != VK_NULL_HANDLE)
            return true;

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 1;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = 1;
        layout.pBindings = &binding;
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

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue_);
        vkDestroyCommandPool(device_, pool, nullptr);
        destroy_buffer(staging);

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_size.descriptorCount = 1;
        VkDescriptorPoolCreateInfo descriptor_pool{};
        descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool.maxSets = 1;
        descriptor_pool.poolSizeCount = 1;
        descriptor_pool.pPoolSizes = &pool_size;
        if (vkCreateDescriptorPool(device_, &descriptor_pool, nullptr, &white_descriptor_pool_) != VK_SUCCESS)
            return false;

        VkDescriptorSetAllocateInfo descriptor_allocate{};
        descriptor_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptor_allocate.descriptorPool = white_descriptor_pool_;
        descriptor_allocate.descriptorSetCount = 1;
        descriptor_allocate.pSetLayouts = &white_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &descriptor_allocate, &white_descriptor_set_) != VK_SUCCESS)
            return false;

        VkDescriptorImageInfo image_info{};
        image_info.sampler = white_sampler_;
        image_info.imageView = white_view_;
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = white_descriptor_set_;
        write.dstBinding = 1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image_info;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        return true;
    }

    bool ensure_mesh_pipeline()
    {
        if (mesh_pipeline_ != VK_NULL_HANDLE)
            return true;
        if (!ensure_white_texture())
            return false;

        VkShaderModule vert = create_shader_module(
            builtin::default_unlit_vert_spv,
            std::size(builtin::default_unlit_vert_spv));
        VkShaderModule frag = create_shader_module(
            builtin::default_unlit_frag_spv,
            std::size(builtin::default_unlit_frag_spv));
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
        std::array<VkVertexInputAttributeDescription, 4> attributes{};
        attributes[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position) };
        attributes[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal) };
        attributes[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord) };
        attributes[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color) };

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

        const VkResult result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &mesh_pipeline_);
        if (result == VK_SUCCESS && capabilities_.fill_mode_non_solid)
        {
            raster.polygonMode = VK_POLYGON_MODE_LINE;
            depth.depthWriteEnable = VK_FALSE;
            const VkResult wire_result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &mesh_wire_pipeline_);
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

    void ensure_viewport(std::uint32_t width, std::uint32_t height)
    {
        width = std::max(1u, width);
        height = std::max(1u, height);
        if (viewport_image_ != VK_NULL_HANDLE && viewport_width_ == width && viewport_height_ == height)
            return;

        vkDeviceWaitIdle(device_);
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

        viewport_descriptor_ = ImGui_ImplVulkan_AddTexture(viewport_sampler_, viewport_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        viewport_width_ = width;
        viewport_height_ = height;
        viewport_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

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
        color_attachment.clearValue.color.float32[0] = 0.118f;
        color_attachment.clearValue.color.float32[1] = 0.118f;
        color_attachment.clearValue.color.float32[2] = 0.118f;
        color_attachment.clearValue.color.float32[3] = 1.0f;

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
        vkCmdBeginRendering(command_buffer, &rendering);

        if (!frame_draws_.empty() && ensure_mesh_pipeline())
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
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline_layout_, 0, 1, &white_descriptor_set_, 0, nullptr);

            const directional_light_event light = frame_directional_lights_.empty()
                ? directional_light_event{}
                : frame_directional_lights_.front();

            const auto draw_with_pipeline = [&](const draw_mesh_event& draw, VkPipeline pipeline) {
                if (pipeline == VK_NULL_HANDLE)
                    return;
                auto found = meshes_.find(resource_key(draw.mesh));
                if (found == meshes_.end())
                    return;

                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                const math::matrix4f mvp = math::matmul(draw.view_projection, draw.model);
                mesh_push_constants constants{};
                std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                std::copy(draw.model.data(), draw.model.data() + 16, constants.model);
                constants.light_direction_intensity[0] = light.direction[0];
                constants.light_direction_intensity[1] = light.direction[1];
                constants.light_direction_intensity[2] = light.direction[2];
                constants.light_direction_intensity[3] = light.intensity;
                constants.light_color[0] = light.color[0];
                constants.light_color[1] = light.color[1];
                constants.light_color[2] = light.color[2];
                constants.visualization[0] = static_cast<float>(draw.visualization);
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

                draw_with_pipeline(draw, mesh_pipeline_);
                if (draw.selected && mesh_wire_pipeline_ != VK_NULL_HANDLE)
                    draw_with_pipeline(draw, mesh_wire_pipeline_);
            }
        }

        vkCmdEndRendering(command_buffer);
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
    std::unordered_map<std::uint64_t, gpu_mesh> meshes_;
    std::vector<draw_mesh_event> frame_draws_;
    std::vector<directional_light_event> frame_directional_lights_;

    VkDescriptorSetLayout white_descriptor_set_layout_{};
    VkDescriptorPool white_descriptor_pool_{};
    VkDescriptorSet white_descriptor_set_{};
    VkImage white_image_{};
    VmaAllocation white_allocation_{};
    VkImageView white_view_{};
    VkSampler white_sampler_{};
    VkPipelineLayout mesh_pipeline_layout_{};
    VkPipeline mesh_pipeline_{};
    VkPipeline mesh_wire_pipeline_{};
    bool wireframe_warning_reported_{};

#if ARC_RENDER_VULKAN_ENABLE_IMGUI
    ImGui_ImplVulkanH_Window window_{};
    bool imgui_initialized_{};
    bool swapchain_rebuild_{};
    std::uint32_t min_image_count_{ 2 };
    VkFormat viewport_format_{ VK_FORMAT_R8G8B8A8_UNORM };
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
    std::uint32_t viewport_width_{};
    std::uint32_t viewport_height_{};
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

std::uint32_t find_graphics_queue_family(VkPhysicalDevice physical_device, VkSurfaceKHR surface = VK_NULL_HANDLE)
{
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());

    for (std::uint32_t index = 0; index < count; ++index)
    {
        if ((families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)
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

render_capabilities query_capabilities(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceVulkan13Features vulkan13{};
    vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13.pNext = &vulkan12;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &vulkan13;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);

    std::uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data());

    render_capabilities capabilities{};
    capabilities.backend = render_backend_type::vulkan;
    capabilities.api_major = VK_VERSION_MAJOR(properties.apiVersion);
    capabilities.api_minor = VK_VERSION_MINOR(properties.apiVersion);
    capabilities.synchronization2 = vulkan13.synchronization2 == VK_TRUE;
    capabilities.timeline_semaphores = vulkan12.timelineSemaphore == VK_TRUE;
    capabilities.dynamic_rendering = vulkan13.dynamicRendering == VK_TRUE;
    capabilities.descriptor_indexing = vulkan12.descriptorIndexing == VK_TRUE;
    capabilities.descriptor_buffer = has_extension(extensions, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
    capabilities.mesh_shaders = has_extension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);
    capabilities.ray_tracing = has_extension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    capabilities.variable_rate_shading = has_extension(extensions, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
    capabilities.fill_mode_non_solid = features.features.fillModeNonSolid == VK_TRUE;
    return capabilities;
}

bool supports_required_features(const render_capabilities& capabilities)
{
    return capabilities.api_major > 1 ||
        (capabilities.api_major == 1 && capabilities.api_minor >= 3) &&
            capabilities.synchronization2 &&
            capabilities.timeline_semaphores &&
            capabilities.dynamic_rendering;
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
    app_info.apiVersion = VK_API_VERSION_1_3;

    const auto instance_extensions = make_c_strings(config.instance_extensions);

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

    auto required_device_extensions = config.device_extensions;
    if (surface != VK_NULL_HANDLE)
        required_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    for (const auto physical_device : physical_devices)
    {
        const auto capabilities = query_capabilities(physical_device);
        const auto queue_family = find_graphics_queue_family(physical_device, surface);
        if (queue_family != UINT32_MAX &&
            supports_required_features(capabilities) &&
            supports_device_extensions(physical_device, required_device_extensions))
        {
            selected_device = physical_device;
            selected_capabilities = capabilities;
            graphics_queue_family = queue_family;
            break;
        }
    }

    if (selected_device == VK_NULL_HANDLE)
    {
        if (surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return { .message = "no Vulkan 1.3 graphics device with synchronization2, timeline semaphores, and dynamic rendering found" };
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = graphics_queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceVulkan12Features vulkan12{};
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.timelineSemaphore = VK_TRUE;
    vulkan12.descriptorIndexing = selected_capabilities.descriptor_indexing ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan13Features vulkan13{};
    vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13.pNext = &vulkan12;
    vulkan13.synchronization2 = VK_TRUE;
    vulkan13.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.fillModeNonSolid = selected_capabilities.fill_mode_non_solid ? VK_TRUE : VK_FALSE;

    auto device_extensions = required_device_extensions;
    if (selected_capabilities.descriptor_buffer)
        device_extensions.push_back(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
    const auto device_extension_names = make_c_strings(device_extensions);

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &vulkan13;
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
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;

    VmaAllocator allocator = VK_NULL_HANDLE;
    if (vmaCreateAllocator(&allocator_info, &allocator) != VK_SUCCESS)
    {
        vkDestroyDevice(device, nullptr);
        if (surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return { .message = "failed to create Vulkan memory allocator" };
    }

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
