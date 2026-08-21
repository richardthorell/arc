#include "vulkan_swapchain.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace arc::render::vulkan::detail
{
namespace
{

VkSurfaceFormatKHR select_surface_format(VkPhysicalDevice physical_device, VkSurfaceKHR surface,
                                         std::span<const VkFormat> preferred_formats)
{
    std::uint32_t count{};
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, nullptr) != VK_SUCCESS || count == 0)
        return {};

    std::vector<VkSurfaceFormatKHR> formats(count);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, formats.data()) != VK_SUCCESS) return {};

    if (formats.size() == 1 && formats.front().format == VK_FORMAT_UNDEFINED && !preferred_formats.empty())
        return {preferred_formats.front(), VK_COLORSPACE_SRGB_NONLINEAR_KHR};

    for (const auto preferred : preferred_formats)
        for (const auto available : formats)
            if (available.format == preferred && available.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
                return available;

    return formats.front();
}

VkPresentModeKHR select_present_mode(VkPhysicalDevice physical_device, VkSurfaceKHR surface, VkPresentModeKHR preferred)
{
    std::uint32_t count{};
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, nullptr) != VK_SUCCESS ||
        count == 0)
        return VK_PRESENT_MODE_FIFO_KHR;

    std::vector<VkPresentModeKHR> modes(count);
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, modes.data()) != VK_SUCCESS)
        return VK_PRESENT_MODE_FIFO_KHR;

    if (std::find(modes.begin(), modes.end(), preferred) != modes.end()) return preferred;
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkCompositeAlphaFlagBitsKHR select_composite_alpha(VkCompositeAlphaFlagsKHR supported) noexcept
{
    constexpr VkCompositeAlphaFlagBitsKHR candidates[]{
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
    for (const auto candidate : candidates)
        if ((supported & candidate) != 0) return candidate;
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

void destroy_frame_resources(VkDevice device, std::vector<vulkan_swapchain_frame>& frames,
                             std::vector<vulkan_swapchain_semaphores>& semaphores) noexcept
{
    for (auto& frame : frames)
    {
        if (frame.fence != VK_NULL_HANDLE) vkDestroyFence(device, frame.fence, nullptr);
        if (frame.command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, frame.command_pool, nullptr);
    }
    for (auto& pair : semaphores)
    {
        if (pair.image_acquired != VK_NULL_HANDLE) vkDestroySemaphore(device, pair.image_acquired, nullptr);
        if (pair.render_complete != VK_NULL_HANDLE) vkDestroySemaphore(device, pair.render_complete, nullptr);
    }
    frames.clear();
    semaphores.clear();
}

bool create_frame_resources(VkDevice device, std::uint32_t queue_family, std::span<const VkImage> images,
                            std::vector<vulkan_swapchain_frame>& frames,
                            std::vector<vulkan_swapchain_semaphores>& semaphores)
{
    frames.resize(images.size());
    semaphores.resize(images.size());

    for (std::size_t index = 0; index < images.size(); ++index)
    {
        auto& frame = frames[index];
        frame.backbuffer = images[index];

        VkCommandPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = queue_family;
        if (vkCreateCommandPool(device, &pool, nullptr, &frame.command_pool) != VK_SUCCESS) return false;

        VkCommandBufferAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool = frame.command_pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &allocate, &frame.command_buffer) != VK_SUCCESS) return false;

        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(device, &fence, nullptr, &frame.fence) != VK_SUCCESS) return false;

        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semaphore, nullptr, &semaphores[index].image_acquired) != VK_SUCCESS)
            return false;
        if (vkCreateSemaphore(device, &semaphore, nullptr, &semaphores[index].render_complete) != VK_SUCCESS)
            return false;
    }
    return true;
}

} // namespace

bool vulkan_swapchain::create_or_resize(VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR new_surface,
                                        std::uint32_t queue_family, std::uint32_t width, std::uint32_t height,
                                        std::uint32_t minimum_image_count, VkImageUsageFlags usage,
                                        std::span<const VkFormat> preferred_formats,
                                        VkPresentModeKHR preferred_present_mode, std::string& message)
{
    message.clear();
    if (new_surface == VK_NULL_HANDLE || width == 0 || height == 0)
    {
        message = "invalid Vulkan surface or swapchain extent";
        return false;
    }

    VkSurfaceCapabilitiesKHR capabilities{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, new_surface, &capabilities) != VK_SUCCESS)
    {
        message = "failed to query Vulkan surface capabilities";
        return false;
    }
    if ((capabilities.supportedUsageFlags & usage) != usage)
    {
        message = "Vulkan surface does not support the required swapchain image usage";
        return false;
    }

    const auto selected_format = select_surface_format(physical_device, new_surface, preferred_formats);
    if (selected_format.format == VK_FORMAT_UNDEFINED)
    {
        message = "Vulkan surface exposes no usable formats";
        return false;
    }
    const auto selected_present_mode = select_present_mode(physical_device, new_surface, preferred_present_mode);

    VkExtent2D selected_extent{};
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
        selected_extent = capabilities.currentExtent;
    else
    {
        selected_extent.width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        selected_extent.height =
            std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    std::uint32_t selected_image_count = std::max(minimum_image_count, capabilities.minImageCount);
    if (capabilities.maxImageCount != 0)
        selected_image_count = std::min(selected_image_count, capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR create{};
    create.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create.surface = new_surface;
    create.minImageCount = selected_image_count;
    create.imageFormat = selected_format.format;
    create.imageColorSpace = selected_format.colorSpace;
    create.imageExtent = selected_extent;
    create.imageArrayLayers = 1;
    create.imageUsage = usage;
    create.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create.preTransform = capabilities.currentTransform;
    create.compositeAlpha = select_composite_alpha(capabilities.supportedCompositeAlpha);
    create.presentMode = selected_present_mode;
    create.clipped = VK_TRUE;
    create.oldSwapchain = handle;

    VkSwapchainKHR replacement{};
    if (vkCreateSwapchainKHR(device, &create, nullptr, &replacement) != VK_SUCCESS)
    {
        message = "failed to create Vulkan swapchain";
        return false;
    }

    std::uint32_t image_count{};
    if (vkGetSwapchainImagesKHR(device, replacement, &image_count, nullptr) != VK_SUCCESS || image_count == 0)
    {
        vkDestroySwapchainKHR(device, replacement, nullptr);
        message = "Vulkan swapchain returned no images";
        return false;
    }
    std::vector<VkImage> images(image_count);
    if (vkGetSwapchainImagesKHR(device, replacement, &image_count, images.data()) != VK_SUCCESS)
    {
        vkDestroySwapchainKHR(device, replacement, nullptr);
        message = "failed to enumerate Vulkan swapchain images";
        return false;
    }

    std::vector<vulkan_swapchain_frame> replacement_frames;
    std::vector<vulkan_swapchain_semaphores> replacement_semaphores;
    if (!create_frame_resources(device, queue_family, images, replacement_frames, replacement_semaphores))
    {
        destroy_frame_resources(device, replacement_frames, replacement_semaphores);
        vkDestroySwapchainKHR(device, replacement, nullptr);
        message = "failed to create Vulkan swapchain frame resources";
        return false;
    }

    if (vkDeviceWaitIdle(device) != VK_SUCCESS)
    {
        destroy_frame_resources(device, replacement_frames, replacement_semaphores);
        vkDestroySwapchainKHR(device, replacement, nullptr);
        message = "failed to idle Vulkan device before replacing the swapchain";
        return false;
    }

    destroy_frame_resources(device, frames, semaphores);
    if (handle != VK_NULL_HANDLE) vkDestroySwapchainKHR(device, handle, nullptr);

    surface = new_surface;
    surface_format = selected_format;
    present_mode = selected_present_mode;
    handle = replacement;
    extent = selected_extent;
    frames = std::move(replacement_frames);
    semaphores = std::move(replacement_semaphores);
    frame_index = 0;
    semaphore_index = 0;
    return true;
}

void vulkan_swapchain::destroy(VkDevice device) noexcept
{
    destroy_frame_resources(device, frames, semaphores);
    if (handle != VK_NULL_HANDLE) vkDestroySwapchainKHR(device, handle, nullptr);
    surface = VK_NULL_HANDLE;
    surface_format = {};
    present_mode = VK_PRESENT_MODE_FIFO_KHR;
    handle = VK_NULL_HANDLE;
    extent = {};
    frame_index = 0;
    semaphore_index = 0;
}

bool vulkan_swapchain::valid() const noexcept
{
    return handle != VK_NULL_HANDLE && !frames.empty() && frames.size() == semaphores.size();
}

std::uint32_t vulkan_swapchain::image_count() const noexcept
{
    return static_cast<std::uint32_t>(frames.size());
}

} // namespace arc::render::vulkan::detail
