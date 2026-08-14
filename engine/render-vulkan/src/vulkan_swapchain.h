#pragma once

#include <volk.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace arc::render::vulkan::detail
{

struct vulkan_swapchain_frame
{
    VkImage backbuffer{};
    VkCommandPool command_pool{};
    VkCommandBuffer command_buffer{};
    VkFence fence{};
};

struct vulkan_swapchain_semaphores
{
    VkSemaphore image_acquired{};
    VkSemaphore render_complete{};
};

class vulkan_swapchain
{
public:
    vulkan_swapchain() = default;
    vulkan_swapchain(const vulkan_swapchain&) = delete;
    vulkan_swapchain& operator=(const vulkan_swapchain&) = delete;

    [[nodiscard]] bool create_or_resize(VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface,
                                        std::uint32_t queue_family, std::uint32_t width, std::uint32_t height,
                                        std::uint32_t minimum_image_count, VkImageUsageFlags usage,
                                        std::span<const VkFormat> preferred_formats,
                                        VkPresentModeKHR preferred_present_mode, std::string& message);
    void destroy(VkDevice device) noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t image_count() const noexcept;

    VkSurfaceKHR surface{};
    VkSurfaceFormatKHR surface_format{};
    VkPresentModeKHR present_mode{VK_PRESENT_MODE_FIFO_KHR};
    VkSwapchainKHR handle{};
    VkExtent2D extent{};
    std::vector<vulkan_swapchain_frame> frames;
    std::vector<vulkan_swapchain_semaphores> semaphores;
    std::uint32_t frame_index{};
    std::uint32_t semaphore_index{};
};

} // namespace arc::render::vulkan::detail
