#pragma once

#include <arc/render/render_backend.h>

#include <volk.h>

#include <cstdint>
#include <string>
#include <vector>

namespace arc::render::vulkan
{

using vulkan_surface_create_fn = bool (*)(VkInstance instance, VkSurfaceKHR* surface, void* user_data);

/**
 * @brief Vulkan backend startup configuration.
 */
struct vulkan_backend_config
{
    bool enable_validation{};
    std::vector<std::string> instance_extensions;
    std::vector<std::string> device_extensions;
    vulkan_surface_create_fn create_surface{};
    void* surface_user_data{};
};

/**
 * @brief Vulkan handles and editor presentation hooks exposed by the Vulkan backend.
 */
class vulkan_backend : public render_backend
{
public:
    ~vulkan_backend() override = default;

    /**
     * @brief Return the Vulkan instance owned by this backend.
     */
    virtual VkInstance instance() const noexcept = 0;

    /**
     * @brief Return the selected Vulkan physical device.
     */
    virtual VkPhysicalDevice physical_device() const noexcept = 0;

    /**
     * @brief Return the logical Vulkan device.
     */
    virtual VkDevice device() const noexcept = 0;

    /**
     * @brief Return the primary graphics/present queue family.
     */
    virtual std::uint32_t queue_family() const noexcept = 0;

    /**
     * @brief Return the primary graphics/present queue.
     */
    virtual VkQueue queue() const noexcept = 0;

    /**
     * @brief Initialize Dear ImGui's Vulkan renderer against the backend swapchain.
     */
    virtual bool initialize_imgui(std::uint32_t width, std::uint32_t height, std::string& message) = 0;

    /**
     * @brief Start a Dear ImGui Vulkan frame.
     */
    virtual void new_imgui_frame() = 0;

    /**
     * @brief Submit the current Dear ImGui draw data and present the swapchain.
     */
    virtual bool render_imgui_frame(void* draw_data, std::uint32_t width, std::uint32_t height, std::string& message) = 0;

    /**
     * @brief Shut down Dear ImGui's Vulkan renderer integration.
     */
    virtual void shutdown_imgui() noexcept = 0;
};

/**
 * @brief Return whether the Vulkan loader can be initialized.
 */
bool vulkan_loader_available() noexcept;

/**
 * @brief Create the Vulkan render backend.
 */
render_backend_create_result create_vulkan_backend(const vulkan_backend_config& config = {});

/**
 * @brief Return a Vulkan backend interface when a generic backend is Vulkan.
 */
vulkan_backend* as_vulkan_backend(render_backend* backend) noexcept;

} // namespace arc::render::vulkan
