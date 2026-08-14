#pragma once

#include <arc/render/render_backend.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace arc::render::vulkan
{

/**
 * @brief Create a presentation surface using the backend's initialized Vulkan procedure resolver.
 * @param instance Vulkan instance that owns the new surface.
 * @param get_instance_proc_address Procedure resolver initialized for @p instance.
 * @param surface Destination receiving the created surface.
 * @param user_data Platform data supplied through @ref vulkan_backend_config::surface_user_data.
 * @return `true` when @p surface contains a valid presentation surface.
 */
using vulkan_surface_create_fn = bool (*)(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_address,
                                          VkSurfaceKHR* surface, void* user_data);

/**
 * @brief Vulkan backend startup configuration.
 */
struct vulkan_backend_config
{
    bool enable_validation{};
    std::optional<std::uint32_t> adapter_index;
    bool force_disable_optional_features{};
    std::vector<std::string> instance_extensions;
    std::vector<std::string> device_extensions;
    vulkan_surface_create_fn create_surface{};
    void* surface_user_data{};
    viewport_output_type viewport_output{viewport_output_type::native_window};
};

/**
 * @brief Return whether the Vulkan loader can be initialized.
 */
[[nodiscard]] bool vulkan_loader_available() noexcept;

/**
 * @brief Create the Vulkan render backend.
 */
[[nodiscard]] render_backend_create_result create_vulkan_backend(const vulkan_backend_config& config = {});

} // namespace arc::render::vulkan
