#pragma once

#include "android_vulkan_window.h"

#include <vulkan/vulkan.h>

namespace arc::platform::android
{
// Matches arc::render::vulkan::vulkan_surface_create_fn without depending on
// renderer internals. Surface destruction is owned by the Vulkan backend.
[[nodiscard]] bool create_android_vulkan_surface(VkInstance instance,
                                                 PFN_vkGetInstanceProcAddr get_proc,
                                                 VkSurfaceKHR* surface, void* user_data) noexcept;
} // namespace arc::platform::android
