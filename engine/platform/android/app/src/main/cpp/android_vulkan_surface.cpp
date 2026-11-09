#include "android_vulkan_surface.h"

#include <android/native_window.h>

namespace arc::platform::android
{
bool create_android_vulkan_surface(VkInstance instance, PFN_vkGetInstanceProcAddr get_proc, VkSurfaceKHR* surface,
                                   void* user_data) noexcept
{
    if (surface) *surface = VK_NULL_HANDLE;
    if (!instance || !get_proc || !surface || !user_data) return false;
    const auto* binding = static_cast<const vulkan_native_window*>(user_data);
    if (!binding->window || binding->generation == 0) return false;

    // Use ARC's initialized instance resolver, not a second Vulkan loader.
    const auto create =
        reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(get_proc(instance, "vkCreateAndroidSurfaceKHR"));
    if (!create) return false;

    VkAndroidSurfaceCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    info.window = binding->window;
    return create(instance, &info, nullptr, surface) == VK_SUCCESS && *surface != VK_NULL_HANDLE;
}
} // namespace arc::platform::android
