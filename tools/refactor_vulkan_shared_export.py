from pathlib import Path

path = Path("engine/render-vulkan/src/vulkan_backend.cpp")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    text = text.replace(old, new, 1)


def replace_block(start_marker: str, end_marker: str, replacement: str, label: str) -> None:
    global text
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"{label}: start marker not found")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{label}: end marker not found")
    if text.find(start_marker, start + len(start_marker)) >= 0:
        raise RuntimeError(f"{label}: start marker is not unique")
    text = text[:start] + replacement + text[end:]


replace_once(
    """#if defined(_WIN32)\n#include <windows.h>\n#include <d3d11.h>\n#include <dxgi1_2.h>\n#include <wrl/client.h>\n#endif\n""",
    """#if defined(_WIN32)\n#include <windows.h>\n#endif\n""",
    "Windows graphics includes",
)

replace_once(
    "constexpr VkDeviceSize upload_staging_capacity = 64u * 1024u * 1024u;\n",
    """constexpr VkDeviceSize upload_staging_capacity = 64u * 1024u * 1024u;\n#if ARC_VULKAN_SHARED_VIEWPORT\n// Electron/Chromium consumes a Windows NT shared-texture handle. Vulkan can\n// export that D3D-compatible handle type directly without creating a D3D device.\nconstexpr VkExternalMemoryHandleTypeFlagBits shared_viewport_external_handle_type =\n    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;\n#endif\n""",
    "shared viewport handle type",
)

replace_once(
    "        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;\n",
    "",
    "shared viewport D3D texture member",
)

replace_block(
    "    void query_shared_viewport_support()\n",
    "    std::uint32_t shared_memory_type",
    """    void query_shared_viewport_support()\n    {\n        shared_viewport_supported_ = false;\n        get_memory_win32_handle_ = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(\n            vkGetDeviceProcAddr(device_, \"vkGetMemoryWin32HandleKHR\"));\n        if (get_memory_win32_handle_ == nullptr)\n        {\n            shared_viewport_failure_ = \"VK_KHR_external_memory_win32 export is unavailable\";\n            return;\n        }\n\n        VkPhysicalDeviceExternalImageFormatInfo external{};\n        external.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;\n        external.handleType = shared_viewport_external_handle_type;\n        VkPhysicalDeviceImageFormatInfo2 image{};\n        image.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;\n        image.pNext = &external;\n        image.format = VK_FORMAT_B8G8R8A8_UNORM;\n        image.type = VK_IMAGE_TYPE_2D;\n        image.tiling = VK_IMAGE_TILING_OPTIMAL;\n        image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;\n        VkExternalImageFormatProperties external_properties{};\n        external_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;\n        VkImageFormatProperties2 properties{};\n        properties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;\n        properties.pNext = &external_properties;\n        const auto result = vkGetPhysicalDeviceImageFormatProperties2(physical_device_, &image, &properties);\n        const auto features = external_properties.externalMemoryProperties.externalMemoryFeatures;\n        const auto compatible = external_properties.externalMemoryProperties.compatibleHandleTypes;\n        if (result != VK_SUCCESS || (features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0 ||\n            (compatible & shared_viewport_external_handle_type) == 0)\n        {\n            std::ostringstream diagnostic;\n            diagnostic << \"selected Vulkan adapter cannot export BGRA8 Windows shared textures (query=\"\n                       << describe_vk_result(result) << \", features=0x\" << std::hex << features\n                       << \", compatible=0x\" << compatible << ')';\n            shared_viewport_failure_ = std::move(diagnostic).str();\n            return;\n        }\n\n        shared_viewport_supported_ = true;\n        shared_viewport_failure_.clear();\n    }\n\n""",
    "shared viewport support query",
)

replace_block(
    "    bool create_shared_output_slots(shared_viewport_output& output)\n",
    "    void poll_shared_output_fences",
    """    bool create_shared_output_slots(shared_viewport_output& output)\n    {\n        for (auto& slot : output.slots)\n        {\n            VkExternalMemoryImageCreateInfo external_image{};\n            external_image.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;\n            external_image.handleTypes = shared_viewport_external_handle_type;\n            VkImageCreateInfo image{};\n            image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;\n            image.pNext = &external_image;\n            image.imageType = VK_IMAGE_TYPE_2D;\n            image.format = VK_FORMAT_B8G8R8A8_UNORM;\n            image.extent = {output.width, output.height, 1};\n            image.mipLevels = 1;\n            image.arrayLayers = 1;\n            image.samples = VK_SAMPLE_COUNT_1_BIT;\n            image.tiling = VK_IMAGE_TILING_OPTIMAL;\n            image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;\n            if (vkCreateImage(device_, &image, nullptr, &slot.image) != VK_SUCCESS) return false;\n\n            VkMemoryRequirements requirements{};\n            vkGetImageMemoryRequirements(device_, slot.image, &requirements);\n            const auto memory_type = shared_memory_type(requirements.memoryTypeBits);\n            if (memory_type == UINT32_MAX) return false;\n\n            VkExportMemoryAllocateInfo export_memory{};\n            export_memory.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;\n            export_memory.handleTypes = shared_viewport_external_handle_type;\n            VkMemoryDedicatedAllocateInfo dedicated{};\n            dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;\n            dedicated.pNext = &export_memory;\n            dedicated.image = slot.image;\n            VkMemoryAllocateInfo allocation{};\n            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;\n            allocation.pNext = &dedicated;\n            allocation.allocationSize = requirements.size;\n            allocation.memoryTypeIndex = memory_type;\n            if (vkAllocateMemory(device_, &allocation, nullptr, &slot.memory) != VK_SUCCESS ||\n                vkBindImageMemory(device_, slot.image, slot.memory, 0) != VK_SUCCESS)\n                return false;\n\n            VkMemoryGetWin32HandleInfoKHR handle_info{};\n            handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;\n            handle_info.memory = slot.memory;\n            handle_info.handleType = shared_viewport_external_handle_type;\n            if (get_memory_win32_handle_(device_, &handle_info, &slot.shared_handle) != VK_SUCCESS) return false;\n\n            VkCommandPoolCreateInfo pool{};\n            pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;\n            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;\n            pool.queueFamilyIndex = graphics_queue_family_;\n            if (vkCreateCommandPool(device_, &pool, nullptr, &slot.command_pool) != VK_SUCCESS) return false;\n            VkCommandBufferAllocateInfo command{};\n            command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;\n            command.commandPool = slot.command_pool;\n            command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;\n            command.commandBufferCount = 1;\n            if (vkAllocateCommandBuffers(device_, &command, &slot.command_buffer) != VK_SUCCESS) return false;\n            VkFenceCreateInfo fence{};\n            fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;\n            fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;\n            if (vkCreateFence(device_, &fence, nullptr, &slot.fence) != VK_SUCCESS) return false;\n            slot.state = shared_viewport_frame_state::available;\n        }\n        return true;\n    }\n\n""",
    "shared viewport slot creation",
)

replace_once(
    "            slot.texture.Reset();\n",
    "",
    "shared viewport D3D texture cleanup",
)

replace_once(
    """    PFN_vkGetMemoryWin32HandlePropertiesKHR get_memory_win32_handle_properties_{};\n    Microsoft::WRL::ComPtr<ID3D11Device> shared_d3d_device_;\n""",
    "    PFN_vkGetMemoryWin32HandleKHR get_memory_win32_handle_{};\n",
    "shared viewport function pointer",
)

for forbidden in (
    "#include <d3d11.h>",
    "#include <dxgi1_2.h>",
    "#include <wrl/client.h>",
    "D3D11CreateDevice",
    "D3D11_TEXTURE2D_DESC",
    "IDXGI",
    "ID3D11",
    "Microsoft::WRL",
    "VkImportMemoryWin32HandleInfoKHR",
    "VkMemoryWin32HandlePropertiesKHR",
    "get_memory_win32_handle_properties_",
    "shared_d3d_device_",
):
    if forbidden in text:
        raise RuntimeError(f"legacy D3D/import path remains: {forbidden}")

required = (
    "PFN_vkGetMemoryWin32HandleKHR get_memory_win32_handle_{};",
    "VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT",
    "VkExportMemoryAllocateInfo export_memory{};",
    "VkMemoryDedicatedAllocateInfo dedicated{};",
    "VkMemoryGetWin32HandleInfoKHR handle_info{};",
)
for token in required:
    if token not in text:
        raise RuntimeError(f"expected direct Vulkan export token missing: {token}")

if text.count("VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT") != 1:
    raise RuntimeError("D3D-compatible Vulkan handle type should be isolated to one declaration")

path.write_text(text, encoding="utf-8")
