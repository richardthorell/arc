#include <arc/render/vulkan/vulkan_backend.h>

#include <arc/diagnostics/log.h>

#include "vulkan_backend_internal.h"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace arc::render::vulkan
{
namespace
{
bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name)
{
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension)
                       { return std::strcmp(extension.extensionName, name) == 0; });
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
    if (std::find(extensions.begin(), extensions.end(), name) == extensions.end()) extensions.emplace_back(name);
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
            if (present_supported != VK_TRUE) continue;
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
        if (!has_extension(extensions, required.c_str())) return false;
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
    auto append_feature = [&](auto& feature)
    {
        tail->pNext = reinterpret_cast<VkBaseOutStructure*>(&feature);
        tail = reinterpret_cast<VkBaseOutStructure*>(&feature);
    };
    const bool vulkan12_or_newer = properties.apiVersion >= VK_API_VERSION_1_2;
    const bool vulkan13_or_newer = properties.apiVersion >= VK_API_VERSION_1_3;
    if (vulkan12_or_newer) append_feature(vulkan12);
    if (vulkan13_or_newer || has_extension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
        append_feature(dynamic_rendering);
    if (vulkan13_or_newer || has_extension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME))
        append_feature(synchronization2);
    if (has_extension(extensions, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) append_feature(descriptor_buffer);
    if (has_extension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME)) append_feature(mesh_shader);
    if (has_extension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) append_feature(ray_tracing);
    if (has_extension(extensions, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) append_feature(fragment_shading_rate);
    vkGetPhysicalDeviceFeatures2(physical_device, &features);

    VkPhysicalDeviceDriverProperties driver_properties{};
    driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceDescriptorIndexingProperties descriptor_indexing_properties{};
    descriptor_indexing_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
    if (vulkan12_or_newer)
    {
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &driver_properties;
        driver_properties.pNext = &descriptor_indexing_properties;
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    }

    VkPhysicalDeviceMemoryProperties2 memory_properties{};
    memory_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT memory_budget{};
    memory_budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    const bool has_memory_budget = has_extension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (has_memory_budget) memory_properties.pNext = &memory_budget;
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
    if (surface == VK_NULL_HANDLE) capabilities.presentation = true;

    capabilities.draw_indirect = properties.limits.maxDrawIndirectCount > 0;
    capabilities.draw_indirect_count =
        vulkan12_or_newer || has_extension(extensions, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
    capabilities.compute_shaders = capabilities.compute_queue;
    capabilities.storage_buffers = properties.limits.maxStorageBufferRange >= 128u * 1024u * 1024u;
    capabilities.storage_images = properties.limits.maxPerStageDescriptorStorageImages > 0;
    capabilities.texture_mip_streaming = capabilities.graphics_queue && capabilities.transfer_queue &&
                                         capabilities.compute_shaders && capabilities.storage_buffers;
    // Software page-table sampling is advertised only once feedback compaction,
    // cache publication, and the shared sampling ABI are all executable.
    capabilities.virtual_texture_feedback = false;
    capabilities.virtual_texture_sampling = false;
    capabilities.shader_draw_parameters = properties.apiVersion >= VK_API_VERSION_1_1;
    capabilities.gpu_scene_indirect =
        capabilities.compute_shaders && capabilities.storage_buffers && capabilities.draw_indirect;
    capabilities.gpu_scene_indirect_count = capabilities.gpu_scene_indirect && capabilities.draw_indirect_count;
    VkFormatProperties hzb_format{};
    vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R32G32_SFLOAT, &hzb_format);
    const auto required_hzb_features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    capabilities.hzb_occlusion = capabilities.compute_shaders && capabilities.storage_images &&
                                 (hzb_format.optimalTilingFeatures & required_hzb_features) == required_hzb_features;
    const auto supports_storage_sampled = [&](VkFormat format)
    {
        VkFormatProperties format_properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &format_properties);
        const auto required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        return (format_properties.optimalTilingFeatures & required) == required;
    };
    capabilities.temporal_resolve =
        capabilities.compute_shaders && capabilities.storage_images &&
        supports_storage_sampled(VK_FORMAT_R16G16B16A16_SFLOAT) && supports_storage_sampled(VK_FORMAT_R16G16_SFLOAT) &&
        supports_storage_sampled(VK_FORMAT_R32_SFLOAT) && supports_storage_sampled(VK_FORMAT_R8_UNORM);
    capabilities.temporal_upscale = capabilities.temporal_resolve;
    // FXAA is implemented as the final linear-LDR stage fused into the output
    // transform and selected by the executable graph's FXAA pass.
    capabilities.fxaa = true;
    capabilities.virtual_geometry_compute = false;
    capabilities.virtual_geometry_mesh_shader = false;
    capabilities.virtual_geometry_streaming = false;
    // VSM support is advertised only after allocation, feedback, caster rendering,
    // sampling, and contact-shadow pipelines are all executable. Resource plumbing
    // alone must not cause Ultra to select an incomplete path.
    capabilities.virtual_shadow_allocation = false;
    capabilities.virtual_shadow_feedback = false;
    capabilities.virtual_shadow_rendering = false;
    capabilities.virtual_shadow_sampling = false;
    capabilities.virtual_shadow_virtual_geometry = false;
    capabilities.screen_space_contact_shadows = false;
    capabilities.sampler_anisotropy = features.features.samplerAnisotropy == VK_TRUE;
    capabilities.texture_compression_bc = features.features.textureCompressionBC == VK_TRUE;
    capabilities.synchronization2 = synchronization2.synchronization2 == VK_TRUE;
    capabilities.timeline_semaphores = vulkan12.timelineSemaphore == VK_TRUE;
    capabilities.dynamic_rendering = dynamic_rendering.dynamicRendering == VK_TRUE;
    constexpr std::uint32_t minimum_bindless_sampled_images = 4096u;
    constexpr std::uint32_t minimum_bindless_samplers = 256u;
    const bool complete_descriptor_indexing =
        vulkan12.descriptorIndexing == VK_TRUE && vulkan12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
        vulkan12.runtimeDescriptorArray == VK_TRUE && vulkan12.descriptorBindingPartiallyBound == VK_TRUE &&
        vulkan12.descriptorBindingVariableDescriptorCount == VK_TRUE &&
        vulkan12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE &&
        descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindSampledImages >=
            minimum_bindless_sampled_images &&
        descriptor_indexing_properties.maxPerStageDescriptorUpdateAfterBindSampledImages >=
            minimum_bindless_sampled_images &&
        descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindSamplers >= minimum_bindless_samplers;
    capabilities.descriptor_indexing = complete_descriptor_indexing;
    capabilities.bindless_sampled_images = complete_descriptor_indexing;
    capabilities.bindless_samplers = complete_descriptor_indexing;
    capabilities.bindless_material_tables = complete_descriptor_indexing && capabilities.compute_shaders &&
                                            capabilities.storage_buffers && capabilities.storage_images;
    capabilities.bindless_geometry_tables =
        complete_descriptor_indexing && capabilities.gpu_scene_indirect_count && capabilities.shader_draw_parameters;
    const bool complete_virtual_geometry_compute =
        capabilities.bindless_material_tables && capabilities.hzb_occlusion && capabilities.transfer_queue &&
        supports_storage_sampled(VK_FORMAT_R16G16B16A16_SFLOAT) && supports_storage_sampled(VK_FORMAT_R16G16_SFLOAT) &&
        supports_storage_sampled(VK_FORMAT_R32_UINT);
    capabilities.virtual_geometry_compute = complete_virtual_geometry_compute;
    capabilities.virtual_geometry_streaming = complete_virtual_geometry_compute;
    capabilities.gpu_visibility_compaction =
        capabilities.bindless_geometry_tables && capabilities.bindless_material_tables;
    capabilities.gpu_transparent_sorting =
        capabilities.gpu_visibility_compaction && properties.limits.maxComputeSharedMemorySize >= 28u * 1024u;
    capabilities.gpu_skinning = capabilities.gpu_visibility_compaction &&
                                properties.limits.maxPerStageDescriptorStorageBuffers >= 7u &&
                                properties.limits.maxComputeWorkGroupInvocations >= 64u;
    capabilities.descriptor_buffer = descriptor_buffer.descriptorBuffer == VK_TRUE;
    capabilities.mesh_shaders = mesh_shader.meshShader == VK_TRUE;
    // Capability facts describe executable ARC paths. Ray-query acceleration structures and
    // their graph execution are enabled together by the lighting backend; a driver extension
    // alone must never select the hybrid path.
    capabilities.screen_space_indirect_lighting = false;
    capabilities.surface_cache = false;
    capabilities.radiance_cache = false;
    capabilities.software_ray_tracing = false;
    capabilities.hardware_ray_query = false;
    capabilities.ray_tracing = false;
    capabilities.sparse_resources = features.features.sparseBinding == VK_TRUE;
    capabilities.variable_rate_shading = fragment_shading_rate.pipelineFragmentShadingRate == VK_TRUE;
    capabilities.fill_mode_non_solid = features.features.fillModeNonSolid == VK_TRUE;
    return capabilities;
}

bool supports_required_attachment_formats(VkPhysicalDevice physical_device)
{
    const auto supports = [&](VkFormat format, VkFormatFeatureFlags features)
    {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        return (properties.optimalTilingFeatures & features) == features;
    };
    return supports(VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
           supports(VK_FORMAT_R16G16_SFLOAT,
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
           supports(VK_FORMAT_R32_UINT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
           supports(VK_FORMAT_D32_SFLOAT,
                    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
}

bool supports_required_features(const render_capabilities& capabilities, VkPhysicalDevice physical_device)
{
    const bool vulkan12 = capabilities.api_major > 1 || (capabilities.api_major == 1 && capabilities.api_minor >= 2);
    return vulkan12 && capabilities.graphics_queue && capabilities.compute_queue && capabilities.presentation &&
           capabilities.dynamic_rendering && capabilities.max_color_attachments >= 5 &&
           supports_required_attachment_formats(physical_device);
}

std::uint64_t adapter_score(const render_capabilities& capabilities)
{
    std::uint64_t score = capabilities.discrete_gpu     ? 1'000'000ull
                          : capabilities.integrated_gpu ? 500'000ull
                                                        : 100'000ull;
    score += std::min<std::uint64_t>(capabilities.memory_budget / (1024ull * 1024ull), 250'000ull);
    score += capabilities.timeline_semaphores ? 10'000ull : 0ull;
    score += capabilities.synchronization2 ? 10'000ull : 0ull;
    score += capabilities.descriptor_indexing ? 5'000ull : 0ull;
    return score;
}

bool instance_extension_available(const char* name)
{
    std::uint32_t extension_count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr) != VK_SUCCESS) return false;

    std::vector<VkExtensionProperties> extensions(extension_count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data()) != VK_SUCCESS)
        return false;

    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension)
                       { return std::strcmp(extension.extensionName, name) == 0; });
}

} // namespace

bool vulkan_loader_available() noexcept
{
    return volkInitialize() == VK_SUCCESS;
}

render_backend_create_result create_vulkan_backend(const vulkan_backend_config& config)
{
    if (volkInitialize() != VK_SUCCESS)
        return render_backend_create_result::failure(
            {render_backend_create_error_code::loader_unavailable, "failed to initialize Vulkan loader"});

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ARC";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "ARC";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    auto requested_instance_extensions = config.instance_extensions;
    if (instance_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME) &&
        std::find(requested_instance_extensions.begin(), requested_instance_extensions.end(),
                  VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == requested_instance_extensions.end())
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
        return render_backend_create_result::failure(
            {render_backend_create_error_code::instance_creation_failed, "failed to create Vulkan instance"});

    volkLoadInstance(instance);

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (config.create_surface)
    {
        if (!config.create_surface(instance, vkGetInstanceProcAddr, &surface, config.surface_user_data) ||
            surface == VK_NULL_HANDLE)
        {
            vkDestroyInstance(instance, nullptr);
            return render_backend_create_result::failure({render_backend_create_error_code::surface_creation_failed,
                                                          "failed to create Vulkan presentation surface"});
        }
    }

    std::uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);
    if (physical_device_count == 0)
    {
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return render_backend_create_result::failure(
            {render_backend_create_error_code::adapter_unavailable, "no Vulkan physical devices found"});
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data());

    VkPhysicalDevice selected_device = VK_NULL_HANDLE;
    render_capabilities selected_capabilities{};
    std::uint32_t graphics_queue_family = UINT32_MAX;
    std::uint64_t selected_score{};
    std::vector<std::string> selected_device_extensions;

    auto required_device_extensions = config.device_extensions;
    if (surface != VK_NULL_HANDLE) append_unique_extension(required_device_extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#if ARC_VULKAN_SHARED_VIEWPORT
    if (config.viewport_output == viewport_output_type::shared_texture)
    {
        append_unique_extension(required_device_extensions, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        append_unique_extension(required_device_extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        append_unique_extension(required_device_extensions, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        append_unique_extension(required_device_extensions, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    }
#endif

    for (std::uint32_t adapter_index = 0; adapter_index < physical_devices.size(); ++adapter_index)
    {
        if (config.adapter_index && *config.adapter_index != adapter_index) continue;

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
            arc::diagnostics::warn("render.vulkan", "Rejected adapter " + std::to_string(adapter_index) + " (" +
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
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return render_backend_create_result::failure(
            {render_backend_create_error_code::adapter_unavailable,
             "no Vulkan 1.2 graphics/compute device with required attachment formats and dynamic rendering found"});
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
    vulkan12.timelineSemaphore =
        enable_optional_features && selected_capabilities.timeline_semaphores ? VK_TRUE : VK_FALSE;
    const auto enable_descriptor_indexing = enable_optional_features && selected_capabilities.descriptor_indexing;
    vulkan12.descriptorIndexing = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.shaderSampledImageArrayNonUniformIndexing = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.runtimeDescriptorArray = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.descriptorBindingPartiallyBound = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.descriptorBindingVariableDescriptorCount = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;
    vulkan12.descriptorBindingSampledImageUpdateAfterBind = enable_descriptor_indexing ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceSynchronization2Features synchronization2{};
    synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    synchronization2.synchronization2 =
        enable_optional_features && selected_capabilities.synchronization2 ? VK_TRUE : VK_FALSE;
    dynamic_rendering.pNext = &vulkan12;
    vulkan12.pNext = &synchronization2;

    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.fillModeNonSolid = selected_capabilities.fill_mode_non_solid ? VK_TRUE : VK_FALSE;
    enabled_features.samplerAnisotropy =
        enable_optional_features && selected_capabilities.sampler_anisotropy ? VK_TRUE : VK_FALSE;

    if (synchronization2.synchronization2 == VK_TRUE && selected_capabilities.api_major == 1 &&
        selected_capabilities.api_minor < 3)
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
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return render_backend_create_result::failure(
            {render_backend_create_error_code::device_creation_failed, "failed to create Vulkan device"});
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
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        return render_backend_create_result::failure(
            {render_backend_create_error_code::memory_allocator_creation_failed,
             "failed to create Vulkan memory allocator"});
    }

    arc::diagnostics::info(
        "render.vulkan",
        "Selected adapter " + selected_capabilities.adapter_name + " (Vulkan " +
            std::to_string(selected_capabilities.api_major) + "." + std::to_string(selected_capabilities.api_minor) +
            ", " + std::to_string(selected_capabilities.memory_budget / (1024ull * 1024ull)) + " MiB budget)");
    if (config.force_disable_optional_features)
        arc::diagnostics::info("render.vulkan",
                               "Developer compatibility override left all non-required Vulkan features disabled");
    arc::diagnostics::info("render.vulkan", "Created Vulkan backend");
    return render_backend_create_result::success(std::make_unique<backend_detail::vulkan_render_backend>(
        instance, surface, selected_device, device, queue, allocator, graphics_queue_family, selected_capabilities,
        config.viewport_output));
}

} // namespace arc::render::vulkan
