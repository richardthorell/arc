#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"
#include "vulkan_sky_constants.h"

namespace arc::render::vulkan::backend_detail
{
bool vulkan_render_backend::update_material_parameter_buffer(gpu_buffer& buffer, const material_descriptor* material)
{
    if (buffer.buffer == VK_NULL_HANDLE &&
        !create_buffer(sizeof(material_uniform_data), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                       buffer))
        return false;

    const auto parameters = build_material_parameters(material);
    void* mapped{};
    if (vmaMapMemory(allocator_, buffer.allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, &parameters, sizeof(parameters));
    vmaFlushAllocation(allocator_, buffer.allocation, 0, sizeof(parameters));
    vmaUnmapMemory(allocator_, buffer.allocation);
    return true;
}

bool vulkan_render_backend::ensure_material_parameter_buffers(std::vector<gpu_buffer>& buffers,
                                                              const material_descriptor* material)
{
    const auto count = frame_resource_count();
    if (buffers.size() != count)
    {
        for (auto& buffer : buffers)
            destroy_buffer(buffer);
        buffers.assign(count, {});
    }
    for (auto& buffer : buffers)
    {
        if (buffer.buffer == VK_NULL_HANDLE && !update_material_parameter_buffer(buffer, material)) return false;
    }
    return true;
}

bool vulkan_render_backend::ensure_material_descriptor_sets(gpu_material& material)
{
    if (!ensure_material_parameter_buffers(material.parameter_buffers, &material.data)) return false;
    const auto count = frame_resource_count();
    if (material.descriptor_sets.size() != count) material.descriptor_sets.assign(count, VK_NULL_HANDLE);
    for (auto& set : material.descriptor_sets)
    {
        if (set != VK_NULL_HANDLE) continue;
        set = allocate_material_descriptor_set();
        if (set == VK_NULL_HANDLE) return false;
    }
    return true;
}

bool vulkan_render_backend::ensure_white_descriptor_sets()
{
    if (!ensure_material_parameter_buffers(white_material_parameter_buffers_, nullptr)) return false;
    const auto count = frame_resource_count();
    if (white_descriptor_sets_.size() != count) white_descriptor_sets_.assign(count, VK_NULL_HANDLE);
    for (auto& set : white_descriptor_sets_)
    {
        if (set != VK_NULL_HANDLE) continue;
        set = allocate_material_descriptor_set();
        if (set == VK_NULL_HANDLE) return false;
    }
    return true;
}

bool vulkan_render_backend::ensure_sky_descriptor_sets()
{
    if (!ensure_white_texture()) return false;
    const auto count = frame_resource_count();
    if (sky_descriptor_sets_.size() != count) sky_descriptor_sets_.assign(count, VK_NULL_HANDLE);
    for (auto& set : sky_descriptor_sets_)
    {
        if (set == VK_NULL_HANDLE) set = allocate_material_descriptor_set();
        if (set == VK_NULL_HANDLE) return false;
    }
    return true;
}

VkDescriptorSet vulkan_render_backend::update_current_sky_descriptor_set()
{
    if (!ensure_sky_descriptor_sets()) return VK_NULL_HANDLE;
    const auto slot = current_frame_slot();
    if (slot >= sky_descriptor_sets_.size()) return VK_NULL_HANDLE;
    const auto set = sky_descriptor_sets_[slot];
    update_material_descriptor_set(
        set, nullptr,
        slot < white_material_parameter_buffers_.size() ? &white_material_parameter_buffers_[slot] : nullptr, slot);

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

void vulkan_render_backend::update_material_descriptor_set(VkDescriptorSet descriptor_set,
                                                           const material_descriptor* material,
                                                           const gpu_buffer* material_parameters,
                                                           std::uint32_t frame_slot)
{
    const auto* shadow_buffer_resource = shadow_uniform_buffer_for_slot(frame_slot);
    if (descriptor_set == VK_NULL_HANDLE || white_view_ == VK_NULL_HANDLE ||
        shadow_atlas_.array_view == VK_NULL_HANDLE || local_shadow_atlas_.view == VK_NULL_HANDLE ||
        shadow_buffer_resource == nullptr || shadow_buffer_resource->buffer == VK_NULL_HANDLE ||
        material_parameters == nullptr || material_parameters->buffer == VK_NULL_HANDLE ||
        light_buffer_.buffer == VK_NULL_HANDLE)
        return;

    const auto resolve_texture = [&](texture_handle handle, VkSampler& sampler, VkImageView& view,
                                     std::optional<texture_semantic> expected = std::nullopt)
    {
        sampler = white_sampler_;
        view = white_view_;
        if (!handle.valid()) return;
        if (const auto found = textures_.find(resource_key(handle)); found != textures_.end())
        {
            if (expected && !valid_texture_color_space(*expected, found->second.data.color_space))
            {
                const auto diagnostic_key = resource_key(handle) ^ (static_cast<std::uint64_t>(*expected) << 56u);
                if (texture_semantic_diagnostics_.insert(diagnostic_key).second)
                {
                    arc::diagnostics::warn("render.vulkan", "Texture '" + found->second.data.name +
                                                                "' has a color space incompatible with its "
                                                                "material slot; binding the explicit fallback");
                }
                return;
            }
            if (found->second.view != VK_NULL_HANDLE && found->second.sampler != VK_NULL_HANDLE)
            {
                sampler = found->second.sampler;
                view = found->second.view;
            }
        }
    };

    std::array<VkDescriptorImageInfo, material_image_bindings.size()> image_infos{};
    VkSampler sampler{};
    VkImageView view{};
    resolve_texture({}, sampler, view);
    for (auto& image : image_infos)
        image = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    if (material != nullptr && material->domain == material_domain::terrain)
    {
        for (std::size_t layer = 0; layer < material->terrain_layers.size(); ++layer)
        {
            resolve_texture(material->terrain_layers[layer].base_color_texture, sampler, view,
                            texture_semantic::base_color);
            image_infos[layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material->terrain_layers[layer].normal_texture, sampler, view, texture_semantic::normal);
            image_infos[6u + layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material->terrain_layers[layer].packed_surface_texture, sampler, view,
                            texture_semantic::metallic_roughness);
            image_infos[10u + layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
    }
    else
    {
        resolve_texture(material ? material->base_color_texture : texture_handle{}, sampler, view,
                        texture_semantic::base_color);
        image_infos[0] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->metallic_roughness_texture : texture_handle{}, sampler, view,
                        texture_semantic::metallic_roughness);
        image_infos[1] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->normal_texture : texture_handle{}, sampler, view,
                        texture_semantic::normal);
        image_infos[2] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->occlusion_texture : texture_handle{}, sampler, view,
                        texture_semantic::occlusion);
        image_infos[3] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->emissive_texture : texture_handle{}, sampler, view,
                        texture_semantic::emissive);
        image_infos[4] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->clear_coat_texture : texture_handle{}, sampler, view,
                        texture_semantic::clear_coat);
        image_infos[6] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->clear_coat_roughness_texture : texture_handle{}, sampler, view,
                        texture_semantic::clear_coat);
        image_infos[7] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->clear_coat_normal_texture : texture_handle{}, sampler, view,
                        texture_semantic::normal);
        image_infos[8] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->anisotropy_texture : texture_handle{}, sampler, view,
                        texture_semantic::anisotropy);
        image_infos[9] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->subsurface_texture : texture_handle{}, sampler, view,
                        texture_semantic::thickness);
        image_infos[10] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->thickness_texture : texture_handle{}, sampler, view,
                        texture_semantic::thickness);
        image_infos[11] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->transmission_texture : texture_handle{}, sampler, view,
                        texture_semantic::transmission);
        image_infos[12] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }

    image_infos[5].sampler = shadow_atlas_.sampler;
    image_infos[5].imageView = shadow_atlas_.array_view;
    image_infos[5].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    image_infos.back().sampler = local_shadow_atlas_.sampler;
    image_infos.back().imageView = local_shadow_atlas_.view;
    image_infos.back().imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo shadow_buffer{};
    shadow_buffer.buffer = shadow_buffer_resource->buffer;
    shadow_buffer.offset = 0;
    shadow_buffer.range = sizeof(shadow_uniform_data);

    std::array<VkWriteDescriptorSet, material_image_bindings.size() + 3u> writes{};
    for (std::uint32_t image_index = 0; image_index < image_infos.size(); ++image_index)
    {
        writes[image_index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[image_index].dstSet = descriptor_set;
        writes[image_index].dstBinding = material_image_bindings[image_index];
        writes[image_index].descriptorCount = 1;
        writes[image_index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[image_index].pImageInfo = &image_infos[image_index];
    }
    auto& shadow_write = writes[material_image_bindings.size()];
    shadow_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    shadow_write.dstSet = descriptor_set;
    shadow_write.dstBinding = material_shadow_data_binding;
    shadow_write.descriptorCount = 1;
    shadow_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    shadow_write.pBufferInfo = &shadow_buffer;
    VkDescriptorBufferInfo light_buffer{};
    light_buffer.buffer = light_buffer_.buffer;
    light_buffer.range = sizeof(scene_lighting_data);
    auto& light_write = writes[material_image_bindings.size() + 1u];
    light_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    light_write.dstSet = descriptor_set;
    light_write.dstBinding = material_light_data_binding;
    light_write.descriptorCount = 1;
    light_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    light_write.pBufferInfo = &light_buffer;
    VkDescriptorBufferInfo parameter_buffer{};
    parameter_buffer.buffer = material_parameters->buffer;
    parameter_buffer.range = sizeof(material_uniform_data);
    auto& parameter_write = writes.back();
    parameter_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    parameter_write.dstSet = descriptor_set;
    parameter_write.dstBinding = material_parameters_binding;
    parameter_write.descriptorCount = 1;
    parameter_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    parameter_write.pBufferInfo = &parameter_buffer;
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void vulkan_render_backend::update_material_descriptor_sets(gpu_material& material)
{
    if (!ensure_material_descriptor_sets(material)) return;
    for (std::uint32_t frame_slot = 0; frame_slot < material.descriptor_sets.size(); ++frame_slot)
    {
        if (!update_material_parameter_buffer(material.parameter_buffers[frame_slot], &material.data)) continue;
        update_material_descriptor_set(material.descriptor_sets[frame_slot], &material.data,
                                       &material.parameter_buffers[frame_slot], frame_slot);
    }
}

void vulkan_render_backend::update_white_descriptor_sets()
{
    if (!ensure_white_descriptor_sets()) return;
    for (std::uint32_t frame_slot = 0; frame_slot < white_descriptor_sets_.size(); ++frame_slot)
        update_material_descriptor_set(white_descriptor_sets_[frame_slot], nullptr,
                                       &white_material_parameter_buffers_[frame_slot], frame_slot);
}

void vulkan_render_backend::update_all_material_descriptor_sets()
{
    update_white_descriptor_sets();
    for (auto& [_, material] : materials_)
        update_material_descriptor_sets(material);
}

void vulkan_render_backend::update_current_material_descriptor_sets()
{
    const auto frame_slot = current_frame_slot();
    if (ensure_white_descriptor_sets() && frame_slot < white_descriptor_sets_.size())
        update_material_descriptor_set(white_descriptor_sets_[frame_slot], nullptr,
                                       &white_material_parameter_buffers_[frame_slot], frame_slot);
    for (auto& [_, material] : materials_)
    {
        if (ensure_material_descriptor_sets(material) && frame_slot < material.descriptor_sets.size())
        {
            if (!update_material_parameter_buffer(material.parameter_buffers[frame_slot], &material.data)) continue;
            update_material_descriptor_set(material.descriptor_sets[frame_slot], &material.data,
                                           &material.parameter_buffers[frame_slot], frame_slot);
        }
    }
}

VkDescriptorSet vulkan_render_backend::allocate_material_descriptor_set()
{
    if (white_descriptor_pool_ == VK_NULL_HANDLE || white_descriptor_set_layout_ == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    VkDescriptorSet set{};
    VkDescriptorSetAllocateInfo descriptor_allocate{};
    descriptor_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_allocate.descriptorPool = white_descriptor_pool_;
    descriptor_allocate.descriptorSetCount = 1;
    descriptor_allocate.pSetLayouts = &white_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &descriptor_allocate, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    return set;
}

bool vulkan_render_backend::ensure_white_texture()
{
    const auto shadow_resolution = shadow_atlas_.resolution == 0 ? 2048u : shadow_atlas_.resolution;
    if (!ensure_shadow_uniform_buffers() ||
        !ensure_shadow_resources({.enabled = false, .resolution = shadow_resolution}) ||
        !ensure_local_shadow_resources())
        return false;

    if (white_descriptor_set_layout_ != VK_NULL_HANDLE && white_descriptor_pool_ != VK_NULL_HANDLE &&
        white_view_ != VK_NULL_HANDLE && white_sampler_ != VK_NULL_HANDLE)
    {
        return ensure_white_descriptor_sets();
    }

    std::array<VkDescriptorSetLayoutBinding, material_binding_count> bindings{};
    for (std::uint32_t binding_index = 0; binding_index < material_binding_count; ++binding_index)
    {
        bindings[binding_index].binding = binding_index;
        bindings[binding_index].descriptorType =
            binding_index == material_shadow_data_binding || binding_index == material_parameters_binding
                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : binding_index == material_light_data_binding ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                           : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[binding_index].descriptorCount = 1;
        bindings[binding_index].stageFlags = binding_index == material_shadow_data_binding ||
                                                     binding_index == material_light_data_binding ||
                                                     binding_index == material_parameters_binding
                                                 ? VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                                 : VK_SHADER_STAGE_FRAGMENT_BIT;
    }

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
    image.extent = {1, 1, 1};
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
    if (vkCreateImageView(device_, &view, nullptr, &white_view_) != VK_SUCCESS) return false;

    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = VK_FILTER_NEAREST;
    sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device_, &sampler, nullptr, &white_sampler_) != VK_SUCCESS) return false;

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
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &to_copy);

    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {1, 1, 1};
    vkCmdCopyBufferToImage(command_buffer, staging.buffer, white_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copy);

    VkImageMemoryBarrier to_shader = to_copy;
    to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &to_shader);
    vkEndCommandBuffer(command_buffer);

    submit_upload_commands(command_buffer);
    vkDestroyCommandPool(device_, pool, nullptr);
    destroy_buffer(staging);

    std::array<VkDescriptorPoolSize, 3> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount =
        static_cast<std::uint32_t>(material_image_bindings.size()) * material_descriptor_set_capacity;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = material_descriptor_set_capacity * 2u;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[2].descriptorCount = material_descriptor_set_capacity;
    VkDescriptorPoolCreateInfo descriptor_pool{};
    descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool.maxSets = material_descriptor_set_capacity;
    descriptor_pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    descriptor_pool.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &descriptor_pool, nullptr, &white_descriptor_pool_) != VK_SUCCESS) return false;

    update_all_material_descriptor_sets();
    return true;
}

} // namespace arc::render::vulkan::backend_detail
