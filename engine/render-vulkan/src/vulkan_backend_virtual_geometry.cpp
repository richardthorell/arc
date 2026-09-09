#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"

namespace arc::render::vulkan::backend_detail
{
bool vulkan_render_backend::ensure_virtual_geometry_raster_resources()
{
    if (!capabilities_.storage_images || !ensure_virtual_geometry_traversal_resources()) return false;
    const auto previous_depth_view = virtual_geometry_encoded_depth_.view;
    const auto previous_visibility_view = virtual_geometry_visibility_ids_.view;
    if (!ensure_graph_image(virtual_geometry_encoded_depth_, std::max(viewport_width_, 1u),
                            std::max(viewport_height_, 1u), VK_FORMAT_R32_UINT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !ensure_graph_image(virtual_geometry_visibility_ids_, std::max(viewport_width_, 1u),
                            std::max(viewport_height_, 1u), VK_FORMAT_R32_UINT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (previous_depth_view != virtual_geometry_encoded_depth_.view ||
        previous_visibility_view != virtual_geometry_visibility_ids_.view)
    {
        virtual_geometry_raster_descriptors_dirty_ = true;
        virtual_geometry_material_descriptors_dirty_ = true;
    }

    if (virtual_geometry_raster_bin_capacity_ < virtual_geometry_visible_capacity_ ||
        virtual_geometry_raster_bin_buffer_.buffer == VK_NULL_HANDLE)
    {
        gpu_buffer replacement{};
        if (!create_buffer(buffer_size(virtual_geometry_visible_capacity_, sizeof(virtual_geometry_gpu_raster_bin)),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, replacement))
            return false;
        auto retired = virtual_geometry_raster_bin_buffer_;
        virtual_geometry_raster_bin_buffer_ = replacement;
        virtual_geometry_raster_bin_capacity_ = virtual_geometry_visible_capacity_;
        virtual_geometry_raster_descriptors_dirty_ = true;
        if (retired.buffer != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                     [this, retired]() mutable { destroy_buffer(retired); });
    }

    if (virtual_geometry_raster_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
        for (std::uint32_t binding = 0; binding < 7u; ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[7] = {7u, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[8] = {8u, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &virtual_geometry_raster_descriptor_set_layout_) !=
            VK_SUCCESS)
            return false;
        virtual_geometry_raster_descriptors_dirty_ = true;
    }

    if (virtual_geometry_raster_descriptors_dirty_)
    {
        VkDescriptorPool replacement_pool{};
        VkDescriptorSet replacement_set{};
        const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7u},
                                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2u}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1u;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &replacement_pool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = replacement_pool;
        allocate.descriptorSetCount = 1u;
        allocate.pSetLayouts = &virtual_geometry_raster_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &replacement_set) != VK_SUCCESS)
        {
            vkDestroyDescriptorPool(device_, replacement_pool, nullptr);
            return false;
        }
        const std::array buffers{
            VkDescriptorBufferInfo{virtual_geometry_visible_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_cluster_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_scene_transform_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_page_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_page_heap_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_counter_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_raster_bin_buffer_.buffer, 0u, VK_WHOLE_SIZE},
        };
        std::array<VkWriteDescriptorSet, 7> writes{};
        for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
        {
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = replacement_set;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1u;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &buffers[binding];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        const std::array images{
            VkDescriptorImageInfo{VK_NULL_HANDLE, virtual_geometry_encoded_depth_.view, VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{VK_NULL_HANDLE, virtual_geometry_visibility_ids_.view, VK_IMAGE_LAYOUT_GENERAL},
        };
        std::array<VkWriteDescriptorSet, 2> image_writes{};
        for (std::uint32_t index = 0; index < image_writes.size(); ++index)
        {
            image_writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            image_writes[index].dstSet = replacement_set;
            image_writes[index].dstBinding = 7u + index;
            image_writes[index].descriptorCount = 1u;
            image_writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            image_writes[index].pImageInfo = &images[index];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(image_writes.size()), image_writes.data(), 0u,
                               nullptr);
        const auto retired_pool = virtual_geometry_raster_descriptor_pool_;
        virtual_geometry_raster_descriptor_pool_ = replacement_pool;
        virtual_geometry_raster_descriptor_set_ = replacement_set;
        if (retired_pool != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(), [this, retired_pool]()
                                     { vkDestroyDescriptorPool(device_, retired_pool, nullptr); });
        virtual_geometry_raster_descriptors_dirty_ = false;
    }

    if (virtual_geometry_raster_pipeline_layout_ == VK_NULL_HANDLE)
    {
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(virtual_geometry_raster_push_constants)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1u;
        layout.pSetLayouts = &virtual_geometry_raster_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1u;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &virtual_geometry_raster_pipeline_layout_) != VK_SUCCESS)
            return false;
    }
    const std::array shader_words{
        std::pair{builtin::virtual_geometry_binning_comp_spv, std::size(builtin::virtual_geometry_binning_comp_spv)},
        std::pair{builtin::virtual_geometry_software_depth_comp_spv,
                  std::size(builtin::virtual_geometry_software_depth_comp_spv)},
        std::pair{builtin::virtual_geometry_visibility_resolve_comp_spv,
                  std::size(builtin::virtual_geometry_visibility_resolve_comp_spv)},
    };
    for (std::size_t index = 0; index < virtual_geometry_raster_pipelines_.size(); ++index)
    {
        if (virtual_geometry_raster_pipelines_[index] != VK_NULL_HANDLE) continue;
        const auto shader = create_shader_module(shader_words[index].first, shader_words[index].second);
        if (shader == VK_NULL_HANDLE) return false;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = shader;
        pipeline.stage.pName = "main";
        pipeline.layout = virtual_geometry_raster_pipeline_layout_;
        const auto status = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                     &virtual_geometry_raster_pipelines_[index]);
        vkDestroyShaderModule(device_, shader, nullptr);
        if (status != VK_SUCCESS) return false;
    }
    return true;
}

void vulkan_render_backend::dispatch_virtual_geometry_raster(VkCommandBuffer command_buffer)
{
    if (!ensure_virtual_geometry_raster_resources()) return;
    transition_graph_image(command_buffer, virtual_geometry_encoded_depth_, VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, virtual_geometry_visibility_ids_, VK_IMAGE_LAYOUT_GENERAL);
    VkClearColorValue clear{};
    clear.uint32[0] = std::numeric_limits<std::uint32_t>::max();
    clear.uint32[1] = clear.uint32[0];
    clear.uint32[2] = clear.uint32[0];
    clear.uint32[3] = clear.uint32[0];
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vkCmdClearColorImage(command_buffer, virtual_geometry_encoded_depth_.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1u,
                         &range);
    vkCmdClearColorImage(command_buffer, virtual_geometry_visibility_ids_.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1u,
                         &range);

    std::array<VkImageMemoryBarrier, 2> clear_barriers{};
    const std::array clear_images{virtual_geometry_encoded_depth_.image, virtual_geometry_visibility_ids_.image};
    for (std::size_t index = 0; index < clear_barriers.size(); ++index)
    {
        clear_barriers[index].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        clear_barriers[index].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        clear_barriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        clear_barriers[index].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        clear_barriers[index].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        clear_barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clear_barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clear_barriers[index].image = clear_images[index];
        clear_barriers[index].subresourceRange = range;
    }
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u,
                         nullptr, 0u, nullptr, static_cast<std::uint32_t>(clear_barriers.size()),
                         clear_barriers.data());

    std::array<VkBufferMemoryBarrier, 2> traversal_barriers{};
    const std::array traversal_buffers{virtual_geometry_visible_buffer_.buffer,
                                       virtual_geometry_counter_buffer_.buffer};
    for (std::size_t index = 0; index < traversal_barriers.size(); ++index)
    {
        traversal_barriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        traversal_barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        traversal_barriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        traversal_barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        traversal_barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        traversal_barriers[index].buffer = traversal_buffers[index];
        traversal_barriers[index].size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                         0u, nullptr, static_cast<std::uint32_t>(traversal_barriers.size()), traversal_barriers.data(),
                         0u, nullptr);

    virtual_geometry_raster_push_constants constants{};
    std::copy_n(frame_camera_.view_projection.data(), 16u, constants.view_projection);
    constants.viewport_capacities[0] = viewport_width_;
    constants.viewport_capacities[1] = viewport_height_;
    constants.viewport_capacities[2] = virtual_geometry_visible_capacity_;
    constants.viewport_capacities[3] = virtual_geometry_raster_bin_capacity_;
    const auto bind_and_dispatch = [&](VkPipeline pipeline)
    {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                virtual_geometry_raster_pipeline_layout_, 0u, 1u,
                                &virtual_geometry_raster_descriptor_set_, 0u, nullptr);
        vkCmdPushConstants(command_buffer, virtual_geometry_raster_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, (virtual_geometry_visible_capacity_ + 63u) / 64u, 1u, 1u);
    };
    bind_and_dispatch(virtual_geometry_raster_pipelines_[0]);

    std::array<VkBufferMemoryBarrier, 2> bin_barriers{};
    const std::array bin_buffers{virtual_geometry_raster_bin_buffer_.buffer, virtual_geometry_counter_buffer_.buffer};
    for (std::size_t index = 0; index < bin_barriers.size(); ++index)
    {
        bin_barriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bin_barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bin_barriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bin_barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bin_barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bin_barriers[index].buffer = bin_buffers[index];
        bin_barriers[index].size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                         0u, nullptr, static_cast<std::uint32_t>(bin_barriers.size()), bin_barriers.data(), 0u,
                         nullptr);
    bind_and_dispatch(virtual_geometry_raster_pipelines_[1]);

    VkImageMemoryBarrier depth_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    depth_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    depth_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    depth_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    depth_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    depth_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depth_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depth_barrier.image = virtual_geometry_encoded_depth_.image;
    depth_barrier.subresourceRange = range;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                         0u, nullptr, 0u, nullptr, 1u, &depth_barrier);
    bind_and_dispatch(virtual_geometry_raster_pipelines_[2]);
}

bool vulkan_render_backend::ensure_virtual_geometry_material_resources()
{
    if (!capabilities_.virtual_geometry_compute || virtual_geometry_visibility_ids_.view == VK_NULL_HANDLE ||
        gbuffer_albedo_.view == VK_NULL_HANDLE || gpu_scene_visibility_buffer_.buffer == VK_NULL_HANDLE)
        return false;
    const auto& material_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::material)];
    const auto& texture_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::texture)];
    if (material_table.storage.buffer == VK_NULL_HANDLE ||
        material_table.element_stride != sizeof(gpu_material_table_record))
        return false;
    if (virtual_geometry_material_frame_buffer_.buffer == VK_NULL_HANDLE &&
        !create_buffer(sizeof(virtual_geometry_material_frame_data), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       VMA_MEMORY_USAGE_CPU_TO_GPU, virtual_geometry_material_frame_buffer_))
        return false;

    if (virtual_geometry_material_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 18> bindings{};
        bindings[0] = {0u, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1u, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        for (std::uint32_t binding = 2u; binding <= 9u; ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[10] = {10u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        for (std::uint32_t binding = 11u; binding <= 16u; ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[17] = {17u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, virtual_geometry_bindless_texture_capacity,
                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        std::array<VkDescriptorBindingFlags, 18> binding_flags{};
        binding_flags[17] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT |
                            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo flags{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        flags.bindingCount = static_cast<std::uint32_t>(binding_flags.size());
        flags.pBindingFlags = binding_flags.data();
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.pNext = &flags;
        layout.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &virtual_geometry_material_descriptor_set_layout_) !=
            VK_SUCCESS)
            return false;
        virtual_geometry_material_descriptors_dirty_ = true;
    }

    if (virtual_geometry_material_descriptors_dirty_)
    {
        VkDescriptorPool replacement_pool{};
        VkDescriptorSet replacement_set{};
        const std::array pool_sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8u},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8u},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, virtual_geometry_bindless_texture_capacity},
        };
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = 1u;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &replacement_pool) != VK_SUCCESS) return false;

        const std::uint32_t descriptor_count = virtual_geometry_bindless_texture_capacity;
        VkDescriptorSetVariableDescriptorCountAllocateInfo variable_count{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};
        variable_count.descriptorSetCount = 1u;
        variable_count.pDescriptorCounts = &descriptor_count;
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.pNext = &variable_count;
        allocate.descriptorPool = replacement_pool;
        allocate.descriptorSetCount = 1u;
        allocate.pSetLayouts = &virtual_geometry_material_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &replacement_set) != VK_SUCCESS)
        {
            vkDestroyDescriptorPool(device_, replacement_pool, nullptr);
            return false;
        }

        const auto texture_table_buffer = texture_table.storage.buffer != VK_NULL_HANDLE
                                              ? texture_table.storage.buffer
                                              : material_table.storage.buffer;
        const std::array buffer_infos{
            VkDescriptorBufferInfo{virtual_geometry_visible_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_cluster_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_scene_transform_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_page_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_page_heap_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_scene_visibility_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{material_table.storage.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{texture_table_buffer, 0u, VK_WHOLE_SIZE},
        };
        std::array<VkWriteDescriptorSet, 8> buffer_writes{};
        for (std::uint32_t index = 0u; index < buffer_writes.size(); ++index)
        {
            buffer_writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            buffer_writes[index].dstSet = replacement_set;
            buffer_writes[index].dstBinding = 2u + index;
            buffer_writes[index].descriptorCount = 1u;
            buffer_writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            buffer_writes[index].pBufferInfo = &buffer_infos[index];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(buffer_writes.size()), buffer_writes.data(), 0u,
                               nullptr);

        const VkDescriptorBufferInfo frame_info{virtual_geometry_material_frame_buffer_.buffer, 0u,
                                                sizeof(virtual_geometry_material_frame_data)};
        VkWriteDescriptorSet frame_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        frame_write.dstSet = replacement_set;
        frame_write.dstBinding = 10u;
        frame_write.descriptorCount = 1u;
        frame_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        frame_write.pBufferInfo = &frame_info;
        vkUpdateDescriptorSets(device_, 1u, &frame_write, 0u, nullptr);

        const std::array image_infos{
            VkDescriptorImageInfo{VK_NULL_HANDLE, virtual_geometry_visibility_ids_.view, VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{VK_NULL_HANDLE, virtual_geometry_encoded_depth_.view, VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{VK_NULL_HANDLE, gbuffer_albedo_.view, VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{VK_NULL_HANDLE, gbuffer_normal_.view, VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{VK_NULL_HANDLE, gbuffer_material_.view, VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{VK_NULL_HANDLE, gbuffer_emissive_.view, VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{VK_NULL_HANDLE, gbuffer_motion_.view, VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{VK_NULL_HANDLE, gbuffer_object_id_.view, VK_IMAGE_LAYOUT_GENERAL},
        };
        std::array<VkWriteDescriptorSet, 8> image_writes{};
        for (std::uint32_t index = 0u; index < image_writes.size(); ++index)
        {
            image_writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            image_writes[index].dstSet = replacement_set;
            image_writes[index].dstBinding = index < 2u ? index : 9u + index;
            image_writes[index].descriptorCount = 1u;
            image_writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            image_writes[index].pImageInfo = &image_infos[index];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(image_writes.size()), image_writes.data(), 0u,
                               nullptr);

        std::vector<VkDescriptorImageInfo> texture_infos;
        std::vector<VkWriteDescriptorSet> texture_writes;
        texture_infos.reserve(textures_.size());
        texture_writes.reserve(textures_.size());
        for (const auto& [_, texture] : textures_)
        {
            if (!texture.handle.valid() || texture.handle.index >= virtual_geometry_bindless_texture_capacity ||
                texture.view == VK_NULL_HANDLE || texture.sampler == VK_NULL_HANDLE)
                continue;
            texture_infos.push_back({texture.sampler, texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = replacement_set;
            write.dstBinding = 17u;
            write.dstArrayElement = texture.handle.index;
            write.descriptorCount = 1u;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            texture_writes.push_back(write);
        }
        for (std::size_t index = 0u; index < texture_writes.size(); ++index)
            texture_writes[index].pImageInfo = &texture_infos[index];
        if (!texture_writes.empty())
            vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(texture_writes.size()), texture_writes.data(),
                                   0u, nullptr);

        const auto retired_pool = virtual_geometry_material_descriptor_pool_;
        virtual_geometry_material_descriptor_pool_ = replacement_pool;
        virtual_geometry_material_descriptor_set_ = replacement_set;
        if (retired_pool != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(), [this, retired_pool]()
                                     { vkDestroyDescriptorPool(device_, retired_pool, nullptr); });
        virtual_geometry_material_descriptors_dirty_ = false;
    }

    if (virtual_geometry_material_pipeline_layout_ == VK_NULL_HANDLE)
    {
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1u;
        layout.pSetLayouts = &virtual_geometry_material_descriptor_set_layout_;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &virtual_geometry_material_pipeline_layout_) !=
            VK_SUCCESS)
            return false;
    }
    if (virtual_geometry_material_pipeline_ == VK_NULL_HANDLE)
    {
        const auto shader = create_shader_module(builtin::virtual_geometry_material_resolve_comp_spv,
                                                 std::size(builtin::virtual_geometry_material_resolve_comp_spv));
        if (shader == VK_NULL_HANDLE) return false;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = shader;
        pipeline.stage.pName = "main";
        pipeline.layout = virtual_geometry_material_pipeline_layout_;
        const auto status = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                     &virtual_geometry_material_pipeline_);
        vkDestroyShaderModule(device_, shader, nullptr);
        if (status != VK_SUCCESS) return false;
    }
    return true;
}

bool vulkan_render_backend::dispatch_virtual_geometry_material_resolve(VkCommandBuffer command_buffer)
{
    if (!resolved_config_.features.virtual_geometry || !ensure_virtual_geometry_material_resources()) return false;
    virtual_geometry_material_frame_data frame{};
    std::copy_n(frame_camera_.view_projection.data(), 16u, frame.view_projection);
    std::copy_n(frame_camera_.previous_view_projection.data(), 16u, frame.previous_view_projection);
    frame.viewport_material_texture_debug[0] = viewport_width_;
    frame.viewport_material_texture_debug[1] = viewport_height_;
    const auto& material_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::material)];
    const auto& texture_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::texture)];
    frame.viewport_material_texture_debug[2] = static_cast<std::uint32_t>(material_table.generations.size());
    frame.viewport_material_texture_debug[3] = std::min(static_cast<std::uint32_t>(texture_table.generations.size()),
                                                        virtual_geometry_bindless_texture_capacity);
    if (!update_host_visible_buffer(virtual_geometry_material_frame_buffer_, &frame, sizeof(frame))) return false;

    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    std::array<VkImageMemoryBarrier, 2> visibility_barriers{};
    const std::array visibility_images{virtual_geometry_visibility_ids_.image, virtual_geometry_encoded_depth_.image};
    for (std::size_t index = 0; index < visibility_barriers.size(); ++index)
    {
        visibility_barriers[index].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        visibility_barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        visibility_barriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        visibility_barriers[index].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        visibility_barriers[index].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        visibility_barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        visibility_barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        visibility_barriers[index].image = visibility_images[index];
        visibility_barriers[index].subresourceRange = range;
    }
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                         0u, nullptr, 0u, nullptr, static_cast<std::uint32_t>(visibility_barriers.size()),
                         visibility_barriers.data());

    transition_graph_image(command_buffer, gbuffer_albedo_, VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, gbuffer_normal_, VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, gbuffer_material_, VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, gbuffer_emissive_, VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, virtual_geometry_material_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, virtual_geometry_material_pipeline_layout_,
                            0u, 1u, &virtual_geometry_material_descriptor_set_, 0u, nullptr);
    vkCmdDispatch(command_buffer, (viewport_width_ + 7u) / 8u, (viewport_height_ + 7u) / 8u, 1u);
    return true;
}

bool vulkan_render_backend::ensure_virtual_geometry_traversal_resources()
{
    if (!capabilities_.virtual_geometry_streaming || virtual_meshes_.empty() || gpu_scene_capacity_ == 0u ||
        virtual_geometry_resource_buffer_.buffer == VK_NULL_HANDLE)
        return false;
    const auto requested_visible_capacity =
        std::clamp(std::bit_ceil(std::max(1u, static_cast<std::uint32_t>(virtual_geometry_cluster_mirror_.size()))),
                   256u, 1u << 20u);
    constexpr std::uint32_t requested_page_capacity = 4096u;
    if (virtual_geometry_visible_capacity_ < requested_visible_capacity ||
        virtual_geometry_request_capacity_ < requested_page_capacity ||
        virtual_geometry_visible_buffer_.buffer == VK_NULL_HANDLE)
    {
        gpu_buffer visible{};
        gpu_buffer requests{};
        gpu_buffer counters{};
        if (!create_buffer(buffer_size(requested_visible_capacity, sizeof(virtual_geometry_visible_cluster_record)),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VMA_MEMORY_USAGE_GPU_ONLY, visible) ||
            !create_buffer(buffer_size(requested_page_capacity, sizeof(virtual_geometry_gpu_page_request)),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VMA_MEMORY_USAGE_GPU_ONLY, requests) ||
            !create_buffer(sizeof(virtual_geometry_traversal_counter_data),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VMA_MEMORY_USAGE_GPU_ONLY, counters))
        {
            destroy_buffer(visible);
            destroy_buffer(requests);
            destroy_buffer(counters);
            return false;
        }
        auto retired_visible = virtual_geometry_visible_buffer_;
        auto retired_requests = virtual_geometry_request_buffer_;
        auto retired_counters = virtual_geometry_counter_buffer_;
        virtual_geometry_visible_buffer_ = visible;
        virtual_geometry_request_buffer_ = requests;
        virtual_geometry_counter_buffer_ = counters;
        virtual_geometry_visible_capacity_ = requested_visible_capacity;
        virtual_geometry_request_capacity_ = requested_page_capacity;
        virtual_geometry_traversal_descriptors_dirty_ = true;
        virtual_geometry_raster_descriptors_dirty_ = true;
        virtual_geometry_material_descriptors_dirty_ = true;
        if (retired_visible.buffer != VK_NULL_HANDLE || retired_requests.buffer != VK_NULL_HANDLE ||
            retired_counters.buffer != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                     [this, retired_visible, retired_requests, retired_counters]() mutable
                                     {
                                         destroy_buffer(retired_visible);
                                         destroy_buffer(retired_requests);
                                         destroy_buffer(retired_counters);
                                     });
    }

    if (virtual_geometry_traversal_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 11> bindings{};
        for (std::uint32_t binding = 0; binding < 10u; ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[10] = {10u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr,
                                        &virtual_geometry_traversal_descriptor_set_layout_) != VK_SUCCESS)
            return false;
        virtual_geometry_traversal_descriptors_dirty_ = true;
    }

    if (virtual_geometry_traversal_descriptors_dirty_)
    {
        VkDescriptorPool replacement_pool{};
        VkDescriptorSet replacement_set{};
        const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10u},
                                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2u}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1u;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &replacement_pool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = replacement_pool;
        allocate.descriptorSetCount = 1u;
        allocate.pSetLayouts = &virtual_geometry_traversal_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &replacement_set) != VK_SUCCESS)
        {
            vkDestroyDescriptorPool(device_, replacement_pool, nullptr);
            return false;
        }
        const std::array buffers{
            VkDescriptorBufferInfo{gpu_scene_visibility_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_scene_transform_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_resource_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_node_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_child_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_root_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_page_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_visible_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_request_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{virtual_geometry_counter_buffer_.buffer, 0, VK_WHOLE_SIZE},
        };
        std::array<VkWriteDescriptorSet, 10> writes{};
        for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
        {
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = replacement_set;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1u;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &buffers[binding];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        const VkDescriptorImageInfo fallback{white_sampler_, white_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        std::array<VkDescriptorImageInfo, 2> hzb_images{fallback, fallback};
        if (ensure_hzb_resources(viewport_width_, viewport_height_))
            for (std::size_t index = 0; index < hzb_images.size(); ++index)
                hzb_images[index] = {hzb_sampler_, hzb_history_[index].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet image_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        image_write.dstSet = replacement_set;
        image_write.dstBinding = 10u;
        image_write.descriptorCount = static_cast<std::uint32_t>(hzb_images.size());
        image_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        image_write.pImageInfo = hzb_images.data();
        vkUpdateDescriptorSets(device_, 1u, &image_write, 0u, nullptr);
        const auto retired_pool = virtual_geometry_traversal_descriptor_pool_;
        virtual_geometry_traversal_descriptor_pool_ = replacement_pool;
        virtual_geometry_traversal_descriptor_set_ = replacement_set;
        if (retired_pool != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(), [this, retired_pool]()
                                     { vkDestroyDescriptorPool(device_, retired_pool, nullptr); });
        virtual_geometry_traversal_descriptors_dirty_ = false;
    }

    if (virtual_geometry_traversal_pipeline_ == VK_NULL_HANDLE)
    {
        const auto shader = create_shader_module(builtin::virtual_geometry_traversal_comp_spv,
                                                 std::size(builtin::virtual_geometry_traversal_comp_spv));
        if (shader == VK_NULL_HANDLE) return false;
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(virtual_geometry_traversal_push_constants)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1u;
        layout.pSetLayouts = &virtual_geometry_traversal_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1u;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &virtual_geometry_traversal_pipeline_layout_) !=
            VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, shader, nullptr);
            return false;
        }
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = shader;
        pipeline.stage.pName = "main";
        pipeline.layout = virtual_geometry_traversal_pipeline_layout_;
        const auto status = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                     &virtual_geometry_traversal_pipeline_);
        vkDestroyShaderModule(device_, shader, nullptr);
        if (status != VK_SUCCESS) return false;
    }
    return true;
}

bool vulkan_render_backend::ensure_virtual_geometry_feedback_frame(virtual_geometry_feedback_frame& frame)
{
    if (frame.requests.buffer == VK_NULL_HANDLE &&
        !create_buffer(buffer_size(virtual_geometry_request_capacity_, sizeof(virtual_geometry_gpu_page_request)),
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU, frame.requests))
        return false;
    return frame.counters.buffer != VK_NULL_HANDLE ||
           create_buffer(sizeof(virtual_geometry_traversal_counter_data), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VMA_MEMORY_USAGE_GPU_TO_CPU, frame.counters);
}

void vulkan_render_backend::dispatch_virtual_geometry_traversal(VkCommandBuffer command_buffer)
{
    if (!ensure_virtual_geometry_traversal_resources()) return;
    vkCmdFillBuffer(command_buffer, virtual_geometry_counter_buffer_.buffer, 0u, VK_WHOLE_SIZE, 0u);
    VkBufferMemoryBarrier counter_input{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    counter_input.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    counter_input.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    counter_input.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    counter_input.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    counter_input.buffer = virtual_geometry_counter_buffer_.buffer;
    counter_input.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u,
                         nullptr, 1u, &counter_input, 0u, nullptr);

    virtual_geometry_traversal_push_constants constants{};
    std::copy_n(frame_camera_.view_projection.data(), 16u, constants.view_projection);
    constants.camera_position_and_error[0] = frame_camera_.position[0];
    constants.camera_position_and_error[1] = frame_camera_.position[1];
    constants.camera_position_and_error[2] = frame_camera_.position[2];
    constants.camera_position_and_error[3] = resolved_config_.geometry_error_threshold;
    constants.capacities[0] = gpu_scene_capacity_;
    constants.capacities[1] = static_cast<std::uint32_t>(virtual_geometry_resource_mirror_.size());
    constants.capacities[2] = virtual_geometry_visible_capacity_;
    constants.capacities[3] = virtual_geometry_request_capacity_;
    constants.viewport_hzb[0] = static_cast<float>(viewport_width_);
    constants.viewport_hzb[1] = static_cast<float>(viewport_height_);
    constants.viewport_hzb[2] = static_cast<float>(hzb_mip_count_);
    constants.viewport_hzb[3] = hzb_history_valid_ && !frame_camera_.camera_cut ? 1.0f : 0.0f;
    constants.hzb_generation =
        static_cast<std::uint32_t>((last_profile_.frame_index + hzb_history_.size() - 1u) % hzb_history_.size());
    constants.camera_cut = frame_camera_.camera_cut ? 1u : 0u;
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, virtual_geometry_traversal_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, virtual_geometry_traversal_pipeline_layout_,
                            0u, 1u, &virtual_geometry_traversal_descriptor_set_, 0u, nullptr);
    vkCmdPushConstants(command_buffer, virtual_geometry_traversal_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                       sizeof(constants), &constants);
    vkCmdDispatch(command_buffer, (gpu_scene_capacity_ + 63u) / 64u, 1u, 1u);

    dispatch_virtual_geometry_raster(command_buffer);

    std::array<VkBufferMemoryBarrier, 2> outputs{};
    const std::array output_buffers{virtual_geometry_request_buffer_.buffer, virtual_geometry_counter_buffer_.buffer};
    for (std::size_t index = 0; index < outputs.size(); ++index)
    {
        outputs[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        outputs[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        outputs[index].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        outputs[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outputs[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outputs[index].buffer = output_buffers[index];
        outputs[index].size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u,
                         nullptr, static_cast<std::uint32_t>(outputs.size()), outputs.data(), 0u, nullptr);
    if (virtual_geometry_feedback_frames_.size() < frame_resource_count())
        virtual_geometry_feedback_frames_.resize(frame_resource_count());
    auto& feedback = virtual_geometry_feedback_frames_[current_frame_slot()];
    if (ensure_virtual_geometry_feedback_frame(feedback))
    {
        VkBufferCopy request_copy{
            .size = buffer_size(virtual_geometry_request_capacity_, sizeof(virtual_geometry_gpu_page_request))};
        VkBufferCopy counter_copy{.size = sizeof(virtual_geometry_traversal_counter_data)};
        vkCmdCopyBuffer(command_buffer, virtual_geometry_request_buffer_.buffer, feedback.requests.buffer, 1u,
                        &request_copy);
        vkCmdCopyBuffer(command_buffer, virtual_geometry_counter_buffer_.buffer, feedback.counters.buffer, 1u,
                        &counter_copy);
        std::array<VkBufferMemoryBarrier, 2> host_barriers{};
        const std::array host_buffers{feedback.requests.buffer, feedback.counters.buffer};
        for (std::size_t index = 0; index < host_barriers.size(); ++index)
        {
            host_barriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            host_barriers[index].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            host_barriers[index].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            host_barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_barriers[index].buffer = host_buffers[index];
            host_barriers[index].size = VK_WHOLE_SIZE;
        }
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u,
                             nullptr, static_cast<std::uint32_t>(host_barriers.size()), host_barriers.data(), 0u,
                             nullptr);
        feedback.submitted_frame = last_profile_.frame_index;
    }
}

void vulkan_render_backend::collect_virtual_geometry_feedback(std::uint32_t frame_index)
{
    if (frame_index >= virtual_geometry_feedback_frames_.size()) return;
    auto& frame = virtual_geometry_feedback_frames_[frame_index];
    if (frame.submitted_frame == 0u) return;
    void* counters_mapped{};
    if (vmaMapMemory(allocator_, frame.counters.allocation, &counters_mapped) != VK_SUCCESS) return;
    vmaInvalidateAllocation(allocator_, frame.counters.allocation, 0u, sizeof(virtual_geometry_traversal_counter_data));
    virtual_geometry_traversal_counter_data counters{};
    std::memcpy(&counters, counters_mapped, sizeof(counters));
    vmaUnmapMemory(allocator_, frame.counters.allocation);
    const auto request_count = std::min(counters.request_count, virtual_geometry_request_capacity_);
    completed_virtual_geometry_feedback_ = {.frame_index = frame.submitted_frame,
                                            .overflow = {.visible_cluster_overflow = counters.visible_overflow,
                                                         .page_request_overflow = counters.request_overflow,
                                                         .fallback_instance_count = counters.fallback_instances}};
    if (request_count != 0u)
    {
        void* requests_mapped{};
        if (vmaMapMemory(allocator_, frame.requests.allocation, &requests_mapped) == VK_SUCCESS)
        {
            const auto bytes = buffer_size(request_count, sizeof(virtual_geometry_gpu_page_request));
            vmaInvalidateAllocation(allocator_, frame.requests.allocation, 0u, bytes);
            const auto* requests = static_cast<const virtual_geometry_gpu_page_request*>(requests_mapped);
            completed_virtual_geometry_feedback_.page_requests.assign(requests, requests + request_count);
            vmaUnmapMemory(allocator_, frame.requests.allocation);
        }
    }
    auto& profile = last_profile_.virtual_geometry;
    profile.visible_clusters = std::min(counters.visible_count, virtual_geometry_visible_capacity_);
    profile.frustum_rejected = counters.frustum_rejected;
    profile.cone_rejected = counters.cone_rejected;
    profile.hzb_rejected = counters.hzb_rejected;
    profile.projected_size_rejected = counters.projected_size_rejected;
    profile.requested_pages = request_count;
    profile.parent_fallbacks = counters.parent_fallbacks;
    profile.overflowed_clusters = counters.visible_overflow + counters.traversal_overflow;
    profile.fallback_instances = counters.fallback_instances;
    frame.submitted_frame = 0u;
}

} // namespace arc::render::vulkan::backend_detail
