#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"
#include "vulkan_sky_constants.h"

namespace arc::render::vulkan::backend_detail
{
bool vulkan_render_backend::ensure_output_transform_pipeline()
{
    if (exposure_buffer_.buffer == VK_NULL_HANDLE &&
        !create_buffer(exposure_buffer_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_GPU_ONLY, exposure_buffer_))
        return false;

    if (output_transform_pipeline_ != VK_NULL_HANDLE)
    {
        VkDescriptorImageInfo image{viewport_sampler_,
                                    temporal_output_view_ != VK_NULL_HANDLE ? temporal_output_view_ : scene_color_.view,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo mask{viewport_sampler_, selection_mask_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (auto& write : writes)
        {
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = output_transform_descriptor_set_;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        writes[0].dstBinding = 0;
        writes[0].pImageInfo = &image;
        writes[1].dstBinding = 2;
        writes[1].pImageInfo = &mask;
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    }
    if (scene_color_.view == VK_NULL_HANDLE || selection_mask_.view == VK_NULL_HANDLE ||
        viewport_sampler_ == VK_NULL_HANDLE)
        return false;

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo descriptor_layout{};
    descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptor_layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr, &output_transform_descriptor_set_layout_) !=
        VK_SUCCESS)
        return false;

    std::array<VkDescriptorPoolSize, 2> pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
                                                   VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 1;
    pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &output_transform_descriptor_pool_) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = output_transform_descriptor_pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &output_transform_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &allocate, &output_transform_descriptor_set_) != VK_SUCCESS) return false;

    VkDescriptorImageInfo image{viewport_sampler_,
                                temporal_output_view_ != VK_NULL_HANDLE ? temporal_output_view_ : scene_color_.view,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo mask{viewport_sampler_, selection_mask_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo exposure_buffer_info{exposure_buffer_.buffer, 0, exposure_buffer_bytes};
    std::array<VkWriteDescriptorSet, 3> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = output_transform_descriptor_set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &image;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = output_transform_descriptor_set_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &exposure_buffer_info;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = output_transform_descriptor_set_;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &mask;
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    const auto vert =
        create_shader_module(builtin::deferred_lighting_vert_spv, std::size(builtin::deferred_lighting_vert_spv));
    const auto frag =
        create_shader_module(builtin::output_transform_frag_spv, std::size(builtin::output_transform_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push.size = sizeof(output_transform_push_constants);
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &output_transform_descriptor_set_layout_;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &output_transform_pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_attachment;
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &viewport_format_;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = output_transform_pipeline_layout_;
    const auto result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &output_transform_pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (result != VK_SUCCESS)
        arc::diagnostics::warn("render.vulkan", "Vulkan output-transform pipeline creation failed");
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_exposure_pipelines()
{
    if (luminance_histogram_pipeline_ != VK_NULL_HANDLE && exposure_resolve_pipeline_ != VK_NULL_HANDLE) return true;
    if (!ensure_output_transform_pipeline()) return false;

    const auto create_compute = [&](const std::uint32_t* words, std::size_t word_count, std::uint32_t push_size,
                                    VkPipelineLayout& pipeline_layout, VkPipeline& pipeline)
    {
        const auto shader = create_shader_module(words, word_count);
        if (shader == VK_NULL_HANDLE) return false;
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = push_size;
        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &output_transform_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &pipeline_layout) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, shader, nullptr);
            return false;
        }
        VkComputePipelineCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        create.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        create.stage.module = shader;
        create.stage.pName = "main";
        create.layout = pipeline_layout;
        const auto result = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &create, nullptr, &pipeline);
        vkDestroyShaderModule(device_, shader, nullptr);
        return result == VK_SUCCESS;
    };

    if (!create_compute(builtin::luminance_histogram_comp_spv, std::size(builtin::luminance_histogram_comp_spv),
                        sizeof(histogram_push_constants), luminance_histogram_pipeline_layout_,
                        luminance_histogram_pipeline_))
        return false;
    if (!create_compute(builtin::exposure_resolve_comp_spv, std::size(builtin::exposure_resolve_comp_spv),
                        sizeof(exposure_resolve_push_constants), exposure_resolve_pipeline_layout_,
                        exposure_resolve_pipeline_))
        return false;
    return true;
}

void vulkan_render_backend::dispatch_exposure(VkCommandBuffer command_buffer)
{
    if (!ensure_exposure_pipelines()) return;

    // The exposure state is persistent and shared between frames.
    // Serialize the transfer clear after the previous frame's resolve and
    // output-transform read before reusing the buffer.
    VkBufferMemoryBarrier reuse_barrier{};
    reuse_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    reuse_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    reuse_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    reuse_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    reuse_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    reuse_barrier.buffer = exposure_buffer_.buffer;
    reuse_barrier.size = exposure_buffer_bytes;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &reuse_barrier, 0, nullptr);

    const VkDeviceSize clear_size = exposure_needs_reset_ ? exposure_buffer_bytes : exposure_histogram_bytes;
    vkCmdFillBuffer(command_buffer, exposure_buffer_.buffer, 0, clear_size, 0u);
    VkBufferMemoryBarrier clear_barrier{};
    clear_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    clear_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    clear_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    clear_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clear_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clear_barrier.buffer = exposure_buffer_.buffer;
    clear_barrier.size = exposure_buffer_bytes;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 1, &clear_barrier, 0, nullptr);

    histogram_push_constants histogram{};
    histogram.log_luminance_extent[2] = static_cast<float>(viewport_width_);
    histogram.log_luminance_extent[3] = static_cast<float>(viewport_height_);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_histogram_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_histogram_pipeline_layout_, 0, 1,
                            &output_transform_descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer, luminance_histogram_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(histogram), &histogram);
    const std::uint32_t sample_width = (viewport_width_ + 3u) / 4u;
    const std::uint32_t sample_height = (viewport_height_ + 3u) / 4u;
    vkCmdDispatch(command_buffer, (sample_width + 15u) / 16u, (sample_height + 15u) / 16u, 1u);

    VkBufferMemoryBarrier histogram_barrier = clear_barrier;
    histogram_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    histogram_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 1, &histogram_barrier, 0, nullptr);

    exposure_resolve_push_constants resolve{};
    resolve.limits_speeds[0] = frame_camera_.exposure.minimum_ev100;
    resolve.limits_speeds[1] = frame_camera_.exposure.maximum_ev100;
    resolve.limits_speeds[2] = frame_camera_.exposure.brighten_speed;
    resolve.limits_speeds[3] = frame_camera_.exposure.darken_speed;
    resolve.timing_mode[1] = frame_camera_.exposure.mode == exposure_mode::automatic ? 1.0f : 0.0f;
    resolve.timing_mode[2] = frame_camera_.exposure.manual_ev100;
    resolve.timing_mode[3] = exposure_needs_reset_ ? 1.0f : 0.0f;
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_resolve_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_resolve_pipeline_layout_, 0, 1,
                            &output_transform_descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer, exposure_resolve_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(resolve), &resolve);
    vkCmdDispatch(command_buffer, 1u, 1u, 1u);

    VkBufferMemoryBarrier output_barrier = histogram_barrier;
    output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 1, &output_barrier, 0, nullptr);
    exposure_needs_reset_ = false;
}

} // namespace arc::render::vulkan::backend_detail
