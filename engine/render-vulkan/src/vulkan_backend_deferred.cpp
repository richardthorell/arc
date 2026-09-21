#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"
#include "vulkan_sky_constants.h"

namespace arc::render::vulkan::backend_detail
{
bool vulkan_render_backend::ensure_gbuffer_pipeline()
{
    if (gbuffer_pipeline_ != VK_NULL_HANDLE) return true;
    if (!ensure_mesh_pipeline()) return false;

    VkShaderModule vert = create_shader_module(builtin::gbuffer_vert_spv, std::size(builtin::gbuffer_vert_spv));
    VkShaderModule frag = create_shader_module(builtin::gbuffer_frag_spv, std::size(builtin::gbuffer_frag_spv));
    VkShaderModule terrain_surface_frag = create_shader_module(builtin::terrain_surface_gbuffer_frag_spv,
                                                               std::size(builtin::terrain_surface_gbuffer_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
    {
        if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
        if (terrain_surface_frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_surface_frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(mesh_vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 5> attributes{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)};
    attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)};
    attributes[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)};
    attributes[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)};
    attributes[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)};

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    std::array<VkPipelineColorBlendAttachmentState, 6> attachments{};
    for (auto& attachment : attachments)
    {
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    color_blend.pAttachments = attachments.data();

    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    const std::array<VkFormat, 6> color_formats{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_FORMAT_R16G16_SFLOAT,       VK_FORMAT_R32_UINT};
    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size());
    rendering.pColorAttachmentFormats = color_formats.data();
    rendering.depthAttachmentFormat = depth_format_;

    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = &rendering;
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = mesh_pipeline_layout_;
    pipeline.renderPass = VK_NULL_HANDLE;

    const VkResult result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &gbuffer_pipeline_);
    if (result == VK_SUCCESS && terrain_surface_frag != VK_NULL_HANDLE)
    {
        stages[0].module = vert;
        stages[1].module = terrain_surface_frag;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.layout = terrain_surface_pipeline_layout_;
        if (vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr,
                                      &terrain_surface_gbuffer_pipeline_) != VK_SUCCESS)
        {
            terrain_surface_gbuffer_pipeline_ = VK_NULL_HANDLE;
            arc::diagnostics::warn("render.vulkan",
                                   "Vulkan terrain surface G-buffer pipeline creation failed; using mesh fallback");
        }
    }
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (terrain_surface_frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_surface_frag, nullptr);
    if (result != VK_SUCCESS)
        arc::diagnostics::warn("render.vulkan",
                               "Vulkan G-buffer pipeline creation failed; falling back to forward rendering");
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_gbuffer_descriptor_set()
{
    if (gbuffer_descriptor_set_ != VK_NULL_HANDLE) return true;
    if (!ensure_white_texture() || gbuffer_albedo_.view == VK_NULL_HANDLE || gbuffer_normal_.view == VK_NULL_HANDLE ||
        gbuffer_material_.view == VK_NULL_HANDLE || gbuffer_emissive_.view == VK_NULL_HANDLE ||
        gbuffer_motion_.view == VK_NULL_HANDLE || gbuffer_object_id_.view == VK_NULL_HANDLE ||
        viewport_depth_view_ == VK_NULL_HANDLE || shadow_atlas_.array_view == VK_NULL_HANDLE ||
        shadow_atlas_.sampler == VK_NULL_HANDLE || local_shadow_atlas_.view == VK_NULL_HANDLE ||
        local_shadow_atlas_.sampler == VK_NULL_HANDLE || current_shadow_uniform_buffer() == nullptr)
        return false;

    if (gbuffer_sampler_ == VK_NULL_HANDLE)
    {
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &sampler, nullptr, &gbuffer_sampler_) != VK_SUCCESS) return false;
    }

    if (gbuffer_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
        for (std::uint32_t index = 0; index < 7; ++index)
        {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        bindings[7].binding = 7;
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[8].binding = 8;
        bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[9].binding = 9;
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[10].binding = 10;
        bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[10].descriptorCount = 1;
        bindings[10].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[11].binding = 11;
        bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[11].descriptorCount = 1;
        bindings[11].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &gbuffer_descriptor_set_layout_) != VK_SUCCESS)
            return false;
    }

    if (gbuffer_descriptor_pool_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorPoolSize, 3> pool_sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 1;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &gbuffer_descriptor_pool_) != VK_SUCCESS) return false;
    }

    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = gbuffer_descriptor_pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &gbuffer_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &allocate, &gbuffer_descriptor_set_) != VK_SUCCESS) return false;

    update_gbuffer_descriptor_set();
    return true;
}

void vulkan_render_backend::update_gbuffer_descriptor_set()
{
    if (gbuffer_descriptor_set_ == VK_NULL_HANDLE || light_buffer_.buffer == VK_NULL_HANDLE) return;

    std::array<VkDescriptorImageInfo, 10> images{};
    const VkSampler sampler = gbuffer_sampler_ != VK_NULL_HANDLE ? gbuffer_sampler_ : white_sampler_;
    images[0] = {sampler, gbuffer_albedo_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[1] = {sampler, gbuffer_normal_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[2] = {sampler, gbuffer_material_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[3] = {sampler, gbuffer_emissive_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[4] = {sampler, gbuffer_object_id_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[5] = {sampler, gbuffer_motion_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[6] = {sampler, viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[7] = {white_sampler_, white_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[8] = {shadow_atlas_.sampler, shadow_atlas_.array_view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    images[9] = {local_shadow_atlas_.sampler, local_shadow_atlas_.view,
                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    const environment_descriptor* environment{};
    if (frame_environment_.lighting.environment.valid())
    {
        if (const auto found = environments_.find(resource_key(frame_environment_.lighting.environment));
            found != environments_.end())
            environment = &found->second.data;
    }
    if (!environment) environment = active_environment();
    if (environment)
    {
        if (const auto found = textures_.find(resource_key(environment->equirectangular_texture));
            found != textures_.end() && found->second.view != VK_NULL_HANDLE)
        {
            images[7] = {found->second.sampler, found->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
    }

    std::array<VkWriteDescriptorSet, 10> writes{};
    for (std::uint32_t index = 0; index < 7; ++index)
    {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = gbuffer_descriptor_set_;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[index].pImageInfo = &images[index];
    }
    writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[7].dstSet = gbuffer_descriptor_set_;
    writes[7].dstBinding = 8;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &images[7];
    writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[8].dstSet = gbuffer_descriptor_set_;
    writes[8].dstBinding = 9;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &images[8];
    writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[9].dstSet = gbuffer_descriptor_set_;
    writes[9].dstBinding = 11;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[9].pImageInfo = &images[9];
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    VkDescriptorBufferInfo lights{light_buffer_.buffer, 0, sizeof(scene_lighting_data)};
    VkWriteDescriptorSet light_write{};
    light_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    light_write.dstSet = gbuffer_descriptor_set_;
    light_write.dstBinding = 7;
    light_write.descriptorCount = 1;
    light_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    light_write.pBufferInfo = &lights;
    const auto* shadow_buffer = current_shadow_uniform_buffer();
    if (shadow_buffer == nullptr) return;
    VkDescriptorBufferInfo shadow{shadow_buffer->buffer, 0, sizeof(shadow_uniform_data)};
    VkWriteDescriptorSet shadow_write{};
    shadow_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    shadow_write.dstSet = gbuffer_descriptor_set_;
    shadow_write.dstBinding = 10;
    shadow_write.descriptorCount = 1;
    shadow_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    shadow_write.pBufferInfo = &shadow;
    const std::array buffer_writes{light_write, shadow_write};
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(buffer_writes.size()), buffer_writes.data(), 0, nullptr);
}

bool vulkan_render_backend::ensure_deferred_pipeline()
{
    if (deferred_pipeline_ != VK_NULL_HANDLE) return true;
    if (!ensure_gbuffer_descriptor_set()) return false;

    VkShaderModule vert =
        create_shader_module(builtin::deferred_lighting_vert_spv, std::size(builtin::deferred_lighting_vert_spv));
    VkShaderModule frag =
        create_shader_module(builtin::deferred_lighting_frag_spv, std::size(builtin::deferred_lighting_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = sizeof(deferred_push_constants);

    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &gbuffer_descriptor_set_layout_;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &deferred_pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_attachment;
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &scene_color_format_;

    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = &rendering;
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = deferred_pipeline_layout_;
    pipeline.renderPass = VK_NULL_HANDLE;

    const VkResult result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &deferred_pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (result != VK_SUCCESS)
        arc::diagnostics::warn("render.vulkan",
                               "Vulkan deferred lighting pipeline creation failed; falling back to forward rendering");
    return result == VK_SUCCESS;
}

} // namespace arc::render::vulkan::backend_detail
