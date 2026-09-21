#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"
#include "vulkan_sky_constants.h"

namespace arc::render::vulkan::backend_detail
{
bool vulkan_render_backend::ensure_mesh_pipeline()
{
    if (mesh_pipeline_ != VK_NULL_HANDLE) return true;
    if (max_push_constant_bytes_ < sizeof(mesh_push_constants))
    {
        if (!push_constant_limit_warning_reported_)
        {
            arc::diagnostics::error("render.vulkan",
                                    "The selected adapter exposes only " + std::to_string(max_push_constant_bytes_) +
                                        " push-constant bytes; ARC's raster mesh path currently requires " +
                                        std::to_string(sizeof(mesh_push_constants)));
            push_constant_limit_warning_reported_ = true;
        }
        return false;
    }
    if (!ensure_white_texture()) return false;

    if (material_attribute_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        const VkDescriptorSetLayoutBinding binding{0u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo descriptor_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptor_layout.bindingCount = 1u;
        descriptor_layout.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr,
                                        &material_attribute_descriptor_set_layout_) != VK_SUCCESS)
            return false;
    }
    if (material_attribute_descriptor_pool_ == VK_NULL_HANDLE)
    {
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                             material_attribute_descriptor_set_capacity};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = material_attribute_descriptor_set_capacity;
        pool.poolSizeCount = 1u;
        pool.pPoolSizes = &pool_size;
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &material_attribute_descriptor_pool_) != VK_SUCCESS)
            return false;
    }

    VkShaderModule vert =
        create_shader_module(builtin::default_phong_vert_spv, std::size(builtin::default_phong_vert_spv));
    VkShaderModule frag =
        create_shader_module(builtin::default_phong_frag_spv, std::size(builtin::default_phong_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = sizeof(mesh_push_constants);

    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &white_descriptor_set_layout_;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &mesh_pipeline_layout_) != VK_SUCCESS) return false;
    const std::array terrain_surface_set_layouts{white_descriptor_set_layout_,
                                                 material_attribute_descriptor_set_layout_};
    layout.setLayoutCount = static_cast<std::uint32_t>(terrain_surface_set_layouts.size());
    layout.pSetLayouts = terrain_surface_set_layouts.data();
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &terrain_surface_pipeline_layout_) != VK_SUCCESS)
        return false;
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
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

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
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_pipeline_);
    if (result == VK_SUCCESS)
    {
        depth.depthWriteEnable = VK_FALSE;
        color_attachment.blendEnable = VK_TRUE;
        color_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        color_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        color_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        color_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        color_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
        const VkResult blend_result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_transparent_pipeline_);
        if (blend_result != VK_SUCCESS)
            arc::diagnostics::warn(
                "render.vulkan",
                "Vulkan transparent mesh pipeline creation failed; blended materials will render opaque");
        color_attachment = {};
        color_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        depth.depthWriteEnable = VK_TRUE;
    }
    if (result == VK_SUCCESS && capabilities_.fill_mode_non_solid)
    {
        raster.polygonMode = VK_POLYGON_MODE_LINE;
        depth.depthWriteEnable = VK_FALSE;
        const VkResult wire_result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_wire_pipeline_);
        if (wire_result != VK_SUCCESS)
            arc::diagnostics::warn("render.vulkan",
                                   "Vulkan wireframe pipeline creation failed; shaded rendering will continue");
    }
    else if (result == VK_SUCCESS && !capabilities_.fill_mode_non_solid && !wireframe_warning_reported_)
    {
        arc::diagnostics::warn("render.vulkan",
                               "Vulkan device does not support fillModeNonSolid; wireframe rendering is disabled");
        wireframe_warning_reported_ = true;
    }

    if (result == VK_SUCCESS)
    {
        const auto selection_frag =
            create_shader_module(builtin::selection_mask_frag_spv, std::size(builtin::selection_mask_frag_spv));
        if (selection_frag != VK_NULL_HANDLE)
        {
            stages[1].module = selection_frag;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            depth.depthWriteEnable = VK_FALSE;
            color_attachment = {};
            color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
            const VkFormat selection_format = VK_FORMAT_R8_UNORM;
            rendering.pColorAttachmentFormats = &selection_format;
            pipeline.layout = mesh_pipeline_layout_;
            if (vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr,
                                          &selection_mask_pipeline_) != VK_SUCCESS)
                arc::diagnostics::warn("render.vulkan",
                                       "Vulkan selection-mask pipeline creation failed; outlines are disabled");
            vkDestroyShaderModule(device_, selection_frag, nullptr);
            stages[1].module = frag;
            rendering.pColorAttachmentFormats = &scene_color_format_;
        }
    }

    if (result == VK_SUCCESS)
    {
        VkShaderModule terrain_surface_frag = create_shader_module(
            builtin::terrain_surface_forward_frag_spv, std::size(builtin::terrain_surface_forward_frag_spv));
        if (terrain_surface_frag != VK_NULL_HANDLE)
        {
            stages[0].module = vert;
            stages[1].module = terrain_surface_frag;
            pipeline.pVertexInputState = &vertex_input;
            pipeline.layout = terrain_surface_pipeline_layout_;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            depth.depthWriteEnable = VK_TRUE;
            color_attachment = {};
            color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            if (vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr,
                                          &terrain_surface_pipeline_) != VK_SUCCESS)
            {
                terrain_surface_pipeline_ = VK_NULL_HANDLE;
                arc::diagnostics::warn("render.vulkan",
                                       "Vulkan terrain surface pipeline creation failed; using mesh fallback");
            }
            vkDestroyShaderModule(device_, terrain_surface_frag, nullptr);
        }
    }

    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_debug_overlay_pipeline()
{
    if (debug_overlay_line_pipeline_ != VK_NULL_HANDLE && debug_overlay_triangle_pipeline_ != VK_NULL_HANDLE &&
        debug_overlay_output_line_pipeline_ != VK_NULL_HANDLE &&
        debug_overlay_output_triangle_pipeline_ != VK_NULL_HANDLE)
        return true;
    VkShaderModule vert =
        create_shader_module(builtin::debug_overlay_vert_spv, std::size(builtin::debug_overlay_vert_spv));
    VkShaderModule frag =
        create_shader_module(builtin::debug_overlay_frag_spv, std::size(builtin::debug_overlay_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
    {
        if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.size = sizeof(float) * 16u;
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &debug_overlay_pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 nullptr,
                 0,
                 VK_SHADER_STAGE_VERTEX_BIT,
                 vert,
                 "main",
                 nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 nullptr,
                 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT,
                 frag,
                 "main",
                 nullptr};
    VkVertexInputBindingDescription binding{0, sizeof(debug_overlay_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array<VkVertexInputAttributeDescription, 2> attributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(debug_overlay_vertex, position)},
        VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(debug_overlay_vertex, color)}};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &blend;
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    const auto create_pipeline =
        [&](VkPrimitiveTopology topology, VkFormat color_format, bool depth_test, VkPipeline& destination)
    {
        input.topology = topology;
        depth.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &color_format;
        rendering.depthAttachmentFormat = depth_test ? depth_format_ : VK_FORMAT_UNDEFINED;
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline.pNext = &rendering;
        pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
        pipeline.pStages = stages.data();
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &depth;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = debug_overlay_pipeline_layout_;
        return vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &destination);
    };

    const auto tested_line_result =
        create_pipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, scene_color_format_, true, debug_overlay_line_pipeline_);
    const auto tested_triangle_result = tested_line_result == VK_SUCCESS
                                            ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, scene_color_format_,
                                                              true, debug_overlay_triangle_pipeline_)
                                            : VK_ERROR_INITIALIZATION_FAILED;
    const auto output_line_result = tested_triangle_result == VK_SUCCESS
                                        ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, viewport_format_, false,
                                                          debug_overlay_output_line_pipeline_)
                                        : VK_ERROR_INITIALIZATION_FAILED;
    const auto output_triangle_result = output_line_result == VK_SUCCESS
                                            ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, viewport_format_,
                                                              false, debug_overlay_output_triangle_pipeline_)
                                            : VK_ERROR_INITIALIZATION_FAILED;
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (tested_line_result != VK_SUCCESS || tested_triangle_result != VK_SUCCESS || output_line_result != VK_SUCCESS ||
        output_triangle_result != VK_SUCCESS)
    {
        arc::diagnostics::warn("render.vulkan", "Vulkan debug-overlay pipeline creation failed");
        const auto destroy_pipeline = [&](VkPipeline& pipeline)
        {
            if (pipeline == VK_NULL_HANDLE) return;
            vkDestroyPipeline(device_, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        };
        destroy_pipeline(debug_overlay_line_pipeline_);
        destroy_pipeline(debug_overlay_triangle_pipeline_);
        destroy_pipeline(debug_overlay_output_line_pipeline_);
        destroy_pipeline(debug_overlay_output_triangle_pipeline_);
        vkDestroyPipelineLayout(device_, debug_overlay_pipeline_layout_, nullptr);
        debug_overlay_pipeline_layout_ = VK_NULL_HANDLE;
    }
    return tested_line_result == VK_SUCCESS && tested_triangle_result == VK_SUCCESS &&
           output_line_result == VK_SUCCESS && output_triangle_result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_gpu_bindless_pipelines()
{
    if (!resolved_config_.features.gpu_visibility_compaction ||
        shared_geometry_buffers_.vertices.buffer == VK_NULL_HANDLE ||
        shared_geometry_buffers_.indices.buffer == VK_NULL_HANDLE ||
        gpu_scene_visibility_buffer_.buffer == VK_NULL_HANDLE)
        return false;
    const auto& material_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::material)];
    const auto& texture_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::texture)];
    if (material_table.storage.buffer == VK_NULL_HANDLE) return false;

    if (gpu_bindless_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        for (std::uint32_t binding = 0u; binding < 4u; ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[4] = {4u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, virtual_geometry_bindless_texture_capacity,
                       VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        std::array<VkDescriptorBindingFlags, 5> binding_flags{};
        binding_flags[4] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
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
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &gpu_bindless_descriptor_set_layout_) != VK_SUCCESS)
            return false;
        gpu_bindless_descriptors_dirty_ = true;
    }

    if (gpu_bindless_descriptors_dirty_)
    {
        VkDescriptorPool replacement_pool{};
        VkDescriptorSet replacement_set{};
        const std::array pool_sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4u},
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
        allocate.pSetLayouts = &gpu_bindless_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &replacement_set) != VK_SUCCESS)
        {
            vkDestroyDescriptorPool(device_, replacement_pool, nullptr);
            return false;
        }
        const auto texture_buffer = texture_table.storage.buffer != VK_NULL_HANDLE ? texture_table.storage.buffer
                                                                                   : material_table.storage.buffer;
        const std::array buffer_infos{
            VkDescriptorBufferInfo{gpu_scene_visibility_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_scene_transform_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{material_table.storage.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{texture_buffer, 0u, VK_WHOLE_SIZE},
        };
        std::array<VkWriteDescriptorSet, 4> buffer_writes{};
        for (std::uint32_t binding = 0u; binding < buffer_writes.size(); ++binding)
        {
            buffer_writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            buffer_writes[binding].dstSet = replacement_set;
            buffer_writes[binding].dstBinding = binding;
            buffer_writes[binding].descriptorCount = 1u;
            buffer_writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            buffer_writes[binding].pBufferInfo = &buffer_infos[binding];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(buffer_writes.size()), buffer_writes.data(), 0u,
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
            write.dstBinding = 4u;
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
        const auto retired_pool = gpu_bindless_descriptor_pool_;
        gpu_bindless_descriptor_pool_ = replacement_pool;
        gpu_bindless_descriptor_set_ = replacement_set;
        if (retired_pool != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(), [this, retired_pool]()
                                     { vkDestroyDescriptorPool(device_, retired_pool, nullptr); });
        gpu_bindless_descriptors_dirty_ = false;
    }

    if (gpu_bindless_pipeline_layout_ == VK_NULL_HANDLE)
    {
        VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0u, sizeof(float) * 32u};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1u;
        layout.pSetLayouts = &gpu_bindless_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1u;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &gpu_bindless_pipeline_layout_) != VK_SUCCESS)
            return false;
    }
    if (gpu_bindless_gbuffer_pipeline_ != VK_NULL_HANDLE && gpu_bindless_transparent_pipeline_ != VK_NULL_HANDLE)
        return true;

    const auto vertex_shader =
        create_shader_module(builtin::gpu_scene_bindless_vert_spv, std::size(builtin::gpu_scene_bindless_vert_spv));
    const auto gbuffer_shader = create_shader_module(builtin::gpu_scene_bindless_gbuffer_frag_spv,
                                                     std::size(builtin::gpu_scene_bindless_gbuffer_frag_spv));
    const auto transparent_shader = create_shader_module(builtin::gpu_scene_bindless_transparent_frag_spv,
                                                         std::size(builtin::gpu_scene_bindless_transparent_frag_spv));
    if (vertex_shader == VK_NULL_HANDLE || gbuffer_shader == VK_NULL_HANDLE || transparent_shader == VK_NULL_HANDLE)
    {
        if (vertex_shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vertex_shader, nullptr);
        if (gbuffer_shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, gbuffer_shader, nullptr);
        if (transparent_shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, transparent_shader, nullptr);
        return false;
    }
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_shader;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{0u, sizeof(mesh_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array attributes{
        VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)},
        VkVertexInputAttributeDescription{1u, 0u, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)},
        VkVertexInputAttributeDescription{2u, 0u, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)},
        VkVertexInputAttributeDescription{3u, 0u, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)},
        VkVertexInputAttributeDescription{4u, 0u, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)},
    };
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1u;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1u;
    viewport.scissorCount = 1u;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    std::array<VkPipelineColorBlendAttachmentState, 6> gbuffer_attachments{};
    for (auto& attachment : gbuffer_attachments)
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = static_cast<std::uint32_t>(gbuffer_attachments.size());
    blend.pAttachments = gbuffer_attachments.data();
    const std::array<VkFormat, 6> gbuffer_formats{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                  VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                  VK_FORMAT_R16G16_SFLOAT,       VK_FORMAT_R32_UINT};
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(gbuffer_formats.size());
    rendering.pColorAttachmentFormats = gbuffer_formats.data();
    rendering.depthAttachmentFormat = depth_format_;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = gpu_bindless_pipeline_layout_;
    stages[1].module = gbuffer_shader;
    const auto gbuffer_result = gpu_bindless_gbuffer_pipeline_ != VK_NULL_HANDLE
                                    ? VK_SUCCESS
                                    : vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                                &gpu_bindless_gbuffer_pipeline_);

    VkPipelineColorBlendAttachmentState transparent_attachment{};
    transparent_attachment.blendEnable = VK_TRUE;
    transparent_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    transparent_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    transparent_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    transparent_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    transparent_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    transparent_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    transparent_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend.attachmentCount = 1u;
    blend.pAttachments = &transparent_attachment;
    rendering.colorAttachmentCount = 1u;
    rendering.pColorAttachmentFormats = &scene_color_format_;
    stages[1].module = transparent_shader;
    const auto transparent_result = gpu_bindless_transparent_pipeline_ != VK_NULL_HANDLE
                                        ? VK_SUCCESS
                                        : vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                                    &gpu_bindless_transparent_pipeline_);
    vkDestroyShaderModule(device_, vertex_shader, nullptr);
    vkDestroyShaderModule(device_, gbuffer_shader, nullptr);
    vkDestroyShaderModule(device_, transparent_shader, nullptr);
    return gbuffer_result == VK_SUCCESS && transparent_result == VK_SUCCESS;
}

} // namespace arc::render::vulkan::backend_detail
