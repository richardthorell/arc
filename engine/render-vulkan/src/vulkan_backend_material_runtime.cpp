#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"
#include "vulkan_sky_constants.h"

namespace arc::render::vulkan::backend_detail
{
namespace
{
std::optional<shader_parameter_type> material_texture_resource_type(std::string_view name) noexcept
{
    if (name == "arcMaterialTextures2D") return shader_parameter_type::texture_2d;
    if (name == "arcMaterialTexturesCube") return shader_parameter_type::texture_cube;
    if (name == "arcMaterialTextures3D") return shader_parameter_type::texture_3d;
    return std::nullopt;
}

texture_dimension material_texture_dimension(shader_parameter_type type) noexcept
{
    switch (type)
    {
        case shader_parameter_type::texture_cube:
            return texture_dimension::cube;
        case shader_parameter_type::texture_3d:
            return texture_dimension::texture_3d;
        default:
            return texture_dimension::texture_2d;
    }
}

std::string_view material_texture_type_name(shader_parameter_type type) noexcept
{
    switch (type)
    {
        case shader_parameter_type::texture_cube:
            return "TextureCube";
        case shader_parameter_type::texture_3d:
            return "Texture3D";
        default:
            return "Texture2D";
    }
}
} // namespace

void vulkan_render_backend::destroy_material_runtime(gpu_material_runtime& runtime) noexcept
{
    if (runtime.gbuffer_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, runtime.gbuffer_pipeline, nullptr);
    if (runtime.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, runtime.pipeline_layout, nullptr);
    if (runtime.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, runtime.descriptor_pool, nullptr);
    if (runtime.descriptor_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, runtime.descriptor_set_layout, nullptr);
    for (auto& buffer : runtime.parameter_buffers)
        destroy_buffer(buffer);
    for (auto& buffer : runtime.frame_buffers)
        destroy_buffer(buffer);
    runtime = {};
}

bool vulkan_render_backend::reject_runtime_material(gpu_material& material, std::string reason)
{
    const auto generation = material.data.runtime_program ? material.data.runtime_program->generation : 0u;
    destroy_material_runtime(material.runtime);
    material.runtime.generation = generation;
    material.runtime.failed = true;
    arc::diagnostics::warn("render.vulkan", "Compiled Material ABI G-buffer fallback for '" + material.data.name +
                                                "': " + std::move(reason));
    return false;
}

const material_runtime_pass* vulkan_render_backend::runtime_gbuffer_pass(const gpu_material& material) const noexcept
{
    if (!material.data.runtime_program) return nullptr;
    const auto& program = *material.data.runtime_program;
    if (!material_runtime_program_compatible(program)) return nullptr;
    const auto found = std::ranges::find(program.passes, material_pass::gbuffer, &material_runtime_pass::pass);
    return found == program.passes.end() ? nullptr : &*found;
}

bool vulkan_render_backend::update_runtime_parameter_buffer(gpu_buffer& buffer, const material_descriptor& material,
                                                            const material_runtime_program& program)
{
    const auto byte_size = std::max<std::size_t>(program.parameter_block_size, 16u);
    if (buffer.buffer == VK_NULL_HANDLE &&
        !create_buffer(byte_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, buffer))
        return false;

    void* mapped{};
    if (vmaMapMemory(allocator_, buffer.allocation, &mapped) != VK_SUCCESS) return false;
    std::memset(mapped, 0, byte_size);
    if (!program.parameter_defaults.empty())
        std::memcpy(mapped, program.parameter_defaults.data(), std::min(byte_size, program.parameter_defaults.size()));

    const auto copy_override = [&](const shader_parameter_descriptor& parameter, const material_parameter_value& value)
    {
        if (parameter.offset >= byte_size) return;
        const auto destination = static_cast<std::byte*>(mapped) + parameter.offset;
        const auto available = std::min<std::size_t>(parameter.size, byte_size - parameter.offset);
        const auto copy = [&](const void* source, std::size_t size)
        { std::memcpy(destination, source, std::min(size, available)); };
        switch (parameter.type)
        {
            case shader_parameter_type::boolean:
                if (const auto* typed = std::get_if<bool>(&value))
                {
                    const std::uint32_t packed = *typed ? 1u : 0u;
                    copy(&packed, sizeof(packed));
                }
                break;
            case shader_parameter_type::int32:
                if (const auto* typed = std::get_if<std::int32_t>(&value)) copy(typed, sizeof(*typed));
                break;
            case shader_parameter_type::uint32:
                if (const auto* typed = std::get_if<std::uint32_t>(&value)) copy(typed, sizeof(*typed));
                break;
            case shader_parameter_type::float32:
                if (const auto* typed = std::get_if<float>(&value)) copy(typed, sizeof(*typed));
                break;
            case shader_parameter_type::float2:
                if (const auto* typed = std::get_if<math::vector2f>(&value)) copy(typed->data(), sizeof(float) * 2u);
                break;
            case shader_parameter_type::float3:
                if (const auto* typed = std::get_if<math::vector3f>(&value)) copy(typed->data(), sizeof(float) * 3u);
                break;
            case shader_parameter_type::float4:
                if (const auto* typed = std::get_if<math::vector4f>(&value)) copy(typed->data(), sizeof(float) * 4u);
                break;
            case shader_parameter_type::matrix4x4:
                if (const auto* typed = std::get_if<math::matrix4x4f>(&value)) copy(typed->data(), sizeof(float) * 16u);
                break;
            case shader_parameter_type::texture_2d:
            case shader_parameter_type::texture_cube:
            case shader_parameter_type::texture_3d:
            case shader_parameter_type::sampler:
                break;
        }
    };
    for (const auto& override : material.parameters)
    {
        const auto parameter = std::ranges::find(program.parameters, override.id, &shader_parameter_descriptor::id);
        if (parameter != program.parameters.end()) copy_override(*parameter, override.value);
    }

    vmaFlushAllocation(allocator_, buffer.allocation, 0, byte_size);
    vmaUnmapMemory(allocator_, buffer.allocation);
    return true;
}

bool vulkan_render_backend::update_runtime_frame_buffer(gpu_buffer& buffer)
{
    constexpr VkDeviceSize byte_size = sizeof(float) * 4u;
    if (buffer.buffer == VK_NULL_HANDLE &&
        !create_buffer(byte_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, buffer))
        return false;
    const std::array<float, 4> frame{static_cast<float>(last_profile_.frame_index) / 60.0f, 0.0f, 0.0f, 0.0f};
    void* mapped{};
    if (vmaMapMemory(allocator_, buffer.allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, frame.data(), sizeof(frame));
    vmaFlushAllocation(allocator_, buffer.allocation, 0, sizeof(frame));
    vmaUnmapMemory(allocator_, buffer.allocation);
    return true;
}

bool vulkan_render_backend::update_runtime_material_buffers(gpu_material& material)
{
    if (!material.data.runtime_program) return false;
    const auto slot = current_frame_slot();
    if (!material.runtime.parameter_buffers.empty())
    {
        if (slot >= material.runtime.parameter_buffers.size() ||
            !update_runtime_parameter_buffer(material.runtime.parameter_buffers[slot], material.data,
                                             *material.data.runtime_program))
            return false;
    }
    if (!material.runtime.frame_buffers.empty())
    {
        if (slot >= material.runtime.frame_buffers.size() ||
            !update_runtime_frame_buffer(material.runtime.frame_buffers[slot]))
            return false;
    }
    if (material.data.runtime_program->uses_texture_sampling && !update_runtime_texture_descriptors(material, slot))
        return false;
    return true;
}

texture_handle vulkan_render_backend::runtime_texture_handle(const gpu_material& material,
                                                             std::uint32_t slot) const noexcept
{
    texture_handle handle{};
    if (slot < material.data.runtime_textures.size()) handle = material.data.runtime_textures[slot];
    if (!material.data.runtime_program) return handle;
    const auto binding = std::ranges::find(material.data.runtime_program->texture_bindings, slot,
                                           &material_runtime_texture_binding::slot);
    if (binding == material.data.runtime_program->texture_bindings.end() ||
        binding->parameter_id.representation() == 0u)
        return handle;
    const auto override =
        std::ranges::find(material.data.parameters, binding->parameter_id, &material_parameter_override::id);
    if (override == material.data.parameters.end()) return handle;
    if (const auto* resource = std::get_if<resource_handle>(&override->value)) return *resource;
    return handle;
}

VkDescriptorType vulkan_render_backend::runtime_descriptor_type(shader_resource_kind kind) const noexcept
{
    switch (kind)
    {
        case shader_resource_kind::constant_buffer:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case shader_resource_kind::sampled_texture:
            return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case shader_resource_kind::sampler:
            return VK_DESCRIPTOR_TYPE_SAMPLER;
        default:
            return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
}

bool vulkan_render_backend::update_runtime_texture_descriptors(gpu_material& material, std::uint32_t frame_slot)
{
    if (!material.data.runtime_program || frame_slot >= material.runtime.descriptor_sets.size()) return false;
    const auto* pass = runtime_gbuffer_pass(material);
    if (pass == nullptr) return false;

    const auto& resources = pass->compiled.reflection.resources;
    std::vector<std::vector<VkDescriptorImageInfo>> image_infos(resources.size());
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(resources.size());

    VkSampler graph_sampler = white_sampler_;
    for (std::uint32_t texture_slot = 0;
         texture_slot < static_cast<std::uint32_t>(material.data.runtime_program->texture_bindings.size());
         ++texture_slot)
    {
        const auto handle = runtime_texture_handle(material, texture_slot);
        if (!handle.valid()) continue;
        const auto found = textures_.find(resource_key(handle));
        if (found != textures_.end() && found->second.sampler != VK_NULL_HANDLE)
        {
            graph_sampler = found->second.sampler;
            break;
        }
    }

    for (std::size_t resource_index = 0; resource_index < resources.size(); ++resource_index)
    {
        const auto& resource = resources[resource_index];
        if (resource.kind != shader_resource_kind::sampled_texture && resource.kind != shader_resource_kind::sampler)
            continue;

        auto& infos = image_infos[resource_index];
        infos.resize(resource.kind == shader_resource_kind::sampled_texture ? resource.count : 1u);
        if (resource.kind == shader_resource_kind::sampled_texture)
        {
            const auto expected_type = material_texture_resource_type(resource.name);
            if (!expected_type)
                return reject_runtime_material(material, "unsupported reflected Material ABI texture resource '" +
                                                             resource.name + "'");
            for (std::uint32_t dimension_slot = 0; dimension_slot < resource.count; ++dimension_slot)
            {
                const auto binding = std::ranges::find_if(
                    material.data.runtime_program->texture_bindings,
                    [expected_type, dimension_slot](const material_runtime_texture_binding& value)
                    { return value.type == *expected_type && value.dimension_slot == dimension_slot; });
                if (binding == material.data.runtime_program->texture_bindings.end())
                    return reject_runtime_material(material,
                                                   "compiled Material ABI texture binding table is incomplete");

                VkImageView view = white_view_;
                const auto handle = runtime_texture_handle(material, binding->slot);
                if (!handle.valid())
                {
                    if (*expected_type != shader_parameter_type::texture_2d) return false;
                }
                else
                {
                    const auto found = textures_.find(resource_key(handle));
                    if (found == textures_.end() || found->second.view == VK_NULL_HANDLE)
                    {
                        if (*expected_type != shader_parameter_type::texture_2d) return false;
                    }
                    else
                    {
                        if (found->second.data.dimension != material_texture_dimension(*expected_type))
                            return reject_runtime_material(
                                material, std::string(material_texture_type_name(*expected_type)) +
                                              " material binding received a texture with incompatible dimensionality");
                        view = found->second.view;
                    }
                }
                infos[dimension_slot] = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
        }
        else
            infos[0] = {graph_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = material.runtime.descriptor_sets[frame_slot];
        write.dstBinding = resource.binding;
        write.descriptorCount = static_cast<std::uint32_t>(infos.size());
        write.descriptorType = runtime_descriptor_type(resource.kind);
        write.pImageInfo = infos.data();
        writes.push_back(write);
    }
    if (!writes.empty())
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    return true;
}

bool vulkan_render_backend::create_runtime_material_descriptors(gpu_material& material,
                                                                const material_runtime_pass& pass)
{
    const auto& reflection = pass.compiled.reflection;
    std::vector<const shader_resource_descriptor*> resources;
    resources.reserve(reflection.resources.size());
    for (const auto& resource : reflection.resources)
    {
        if (resource.set != 0u)
            return reject_runtime_material(material, "compiled preview resources must currently use descriptor set 0");
        const auto descriptor_type = runtime_descriptor_type(resource.kind);
        if (descriptor_type == VK_DESCRIPTOR_TYPE_MAX_ENUM)
            return reject_runtime_material(material,
                                           "unsupported reflected Material ABI resource '" + resource.name + "'");
        const bool supported_name =
            (resource.kind == shader_resource_kind::constant_buffer &&
             (resource.name == "arcMaterialParameters" || resource.name == "arcFrame")) ||
            (resource.kind == shader_resource_kind::sampled_texture &&
             material_texture_resource_type(resource.name).has_value()) ||
            (resource.kind == shader_resource_kind::sampler && resource.name == "arcMaterialSampler");
        if (!supported_name)
            return reject_runtime_material(material,
                                           "unsupported reflected Material ABI resource '" + resource.name + "'");
        resources.push_back(&resource);
    }
    if (resources.empty()) return true;

    std::ranges::sort(resources, {}, [](const shader_resource_descriptor* resource) { return resource->binding; });
    for (std::size_t index = 1; index < resources.size(); ++index)
        if (resources[index - 1]->binding == resources[index]->binding)
            return reject_runtime_material(material,
                                           "compiled Material ABI reflection contains duplicate descriptor bindings");

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(resources.size());
    for (const auto* resource : resources)
        bindings.push_back({resource->binding, runtime_descriptor_type(resource->kind), resource->count,
                            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &material.runtime.descriptor_set_layout) != VK_SUCCESS)
        return reject_runtime_material(material, "failed to create reflected Material ABI descriptor layout");

    const auto frame_count = frame_resource_count();
    std::array<std::uint32_t, 3> descriptor_counts{};
    for (const auto* resource : resources)
    {
        const auto count = resource->count * frame_count;
        switch (runtime_descriptor_type(resource->kind))
        {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                descriptor_counts[0] += count;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                descriptor_counts[1] += count;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                descriptor_counts[2] += count;
                break;
            default:
                break;
        }
    }
    std::vector<VkDescriptorPoolSize> pool_sizes;
    if (descriptor_counts[0]) pool_sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_counts[0]});
    if (descriptor_counts[1]) pool_sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, descriptor_counts[1]});
    if (descriptor_counts[2]) pool_sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, descriptor_counts[2]});
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = frame_count;
    pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &material.runtime.descriptor_pool) != VK_SUCCESS)
        return reject_runtime_material(material, "failed to create reflected Material ABI descriptor pool");

    material.runtime.descriptor_sets.resize(frame_count);
    std::vector<VkDescriptorSetLayout> layouts(frame_count, material.runtime.descriptor_set_layout);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = material.runtime.descriptor_pool;
    allocate.descriptorSetCount = frame_count;
    allocate.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device_, &allocate, material.runtime.descriptor_sets.data()) != VK_SUCCESS)
        return reject_runtime_material(material, "failed to allocate reflected Material ABI descriptor sets");

    const bool needs_parameters =
        std::ranges::any_of(resources, [](const auto* resource) { return resource->name == "arcMaterialParameters"; });
    const bool needs_frame =
        std::ranges::any_of(resources, [](const auto* resource) { return resource->name == "arcFrame"; });
    if (needs_parameters) material.runtime.parameter_buffers.resize(frame_count);
    if (needs_frame) material.runtime.frame_buffers.resize(frame_count);

    for (std::uint32_t slot = 0; slot < frame_count; ++slot)
    {
        if (needs_parameters && !update_runtime_parameter_buffer(material.runtime.parameter_buffers[slot],
                                                                 material.data, *material.data.runtime_program))
            return reject_runtime_material(material, "failed to allocate Material ABI parameter buffer");
        if (needs_frame && !update_runtime_frame_buffer(material.runtime.frame_buffers[slot]))
            return reject_runtime_material(material, "failed to allocate Material ABI frame buffer");

        std::vector<VkDescriptorBufferInfo> infos;
        std::vector<VkWriteDescriptorSet> writes;
        infos.reserve(resources.size());
        writes.reserve(resources.size());
        for (const auto* resource : resources)
        {
            if (resource->kind != shader_resource_kind::constant_buffer) continue;
            const bool parameters = resource->name == "arcMaterialParameters";
            const auto& buffer =
                parameters ? material.runtime.parameter_buffers[slot] : material.runtime.frame_buffers[slot];
            const auto range = parameters ? static_cast<VkDeviceSize>(std::max<std::size_t>(
                                                material.data.runtime_program->parameter_block_size, 16u))
                                          : static_cast<VkDeviceSize>(sizeof(float) * 4u);
            infos.push_back({buffer.buffer, 0u, range});
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = material.runtime.descriptor_sets[slot];
            write.dstBinding = resource->binding;
            write.descriptorCount = 1u;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &infos.back();
            writes.push_back(write);
        }
        if (!writes.empty())
            vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        if (!update_runtime_texture_descriptors(material, slot))
            return reject_runtime_material(material, "failed to update Material ABI texture descriptors");
    }
    return true;
}

bool vulkan_render_backend::create_runtime_gbuffer_pipeline(gpu_material& material, const material_runtime_pass& pass)
{
    VkShaderModule vert = create_shader_module(builtin::gbuffer_vert_spv, std::size(builtin::gbuffer_vert_spv));
    VkShaderModule frag = create_shader_module(pass.compiled.bytecode);
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
    {
        if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
        return reject_runtime_material(material, "failed to create compiled Material ABI shader module");
    }

    VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0u,
                             sizeof(mesh_push_constants)};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (material.runtime.descriptor_set_layout != VK_NULL_HANDLE)
    {
        layout.setLayoutCount = 1u;
        layout.pSetLayouts = &material.runtime.descriptor_set_layout;
    }
    layout.pushConstantRangeCount = 1u;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &material.runtime.pipeline_layout) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return reject_runtime_material(material, "failed to create compiled Material ABI pipeline layout");
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

    VkVertexInputBindingDescription binding{0u, sizeof(mesh_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array<VkVertexInputAttributeDescription, 5> attributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)},
        VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)},
        VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)},
        VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)},
        VkVertexInputAttributeDescription{4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)}};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1u;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
    std::array<VkPipelineColorBlendAttachmentState, 6> attachments{};
    for (auto& attachment : attachments)
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color_blend.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    color_blend.pAttachments = attachments.data();
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();
    const std::array<VkFormat, 6> color_formats{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_FORMAT_R16G16_SFLOAT,       VK_FORMAT_R32_UINT};
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size());
    rendering.pColorAttachmentFormats = color_formats.data();
    rendering.depthAttachmentFormat = depth_format_;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = material.runtime.pipeline_layout;
    const auto result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                  &material.runtime.gbuffer_pipeline);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (result != VK_SUCCESS)
        return reject_runtime_material(material, "failed to create compiled Material ABI G-buffer pipeline: " +
                                                     describe_vk_result(result));
    return true;
}

bool vulkan_render_backend::ensure_runtime_gbuffer_pipeline(gpu_material& material)
{
    const auto* program = material.data.runtime_program.get();
    if (program == nullptr) return false;
    if (material.runtime.failed && material.runtime.generation == program->generation) return false;
    if (material.runtime.gbuffer_pipeline != VK_NULL_HANDLE && material.runtime.generation == program->generation &&
        (material.runtime.descriptor_set_layout == VK_NULL_HANDLE ||
         material.runtime.descriptor_sets.size() == frame_resource_count()))
        return true;

    if (material.runtime.gbuffer_pipeline != VK_NULL_HANDLE || material.runtime.pipeline_layout != VK_NULL_HANDLE ||
        material.runtime.descriptor_pool != VK_NULL_HANDLE ||
        material.runtime.descriptor_set_layout != VK_NULL_HANDLE || !material.runtime.parameter_buffers.empty() ||
        !material.runtime.frame_buffers.empty())
    {
        wait_for_in_flight_frames();
        destroy_material_runtime(material.runtime);
    }
    material.runtime.generation = program->generation;

    if (!material_runtime_program_compatible(*program))
        return reject_runtime_material(material, "unsupported compiled Material ABI contract version (got pass " +
                                                     std::to_string(program->contract_version) + " / ABI " +
                                                     std::to_string(program->material_abi) + ", expected pass " +
                                                     std::to_string(material_pass_contract_version) + " / ABI " +
                                                     std::to_string(material_abi_version) + ')');
    if (material.data.alpha_mode != material_alpha_mode::opaque)
        return reject_runtime_material(material, "compiled preview execution currently requires an opaque material");
    const auto* pass = runtime_gbuffer_pass(material);
    if (pass == nullptr || pass->compiled.bytecode.empty())
        return reject_runtime_material(material, "compiled material does not provide an executable G-buffer pass");
    if (!create_runtime_material_descriptors(material, *pass)) return false;
    if (!create_runtime_gbuffer_pipeline(material, *pass)) return false;
    arc::diagnostics::debug("render.vulkan",
                            "Using compiled Material ABI G-buffer pass for '" + material.data.name + "'");
    return true;
}

material_uniform_data
vulkan_render_backend::build_material_parameters(const material_descriptor* material) const noexcept
{
    material_uniform_data parameters{};
    if (material == nullptr || material->domain == material_domain::terrain) return parameters;

    parameters.emissive_factor[0] = material->emissive_factor[0];
    parameters.emissive_factor[1] = material->emissive_factor[1];
    parameters.emissive_factor[2] = material->emissive_factor[2];
    parameters.emissive_factor[3] = material->emissive_luminance_nits > 0.0f
                                        ? material->emissive_luminance_nits / 100.0f
                                        : material->emissive_strength;
    parameters.material_lobes[0] = material->clear_coat_factor;
    parameters.material_lobes[1] = material->clear_coat_roughness;
    parameters.material_lobes[2] = material->anisotropy_factor;
    parameters.material_lobes[3] = material->transmission_factor;
    parameters.volume_params[0] = static_cast<float>(material->shading_model);
    parameters.volume_params[1] = material->index_of_refraction;
    parameters.volume_params[2] = material->thickness_factor;
    parameters.volume_params[3] = material->attenuation_distance;
    parameters.subsurface_color_factor[0] = material->subsurface_color[0];
    parameters.subsurface_color_factor[1] = material->subsurface_color[1];
    parameters.subsurface_color_factor[2] = material->subsurface_color[2];
    parameters.subsurface_color_factor[3] = material->subsurface_factor;
    parameters.attenuation_color[0] = material->attenuation_color[0];
    parameters.attenuation_color[1] = material->attenuation_color[1];
    parameters.attenuation_color[2] = material->attenuation_color[2];
    parameters.attenuation_color[3] = (texture_ready(material->clear_coat_texture) ? 1.0f : 0.0f) +
                                      (texture_ready(material->clear_coat_roughness_texture) ? 2.0f : 0.0f) +
                                      (texture_ready(material->clear_coat_normal_texture) ? 4.0f : 0.0f) +
                                      (texture_ready(material->anisotropy_texture) ? 8.0f : 0.0f) +
                                      (texture_ready(material->subsurface_texture) ? 16.0f : 0.0f) +
                                      (texture_ready(material->thickness_texture) ? 32.0f : 0.0f) +
                                      (texture_ready(material->transmission_texture) ? 64.0f : 0.0f);
    return parameters;
}

} // namespace arc::render::vulkan::backend_detail
