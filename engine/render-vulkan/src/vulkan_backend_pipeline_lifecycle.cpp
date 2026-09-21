#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"
#include "vulkan_sky_constants.h"

namespace arc::render::vulkan::backend_detail
{
void vulkan_render_backend::destroy_mesh_pipeline() noexcept
{
    if (water_surface_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, water_surface_pipeline_, nullptr);
        water_surface_pipeline_ = VK_NULL_HANDLE;
    }
    if (water_surface_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, water_surface_pipeline_layout_, nullptr);
        water_surface_pipeline_layout_ = VK_NULL_HANDLE;
    }
    const auto destroy_debug_pipeline = [&](VkPipeline& pipeline)
    {
        if (pipeline == VK_NULL_HANDLE) return;
        vkDestroyPipeline(device_, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    };
    destroy_debug_pipeline(debug_overlay_line_pipeline_);
    destroy_debug_pipeline(debug_overlay_triangle_pipeline_);
    destroy_debug_pipeline(debug_overlay_output_line_pipeline_);
    destroy_debug_pipeline(debug_overlay_output_triangle_pipeline_);
    if (debug_overlay_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, debug_overlay_pipeline_layout_, nullptr);
        debug_overlay_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (deferred_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, deferred_pipeline_, nullptr);
        deferred_pipeline_ = VK_NULL_HANDLE;
    }
    if (deferred_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, deferred_pipeline_layout_, nullptr);
        deferred_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (output_transform_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, output_transform_pipeline_, nullptr);
        output_transform_pipeline_ = VK_NULL_HANDLE;
    }
    if (output_transform_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, output_transform_pipeline_layout_, nullptr);
        output_transform_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (luminance_histogram_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, luminance_histogram_pipeline_, nullptr);
        luminance_histogram_pipeline_ = VK_NULL_HANDLE;
    }
    if (luminance_histogram_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, luminance_histogram_pipeline_layout_, nullptr);
        luminance_histogram_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (exposure_resolve_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, exposure_resolve_pipeline_, nullptr);
        exposure_resolve_pipeline_ = VK_NULL_HANDLE;
    }
    if (exposure_resolve_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, exposure_resolve_pipeline_layout_, nullptr);
        exposure_resolve_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (output_transform_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, output_transform_descriptor_pool_, nullptr);
        output_transform_descriptor_pool_ = VK_NULL_HANDLE;
        output_transform_descriptor_set_ = VK_NULL_HANDLE;
    }
    if (output_transform_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, output_transform_descriptor_set_layout_, nullptr);
        output_transform_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (gbuffer_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, gbuffer_pipeline_, nullptr);
        gbuffer_pipeline_ = VK_NULL_HANDLE;
    }
    if (terrain_surface_gbuffer_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, terrain_surface_gbuffer_pipeline_, nullptr);
        terrain_surface_gbuffer_pipeline_ = VK_NULL_HANDLE;
    }
    if (gbuffer_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, gbuffer_descriptor_pool_, nullptr);
        gbuffer_descriptor_pool_ = VK_NULL_HANDLE;
        gbuffer_descriptor_set_ = VK_NULL_HANDLE;
    }
    if (gbuffer_sampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, gbuffer_sampler_, nullptr);
        gbuffer_sampler_ = VK_NULL_HANDLE;
    }
    if (gbuffer_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, gbuffer_descriptor_set_layout_, nullptr);
        gbuffer_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (shadow_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, shadow_pipeline_, nullptr);
        shadow_pipeline_ = VK_NULL_HANDLE;
    }
    if (shadow_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, shadow_pipeline_layout_, nullptr);
        shadow_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, mesh_wire_pipeline_, nullptr);
        mesh_wire_pipeline_ = VK_NULL_HANDLE;
    }
    if (selection_mask_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, selection_mask_pipeline_, nullptr);
        selection_mask_pipeline_ = VK_NULL_HANDLE;
    }
    if (mesh_transparent_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, mesh_transparent_pipeline_, nullptr);
        mesh_transparent_pipeline_ = VK_NULL_HANDLE;
    }
    if (terrain_surface_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, terrain_surface_pipeline_, nullptr);
        terrain_surface_pipeline_ = VK_NULL_HANDLE;
    }
    if (mesh_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, mesh_pipeline_, nullptr);
        mesh_pipeline_ = VK_NULL_HANDLE;
    }
    if (mesh_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, mesh_pipeline_layout_, nullptr);
        mesh_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (terrain_surface_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, terrain_surface_pipeline_layout_, nullptr);
        terrain_surface_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (material_attribute_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, material_attribute_descriptor_pool_, nullptr);
        material_attribute_descriptor_pool_ = VK_NULL_HANDLE;
        material_attribute_descriptor_sets_.clear();
    }
    if (material_attribute_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, material_attribute_descriptor_set_layout_, nullptr);
        material_attribute_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (sky_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, sky_pipeline_, nullptr);
        sky_pipeline_ = VK_NULL_HANDLE;
    }
    if (sky_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, sky_pipeline_layout_, nullptr);
        sky_pipeline_layout_ = VK_NULL_HANDLE;
    }
}

void vulkan_render_backend::destroy_white_texture() noexcept
{
    for (auto& parameters : white_material_parameter_buffers_)
        destroy_buffer(parameters);
    white_material_parameter_buffers_.clear();
    if (white_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, white_descriptor_pool_, nullptr);
        white_descriptor_pool_ = VK_NULL_HANDLE;
        white_descriptor_sets_.clear();
        sky_descriptor_sets_.clear();
        for (auto& [_, material] : materials_)
            material.descriptor_sets.clear();
    }
    if (white_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, white_descriptor_set_layout_, nullptr);
        white_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (white_sampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, white_sampler_, nullptr);
        white_sampler_ = VK_NULL_HANDLE;
    }
    if (white_view_ != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device_, white_view_, nullptr);
        white_view_ = VK_NULL_HANDLE;
    }
    if (white_image_ != VK_NULL_HANDLE)
    {
        vmaDestroyImage(allocator_, white_image_, white_allocation_);
        white_image_ = VK_NULL_HANDLE;
        white_allocation_ = VK_NULL_HANDLE;
    }
}

VkShaderModule vulkan_render_backend::create_shader_module(const std::uint32_t* code, std::size_t word_count)
{
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = word_count * sizeof(std::uint32_t);
    info.pCode = code;

    VkShaderModule module{};
    if (vkCreateShaderModule(device_, &info, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    return module;
}

VkShaderModule vulkan_render_backend::create_shader_module(const std::vector<std::uint8_t>& bytecode)
{
    if (bytecode.empty() || bytecode.size() % sizeof(std::uint32_t) != 0) return VK_NULL_HANDLE;
    std::vector<std::uint32_t> words(bytecode.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytecode.data(), bytecode.size());
    return create_shader_module(words.data(), words.size());
}

} // namespace arc::render::vulkan::backend_detail
