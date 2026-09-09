#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"
#include "vulkan_pick_utils.h"
#include "vulkan_sky_constants.h"

namespace arc::render::vulkan::backend_detail
{
void vulkan_render_backend::destroy_graph_image(graph_image& image) noexcept
{
    for (const auto view : image.mip_views)
        if (view != VK_NULL_HANDLE) vkDestroyImageView(device_, view, nullptr);
    image.mip_views.clear();
    if (image.view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device_, image.view, nullptr);
        image.view = VK_NULL_HANDLE;
    }
    if (image.image != VK_NULL_HANDLE)
    {
        vmaDestroyImage(allocator_, image.image, image.allocation);
        image.image = VK_NULL_HANDLE;
        image.allocation = VK_NULL_HANDLE;
    }
    image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.width = 0;
    image.height = 0;
    image.mip_levels = 1;
}

bool vulkan_render_backend::ensure_graph_image(graph_image& target, std::uint32_t width, std::uint32_t height,
                                               VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                               std::uint32_t mip_levels)
{
    if (target.image != VK_NULL_HANDLE && target.format == format && target.aspect == aspect && target.width == width &&
        target.height == height && target.mip_levels == mip_levels)
        return true;

    destroy_graph_image(target);

    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = format;
    image.extent = {width, height, 1};
    image.mipLevels = mip_levels;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = usage;

    VmaAllocationCreateInfo allocation{};
    allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(allocator_, &image, &allocation, &target.image, &target.allocation, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = target.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange.aspectMask = aspect;
    view.subresourceRange.levelCount = mip_levels;
    view.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_, &view, nullptr, &target.view) != VK_SUCCESS)
    {
        destroy_graph_image(target);
        return false;
    }

    if (mip_levels > 1)
    {
        target.mip_views.resize(mip_levels);
        view.subresourceRange.levelCount = 1;
        for (std::uint32_t mip = 0; mip < mip_levels; ++mip)
        {
            view.subresourceRange.baseMipLevel = mip;
            if (vkCreateImageView(device_, &view, nullptr, &target.mip_views[mip]) != VK_SUCCESS)
            {
                destroy_graph_image(target);
                return false;
            }
        }
    }

    target.format = format;
    target.aspect = aspect;
    target.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    target.width = width;
    target.height = height;
    target.mip_levels = mip_levels;
    return true;
}

void vulkan_render_backend::transition_graph_image(VkCommandBuffer command_buffer, graph_image& image,
                                                   VkImageLayout new_layout)
{
    if (image.image == VK_NULL_HANDLE || image.layout == new_layout) return;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = image.layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = image.aspect;
    barrier.subresourceRange.levelCount = image.mip_levels;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (image.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else if (image.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (image.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        // Graph images are consumed by both raster passes and compute
        // post-processing (notably the luminance histogram). Restricting
        // this dependency to fragment shaders lets the next frame start
        // overwriting scene_color while the histogram still reads it.
        src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (image.layout == VK_IMAGE_LAYOUT_GENERAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    if (new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_GENERAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    image.layout = new_layout;
}

bool vulkan_render_backend::ensure_deferred_targets(std::uint32_t width, std::uint32_t height)
{
    const VkImageUsageFlags sampled_color_usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkImageUsageFlags gbuffer_usage =
        sampled_color_usage |
        (capabilities_.virtual_geometry_compute ? static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_STORAGE_BIT)
                                                : VkImageUsageFlags{0});
    const std::array previous_views{gbuffer_albedo_.view,   gbuffer_normal_.view, gbuffer_material_.view,
                                    gbuffer_emissive_.view, gbuffer_motion_.view, gbuffer_object_id_.view};
    const bool ok = ensure_graph_image(scene_color_, width, height, scene_color_format_, sampled_color_usage,
                                       VK_IMAGE_ASPECT_COLOR_BIT) &&
                    ensure_graph_image(gbuffer_albedo_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, gbuffer_usage,
                                       VK_IMAGE_ASPECT_COLOR_BIT) &&
                    ensure_graph_image(gbuffer_normal_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, gbuffer_usage,
                                       VK_IMAGE_ASPECT_COLOR_BIT) &&
                    ensure_graph_image(gbuffer_material_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, gbuffer_usage,
                                       VK_IMAGE_ASPECT_COLOR_BIT) &&
                    ensure_graph_image(gbuffer_emissive_, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, gbuffer_usage,
                                       VK_IMAGE_ASPECT_COLOR_BIT) &&
                    ensure_graph_image(gbuffer_motion_, width, height, VK_FORMAT_R16G16_SFLOAT, gbuffer_usage,
                                       VK_IMAGE_ASPECT_COLOR_BIT) &&
                    ensure_graph_image(gbuffer_object_id_, width, height, VK_FORMAT_R32_UINT, gbuffer_usage,
                                       VK_IMAGE_ASPECT_COLOR_BIT) &&
                    ensure_graph_image(selection_mask_, width, height, VK_FORMAT_R8_UNORM, sampled_color_usage,
                                       VK_IMAGE_ASPECT_COLOR_BIT);
    if (ok)
    {
        const std::array current_views{gbuffer_albedo_.view,   gbuffer_normal_.view, gbuffer_material_.view,
                                       gbuffer_emissive_.view, gbuffer_motion_.view, gbuffer_object_id_.view};
        if (current_views != previous_views) virtual_geometry_material_descriptors_dirty_ = true;
        update_gbuffer_descriptor_set();
    }
    return ok;
}

void vulkan_render_backend::destroy_hzb_resources() noexcept
{
    for (auto& image : hzb_history_)
        destroy_graph_image(image);
    hzb_descriptor_sets_.clear();
    if (hzb_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, hzb_pipeline_, nullptr);
    if (hzb_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, hzb_pipeline_layout_, nullptr);
    if (hzb_descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, hzb_descriptor_pool_, nullptr);
    if (hzb_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, hzb_descriptor_set_layout_, nullptr);
    if (hzb_sampler_ != VK_NULL_HANDLE) vkDestroySampler(device_, hzb_sampler_, nullptr);
    hzb_pipeline_ = VK_NULL_HANDLE;
    hzb_pipeline_layout_ = VK_NULL_HANDLE;
    hzb_descriptor_pool_ = VK_NULL_HANDLE;
    hzb_descriptor_set_layout_ = VK_NULL_HANDLE;
    hzb_sampler_ = VK_NULL_HANDLE;
    hzb_mip_count_ = 0;
    hzb_history_valid_ = false;
}

bool vulkan_render_backend::ensure_hzb_resources(std::uint32_t width, std::uint32_t height)
{
    if (!capabilities_.hzb_occlusion || viewport_depth_view_ == VK_NULL_HANDLE) return false;
    const auto mip_count = hzb_mip_count(width, height);
    const bool extent_changed =
        hzb_mip_count_ != mip_count || hzb_history_[0].width != width || hzb_history_[0].height != height;

    if (hzb_sampler_ == VK_NULL_HANDLE)
    {
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = static_cast<float>(mip_count);
        if (vkCreateSampler(device_, &sampler, nullptr, &hzb_sampler_) != VK_SUCCESS) return false;
    }

    if (hzb_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        const std::array bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                         nullptr},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &hzb_descriptor_set_layout_) != VK_SUCCESS)
            return false;
    }

    if (hzb_pipeline_ == VK_NULL_HANDLE)
    {
        const auto shader =
            create_shader_module(builtin::depth_pyramid_comp_spv, std::size(builtin::depth_pyramid_comp_spv));
        if (shader == VK_NULL_HANDLE) return false;
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hzb_reduce_push_constants)};
        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &hzb_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &hzb_pipeline_layout_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, shader, nullptr);
            return false;
        }
        VkComputePipelineCreateInfo pipeline{};
        pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = shader;
        pipeline.stage.pName = "main";
        pipeline.layout = hzb_pipeline_layout_;
        const auto result =
            vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &hzb_pipeline_);
        vkDestroyShaderModule(device_, shader, nullptr);
        if (result != VK_SUCCESS) return false;
    }

    if (!extent_changed && hzb_descriptor_pool_ != VK_NULL_HANDLE) return true;
    for (auto& image : hzb_history_)
        destroy_graph_image(image);
    if (hzb_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, hzb_descriptor_pool_, nullptr);
        hzb_descriptor_pool_ = VK_NULL_HANDLE;
    }
    hzb_descriptor_sets_.clear();
    const auto usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    for (auto& image : hzb_history_)
        if (!ensure_graph_image(image, width, height, VK_FORMAT_R32G32_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT,
                                mip_count))
            return false;

    const std::uint32_t set_count = mip_count * static_cast<std::uint32_t>(hzb_history_.size());
    const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, set_count},
                                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, set_count}};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = set_count;
    pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &hzb_descriptor_pool_) != VK_SUCCESS) return false;
    hzb_descriptor_sets_.resize(set_count);
    std::vector<VkDescriptorSetLayout> layouts(set_count, hzb_descriptor_set_layout_);
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = hzb_descriptor_pool_;
    allocate.descriptorSetCount = set_count;
    allocate.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device_, &allocate, hzb_descriptor_sets_.data()) != VK_SUCCESS) return false;

    for (std::uint32_t generation = 0; generation < hzb_history_.size(); ++generation)
    {
        auto& image = hzb_history_[generation];
        for (std::uint32_t mip = 0; mip < mip_count; ++mip)
        {
            const auto index = generation * mip_count + mip;
            const VkDescriptorImageInfo source{hzb_sampler_, mip == 0 ? viewport_depth_view_ : image.view,
                                               mip == 0 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                        : VK_IMAGE_LAYOUT_GENERAL};
            const VkDescriptorImageInfo destination{VK_NULL_HANDLE, image.mip_views[mip], VK_IMAGE_LAYOUT_GENERAL};
            std::array writes{VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
                              VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            writes[0].dstSet = hzb_descriptor_sets_[index];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].pImageInfo = &source;
            writes[1].dstSet = hzb_descriptor_sets_[index];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].pImageInfo = &destination;
            vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }
    hzb_mip_count_ = mip_count;
    hzb_history_valid_ = false;
    gpu_visibility_descriptors_dirty_ = true;
    return true;
}

void vulkan_render_backend::dispatch_hzb(VkCommandBuffer command_buffer)
{
    if (!ensure_hzb_resources(viewport_width_, viewport_height_))
    {
        last_profile_.gpu_scene.fallback_reason = "HZB resources are unavailable; occlusion is disabled";
        return;
    }
    transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % hzb_history_.size());
    auto& image = hzb_history_[generation];
    transition_graph_image(command_buffer, image, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, hzb_pipeline_);
    for (std::uint32_t mip = 0; mip < hzb_mip_count_; ++mip)
    {
        const std::uint32_t destination_width = std::max(1u, viewport_width_ >> mip);
        const std::uint32_t destination_height = std::max(1u, viewport_height_ >> mip);
        const std::uint32_t source_width = mip == 0 ? viewport_width_ : std::max(1u, viewport_width_ >> (mip - 1u));
        const std::uint32_t source_height = mip == 0 ? viewport_height_ : std::max(1u, viewport_height_ >> (mip - 1u));
        const hzb_reduce_push_constants constants{
            static_cast<std::int32_t>(destination_width), static_cast<std::int32_t>(destination_height),
            static_cast<std::int32_t>(source_width), static_cast<std::int32_t>(source_height),
            mip == 0 ? -1 : static_cast<std::int32_t>(mip - 1u)};
        const auto set = hzb_descriptor_sets_[generation * hzb_mip_count_ + mip];
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, hzb_pipeline_layout_, 0, 1, &set, 0,
                                nullptr);
        vkCmdPushConstants(command_buffer, hzb_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants),
                           &constants);
        vkCmdDispatch(command_buffer, (destination_width + 7u) / 8u, (destination_height + 7u) / 8u, 1u);
        if (mip + 1u < hzb_mip_count_)
        {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = mip;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }
    }
    hzb_history_valid_ = !frame_camera_.camera_cut;
    last_profile_.gpu_scene.history_valid = hzb_history_valid_;
    last_profile_.temporal.hzb_mip_count = hzb_mip_count_;
}

void vulkan_render_backend::destroy_temporal_resources() noexcept
{
    for (auto& image : temporal_dilated_motion_)
        destroy_graph_image(image);
    for (auto& image : temporal_reactive_)
        destroy_graph_image(image);
    for (auto& image : temporal_disocclusion_)
        destroy_graph_image(image);
    for (auto& image : temporal_color_history_)
        destroy_graph_image(image);
    for (auto& image : temporal_depth_history_)
        destroy_graph_image(image);
    for (auto& image : temporal_moments_history_)
        destroy_graph_image(image);
    for (auto& image : temporal_confidence_history_)
        destroy_graph_image(image);
    for (auto& image : temporal_sharpened_)
        destroy_graph_image(image);
    if (temporal_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, temporal_descriptor_pool_, nullptr);
    if (temporal_mask_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, temporal_mask_pipeline_, nullptr);
    if (temporal_velocity_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, temporal_velocity_pipeline_, nullptr);
    if (temporal_resolve_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, temporal_resolve_pipeline_, nullptr);
    if (temporal_sharpen_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, temporal_sharpen_pipeline_, nullptr);
    if (temporal_mask_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, temporal_mask_pipeline_layout_, nullptr);
    if (temporal_velocity_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, temporal_velocity_pipeline_layout_, nullptr);
    if (temporal_resolve_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, temporal_resolve_pipeline_layout_, nullptr);
    if (temporal_sharpen_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, temporal_sharpen_pipeline_layout_, nullptr);
    if (temporal_mask_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, temporal_mask_descriptor_layout_, nullptr);
    if (temporal_velocity_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, temporal_velocity_descriptor_layout_, nullptr);
    if (temporal_resolve_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, temporal_resolve_descriptor_layout_, nullptr);
    if (temporal_sharpen_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, temporal_sharpen_descriptor_layout_, nullptr);
    temporal_descriptor_pool_ = VK_NULL_HANDLE;
    temporal_mask_pipeline_ = VK_NULL_HANDLE;
    temporal_velocity_pipeline_ = VK_NULL_HANDLE;
    temporal_resolve_pipeline_ = VK_NULL_HANDLE;
    temporal_sharpen_pipeline_ = VK_NULL_HANDLE;
    temporal_mask_pipeline_layout_ = VK_NULL_HANDLE;
    temporal_velocity_pipeline_layout_ = VK_NULL_HANDLE;
    temporal_resolve_pipeline_layout_ = VK_NULL_HANDLE;
    temporal_sharpen_pipeline_layout_ = VK_NULL_HANDLE;
    temporal_mask_descriptor_layout_ = VK_NULL_HANDLE;
    temporal_velocity_descriptor_layout_ = VK_NULL_HANDLE;
    temporal_resolve_descriptor_layout_ = VK_NULL_HANDLE;
    temporal_sharpen_descriptor_layout_ = VK_NULL_HANDLE;
    temporal_mask_sets_ = {};
    temporal_velocity_sets_ = {};
    temporal_resolve_sets_ = {};
    temporal_sharpen_sets_ = {};
    temporal_input_width_ = temporal_input_height_ = temporal_output_width_ = temporal_output_height_ = 0;
    temporal_history_valid_ = false;
    temporal_resources_initialized_ = false;
    temporal_output_view_ = VK_NULL_HANDLE;
}

bool vulkan_render_backend::create_temporal_pipeline(const std::uint32_t* code, std::size_t code_words,
                                                     VkDescriptorSetLayout set_layout, std::uint32_t push_size,
                                                     VkPipelineLayout& pipeline_layout, VkPipeline& pipeline)
{
    const auto shader = create_shader_module(code, code_words);
    if (shader == VK_NULL_HANDLE) return false;
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &set_layout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &pipeline_layout) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, shader, nullptr);
        return false;
    }
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = shader;
    info.stage.pName = "main";
    info.layout = pipeline_layout;
    const auto result = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(device_, shader, nullptr);
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_temporal_pipelines()
{
    if (temporal_resolve_pipeline_ != VK_NULL_HANDLE) return true;
    const auto make_layout =
        [&](std::uint32_t sampled_count, std::uint32_t storage_count, VkDescriptorSetLayout& result)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(sampled_count + storage_count);
        for (std::uint32_t binding = 0; binding < bindings.size(); ++binding)
        {
            bindings[binding].binding = binding;
            bindings[binding].descriptorType =
                binding < sampled_count ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[binding].descriptorCount = 1;
            bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        return vkCreateDescriptorSetLayout(device_, &info, nullptr, &result) == VK_SUCCESS;
    };
    if (!make_layout(2, 1, temporal_velocity_descriptor_layout_) ||
        !make_layout(4, 2, temporal_mask_descriptor_layout_) ||
        !make_layout(8, 4, temporal_resolve_descriptor_layout_) ||
        !make_layout(1, 1, temporal_sharpen_descriptor_layout_))
        return false;
    return create_temporal_pipeline(builtin::velocity_dilation_comp_spv, std::size(builtin::velocity_dilation_comp_spv),
                                    temporal_velocity_descriptor_layout_, sizeof(velocity_dilation_push_constants),
                                    temporal_velocity_pipeline_layout_, temporal_velocity_pipeline_) &&
           create_temporal_pipeline(builtin::temporal_masks_comp_spv, std::size(builtin::temporal_masks_comp_spv),
                                    temporal_mask_descriptor_layout_, sizeof(temporal_mask_push_constants),
                                    temporal_mask_pipeline_layout_, temporal_mask_pipeline_) &&
           create_temporal_pipeline(builtin::temporal_resolve_comp_spv, std::size(builtin::temporal_resolve_comp_spv),
                                    temporal_resolve_descriptor_layout_, sizeof(temporal_resolve_push_constants),
                                    temporal_resolve_pipeline_layout_, temporal_resolve_pipeline_) &&
           create_temporal_pipeline(builtin::spatial_sharpen_comp_spv, std::size(builtin::spatial_sharpen_comp_spv),
                                    temporal_sharpen_descriptor_layout_, sizeof(sharpen_push_constants),
                                    temporal_sharpen_pipeline_layout_, temporal_sharpen_pipeline_);
}

bool vulkan_render_backend::ensure_temporal_resources(std::uint32_t input_width, std::uint32_t input_height,
                                                      std::uint32_t output_width, std::uint32_t output_height)
{
    if (!capabilities_.temporal_resolve || !ensure_temporal_pipelines()) return false;
    if (temporal_descriptor_pool_ != VK_NULL_HANDLE && temporal_input_width_ == input_width &&
        temporal_input_height_ == input_height && temporal_output_width_ == output_width &&
        temporal_output_height_ == output_height)
        return true;

    for (auto& image : temporal_dilated_motion_)
        destroy_graph_image(image);
    for (auto& image : temporal_reactive_)
        destroy_graph_image(image);
    for (auto& image : temporal_disocclusion_)
        destroy_graph_image(image);
    for (auto& image : temporal_color_history_)
        destroy_graph_image(image);
    for (auto& image : temporal_depth_history_)
        destroy_graph_image(image);
    for (auto& image : temporal_moments_history_)
        destroy_graph_image(image);
    for (auto& image : temporal_confidence_history_)
        destroy_graph_image(image);
    for (auto& image : temporal_sharpened_)
        destroy_graph_image(image);
    if (temporal_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, temporal_descriptor_pool_, nullptr);
    temporal_descriptor_pool_ = VK_NULL_HANDLE;
    const auto usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    for (std::uint32_t generation = 0; generation < 2; ++generation)
    {
        if (!ensure_graph_image(temporal_dilated_motion_[generation], input_width, input_height,
                                VK_FORMAT_R16G16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
            !ensure_graph_image(temporal_reactive_[generation], input_width, input_height, VK_FORMAT_R8_UNORM, usage,
                                VK_IMAGE_ASPECT_COLOR_BIT) ||
            !ensure_graph_image(temporal_disocclusion_[generation], input_width, input_height, VK_FORMAT_R8_UNORM,
                                usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
            !ensure_graph_image(temporal_color_history_[generation], output_width, output_height,
                                VK_FORMAT_R16G16B16A16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
            !ensure_graph_image(temporal_depth_history_[generation], output_width, output_height, VK_FORMAT_R32_SFLOAT,
                                usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
            !ensure_graph_image(temporal_moments_history_[generation], output_width, output_height,
                                VK_FORMAT_R16G16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
            !ensure_graph_image(temporal_confidence_history_[generation], output_width, output_height,
                                VK_FORMAT_R8_UNORM, usage, VK_IMAGE_ASPECT_COLOR_BIT) ||
            !ensure_graph_image(temporal_sharpened_[generation], output_width, output_height,
                                VK_FORMAT_R16G16B16A16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
    }
    const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 30},
                                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16}};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 8;
    pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &temporal_descriptor_pool_) != VK_SUCCESS) return false;
    const std::array layouts{temporal_velocity_descriptor_layout_, temporal_mask_descriptor_layout_,
                             temporal_resolve_descriptor_layout_,  temporal_sharpen_descriptor_layout_,
                             temporal_velocity_descriptor_layout_, temporal_mask_descriptor_layout_,
                             temporal_resolve_descriptor_layout_,  temporal_sharpen_descriptor_layout_};
    std::array<VkDescriptorSet, 8> sets{};
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = temporal_descriptor_pool_;
    allocate.descriptorSetCount = static_cast<std::uint32_t>(sets.size());
    allocate.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device_, &allocate, sets.data()) != VK_SUCCESS) return false;
    for (std::uint32_t generation = 0; generation < 2; ++generation)
    {
        temporal_velocity_sets_[generation] = sets[generation * 4];
        temporal_mask_sets_[generation] = sets[generation * 4 + 1];
        temporal_resolve_sets_[generation] = sets[generation * 4 + 2];
        temporal_sharpen_sets_[generation] = sets[generation * 4 + 3];
    }
    temporal_input_width_ = input_width;
    temporal_input_height_ = input_height;
    temporal_output_width_ = output_width;
    temporal_output_height_ = output_height;
    temporal_history_valid_ = false;
    temporal_resources_initialized_ = false;
    return true;
}

void vulkan_render_backend::update_temporal_descriptors(std::uint32_t generation)
{
    const auto previous = (generation + 1u) % 2u;
    const auto sampled = [&](VkImageView view, VkImageLayout layout)
    { return VkDescriptorImageInfo{viewport_sampler_, view, layout}; };
    const auto storage = [](VkImageView view)
    { return VkDescriptorImageInfo{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL}; };

    const std::array velocity_images{sampled(gbuffer_motion_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                                     sampled(viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                                     storage(temporal_dilated_motion_[generation].view)};
    const std::array mask_images{
        sampled(scene_color_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        sampled(viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        sampled(temporal_depth_history_[previous].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        sampled(temporal_dilated_motion_[generation].view, VK_IMAGE_LAYOUT_GENERAL),
        storage(temporal_reactive_[generation].view),
        storage(temporal_disocclusion_[generation].view)};
    const std::array resolve_images{
        sampled(scene_color_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        sampled(temporal_color_history_[previous].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        sampled(temporal_dilated_motion_[generation].view, VK_IMAGE_LAYOUT_GENERAL),
        sampled(temporal_reactive_[generation].view, VK_IMAGE_LAYOUT_GENERAL),
        sampled(temporal_disocclusion_[generation].view, VK_IMAGE_LAYOUT_GENERAL),
        sampled(viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        sampled(temporal_moments_history_[previous].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        sampled(temporal_confidence_history_[previous].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        storage(temporal_color_history_[generation].view),
        storage(temporal_depth_history_[generation].view),
        storage(temporal_moments_history_[generation].view),
        storage(temporal_confidence_history_[generation].view)};
    const std::array sharpen_images{
        sampled(temporal_color_history_[generation].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        storage(temporal_sharpened_[generation].view)};
    const auto write_images = [&](VkDescriptorSet set, const auto& images, std::uint32_t sampled_count)
    {
        std::vector<VkWriteDescriptorSet> writes(images.size());
        for (std::uint32_t binding = 0; binding < images.size(); ++binding)
        {
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = set;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType =
                binding < sampled_count ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[binding].pImageInfo = &images[binding];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    };
    write_images(temporal_velocity_sets_[generation], velocity_images, 2);
    write_images(temporal_mask_sets_[generation], mask_images, 4);
    write_images(temporal_resolve_sets_[generation], resolve_images, 8);
    write_images(temporal_sharpen_sets_[generation], sharpen_images, 1);
}

void vulkan_render_backend::prepare_temporal_images(VkCommandBuffer command_buffer, std::uint32_t generation)
{
    const auto previous = (generation + 1u) % 2u;
    transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_depth(command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, temporal_dilated_motion_[generation], VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, temporal_color_history_[previous], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, temporal_depth_history_[previous], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, temporal_moments_history_[previous],
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, temporal_confidence_history_[previous],
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, temporal_reactive_[generation], VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, temporal_disocclusion_[generation], VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, temporal_color_history_[generation], VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, temporal_depth_history_[generation], VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, temporal_moments_history_[generation], VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, temporal_confidence_history_[generation], VK_IMAGE_LAYOUT_GENERAL);
    transition_graph_image(command_buffer, temporal_sharpened_[generation], VK_IMAGE_LAYOUT_GENERAL);
}

void vulkan_render_backend::dispatch_velocity_dilation(VkCommandBuffer command_buffer)
{
    if (!ensure_temporal_resources(viewport_width_, viewport_height_, output_viewport_width_, output_viewport_height_))
        return;
    const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % 2u);
    prepare_temporal_images(command_buffer, generation);
    update_temporal_descriptors(generation);
    const velocity_dilation_push_constants constants{static_cast<std::int32_t>(viewport_width_),
                                                     static_cast<std::int32_t>(viewport_height_)};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_velocity_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_velocity_pipeline_layout_, 0, 1,
                            &temporal_velocity_sets_[generation], 0, nullptr);
    vkCmdPushConstants(command_buffer, temporal_velocity_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(constants), &constants);
    vkCmdDispatch(command_buffer, (viewport_width_ + 7u) / 8u, (viewport_height_ + 7u) / 8u, 1u);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &barrier, 0, nullptr, 0, nullptr);
}

void vulkan_render_backend::dispatch_temporal_masks(VkCommandBuffer command_buffer)
{
    if (!ensure_temporal_resources(viewport_width_, viewport_height_, output_viewport_width_, output_viewport_height_))
        return;
    const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % 2u);
    prepare_temporal_images(command_buffer, generation);
    update_temporal_descriptors(generation);
    const temporal_mask_push_constants constants{
        static_cast<std::int32_t>(viewport_width_), static_cast<std::int32_t>(viewport_height_),
        temporal_history_valid_ && frame_camera_.history_valid ? 1u : 0u,
        resolved_config_.temporal.disocclusion_threshold, resolved_config_.temporal.reactive_response};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_mask_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_mask_pipeline_layout_, 0, 1,
                            &temporal_mask_sets_[generation], 0, nullptr);
    vkCmdPushConstants(command_buffer, temporal_mask_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(constants), &constants);
    vkCmdDispatch(command_buffer, (viewport_width_ + 7u) / 8u, (viewport_height_ + 7u) / 8u, 1u);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &barrier, 0, nullptr, 0, nullptr);
}

void vulkan_render_backend::dispatch_temporal_resolve(VkCommandBuffer command_buffer)
{
    const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % 2u);
    if (temporal_resolve_pipeline_ == VK_NULL_HANDLE || temporal_resolve_sets_[generation] == VK_NULL_HANDLE) return;
    const temporal_resolve_push_constants constants{static_cast<std::int32_t>(output_viewport_width_),
                                                    static_cast<std::int32_t>(output_viewport_height_),
                                                    static_cast<float>(viewport_width_),
                                                    static_cast<float>(viewport_height_),
                                                    temporal_history_valid_ && frame_camera_.history_valid ? 1u : 0u,
                                                    resolved_config_.temporal.history_weight};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_resolve_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_resolve_pipeline_layout_, 0, 1,
                            &temporal_resolve_sets_[generation], 0, nullptr);
    vkCmdPushConstants(command_buffer, temporal_resolve_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(constants), &constants);
    vkCmdDispatch(command_buffer, (output_viewport_width_ + 7u) / 8u, (output_viewport_height_ + 7u) / 8u, 1u);
    transition_graph_image(command_buffer, temporal_color_history_[generation],
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    temporal_output_view_ = temporal_color_history_[generation].view;
    temporal_history_valid_ = true;
    last_profile_.temporal.enabled = true;
    last_profile_.temporal.upscaling =
        viewport_width_ != output_viewport_width_ || viewport_height_ != output_viewport_height_;
    last_profile_.temporal.effective_method =
        last_profile_.temporal.upscaling ? anti_aliasing_method::taau : anti_aliasing_method::taa;
    last_profile_.temporal.history_valid = frame_camera_.history_valid && !frame_camera_.camera_cut;
}

void vulkan_render_backend::dispatch_temporal_sharpen(VkCommandBuffer command_buffer)
{
    const auto generation = static_cast<std::uint32_t>(last_profile_.frame_index % 2u);
    if (temporal_sharpen_pipeline_ == VK_NULL_HANDLE || temporal_sharpen_sets_[generation] == VK_NULL_HANDLE) return;
    transition_graph_image(command_buffer, temporal_sharpened_[generation], VK_IMAGE_LAYOUT_GENERAL);
    const sharpen_push_constants constants{static_cast<std::int32_t>(output_viewport_width_),
                                           static_cast<std::int32_t>(output_viewport_height_),
                                           resolved_config_.temporal.sharpening, 0.25f};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_sharpen_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_sharpen_pipeline_layout_, 0, 1,
                            &temporal_sharpen_sets_[generation], 0, nullptr);
    vkCmdPushConstants(command_buffer, temporal_sharpen_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(constants), &constants);
    vkCmdDispatch(command_buffer, (output_viewport_width_ + 7u) / 8u, (output_viewport_height_ + 7u) / 8u, 1u);
    transition_graph_image(command_buffer, temporal_sharpened_[generation], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    temporal_output_view_ = temporal_sharpened_[generation].view;
}

void vulkan_render_backend::ensure_viewport(std::uint32_t width, std::uint32_t height)
{
    width = std::max(1u, width);
    height = std::max(1u, height);
    if (viewport_image_ != VK_NULL_HANDLE && viewport_width_ == width && viewport_height_ == height) return;

    wait_for_in_flight_frames();
    destroy_viewport();

    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = viewport_format_;
    image.extent = {width, height, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VmaAllocationCreateInfo allocation{};
    allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(allocator_, &image, &allocation, &viewport_image_, &viewport_allocation_, nullptr) != VK_SUCCESS)
        return;

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = viewport_image_;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = viewport_format_;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_, &view, nullptr, &viewport_view_) != VK_SUCCESS)
    {
        destroy_viewport();
        return;
    }

    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device_, &sampler, nullptr, &viewport_sampler_) != VK_SUCCESS)
    {
        destroy_viewport();
        return;
    }

    viewport_width_ = width;
    viewport_height_ = height;
    exposure_needs_reset_ = true;
    viewport_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    ensure_deferred_targets(width, height);

    VkImageCreateInfo depth_image{};
    depth_image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depth_image.imageType = VK_IMAGE_TYPE_2D;
    depth_image.format = depth_format_;
    depth_image.extent = {width, height, 1};
    depth_image.mipLevels = 1;
    depth_image.arrayLayers = 1;
    depth_image.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_image.tiling = VK_IMAGE_TILING_OPTIMAL;
    depth_image.usage =
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo depth_allocation{};
    depth_allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(allocator_, &depth_image, &depth_allocation, &viewport_depth_image_, &viewport_depth_allocation_,
                       nullptr) != VK_SUCCESS)
    {
        destroy_viewport();
        return;
    }

    VkImageViewCreateInfo depth_view{};
    depth_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depth_view.image = viewport_depth_image_;
    depth_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depth_view.format = depth_format_;
    depth_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_view.subresourceRange.levelCount = 1;
    depth_view.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_, &depth_view, nullptr, &viewport_depth_view_) != VK_SUCCESS)
    {
        destroy_viewport();
        return;
    }
    viewport_depth_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void vulkan_render_backend::destroy_viewport() noexcept
{
    if (viewport_sampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, viewport_sampler_, nullptr);
        viewport_sampler_ = VK_NULL_HANDLE;
    }
    if (viewport_view_ != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device_, viewport_view_, nullptr);
        viewport_view_ = VK_NULL_HANDLE;
    }
    if (viewport_image_ != VK_NULL_HANDLE)
    {
        vmaDestroyImage(allocator_, viewport_image_, viewport_allocation_);
        viewport_image_ = VK_NULL_HANDLE;
        viewport_allocation_ = VK_NULL_HANDLE;
    }
    if (viewport_depth_view_ != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device_, viewport_depth_view_, nullptr);
        viewport_depth_view_ = VK_NULL_HANDLE;
    }
    if (viewport_depth_image_ != VK_NULL_HANDLE)
    {
        vmaDestroyImage(allocator_, viewport_depth_image_, viewport_depth_allocation_);
        viewport_depth_image_ = VK_NULL_HANDLE;
        viewport_depth_allocation_ = VK_NULL_HANDLE;
    }
    viewport_width_ = 0;
    viewport_height_ = 0;
    viewport_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    viewport_depth_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    destroy_graph_image(gbuffer_albedo_);
    destroy_graph_image(gbuffer_normal_);
    destroy_graph_image(gbuffer_material_);
    destroy_graph_image(gbuffer_emissive_);
    destroy_graph_image(gbuffer_motion_);
    destroy_graph_image(gbuffer_object_id_);
    destroy_graph_image(selection_mask_);
    destroy_graph_image(scene_color_);
    if (gbuffer_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, gbuffer_descriptor_pool_, nullptr);
        gbuffer_descriptor_pool_ = VK_NULL_HANDLE;
        gbuffer_descriptor_set_ = VK_NULL_HANDLE;
    }
}

void vulkan_render_backend::transition_viewport(VkCommandBuffer command_buffer, VkImageLayout new_layout)
{
    if (viewport_image_ == VK_NULL_HANDLE || viewport_layout_ == new_layout) return;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = viewport_layout_;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = viewport_image_;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (viewport_layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (viewport_layout_ == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else if (viewport_layout_ == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        if (viewport_layout_ == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        else
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = viewport_layout_ == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                        ? VK_PIPELINE_STAGE_TRANSFER_BIT
                        : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    viewport_layout_ = new_layout;
}

void vulkan_render_backend::transition_depth(VkCommandBuffer command_buffer, VkImageLayout new_layout)
{
    if (viewport_depth_image_ == VK_NULL_HANDLE || viewport_depth_layout_ == new_layout) return;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = viewport_depth_layout_;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = viewport_depth_image_;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    VkPipelineStageFlags source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destination_stage =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    if (viewport_depth_layout_ == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        source_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    }
    else if (viewport_depth_layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (viewport_depth_layout_ == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else
    {
        barrier.dstAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    viewport_depth_layout_ = new_layout;
}

void vulkan_render_backend::set_viewport_and_scissor(VkCommandBuffer command_buffer) const
{
    VkViewport viewport{};
    viewport.y = static_cast<float>(viewport_height_);
    viewport.width = static_cast<float>(viewport_width_);
    viewport.height = -static_cast<float>(viewport_height_);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{};
    scissor.extent = {viewport_width_, viewport_height_};
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);
}

void vulkan_render_backend::draw_debug_overlay(VkCommandBuffer command_buffer, debug_overlay_depth_mode mode)
{
    const auto slot = current_frame_slot();
    if (slot >= debug_overlay_buffers_.size() || debug_overlay_pipeline_layout_ == VK_NULL_HANDLE) return;
    const auto& buffer = debug_overlay_buffers_[slot];
    if (buffer.vertices.buffer == VK_NULL_HANDLE) return;
    set_viewport_and_scissor(command_buffer);
    vkCmdPushConstants(command_buffer, debug_overlay_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(float) * 16u, frame_camera_.view_projection.data());
    const VkDeviceSize offset{};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, &buffer.vertices.buffer, &offset);
    const auto draw_range = [&](VkPipeline pipeline, std::uint32_t count, std::uint32_t first)
    {
        if (pipeline == VK_NULL_HANDLE || count == 0) return;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdDraw(command_buffer, count, 1, first, 0);
    };
    if (mode == debug_overlay_depth_mode::tested)
    {
        draw_range(debug_overlay_line_pipeline_, buffer.tested_line_count, buffer.tested_line_offset);
        draw_range(debug_overlay_triangle_pipeline_, buffer.tested_triangle_count, buffer.tested_triangle_offset);
    }
    else
    {
        draw_range(debug_overlay_output_line_pipeline_, buffer.output_line_count, buffer.output_line_offset);
        draw_range(debug_overlay_output_triangle_pipeline_, buffer.output_triangle_count,
                   buffer.output_triangle_offset);
    }
}

void vulkan_render_backend::draw_indexed_mesh(VkCommandBuffer command_buffer, const draw_mesh_event& draw,
                                              VkPipelineLayout layout, VkShaderStageFlags stages, bool gpu_culled,
                                              bool write_motion)
{
    const auto found = meshes_.find(resource_key(draw.mesh));
    if (found == meshes_.end()) return;

    auto constants = build_mesh_constants(draw);
    if (write_motion)
    {
        const auto previous_mvp = math::matmul(draw.previous_view_projection, draw.previous_model);
        const auto* values = previous_mvp.data();
        std::copy(values, values + 4, constants.light_direction_intensity);
        std::copy(values + 4, values + 7, constants.light_color);
        constants.camera_position[0] = values[7];
        std::copy(values + 8, values + 11, constants.camera_position + 1);
        constants.fog_color_density[0] = values[11];
        std::copy(values + 12, values + 15, constants.fog_color_density + 1);
        constants.fog_params[0] = values[15];
    }
    vkCmdPushConstants(command_buffer, layout, stages, 0, sizeof(constants), &constants);
    const VkDeviceSize offset = 0;
    const VkBuffer vertex_buffer = mesh_vertex_buffer(found->second, draw.gpu_scene_instance);
    if (vertex_buffer == VK_NULL_HANDLE) return;
    vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
    vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    if (!gpu_culled || !draw_gpu_visibility_command(command_buffer, draw.gpu_scene_instance))
        vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
}

void vulkan_render_backend::draw_indexed_virtual_cluster(VkCommandBuffer command_buffer,
                                                         const virtual_cluster_draw& draw, VkPipelineLayout layout,
                                                         VkShaderStageFlags stages, bool gpu_culled, bool write_motion)
{
    const auto found = virtual_meshes_.find(resource_key(draw.mesh));
    if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size()) return;

    const auto& cluster = found->second.clusters[draw.cluster_index];
    if (cluster.index_count == 0 || cluster.first_index + cluster.index_count > found->second.index_count) return;

    auto constants = build_mesh_constants(draw.draw);
    if (write_motion)
    {
        const auto previous_mvp = math::matmul(draw.draw.previous_view_projection, draw.draw.previous_model);
        const auto* values = previous_mvp.data();
        std::copy(values, values + 4, constants.light_direction_intensity);
        std::copy(values + 4, values + 7, constants.light_color);
        constants.camera_position[0] = values[7];
        std::copy(values + 8, values + 11, constants.camera_position + 1);
        constants.fog_color_density[0] = values[11];
        std::copy(values + 12, values + 15, constants.fog_color_density + 1);
        constants.fog_params[0] = values[15];
    }
    vkCmdPushConstants(command_buffer, layout, stages, 0, sizeof(constants), &constants);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
    vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    if (!gpu_culled || !draw_gpu_visibility_command(command_buffer, draw.draw.gpu_scene_instance))
        vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
}

bool vulkan_render_backend::render_deferred_scene(VkCommandBuffer command_buffer)
{
    if ((frame_draws_.empty() && frame_virtual_draws_.empty() && frame_terrain_draws_.empty()) ||
        !ensure_deferred_targets(viewport_width_, viewport_height_) || !ensure_shadow_pipeline() ||
        !ensure_gbuffer_pipeline() || !ensure_gbuffer_descriptor_set() || !ensure_deferred_pipeline())
        return false;

    bool has_opaque_draws = false;
    for (const auto& draw : frame_draws_)
    {
        if (draw.mode != render_mode::wireframe && material_alpha_mode_for(draw) != material_alpha_mode::blend &&
            !material_requires_forward(draw))
        {
            has_opaque_draws = true;
            break;
        }
    }
    for (const auto& draw : frame_virtual_draws_)
    {
        if (draw.draw.mode != render_mode::wireframe &&
            material_alpha_mode_for(draw.draw) != material_alpha_mode::blend && !material_requires_forward(draw.draw))
        {
            has_opaque_draws = true;
            break;
        }
    }
    has_opaque_draws = has_opaque_draws || !frame_terrain_draws_.empty();
    if (!has_opaque_draws) return false;

    transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    {
        VkRenderingAttachmentInfo depth_attachment{};
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = viewport_depth_view_;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue.depthStencil.depth = 1.0f;

        VkRenderingInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering.renderArea.extent = {viewport_width_, viewport_height_};
        rendering.layerCount = 1;
        rendering.pDepthAttachment = &depth_attachment;
        cmd_begin_rendering(command_buffer, &rendering);
        set_viewport_and_scissor(command_buffer);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);

        for (const auto& draw : frame_draws_)
        {
            if (draw.mode == render_mode::wireframe) continue;
            VkDescriptorSet descriptor_set = material_descriptor_set_for(draw);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0, 1,
                                    &descriptor_set, 0, nullptr);
            draw_indexed_mesh(command_buffer, draw, shadow_pipeline_layout_,
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true);
        }
        for (const auto& draw : frame_virtual_draws_)
        {
            if (draw.draw.mode == render_mode::wireframe) continue;
            VkDescriptorSet descriptor_set = material_descriptor_set_for(draw.draw);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0, 1,
                                    &descriptor_set, 0, nullptr);
            draw_indexed_virtual_cluster(command_buffer, draw, shadow_pipeline_layout_,
                                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true);
        }
        if (terrain_shadow_pipeline_ != VK_NULL_HANDLE)
        {
            for (const auto& draw : frame_terrain_draws_)
                draw_terrain_patch(command_buffer, draw, terrain_shadow_pipeline_, false);
        }
        cmd_end_rendering(command_buffer);
    }

    transition_graph_image(command_buffer, gbuffer_albedo_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_normal_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_material_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_emissive_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    {
        std::array<VkRenderingAttachmentInfo, 6> color_attachments{};
        graph_image* images[6]{&gbuffer_albedo_,   &gbuffer_normal_, &gbuffer_material_,
                               &gbuffer_emissive_, &gbuffer_motion_, &gbuffer_object_id_};
        for (std::size_t index = 0; index < color_attachments.size(); ++index)
        {
            color_attachments[index].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachments[index].imageView = images[index]->view;
            color_attachments[index].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attachments[index].clearValue.color.float32[0] = 0.0f;
            color_attachments[index].clearValue.color.float32[1] = 0.0f;
            color_attachments[index].clearValue.color.float32[2] = 0.0f;
            color_attachments[index].clearValue.color.float32[3] = 0.0f;
        }

        VkRenderingAttachmentInfo depth_attachment{};
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = viewport_depth_view_;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering.renderArea.extent = {viewport_width_, viewport_height_};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_attachments.size());
        rendering.pColorAttachments = color_attachments.data();
        rendering.pDepthAttachment = &depth_attachment;
        cmd_begin_rendering(command_buffer, &rendering);
        set_viewport_and_scissor(command_buffer);
        const bool bindless_opaque_drawn = draw_gpu_bindless_batch(command_buffer, false);
        for (const auto& draw : frame_draws_)
        {
            if (draw.mode == render_mode::wireframe || material_alpha_mode_for(draw) == material_alpha_mode::blend)
                continue;
            if (bindless_opaque_drawn && gpu_bindless_draw_compatible(draw, false)) continue;
            const bool terrain_surface = material_is_terrain(draw) && draw.material_attribute_texture.valid();
            if (!material_is_terrain(draw) && draw_runtime_material_gbuffer(command_buffer, draw)) continue;
            if (terrain_surface && terrain_surface_gbuffer_pipeline_ != VK_NULL_HANDLE)
            {
                const auto attributes = material_attribute_descriptor_set_for(draw.material_attribute_texture);
                if (attributes != VK_NULL_HANDLE)
                {
                    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                      terrain_surface_gbuffer_pipeline_);
                    const std::array descriptor_sets{material_descriptor_set_for(draw), attributes};
                    vkCmdBindDescriptorSets(
                        command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain_surface_pipeline_layout_, 0,
                        static_cast<std::uint32_t>(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
                    draw_indexed_mesh(command_buffer, draw, terrain_surface_pipeline_layout_,
                                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true, true);
                    continue;
                }
            }
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, gbuffer_pipeline_);
            VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline_layout_, 0, 1,
                                    &material_descriptor_set, 0, nullptr);
            draw_indexed_mesh(command_buffer, draw, mesh_pipeline_layout_,
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true, true);
        }
        for (const auto& draw : frame_virtual_draws_)
        {
            if (draw.draw.mode == render_mode::wireframe ||
                material_alpha_mode_for(draw.draw) == material_alpha_mode::blend)
                continue;
            if (!material_is_terrain(draw.draw) && draw_runtime_material_gbuffer(command_buffer, draw)) continue;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              material_is_terrain(draw.draw) && terrain_gbuffer_pipeline_ != VK_NULL_HANDLE
                                  ? terrain_gbuffer_pipeline_
                                  : gbuffer_pipeline_);
            VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw.draw);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline_layout_, 0, 1,
                                    &material_descriptor_set, 0, nullptr);
            draw_indexed_virtual_cluster(command_buffer, draw, mesh_pipeline_layout_,
                                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true, true);
        }
        for (const auto& draw : frame_gpu_terrain_draws_)
            draw_gpu_terrain(command_buffer, draw, terrain_gbuffer_pipeline_, true);
        for (const auto& draw : frame_terrain_draws_)
            if (!gpu_terrain_active_instances_.contains(gpu_scene_key(draw.terrain.gpu_scene_instance)))
                draw_terrain_patch(command_buffer, draw, terrain_gbuffer_pipeline_, true);

        cmd_end_rendering(command_buffer);
    }

    if (resolved_config_.features.virtual_geometry && !dispatch_virtual_geometry_material_resolve(command_buffer))
        last_profile_.virtual_geometry.fallback_reason =
            "virtual-geometry material resolve is unavailable; using conventional material draws";

    transition_graph_image(command_buffer, gbuffer_albedo_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_normal_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_material_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_emissive_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_motion_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition_depth(command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    update_gbuffer_descriptor_set();

    {
        transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color_attachment{};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = scene_color_.view;
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering.renderArea.extent = {viewport_width_, viewport_height_};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color_attachment;
        cmd_begin_rendering(command_buffer, &rendering);
        set_viewport_and_scissor(command_buffer);
        deferred_push_constants constants{};
        std::copy(frame_camera_.inverse_view_projection.data(), frame_camera_.inverse_view_projection.data() + 16,
                  constants.inverse_view_projection);
        constants.camera_position[0] = frame_camera_.position[0];
        constants.camera_position[1] = frame_camera_.position[1];
        constants.camera_position[2] = frame_camera_.position[2];
        constants.light_direction_intensity[0] = 0.35f;
        constants.light_direction_intensity[1] = -0.85f;
        constants.light_direction_intensity[2] = -0.40f;
        constants.light_direction_intensity[3] = frame_shadows_enabled_ ? 1.0f : 0.0f;
        if (!frame_directional_lights_.empty())
        {
            const auto& light = frame_directional_lights_.front();
            constants.light_direction_intensity[0] = light.direction[0];
            constants.light_direction_intensity[1] = light.direction[1];
            constants.light_direction_intensity[2] = light.direction[2];
            constants.light_color[0] = light.color[0];
            constants.light_color[1] = light.color[1];
            constants.light_color[2] = light.color[2];
        }
        constants.ambient_visualization[0] =
            frame_lighting_.ambient_color_intensity[0] * frame_lighting_.ambient_color_intensity[3];
        constants.ambient_visualization[1] =
            frame_lighting_.ambient_color_intensity[1] * frame_lighting_.ambient_color_intensity[3];
        constants.ambient_visualization[2] =
            frame_lighting_.ambient_color_intensity[2] * frame_lighting_.ambient_color_intensity[3];
        if (const auto* environment = active_environment())
        {
            const auto found = textures_.find(resource_key(environment->equirectangular_texture));
            constants.light_color[3] = found != textures_.end() && found->second.view != VK_NULL_HANDLE ? 1.0f : 0.0f;
        }
        constants.ambient_visualization[3] = !frame_draws_.empty()
                                                 ? static_cast<float>(frame_draws_.front().visualization)
                                                 : static_cast<float>(frame_virtual_draws_.front().draw.visualization);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferred_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, deferred_pipeline_layout_, 0, 1,
                                &gbuffer_descriptor_set_, 0, nullptr);
        vkCmdPushConstants(command_buffer, deferred_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(constants), &constants);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        cmd_end_rendering(command_buffer);
    }

    if (pending_pick_request_ && ensure_pick_readback_buffer())
    {
        const auto request = *pending_pick_request_;
        pending_pick_request_.reset();

        if (request.x < output_viewport_width_ && request.y < output_viewport_height_ && gbuffer_object_id_.width > 0 &&
            gbuffer_object_id_.height > 0)
        {
            object_pick_readback readback{};
            readback.request = request;
            readback.frame_index = last_profile_.frame_index;
            readback.frame_slot = active_frame_index_;
            readback.active = true;

            for (const auto& draw : frame_draws_)
            {
                if (draw.object_id.valid()) readback.objects.emplace(draw.object_id.index + 1u, draw.object_id);
            }
            for (const auto& draw : frame_virtual_draws_)
            {
                if (draw.draw.object_id.valid())
                    readback.objects.emplace(draw.draw.object_id.index + 1u, draw.draw.object_id);
            }

            transition_graph_image(command_buffer, gbuffer_object_id_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            const auto render_x =
                detail::map_output_pixel_to_render_pixel(request.x, output_viewport_width_, gbuffer_object_id_.width);
            const auto render_y =
                detail::map_output_pixel_to_render_pixel(request.y, output_viewport_height_, gbuffer_object_id_.height);
            region.imageOffset = {static_cast<std::int32_t>(render_x), static_cast<std::int32_t>(render_y), 0};
            region.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(command_buffer, gbuffer_object_id_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   pick_readback_buffer_.buffer, 1, &region);

            in_flight_pick_ = std::move(readback);
        }
        else
        {
            last_pick_result_ = {.request_id = request.request_id,
                                 .available = true,
                                 .hit = false,
                                 .object = {},
                                 .x = request.x,
                                 .y = request.y,
                                 .frame_index = last_profile_.frame_index};
        }
    }

    return true;
}

void vulkan_render_backend::render_viewport(VkCommandBuffer command_buffer, bool render_scene, bool render_output)
{
    if (viewport_image_ == VK_NULL_HANDLE) return;

    if (render_scene)
    {
        transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color_attachment{};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = scene_color_.view;
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        for (std::uint32_t channel = 0; channel < 4; ++channel)
            color_attachment.clearValue.color.float32[channel] = frame_camera_.clear_color[channel];

        VkRenderingAttachmentInfo depth_attachment{};
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = viewport_depth_view_;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue.depthStencil.depth = 1.0f;

        VkRenderingInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering.renderArea.extent = {viewport_width_, viewport_height_};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color_attachment;
        rendering.pDepthAttachment = &depth_attachment;
        cmd_begin_rendering(command_buffer, &rendering);

        if (frame_environment_.enabled && frame_environment_.sky_visible && ensure_sky_pipeline())
        {
            VkViewport viewport{};
            viewport.y = static_cast<float>(viewport_height_);
            viewport.width = static_cast<float>(viewport_width_);
            viewport.height = -static_cast<float>(viewport_height_);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.extent = {viewport_width_, viewport_height_};
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);

            math::vector3f sun_direction_override{};
            if (!frame_environment_.celestial.enabled && !frame_directional_lights_.empty())
                sun_direction_override = frame_directional_lights_.front().direction;
            const auto constants = detail::build_sky_push_constants(
                frame_environment_, frame_camera_, viewport_width_, viewport_height_,
                resolved_config_.quality != render_quality_tier::low, sun_direction_override);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_);
            const auto sky_descriptor = update_current_sky_descriptor_set();
            if (sky_descriptor != VK_NULL_HANDLE)
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_layout_, 0, 1,
                                        &sky_descriptor, 0, nullptr);
            vkCmdPushConstants(command_buffer, sky_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                               &constants);
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
        }

        cmd_end_rendering(command_buffer);
        const bool deferred_rendered =
            resolved_config_.path == render_path::deferred && render_deferred_scene(command_buffer);

        transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition_depth(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        cmd_begin_rendering(command_buffer, &rendering);

        if ((!frame_draws_.empty() || !frame_virtual_draws_.empty()) && mesh_pipeline_ != VK_NULL_HANDLE &&
            !white_descriptor_sets_.empty())
        {
            VkViewport viewport{};
            viewport.y = static_cast<float>(viewport_height_);
            viewport.width = static_cast<float>(viewport_width_);
            viewport.height = -static_cast<float>(viewport_height_);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.extent = {viewport_width_, viewport_height_};
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);

            const auto draw_with_pipeline = [&](const draw_mesh_event& draw, VkPipeline pipeline)
            {
                if (pipeline == VK_NULL_HANDLE) return;
                auto found = meshes_.find(resource_key(draw.mesh));
                if (found == meshes_.end()) return;

                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                auto constants = build_mesh_constants(draw);
                if (pipeline == mesh_wire_pipeline_)
                {
                    std::copy(draw.wire_color.data(), draw.wire_color.data() + 4, constants.base_color);
                    constants.light_color[3] = 0.0f;
                    constants.visualization[0] = static_cast<float>(mesh_visualization_mode::albedo);
                    constants.fog_color_density[3] = 0.0f;
                    constants.material_params[3] = static_cast<float>(material_alpha_mode::opaque);
                }
                VkPipelineLayout pipeline_layout = mesh_pipeline_layout_;
                VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw);
                if (pipeline == terrain_surface_pipeline_)
                {
                    const auto attributes = material_attribute_descriptor_set_for(draw.material_attribute_texture);
                    if (attributes == VK_NULL_HANDLE) return;
                    const std::array descriptor_sets{material_descriptor_set, attributes};
                    pipeline_layout = terrain_surface_pipeline_layout_;
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0,
                                            static_cast<std::uint32_t>(descriptor_sets.size()), descriptor_sets.data(),
                                            0, nullptr);
                }
                else
                {
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                                            &material_descriptor_set, 0, nullptr);
                }
                vkCmdPushConstants(command_buffer, pipeline_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                   &constants);
                const VkDeviceSize offset = 0;
                const VkBuffer vertex_buffer = mesh_vertex_buffer(found->second, draw.gpu_scene_instance);
                if (vertex_buffer == VK_NULL_HANDLE) return;
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
                vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                if (!draw_gpu_visibility_command(command_buffer, draw.gpu_scene_instance))
                    vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
            };
            const auto draw_virtual_with_pipeline = [&](const virtual_cluster_draw& draw, VkPipeline pipeline)
            {
                if (pipeline == VK_NULL_HANDLE) return;
                const auto found = virtual_meshes_.find(resource_key(draw.mesh));
                if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size()) return;

                const auto& cluster = found->second.clusters[draw.cluster_index];
                if (cluster.index_count == 0 || cluster.first_index + cluster.index_count > found->second.index_count)
                    return;

                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                auto constants = build_mesh_constants(draw.draw);
                if (pipeline == mesh_wire_pipeline_)
                {
                    std::copy(draw.draw.wire_color.data(), draw.draw.wire_color.data() + 4, constants.base_color);
                    constants.light_color[3] = 0.0f;
                    constants.visualization[0] = static_cast<float>(mesh_visualization_mode::albedo);
                    constants.fog_color_density[3] = 0.0f;
                    constants.material_params[3] = static_cast<float>(material_alpha_mode::opaque);
                }
                VkDescriptorSet material_descriptor_set = material_descriptor_set_for(draw.draw);
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline_layout_, 0, 1,
                                        &material_descriptor_set, 0, nullptr);
                vkCmdPushConstants(command_buffer, mesh_pipeline_layout_,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                   &constants);
                const VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                if (!draw_gpu_visibility_command(command_buffer, draw.draw.gpu_scene_instance))
                    vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
            };

            for (const auto& draw : frame_draws_)
            {
                if (draw.mode == render_mode::wireframe)
                {
                    if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
                        draw_with_pipeline(draw, mesh_wire_pipeline_);
                    else
                        draw_with_pipeline(draw, mesh_pipeline_);
                    continue;
                }

                if (material_alpha_mode_for(draw) == material_alpha_mode::blend) continue;

                if (deferred_rendered && !material_requires_forward(draw))
                {
                    continue;
                }

                draw_with_pipeline(draw, material_is_terrain(draw) && draw.material_attribute_texture.valid() &&
                                                 terrain_surface_pipeline_ != VK_NULL_HANDLE
                                             ? terrain_surface_pipeline_
                                             : mesh_pipeline_);
            }

            for (const auto& draw : frame_virtual_draws_)
            {
                if (draw.draw.mode == render_mode::wireframe)
                {
                    if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
                        draw_virtual_with_pipeline(draw, mesh_wire_pipeline_);
                    else
                        draw_virtual_with_pipeline(draw, mesh_pipeline_);
                    continue;
                }

                if (material_alpha_mode_for(draw.draw) == material_alpha_mode::blend) continue;

                if (deferred_rendered && !material_requires_forward(draw.draw)) continue;

                draw_virtual_with_pipeline(draw, material_is_terrain(draw.draw) && terrain_pipeline_ != VK_NULL_HANDLE
                                                     ? terrain_pipeline_
                                                     : mesh_pipeline_);
            }

            if (!deferred_rendered)
            {
                for (const auto& draw : frame_gpu_terrain_draws_)
                    draw_gpu_terrain(command_buffer, draw, terrain_pipeline_, false);
                for (const auto& draw : frame_terrain_draws_)
                    if (!gpu_terrain_active_instances_.contains(gpu_scene_key(draw.terrain.gpu_scene_instance)))
                        draw_terrain_patch(command_buffer, draw, terrain_pipeline_, false);
            }

            const bool bindless_transparent_drawn = draw_gpu_bindless_batch(command_buffer, true);
            std::vector<const draw_mesh_event*> transparent_draws;
            for (const auto& draw : frame_draws_)
            {
                if (draw.mode == render_mode::wireframe || material_alpha_mode_for(draw) != material_alpha_mode::blend)
                    continue;
                if (bindless_transparent_drawn && gpu_bindless_draw_compatible(draw, true)) continue;
                transparent_draws.push_back(&draw);
            }
            std::sort(transparent_draws.begin(), transparent_draws.end(),
                      [&](const draw_mesh_event* lhs, const draw_mesh_event* rhs)
                      {
                          const auto lhs_delta = math::sub(matrix_translation(lhs->model), frame_camera_.position);
                          const auto rhs_delta = math::sub(matrix_translation(rhs->model), frame_camera_.position);
                          return math::length_squared(lhs_delta) > math::length_squared(rhs_delta);
                      });
            for (const auto* draw : transparent_draws)
            {
                draw_with_pipeline(*draw, mesh_transparent_pipeline_ != VK_NULL_HANDLE ? mesh_transparent_pipeline_
                                                                                       : mesh_pipeline_);
            }

            // Selection is an editor overlay, not part of the material path.
            // Draw it after deferred, forward, and transparent geometry so
            // ordinary deferred objects cannot skip their highlight.
            if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
            {
                for (const auto& draw : frame_draws_)
                    if (draw.selected && draw.mode != render_mode::wireframe && !material_is_terrain(draw))
                        draw_with_pipeline(draw, mesh_wire_pipeline_);
                for (const auto& draw : frame_virtual_draws_)
                    if (draw.draw.selected && draw.draw.mode != render_mode::wireframe &&
                        !material_is_terrain(draw.draw))
                        draw_virtual_with_pipeline(draw, mesh_wire_pipeline_);
            }
        }

        draw_debug_overlay(command_buffer, debug_overlay_depth_mode::tested);
        cmd_end_rendering(command_buffer);
    }

    if (!render_output) return;
    transition_graph_image(command_buffer, scene_color_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!ensure_output_transform_pipeline())
    {
        arc::diagnostics::warn("render.vulkan",
                               "Output transform unavailable; the viewport retains its previous valid image");
        return;
    }
    dispatch_exposure(command_buffer);
    transition_viewport(command_buffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo output_attachment{};
    output_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    output_attachment.imageView = viewport_view_;
    output_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    output_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    output_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo output_rendering{};
    output_rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    output_rendering.renderArea.extent = {viewport_width_, viewport_height_};
    output_rendering.layerCount = 1;
    output_rendering.colorAttachmentCount = 1;
    output_rendering.pColorAttachments = &output_attachment;
    cmd_begin_rendering(command_buffer, &output_rendering);
    set_viewport_and_scissor(command_buffer);
    output_transform_push_constants output_constants{};
    output_constants.exposure_output[0] =
        frame_camera_.exposure.mode == exposure_mode::manual
            ? exposure_multiplier(frame_camera_.exposure.manual_ev100, frame_camera_.exposure.compensation_ev)
            : 1.0f;
    output_constants.exposure_output[1] = frame_camera_.exposure.mode == exposure_mode::automatic ? 1.0f : 0.0f;
    output_constants.exposure_output[2] = frame_camera_.exposure.compensation_ev;
    const auto visualization = !frame_draws_.empty()           ? frame_draws_.front().visualization
                               : !frame_virtual_draws_.empty() ? frame_virtual_draws_.front().draw.visualization
                                                               : mesh_visualization_mode::standard;
    output_constants.exposure_output[3] = visualization == mesh_visualization_mode::standard ? 0.0f : 1.0f;
    output_constants.post_process[0] = frame_fxaa_enabled_ ? 1.0f : 0.0f;
    output_constants.post_process[1] = 1.0f / static_cast<float>(std::max(viewport_width_, 1u));
    output_constants.post_process[2] = 1.0f / static_cast<float>(std::max(viewport_height_, 1u));
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, output_transform_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, output_transform_pipeline_layout_, 0, 1,
                            &output_transform_descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer, output_transform_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(output_constants), &output_constants);
    vkCmdDraw(command_buffer, 3, 1, 0, 0);
    draw_debug_overlay(command_buffer, debug_overlay_depth_mode::always);
    cmd_end_rendering(command_buffer);
    record_frame_capture(command_buffer);
    transition_viewport(command_buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace arc::render::vulkan::backend_detail
