#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"

namespace arc::render::vulkan::backend_detail
{
void vulkan_render_backend::destroy_virtual_shadow_resources(vulkan_virtual_shadow_resources& resources) noexcept
{
    destroy_buffer(resources.page_table);
    destroy_buffer(resources.requests);
    destroy_buffer(resources.feedback);
    if (resources.sampler != VK_NULL_HANDLE) vkDestroySampler(device_, resources.sampler, nullptr);
    if (resources.static_view != VK_NULL_HANDLE) vkDestroyImageView(device_, resources.static_view, nullptr);
    if (resources.dynamic_view != VK_NULL_HANDLE) vkDestroyImageView(device_, resources.dynamic_view, nullptr);
    if (resources.static_image != VK_NULL_HANDLE)
        vmaDestroyImage(allocator_, resources.static_image, resources.static_allocation);
    if (resources.dynamic_image != VK_NULL_HANDLE)
        vmaDestroyImage(allocator_, resources.dynamic_image, resources.dynamic_allocation);
    resources = {};
}

void vulkan_render_backend::retire_virtual_shadow_resources()
{
    if (virtual_shadow_resources_.static_image == VK_NULL_HANDLE &&
        virtual_shadow_resources_.page_table.buffer == VK_NULL_HANDLE)
        return;
    auto retired = virtual_shadow_resources_;
    virtual_shadow_resources_ = {};
    deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                             [this, retired]() mutable { destroy_virtual_shadow_resources(retired); });
}

bool vulkan_render_backend::ensure_virtual_shadow_resources()
{
    if (!virtual_shadow_cache_ || virtual_shadow_cache_->physical_page_capacity() == 0) return false;
    const std::uint32_t page_capacity = virtual_shadow_cache_->physical_page_capacity();
    const std::uint32_t physical_page_extent = virtual_shadow_page_texels + virtual_shadow_page_guard_texels * 2u;
    const std::uint32_t pages_per_axis =
        static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<double>(page_capacity))));
    const std::uint32_t atlas_extent = pages_per_axis * physical_page_extent;
    if (atlas_extent == 0 || atlas_extent > capabilities_.max_texture_dimension_2d) return false;
    if (virtual_shadow_resources_.static_image != VK_NULL_HANDLE &&
        virtual_shadow_resources_.physical_page_capacity == page_capacity &&
        virtual_shadow_resources_.atlas_extent == atlas_extent)
        return true;

    retire_virtual_shadow_resources();
    vulkan_virtual_shadow_resources resources{};
    VkFormatProperties d16_properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device_, VK_FORMAT_D16_UNORM, &d16_properties);
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    resources.format =
        (d16_properties.optimalTilingFeatures & required) == required ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
    VkFormatProperties selected_properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device_, resources.format, &selected_properties);
    if ((selected_properties.optimalTilingFeatures & required) != required) return false;

    const auto create_depth_atlas = [&](VkImage& image, VmaAllocation& allocation, VkImageView& view) -> bool
    {
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = resources.format;
        image_info.extent = {atlas_extent, atlas_extent, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo allocation_info{};
        allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator_, &image_info, &allocation_info, &image, &allocation, nullptr) != VK_SUCCESS)
            return false;
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = resources.format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.layerCount = 1;
        return vkCreateImageView(device_, &view_info, nullptr, &view) == VK_SUCCESS;
    };
    if (!create_depth_atlas(resources.static_image, resources.static_allocation, resources.static_view) ||
        !create_depth_atlas(resources.dynamic_image, resources.dynamic_allocation, resources.dynamic_view))
    {
        destroy_virtual_shadow_resources(resources);
        return false;
    }

    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    const bool linear_filter =
        (selected_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    sampler.magFilter = linear_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sampler.minFilter = sampler.magFilter;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sampler.compareEnable = VK_TRUE;
    sampler.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (vkCreateSampler(device_, &sampler, nullptr, &resources.sampler) != VK_SUCCESS)
    {
        destroy_virtual_shadow_resources(resources);
        return false;
    }

    resources.page_table_capacity = static_cast<VkDeviceSize>(page_capacity) * sizeof(gpu_virtual_shadow_page_mapping);
    const VkDeviceSize request_capacity =
        static_cast<VkDeviceSize>(std::max(4096u, resolved_config_.virtual_shadow_page_render_budget * 2u)) *
        sizeof(virtual_shadow_page_request);
    if (!create_buffer(resources.page_table_capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                       resources.page_table) ||
        !create_buffer(request_capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VMA_MEMORY_USAGE_GPU_ONLY, resources.requests) ||
        !create_buffer(request_capacity, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU,
                       resources.feedback))
    {
        destroy_virtual_shadow_resources(resources);
        return false;
    }
    resources.atlas_extent = atlas_extent;
    resources.physical_page_capacity = page_capacity;
    virtual_shadow_resources_ = resources;
    return true;
}

std::uint64_t vulkan_render_backend::virtual_shadow_light_key(shadow_light_kind kind, render_object_id object) noexcept
{
    std::uint64_t key = light_shadow_key(object);
    key ^= static_cast<std::uint64_t>(kind) + 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u);
    return key;
}

void vulkan_render_backend::prepare_virtual_shadow_cache(std::uint64_t frame_index)
{
    pending_virtual_shadow_pages_.clear();
    if (!resolved_config_.features.virtual_shadow_maps || !virtual_shadow_cache_) return;

    std::vector<virtual_shadow_page_request> requests;
    const auto append_light =
        [&](shadow_light_kind kind, render_object_id object, render_mobility mobility, const shadow_settings& settings)
    {
        if (!object.valid() || !settings.enabled || settings.map_method == shadow_map_method::conventional) return;
        const std::uint64_t key = virtual_shadow_light_key(kind, object);
        auto found = virtual_shadow_lights_.find(key);
        if (found == virtual_shadow_lights_.end())
        {
            const auto address_space = virtual_shadow_cache_->create_address_space(
                {.light_kind = kind,
                 .light_key = key,
                 .mobility = mobility,
                 .virtual_resolution =
                     kind == shadow_light_kind::directional ? 16384u : std::max(settings.resolution, 2048u),
                 .level_count = virtual_shadow_directional_clip_levels,
                 .priority = settings.priority});
            if (!address_space) return;
            found = virtual_shadow_lights_.emplace(key, virtual_shadow_light_state{*address_space, frame_index}).first;
        }
        found->second.last_seen_frame = frame_index;
        const auto* descriptor = virtual_shadow_cache_->address_space(found->second.address_space);
        if (!descriptor) return;
        const std::uint8_t root_level = static_cast<std::uint8_t>(descriptor->level_count - 1u);
        const auto append_layer = [&](virtual_shadow_page_layer layer, std::uint8_t face)
        {
            const std::uint64_t revision = layer == virtual_shadow_page_layer::dynamic_depth
                                               ? shadow_resource_revision_ ^ frame_index
                                               : shadow_resource_revision_;
            requests.push_back({.key = {.address_space = found->second.address_space,
                                        .coordinate = {.x = 0, .y = 0, .level = root_level, .face = face},
                                        .layer = layer},
                                .frame_index = frame_index,
                                .content_revision = revision,
                                .projected_coverage = 1.0f,
                                .light_priority = settings.priority,
                                .coarse_page = true});
        };
        for (std::uint8_t face = 0; face < descriptor->face_count; ++face)
        {
            if (mobility != render_mobility::movable) append_layer(virtual_shadow_page_layer::static_depth, face);
            if (mobility != render_mobility::static_object)
                append_layer(virtual_shadow_page_layer::dynamic_depth, face);
        }
    };

    for (const auto& light : frame_directional_lights_)
        if (light.enabled && light.casts_shadows)
            append_light(shadow_light_kind::directional, light.object_id, light.mobility, light.shadow);
    for (const auto& light : frame_point_lights_)
        if (light.enabled && light.casts_shadows)
            append_light(shadow_light_kind::point, light.object_id, light.mobility, light.shadow);
    for (const auto& light : frame_spot_lights_)
        if (light.enabled && light.casts_shadows)
            append_light(shadow_light_kind::spot, light.object_id, light.mobility, light.shadow);

    for (auto iterator = virtual_shadow_lights_.begin(); iterator != virtual_shadow_lights_.end();)
    {
        if (iterator->second.last_seen_frame == frame_index)
        {
            ++iterator;
            continue;
        }
        const bool destroyed = virtual_shadow_cache_->destroy_address_space(iterator->second.address_space);
        (void)destroyed; // Stale light cleanup is best-effort; generation checks reject already-retired spaces.
        iterator = virtual_shadow_lights_.erase(iterator);
    }

    const auto result = virtual_shadow_cache_->resolve_requests(requests, frame_index);
    pending_virtual_shadow_pages_ = result.render_pages;
    if (pending_virtual_shadow_pages_.size() > resolved_config_.virtual_shadow_page_render_budget)
        pending_virtual_shadow_pages_.resize(resolved_config_.virtual_shadow_page_render_budget);
    for (const auto& mapping : pending_virtual_shadow_pages_)
    {
        const bool marked = virtual_shadow_cache_->set_in_flight(mapping.key, true);
        (void)marked; // A concurrent invalidation may have retired the request before submission.
    }

    const auto stats = virtual_shadow_cache_->statistics();
    auto& profile = last_profile_.shadows;
    profile.virtual_shadow_maps = true;
    profile.virtual_address_space_count = stats.address_space_count;
    profile.virtual_page_capacity = stats.physical_page_capacity;
    profile.virtual_resident_pages = stats.resident_pages;
    profile.virtual_dirty_pages = stats.dirty_pages;
    profile.virtual_reused_pages = result.cache_hits;
    profile.virtual_evictions = stats.eviction_count;
    profile.virtual_parent_fallbacks = stats.parent_fallbacks;
    profile.virtual_failed_requests = stats.failed_requests;
    profile.virtual_memory_bytes = stats.physical_memory_bytes;
}

void vulkan_render_backend::transition_virtual_shadow_image(VkCommandBuffer command_buffer, VkImage image,
                                                            VkImageLayout& current_layout, VkImageLayout new_layout)
{
    if (image == VK_NULL_HANDLE || current_layout == new_layout) return;
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = current_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    VkPipelineStageFlags source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    if (current_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        source_stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    }
    else if (current_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else
    {
        barrier.dstAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    current_layout = new_layout;
}

void vulkan_render_backend::clear_virtual_shadow_render_pages(VkCommandBuffer command_buffer,
                                                              virtual_shadow_page_layer layer)
{
    auto& resources = virtual_shadow_resources_;
    VkImage image = layer == virtual_shadow_page_layer::static_depth ? resources.static_image : resources.dynamic_image;
    VkImageView view =
        layer == virtual_shadow_page_layer::static_depth ? resources.static_view : resources.dynamic_view;
    auto& layout =
        layer == virtual_shadow_page_layer::static_depth ? resources.static_layout : resources.dynamic_layout;
    if (image == VK_NULL_HANDLE || view == VK_NULL_HANDLE || pending_virtual_shadow_pages_.empty()) return;
    transition_virtual_shadow_image(command_buffer, image, layout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.extent = {resources.atlas_extent, resources.atlas_extent};
    rendering.layerCount = 1;
    rendering.pDepthAttachment = &depth_attachment;
    cmd_begin_rendering(command_buffer, &rendering);
    const std::uint32_t physical_extent = virtual_shadow_page_texels + virtual_shadow_page_guard_texels * 2u;
    const std::uint32_t pages_per_axis = resources.atlas_extent / physical_extent;
    VkClearAttachment attachment{};
    attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    attachment.clearValue.depthStencil.depth = 1.0f;
    for (const auto& mapping : pending_virtual_shadow_pages_)
    {
        if (mapping.key.layer != layer || !mapping.physical_page.valid()) continue;
        const std::uint32_t x = mapping.physical_page.index % pages_per_axis;
        const std::uint32_t y = mapping.physical_page.index / pages_per_axis;
        VkClearRect rect{};
        rect.rect.offset = {static_cast<std::int32_t>(x * physical_extent),
                            static_cast<std::int32_t>(y * physical_extent)};
        rect.rect.extent = {physical_extent, physical_extent};
        rect.layerCount = 1;
        vkCmdClearAttachments(command_buffer, 1, &attachment, 1, &rect);
    }
    cmd_end_rendering(command_buffer);
}

void vulkan_render_backend::publish_virtual_shadow_pages(VkCommandBuffer command_buffer)
{
    if (!virtual_shadow_cache_) return;
    for (const auto& mapping : pending_virtual_shadow_pages_)
    {
        const bool published = virtual_shadow_cache_->publish(mapping.key, mapping.content_revision);
        (void)published; // Stale generations are intentionally discarded instead of being published.
    }
    last_profile_.shadows.virtual_rendered_pages = static_cast<std::uint32_t>(pending_virtual_shadow_pages_.size());
    pending_virtual_shadow_pages_.clear();

    const auto mappings = virtual_shadow_cache_->mappings();
    const std::size_t count = std::min<std::size_t>(mappings.size(), virtual_shadow_resources_.page_table_capacity /
                                                                         sizeof(gpu_virtual_shadow_page_mapping));
    if (count > 0 && virtual_shadow_resources_.page_table.allocation != VK_NULL_HANDLE)
    {
        void* mapped{};
        if (vmaMapMemory(allocator_, virtual_shadow_resources_.page_table.allocation, &mapped) == VK_SUCCESS)
        {
            auto* output = static_cast<gpu_virtual_shadow_page_mapping*>(mapped);
            for (std::size_t index = 0; index < count; ++index)
            {
                const auto& input = mappings[index];
                output[index] = {.address_space_index = input.key.address_space.index,
                                 .address_space_generation = input.key.address_space.generation,
                                 .physical_page_index = input.physical_page.index,
                                 .physical_page_generation = input.physical_page.generation,
                                 .packed_coordinate = static_cast<std::uint32_t>(input.key.coordinate.x) |
                                                      (static_cast<std::uint32_t>(input.key.coordinate.y) << 12u) |
                                                      (static_cast<std::uint32_t>(input.key.coordinate.level) << 24u) |
                                                      (static_cast<std::uint32_t>(input.key.coordinate.face) << 28u),
                                 .flags = (input.resident ? 1u : 0u) | (input.pinned ? 2u : 0u) |
                                          (input.key.layer == virtual_shadow_page_layer::dynamic_depth ? 4u : 0u),
                                 .content_revision_low = static_cast<std::uint32_t>(input.content_revision),
                                 .content_revision_high = static_cast<std::uint32_t>(input.content_revision >> 32u)};
            }
            vmaFlushAllocation(allocator_, virtual_shadow_resources_.page_table.allocation, 0,
                               count * sizeof(gpu_virtual_shadow_page_mapping));
            vmaUnmapMemory(allocator_, virtual_shadow_resources_.page_table.allocation);
        }
    }
    transition_virtual_shadow_image(command_buffer, virtual_shadow_resources_.static_image,
                                    virtual_shadow_resources_.static_layout,
                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    transition_virtual_shadow_image(command_buffer, virtual_shadow_resources_.dynamic_image,
                                    virtual_shadow_resources_.dynamic_layout,
                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    const auto stats = virtual_shadow_cache_->statistics();
    last_profile_.shadows.virtual_resident_pages = stats.resident_pages;
    last_profile_.shadows.virtual_dirty_pages = stats.dirty_pages;
}

void vulkan_render_backend::destroy_shadow_resources() noexcept
{
    for (auto& view : shadow_atlas_.cascade_views)
    {
        if (view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, view, nullptr);
            view = VK_NULL_HANDLE;
        }
    }
    if (shadow_atlas_.array_view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device_, shadow_atlas_.array_view, nullptr);
        shadow_atlas_.array_view = VK_NULL_HANDLE;
    }
    if (shadow_atlas_.sampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, shadow_atlas_.sampler, nullptr);
        shadow_atlas_.sampler = VK_NULL_HANDLE;
    }
    if (shadow_atlas_.image != VK_NULL_HANDLE)
    {
        vmaDestroyImage(allocator_, shadow_atlas_.image, shadow_atlas_.allocation);
        shadow_atlas_.image = VK_NULL_HANDLE;
        shadow_atlas_.allocation = VK_NULL_HANDLE;
    }
    shadow_atlas_.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    shadow_atlas_.resolution = 0;
    shadow_cache_.static_layers_valid = false;
}

void vulkan_render_backend::destroy_local_shadow_resources() noexcept
{
    if (local_shadow_atlas_.sampler != VK_NULL_HANDLE) vkDestroySampler(device_, local_shadow_atlas_.sampler, nullptr);
    if (local_shadow_atlas_.view != VK_NULL_HANDLE) vkDestroyImageView(device_, local_shadow_atlas_.view, nullptr);
    if (local_shadow_atlas_.image != VK_NULL_HANDLE)
        vmaDestroyImage(allocator_, local_shadow_atlas_.image, local_shadow_atlas_.allocation);
    local_shadow_atlas_ = {};
}

bool vulkan_render_backend::ensure_local_shadow_resources()
{
    const std::uint32_t resolution =
        active_local_shadows_.empty() ? 1u : std::max(resolved_config_.local_shadow_atlas_resolution, 128u);
    if (local_shadow_atlas_.image != VK_NULL_HANDLE && local_shadow_atlas_.resolution == resolution) return true;

    wait_for_in_flight_frames();
    destroy_local_shadow_resources();
    for (auto& shadow : active_local_shadows_)
        shadow.redraw = true;

    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = depth_format_;
    image.extent = {resolution, resolution, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo allocation{};
    allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(allocator_, &image, &allocation, &local_shadow_atlas_.image, &local_shadow_atlas_.allocation,
                       nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = local_shadow_atlas_.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = depth_format_;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_, &view, nullptr, &local_shadow_atlas_.view) != VK_SUCCESS)
    {
        destroy_local_shadow_resources();
        return false;
    }

    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sampler.compareEnable = VK_TRUE;
    sampler.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (vkCreateSampler(device_, &sampler, nullptr, &local_shadow_atlas_.sampler) != VK_SUCCESS)
    {
        destroy_local_shadow_resources();
        return false;
    }
    local_shadow_atlas_.resolution = resolution;
    local_shadow_atlas_.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

std::uint32_t vulkan_render_backend::frame_resource_count() const noexcept
{
    return std::max(1u, swapchain_.image_count());
}

std::uint32_t vulkan_render_backend::current_frame_slot() const noexcept
{
    return active_frame_index_ % frame_resource_count();
}

bool vulkan_render_backend::ensure_shadow_uniform_buffers()
{
    const auto count = frame_resource_count();
    if (shadow_uniform_buffers_.size() == count)
    {
        bool ready = true;
        for (const auto& buffer : shadow_uniform_buffers_)
            ready = ready && buffer.buffer != VK_NULL_HANDLE;
        if (ready) return true;
    }

    wait_for_in_flight_frames();
    for (auto& buffer : shadow_uniform_buffers_)
        destroy_buffer(buffer);
    shadow_uniform_buffers_.clear();
    shadow_uniform_buffers_.resize(count);

    for (auto& buffer : shadow_uniform_buffers_)
    {
        if (!create_buffer(sizeof(shadow_uniform_data), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                           buffer))
            return false;
    }
    return true;
}

vulkan_render_backend::gpu_buffer* vulkan_render_backend::current_shadow_uniform_buffer() noexcept
{
    const auto slot = current_frame_slot();
    if (slot >= shadow_uniform_buffers_.size()) return nullptr;
    return &shadow_uniform_buffers_[slot];
}

bool vulkan_render_backend::update_debug_overlay_buffer()
{
    const auto count = frame_resource_count();
    if (debug_overlay_buffers_.size() != count)
    {
        wait_for_in_flight_frames();
        for (auto& buffer : debug_overlay_buffers_)
            destroy_buffer(buffer.vertices);
        debug_overlay_buffers_.clear();
        debug_overlay_buffers_.resize(count);
    }
    auto& target = debug_overlay_buffers_[current_frame_slot()];
    std::vector<debug_overlay_vertex> vertices;
    vertices.reserve(frame_debug_overlay_lines_.size() * 2u + frame_debug_overlay_triangles_.size() * 3u);
    const auto append_lines = [&](debug_overlay_depth_mode mode)
    {
        for (const auto& line : frame_debug_overlay_lines_)
        {
            if (line.depth != mode) continue;
            vertices.push_back({line.start, line.color});
            vertices.push_back({line.end, line.color});
        }
    };
    const auto append_triangles = [&](debug_overlay_depth_mode mode)
    {
        for (const auto& triangle : frame_debug_overlay_triangles_)
        {
            if (triangle.depth != mode) continue;
            vertices.push_back({triangle.first, triangle.color});
            vertices.push_back({triangle.second, triangle.color});
            vertices.push_back({triangle.third, triangle.color});
        }
    };
    const auto append_range =
        [&](auto&& append, debug_overlay_depth_mode mode, std::uint32_t& offset, std::uint32_t& count)
    {
        offset = static_cast<std::uint32_t>(vertices.size());
        append(mode);
        count = static_cast<std::uint32_t>(vertices.size()) - offset;
    };
    append_range(append_lines, debug_overlay_depth_mode::tested, target.tested_line_offset, target.tested_line_count);
    append_range(append_triangles, debug_overlay_depth_mode::tested, target.tested_triangle_offset,
                 target.tested_triangle_count);
    append_range(append_lines, debug_overlay_depth_mode::always, target.output_line_offset, target.output_line_count);
    append_range(append_triangles, debug_overlay_depth_mode::always, target.output_triangle_offset,
                 target.output_triangle_count);
    if (vertices.empty()) return true;
    const VkDeviceSize bytes = vertices.size() * sizeof(debug_overlay_vertex);
    if (target.capacity < bytes)
    {
        destroy_buffer(target.vertices);
        target.capacity = std::max<VkDeviceSize>(4096u, std::bit_ceil(static_cast<std::uint64_t>(bytes)));
        if (!create_buffer(target.capacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                           target.vertices))
        {
            target.capacity = 0;
            return false;
        }
    }
    void* mapped{};
    if (vmaMapMemory(allocator_, target.vertices.allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(bytes));
    vmaFlushAllocation(allocator_, target.vertices.allocation, 0, bytes);
    vmaUnmapMemory(allocator_, target.vertices.allocation);
    return true;
}

const vulkan_render_backend::gpu_buffer*
vulkan_render_backend::shadow_uniform_buffer_for_slot(std::uint32_t slot) const noexcept
{
    if (slot >= shadow_uniform_buffers_.size()) return nullptr;
    return &shadow_uniform_buffers_[slot];
}

bool vulkan_render_backend::ensure_shadow_resources(const shadow_settings& settings)
{
    const std::uint32_t resolution = std::clamp(settings.resolution, 256u, 8192u);
    if (shadow_atlas_.image != VK_NULL_HANDLE && shadow_atlas_.resolution == resolution) return true;

    wait_for_in_flight_frames();
    destroy_shadow_resources();

    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = depth_format_;
    image.extent = {resolution, resolution, 1};
    image.mipLevels = 1;
    image.arrayLayers = directional_shadow_layer_count;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    VmaAllocationCreateInfo allocation{};
    allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(allocator_, &image, &allocation, &shadow_atlas_.image, &shadow_atlas_.allocation, nullptr) !=
        VK_SUCCESS)
    {
        arc::diagnostics::warn("render.vulkan", "Failed to allocate directional shadow atlas");
        return false;
    }

    VkImageViewCreateInfo array_view{};
    array_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    array_view.image = shadow_atlas_.image;
    array_view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    array_view.format = depth_format_;
    array_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    array_view.subresourceRange.levelCount = 1;
    array_view.subresourceRange.layerCount = directional_shadow_layer_count;
    if (vkCreateImageView(device_, &array_view, nullptr, &shadow_atlas_.array_view) != VK_SUCCESS)
    {
        destroy_shadow_resources();
        return false;
    }

    for (std::uint32_t layer = 0; layer < directional_shadow_layer_count; ++layer)
    {
        VkImageViewCreateInfo layer_view = array_view;
        layer_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        layer_view.subresourceRange.baseArrayLayer = layer;
        layer_view.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device_, &layer_view, nullptr, &shadow_atlas_.cascade_views[layer]) != VK_SUCCESS)
        {
            destroy_shadow_resources();
            return false;
        }
    }

    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sampler.compareEnable = VK_TRUE;
    sampler.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (vkCreateSampler(device_, &sampler, nullptr, &shadow_atlas_.sampler) != VK_SUCCESS)
    {
        destroy_shadow_resources();
        return false;
    }

    shadow_atlas_.resolution = resolution;
    shadow_atlas_.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

const directional_light_event* vulkan_render_backend::active_directional_shadow_light() const noexcept
{
    if (!frame_shadows_enabled_) return nullptr;
    for (const auto& light : frame_directional_lights_)
    {
        if (light.enabled && light.casts_shadows && light.shadow.enabled) return &light;
    }
    return nullptr;
}

void vulkan_render_backend::execute_compiled_graph(VkCommandBuffer command_buffer)
{
    bool directional_shadows_executed{};
    bool viewport_executed{};
    bool scene_executed{};
    bool point_shadows_executed{};
    bool spot_shadows_executed{};
    bool gpu_visibility_executed{};
    bool virtual_shadow_pages_published{};
    frame_fxaa_enabled_ = std::any_of(last_profile_.graph.passes.begin(), last_profile_.graph.passes.end(),
                                      [](const auto& pass) { return pass.builtin == builtin_render_pass::fxaa; });

    dispatch_texture_mip_feedback(command_buffer);

    for (const auto& pass : last_profile_.graph.passes)
    {
        const auto scope = begin_gpu_scope(command_buffer, pass.name.c_str());
        switch (pass.builtin)
        {
            case builtin_render_pass::virtual_shadow_page_marking:
                prepare_virtual_shadow_cache(last_profile_.frame_index);
                break;
            case builtin_render_pass::virtual_shadow_static_render:
                clear_virtual_shadow_render_pages(command_buffer, virtual_shadow_page_layer::static_depth);
                break;
            case builtin_render_pass::virtual_shadow_dynamic_render:
                clear_virtual_shadow_render_pages(command_buffer, virtual_shadow_page_layer::dynamic_depth);
                break;
            case builtin_render_pass::virtual_shadow_page_table_publication:
                publish_virtual_shadow_pages(command_buffer);
                virtual_shadow_pages_published = true;
                break;
            case builtin_render_pass::directional_shadow_static:
            case builtin_render_pass::directional_shadow_dynamic:
                if (!directional_shadows_executed)
                {
                    render_shadow_maps(command_buffer);
                    directional_shadows_executed = true;
                }
                break;
            case builtin_render_pass::point_shadow:
                if (!point_shadows_executed)
                {
                    render_local_shadow_maps(command_buffer, shadow_light_kind::point);
                    point_shadows_executed = true;
                }
                break;
            case builtin_render_pass::spot_shadow:
                if (!spot_shadows_executed)
                {
                    render_local_shadow_maps(command_buffer, shadow_light_kind::spot);
                    spot_shadows_executed = true;
                }
                break;
            case builtin_render_pass::gpu_frustum_distance_cull:
                if (!gpu_visibility_executed)
                {
                    dispatch_gpu_visibility(command_buffer);
                    dispatch_virtual_geometry_traversal(command_buffer);
                    gpu_visibility_executed = true;
                }
                break;
            case builtin_render_pass::gpu_skinning:
                dispatch_gpu_skinning(command_buffer);
                break;
            case builtin_render_pass::water_spectrum_update:
                dispatch_water_spectrum_update(command_buffer);
                break;
            case builtin_render_pass::water_inverse_fft:
                dispatch_water_inverse_fft(command_buffer);
                break;
            case builtin_render_pass::water_foam_update:
                dispatch_water_foam_update(command_buffer);
                break;
            case builtin_render_pass::depth_prepass:
                if (!scene_executed)
                {
                    // The current Vulkan raster path records depth and material outputs
                    // together. Starting it at the graph's depth boundary makes the
                    // produced depth available to the following HZB pass.
                    render_viewport(command_buffer, true, false);
                    scene_executed = true;
                }
                break;
            case builtin_render_pass::depth_pyramid:
                dispatch_hzb(command_buffer);
                break;
            case builtin_render_pass::velocity_dilation:
                dispatch_velocity_dilation(command_buffer);
                break;
            case builtin_render_pass::reactive_mask:
                dispatch_temporal_masks(command_buffer);
                break;
            case builtin_render_pass::temporal_antialiasing:
            case builtin_render_pass::temporal_upscale:
                dispatch_temporal_resolve(command_buffer);
                break;
            case builtin_render_pass::spatial_sharpen:
                dispatch_temporal_sharpen(command_buffer);
                break;
            case builtin_render_pass::gbuffer:
            case builtin_render_pass::forward_opaque:
                if (!scene_executed)
                {
                    render_viewport(command_buffer, true, false);
                    scene_executed = true;
                }
                break;
            case builtin_render_pass::output_transform:
                // FXAA is folded into output conversion so it can filter
                // tone-mapped linear color immediately before the single
                // sRGB conversion. Its graph pass remains the execution
                // boundary for this fused implementation.
                if (frame_fxaa_enabled_) break;
                [[fallthrough]];
            case builtin_render_pass::fxaa:
                if (!viewport_executed)
                {
                    if (!scene_executed)
                    {
                        render_viewport(command_buffer, true, false);
                        scene_executed = true;
                    }
                    render_viewport(command_buffer, false, true);
                    viewport_executed = true;
                }
                break;
            default:
                break;
        }
        end_gpu_scope(command_buffer, scope);
    }

    if (!directional_shadows_executed)
        transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    if (!point_shadows_executed && !spot_shadows_executed)
        transition_local_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    if (resolved_config_.features.virtual_shadow_maps && !virtual_shadow_pages_published)
    {
        transition_virtual_shadow_image(command_buffer, virtual_shadow_resources_.static_image,
                                        virtual_shadow_resources_.static_layout,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        transition_virtual_shadow_image(command_buffer, virtual_shadow_resources_.dynamic_image,
                                        virtual_shadow_resources_.dynamic_layout,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }
}

void vulkan_render_backend::prepare_frame_gpu_resources()
{
    update_dynamic_mesh_vertices();
    if (!resolved_config_.features.gpu_skinning || !ensure_gpu_skinning_pipeline()) update_cpu_skinned_vertices();
    if (!virtual_meshes_.empty() && !ensure_virtual_geometry_raster_resources())
        last_profile_.virtual_geometry.fallback_reason =
            "virtual-geometry software visibility resources are unavailable; using conventional LODs";
    const auto* light = active_directional_shadow_light();
    auto settings = light ? light->shadow : shadow_settings{.enabled = false, .resolution = 2048};
    settings.resolution =
        std::min(std::bit_ceil(std::max(settings.resolution, 1u)), resolved_config_.directional_shadow_resolution);
    if (ensure_shadow_uniform_buffers() && ensure_shadow_resources(settings))
    {
        update_shadow_uniform(build_shadow_uniform(light));
        update_gbuffer_descriptor_set();
    }
    if (!ensure_local_shadow_resources() && !active_local_shadows_.empty())
    {
        last_profile_.shadows.fallback_reason =
            "local shadow atlas allocation failed; affected lights render unshadowed";
        active_local_shadows_.clear();
        frame_lighting_.local_shadow_face_count = 0u;
        update_light_buffer();
    }
    else
        update_gbuffer_descriptor_set();

    if (ensure_mesh_pipeline()) update_current_material_descriptor_sets();

    if ((light && !frame_shadow_draws_.empty()) || !active_local_shadows_.empty()) ensure_shadow_pipeline();
    const auto clear_overlay_counts = [&]
    {
        const auto slot = current_frame_slot();
        if (slot < debug_overlay_buffers_.size())
        {
            auto& buffer = debug_overlay_buffers_[slot];
            buffer.tested_line_count = 0;
            buffer.tested_triangle_count = 0;
            buffer.output_line_count = 0;
            buffer.output_triangle_count = 0;
        }
    };
    if (!frame_debug_overlay_lines_.empty() || !frame_debug_overlay_triangles_.empty())
    {
        if (!ensure_debug_overlay_pipeline() || !update_debug_overlay_buffer()) clear_overlay_counts();
    }
    else
        clear_overlay_counts();
}

void vulkan_render_backend::transition_shadow_atlas(VkCommandBuffer command_buffer, VkImageLayout new_layout)
{
    if (shadow_atlas_.image == VK_NULL_HANDLE || shadow_atlas_.layout == new_layout) return;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = shadow_atlas_.layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = shadow_atlas_.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = directional_shadow_layer_count;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dst_stage =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    if (shadow_atlas_.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (shadow_atlas_.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    }

    if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = shadow_atlas_.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                                      : VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }

    vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    shadow_atlas_.layout = new_layout;
}

void vulkan_render_backend::transition_local_shadow_atlas(VkCommandBuffer command_buffer, VkImageLayout new_layout)
{
    if (local_shadow_atlas_.image == VK_NULL_HANDLE || local_shadow_atlas_.layout == new_layout) return;
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = local_shadow_atlas_.layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = local_shadow_atlas_.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    VkPipelineStageFlags source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destination_stage =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    if (local_shadow_atlas_.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (local_shadow_atlas_.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        source_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    }
    if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
        barrier.dstAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    local_shadow_atlas_.layout = new_layout;
}

shadow_uniform_data vulkan_render_backend::build_shadow_uniform(const directional_light_event* light) const noexcept
{
    shadow_uniform_data data{};
    const auto identity = math::identity<float, 4>();
    for (auto& matrix : data.light_view_projection)
        std::copy(identity.data(), identity.data() + 16, matrix);

    if (!light)
    {
        data.params[0] = 0.0f;
        return data;
    }

    auto cascade_settings = light->cascades;
    cascade_settings.cascade_count =
        std::min(cascade_settings.cascade_count, resolved_config_.directional_shadow_cascades);
    cascade_settings.maximum_distance =
        std::min(cascade_settings.maximum_distance, resolved_config_.directional_shadow_distance);
    const auto layout =
        fit_directional_shadow_cascades({.inverse_view_projection = frame_camera_.inverse_view_projection,
                                         .near_plane = frame_camera_.near_plane,
                                         .far_plane = frame_camera_.far_plane},
                                        light->direction, cascade_settings, shadow_atlas_.resolution);

    for (std::uint32_t cascade = 0; cascade < layout.cascade_count; ++cascade)
    {
        const auto& fitted = layout.cascades[cascade];
        std::copy(fitted.light_view_projection.data(), fitted.light_view_projection.data() + 16,
                  data.light_view_projection[cascade]);
        data.cascade_splits[cascade] = fitted.split_depth;
        data.cascade_blend_starts[cascade] = fitted.blend_start_depth;
        data.cascade_texel_size[cascade] = fitted.texel_world_size;
    }

    data.params[0] = std::clamp(light->shadow.strength, 0.0f, 1.0f);
    data.params[1] = std::max(0.0f, light->shadow.bias);
    data.params[2] = std::max(0.0f, light->shadow.normal_bias);
    const auto filter = resolved_config_.quality == render_quality_tier::low
                            ? shadow_filter::pcf_3x3
                            : static_cast<shadow_filter>(std::min(static_cast<unsigned>(light->shadow.filter),
                                                                  static_cast<unsigned>(shadow_filter::pcf_5x5)));
    data.params[3] = static_cast<float>(filter);
    data.configuration[0] = static_cast<float>(layout.cascade_count);
    // Cascade splits are authored in camera view depth, not radial
    // distance. The remaining configuration lanes carry the normalized
    // camera forward vector so every lighting path selects the same
    // frustum slice without expanding the Vulkan 1.2-safe uniform.
    data.configuration[1] = frame_camera_.forward[0];
    data.configuration[2] = frame_camera_.forward[1];
    data.configuration[3] = frame_camera_.forward[2];
    return data;
}

void vulkan_render_backend::update_shadow_uniform(const shadow_uniform_data& data)
{
    if (!ensure_shadow_uniform_buffers()) return;
    auto* shadow_buffer = current_shadow_uniform_buffer();
    if (shadow_buffer == nullptr || shadow_buffer->buffer == VK_NULL_HANDLE) return;
    void* mapped{};
    if (vmaMapMemory(allocator_, shadow_buffer->allocation, &mapped) != VK_SUCCESS) return;
    std::memcpy(mapped, &data, sizeof(data));
    vmaFlushAllocation(allocator_, shadow_buffer->allocation, 0, sizeof(data));
    vmaUnmapMemory(allocator_, shadow_buffer->allocation);
}

bool vulkan_render_backend::ensure_shadow_pipeline()
{
    if (shadow_pipeline_ != VK_NULL_HANDLE) return true;
    if (max_push_constant_bytes_ < sizeof(mesh_push_constants) || !ensure_mesh_pipeline()) return false;

    VkShaderModule vert =
        create_shader_module(builtin::shadow_depth_vert_spv, std::size(builtin::shadow_depth_vert_spv));
    VkShaderModule frag =
        create_shader_module(builtin::shadow_depth_frag_spv, std::size(builtin::shadow_depth_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
    {
        if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

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
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &shadow_pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
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
    std::array<VkVertexInputAttributeDescription, 2> attributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)},
        VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)}};

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
    raster.depthBiasEnable = VK_TRUE;
    raster.depthBiasConstantFactor = 1.25f;
    raster.depthBiasSlopeFactor = 1.75f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.depthAttachmentFormat = depth_format_;

    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = shadow_pipeline_layout_;
    pipeline.renderPass = VK_NULL_HANDLE;

    const VkResult result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &shadow_pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (result != VK_SUCCESS)
    {
        arc::diagnostics::warn("render.vulkan",
                               "Vulkan shadow pipeline creation failed; rendering will continue without shadows");
        return false;
    }
    return true;
}

void vulkan_render_backend::render_shadow_maps(VkCommandBuffer command_buffer)
{
    const auto* light = active_directional_shadow_light();
    const shadow_settings settings = light ? light->shadow : shadow_settings{.enabled = false, .resolution = 2048};
    if (shadow_atlas_.image == VK_NULL_HANDLE) return;
    const auto uniform = build_shadow_uniform(light);

    if (!light || shadow_pipeline_ == VK_NULL_HANDLE)
    {
        transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        return;
    }

    transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    const float resolution = static_cast<float>(shadow_atlas_.resolution);
    VkViewport viewport{};
    viewport.width = resolution;
    viewport.height = resolution;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{};
    scissor.extent = {shadow_atlas_.resolution, shadow_atlas_.resolution};

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    const auto cascade_count = static_cast<std::uint32_t>(
        std::clamp(uniform.configuration[0], 1.0f, static_cast<float>(directional_shadow_cascade_count)));
    const auto within_shadow_distance = [&](const draw_mesh_event& draw)
    {
        if (draw.maximum_shadow_distance <= 0.0f) return true;
        // Draw events intentionally stay compact and do not duplicate the
        // render world's bounds. The model origin is a conservative,
        // stable distance proxy until shadow draws carry their selected
        // shadow LOD's tight bounds.
        const float x = draw.model(0, 3) - frame_camera_.position[0];
        const float y = draw.model(1, 3) - frame_camera_.position[1];
        const float z = draw.model(2, 3) - frame_camera_.position[2];
        return x * x + y * y + z * z <= draw.maximum_shadow_distance * draw.maximum_shadow_distance;
    };
    const auto is_static_caster = [&](const draw_mesh_event& draw)
    { return draw.casts_shadows && within_shadow_distance(draw) && draw.mobility == render_mobility::static_object; };
    const auto is_dynamic_caster = [&](const draw_mesh_event& draw)
    {
        if (!draw.casts_shadows || !within_shadow_distance(draw)) return false;
        if (light->mobility == render_mobility::static_object) return false;
        if (light->mobility == render_mobility::movable) return true;
        return draw.mobility != render_mobility::static_object;
    };
    const auto intersects_cascade = [](const draw_mesh_event& draw, const math::matrix4f& matrix)
    {
        const auto bounds_size = geometric::size(draw.world_bounds);
        if (math::length_squared(bounds_size) <= 1.0e-8f) return true;

        bool outside_left = true;
        bool outside_right = true;
        bool outside_bottom = true;
        bool outside_top = true;
        bool outside_near = true;
        bool outside_far = true;
        for (std::uint32_t corner = 0; corner < 8u; ++corner)
        {
            const float x = (corner & 1u) ? draw.world_bounds.max[0] : draw.world_bounds.min[0];
            const float y = (corner & 2u) ? draw.world_bounds.max[1] : draw.world_bounds.min[1];
            const float z = (corner & 4u) ? draw.world_bounds.max[2] : draw.world_bounds.min[2];
            const float clip_x = matrix(0, 0) * x + matrix(0, 1) * y + matrix(0, 2) * z + matrix(0, 3);
            const float clip_y = matrix(1, 0) * x + matrix(1, 1) * y + matrix(1, 2) * z + matrix(1, 3);
            const float clip_z = matrix(2, 0) * x + matrix(2, 1) * y + matrix(2, 2) * z + matrix(2, 3);
            const float clip_w = matrix(3, 0) * x + matrix(3, 1) * y + matrix(3, 2) * z + matrix(3, 3);
            outside_left &= clip_x < -clip_w;
            outside_right &= clip_x > clip_w;
            outside_bottom &= clip_y < -clip_w;
            outside_top &= clip_y > clip_w;
            outside_near &= clip_z < 0.0f;
            outside_far &= clip_z > clip_w;
        }
        return !(outside_left || outside_right || outside_bottom || outside_top || outside_near || outside_far);
    };

    std::uint64_t static_signature = 1469598103934665603ull;
    const auto hash_bytes = [&](const void* bytes, std::size_t count)
    {
        const auto* data = static_cast<const std::byte*>(bytes);
        for (std::size_t index = 0; index < count; ++index)
        {
            static_signature ^= static_cast<std::uint64_t>(std::to_integer<unsigned char>(data[index]));
            static_signature *= 1099511628211ull;
        }
    };
    hash_bytes(&uniform, sizeof(uniform));
    hash_bytes(&shadow_resource_revision_, sizeof(shadow_resource_revision_));
    for (const auto& draw : frame_shadow_draws_)
    {
        if (!is_static_caster(draw)) continue;
        hash_bytes(&draw.object_id, sizeof(draw.object_id));
        hash_bytes(draw.model.data(), sizeof(float) * 16u);
        hash_bytes(&draw.mesh, sizeof(draw.mesh));
        hash_bytes(&draw.material, sizeof(draw.material));
    }
    for (const auto& draw : frame_virtual_shadow_draws_)
    {
        if (!is_static_caster(draw.draw)) continue;
        hash_bytes(&draw.draw.object_id, sizeof(draw.draw.object_id));
        hash_bytes(draw.draw.model.data(), sizeof(float) * 16u);
        hash_bytes(&draw.mesh, sizeof(draw.mesh));
        hash_bytes(&draw.cluster_index, sizeof(draw.cluster_index));
        hash_bytes(&draw.draw.material, sizeof(draw.draw.material));
    }
    const bool redraw_static =
        !shadow_cache_.static_layers_valid || shadow_cache_.static_signature != static_signature ||
        light->mobility == render_mobility::movable || settings.cache_mode == shadow_cache_mode::always_update;
    last_static_shadow_cache_hit_ = !redraw_static;

    const auto render_layers = [&](std::uint32_t layer_offset, const auto& accepts_draw, bool enabled)
    {
        for (std::uint32_t cascade = 0; cascade < cascade_count; ++cascade)
        {
            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = shadow_atlas_.cascade_views[layer_offset + cascade];
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil.depth = 1.0f;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.extent = {shadow_atlas_.resolution, shadow_atlas_.resolution};
            rendering.layerCount = 1;
            rendering.pDepthAttachment = &depth_attachment;
            cmd_begin_rendering(command_buffer, &rendering);

            if (enabled)
            {
                math::matrix4f cascade_matrix;
                std::copy(uniform.light_view_projection[cascade], uniform.light_view_projection[cascade] + 16,
                          cascade_matrix.data());
                for (const auto& draw : frame_shadow_draws_)
                {
                    if (!accepts_draw(draw) || !intersects_cascade(draw, cascade_matrix)) continue;

                    auto found = meshes_.find(resource_key(draw.mesh));
                    if (found == meshes_.end()) continue;

                    const math::matrix4f mvp = math::matmul(cascade_matrix, draw.model);
                    mesh_push_constants constants = build_mesh_constants(draw);
                    std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                    VkDescriptorSet descriptor_set = material_descriptor_set_for(draw);
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0,
                                            1, &descriptor_set, 0, nullptr);
                    vkCmdPushConstants(command_buffer, shadow_pipeline_layout_,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                       &constants);

                    const VkDeviceSize offset = 0;
                    const VkBuffer vertex_buffer = mesh_vertex_buffer(found->second, draw.gpu_scene_instance);
                    if (vertex_buffer == VK_NULL_HANDLE) continue;
                    vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
                    vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(command_buffer, found->second.index_count, 1, 0, 0, 0);
                }
                for (const auto& draw : frame_virtual_shadow_draws_)
                {
                    if (!accepts_draw(draw.draw) || !intersects_cascade(draw.draw, cascade_matrix)) continue;
                    const auto found = virtual_meshes_.find(resource_key(draw.mesh));
                    if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size()) continue;
                    const auto& cluster = found->second.clusters[draw.cluster_index];
                    if (cluster.index_count == 0 ||
                        cluster.first_index + cluster.index_count > found->second.index_count)
                        continue;

                    const math::matrix4f mvp = math::matmul(cascade_matrix, draw.draw.model);
                    mesh_push_constants constants = build_mesh_constants(draw.draw);
                    std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                    VkDescriptorSet descriptor_set = material_descriptor_set_for(draw.draw);
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0,
                                            1, &descriptor_set, 0, nullptr);
                    vkCmdPushConstants(command_buffer, shadow_pipeline_layout_,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                       &constants);
                    const VkDeviceSize offset = 0;
                    vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                    vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
                }
            }
            cmd_end_rendering(command_buffer);
        }
    };

    if (redraw_static)
    {
        render_layers(0u, is_static_caster, light->mobility != render_mobility::movable);
        shadow_cache_.static_signature = static_signature;
        shadow_cache_.static_layers_valid = true;
    }
    render_layers(directional_shadow_cascade_count, is_dynamic_caster, true);

    transition_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    shadow_cache_.last_directional_key = {
        .light_index = 0, .resolution = shadow_atlas_.resolution, .filter = settings.filter};
    shadow_cache_.has_directional_key = true;
}

void vulkan_render_backend::render_local_shadow_maps(VkCommandBuffer command_buffer, shadow_light_kind requested_kind)
{
    if (active_local_shadows_.empty() || local_shadow_atlas_.image == VK_NULL_HANDLE ||
        shadow_pipeline_ == VK_NULL_HANDLE)
        return;

    transition_local_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);

    std::uint32_t packed_face_index{};
    for (const auto& shadow : active_local_shadows_)
    {
        if (shadow.kind != requested_kind)
        {
            packed_face_index += shadow.allocation.face_count;
            continue;
        }
        if (!shadow.redraw)
        {
            packed_face_index += shadow.allocation.face_count;
            continue;
        }

        const auto in_light_range = [&](const draw_mesh_event& draw)
        {
            const auto bounds_size = geometric::size(draw.world_bounds);
            math::vector3f center = matrix_translation(draw.model);
            if (math::length_squared(bounds_size) > 1.0e-8f)
            {
                const auto bounds_center = geometric::center(draw.world_bounds);
                center = {bounds_center[0], bounds_center[1], bounds_center[2]};
            }
            return math::length_squared(math::sub(center, shadow.position)) <= shadow.range * shadow.range;
        };
        for (std::uint32_t face = 0; face < shadow.allocation.face_count; ++face)
        {
            if (packed_face_index >= frame_lighting_.local_shadow_face_count) break;
            const auto& packed = frame_lighting_.local_shadow_faces[packed_face_index++];
            const auto& rect = shadow.allocation.faces[face];
            VkViewport viewport{};
            viewport.x = static_cast<float>(rect.content_x());
            viewport.y = static_cast<float>(rect.content_y());
            viewport.width = static_cast<float>(rect.content_size());
            viewport.height = static_cast<float>(rect.content_size());
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.offset = {static_cast<std::int32_t>(rect.content_x()), static_cast<std::int32_t>(rect.content_y())};
            scissor.extent = {rect.content_size(), rect.content_size()};
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);

            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = local_shadow_atlas_.view;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.offset = {static_cast<std::int32_t>(rect.x), static_cast<std::int32_t>(rect.y)};
            rendering.renderArea.extent = {rect.size, rect.size};
            rendering.layerCount = 1;
            rendering.pDepthAttachment = &depth_attachment;
            cmd_begin_rendering(command_buffer, &rendering);

            const auto draw_mesh = [&](const draw_mesh_event& draw, const gpu_mesh& mesh)
            {
                if (!draw.casts_shadows || !in_light_range(draw) ||
                    (shadow.mobility == render_mobility::static_object &&
                     draw.mobility != render_mobility::static_object))
                    return;
                mesh_push_constants constants = build_mesh_constants(draw);
                const auto mvp = math::matmul(packed.light_view_projection, draw.model);
                std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                VkDescriptorSet descriptor_set = material_descriptor_set_for(draw);
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0, 1,
                                        &descriptor_set, 0, nullptr);
                vkCmdPushConstants(command_buffer, shadow_pipeline_layout_,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                   &constants);
                const VkDeviceSize offset{};
                const auto vertex_buffer = mesh_vertex_buffer(mesh, draw.gpu_scene_instance);
                if (vertex_buffer == VK_NULL_HANDLE) return;
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
                vkCmdBindIndexBuffer(command_buffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
            };
            for (const auto& draw : frame_shadow_draws_)
            {
                const auto found = meshes_.find(resource_key(draw.mesh));
                if (found != meshes_.end()) draw_mesh(draw, found->second);
            }
            for (const auto& draw : frame_virtual_shadow_draws_)
            {
                if (!draw.draw.casts_shadows || !in_light_range(draw.draw) ||
                    (shadow.mobility == render_mobility::static_object &&
                     draw.draw.mobility != render_mobility::static_object))
                    continue;
                const auto found = virtual_meshes_.find(resource_key(draw.mesh));
                if (found == virtual_meshes_.end() || draw.cluster_index >= found->second.clusters.size()) continue;
                const auto& cluster = found->second.clusters[draw.cluster_index];
                mesh_push_constants constants = build_mesh_constants(draw.draw);
                const auto mvp = math::matmul(packed.light_view_projection, draw.draw.model);
                std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
                VkDescriptorSet descriptor_set = material_descriptor_set_for(draw.draw);
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout_, 0, 1,
                                        &descriptor_set, 0, nullptr);
                vkCmdPushConstants(command_buffer, shadow_pipeline_layout_,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants),
                                   &constants);
                const VkDeviceSize offset{};
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &found->second.vertices.buffer, &offset);
                vkCmdBindIndexBuffer(command_buffer, found->second.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(command_buffer, cluster.index_count, 1, cluster.first_index, 0, 0);
            }
            cmd_end_rendering(command_buffer);
        }
    }
    transition_local_shadow_atlas(command_buffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
}

} // namespace arc::render::vulkan::backend_detail
