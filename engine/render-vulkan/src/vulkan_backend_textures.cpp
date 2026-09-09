#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"

namespace arc::render::vulkan::backend_detail
{
void vulkan_render_backend::defer_texture_release(gpu_texture texture)
{
    if (texture.image == VK_NULL_HANDLE && texture.view == VK_NULL_HANDLE && texture.sampler == VK_NULL_HANDLE) return;
    deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                             [this, texture = std::move(texture)]() mutable { destroy_texture(texture); });
}

void vulkan_render_backend::collect_texture_feedback_slots()
{
    for (std::size_t index = 0; index < retired_texture_feedback_slots_.size();)
    {
        if (retired_texture_feedback_slots_[index].reuse_after_frame > last_completed_frame_)
        {
            ++index;
            continue;
        }
        free_texture_feedback_slots_.push_back(retired_texture_feedback_slots_[index].slot);
        retired_texture_feedback_slots_[index] = retired_texture_feedback_slots_.back();
        retired_texture_feedback_slots_.pop_back();
    }
}

std::uint32_t vulkan_render_backend::allocate_texture_feedback_slot(texture_handle resource,
                                                                    std::uint32_t content_generation,
                                                                    std::uint32_t mip_count)
{
    collect_texture_feedback_slots();
    std::uint32_t index{};
    if (!free_texture_feedback_slots_.empty())
    {
        index = free_texture_feedback_slots_.back();
        free_texture_feedback_slots_.pop_back();
        auto& slot = texture_feedback_slots_[index];
        slot.slot_generation =
            slot.slot_generation == std::numeric_limits<std::uint32_t>::max() ? 1u : slot.slot_generation + 1u;
    }
    else
    {
        index = static_cast<std::uint32_t>(texture_feedback_slots_.size());
        texture_feedback_slots_.push_back({});
    }
    auto& slot = texture_feedback_slots_[index];
    slot.resource = resource;
    slot.content_generation = content_generation;
    slot.mip_count = mip_count;
    slot.active = true;
    return index;
}

void vulkan_render_backend::retire_texture_feedback_slot(std::uint32_t slot)
{
    if (slot >= texture_feedback_slots_.size() || !texture_feedback_slots_[slot].active) return;
    texture_feedback_slots_[slot].active = false;
    texture_feedback_slots_[slot].resource = {};
    retired_texture_feedback_slots_.push_back(
        {.slot = slot, .reuse_after_frame = last_profile_.frame_index + frame_resource_count()});
}

void vulkan_render_backend::retire_texture(texture_handle handle)
{
    const auto found = textures_.find(resource_key(handle));
    if (found == textures_.end()) return;
    retire_texture_feedback_slot(found->second.feedback_slot);
    retire_virtual_texture(found->second);
    auto retired = std::move(found->second);
    textures_.erase(found);
    defer_texture_release(std::move(retired));
    virtual_geometry_material_descriptors_dirty_ = true;
    gpu_bindless_descriptors_dirty_ = true;
}

void vulkan_render_backend::register_streamed_texture(const texture_stream_register_event& event)
{
    if (!event.descriptor) return;
    gpu_texture texture;
    texture.handle = event.handle;
    texture.streaming = *event.descriptor;
    texture.streamable = true;
    texture.mip_window_base = texture.streaming.artifact.mip_count;
    texture.streamed_mips.resize(texture.streaming.artifact.mip_count);
    texture.data.name = texture.streaming.texture.name;
    texture.data.width = texture.streaming.texture.width;
    texture.data.height = texture.streaming.texture.height;
    texture.data.depth = texture.streaming.texture.depth;
    texture.data.dimension = texture.streaming.texture.dimension;
    texture.data.format = texture.streaming.texture.format;
    texture.data.color_space = texture.streaming.texture.color_space;
    texture.data.semantic = texture.streaming.texture.semantic;
    texture.data.mip_levels = texture.streaming.texture.mip_levels;
    texture.feedback_slot = allocate_texture_feedback_slot(event.handle, texture.streaming.content_generation,
                                                           texture.streaming.artifact.mip_count);
    const auto key = resource_key(event.handle);
    if (const auto found = textures_.find(key); found != textures_.end())
    {
        retire_texture_feedback_slot(found->second.feedback_slot);
        retire_virtual_texture(found->second);
        defer_texture_release(std::move(found->second));
    }
    register_virtual_texture(texture);
    textures_[key] = std::move(texture);
    virtual_geometry_material_descriptors_dirty_ = true;
    gpu_bindless_descriptors_dirty_ = true;
}

void vulkan_render_backend::update_gpu_texture_table_window(const gpu_texture& texture)
{
    if (!texture.handle.valid()) return;
    auto& table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::texture)];
    if (table.element_stride != sizeof(gpu_texture_table_record) || texture.handle.index >= table.live.size() ||
        !table.live[texture.handle.index] || table.generations[texture.handle.index] != texture.handle.generation)
        return;
    const auto offset = static_cast<std::size_t>(texture.handle.index) * table.element_stride;
    if (offset > table.mirror.size() || sizeof(gpu_texture_table_record) > table.mirror.size() - offset) return;
    gpu_texture_table_record record{};
    std::memcpy(&record, table.mirror.data() + offset, sizeof(record));
    record.mip_window_base = texture.mip_window_base;
    record.mip_count = texture.mip_count;
    if (++record.descriptor_generation == 0u) record.descriptor_generation = 1u;
    std::memcpy(table.mirror.data() + offset, &record, sizeof(record));
    table.dirty = true;
    last_profile_.gpu_scene.uploaded_bytes += sizeof(record);
    ++last_profile_.gpu_scene.uploaded_ranges;
}

bool vulkan_render_backend::rebuild_streamed_mip_window(gpu_texture& texture)
{
    if (!texture.streamable || texture.streamed_mips.empty()) return false;
    std::uint32_t base = static_cast<std::uint32_t>(texture.streamed_mips.size());
    for (std::uint32_t mip = static_cast<std::uint32_t>(texture.streamed_mips.size()); mip > 0; --mip)
    {
        if (!texture.streamed_mips[mip - 1]) break;
        base = mip - 1;
    }
    if (base == texture.streamed_mips.size()) return false;
    if (base == texture.mip_window_base && texture.image != VK_NULL_HANDLE) return true;

    texture_data data;
    data.name = texture.streaming.texture.name;
    data.width = std::max(1u, texture.streaming.texture.width >> base);
    data.height = std::max(1u, texture.streaming.texture.height >> base);
    data.depth = 1;
    data.dimension = texture_dimension::texture_2d;
    data.format = texture.streaming.texture.format;
    data.color_space = texture.streaming.texture.color_space;
    data.semantic = texture.streaming.texture.semantic;
    data.array_layers = 1;
    data.mip_levels = static_cast<std::uint32_t>(texture.streamed_mips.size()) - base;
    data.mips.reserve(data.mip_levels);
    std::size_t offset{};
    for (std::uint32_t mip = base; mip < texture.streamed_mips.size(); ++mip)
    {
        const auto& bytes = texture.streamed_mips[mip];
        if (!bytes) return false;
        const auto& artifact_mip = texture.streaming.artifact.mips[mip];
        data.mips.push_back(
            {.width = artifact_mip.width, .height = artifact_mip.height, .offset = offset, .size = bytes->size()});
        data.encoded.insert(data.encoded.end(), bytes->begin(), bytes->end());
        offset += bytes->size();
    }

    gpu_texture replacement;
    replacement.streaming = texture.streaming;
    replacement.handle = texture.handle;
    replacement.streamed_mips = texture.streamed_mips;
    replacement.streamable = true;
    replacement.mip_window_base = base;
    replacement.feedback_slot = texture.feedback_slot;
    replacement.virtual_metadata_index = texture.virtual_metadata_index;
    replacement.virtual_page_base = texture.virtual_page_base;
    replacement.virtual_page_count = texture.virtual_page_count;
    replacement.data = data;
    if (!upload_texture_image(data, replacement)) return false;
    auto retired = std::move(texture);
    texture = std::move(replacement);
    defer_texture_release(std::move(retired));
    update_gpu_texture_table_window(texture);
    virtual_geometry_material_descriptors_dirty_ = true;
    gpu_bindless_descriptors_dirty_ = true;
    return true;
}

void vulkan_render_backend::upload_streamed_texture(const texture_stream_upload_event& event)
{
    const auto& upload = event.upload;
    texture_stream_upload_result result{.resource = upload.resource,
                                        .content_generation = upload.content_generation,
                                        .kind = upload.kind,
                                        .mip = upload.mip,
                                        .x = upload.x,
                                        .y = upload.y};
    const auto found = textures_.find(resource_key(upload.resource));
    if (found == textures_.end() || !found->second.streamable || !upload.bytes ||
        found->second.streaming.content_generation != upload.content_generation)
    {
        frame_texture_upload_results_.push_back(result);
        return;
    }
    auto& texture = found->second;
    if (upload.kind == texture_subresource_kind::tile)
    {
        result.succeeded = upload_virtual_texture_page(texture, upload, result);
        frame_texture_upload_results_.push_back(result);
        return;
    }
    if (upload.kind != texture_subresource_kind::mip || upload.mip >= texture.streamed_mips.size())
    {
        frame_texture_upload_results_.push_back(result);
        return;
    }
    texture.streamed_mips[upload.mip] = upload.bytes;
    result.succeeded = rebuild_streamed_mip_window(texture);
    result.gpu_bytes = texture.streaming.artifact.mips[upload.mip].decoded_size;
    frame_texture_upload_results_.push_back(result);
}

void vulkan_render_backend::evict_streamed_texture(const texture_stream_evict_event& event)
{
    const auto& eviction = event.eviction;
    const auto found = textures_.find(resource_key(eviction.resource));
    if (found == textures_.end() || !found->second.streamable ||
        found->second.streaming.content_generation != eviction.content_generation)
        return;
    auto& texture = found->second;
    if (eviction.kind == texture_subresource_kind::tile)
    {
        if (texture.virtual_page_base == resource_handle::invalid_index) return;
        const auto tile = std::ranges::find_if(
            texture.streaming.artifact.tiles, [&](const texture_artifact_tile_range& candidate)
            { return candidate.mip == eviction.mip && candidate.x == eviction.x && candidate.y == eviction.y; });
        if (tile == texture.streaming.artifact.tiles.end()) return;
        const auto page_index =
            texture.virtual_page_base + static_cast<std::uint32_t>(tile - texture.streaming.artifact.tiles.begin());
        if (page_index >= virtual_texture_pages_.size()) return;
        auto& page = virtual_texture_pages_[page_index];
        if (virtual_texture_page_resident(page) && page.cache_descriptor < virtual_texture_caches_.size() &&
            page.cache_layer < virtual_texture_caches_[page.cache_descriptor].slots.size())
        {
            auto& slot = virtual_texture_caches_[page.cache_descriptor].slots[page.cache_layer];
            slot.page = resource_handle::invalid_index;
            slot.reusable_after_frame = last_profile_.frame_index + frame_resource_count();
        }
        page.cache_descriptor = resource_handle::invalid_index;
        page.cache_layer = resource_handle::invalid_index;
        page.flags = virtual_texture_page_flag::none;
        (void)ensure_virtual_texture_table_capacity();
        return;
    }
    if (eviction.kind != texture_subresource_kind::mip || eviction.mip >= texture.streamed_mips.size()) return;
    texture.streamed_mips[eviction.mip].reset();
    if (!rebuild_streamed_mip_window(texture))
        arc::diagnostics::warn("render.vulkan", "Failed to shrink a streamed texture mip window");
}

bool vulkan_render_backend::update_host_visible_buffer(gpu_buffer& buffer, const void* data, VkDeviceSize bytes)
{
    if (buffer.buffer == VK_NULL_HANDLE || buffer.allocation == VK_NULL_HANDLE || !data || bytes == 0) return false;
    void* mapped{};
    if (vmaMapMemory(allocator_, buffer.allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, data, static_cast<std::size_t>(bytes));
    vmaFlushAllocation(allocator_, buffer.allocation, 0, bytes);
    vmaUnmapMemory(allocator_, buffer.allocation);
    return true;
}

bool vulkan_render_backend::ensure_virtual_texture_table_capacity()
{
    const auto metadata_capacity =
        std::max(64u, std::bit_ceil(std::max(static_cast<std::uint32_t>(virtual_texture_metadata_.size()), 1u)));
    const auto page_capacity =
        std::max(256u, std::bit_ceil(std::max(static_cast<std::uint32_t>(virtual_texture_pages_.size()), 1u)));
    const auto replace = [&](gpu_buffer& buffer, std::uint32_t& capacity, std::uint32_t required, VkDeviceSize stride,
                             const void* contents, std::size_t count)
    {
        if (buffer.buffer != VK_NULL_HANDLE && capacity >= required)
            return count == 0 || update_host_visible_buffer(buffer, contents, buffer_size(count, stride));
        gpu_buffer replacement{};
        if (!create_buffer(buffer_size(required, stride),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_CPU_TO_GPU, replacement) ||
            (count != 0 && !update_host_visible_buffer(replacement, contents, buffer_size(count, stride))))
        {
            destroy_buffer(replacement);
            return false;
        }
        auto retired = buffer;
        buffer = replacement;
        capacity = required;
        if (retired.buffer != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                     [this, retired]() mutable { destroy_buffer(retired); });
        virtual_texture_descriptors_dirty_ = true;
        return true;
    };
    return replace(virtual_texture_metadata_buffer_, virtual_texture_metadata_capacity_, metadata_capacity,
                   sizeof(virtual_texture_gpu_metadata), virtual_texture_metadata_.data(),
                   virtual_texture_metadata_.size()) &&
           replace(virtual_texture_page_table_buffer_, virtual_texture_page_capacity_, page_capacity,
                   sizeof(virtual_texture_page_table_entry), virtual_texture_pages_.data(),
                   virtual_texture_pages_.size());
}

void vulkan_render_backend::register_virtual_texture(gpu_texture& texture)
{
    if (texture.streaming.mode != texture_streaming_mode::virtual_tiles) return;
    texture.virtual_metadata_index = static_cast<std::uint32_t>(virtual_texture_metadata_.size());
    texture.virtual_page_base = static_cast<std::uint32_t>(virtual_texture_pages_.size());
    texture.virtual_page_count = static_cast<std::uint32_t>(texture.streaming.artifact.tiles.size());
    virtual_texture_metadata_.push_back({.width = texture.streaming.texture.width,
                                         .height = texture.streaming.texture.height,
                                         .mip_count = texture.streaming.artifact.mip_count,
                                         .tail_first_mip = texture.streaming.artifact.tail_first_mip,
                                         .page_table_base = texture.virtual_page_base,
                                         .page_count = texture.virtual_page_count,
                                         .feedback_slot = texture.feedback_slot,
                                         .generation = texture.streaming.content_generation});
    for (const auto& tile : texture.streaming.artifact.tiles)
    {
        std::uint32_t parent = resource_handle::invalid_index;
        if (tile.mip + 1u < texture.streaming.artifact.tail_first_mip)
        {
            const auto found = std::ranges::find_if(
                texture.streaming.artifact.tiles, [&](const texture_artifact_tile_range& candidate)
                { return candidate.mip == tile.mip + 1u && candidate.x == tile.x / 2u && candidate.y == tile.y / 2u; });
            if (found != texture.streaming.artifact.tiles.end())
                parent = texture.virtual_page_base +
                         static_cast<std::uint32_t>(found - texture.streaming.artifact.tiles.begin());
        }
        virtual_texture_pages_.push_back({.generation = texture.streaming.content_generation,
                                          .parent_page = parent,
                                          .mip = tile.mip,
                                          .x = tile.x,
                                          .y = tile.y});
    }
    if (!ensure_virtual_texture_table_capacity())
        arc::diagnostics::warn("render.vulkan", "Failed to publish virtual-texture metadata tables");
}

void vulkan_render_backend::retire_virtual_texture(gpu_texture& texture)
{
    if (texture.virtual_metadata_index < virtual_texture_metadata_.size())
        virtual_texture_metadata_[texture.virtual_metadata_index].generation = 0;
    if (texture.virtual_page_base == resource_handle::invalid_index) return;
    for (std::uint32_t index = 0; index < texture.virtual_page_count; ++index)
    {
        const auto page_index = texture.virtual_page_base + index;
        if (page_index >= virtual_texture_pages_.size()) break;
        auto& page = virtual_texture_pages_[page_index];
        if (virtual_texture_page_resident(page) && page.cache_descriptor < virtual_texture_caches_.size() &&
            page.cache_layer < virtual_texture_caches_[page.cache_descriptor].slots.size())
        {
            auto& slot = virtual_texture_caches_[page.cache_descriptor].slots[page.cache_layer];
            slot.page = resource_handle::invalid_index;
            slot.reusable_after_frame = last_profile_.frame_index + frame_resource_count();
        }
        page.flags = virtual_texture_page_flag::none;
        page.generation = 0;
    }
    (void)ensure_virtual_texture_table_capacity();
}

bool vulkan_render_backend::ensure_virtual_texture_cache(texture_format source_format, std::uint32_t& cache_index)
{
    const auto format = vulkan_texture_format(source_format);
    if (!format || !texture_format_supported(*format)) return false;
    if (const auto found = virtual_texture_cache_lookup_.find(static_cast<std::uint32_t>(*format));
        found != virtual_texture_cache_lookup_.end())
    {
        cache_index = found->second;
        return true;
    }
    constexpr std::uint32_t cache_layers = 512u;
    virtual_texture_physical_cache cache;
    cache.format = *format;
    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = *format;
    image.extent = {virtual_texture_tile_size + virtual_texture_tile_border * 2u,
                    virtual_texture_tile_size + virtual_texture_tile_border * 2u, 1u};
    image.mipLevels = 1;
    image.arrayLayers = cache_layers;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo allocation{};
    allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(allocator_, &image, &allocation, &cache.image, &cache.allocation, nullptr) != VK_SUCCESS)
        return false;
    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = cache.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    view.format = cache.format;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = cache_layers;
    if (vkCreateImageView(device_, &view, nullptr, &cache.view) != VK_SUCCESS)
    {
        vmaDestroyImage(allocator_, cache.image, cache.allocation);
        return false;
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (resolved_config_.features.sampler_anisotropy)
    {
        sampler.anisotropyEnable = VK_TRUE;
        sampler.maxAnisotropy = std::min(4.0f, properties.limits.maxSamplerAnisotropy);
    }
    if (vkCreateSampler(device_, &sampler, nullptr, &cache.sampler) != VK_SUCCESS)
    {
        vkDestroyImageView(device_, cache.view, nullptr);
        vmaDestroyImage(allocator_, cache.image, cache.allocation);
        return false;
    }
    cache.slots.resize(cache_layers);
    cache.free_slots.reserve(cache_layers);
    for (std::uint32_t layer = cache_layers; layer > 0; --layer)
        cache.free_slots.push_back(layer - 1u);
    cache_index = static_cast<std::uint32_t>(virtual_texture_caches_.size());
    virtual_texture_cache_lookup_.emplace(static_cast<std::uint32_t>(*format), cache_index);
    virtual_texture_caches_.push_back(std::move(cache));
    virtual_texture_descriptors_dirty_ = true;
    return true;
}

std::optional<std::uint32_t>
vulkan_render_backend::allocate_virtual_texture_cache_slot(virtual_texture_physical_cache& cache)
{
    for (std::uint32_t index = 0; index < cache.slots.size(); ++index)
        if (cache.slots[index].page == resource_handle::invalid_index && cache.slots[index].reusable_after_frame != 0 &&
            cache.slots[index].reusable_after_frame <= last_completed_frame_)
        {
            cache.slots[index].reusable_after_frame = 0;
            cache.free_slots.push_back(index);
        }
    if (cache.free_slots.empty()) return std::nullopt;
    const auto result = cache.free_slots.back();
    cache.free_slots.pop_back();
    return result;
}

bool vulkan_render_backend::upload_virtual_texture_page(gpu_texture& texture, const texture_stream_upload& upload,
                                                        texture_stream_upload_result& result)
{
    if (texture.virtual_page_base == resource_handle::invalid_index || !upload.bytes) return false;
    const auto tile = std::ranges::find_if(
        texture.streaming.artifact.tiles, [&](const texture_artifact_tile_range& candidate)
        { return candidate.mip == upload.mip && candidate.x == upload.x && candidate.y == upload.y; });
    if (tile == texture.streaming.artifact.tiles.end()) return false;
    const auto local_page = static_cast<std::uint32_t>(tile - texture.streaming.artifact.tiles.begin());
    const auto page_index = texture.virtual_page_base + local_page;
    if (page_index >= virtual_texture_pages_.size()) return false;
    std::uint32_t cache_index{};
    if (!ensure_virtual_texture_cache(texture.streaming.texture.format, cache_index)) return false;
    auto& cache = virtual_texture_caches_[cache_index];
    const auto layer = allocate_virtual_texture_cache_slot(cache);
    if (!layer) return false;
    const auto staging = reserve_upload(upload.bytes->size(), 16u);
    if (!staging)
    {
        cache.free_slots.push_back(*layer);
        return false;
    }
    std::memcpy(staging.bytes.data(), upload.bytes->data(), upload.bytes->size());
    vmaFlushAllocation(allocator_, upload_staging_.allocation, static_cast<VkDeviceSize>(staging.offset),
                       upload.bytes->size());
    VkImageMemoryBarrier to_copy{};
    to_copy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    const bool reused = cache.slots[*layer].generation != 0;
    to_copy.oldLayout = reused ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_copy.srcAccessMask = reused ? VK_ACCESS_SHADER_READ_BIT : 0;
    to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.image = cache.image;
    to_copy.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    to_copy.subresourceRange.baseArrayLayer = *layer;
    to_copy.subresourceRange.layerCount = 1;
    to_copy.subresourceRange.levelCount = 1;
    vkCmdPipelineBarrier(upload_command_buffer_,
                         reused ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_copy);
    VkBufferImageCopy copy{};
    copy.bufferOffset = static_cast<VkDeviceSize>(staging.offset);
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.baseArrayLayer = *layer;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {tile->width, tile->height, 1};
    vkCmdCopyBufferToImage(upload_command_buffer_, upload_staging_.buffer, cache.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    auto to_shader = to_copy;
    to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(upload_command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_shader);
    upload_batch_has_work_ = true;
    // Page-table publication is an acknowledgement boundary: a failed copy
    // must never make a page visible to shaders.
    if (!flush_upload_batch())
    {
        cache.slots[*layer].page = resource_handle::invalid_index;
        cache.slots[*layer].generation = texture.streaming.content_generation;
        cache.slots[*layer].reusable_after_frame = last_profile_.frame_index + frame_resource_count();
        return false;
    }
    cache.slots[*layer] = {.page = page_index, .generation = texture.streaming.content_generation};
    auto& page = virtual_texture_pages_[page_index];
    page.cache_descriptor = cache_index;
    page.cache_layer = *layer;
    page.generation = texture.streaming.content_generation;
    page.flags = virtual_texture_page_flag::resident;
    if (!ensure_virtual_texture_table_capacity())
    {
        page.flags = virtual_texture_page_flag::none;
        cache.slots[*layer].page = resource_handle::invalid_index;
        cache.slots[*layer].reusable_after_frame = last_profile_.frame_index + frame_resource_count();
        return false;
    }
    result.gpu_bytes = tile->decoded_size;
    return true;
}

void vulkan_render_backend::destroy_virtual_texture_resources() noexcept
{
    destroy_buffer(virtual_texture_metadata_buffer_);
    destroy_buffer(virtual_texture_page_table_buffer_);
    for (auto& cache : virtual_texture_caches_)
    {
        if (cache.sampler != VK_NULL_HANDLE) vkDestroySampler(device_, cache.sampler, nullptr);
        if (cache.view != VK_NULL_HANDLE) vkDestroyImageView(device_, cache.view, nullptr);
        if (cache.image != VK_NULL_HANDLE) vmaDestroyImage(allocator_, cache.image, cache.allocation);
    }
    virtual_texture_caches_.clear();
    virtual_texture_cache_lookup_.clear();
    virtual_texture_metadata_.clear();
    virtual_texture_pages_.clear();
}

void vulkan_render_backend::destroy_texture_feedback_resources() noexcept
{
    for (auto& frame : texture_feedback_frames_)
    {
        destroy_buffer(frame.demands);
        destroy_buffer(frame.slots);
    }
    texture_feedback_frames_.clear();
    if (texture_feedback_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, texture_feedback_pipeline_, nullptr);
    if (texture_feedback_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, texture_feedback_pipeline_layout_, nullptr);
    if (texture_feedback_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, texture_feedback_descriptor_pool_, nullptr);
    if (texture_feedback_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, texture_feedback_descriptor_set_layout_, nullptr);
    texture_feedback_pipeline_ = VK_NULL_HANDLE;
    texture_feedback_pipeline_layout_ = VK_NULL_HANDLE;
    texture_feedback_descriptor_pool_ = VK_NULL_HANDLE;
    texture_feedback_descriptor_set_layout_ = VK_NULL_HANDLE;
}

bool vulkan_render_backend::ensure_texture_feedback_pipeline()
{
    if (texture_feedback_pipeline_ != VK_NULL_HANDLE) return true;
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t binding = 0; binding < bindings.size(); ++binding)
        bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo set_layout{};
    set_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    set_layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &set_layout, nullptr, &texture_feedback_descriptor_set_layout_) !=
        VK_SUCCESS)
        return false;

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128u};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = 64u;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &texture_feedback_descriptor_pool_) != VK_SUCCESS) return false;

    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(texture_feedback_push_constants)};
    VkPipelineLayoutCreateInfo pipeline_layout{};
    pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout.setLayoutCount = 1;
    pipeline_layout.pSetLayouts = &texture_feedback_descriptor_set_layout_;
    pipeline_layout.pushConstantRangeCount = 1;
    pipeline_layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &texture_feedback_pipeline_layout_) != VK_SUCCESS)
        return false;

    const auto shader =
        create_shader_module(builtin::texture_mip_feedback_comp_spv, std::size(builtin::texture_mip_feedback_comp_spv));
    if (shader == VK_NULL_HANDLE) return false;
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline.stage = stage;
    pipeline.layout = texture_feedback_pipeline_layout_;
    const auto result =
        vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &texture_feedback_pipeline_);
    vkDestroyShaderModule(device_, shader, nullptr);
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_texture_feedback_frame(texture_feedback_frame& frame, std::uint32_t demand_count,
                                                          std::uint32_t slot_count)
{
    if (!ensure_texture_feedback_pipeline()) return false;
    const auto demand_capacity = std::max(64u, std::bit_ceil(std::max(demand_count, 1u)));
    const auto slot_capacity = std::max(64u, std::bit_ceil(std::max(slot_count, 1u)));
    if (frame.demands.buffer != VK_NULL_HANDLE && frame.demand_capacity >= demand_capacity &&
        frame.slot_capacity >= slot_capacity)
        return true;

    if (frame.descriptor_set != VK_NULL_HANDLE)
        vkFreeDescriptorSets(device_, texture_feedback_descriptor_pool_, 1, &frame.descriptor_set);
    destroy_buffer(frame.demands);
    destroy_buffer(frame.slots);
    frame = {};
    if (!create_buffer(buffer_size(demand_capacity, sizeof(gpu_texture_mip_demand)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VMA_MEMORY_USAGE_CPU_TO_GPU, frame.demands) ||
        !create_buffer(buffer_size(slot_capacity, sizeof(gpu_texture_mip_slot)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VMA_MEMORY_USAGE_CPU_TO_GPU, frame.slots))
    {
        destroy_buffer(frame.demands);
        destroy_buffer(frame.slots);
        return false;
    }
    frame.demand_capacity = demand_capacity;
    frame.slot_capacity = slot_capacity;
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = texture_feedback_descriptor_pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &texture_feedback_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &allocate, &frame.descriptor_set) != VK_SUCCESS) return false;
    const std::array<VkDescriptorBufferInfo, 2> infos{VkDescriptorBufferInfo{frame.demands.buffer, 0, VK_WHOLE_SIZE},
                                                      VkDescriptorBufferInfo{frame.slots.buffer, 0, VK_WHOLE_SIZE}};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
    {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = frame.descriptor_set;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &infos[binding];
    }
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return true;
}

float vulkan_render_backend::projected_texture_extent(const geometric::box3f& bounds) const noexcept
{
    const auto dimensions = geometric::size(bounds);
    const float diameter = std::max({dimensions[0], dimensions[1], dimensions[2]});
    if (!frame_camera_valid_ || !(diameter > 0.0f)) return std::numeric_limits<float>::max();
    const auto center = geometric::center(bounds);
    const math::vector3f offset{center[0] - frame_camera_.position[0], center[1] - frame_camera_.position[1],
                                center[2] - frame_camera_.position[2]};
    const float distance = std::max(std::sqrt(math::length_squared(offset)), diameter * 0.5f);
    const float render_height = static_cast<float>(std::max(frame_camera_.render_height, viewport_height_));
    return diameter * std::abs(frame_camera_.projection(1, 1)) * render_height / std::max(2.0f * distance, 0.001f);
}

std::vector<gpu_texture_mip_demand> vulkan_render_backend::build_texture_mip_demands() const
{
    std::vector<gpu_texture_mip_demand> demands;
    const auto append_texture = [&](texture_handle handle, float extent)
    {
        const auto found = textures_.find(resource_key(handle));
        if (found == textures_.end() || !found->second.streamable ||
            found->second.feedback_slot == resource_handle::invalid_index)
            return;
        const auto slot_index = found->second.feedback_slot;
        if (slot_index >= texture_feedback_slots_.size() || !texture_feedback_slots_[slot_index].active) return;
        const auto& texture = found->second;
        const auto desired = texture_requested_mip(texture.streaming.texture.width, texture.streaming.texture.height,
                                                   texture.streaming.artifact.mip_count, extent);
        const float coverage = std::clamp(
            extent / static_cast<float>(std::max(texture.streaming.texture.width, texture.streaming.texture.height)),
            0.0f, 1.0f);
        demands.push_back({.slot = slot_index,
                           .generation = texture_feedback_slots_[slot_index].slot_generation,
                           .desired_mip = desired,
                           .coverage = static_cast<std::uint32_t>(
                               coverage * static_cast<float>(std::numeric_limits<std::uint32_t>::max()))});
    };
    const auto append_material = [&](material_handle handle, float extent)
    {
        const auto found = materials_.find(resource_key(handle));
        if (found == materials_.end()) return;
        const auto& material = found->second.data;
        for (const auto texture : material.runtime_textures)
            append_texture(texture, extent);
        const std::array textures{material.base_color_texture,
                                  material.metallic_roughness_texture,
                                  material.normal_texture,
                                  material.occlusion_texture,
                                  material.emissive_texture,
                                  material.clear_coat_texture,
                                  material.clear_coat_roughness_texture,
                                  material.clear_coat_normal_texture,
                                  material.anisotropy_texture,
                                  material.subsurface_texture,
                                  material.thickness_texture,
                                  material.transmission_texture};
        for (const auto texture : textures)
            append_texture(texture, extent);
    };
    for (const auto& draw : frame_draws_)
        append_material(draw.material, projected_texture_extent(draw.world_bounds));
    for (const auto& draw : frame_virtual_draws_)
        append_material(draw.draw.material, projected_texture_extent(draw.draw.world_bounds));
    for (const auto& draw : frame_terrain_draws_)
        append_material(draw.terrain.material, projected_texture_extent(draw.terrain.world_bounds));
    return demands;
}

void vulkan_render_backend::dispatch_texture_mip_feedback(VkCommandBuffer command_buffer)
{
    if (!resolved_config_.features.texture_streaming || texture_feedback_slots_.empty()) return;
    auto demands = build_texture_mip_demands();
    if (demands.empty()) return;
    if (texture_feedback_frames_.size() < frame_resource_count())
        texture_feedback_frames_.resize(frame_resource_count());
    auto& frame = texture_feedback_frames_[current_frame_slot()];
    const auto slot_count = static_cast<std::uint32_t>(texture_feedback_slots_.size());
    if (!ensure_texture_feedback_frame(frame, static_cast<std::uint32_t>(demands.size()), slot_count))
    {
        last_profile_.texture_streaming.fallback_reason = "texture feedback resources are unavailable";
        return;
    }

    std::vector<gpu_texture_mip_slot> slots(slot_count);
    for (std::uint32_t index = 0; index < slot_count; ++index)
        slots[index].generation =
            texture_feedback_slots_[index].active ? texture_feedback_slots_[index].slot_generation : 0u;
    void* mapped{};
    if (vmaMapMemory(allocator_, frame.demands.allocation, &mapped) != VK_SUCCESS) return;
    std::memcpy(mapped, demands.data(), demands.size() * sizeof(gpu_texture_mip_demand));
    vmaFlushAllocation(allocator_, frame.demands.allocation, 0, demands.size() * sizeof(gpu_texture_mip_demand));
    vmaUnmapMemory(allocator_, frame.demands.allocation);
    if (vmaMapMemory(allocator_, frame.slots.allocation, &mapped) != VK_SUCCESS) return;
    std::memcpy(mapped, slots.data(), slots.size() * sizeof(gpu_texture_mip_slot));
    vmaFlushAllocation(allocator_, frame.slots.allocation, 0, slots.size() * sizeof(gpu_texture_mip_slot));
    vmaUnmapMemory(allocator_, frame.slots.allocation);

    std::array<VkBufferMemoryBarrier, 2> input_barriers{};
    const std::array<gpu_buffer, 2> buffers{frame.demands, frame.slots};
    for (std::size_t index = 0; index < input_barriers.size(); ++index)
    {
        input_barriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        input_barriers[index].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        input_barriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        input_barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        input_barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        input_barriers[index].buffer = buffers[index].buffer;
        input_barriers[index].size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, static_cast<std::uint32_t>(input_barriers.size()), input_barriers.data(), 0, nullptr);
    const texture_feedback_push_constants constants{static_cast<std::uint32_t>(demands.size()), slot_count};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, texture_feedback_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, texture_feedback_pipeline_layout_, 0, 1,
                            &frame.descriptor_set, 0, nullptr);
    vkCmdPushConstants(command_buffer, texture_feedback_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(constants), &constants);
    vkCmdDispatch(command_buffer, (constants.demand_count + 63u) / 64u, 1u, 1u);
    VkBufferMemoryBarrier output{};
    output.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    output.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    output.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    output.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    output.buffer = frame.slots.buffer;
    output.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                         nullptr, 1, &output, 0, nullptr);
    frame.submitted_slot_count = slot_count;
    frame.submitted_frame = last_profile_.frame_index;
}

void vulkan_render_backend::collect_texture_mip_feedback(std::uint32_t frame_index)
{
    if (frame_index >= texture_feedback_frames_.size()) return;
    auto& frame = texture_feedback_frames_[frame_index];
    if (frame.submitted_frame == 0 || frame.submitted_slot_count == 0) return;
    void* mapped{};
    if (vmaMapMemory(allocator_, frame.slots.allocation, &mapped) != VK_SUCCESS) return;
    vmaInvalidateAllocation(allocator_, frame.slots.allocation, 0,
                            frame.submitted_slot_count * sizeof(gpu_texture_mip_slot));
    const auto* slots = static_cast<const gpu_texture_mip_slot*>(mapped);
    completed_texture_feedback_.frame_index = frame.submitted_frame;
    for (std::uint32_t index = 0; index < frame.submitted_slot_count && index < texture_feedback_slots_.size(); ++index)
    {
        const auto& owner = texture_feedback_slots_[index];
        if (!owner.active || slots[index].generation != owner.slot_generation ||
            slots[index].desired_mip == std::numeric_limits<std::uint32_t>::max())
            continue;
        completed_texture_feedback_.mips.push_back(
            {.resource = owner.resource,
             .content_generation = owner.content_generation,
             .desired_mip = std::min(slots[index].desired_mip, owner.mip_count - 1u),
             .screen_coverage = static_cast<float>(slots[index].coverage) /
                                static_cast<float>(std::numeric_limits<std::uint32_t>::max())});
    }
    vmaUnmapMemory(allocator_, frame.slots.allocation);
    frame.submitted_frame = 0;
    frame.submitted_slot_count = 0;
}

void vulkan_render_backend::upload_texture(const texture_upload_event& event)
{
    if (!event.texture) return;

    gpu_texture texture{.handle = event.handle, .data = *event.texture};
    const bool uploaded = upload_texture_image(*event.texture, texture);
    if (!uploaded && event.texture->dds && event.texture->compressed)
    {
        arc::diagnostics::warn(
            "render.vulkan",
            "DDS texture '" + event.label +
                "' uses a compressed format unsupported by this Vulkan device; using fallback descriptors");
    }
    else if (!uploaded && !event.texture->has_pixels() && !event.texture->encoded.empty())
    {
        arc::diagnostics::debug("render.vulkan",
                                "Texture '" + event.label + "' kept as encoded data until image decoding is available");
    }

    const std::uint64_t key = resource_key(event.handle);
    if (auto found = textures_.find(key); found != textures_.end())
    {
        retire_texture_feedback_slot(found->second.feedback_slot);
        defer_texture_release(std::move(found->second));
    }
    textures_[key] = std::move(texture);
    virtual_geometry_material_descriptors_dirty_ = true;
    gpu_bindless_descriptors_dirty_ = true;
}

void vulkan_render_backend::upload_material(const material_upload_event& event)
{
    if (!event.material) return;

    auto& material = materials_[resource_key(event.handle)];
    if (material.runtime.gbuffer_pipeline != VK_NULL_HANDLE || material.runtime.pipeline_layout != VK_NULL_HANDLE ||
        material.runtime.descriptor_pool != VK_NULL_HANDLE ||
        material.runtime.descriptor_set_layout != VK_NULL_HANDLE || !material.runtime.parameter_buffers.empty() ||
        !material.runtime.frame_buffers.empty())
    {
        auto retired = std::move(material.runtime);
        material.runtime = {};
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, retired = std::move(retired)]() mutable { destroy_material_runtime(retired); });
    }
    else
        material.runtime = {};
    material.data = *event.material;
}

void vulkan_render_backend::upload_environment(const environment_upload_event& event)
{
    if (!event.environment) return;

    auto environment = *event.environment;
    if (!environment.prefiltered)
    {
        environment.diffuse_irradiance = environment.fallback_color;
        environment.diffuse_intensity = environment.intensity;
    }
    environments_[resource_key(event.handle)] = gpu_environment{.data = std::move(environment)};
    active_environment_ = event.handle;
}

const environment_descriptor* vulkan_render_backend::active_environment() const noexcept
{
    const auto found = environments_.find(resource_key(active_environment_));
    return found == environments_.end() ? nullptr : &found->second.data;
}

void vulkan_render_backend::update_environment_profile(const environment_descriptor* lighting_environment)
{
    auto& profile = last_profile_.environment;
    profile = {};
    profile.enabled = frame_environment_.enabled;
    profile.sky_visible = frame_environment_.enabled && frame_environment_.sky_visible;
    profile.affects_lighting = frame_environment_.affect_lighting && frame_environment_.lighting.enabled;
    switch (frame_environment_.source)
    {
        case sky_source_mode::physical_atmosphere:
            profile.source = "Physical atmosphere";
            break;
        case sky_source_mode::hdri:
            profile.source = "HDRI";
            break;
        case sky_source_mode::solid_color:
            profile.source = "Solid color";
            break;
    }

    if (!profile.enabled)
    {
        profile.quality_path = "Disabled";
        profile.atmosphere_lut_state = "Not required";
    }
    else if (frame_environment_.source == sky_source_mode::physical_atmosphere)
    {
        profile.quality_path = resolved_config_.quality == render_quality_tier::low ? "Analytic low-tier"
                                                                                    : "Analytic compatibility fallback";
        profile.atmosphere_lut_state = resolved_config_.quality == render_quality_tier::low
                                           ? "Not required by low tier"
                                           : "Graph scheduled; Vulkan execution pending";
    }
    else
    {
        profile.quality_path = "Texture/constant composite";
        profile.atmosphere_lut_state = "Not required";
    }

    if (!profile.affects_lighting)
        profile.environment_lighting_state = "Disabled";
    else if (lighting_environment && lighting_environment->prefiltered)
        profile.environment_lighting_state = "Prefiltered environment";
    else
        profile.environment_lighting_state = "Diffuse fallback";

    // The graph owns the future standard-tier cloud shadow pass, but the
    // current Vulkan executor does not allocate or sample that texture yet.
    profile.cloud_shadow_resolution = 0;
    profile.fallback_reason = frame_environment_.fallback_reason;
    if (frame_environment_.source == sky_source_mode::hdri &&
        (!frame_environment_.hdri_texture.valid() ||
         textures_.find(resource_key(frame_environment_.hdri_texture)) == textures_.end()))
    {
        profile.fallback_reason = "HDRI texture is unavailable; using the visible fallback color";
    }
    else if (frame_environment_.source == sky_source_mode::physical_atmosphere &&
             resolved_config_.quality != render_quality_tier::low && profile.fallback_reason.empty())
    {
        profile.fallback_reason = "Atmosphere LUT execution is not available in Vulkan yet; using the analytic sky";
    }
}

} // namespace arc::render::vulkan::backend_detail
