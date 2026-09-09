#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"

namespace arc::render::vulkan::backend_detail
{
bool vulkan_render_backend::ensure_terrain_descriptors()
{
    if (terrain_descriptor_set_layout_ != VK_NULL_HANDLE && terrain_descriptor_pool_ != VK_NULL_HANDLE) return true;
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    bindings[0] = {0u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    bindings[1] = {1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    bindings[2] = {2u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    bindings[3] = {3u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &terrain_descriptor_set_layout_) != VK_SUCCESS)
        return false;

    constexpr std::uint32_t capacity = 2048u;
    const std::array<VkDescriptorPoolSize, 2> sizes{
        {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, capacity * 3u}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, capacity}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = capacity;
    pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &terrain_descriptor_pool_) != VK_SUCCESS)
    {
        vkDestroyDescriptorSetLayout(device_, terrain_descriptor_set_layout_, nullptr);
        terrain_descriptor_set_layout_ = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool vulkan_render_backend::allocate_terrain_draw_descriptor(const gpu_terrain& terrain, VkBuffer patches,
                                                             VkDescriptorSet& descriptor)
{
    if (!ensure_terrain_descriptors()) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = terrain_descriptor_pool_;
    allocate.descriptorSetCount = 1u;
    allocate.pSetLayouts = &terrain_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &allocate, &descriptor) != VK_SUCCESS) return false;
    const VkDescriptorBufferInfo heights{terrain.heights.buffer, 0u, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo weights{terrain.weights.buffer, 0u, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo parameters{terrain.parameters.buffer, 0u, sizeof(terrain_resource_uniform)};
    const VkDescriptorBufferInfo selected_patches{patches, 0u, VK_WHOLE_SIZE};
    std::array<VkWriteDescriptorSet, 4> writes{};
    const std::array<const VkDescriptorBufferInfo*, 4> infos{&heights, &weights, &parameters, &selected_patches};
    const std::array<VkDescriptorType, 4> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    for (std::size_t index = 0; index < writes.size(); ++index)
    {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = descriptor;
        writes[index].dstBinding = static_cast<std::uint32_t>(index);
        writes[index].descriptorCount = 1u;
        writes[index].descriptorType = types[index];
        writes[index].pBufferInfo = infos[index];
    }
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    return true;
}

bool vulkan_render_backend::allocate_terrain_descriptor(gpu_terrain& terrain)
{
    return allocate_terrain_draw_descriptor(terrain, terrain.fallback_patch.buffer, terrain.descriptor_set);
}

void vulkan_render_backend::destroy_terrain_buffers(gpu_terrain& terrain) noexcept
{
    destroy_buffer(terrain.heights);
    destroy_buffer(terrain.weights);
    destroy_buffer(terrain.parameters);
    destroy_buffer(terrain.hierarchy);
    destroy_buffer(terrain.fallback_patch);
}

void vulkan_render_backend::upload_terrain(const terrain_upload_event& event)
{
    if (!event.terrain || event.terrain->heights.empty() || event.terrain->weights.empty()) return;
    gpu_terrain terrain;
    terrain.sample_resolution = event.terrain->sample_resolution;
    terrain.patch_quads = event.terrain->lod.patch_quads;
    const auto packed_hierarchy = make_terrain_gpu_hierarchy(event.terrain->hierarchy);
    terrain.hierarchy_node_count = static_cast<std::uint32_t>(packed_hierarchy.nodes.size());
    terrain.hierarchy_leaf_count = packed_hierarchy.leaf_count;
    terrain.geometric_error_multiplier = event.terrain->lod.geometric_error_multiplier;
    const terrain_resource_uniform parameters{event.terrain->sample_resolution,
                                              event.terrain->lod.patch_quads,
                                              packed_hierarchy.root,
                                              static_cast<std::uint32_t>(packed_hierarchy.nodes.size()),
                                              event.terrain->width,
                                              event.terrain->depth,
                                              {},
                                              packed_hierarchy.leaf_count,
                                              {}};
    const gpu_terrain_patch_record fallback_patch{};
    const auto height_bytes = buffer_size(event.terrain->heights.size(), sizeof(float));
    const auto weight_bytes = buffer_size(event.terrain->weights.size(), sizeof(event.terrain->weights[0]));
    if (!ensure_terrain_topologies(terrain.patch_quads) ||
        !upload_buffer(event.terrain->heights.data(), height_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       terrain.heights) ||
        !upload_buffer(event.terrain->weights.data(), weight_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       terrain.weights) ||
        !upload_buffer(&parameters, sizeof(parameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, terrain.parameters) ||
        !upload_buffer(packed_hierarchy.nodes.data(),
                       buffer_size(packed_hierarchy.nodes.size(), sizeof(gpu_terrain_node_record)),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, terrain.hierarchy) ||
        !upload_buffer(&fallback_patch, sizeof(fallback_patch), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       terrain.fallback_patch) ||
        !allocate_terrain_descriptor(terrain))
    {
        destroy_terrain_buffers(terrain);
        arc::diagnostics::error("render.vulkan", "Failed to upload terrain '" + event.label + "'");
        return;
    }
    const auto key = resource_key(event.handle);
    bool has_instances{};
    for (const auto& [_, instance] : gpu_terrain_instances_)
        has_instances = has_instances || instance.terrain == event.handle;
    if (has_instances)
    {
        wait_for_in_flight_frames();
        for (auto instance = gpu_terrain_instances_.begin(); instance != gpu_terrain_instances_.end();)
        {
            if (instance->second.terrain != event.handle)
            {
                ++instance;
                continue;
            }
            destroy_gpu_terrain_instance(instance->second);
            instance = gpu_terrain_instances_.erase(instance);
        }
    }
    if (auto found = terrains_.find(key); found != terrains_.end())
    {
        auto replaced = std::move(found->second);
        const auto descriptor = replaced.descriptor_set;
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, replaced, descriptor]() mutable
                                 {
                                     destroy_terrain_buffers(replaced);
                                     if (descriptor != VK_NULL_HANDLE && terrain_descriptor_pool_ != VK_NULL_HANDLE)
                                         vkFreeDescriptorSets(device_, terrain_descriptor_pool_, 1u, &descriptor);
                                 });
    }
    terrains_[key] = std::move(terrain);
    last_profile_.terrain.height_bytes += static_cast<std::uint64_t>(height_bytes);
    last_profile_.terrain.weight_bytes += static_cast<std::uint64_t>(weight_bytes);
    last_profile_.terrain.uploaded_height_bytes += static_cast<std::uint64_t>(height_bytes);
    last_profile_.terrain.uploaded_weight_bytes += static_cast<std::uint64_t>(weight_bytes);
}

bool vulkan_render_backend::ensure_terrain_topologies(std::uint32_t patch_quads)
{
    for (std::uint8_t mask = 0u; mask < 16u; ++mask)
    {
        const auto key = (patch_quads << 8u) | mask;
        if (terrain_topologies_.contains(key)) continue;
        const auto indices = make_terrain_patch_indices(patch_quads, mask);
        terrain_topology topology;
        if (indices.empty() || !upload_buffer(indices.data(), buffer_size(indices.size(), sizeof(std::uint32_t)),
                                              VK_BUFFER_USAGE_INDEX_BUFFER_BIT, topology.indices))
            return false;
        topology.index_count = static_cast<std::uint32_t>(indices.size());
        terrain_topologies_.emplace(key, std::move(topology));
    }
    return true;
}

template <typename T>
bool vulkan_render_backend::update_terrain_rows(VkBuffer destination, std::uint32_t destination_resolution,
                                                const terrain_sample_region& region, std::uint32_t row_stride,
                                                const std::vector<T>& values)
{
    if (destination == VK_NULL_HANDLE || row_stride < region.width() ||
        values.size() < static_cast<std::size_t>(row_stride) * region.height())
        return false;
    const auto byte_size = buffer_size(values.size(), sizeof(T));
    const auto staging = reserve_upload(byte_size, alignof(T));
    if (!staging) return false;
    std::memcpy(staging.bytes.data(), values.data(), static_cast<std::size_t>(byte_size));
    vmaFlushAllocation(allocator_, upload_staging_.allocation, static_cast<VkDeviceSize>(staging.offset), byte_size);
    std::vector<VkBufferCopy> copies(region.height());
    for (std::uint32_t row = 0; row < region.height(); ++row)
        copies[row] = {
            .srcOffset =
                static_cast<VkDeviceSize>(staging.offset) + static_cast<VkDeviceSize>(row) * row_stride * sizeof(T),
            .dstOffset =
                (static_cast<VkDeviceSize>(region.min_z + row) * destination_resolution + region.min_x) * sizeof(T),
            .size = static_cast<VkDeviceSize>(region.width()) * sizeof(T)};
    vkCmdCopyBuffer(upload_command_buffer_, upload_staging_.buffer, destination,
                    static_cast<std::uint32_t>(copies.size()), copies.data());
    upload_batch_has_work_ = true;
    return true;
}

void vulkan_render_backend::update_terrain_heights(const terrain_height_update_event& event)
{
    if (!event.update) return;
    const auto found = terrains_.find(resource_key(event.handle));
    if (found == terrains_.end()) return;
    const bool heights_updated =
        update_terrain_rows(found->second.heights.buffer, found->second.sample_resolution, event.update->region,
                            event.update->row_stride, event.update->values);
    const bool hierarchy_updated =
        !event.hierarchy ||
        (event.hierarchy->nodes.size() == found->second.hierarchy_node_count &&
         upload_buffer_region(event.hierarchy->nodes.data(),
                              buffer_size(event.hierarchy->nodes.size(), sizeof(gpu_terrain_node_record)),
                              found->second.hierarchy, 0u));
    if (!heights_updated || !hierarchy_updated)
        arc::diagnostics::warn("render.vulkan", "Failed to upload a terrain height region");
    else
        last_profile_.terrain.uploaded_height_bytes +=
            static_cast<std::uint64_t>(event.update->region.width()) * event.update->region.height() * sizeof(float);
}

void vulkan_render_backend::update_terrain_weights(const terrain_weight_update_event& event)
{
    if (!event.update) return;
    const auto found = terrains_.find(resource_key(event.handle));
    if (found == terrains_.end()) return;
    if (!update_terrain_rows(found->second.weights.buffer, found->second.sample_resolution, event.update->region,
                             event.update->row_stride, event.update->values))
        arc::diagnostics::warn("render.vulkan", "Failed to upload a terrain weight region");
    else
        last_profile_.terrain.uploaded_weight_bytes += static_cast<std::uint64_t>(event.update->region.width()) *
                                                       event.update->region.height() * sizeof(event.update->values[0]);
}

void vulkan_render_backend::retire_terrain(terrain_handle handle)
{
    const auto found = terrains_.find(resource_key(handle));
    if (found == terrains_.end()) return;
    auto retired = std::move(found->second);
    terrains_.erase(found);
    const auto descriptor = retired.descriptor_set;
    deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                             [this, retired, descriptor]() mutable
                             {
                                 destroy_terrain_buffers(retired);
                                 if (descriptor != VK_NULL_HANDLE && terrain_descriptor_pool_ != VK_NULL_HANDLE)
                                     vkFreeDescriptorSets(device_, terrain_descriptor_pool_, 1u, &descriptor);
                             });
    const bool has_instances =
        std::ranges::any_of(gpu_terrain_instances_, [&](const auto& value) { return value.second.terrain == handle; });
    if (has_instances) wait_for_in_flight_frames();
    for (auto instance = gpu_terrain_instances_.begin(); instance != gpu_terrain_instances_.end();)
    {
        if (instance->second.terrain != handle)
        {
            ++instance;
            continue;
        }
        destroy_gpu_terrain_instance(instance->second);
        instance = gpu_terrain_instances_.erase(instance);
    }
}

void vulkan_render_backend::destroy_gpu_terrain_frame(gpu_terrain_traversal_frame& frame) noexcept
{
    destroy_buffer(frame.stack);
    destroy_buffer(frame.patches);
    destroy_buffer(frame.counters);
    destroy_buffer(frame.indirect);
    destroy_buffer(frame.readback);
    if (frame.traversal_descriptor != VK_NULL_HANDLE && gpu_terrain_descriptor_pool_ != VK_NULL_HANDLE)
        vkFreeDescriptorSets(device_, gpu_terrain_descriptor_pool_, 1u, &frame.traversal_descriptor);
    if (frame.draw_descriptor != VK_NULL_HANDLE && terrain_descriptor_pool_ != VK_NULL_HANDLE)
        vkFreeDescriptorSets(device_, terrain_descriptor_pool_, 1u, &frame.draw_descriptor);
    frame = {};
}

void vulkan_render_backend::destroy_gpu_terrain_instance(gpu_terrain_instance& instance) noexcept
{
    for (auto& frame : instance.frames)
        destroy_gpu_terrain_frame(frame);
    instance.frames.clear();
}

bool vulkan_render_backend::ensure_gpu_terrain_pipeline()
{
    if (gpu_terrain_traversal_pipeline_ != VK_NULL_HANDLE) return true;
    if (!resolved_config_.features.gpu_terrain_traversal) return false;

    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    for (std::uint32_t binding = 0u; binding < 5u; ++binding)
        bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[5] = {5u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo descriptor_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptor_layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr, &gpu_terrain_descriptor_set_layout_) !=
        VK_SUCCESS)
        return false;

    constexpr std::uint32_t maximum_sets = 4096u;
    const std::array<VkDescriptorPoolSize, 2> sizes{
        {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maximum_sets * 5u}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maximum_sets}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = maximum_sets;
    pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &gpu_terrain_descriptor_pool_) != VK_SUCCESS) return false;

    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(gpu_terrain_traversal_push_constants)};
    VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout.setLayoutCount = 1u;
    pipeline_layout.pSetLayouts = &gpu_terrain_descriptor_set_layout_;
    pipeline_layout.pushConstantRangeCount = 1u;
    pipeline_layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &gpu_terrain_traversal_pipeline_layout_) !=
        VK_SUCCESS)
        return false;

    const auto shader = create_shader_module(builtin::gpu_terrain_traversal_comp_spv,
                                             std::size(builtin::gpu_terrain_traversal_comp_spv));
    if (shader == VK_NULL_HANDLE) return false;
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.stage = stage;
    pipeline.layout = gpu_terrain_traversal_pipeline_layout_;
    const auto result =
        vkCreateComputePipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr, &gpu_terrain_traversal_pipeline_);
    vkDestroyShaderModule(device_, shader, nullptr);
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::allocate_gpu_terrain_frame(const gpu_terrain& terrain, gpu_terrain_traversal_frame& frame)
{
    const auto node_bytes = buffer_size(terrain.hierarchy_node_count, sizeof(std::uint32_t));
    const auto patch_bytes = buffer_size(terrain.hierarchy_leaf_count, sizeof(gpu_terrain_patch_record));
    const auto indirect_bytes = buffer_size(terrain.hierarchy_leaf_count, sizeof(VkDrawIndexedIndirectCommand));
    if (node_bytes == 0u || patch_bytes == 0u || indirect_bytes == 0u ||
        !create_buffer(node_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, frame.stack) ||
        !create_buffer(patch_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, frame.patches) ||
        !create_buffer(sizeof(gpu_terrain_counter_data),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                       VMA_MEMORY_USAGE_GPU_ONLY, frame.counters) ||
        !create_buffer(indirect_bytes,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                       VMA_MEMORY_USAGE_GPU_ONLY, frame.indirect) ||
        !create_buffer(sizeof(gpu_terrain_counter_data), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU,
                       frame.readback))
    {
        destroy_gpu_terrain_frame(frame);
        return false;
    }

    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = gpu_terrain_descriptor_pool_;
    allocate.descriptorSetCount = 1u;
    allocate.pSetLayouts = &gpu_terrain_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &allocate, &frame.traversal_descriptor) != VK_SUCCESS ||
        !allocate_terrain_draw_descriptor(terrain, frame.patches.buffer, frame.draw_descriptor))
    {
        destroy_gpu_terrain_frame(frame);
        return false;
    }

    const std::array<VkDescriptorBufferInfo, 6> infos{
        {{terrain.hierarchy.buffer, 0u, VK_WHOLE_SIZE},
         {frame.stack.buffer, 0u, VK_WHOLE_SIZE},
         {frame.patches.buffer, 0u, VK_WHOLE_SIZE},
         {frame.counters.buffer, 0u, VK_WHOLE_SIZE},
         {frame.indirect.buffer, 0u, VK_WHOLE_SIZE},
         {terrain.parameters.buffer, 0u, sizeof(terrain_resource_uniform)}}};
    std::array<VkWriteDescriptorSet, 6> writes{};
    for (std::uint32_t binding = 0u; binding < writes.size(); ++binding)
    {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = frame.traversal_descriptor;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1u;
        writes[binding].descriptorType =
            binding == 5u ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &infos[binding];
    }
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    return true;
}

vulkan_render_backend::gpu_terrain_instance*
vulkan_render_backend::ensure_gpu_terrain_instance(const gpu_terrain_draw& draw)
{
    if (!draw.terrain.gpu_scene_instance.valid() || !ensure_gpu_terrain_pipeline()) return nullptr;
    const auto terrain_found = terrains_.find(resource_key(draw.terrain.terrain));
    if (terrain_found == terrains_.end() || terrain_found->second.hierarchy_leaf_count == 0u) return nullptr;
    constexpr std::uint32_t terrain_patch_capacity = 2048u;
    if (terrain_found->second.hierarchy_leaf_count > std::min(terrain_patch_capacity, max_indirect_draw_count_))
    {
        last_profile_.gpu_scene.fallback_reason =
            "terrain traversal capacity exceeded; using deterministic CPU patch submission";
        return nullptr;
    }
    const auto key = gpu_scene_key(draw.terrain.gpu_scene_instance);
    auto& instance = gpu_terrain_instances_[key];
    const auto& terrain = terrain_found->second;
    const auto frame_count = frame_resource_count();
    if (instance.terrain != draw.terrain.terrain || instance.node_capacity != terrain.hierarchy_node_count ||
        instance.patch_capacity != terrain.hierarchy_leaf_count || instance.frames.size() != frame_count)
    {
        destroy_gpu_terrain_instance(instance);
        instance.terrain = draw.terrain.terrain;
        instance.node_capacity = terrain.hierarchy_node_count;
        instance.patch_capacity = terrain.hierarchy_leaf_count;
        instance.frames.resize(frame_count);
        for (auto& frame : instance.frames)
            if (!allocate_gpu_terrain_frame(terrain, frame))
            {
                destroy_gpu_terrain_instance(instance);
                return nullptr;
            }
    }
    if (instance.overflowed)
    {
        last_profile_.terrain.fallback_reason =
            "GPU terrain traversal overflowed; using deterministic CPU patch submission";
        return nullptr;
    }
    return &instance;
}

void vulkan_render_backend::collect_gpu_terrain_feedback(std::uint32_t frame_index)
{
    gpu_terrain_counter_data collected{};
    bool has_feedback{};
    for (auto& [_, instance] : gpu_terrain_instances_)
    {
        if (frame_index >= instance.frames.size()) continue;
        auto& frame = instance.frames[frame_index];
        if (!frame.readback_pending || frame.readback.buffer == VK_NULL_HANDLE) continue;
        vmaInvalidateAllocation(allocator_, frame.readback.allocation, 0u, sizeof(gpu_terrain_counter_data));
        void* mapped{};
        if (vmaMapMemory(allocator_, frame.readback.allocation, &mapped) != VK_SUCCESS) continue;
        gpu_terrain_counter_data counters{};
        std::memcpy(&counters, mapped, sizeof(counters));
        vmaUnmapMemory(allocator_, frame.readback.allocation);
        collected.selected_count += counters.selected_count;
        collected.culled_count += counters.culled_count;
        collected.overflow_count += counters.overflow_count != 0u ? 1u : 0u;
        collected.draw_count += counters.draw_count;
        instance.overflowed = instance.overflowed || counters.overflow_count != 0u;
        frame.readback_pending = false;
        frame.dispatched = false;
        has_feedback = true;
    }
    if (has_feedback) completed_gpu_terrain_statistics_ = collected;
}

void vulkan_render_backend::dispatch_gpu_terrain_traversal(VkCommandBuffer command_buffer)
{
    gpu_terrain_active_instances_.clear();
    const auto slot = current_frame_slot();
    for (const auto& draw : frame_gpu_terrain_draws_)
    {
        auto* instance = ensure_gpu_terrain_instance(draw);
        if (!instance || slot >= instance->frames.size())
        {
            ++last_profile_.terrain.gpu_fallback_instances;
            if (last_profile_.terrain.fallback_reason.empty())
                last_profile_.terrain.fallback_reason =
                    "GPU terrain resources are unavailable; using deterministic CPU patch submission";
            continue;
        }
        const auto terrain_found = terrains_.find(resource_key(draw.terrain.terrain));
        if (terrain_found == terrains_.end()) continue;
        auto& frame = instance->frames[slot];

        vkCmdFillBuffer(command_buffer, frame.counters.buffer, 0u, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(command_buffer, frame.indirect.buffer, 0u, VK_WHOLE_SIZE, 0u);
        std::array<VkBufferMemoryBarrier, 2> clears{};
        for (std::uint32_t index = 0u; index < clears.size(); ++index)
        {
            clears[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            clears[index].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            clears[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            clears[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            clears[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            clears[index].buffer = index == 0u ? frame.counters.buffer : frame.indirect.buffer;
            clears[index].size = VK_WHOLE_SIZE;
        }
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                             0u, nullptr, static_cast<std::uint32_t>(clears.size()), clears.data(), 0u, nullptr);

        gpu_terrain_traversal_push_constants constants{};
        const auto mvp = math::matmul(draw.view_projection, draw.terrain.model);
        std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
        for (std::uint32_t row = 0u; row < 3u; ++row)
            for (std::uint32_t column = 0u; column < 4u; ++column)
                constants.model_rows[row * 4u + column] = draw.terrain.model(row, column);
        constants.camera_and_error[0] = frame_camera_.position[0];
        constants.camera_and_error[1] = frame_camera_.position[1];
        constants.camera_and_error[2] = frame_camera_.position[2];
        const float projection_scale =
            std::abs(frame_camera_.projection(1, 1)) * 0.5f * std::max(frame_camera_.render_height, 1u);
        constants.camera_and_error[3] = projection_scale * terrain_found->second.geometric_error_multiplier /
                                        std::max(resolved_config_.geometry_error_threshold, 0.01f);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_terrain_traversal_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_terrain_traversal_pipeline_layout_,
                                0u, 1u, &frame.traversal_descriptor, 0u, nullptr);
        vkCmdPushConstants(command_buffer, gpu_terrain_traversal_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, 1u, 1u, 1u);

        std::array<VkBufferMemoryBarrier, 3> outputs{};
        const std::array<VkBuffer, 3> output_buffers{frame.patches.buffer, frame.counters.buffer,
                                                     frame.indirect.buffer};
        for (std::uint32_t index = 0u; index < outputs.size(); ++index)
        {
            outputs[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            outputs[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            outputs[index].dstAccessMask =
                index == 0u
                    ? static_cast<VkAccessFlags>(VK_ACCESS_SHADER_READ_BIT)
                    : static_cast<VkAccessFlags>(VK_ACCESS_INDIRECT_COMMAND_READ_BIT) |
                          (index == 1u ? static_cast<VkAccessFlags>(VK_ACCESS_TRANSFER_READ_BIT) : VkAccessFlags{0});
            outputs[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            outputs[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            outputs[index].buffer = output_buffers[index];
            outputs[index].size = VK_WHOLE_SIZE;
        }
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0u, 0u, nullptr, static_cast<std::uint32_t>(outputs.size()), outputs.data(), 0u, nullptr);
        const VkBufferCopy readback_copy{.size = sizeof(gpu_terrain_counter_data)};
        vkCmdCopyBuffer(command_buffer, frame.counters.buffer, frame.readback.buffer, 1u, &readback_copy);
        frame.dispatched = true;
        frame.readback_pending = true;
        gpu_terrain_active_instances_.insert(gpu_scene_key(draw.terrain.gpu_scene_instance));
    }
}

VkBuffer vulkan_render_backend::mesh_vertex_buffer(const gpu_mesh& mesh,
                                                   gpu_scene_instance_handle instance) const noexcept
{
    if (instance.valid())
    {
        const auto key = (static_cast<std::uint64_t>(instance.generation) << 32u) | instance.index;
        const auto skinned = gpu_skinned_instances_.find(key);
        const auto slot = current_frame_slot();
        if (skinned != gpu_skinned_instances_.end() && slot < skinned->second.current_vertices.size() &&
            skinned->second.current_vertices[slot].buffer != VK_NULL_HANDLE)
            return skinned->second.current_vertices[slot].buffer;
    }
    if (!mesh.dynamic) return mesh.vertices.buffer;
    const auto slot = current_frame_slot();
    return slot < mesh.dynamic_vertices.size() ? mesh.dynamic_vertices[slot].buffer : VK_NULL_HANDLE;
}

void vulkan_render_backend::update_dynamic_mesh_vertices()
{
    const auto slot = current_frame_slot();
    const auto frame_count = frame_resource_count();
    for (auto& [_, mesh] : meshes_)
    {
        if (mesh.dynamic && mesh.dynamic_vertices.size() != frame_count)
        {
            wait_for_in_flight_frames();
            for (auto& vertices : mesh.dynamic_vertices)
                destroy_buffer(vertices);
            mesh.dynamic_vertices.assign(frame_count, {});
            mesh.uploaded_revisions.assign(frame_count, 0u);
            const auto bytes = buffer_size(mesh.pending_vertices.size(), sizeof(mesh_vertex));
            for (auto& vertices : mesh.dynamic_vertices)
            {
                if (!create_buffer(bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VMA_MEMORY_USAGE_CPU_TO_GPU, vertices))
                    arc::diagnostics::error("render.vulkan", "Failed to resize per-frame dynamic mesh buffers");
            }
        }
        if (!mesh.dynamic || slot >= mesh.dynamic_vertices.size() || slot >= mesh.uploaded_revisions.size() ||
            mesh.uploaded_revisions[slot] == mesh.vertex_revision)
            continue;
        auto& target = mesh.dynamic_vertices[slot];
        void* mapped{};
        if (target.buffer == VK_NULL_HANDLE || vmaMapMemory(allocator_, target.allocation, &mapped) != VK_SUCCESS)
            continue;
        const auto bytes = buffer_size(mesh.pending_vertices.size(), sizeof(mesh_vertex));
        std::memcpy(mapped, mesh.pending_vertices.data(), static_cast<std::size_t>(bytes));
        vmaFlushAllocation(allocator_, target.allocation, 0, bytes);
        vmaUnmapMemory(allocator_, target.allocation);
        mesh.uploaded_revisions[slot] = mesh.vertex_revision;
    }
}

void vulkan_render_backend::destroy_virtual_mesh_buffers(gpu_virtual_mesh& mesh) noexcept
{
    destroy_buffer(mesh.vertices);
    destroy_buffer(mesh.indices);
    destroy_buffer(mesh.resources);
    destroy_buffer(mesh.nodes);
    destroy_buffer(mesh.clusters_metadata);
    destroy_buffer(mesh.hierarchy_children);
    destroy_buffer(mesh.roots);
    destroy_buffer(mesh.page_table);
    destroy_buffer(mesh.page_heap);
}

void vulkan_render_backend::upload_virtual_mesh(const virtual_mesh_upload_event& event)
{
    if (!event.mesh || event.mesh->vertices.empty() || event.mesh->indices.empty() || event.mesh->clusters.empty())
        return;

    gpu_virtual_mesh mesh;
    mesh.source = event.mesh;
    mesh.resource_generation = event.resource_generation;
    const VkDeviceSize vertex_size = buffer_size(event.mesh->vertices.size(), sizeof(mesh_vertex));
    const VkDeviceSize index_size = buffer_size(event.mesh->indices.size(), sizeof(std::uint32_t));
    if (!upload_buffer(event.mesh->vertices.data(), vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices) ||
        !upload_buffer(event.mesh->indices.data(), index_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices))
    {
        destroy_virtual_mesh_buffers(mesh);
        arc::diagnostics::error("render.vulkan", "Failed to upload virtual mesh '" + event.label + "'");
        return;
    }

    auto tables = make_virtual_geometry_gpu_table_update(event.handle, *event.mesh, event.resource_generation);
    const auto upload_table = [&](const auto& values, VkBufferUsageFlags usage, gpu_buffer& destination)
    {
        using value_type = typename std::decay_t<decltype(values)>::value_type;
        return values.empty() || upload_buffer(values.data(), buffer_size(values.size(), sizeof(value_type)),
                                               usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, destination);
    };
    if (!upload_table(tables.resources, 0, mesh.resources) || !upload_table(tables.nodes, 0, mesh.nodes) ||
        !upload_table(tables.clusters, 0, mesh.clusters_metadata) ||
        !upload_table(tables.children, 0, mesh.hierarchy_children) || !upload_table(tables.roots, 0, mesh.roots))
    {
        destroy_virtual_mesh_buffers(mesh);
        arc::diagnostics::error("render.vulkan",
                                "Failed to upload virtual-geometry metadata for '" + event.label + "'");
        return;
    }

    mesh.page_records = std::move(tables.pages);
    mesh.page_offsets.resize(mesh.page_records.size());
    mesh.resident_page_bytes.resize(mesh.page_records.size());
    VkDeviceSize heap_size{};
    for (std::size_t page_index = 0; page_index < mesh.page_records.size(); ++page_index)
    {
        heap_size = (heap_size + 255u) & ~VkDeviceSize{255u};
        mesh.page_offsets[page_index] = heap_size;
        auto& page = mesh.page_records[page_index];
        page.heap_index = 0;
        page.heap_byte_offset = static_cast<std::uint32_t>(heap_size);
        heap_size += std::max<VkDeviceSize>(page.decoded_size, 1u);
    }
    if (heap_size == 0 ||
        !create_buffer(heap_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_GPU_ONLY, mesh.page_heap))
    {
        destroy_virtual_mesh_buffers(mesh);
        arc::diagnostics::error("render.vulkan",
                                "Failed to allocate virtual-geometry page heap for '" + event.label + "'");
        return;
    }

    std::vector<std::byte> decoded;
    for (std::uint32_t page_index = 0; page_index < event.mesh->pages.size(); ++page_index)
    {
        if (!event.mesh->pages[page_index].root) continue;
        if (!decode_virtual_geometry_page(*event.mesh, page_index, decoded) ||
            !upload_buffer_region(decoded.data(), decoded.size(), mesh.page_heap, mesh.page_offsets[page_index]))
        {
            destroy_virtual_mesh_buffers(mesh);
            arc::diagnostics::error("render.vulkan",
                                    "Failed to publish pinned virtual-geometry root page for '" + event.label + "'");
            return;
        }
        mesh.resident_page_bytes[page_index] =
            std::make_shared<const std::vector<std::byte>>(decoded.begin(), decoded.end());
    }
    if (!upload_table(mesh.page_records, VK_BUFFER_USAGE_TRANSFER_DST_BIT, mesh.page_table))
    {
        destroy_virtual_mesh_buffers(mesh);
        arc::diagnostics::error("render.vulkan",
                                "Failed to upload virtual-geometry page table for '" + event.label + "'");
        return;
    }

    mesh.index_count = static_cast<std::uint32_t>(event.mesh->indices.size());
    mesh.clusters = event.mesh->clusters;
    const std::uint64_t key = resource_key(event.handle);
    if (auto found = virtual_meshes_.find(key); found != virtual_meshes_.end())
    {
        auto retired = std::move(found->second);
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, retired]() mutable { destroy_virtual_mesh_buffers(retired); });
    }
    virtual_meshes_[key] = std::move(mesh);
    virtual_geometry_tables_dirty_ = true;
}

void vulkan_render_backend::upload_virtual_geometry_page(const virtual_geometry_page_upload_event& event)
{
    virtual_geometry_page_upload_result result{.resource = event.upload.resource,
                                               .resource_generation = event.upload.resource_generation,
                                               .page_index = event.upload.page_index,
                                               .compressed_cpu_bytes = event.upload.compressed_cpu_bytes};
    const auto found = virtual_meshes_.find(resource_key(event.upload.resource));
    if (found == virtual_meshes_.end() || found->second.resource_generation != event.upload.resource_generation ||
        !event.upload.decoded_bytes || event.upload.page_index >= found->second.page_records.size())
    {
        frame_virtual_geometry_upload_results_.push_back(result);
        return;
    }
    auto& mesh = found->second;
    auto& page = mesh.page_records[event.upload.page_index];
    if (event.upload.decoded_bytes->size() != page.decoded_size)
    {
        frame_virtual_geometry_upload_results_.push_back(result);
        return;
    }
    result.gpu_bytes = page.decoded_size;
    if (!upload_buffer_region(event.upload.decoded_bytes->data(), event.upload.decoded_bytes->size(), mesh.page_heap,
                              mesh.page_offsets[event.upload.page_index]))
    {
        frame_virtual_geometry_upload_results_.push_back(result);
        return;
    }
    page.flags = static_cast<virtual_geometry_gpu_page_flag>(
        static_cast<std::uint32_t>(page.flags) | static_cast<std::uint32_t>(virtual_geometry_gpu_page_flag::resident));
    const auto offset = static_cast<VkDeviceSize>(event.upload.page_index) * sizeof(virtual_geometry_gpu_page_record);
    if (!upload_buffer_region(&page, sizeof(page), mesh.page_table, offset))
        arc::diagnostics::warn("render.vulkan", "Failed to update virtual-geometry page-table residency");
    else
    {
        result.succeeded = true;
        mesh.resident_page_bytes[event.upload.page_index] = event.upload.decoded_bytes;
    }
    frame_virtual_geometry_upload_results_.push_back(result);
    if (result.succeeded) virtual_geometry_tables_dirty_ = true;
}

void vulkan_render_backend::retire_virtual_mesh(virtual_mesh_handle handle)
{
    const auto found = virtual_meshes_.find(resource_key(handle));
    if (found == virtual_meshes_.end()) return;
    auto retired = std::move(found->second);
    virtual_meshes_.erase(found);
    virtual_geometry_tables_dirty_ = true;
    deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                             [this, retired]() mutable { destroy_virtual_mesh_buffers(retired); });
}

} // namespace arc::render::vulkan::backend_detail
