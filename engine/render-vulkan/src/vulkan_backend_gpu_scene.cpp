#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"

namespace arc::render::vulkan::backend_detail
{
packed_gpu_scene_instance vulkan_render_backend::pack_gpu_scene_instance(const gpu_scene_instance& source) const
{
    packed_gpu_scene_instance result{};
    std::copy(source.model.data(), source.model.data() + 16, result.transform.model);
    std::copy(source.previous_model.data(), source.previous_model.data() + 16, result.transform.previous_model);
    for (std::uint32_t component = 0; component < 3; ++component)
    {
        result.visibility.bounds_min[component] = source.world_bounds.min[component];
        result.visibility.bounds_max[component] = source.world_bounds.max[component];
    }
    resource_handle geometry{};
    switch (source.geometry_kind)
    {
        case gpu_scene_geometry_kind::mesh:
        case gpu_scene_geometry_kind::skinned_mesh:
            geometry = source.mesh;
            break;
        case gpu_scene_geometry_kind::terrain:
            geometry = source.terrain;
            break;
        case gpu_scene_geometry_kind::virtual_mesh:
            geometry = source.virtual_mesh;
            break;
    }
    result.visibility.geometry[0] = geometry.index;
    result.visibility.geometry[1] = geometry.generation;
    result.visibility.geometry[2] = source.submesh_or_cluster;
    result.visibility.geometry[3] = static_cast<std::uint32_t>(source.geometry_kind);
    result.visibility.material_flags[0] = source.material.index;
    result.visibility.material_flags[1] = source.material.generation;
    result.visibility.material_flags[2] = source.render_layer_mask;
    result.visibility.material_flags[3] = static_cast<std::uint32_t>(source.flags);
    result.visibility.draw_metadata[3] = source.object_id.valid() ? source.object_id.index + 1u : 0u;
    if (source.geometry_kind == gpu_scene_geometry_kind::mesh ||
        source.geometry_kind == gpu_scene_geometry_kind::skinned_mesh)
    {
        const auto found = meshes_.find(resource_key(source.mesh));
        if (found != meshes_.end()) result.visibility.draw_metadata[0] = found->second.index_count;
        const auto& geometry_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::geometry)];
        const auto table_offset = static_cast<std::size_t>(source.mesh.index) * geometry_table.element_stride;
        gpu_geometry_table_record geometry_record{};
        const bool valid_geometry = source.geometry_kind == gpu_scene_geometry_kind::mesh &&
                                    geometry_table.element_stride == sizeof(geometry_record) &&
                                    source.mesh.index < geometry_table.live.size() &&
                                    geometry_table.live[source.mesh.index] &&
                                    geometry_table.generations[source.mesh.index] == source.mesh.generation &&
                                    table_offset <= geometry_table.mirror.size() &&
                                    sizeof(geometry_record) <= geometry_table.mirror.size() - table_offset;
        if (valid_geometry)
        {
            std::memcpy(&geometry_record, geometry_table.mirror.data() + table_offset, sizeof(geometry_record));
            const auto first_index = geometry_record.index_offset / sizeof(std::uint32_t);
            const auto vertex_offset = geometry_record.vertex_offset / sizeof(mesh_vertex);
            if (geometry_record.vertex_stride == sizeof(mesh_vertex) &&
                geometry_record.index_stride == sizeof(std::uint32_t) &&
                first_index <= std::numeric_limits<std::uint32_t>::max() &&
                vertex_offset <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
            {
                result.visibility.draw_metadata[0] = geometry_record.index_count;
                result.visibility.draw_metadata[1] = static_cast<std::uint32_t>(first_index);
                result.visibility.draw_metadata[2] = static_cast<std::uint32_t>(vertex_offset);
                const auto material = materials_.find(resource_key(source.material));
                if (material != materials_.end() && material->second.data.domain == material_domain::surface &&
                    !material->second.data.runtime_program)
                    result.visibility.material_flags[3] |= 1u << 31u;
            }
        }
    }
    else if (source.geometry_kind == gpu_scene_geometry_kind::virtual_mesh)
    {
        const auto found = virtual_meshes_.find(resource_key(source.virtual_mesh));
        if (found != virtual_meshes_.end() && source.submesh_or_cluster < found->second.clusters.size())
        {
            const auto& cluster = found->second.clusters[source.submesh_or_cluster];
            result.visibility.draw_metadata[0] = cluster.index_count;
            result.visibility.draw_metadata[1] = cluster.first_index;
        }
    }
    result.visibility.distance_error[0] = source.maximum_draw_distance;
    result.visibility.distance_error[1] = source.geometry_error_scale;
    result.visibility.material_attribute[0] = source.material_attribute_texture.index;
    result.visibility.material_attribute[1] = source.material_attribute_texture.generation;
    return result;
}

std::size_t vulkan_render_backend::gpu_table_offset(gpu_resource_table_kind table) noexcept
{
    return static_cast<std::size_t>(table);
}

void vulkan_render_backend::apply_gpu_resource_table_update(const gpu_resource_table_update_event& event)
{
    if (!event.batch) return;
    const auto& batch = *event.batch;
    const auto table_index = gpu_table_offset(batch.table);
    if (table_index >= gpu_resource_tables_.size() || batch.element_stride == 0u) return;
    auto& table = gpu_resource_tables_[table_index];
    if (table.element_stride != 0u && table.element_stride != batch.element_stride)
    {
        last_profile_.gpu_scene.fallback_reason =
            "GPU resource table stride changed unexpectedly; retaining the prior table generation";
        return;
    }
    table.element_stride = batch.element_stride;
    table.table_generation = batch.table_generation;
    const auto required_bytes = static_cast<std::size_t>(batch.capacity) * batch.element_stride;
    if (required_bytes > table.mirror.size()) table.mirror.resize(required_bytes);
    if (batch.capacity > table.generations.size())
    {
        table.generations.resize(batch.capacity);
        table.live.resize(batch.capacity);
    }

    for (const auto& update : batch.updates)
    {
        if (update.slot >= table.generations.size()) continue;
        const auto destination_offset = static_cast<std::size_t>(update.slot) * batch.element_stride;
        if (update.kind == gpu_table_update_kind::reset)
        {
            std::fill(table.mirror.begin(), table.mirror.end(), std::byte{});
            std::fill(table.generations.begin(), table.generations.end(), 0u);
            std::fill(table.live.begin(), table.live.end(), false);
            table.live_entries = 0u;
        }
        else if (update.kind == gpu_table_update_kind::tombstone)
        {
            std::fill_n(table.mirror.begin() + static_cast<std::ptrdiff_t>(destination_offset), batch.element_stride,
                        std::byte{});
            if (table.live[update.slot]) --table.live_entries;
            table.live[update.slot] = false;
            table.generations[update.slot] = update.generation;
        }
        else if (update.payload_size == batch.element_stride && update.payload_offset <= batch.payload.size() &&
                 update.payload_size <= batch.payload.size() - update.payload_offset)
        {
            std::copy_n(batch.payload.begin() + static_cast<std::ptrdiff_t>(update.payload_offset), update.payload_size,
                        table.mirror.begin() + static_cast<std::ptrdiff_t>(destination_offset));
            if (!table.live[update.slot]) ++table.live_entries;
            table.live[update.slot] = true;
            table.generations[update.slot] = update.generation;
        }
    }
    if (!batch.updates.empty()) table.dirty = true;
    if (!batch.updates.empty() &&
        (batch.table == gpu_resource_table_kind::material || batch.table == gpu_resource_table_kind::texture))
    {
        virtual_geometry_material_descriptors_dirty_ = true;
        gpu_bindless_descriptors_dirty_ = true;
    }

    if (batch.geometry_heap_generation != 0u)
    {
        shared_geometry_buffers_.generation = batch.geometry_heap_generation;
        if (batch.vertex_heap_capacity > shared_geometry_buffers_.vertex_mirror.size())
            shared_geometry_buffers_.vertex_mirror.resize(static_cast<std::size_t>(batch.vertex_heap_capacity));
        if (batch.index_heap_capacity > shared_geometry_buffers_.index_mirror.size())
            shared_geometry_buffers_.index_mirror.resize(static_cast<std::size_t>(batch.index_heap_capacity));
        for (const auto& update : batch.heap_updates)
        {
            auto& mirror =
                update.index_heap ? shared_geometry_buffers_.index_mirror : shared_geometry_buffers_.vertex_mirror;
            if (update.destination_offset > mirror.size() || update.payload_offset > batch.heap_payload.size() ||
                update.payload_size > mirror.size() - update.destination_offset ||
                update.payload_size > batch.heap_payload.size() - update.payload_offset)
                continue;
            std::copy_n(batch.heap_payload.begin() + static_cast<std::ptrdiff_t>(update.payload_offset),
                        update.payload_size, mirror.begin() + static_cast<std::ptrdiff_t>(update.destination_offset));
            if (update.index_heap)
                shared_geometry_buffers_.indices_dirty = true;
            else
                shared_geometry_buffers_.vertices_dirty = true;
        }
    }

    auto& profile = last_profile_.gpu_scene;
    profile.uploaded_ranges += static_cast<std::uint32_t>(batch.dirty_ranges.size());
    profile.uploaded_bytes += batch.payload.size() + batch.heap_payload.size();
}

bool vulkan_render_backend::replace_gpu_mirror_buffer(gpu_buffer& destination, std::span<const std::byte> mirror,
                                                      VkBufferUsageFlags usage)
{
    if (mirror.empty()) return true;
    gpu_buffer replacement{};
    if (!create_buffer(mirror.size(), usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                       replacement))
        return false;
    void* mapped{};
    if (vmaMapMemory(allocator_, replacement.allocation, &mapped) != VK_SUCCESS)
    {
        destroy_buffer(replacement);
        return false;
    }
    std::memcpy(mapped, mirror.data(), mirror.size());
    vmaFlushAllocation(allocator_, replacement.allocation, 0, mirror.size());
    vmaUnmapMemory(allocator_, replacement.allocation);
    auto retired = destination;
    destination = replacement;
    if (retired.buffer != VK_NULL_HANDLE)
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, retired]() mutable { destroy_buffer(retired); });
    return true;
}

bool vulkan_render_backend::flush_gpu_resource_tables()
{
    bool succeeded = true;
    for (auto& table : gpu_resource_tables_)
    {
        if (!table.dirty) continue;
        if (replace_gpu_mirror_buffer(table.storage, table.mirror, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
            table.dirty = false;
        else
            succeeded = false;
    }
    if (shared_geometry_buffers_.vertices_dirty)
    {
        if (replace_gpu_mirror_buffer(shared_geometry_buffers_.vertices, shared_geometry_buffers_.vertex_mirror,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT))
            shared_geometry_buffers_.vertices_dirty = false;
        else
            succeeded = false;
    }
    if (shared_geometry_buffers_.indices_dirty)
    {
        if (replace_gpu_mirror_buffer(shared_geometry_buffers_.indices, shared_geometry_buffers_.index_mirror,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT))
            shared_geometry_buffers_.indices_dirty = false;
        else
            succeeded = false;
    }

    auto& profile = last_profile_.gpu_scene;
    profile.geometry_table_entries =
        gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::geometry)].live_entries;
    profile.material_table_entries =
        gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::material)].live_entries;
    profile.texture_table_entries =
        gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::texture)].live_entries;
    profile.sampler_table_entries =
        gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::sampler)].live_entries;
    profile.skin_palette_table_entries =
        gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::skin_palette)].live_entries;
    profile.shared_vertex_heap_bytes = shared_geometry_buffers_.vertex_mirror.size();
    profile.shared_index_heap_bytes = shared_geometry_buffers_.index_mirror.size();
    return succeeded;
}

void vulkan_render_backend::destroy_gpu_resource_tables() noexcept
{
    for (auto& table : gpu_resource_tables_)
    {
        destroy_buffer(table.storage);
        table = {};
    }
    destroy_buffer(shared_geometry_buffers_.vertices);
    destroy_buffer(shared_geometry_buffers_.indices);
    shared_geometry_buffers_ = {};
}

bool vulkan_render_backend::ensure_gpu_scene_buffer(std::uint32_t required_capacity)
{
    if (required_capacity <= gpu_scene_capacity_ && gpu_scene_visibility_buffer_.buffer != VK_NULL_HANDLE &&
        gpu_scene_transform_buffer_.buffer != VK_NULL_HANDLE)
        return true;
    const std::uint32_t new_capacity = std::max(256u, std::bit_ceil(std::max(required_capacity, 1u)));
    gpu_buffer replacement_visibility{};
    gpu_buffer replacement_transforms{};
    const auto visibility_bytes = static_cast<VkDeviceSize>(new_capacity) * sizeof(gpu_scene_visibility_record);
    const auto transform_bytes = static_cast<VkDeviceSize>(new_capacity) * sizeof(gpu_scene_transform_record);
    if (!create_buffer(visibility_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_CPU_TO_GPU, replacement_visibility) ||
        !create_buffer(transform_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_CPU_TO_GPU, replacement_transforms))
    {
        destroy_buffer(replacement_visibility);
        destroy_buffer(replacement_transforms);
        return false;
    }

    auto retired_visibility = gpu_scene_visibility_buffer_;
    auto retired_transforms = gpu_scene_transform_buffer_;
    const auto retired_capacity = gpu_scene_capacity_;
    gpu_scene_visibility_buffer_ = replacement_visibility;
    gpu_scene_transform_buffer_ = replacement_transforms;
    gpu_scene_capacity_ = new_capacity;
    gpu_scene_visibility_mirror_.resize(new_capacity);
    gpu_scene_transform_mirror_.resize(new_capacity);
    gpu_visibility_descriptors_dirty_ = true;
    virtual_geometry_traversal_descriptors_dirty_ = true;
    virtual_geometry_raster_descriptors_dirty_ = true;
    virtual_geometry_material_descriptors_dirty_ = true;
    gpu_bindless_descriptors_dirty_ = true;

    const auto upload_mirror = [&](gpu_buffer& destination, const auto& mirror) -> bool
    {
        void* mapped{};
        if (vmaMapMemory(allocator_, destination.allocation, &mapped) != VK_SUCCESS) return false;
        const auto byte_count =
            static_cast<VkDeviceSize>(mirror.size()) * sizeof(typename std::decay_t<decltype(mirror)>::value_type);
        std::memcpy(mapped, mirror.data(), static_cast<std::size_t>(byte_count));
        vmaFlushAllocation(allocator_, destination.allocation, 0, byte_count);
        vmaUnmapMemory(allocator_, destination.allocation);
        return true;
    };
    if (!upload_mirror(gpu_scene_visibility_buffer_, gpu_scene_visibility_mirror_) ||
        !upload_mirror(gpu_scene_transform_buffer_, gpu_scene_transform_mirror_))
    {
        auto failed_visibility = gpu_scene_visibility_buffer_;
        auto failed_transforms = gpu_scene_transform_buffer_;
        gpu_scene_visibility_buffer_ = retired_visibility;
        gpu_scene_transform_buffer_ = retired_transforms;
        gpu_scene_capacity_ = retired_capacity;
        destroy_buffer(failed_visibility);
        destroy_buffer(failed_transforms);
        return false;
    }
    if (retired_visibility.buffer != VK_NULL_HANDLE || retired_transforms.buffer != VK_NULL_HANDLE)
    {
        deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                 [this, retired_visibility, retired_transforms]() mutable
                                 {
                                     destroy_buffer(retired_visibility);
                                     destroy_buffer(retired_transforms);
                                 });
    }
    return true;
}

void vulkan_render_backend::apply_gpu_scene_update(const gpu_scene_update_event& event)
{
    if (!event.batch) return;
    const auto& batch = *event.batch;
    auto& profile = last_profile_.gpu_scene;
    profile.enabled = true;
    profile.hzb_occlusion = resolved_config_.features.hzb_occlusion;
    profile.submission = resolved_config_.features.submission;
    profile.binding_model = resolved_config_.features.gpu_binding_model;
    profile.capacity = batch.capacity;
    profile.active_instances = batch.active_instance_count;
    profile.geometry_table_entries =
        static_cast<std::uint32_t>(meshes_.size() + virtual_meshes_.size() + terrains_.size());
    profile.material_table_entries = static_cast<std::uint32_t>(materials_.size());
    profile.texture_table_entries = static_cast<std::uint32_t>(textures_.size());
    profile.uploaded_ranges += static_cast<std::uint32_t>(batch.dirty_ranges.size());
    if (!ensure_gpu_scene_buffer(batch.capacity))
    {
        profile.fallback_reason = "GPU Scene buffer allocation failed; using CPU draw submission";
        return;
    }

    bool reset{};
    for (const auto& update : batch.updates)
    {
        if (update.kind == gpu_scene_update_kind::reset)
        {
            reset = true;
        }
        else if (update.handle.index < gpu_scene_visibility_mirror_.size())
        {
            if (update.kind == gpu_scene_update_kind::upsert)
            {
                const auto packed = pack_gpu_scene_instance(update.instance);
                gpu_scene_visibility_mirror_[update.handle.index] = packed.visibility;
                gpu_scene_transform_mirror_[update.handle.index] = packed.transform;
                ++profile.uploaded_instances;
            }
            else
            {
                gpu_scene_visibility_mirror_[update.handle.index] = {};
                gpu_scene_transform_mirror_[update.handle.index] = {};
                const auto key = (static_cast<std::uint64_t>(update.handle.generation) << 32u) | update.handle.index;
                if (auto skinned = gpu_skinned_instances_.find(key); skinned != gpu_skinned_instances_.end())
                {
                    destroy_gpu_skinned_instance(skinned->second);
                    gpu_skinned_instances_.erase(skinned);
                }
                ++profile.destroyed_instances;
            }
        }
    }

    if (batch.dirty_ranges.empty()) return;
    void* mapped_visibility{};
    void* mapped_transforms{};
    const auto visibility_map_result =
        vmaMapMemory(allocator_, gpu_scene_visibility_buffer_.allocation, &mapped_visibility);
    const auto transform_map_result =
        vmaMapMemory(allocator_, gpu_scene_transform_buffer_.allocation, &mapped_transforms);
    if (visibility_map_result != VK_SUCCESS || transform_map_result != VK_SUCCESS)
    {
        if (visibility_map_result == VK_SUCCESS) vmaUnmapMemory(allocator_, gpu_scene_visibility_buffer_.allocation);
        if (transform_map_result == VK_SUCCESS) vmaUnmapMemory(allocator_, gpu_scene_transform_buffer_.allocation);
        profile.fallback_reason = "GPU Scene buffer mapping failed; retaining the previous generation";
        return;
    }
    for (const auto& range : batch.dirty_ranges)
    {
        if (range.count == 0u || range.end() > gpu_scene_visibility_mirror_.size()) continue;
        const auto visibility_offset = static_cast<VkDeviceSize>(range.first) * sizeof(gpu_scene_visibility_record);
        const auto visibility_bytes = static_cast<VkDeviceSize>(range.count) * sizeof(gpu_scene_visibility_record);
        const auto transform_offset = static_cast<VkDeviceSize>(range.first) * sizeof(gpu_scene_transform_record);
        const auto transform_bytes = static_cast<VkDeviceSize>(range.count) * sizeof(gpu_scene_transform_record);
        std::memcpy(static_cast<std::byte*>(mapped_visibility) + visibility_offset,
                    gpu_scene_visibility_mirror_.data() + range.first, static_cast<std::size_t>(visibility_bytes));
        std::memcpy(static_cast<std::byte*>(mapped_transforms) + transform_offset,
                    gpu_scene_transform_mirror_.data() + range.first, static_cast<std::size_t>(transform_bytes));
        vmaFlushAllocation(allocator_, gpu_scene_visibility_buffer_.allocation, visibility_offset, visibility_bytes);
        vmaFlushAllocation(allocator_, gpu_scene_transform_buffer_.allocation, transform_offset, transform_bytes);
        profile.uploaded_bytes += visibility_bytes + transform_bytes;
    }
    vmaUnmapMemory(allocator_, gpu_scene_visibility_buffer_.allocation);
    vmaUnmapMemory(allocator_, gpu_scene_transform_buffer_.allocation);
    if (reset) profile.history_valid = false;
}

void vulkan_render_backend::destroy_gpu_skinned_instance(gpu_skinned_instance& instance) noexcept
{
    for (auto& vertices : instance.current_vertices)
        destroy_buffer(vertices);
    for (auto& vertices : instance.previous_vertices)
        destroy_buffer(vertices);
    if (gpu_skinning_descriptor_pool_ != VK_NULL_HANDLE)
    {
        std::vector<VkDescriptorSet> allocated;
        std::ranges::copy_if(instance.descriptor_sets, std::back_inserter(allocated),
                             [](VkDescriptorSet set) { return set != VK_NULL_HANDLE; });
        if (!allocated.empty())
            vkFreeDescriptorSets(device_, gpu_skinning_descriptor_pool_, static_cast<std::uint32_t>(allocated.size()),
                                 allocated.data());
    }
    instance = {};
}

bool vulkan_render_backend::ensure_gpu_skinning_pipeline()
{
    if (gpu_skinning_pipeline_ != VK_NULL_HANDLE) return true;
    if (!resolved_config_.features.gpu_skinning) return false;

    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
    for (std::uint32_t binding = 0; binding < bindings.size(); ++binding)
        bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo descriptor_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptor_layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr, &gpu_skinning_descriptor_set_layout_) !=
        VK_SUCCESS)
        return false;

    constexpr std::uint32_t maximum_skinning_sets = 8192u;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         maximum_skinning_sets * static_cast<std::uint32_t>(bindings.size())};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = maximum_skinning_sets;
    pool.poolSizeCount = 1u;
    pool.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &gpu_skinning_descriptor_pool_) != VK_SUCCESS) return false;

    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(std::uint32_t) * 4u};
    VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout.setLayoutCount = 1u;
    pipeline_layout.pSetLayouts = &gpu_skinning_descriptor_set_layout_;
    pipeline_layout.pushConstantRangeCount = 1u;
    pipeline_layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &gpu_skinning_pipeline_layout_) != VK_SUCCESS)
        return false;

    const auto shader =
        create_shader_module(builtin::gpu_visible_skinning_comp_spv, std::size(builtin::gpu_visible_skinning_comp_spv));
    if (shader == VK_NULL_HANDLE) return false;
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.stage = stage;
    pipeline.layout = gpu_skinning_pipeline_layout_;
    const auto result =
        vkCreateComputePipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr, &gpu_skinning_pipeline_);
    vkDestroyShaderModule(device_, shader, nullptr);
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_gpu_skinned_instance(gpu_scene_instance_handle handle, mesh_handle mesh_handle_value,
                                                        buffer_handle palette_handle, std::uint32_t vertex_count,
                                                        gpu_skinned_instance*& result)
{
    const auto key = (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
    auto& instance = gpu_skinned_instances_[key];
    const auto frame_count = frame_resource_count();
    if (instance.mesh != mesh_handle_value || instance.palette != palette_handle ||
        instance.vertex_count != vertex_count || instance.current_vertices.size() != frame_count ||
        instance.cpu_fallback)
    {
        destroy_gpu_skinned_instance(instance);
        instance.mesh = mesh_handle_value;
        instance.palette = palette_handle;
        instance.vertex_count = vertex_count;
        instance.cpu_fallback = false;
        instance.current_vertices.resize(frame_count);
        instance.previous_vertices.resize(frame_count);
        instance.descriptor_sets.resize(frame_count);
        const auto byte_size = static_cast<VkDeviceSize>(vertex_count) * sizeof(mesh_vertex);
        for (std::uint32_t frame = 0; frame < frame_count; ++frame)
        {
            if (!create_buffer(byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                               VMA_MEMORY_USAGE_GPU_ONLY, instance.current_vertices[frame]) ||
                !create_buffer(byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                               VMA_MEMORY_USAGE_GPU_ONLY, instance.previous_vertices[frame]))
            {
                destroy_gpu_skinned_instance(instance);
                gpu_skinned_instances_.erase(key);
                return false;
            }
        }
        std::vector<VkDescriptorSetLayout> layouts(frame_count, gpu_skinning_descriptor_set_layout_);
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = gpu_skinning_descriptor_pool_;
        allocate.descriptorSetCount = frame_count;
        allocate.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(device_, &allocate, instance.descriptor_sets.data()) != VK_SUCCESS)
        {
            destroy_gpu_skinned_instance(instance);
            gpu_skinned_instances_.erase(key);
            return false;
        }
    }
    result = &instance;
    return true;
}

bool vulkan_render_backend::ensure_cpu_skinned_instance(gpu_scene_instance_handle handle, mesh_handle mesh_handle_value,
                                                        buffer_handle palette_handle, std::uint32_t vertex_count,
                                                        gpu_skinned_instance*& result)
{
    const auto key = (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
    auto& instance = gpu_skinned_instances_[key];
    const auto frame_count = frame_resource_count();
    if (instance.mesh != mesh_handle_value || instance.palette != palette_handle ||
        instance.vertex_count != vertex_count || instance.current_vertices.size() != frame_count ||
        !instance.cpu_fallback)
    {
        destroy_gpu_skinned_instance(instance);
        instance.mesh = mesh_handle_value;
        instance.palette = palette_handle;
        instance.vertex_count = vertex_count;
        instance.cpu_fallback = true;
        instance.current_vertices.resize(frame_count);
        instance.previous_vertices.resize(frame_count);
        const auto byte_size = static_cast<VkDeviceSize>(vertex_count) * sizeof(mesh_vertex);
        for (std::uint32_t frame = 0; frame < frame_count; ++frame)
        {
            if (!create_buffer(byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_TO_GPU, instance.current_vertices[frame]) ||
                !create_buffer(byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                               VMA_MEMORY_USAGE_CPU_TO_GPU, instance.previous_vertices[frame]))
            {
                destroy_gpu_skinned_instance(instance);
                gpu_skinned_instances_.erase(key);
                return false;
            }
        }
    }
    result = &instance;
    return true;
}

void vulkan_render_backend::update_cpu_skinned_vertices()
{
    const auto slot = current_frame_slot();
    std::unordered_set<std::uint64_t> updated;
    std::vector<mesh_vertex> current;
    std::vector<mesh_vertex> previous;
    for (const auto& draw : frame_draws_)
    {
        if (!draw.gpu_scene_instance.valid() || !draw.skin_palette.valid()) continue;
        const auto instance_key =
            (static_cast<std::uint64_t>(draw.gpu_scene_instance.generation) << 32u) | draw.gpu_scene_instance.index;
        if (!updated.insert(instance_key).second) continue;
        const auto mesh = meshes_.find(resource_key(draw.mesh));
        const auto palette = skin_palettes_.find(resource_key(draw.skin_palette));
        if (mesh == meshes_.end() || palette == skin_palettes_.end() ||
            mesh->second.source_vertices.size() != mesh->second.skin_influences.size() ||
            mesh->second.source_vertices.empty())
            continue;

        gpu_skinned_instance* instance{};
        if (!ensure_cpu_skinned_instance(draw.gpu_scene_instance, draw.mesh, draw.skin_palette,
                                         mesh->second.vertex_count, instance) ||
            slot >= instance->current_vertices.size() || slot >= instance->previous_vertices.size())
            continue;
        current.resize(mesh->second.source_vertices.size());
        previous.resize(mesh->second.source_vertices.size());
        const auto requested_joint_count =
            draw.skin_joint_count == 0u ? palette->second.joint_count : draw.skin_joint_count;
        const auto joint_count = std::min<std::size_t>(requested_joint_count, palette->second.current_matrices.size());
        const auto previous_joint_count = std::min(joint_count, palette->second.previous_matrices.size());
        if (!skin_mesh_vertices(mesh->second.source_vertices, mesh->second.skin_influences,
                                std::span{palette->second.current_matrices}.first(joint_count), current) ||
            !skin_mesh_vertices(mesh->second.source_vertices, mesh->second.skin_influences,
                                std::span{palette->second.previous_matrices}.first(previous_joint_count), previous))
            continue;

        const auto upload = [&](gpu_buffer& target, const std::vector<mesh_vertex>& vertices)
        {
            void* mapped{};
            if (target.buffer == VK_NULL_HANDLE || vmaMapMemory(allocator_, target.allocation, &mapped) != VK_SUCCESS)
                return false;
            const auto bytes = buffer_size(vertices.size(), sizeof(mesh_vertex));
            std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(bytes));
            vmaFlushAllocation(allocator_, target.allocation, 0u, bytes);
            vmaUnmapMemory(allocator_, target.allocation);
            return true;
        };
        if (!upload(instance->current_vertices[slot], current) || !upload(instance->previous_vertices[slot], previous))
        {
            arc::diagnostics::warn("render.vulkan", "CPU skinning upload failed; using bind-pose geometry");
            destroy_gpu_skinned_instance(*instance);
            gpu_skinned_instances_.erase(instance_key);
        }
    }
}

void vulkan_render_backend::dispatch_gpu_skinning(VkCommandBuffer command_buffer)
{
    if (!ensure_gpu_skinning_pipeline() || !gpu_visibility_active_ || gpu_visibility_commands_.buffer == VK_NULL_HANDLE)
        return;
    const auto slot = current_frame_slot();
    std::uint32_t skinned_draws{};
    for (const auto& draw : frame_draws_)
    {
        if (!draw.gpu_scene_instance.valid() || !draw.skin_palette.valid()) continue;
        const auto mesh = meshes_.find(resource_key(draw.mesh));
        const auto palette = skin_palettes_.find(resource_key(draw.skin_palette));
        if (mesh == meshes_.end() || palette == skin_palettes_.end() ||
            mesh->second.skin_vertices.buffer == VK_NULL_HANDLE || mesh->second.vertex_count == 0u)
            continue;
        gpu_skinned_instance* instance{};
        if (!ensure_gpu_skinned_instance(draw.gpu_scene_instance, draw.mesh, draw.skin_palette,
                                         mesh->second.vertex_count, instance) ||
            slot >= instance->descriptor_sets.size())
            continue;

        const auto source = mesh_vertex_buffer(mesh->second);
        if (source == VK_NULL_HANDLE) continue;
        const std::array buffer_infos{
            VkDescriptorBufferInfo{source, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{mesh->second.skin_vertices.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{palette->second.current.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{palette->second.previous.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{instance->current_vertices[slot].buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{instance->previous_vertices[slot].buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_visibility_commands_.buffer, 0u, VK_WHOLE_SIZE}};
        std::array<VkWriteDescriptorSet, buffer_infos.size()> writes{};
        for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
        {
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = instance->descriptor_sets[slot];
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1u;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &buffer_infos[binding];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        const auto requested_joint_count =
            draw.skin_joint_count == 0u ? palette->second.joint_count : draw.skin_joint_count;
        const std::array constants{
            mesh->second.vertex_count, std::min(requested_joint_count, palette->second.joint_count),
            draw.casts_shadows ? std::numeric_limits<std::uint32_t>::max() : draw.gpu_scene_instance.index,
            static_cast<std::uint32_t>(sizeof(mesh_vertex) / sizeof(std::uint32_t))};
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_skinning_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_skinning_pipeline_layout_, 0u, 1u,
                                &instance->descriptor_sets[slot], 0u, nullptr);
        vkCmdPushConstants(command_buffer, gpu_skinning_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           sizeof(constants), constants.data());
        vkCmdDispatch(command_buffer, (mesh->second.vertex_count + 63u) / 64u, 1u, 1u);
        ++skinned_draws;
    }
    if (skinned_draws == 0u) return;
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0u,
                         1u, &barrier, 0u, nullptr, 0u, nullptr);
    last_profile_.gpu_scene.skinning_milliseconds = 0.0;
}

void vulkan_render_backend::destroy_gpu_visibility_resources()
{
    destroy_virtual_geometry_traversal_resources();
    for (auto& [_, instance] : gpu_skinned_instances_)
        destroy_gpu_skinned_instance(instance);
    gpu_skinned_instances_.clear();
    destroy_buffer(gpu_visibility_commands_);
    destroy_buffer(gpu_visibility_counters_);
    for (auto& frame : gpu_visibility_feedback_frames_)
        destroy_buffer(frame.counters);
    gpu_visibility_feedback_frames_.clear();
    if (gpu_visibility_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, gpu_visibility_pipeline_, nullptr);
    if (gpu_transparent_sort_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, gpu_transparent_sort_pipeline_, nullptr);
    if (gpu_skinning_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, gpu_skinning_pipeline_, nullptr);
    if (gpu_skinning_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, gpu_skinning_pipeline_layout_, nullptr);
    if (gpu_skinning_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, gpu_skinning_descriptor_pool_, nullptr);
    if (gpu_skinning_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, gpu_skinning_descriptor_set_layout_, nullptr);
    if (gpu_bindless_gbuffer_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, gpu_bindless_gbuffer_pipeline_, nullptr);
    if (gpu_bindless_transparent_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, gpu_bindless_transparent_pipeline_, nullptr);
    if (gpu_bindless_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, gpu_bindless_pipeline_layout_, nullptr);
    if (gpu_bindless_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, gpu_bindless_descriptor_pool_, nullptr);
    if (gpu_bindless_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, gpu_bindless_descriptor_set_layout_, nullptr);
    if (gpu_visibility_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, gpu_visibility_pipeline_layout_, nullptr);
    if (gpu_visibility_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, gpu_visibility_descriptor_pool_, nullptr);
    if (gpu_visibility_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, gpu_visibility_descriptor_set_layout_, nullptr);
    gpu_visibility_pipeline_ = VK_NULL_HANDLE;
    gpu_transparent_sort_pipeline_ = VK_NULL_HANDLE;
    gpu_skinning_pipeline_ = VK_NULL_HANDLE;
    gpu_skinning_pipeline_layout_ = VK_NULL_HANDLE;
    gpu_skinning_descriptor_pool_ = VK_NULL_HANDLE;
    gpu_skinning_descriptor_set_layout_ = VK_NULL_HANDLE;
    gpu_bindless_gbuffer_pipeline_ = VK_NULL_HANDLE;
    gpu_bindless_transparent_pipeline_ = VK_NULL_HANDLE;
    gpu_bindless_pipeline_layout_ = VK_NULL_HANDLE;
    gpu_bindless_descriptor_pool_ = VK_NULL_HANDLE;
    gpu_bindless_descriptor_set_layout_ = VK_NULL_HANDLE;
    gpu_bindless_descriptor_set_ = VK_NULL_HANDLE;
    gpu_visibility_pipeline_layout_ = VK_NULL_HANDLE;
    gpu_visibility_descriptor_pool_ = VK_NULL_HANDLE;
    gpu_visibility_descriptor_set_layout_ = VK_NULL_HANDLE;
    gpu_visibility_descriptor_set_ = VK_NULL_HANDLE;
    gpu_visibility_capacity_ = 0;
    gpu_visibility_active_ = false;
}

void vulkan_render_backend::evict_virtual_geometry_page(const virtual_geometry_page_evict_event& event)
{
    const auto found = virtual_meshes_.find(resource_key(event.eviction.resource));
    if (found == virtual_meshes_.end() || found->second.resource_generation != event.eviction.resource_generation ||
        event.eviction.page_index >= found->second.page_records.size() ||
        event.eviction.page_index >= found->second.resident_page_bytes.size())
        return;
    auto& page = found->second.page_records[event.eviction.page_index];
    if ((static_cast<std::uint32_t>(page.flags) & static_cast<std::uint32_t>(virtual_geometry_gpu_page_flag::root)) !=
        0u)
        return;
    constexpr auto resident_mask = static_cast<std::uint32_t>(virtual_geometry_gpu_page_flag::resident) |
                                   static_cast<std::uint32_t>(virtual_geometry_gpu_page_flag::loading) |
                                   static_cast<std::uint32_t>(virtual_geometry_gpu_page_flag::failed);
    page.flags = static_cast<virtual_geometry_gpu_page_flag>(static_cast<std::uint32_t>(page.flags) & ~resident_mask);
    found->second.resident_page_bytes[event.eviction.page_index].reset();
    virtual_geometry_tables_dirty_ = true;
}

bool vulkan_render_backend::rebuild_virtual_geometry_tables()
{
    std::vector<std::pair<std::uint64_t, gpu_virtual_mesh*>> ordered;
    ordered.reserve(virtual_meshes_.size());
    for (auto& [key, mesh] : virtual_meshes_)
        ordered.emplace_back(key, &mesh);
    std::ranges::sort(ordered, {}, &std::pair<std::uint64_t, gpu_virtual_mesh*>::first);

    std::uint32_t resource_capacity{1u};
    for (const auto& [key, _] : ordered)
        resource_capacity = std::max(resource_capacity, static_cast<std::uint32_t>(key) + 1u);
    virtual_geometry_resource_mirror_.assign(resource_capacity, {});
    virtual_geometry_node_mirror_.clear();
    virtual_geometry_cluster_mirror_.clear();
    virtual_geometry_child_mirror_.clear();
    virtual_geometry_root_mirror_.clear();
    virtual_geometry_page_mirror_.clear();
    virtual_geometry_page_heap_mirror_.clear();

    for (const auto& [key, mesh] : ordered)
    {
        if (!mesh->source) continue;
        const auto resource_index = static_cast<std::uint32_t>(key);
        const auto handle_generation = static_cast<std::uint32_t>(key >> 32u);
        const auto node_base = static_cast<std::uint32_t>(virtual_geometry_node_mirror_.size());
        const auto cluster_base = static_cast<std::uint32_t>(virtual_geometry_cluster_mirror_.size());
        const auto child_base = static_cast<std::uint32_t>(virtual_geometry_child_mirror_.size());
        const auto page_base = static_cast<std::uint32_t>(virtual_geometry_page_mirror_.size());
        const auto root_base = static_cast<std::uint32_t>(virtual_geometry_root_mirror_.size());
        auto tables = make_virtual_geometry_gpu_table_update({resource_index, handle_generation}, *mesh->source,
                                                             mesh->resource_generation);
        if (tables.resources.empty()) continue;
        auto resource = tables.resources.front();
        resource.first_node = node_base;
        resource.first_cluster = cluster_base;
        resource.first_child = child_base;
        resource.first_page = page_base;
        resource.first_root = root_base;
        resource.flags = handle_generation;
        virtual_geometry_resource_mirror_[resource_index] = resource;
        for (auto node : tables.nodes)
        {
            node.first_cluster += cluster_base;
            node.first_child += child_base;
            node.page_index += page_base;
            virtual_geometry_node_mirror_.push_back(node);
        }
        for (auto cluster : tables.clusters)
        {
            cluster.page_index += page_base;
            cluster.hierarchy_node += node_base;
            virtual_geometry_cluster_mirror_.push_back(cluster);
        }
        for (const auto child : tables.children)
            virtual_geometry_child_mirror_.push_back(child + node_base);
        for (const auto root : tables.roots)
            virtual_geometry_root_mirror_.push_back(root + node_base);
        for (std::size_t page_index = 0; page_index < mesh->page_records.size(); ++page_index)
        {
            auto page = mesh->page_records[page_index];
            const auto aligned_offset = (virtual_geometry_page_heap_mirror_.size() + 15u) & ~std::size_t{15u};
            virtual_geometry_page_heap_mirror_.resize(aligned_offset, std::byte{});
            page.heap_index = 0u;
            page.heap_byte_offset = static_cast<std::uint32_t>(aligned_offset);
            if (page_index < mesh->resident_page_bytes.size() && mesh->resident_page_bytes[page_index])
            {
                const auto& bytes = *mesh->resident_page_bytes[page_index];
                virtual_geometry_page_heap_mirror_.insert(virtual_geometry_page_heap_mirror_.end(), bytes.begin(),
                                                          bytes.end());
            }
            virtual_geometry_page_mirror_.push_back(page);
        }
    }

    if (virtual_geometry_node_mirror_.empty()) virtual_geometry_node_mirror_.push_back({});
    if (virtual_geometry_cluster_mirror_.empty()) virtual_geometry_cluster_mirror_.push_back({});
    if (virtual_geometry_child_mirror_.empty()) virtual_geometry_child_mirror_.push_back(0u);
    if (virtual_geometry_root_mirror_.empty()) virtual_geometry_root_mirror_.push_back(0u);
    if (virtual_geometry_page_mirror_.empty()) virtual_geometry_page_mirror_.push_back({});
    if (virtual_geometry_page_heap_mirror_.empty()) virtual_geometry_page_heap_mirror_.resize(4u);

    const auto replace = [&](gpu_buffer& destination, const auto& values)
    {
        return replace_gpu_mirror_buffer(destination, std::as_bytes(std::span{values}),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    };
    const bool succeeded =
        replace(virtual_geometry_resource_buffer_, virtual_geometry_resource_mirror_) &&
        replace(virtual_geometry_node_buffer_, virtual_geometry_node_mirror_) &&
        replace(virtual_geometry_cluster_buffer_, virtual_geometry_cluster_mirror_) &&
        replace(virtual_geometry_child_buffer_, virtual_geometry_child_mirror_) &&
        replace(virtual_geometry_root_buffer_, virtual_geometry_root_mirror_) &&
        replace(virtual_geometry_page_buffer_, virtual_geometry_page_mirror_) &&
        replace_gpu_mirror_buffer(virtual_geometry_page_heap_buffer_, virtual_geometry_page_heap_mirror_,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (succeeded)
    {
        virtual_geometry_tables_dirty_ = false;
        virtual_geometry_traversal_descriptors_dirty_ = true;
        virtual_geometry_raster_descriptors_dirty_ = true;
        virtual_geometry_material_descriptors_dirty_ = true;
    }
    return succeeded;
}

void vulkan_render_backend::destroy_virtual_geometry_traversal_resources()
{
    destroy_buffer(virtual_geometry_resource_buffer_);
    destroy_buffer(virtual_geometry_node_buffer_);
    destroy_buffer(virtual_geometry_cluster_buffer_);
    destroy_buffer(virtual_geometry_child_buffer_);
    destroy_buffer(virtual_geometry_root_buffer_);
    destroy_buffer(virtual_geometry_page_buffer_);
    destroy_buffer(virtual_geometry_page_heap_buffer_);
    destroy_buffer(virtual_geometry_visible_buffer_);
    destroy_buffer(virtual_geometry_request_buffer_);
    destroy_buffer(virtual_geometry_counter_buffer_);
    destroy_buffer(virtual_geometry_raster_bin_buffer_);
    destroy_buffer(virtual_geometry_material_frame_buffer_);
    destroy_graph_image(virtual_geometry_encoded_depth_);
    destroy_graph_image(virtual_geometry_visibility_ids_);
    for (auto& frame : virtual_geometry_feedback_frames_)
    {
        destroy_buffer(frame.requests);
        destroy_buffer(frame.counters);
    }
    virtual_geometry_feedback_frames_.clear();
    if (virtual_geometry_traversal_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, virtual_geometry_traversal_pipeline_, nullptr);
    if (virtual_geometry_traversal_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, virtual_geometry_traversal_pipeline_layout_, nullptr);
    if (virtual_geometry_traversal_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, virtual_geometry_traversal_descriptor_pool_, nullptr);
    if (virtual_geometry_traversal_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, virtual_geometry_traversal_descriptor_set_layout_, nullptr);
    for (const auto pipeline : virtual_geometry_raster_pipelines_)
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline, nullptr);
    if (virtual_geometry_raster_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, virtual_geometry_raster_pipeline_layout_, nullptr);
    if (virtual_geometry_raster_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, virtual_geometry_raster_descriptor_pool_, nullptr);
    if (virtual_geometry_raster_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, virtual_geometry_raster_descriptor_set_layout_, nullptr);
    if (virtual_geometry_material_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, virtual_geometry_material_pipeline_, nullptr);
    if (virtual_geometry_material_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, virtual_geometry_material_pipeline_layout_, nullptr);
    if (virtual_geometry_material_descriptor_pool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device_, virtual_geometry_material_descriptor_pool_, nullptr);
    if (virtual_geometry_material_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, virtual_geometry_material_descriptor_set_layout_, nullptr);
    virtual_geometry_traversal_pipeline_ = VK_NULL_HANDLE;
    virtual_geometry_traversal_pipeline_layout_ = VK_NULL_HANDLE;
    virtual_geometry_traversal_descriptor_pool_ = VK_NULL_HANDLE;
    virtual_geometry_traversal_descriptor_set_layout_ = VK_NULL_HANDLE;
    virtual_geometry_traversal_descriptor_set_ = VK_NULL_HANDLE;
    virtual_geometry_raster_pipelines_.fill(VK_NULL_HANDLE);
    virtual_geometry_raster_pipeline_layout_ = VK_NULL_HANDLE;
    virtual_geometry_raster_descriptor_pool_ = VK_NULL_HANDLE;
    virtual_geometry_raster_descriptor_set_layout_ = VK_NULL_HANDLE;
    virtual_geometry_raster_descriptor_set_ = VK_NULL_HANDLE;
    virtual_geometry_material_pipeline_ = VK_NULL_HANDLE;
    virtual_geometry_material_pipeline_layout_ = VK_NULL_HANDLE;
    virtual_geometry_material_descriptor_pool_ = VK_NULL_HANDLE;
    virtual_geometry_material_descriptor_set_layout_ = VK_NULL_HANDLE;
    virtual_geometry_material_descriptor_set_ = VK_NULL_HANDLE;
    virtual_geometry_visible_capacity_ = 0u;
    virtual_geometry_request_capacity_ = 0u;
    virtual_geometry_raster_bin_capacity_ = 0u;
}

bool vulkan_render_backend::ensure_gpu_visibility_resources()
{
    if (!resolved_config_.features.gpu_driven_rendering || gpu_scene_capacity_ == 0 ||
        gpu_scene_visibility_buffer_.buffer == VK_NULL_HANDLE)
        return false;
    const bool hzb_resources_available =
        resolved_config_.features.hzb_occlusion && ensure_hzb_resources(viewport_width_, viewport_height_);

    if (gpu_visibility_capacity_ < gpu_scene_capacity_ || gpu_visibility_commands_.buffer == VK_NULL_HANDLE)
    {
        const auto capacity = std::max(256u, std::bit_ceil(gpu_scene_capacity_));
        gpu_buffer commands{};
        gpu_buffer counters{};
        if (!create_buffer(static_cast<VkDeviceSize>(capacity) * indexed_indirect_command_stride * 3u,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_GPU_ONLY, commands) ||
            !create_buffer(sizeof(gpu_visibility_counter_data),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                           VMA_MEMORY_USAGE_GPU_ONLY, counters))
        {
            destroy_buffer(commands);
            destroy_buffer(counters);
            return false;
        }
        auto retired_commands = gpu_visibility_commands_;
        auto retired_counters = gpu_visibility_counters_;
        gpu_visibility_commands_ = commands;
        gpu_visibility_counters_ = counters;
        gpu_visibility_capacity_ = capacity;
        gpu_visibility_descriptors_dirty_ = true;
        if (retired_commands.buffer != VK_NULL_HANDLE || retired_counters.buffer != VK_NULL_HANDLE)
        {
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(),
                                     [this, retired_commands, retired_counters]() mutable
                                     {
                                         destroy_buffer(retired_commands);
                                         destroy_buffer(retired_counters);
                                     });
        }
    }

    if (gpu_visibility_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        for (std::uint32_t index = 0; index < 3; ++index)
        {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &gpu_visibility_descriptor_set_layout_) !=
            VK_SUCCESS)
            return false;

        gpu_visibility_descriptors_dirty_ = true;
    }

    if (gpu_visibility_descriptors_dirty_)
    {
        VkDescriptorPool replacement_pool{};
        VkDescriptorSet replacement_set{};
        const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
                                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}};
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 1;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &replacement_pool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = replacement_pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &gpu_visibility_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &replacement_set) != VK_SUCCESS)
        {
            vkDestroyDescriptorPool(device_, replacement_pool, nullptr);
            return false;
        }
        std::array<VkDescriptorBufferInfo, 3> buffers{
            VkDescriptorBufferInfo{gpu_scene_visibility_buffer_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_visibility_commands_.buffer, 0, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_visibility_counters_.buffer, 0, VK_WHOLE_SIZE}};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t index = 0; index < writes.size(); ++index)
        {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = replacement_set;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = &buffers[index];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        const VkDescriptorImageInfo fallback_hzb{white_sampler_, white_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        std::array<VkDescriptorImageInfo, 2> hzb_images{fallback_hzb, fallback_hzb};
        if (hzb_resources_available)
            for (std::size_t index = 0; index < hzb_images.size(); ++index)
                hzb_images[index] = {hzb_sampler_, hzb_history_[index].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet hzb_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        hzb_write.dstSet = replacement_set;
        hzb_write.dstBinding = 3;
        hzb_write.descriptorCount = static_cast<std::uint32_t>(hzb_images.size());
        hzb_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        hzb_write.pImageInfo = hzb_images.data();
        vkUpdateDescriptorSets(device_, 1, &hzb_write, 0, nullptr);
        const auto retired_pool = gpu_visibility_descriptor_pool_;
        gpu_visibility_descriptor_pool_ = replacement_pool;
        gpu_visibility_descriptor_set_ = replacement_set;
        if (retired_pool != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(), [this, retired_pool]()
                                     { vkDestroyDescriptorPool(device_, retired_pool, nullptr); });
        gpu_visibility_descriptors_dirty_ = false;
    }

    if (gpu_visibility_pipeline_ == VK_NULL_HANDLE)
    {
        const auto shader = create_shader_module(builtin::gpu_visibility_indirect_comp_spv,
                                                 std::size(builtin::gpu_visibility_indirect_comp_spv));
        if (shader == VK_NULL_HANDLE) return false;
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpu_visibility_push_constants)};
        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &gpu_visibility_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &gpu_visibility_pipeline_layout_) != VK_SUCCESS)
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
        pipeline.layout = gpu_visibility_pipeline_layout_;
        const auto result =
            vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &gpu_visibility_pipeline_);
        vkDestroyShaderModule(device_, shader, nullptr);
        if (result != VK_SUCCESS) return false;
    }
    if (capabilities_.gpu_transparent_sorting && gpu_transparent_sort_pipeline_ == VK_NULL_HANDLE)
    {
        const auto shader = create_shader_module(builtin::gpu_transparent_sort_comp_spv,
                                                 std::size(builtin::gpu_transparent_sort_comp_spv));
        if (shader == VK_NULL_HANDLE) return false;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = shader;
        pipeline.stage.pName = "main";
        pipeline.layout = gpu_visibility_pipeline_layout_;
        const auto result = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                     &gpu_transparent_sort_pipeline_);
        vkDestroyShaderModule(device_, shader, nullptr);
        if (result != VK_SUCCESS) return false;
    }
    return true;
}

bool vulkan_render_backend::ensure_gpu_visibility_feedback_frame(gpu_visibility_feedback_frame& frame)
{
    return frame.counters.buffer != VK_NULL_HANDLE ||
           create_buffer(sizeof(gpu_visibility_counter_data), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VMA_MEMORY_USAGE_GPU_TO_CPU, frame.counters);
}

void vulkan_render_backend::apply_gpu_visibility_statistics(const gpu_visibility_statistics& statistics)
{
    last_profile_.gpu_scene.candidate_instances = statistics.candidates;
    last_profile_.gpu_scene.visible_instances = statistics.visible;
    last_profile_.gpu_scene.frustum_rejected = statistics.frustum_rejected;
    last_profile_.gpu_scene.distance_rejected = statistics.distance_rejected;
    last_profile_.gpu_scene.occlusion_rejected = statistics.occlusion_rejected;
    last_profile_.gpu_scene.active_pipeline_bins = statistics.active_bins;
    last_profile_.gpu_scene.indirect_commands = statistics.indirect_commands;
    last_profile_.gpu_scene.transparent_records = statistics.transparent_records;
    last_profile_.gpu_scene.overflow_records = statistics.overflow_records;
    last_profile_.gpu_scene.cpu_submissions = statistics.cpu_submissions;
}

void vulkan_render_backend::collect_gpu_visibility_feedback(std::uint32_t frame_index)
{
    if (frame_index >= gpu_visibility_feedback_frames_.size()) return;
    auto& frame = gpu_visibility_feedback_frames_[frame_index];
    if (frame.submitted_frame == 0u || frame.counters.allocation == VK_NULL_HANDLE) return;
    void* mapped{};
    if (vmaMapMemory(allocator_, frame.counters.allocation, &mapped) != VK_SUCCESS) return;
    vmaInvalidateAllocation(allocator_, frame.counters.allocation, 0, sizeof(gpu_visibility_counter_data));
    gpu_visibility_counter_data counters{};
    std::memcpy(&counters, mapped, sizeof(counters));
    vmaUnmapMemory(allocator_, frame.counters.allocation);
    completed_gpu_visibility_statistics_ = {
        .candidates = counters.candidate_count,
        .visible = std::min(counters.visible_count, gpu_visibility_capacity_) +
                   std::min(counters.transparent_count, gpu_visibility_capacity_),
        .frustum_rejected = counters.frustum_rejected,
        .distance_rejected = counters.distance_rejected,
        .occlusion_rejected = counters.occlusion_rejected,
        .active_bins = counters.active_bins,
        .indirect_commands = std::min(counters.visible_count, gpu_visibility_capacity_) +
                             std::min(counters.transparent_count, gpu_visibility_capacity_),
        .transparent_records = std::min(counters.transparent_count, gpu_visibility_capacity_),
        .overflow_records = counters.overflow_count,
        .cpu_submissions = counters.overflow_count,
    };
    apply_gpu_visibility_statistics(completed_gpu_visibility_statistics_);
    frame.submitted_frame = 0u;
}

void vulkan_render_backend::dispatch_gpu_visibility(VkCommandBuffer command_buffer)
{
    gpu_visibility_active_ = false;
    if (!ensure_gpu_visibility_resources())
    {
        if (resolved_config_.features.gpu_driven_rendering)
            last_profile_.gpu_scene.fallback_reason =
                "GPU visibility resources are unavailable; using direct candidate submission";
        return;
    }

    vkCmdFillBuffer(command_buffer, gpu_visibility_counters_.buffer, 0, VK_WHOLE_SIZE, 0u);
    VkBufferMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    input_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    input_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    input_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    input_barrier.buffer = gpu_scene_visibility_buffer_.buffer;
    input_barrier.size = VK_WHOLE_SIZE;
    VkBufferMemoryBarrier counter_barrier = input_barrier;
    counter_barrier.buffer = gpu_visibility_counters_.buffer;
    std::array<VkBufferMemoryBarrier, 2> input_barriers{input_barrier, counter_barrier};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         static_cast<std::uint32_t>(input_barriers.size()), input_barriers.data(), 0, nullptr);

    gpu_visibility_push_constants constants{};
    std::copy(frame_camera_.view_projection.data(), frame_camera_.view_projection.data() + 16,
              constants.view_projection);
    constants.camera_position_and_error[0] = frame_camera_.position[0];
    constants.camera_position_and_error[1] = frame_camera_.position[1];
    constants.camera_position_and_error[2] = frame_camera_.position[2];
    constants.camera_position_and_error[3] = resolved_config_.geometry_error_threshold;
    constants.instance_capacity = gpu_scene_capacity_;
    constants.camera_cut = frame_camera_.camera_cut ? 1u : 0u;
    const bool hzb_available = resolved_config_.features.hzb_occlusion &&
                               ensure_hzb_resources(viewport_width_, viewport_height_) && hzb_history_valid_ &&
                               !frame_camera_.camera_cut;
    if (hzb_available)
    {
        const auto previous_generation =
            static_cast<std::uint32_t>((last_profile_.frame_index + hzb_history_.size() - 1u) % hzb_history_.size());
        auto& previous_hzb = hzb_history_[previous_generation];
        transition_graph_image(command_buffer, previous_hzb, VK_IMAGE_LAYOUT_GENERAL);
        constants.reserved = previous_generation;
    }
    constants.hzb_parameters[0] = static_cast<float>(viewport_width_);
    constants.hzb_parameters[1] = static_cast<float>(viewport_height_);
    constants.hzb_parameters[2] = static_cast<float>(hzb_mip_count_);
    constants.hzb_parameters[3] = hzb_available ? 1.0f : 0.0f;
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_visibility_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_visibility_pipeline_layout_, 0, 1,
                            &gpu_visibility_descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer, gpu_visibility_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(constants), &constants);
    vkCmdDispatch(command_buffer, (gpu_scene_capacity_ + 63u) / 64u, 1u, 1u);

    if (resolved_config_.features.gpu_transparent_sorting && gpu_transparent_sort_pipeline_ != VK_NULL_HANDLE)
    {
        VkBufferMemoryBarrier sort_input{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        sort_input.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        sort_input.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        sort_input.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sort_input.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sort_input.buffer = gpu_visibility_commands_.buffer;
        sort_input.size = VK_WHOLE_SIZE;
        VkBufferMemoryBarrier sort_counter = sort_input;
        sort_counter.buffer = gpu_visibility_counters_.buffer;
        const std::array sort_inputs{sort_input, sort_counter};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0u, 0u, nullptr, static_cast<std::uint32_t>(sort_inputs.size()), sort_inputs.data(), 0u,
                             nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_transparent_sort_pipeline_);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu_visibility_pipeline_layout_, 0u, 1u,
                                &gpu_visibility_descriptor_set_, 0u, nullptr);
        vkCmdPushConstants(command_buffer, gpu_visibility_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, 1u, 1u, 1u);
    }

    VkBufferMemoryBarrier command_barrier{};
    command_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    command_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    command_barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    command_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    command_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    command_barrier.buffer = gpu_visibility_commands_.buffer;
    command_barrier.size = VK_WHOLE_SIZE;
    VkBufferMemoryBarrier counter_output_barrier = command_barrier;
    counter_output_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    counter_output_barrier.buffer = gpu_visibility_counters_.buffer;
    const std::array output_barriers{command_barrier, counter_output_barrier};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                         static_cast<std::uint32_t>(output_barriers.size()), output_barriers.data(), 0, nullptr);

    if (gpu_visibility_feedback_frames_.size() < frame_resource_count())
        gpu_visibility_feedback_frames_.resize(frame_resource_count());
    auto& feedback = gpu_visibility_feedback_frames_[current_frame_slot()];
    if (ensure_gpu_visibility_feedback_frame(feedback))
    {
        VkBufferCopy copy{.size = sizeof(gpu_visibility_counter_data)};
        vkCmdCopyBuffer(command_buffer, gpu_visibility_counters_.buffer, feedback.counters.buffer, 1u, &copy);
        VkBufferMemoryBarrier host_barrier{};
        host_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host_barrier.buffer = feedback.counters.buffer;
        host_barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                             1u, &host_barrier, 0, nullptr);
        feedback.submitted_frame = last_profile_.frame_index;
    }
    gpu_visibility_active_ = true;
    last_profile_.gpu_scene.enabled = true;
    last_profile_.gpu_scene.submission = resolved_config_.features.gpu_visibility_compaction
                                             ? gpu_submission_path::indirect_count
                                             : gpu_submission_path::indirect;
}

bool vulkan_render_backend::draw_gpu_visibility_command(VkCommandBuffer command_buffer,
                                                        gpu_scene_instance_handle handle) const
{
    if (!gpu_visibility_active_ || !handle.valid() || handle.index >= gpu_visibility_capacity_) return false;
    vkCmdDrawIndexedIndirect(command_buffer, gpu_visibility_commands_.buffer,
                             static_cast<VkDeviceSize>(handle.index) * indexed_indirect_command_stride, 1u,
                             static_cast<std::uint32_t>(indexed_indirect_command_stride));
    return true;
}

bool vulkan_render_backend::gpu_bindless_draw_compatible(const draw_mesh_event& draw, bool transparent) const
{
    if (resolved_config_.features.gpu_binding_model != gpu_resource_binding_model::bindless ||
        draw.mode == render_mode::wireframe || !draw.mesh.valid() || !draw.material.valid())
        return false;
    const auto material = materials_.find(resource_key(draw.material));
    if (material == materials_.end() || material->second.data.domain != material_domain::surface ||
        material->second.data.runtime_program)
        return false;
    return (material->second.data.alpha_mode == material_alpha_mode::blend) == transparent;
}

bool vulkan_render_backend::draw_gpu_bindless_batch(VkCommandBuffer command_buffer, bool transparent)
{
    if (!gpu_visibility_active_ || !ensure_gpu_bindless_pipelines()) return false;
    const auto compatible_count = std::ranges::count_if(frame_draws_, [this, transparent](const draw_mesh_event& draw)
                                                        { return gpu_bindless_draw_compatible(draw, transparent); });
    const auto path_capacity =
        transparent ? std::min<std::uint32_t>(1024u, max_indirect_draw_count_) : max_indirect_draw_count_;
    if (compatible_count > path_capacity)
    {
        last_profile_.gpu_scene.fallback_reason =
            transparent ? "transparent GPU sort capacity exceeded; using stable CPU submission"
                        : "indirect-count capacity exceeded; using CPU submission";
        return false;
    }

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      transparent ? gpu_bindless_transparent_pipeline_ : gpu_bindless_gbuffer_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu_bindless_pipeline_layout_, 0u, 1u,
                            &gpu_bindless_descriptor_set_, 0u, nullptr);
    std::array<float, 32> constants{};
    std::copy_n(frame_camera_.view_projection.data(), 16u, constants.data());
    std::copy_n(frame_camera_.previous_view_projection.data(), 16u, constants.data() + 16u);
    vkCmdPushConstants(command_buffer, gpu_bindless_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0u, sizeof(constants),
                       constants.data());
    const VkDeviceSize vertex_offset{};
    vkCmdBindVertexBuffers(command_buffer, 0u, 1u, &shared_geometry_buffers_.vertices.buffer, &vertex_offset);
    vkCmdBindIndexBuffer(command_buffer, shared_geometry_buffers_.indices.buffer, 0u, VK_INDEX_TYPE_UINT32);
    const auto command_offset =
        static_cast<VkDeviceSize>(gpu_visibility_capacity_) * indexed_indirect_command_stride * (transparent ? 2u : 1u);
    const auto count_offset =
        static_cast<VkDeviceSize>(transparent ? offsetof(gpu_visibility_counter_data, transparent_count)
                                              : offsetof(gpu_visibility_counter_data, visible_count));
    vkCmdDrawIndexedIndirectCount(command_buffer, gpu_visibility_commands_.buffer, command_offset,
                                  gpu_visibility_counters_.buffer, count_offset,
                                  std::min(gpu_visibility_capacity_, max_indirect_draw_count_),
                                  static_cast<std::uint32_t>(indexed_indirect_command_stride));
    return true;
}

} // namespace arc::render::vulkan::backend_detail
