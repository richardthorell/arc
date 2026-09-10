#include "vulkan_backend_internal.h"

namespace arc::render::vulkan::backend_detail
{
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
