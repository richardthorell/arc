#include <arc/render/virtual_geometry.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace arc::render
{
namespace
{

std::uint64_t resource_key(virtual_mesh_handle handle) noexcept
{
    return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
}

float request_priority(const virtual_geometry_page_request& request) noexcept
{
    float result = request.projected_error * 8.0f + request.screen_coverage * 4.0f;
    if (request.visible_child) result += 1000.0f;
    if (request.shadow_view) result += 100.0f;
    return result - request.distance * 0.001f;
}

bool sphere_inside_frustum(const math::vector3f& center, float radius,
                           const std::array<math::vector4f, 6>& planes) noexcept
{
    for (const auto& plane : planes)
        if (center[0] * plane[0] + center[1] * plane[1] + center[2] * plane[2] + plane[3] < -radius) return false;
    return true;
}

bool page_resident(std::span<const std::uint8_t> pages, std::uint32_t page) noexcept
{
    return page != invalid_virtual_geometry_index && page < pages.size() && pages[page] != 0;
}

} // namespace

virtual_geometry_reference_result traverse_virtual_geometry_reference(const virtual_mesh_data& geometry,
                                                                      std::span<const std::uint8_t> resident_pages,
                                                                      const virtual_geometry_reference_view& view)
{
    virtual_geometry_reference_result result;
    std::vector<std::uint32_t> stack(geometry.root_nodes.rbegin(), geometry.root_nodes.rend());
    std::vector<std::uint8_t> requested(geometry.pages.size());
    while (!stack.empty())
    {
        const auto node_index = stack.back();
        stack.pop_back();
        if (node_index >= geometry.lod_nodes.size()) continue;
        const auto& node = geometry.lod_nodes[node_index];
        if (!sphere_inside_frustum(node.sphere_center, node.sphere_radius, view.frustum_planes))
        {
            ++result.frustum_rejected;
            continue;
        }

        const auto to_camera = math::sub(view.camera_position, node.sphere_center);
        const auto distance = std::sqrt(std::max(math::length_squared(to_camera), 1.0e-12f));
        if (!view.double_sided && node.cone_cutoff >= 0.0f)
        {
            const auto direction = math::mul(to_camera, 1.0f / distance);
            const auto cone_sine = std::sqrt(std::max(0.0f, 1.0f - node.cone_cutoff * node.cone_cutoff));
            if (math::dot(node.cone_axis, direction) <= -cone_sine)
            {
                ++result.cone_rejected;
                continue;
            }
        }
        if (!view.camera_cut && view.occluded &&
            view.occluded(node.sphere_center, node.sphere_radius, view.occlusion_user_data))
        {
            ++result.hzb_rejected;
            continue;
        }

        const auto nearest_distance = std::max(distance - node.sphere_radius, 1.0e-4f);
        const auto projected_radius = node.sphere_radius * view.projection_scale / nearest_distance;
        if (projected_radius < view.minimum_projected_radius)
        {
            ++result.projected_size_rejected;
            continue;
        }
        const auto projected_error = node.error * view.projection_scale / nearest_distance;

        bool children_resident = node.child_count > 0;
        for (std::uint32_t child_offset = 0; child_offset < node.child_count; ++child_offset)
        {
            const auto hierarchy_offset = node.first_child + child_offset;
            if (hierarchy_offset >= geometry.hierarchy_children.size())
            {
                children_resident = false;
                break;
            }
            const auto child_index = geometry.hierarchy_children[hierarchy_offset];
            if (child_index >= geometry.lod_nodes.size() ||
                !page_resident(resident_pages, geometry.lod_nodes[child_index].page_index))
            {
                children_resident = false;
                if (child_index < geometry.lod_nodes.size())
                {
                    const auto page = geometry.lod_nodes[child_index].page_index;
                    if (page < requested.size() && requested[page] == 0)
                    {
                        requested[page] = 1;
                        result.requested_pages.push_back(page);
                    }
                }
            }
        }

        if (node.child_count > 0 && projected_error > view.geometric_error_threshold && children_resident)
        {
            for (std::uint32_t child_offset = node.child_count; child_offset > 0; --child_offset)
                stack.push_back(geometry.hierarchy_children[node.first_child + child_offset - 1]);
            continue;
        }
        if (node.child_count > 0 && projected_error > view.geometric_error_threshold && !children_resident)
            ++result.parent_fallbacks;

        for (std::uint32_t cluster_offset = 0; cluster_offset < node.cluster_count; ++cluster_offset)
        {
            const auto cluster = node.first_cluster + cluster_offset;
            if (cluster < geometry.clusters.size() &&
                page_resident(resident_pages, geometry.clusters[cluster].page_index))
                result.visible_clusters.push_back(cluster);
        }
    }
    return result;
}

virtual_geometry_gpu_table_update make_virtual_geometry_gpu_table_update(virtual_mesh_handle resource,
                                                                         const virtual_mesh_data& geometry,
                                                                         std::uint32_t resource_generation)
{
    virtual_geometry_gpu_table_update result{.resource = resource, .resource_generation = resource_generation};
    result.resources.push_back({.first_node = 0,
                                .node_count = static_cast<std::uint32_t>(geometry.lod_nodes.size()),
                                .first_cluster = 0,
                                .cluster_count = static_cast<std::uint32_t>(geometry.clusters.size()),
                                .first_child = 0,
                                .child_count = static_cast<std::uint32_t>(geometry.hierarchy_children.size()),
                                .first_page = 0,
                                .page_count = static_cast<std::uint32_t>(geometry.pages.size()),
                                .first_root = 0,
                                .root_count = static_cast<std::uint32_t>(geometry.root_nodes.size()),
                                .generation = resource_generation});
    result.nodes.reserve(geometry.lod_nodes.size());
    for (const auto& node : geometry.lod_nodes)
        result.nodes.push_back(
            {.sphere = {node.sphere_center[0], node.sphere_center[1], node.sphere_center[2], node.sphere_radius},
             .normal_cone = {node.cone_axis[0], node.cone_axis[1], node.cone_axis[2], node.cone_cutoff},
             .geometric_error = node.error,
             .first_cluster = node.first_cluster,
             .cluster_count = node.cluster_count,
             .first_child = node.first_child,
             .child_count = node.child_count,
             .page_index = node.page_index,
             .level = node.level,
             .flags = node.flags});
    result.clusters.reserve(geometry.clusters.size());
    for (const auto& cluster : geometry.clusters)
        result.clusters.push_back(
            {.sphere = {cluster.sphere_center[0], cluster.sphere_center[1], cluster.sphere_center[2],
                        cluster.sphere_radius},
             .normal_cone = {cluster.cone_axis[0], cluster.cone_axis[1], cluster.cone_axis[2], cluster.cone_cutoff},
             .bounds_min_error = {cluster.bounds_min[0], cluster.bounds_min[1], cluster.bounds_min[2],
                                  cluster.geometric_error},
             .bounds_max = {cluster.bounds_max[0], cluster.bounds_max[1], cluster.bounds_max[2], 0.0f},
             .page_index = cluster.page_index,
             .page_byte_offset = cluster.page_byte_offset,
             .vertex_count = cluster.vertex_count,
             .triangle_count = cluster.triangle_count,
             .material_section = static_cast<std::uint32_t>(
                 std::min<std::size_t>(cluster.material_index, std::numeric_limits<std::uint32_t>::max())),
             .hierarchy_node = cluster.hierarchy_node,
             .flags = cluster.flags});
    result.children = geometry.hierarchy_children;
    result.roots = geometry.root_nodes;
    result.pages.reserve(geometry.pages.size());
    for (const auto& page : geometry.pages)
        result.pages.push_back({.stored_size = page.compressed_size,
                                .decoded_size = page.uncompressed_size,
                                .resource_generation = resource_generation,
                                .flags = page.root
                                             ? static_cast<virtual_geometry_gpu_page_flag>(
                                                   static_cast<std::uint32_t>(virtual_geometry_gpu_page_flag::root) |
                                                   static_cast<std::uint32_t>(virtual_geometry_gpu_page_flag::resident))
                                             : virtual_geometry_gpu_page_flag::none});
    return result;
}

virtual_geometry_gpu_reference_result traverse_virtual_geometry_gpu_reference(
    virtual_mesh_handle resource, std::uint32_t resource_generation, std::uint32_t instance_index,
    std::uint32_t material_index, const virtual_mesh_data& geometry, std::span<const std::uint8_t> resident_pages,
    const virtual_geometry_reference_view& view, virtual_geometry_traversal_limits limits)
{
    virtual_geometry_gpu_reference_result result;
    const auto reference = traverse_virtual_geometry_reference(geometry, resident_pages, view);
    result.frustum_rejected = reference.frustum_rejected;
    result.cone_rejected = reference.cone_rejected;
    result.hzb_rejected = reference.hzb_rejected;
    result.projected_size_rejected = reference.projected_size_rejected;
    result.feedback.overflow.fallback_instance_count = 0;
    const auto visible_count =
        std::min<std::size_t>(reference.visible_clusters.size(), limits.maximum_visible_clusters);
    result.visible_clusters.reserve(visible_count);
    for (std::size_t index = 0; index < visible_count; ++index)
    {
        const auto cluster_index = reference.visible_clusters[index];
        const auto& cluster = geometry.clusters[cluster_index];
        result.visible_clusters.push_back({.instance_index = instance_index,
                                           .resource_index = resource.index,
                                           .cluster_index = cluster_index,
                                           .page_index = cluster.page_index,
                                           .material_index = material_index,
                                           .hierarchy_level = cluster.hierarchy_level});
    }
    if (visible_count != reference.visible_clusters.size())
    {
        result.feedback.overflow.visible_cluster_overflow =
            static_cast<std::uint32_t>(reference.visible_clusters.size() - visible_count);
        result.feedback.overflow.fallback_instance_count = 1;
        result.visible_clusters.clear();
    }

    const auto request_count = std::min<std::size_t>(reference.requested_pages.size(), limits.maximum_page_requests);
    result.feedback.page_requests.reserve(request_count);
    for (std::size_t index = 0; index < request_count; ++index)
        result.feedback.page_requests.push_back({.resource_index = resource.index,
                                                 .handle_generation = resource.generation,
                                                 .resource_generation = resource_generation,
                                                 .page_index = reference.requested_pages[index],
                                                 .flags = 1u});
    if (request_count != reference.requested_pages.size())
        result.feedback.overflow.page_request_overflow =
            static_cast<std::uint32_t>(reference.requested_pages.size() - request_count);
    return result;
}

struct virtual_geometry_residency_manager::implementation
{
    struct page_entry
    {
        virtual_geometry_page descriptor{};
        virtual_geometry_page_state state{virtual_geometry_page_state::nonresident};
        std::uint64_t last_used_frame{};
        std::uint32_t gpu_bytes{};
        std::uint32_t cpu_bytes{};
        float priority{};
    };

    struct resource_entry
    {
        virtual_mesh_handle handle{};
        std::uint32_t generation{};
        std::vector<page_entry> pages;
    };

    virtual_geometry_residency_config config{};
    std::unordered_map<std::uint64_t, resource_entry> resources;
    std::uint64_t frame_index{};
    std::uint64_t gpu_bytes{};
    std::uint64_t cpu_bytes{};
    std::uint32_t evictions{};
    std::uint32_t deduplicated_requests{};
    std::uint32_t parent_fallbacks{};
    std::uint32_t stale_requests{};

    page_entry* find(virtual_mesh_handle handle, std::uint32_t generation, std::uint32_t page_index) noexcept
    {
        const auto found = resources.find(resource_key(handle));
        if (found == resources.end() || found->second.generation != generation ||
            page_index >= found->second.pages.size())
            return nullptr;
        return &found->second.pages[page_index];
    }

    const page_entry* find(virtual_mesh_handle handle, std::uint32_t generation,
                           std::uint32_t page_index) const noexcept
    {
        const auto found = resources.find(resource_key(handle));
        if (found == resources.end() || found->second.generation != generation ||
            page_index >= found->second.pages.size())
            return nullptr;
        return &found->second.pages[page_index];
    }

    void trim()
    {
        while (gpu_bytes > config.gpu_budget_bytes || cpu_bytes > config.compressed_cpu_budget_bytes)
        {
            page_entry* victim{};
            for (auto& [_, resource] : resources)
                for (auto& page : resource.pages)
                {
                    if (page.state != virtual_geometry_page_state::resident || page.descriptor.root ||
                        frame_index - std::min(frame_index, page.last_used_frame) <= config.protected_frame_count)
                        continue;
                    if (!victim || page.last_used_frame < victim->last_used_frame ||
                        (page.last_used_frame == victim->last_used_frame && page.priority < victim->priority))
                        victim = &page;
                }
            if (!victim) break;
            gpu_bytes -= std::min<std::uint64_t>(gpu_bytes, victim->gpu_bytes);
            cpu_bytes -= std::min<std::uint64_t>(cpu_bytes, victim->cpu_bytes);
            victim->gpu_bytes = 0;
            victim->cpu_bytes = 0;
            victim->priority = 0.0f;
            victim->state = virtual_geometry_page_state::nonresident;
            ++evictions;
        }
    }
};

virtual_geometry_residency_manager::virtual_geometry_residency_manager(virtual_geometry_residency_config config)
    : implementation_(std::make_unique<implementation>())
{
    configure(config);
}

virtual_geometry_residency_manager::~virtual_geometry_residency_manager() = default;
virtual_geometry_residency_manager::virtual_geometry_residency_manager(virtual_geometry_residency_manager&&) noexcept =
    default;
virtual_geometry_residency_manager&
virtual_geometry_residency_manager::operator=(virtual_geometry_residency_manager&&) noexcept = default;

void virtual_geometry_residency_manager::configure(virtual_geometry_residency_config config)
{
    config.maximum_requests_per_frame = std::max(1u, config.maximum_requests_per_frame);
    implementation_->config = config;
    implementation_->trim();
}

void virtual_geometry_residency_manager::register_resource(virtual_mesh_handle resource, const virtual_mesh_data& data,
                                                           std::uint32_t generation)
{
    unregister_resource(resource);
    implementation::resource_entry entry{.handle = resource, .generation = generation};
    entry.pages.reserve(data.pages.size());
    for (const auto& descriptor : data.pages)
    {
        implementation::page_entry page{.descriptor = descriptor};
        if (descriptor.root)
        {
            page.state = virtual_geometry_page_state::resident;
            page.gpu_bytes = descriptor.uncompressed_size;
            page.cpu_bytes = descriptor.compressed_size;
            page.last_used_frame = implementation_->frame_index;
            implementation_->gpu_bytes += page.gpu_bytes;
            implementation_->cpu_bytes += page.cpu_bytes;
        }
        entry.pages.push_back(page);
    }
    implementation_->resources.emplace(resource_key(resource), std::move(entry));
    implementation_->trim();
}

void virtual_geometry_residency_manager::unregister_resource(virtual_mesh_handle resource)
{
    const auto found = implementation_->resources.find(resource_key(resource));
    if (found == implementation_->resources.end()) return;
    for (const auto& page : found->second.pages)
    {
        implementation_->gpu_bytes -= std::min<std::uint64_t>(implementation_->gpu_bytes, page.gpu_bytes);
        implementation_->cpu_bytes -= std::min<std::uint64_t>(implementation_->cpu_bytes, page.cpu_bytes);
    }
    implementation_->resources.erase(found);
}

void virtual_geometry_residency_manager::begin_frame(std::uint64_t frame_index)
{
    implementation_->frame_index = frame_index;
    implementation_->deduplicated_requests = 0;
    implementation_->parent_fallbacks = 0;
    implementation_->stale_requests = 0;
}

void virtual_geometry_residency_manager::request(std::span<const virtual_geometry_page_request> requests)
{
    for (const auto& request : requests)
    {
        auto* page = implementation_->find(request.resource, request.resource_generation, request.page_index);
        if (!page) continue;
        const auto priority = request_priority(request);
        if (page->state == virtual_geometry_page_state::resident)
        {
            page->last_used_frame = implementation_->frame_index;
            page->priority = std::max(page->priority, priority);
            continue;
        }
        if (page->state == virtual_geometry_page_state::requested ||
            page->state == virtual_geometry_page_state::loading)
        {
            page->priority = std::max(page->priority, priority);
            ++implementation_->deduplicated_requests;
            continue;
        }
        page->state = virtual_geometry_page_state::requested;
        page->priority = priority;
    }
}

void virtual_geometry_residency_manager::request_gpu(std::span<const virtual_geometry_gpu_page_request> requests)
{
    for (const auto& request : requests)
    {
        const virtual_mesh_handle handle{request.resource_index, request.handle_generation};
        if (!implementation_->find(handle, request.resource_generation, request.page_index))
        {
            ++implementation_->stale_requests;
            continue;
        }
        const bool visible_child = (request.flags & 1u) != 0;
        const bool shadow_view = (request.flags & 2u) != 0;
        const virtual_geometry_page_request translated{.resource = handle,
                                                       .resource_generation = request.resource_generation,
                                                       .page_index = request.page_index,
                                                       .projected_error = request.projected_error,
                                                       .screen_coverage = request.screen_coverage,
                                                       .distance = request.distance,
                                                       .visible_child = visible_child,
                                                       .shadow_view = shadow_view};
        this->request(std::span(&translated, 1));
    }
}

std::vector<virtual_geometry_page_load> virtual_geometry_residency_manager::take_load_requests()
{
    std::vector<virtual_geometry_page_load> result;
    for (auto& [_, resource] : implementation_->resources)
        for (std::uint32_t index = 0; index < resource.pages.size(); ++index)
        {
            auto& page = resource.pages[index];
            if (page.state != virtual_geometry_page_state::requested) continue;
            result.push_back({.resource = resource.handle,
                              .resource_generation = resource.generation,
                              .page_index = index,
                              .byte_offset = page.descriptor.compressed_offset,
                              .byte_size = page.descriptor.compressed_size,
                              .priority = page.priority});
        }
    std::stable_sort(result.begin(), result.end(),
                     [](const auto& lhs, const auto& rhs)
                     {
                         if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
                         if (lhs.resource.index != rhs.resource.index) return lhs.resource.index < rhs.resource.index;
                         return lhs.page_index < rhs.page_index;
                     });
    if (result.size() > implementation_->config.maximum_requests_per_frame)
        result.resize(implementation_->config.maximum_requests_per_frame);
    return result;
}

void virtual_geometry_residency_manager::mark_loading(virtual_mesh_handle resource, std::uint32_t generation,
                                                      std::uint32_t page_index)
{
    if (auto* page = implementation_->find(resource, generation, page_index);
        page && page->state == virtual_geometry_page_state::requested)
        page->state = virtual_geometry_page_state::loading;
}

void virtual_geometry_residency_manager::publish(virtual_mesh_handle resource, std::uint32_t generation,
                                                 std::uint32_t page_index, std::uint32_t gpu_bytes,
                                                 std::uint32_t compressed_cpu_bytes)
{
    auto* page = implementation_->find(resource, generation, page_index);
    if (!page) return;
    implementation_->gpu_bytes -= std::min<std::uint64_t>(implementation_->gpu_bytes, page->gpu_bytes);
    implementation_->cpu_bytes -= std::min<std::uint64_t>(implementation_->cpu_bytes, page->cpu_bytes);
    page->state = virtual_geometry_page_state::resident;
    page->gpu_bytes = gpu_bytes;
    page->cpu_bytes = compressed_cpu_bytes;
    page->last_used_frame = implementation_->frame_index;
    implementation_->gpu_bytes += gpu_bytes;
    implementation_->cpu_bytes += compressed_cpu_bytes;
    implementation_->trim();
}

void virtual_geometry_residency_manager::fail(virtual_mesh_handle resource, std::uint32_t generation,
                                              std::uint32_t page_index)
{
    if (auto* page = implementation_->find(resource, generation, page_index); page && !page->descriptor.root)
        page->state = virtual_geometry_page_state::failed;
}

void virtual_geometry_residency_manager::touch(virtual_mesh_handle resource, std::uint32_t generation,
                                               std::uint32_t page_index)
{
    if (auto* page = implementation_->find(resource, generation, page_index);
        page && page->state == virtual_geometry_page_state::resident)
        page->last_used_frame = implementation_->frame_index;
}

bool virtual_geometry_residency_manager::resident(virtual_mesh_handle resource, std::uint32_t generation,
                                                  std::uint32_t page_index) const noexcept
{
    const auto* page = implementation_->find(resource, generation, page_index);
    return page && page->state == virtual_geometry_page_state::resident;
}

void virtual_geometry_residency_manager::note_parent_fallback() noexcept
{
    ++implementation_->parent_fallbacks;
}

virtual_geometry_residency_snapshot virtual_geometry_residency_manager::snapshot() const noexcept
{
    virtual_geometry_residency_snapshot result{
        .frame_index = implementation_->frame_index,
        .gpu_budget_bytes = implementation_->config.gpu_budget_bytes,
        .gpu_resident_bytes = implementation_->gpu_bytes,
        .compressed_cpu_budget_bytes = implementation_->config.compressed_cpu_budget_bytes,
        .compressed_cpu_resident_bytes = implementation_->cpu_bytes,
        .resource_count = static_cast<std::uint32_t>(implementation_->resources.size()),
        .evictions = implementation_->evictions,
        .deduplicated_requests = implementation_->deduplicated_requests,
        .parent_fallbacks = implementation_->parent_fallbacks,
        .stale_requests = implementation_->stale_requests};
    for (const auto& [_, resource] : implementation_->resources)
        for (const auto& page : resource.pages)
        {
            ++result.page_count;
            if (page.state == virtual_geometry_page_state::resident) ++result.resident_pages;
            if (page.state == virtual_geometry_page_state::resident &&
                (page.descriptor.root ||
                 implementation_->frame_index - std::min(implementation_->frame_index, page.last_used_frame) <=
                     implementation_->config.protected_frame_count))
                ++result.protected_pages;
            if (page.state == virtual_geometry_page_state::requested ||
                page.state == virtual_geometry_page_state::loading)
                ++result.requested_pages;
            if (page.state == virtual_geometry_page_state::failed) ++result.failed_pages;
        }
    return result;
}

} // namespace arc::render
