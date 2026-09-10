#include <arc/render/renderer.h>
#include <arc/render/virtual_geometry.h>
#include <arc/scene/terrain.h>
#include <arc/scene/terrain_streaming_prediction.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{

constexpr float stress_world_size = 16'384.0f;
constexpr float stress_region_size = 256.0f;
constexpr std::uint32_t stress_regions_per_axis = 64u;
constexpr std::uint32_t stress_region_count = stress_regions_per_axis * stress_regions_per_axis;
constexpr std::uint32_t root_gpu_bytes = 64u;
constexpr std::uint32_t root_cpu_bytes = 16u;
constexpr std::uint32_t detail_gpu_bytes = 4u * 1024u;
constexpr std::uint32_t detail_cpu_bytes = 1024u;

arc::render::virtual_mesh_data make_stress_region_geometry(float minimum_x, float minimum_z)
{
    using namespace arc::render;
    virtual_mesh_data geometry;

    const auto maximum_x = minimum_x + stress_region_size;
    const auto maximum_z = minimum_z + stress_region_size;
    const arc::math::vector3f center{(minimum_x + maximum_x) * 0.5f, 8.0f, (minimum_z + maximum_z) * 0.5f};

    virtual_mesh_cluster root;
    root.bounds_min = {minimum_x, 0.0f, minimum_z};
    root.bounds_max = {maximum_x, 16.0f, maximum_z};
    root.sphere_center = center;
    root.sphere_radius = stress_region_size;
    root.geometric_error = 64.0f;
    root.page_index = 0u;
    root.hierarchy_node = 0u;

    auto detail = root;
    detail.geometric_error = 8.0f;
    detail.page_index = 1u;
    detail.hierarchy_node = 1u;
    geometry.clusters = {root, detail};

    geometry.pages = {{.first_cluster = 0u,
                       .cluster_count = 1u,
                       .uncompressed_size = root_gpu_bytes,
                       .compressed_size = root_cpu_bytes,
                       .root = true},
                      {.first_cluster = 1u,
                       .cluster_count = 1u,
                       .uncompressed_size = detail_gpu_bytes,
                       .compressed_size = detail_cpu_bytes}};
    return geometry;
}

arc::render::virtual_mesh_data make_fallback_geometry()
{
    using namespace arc::render;
    virtual_mesh_data geometry;

    virtual_mesh_cluster root_cluster;
    root_cluster.bounds_min = {-128.0f, 0.0f, -128.0f};
    root_cluster.bounds_max = {128.0f, 16.0f, 128.0f};
    root_cluster.sphere_center = {0.0f, 8.0f, 0.0f};
    root_cluster.sphere_radius = 192.0f;
    root_cluster.geometric_error = 64.0f;
    root_cluster.page_index = 0u;
    root_cluster.hierarchy_node = 0u;

    auto detail_cluster = root_cluster;
    detail_cluster.geometric_error = 4.0f;
    detail_cluster.page_index = 1u;
    detail_cluster.hierarchy_node = 1u;
    geometry.clusters = {root_cluster, detail_cluster};

    virtual_mesh_lod_node root_node;
    root_node.first_cluster = 0u;
    root_node.cluster_count = 1u;
    root_node.first_child = 0u;
    root_node.child_count = 1u;
    root_node.page_index = 0u;
    root_node.error = 64.0f;
    root_node.sphere_center = root_cluster.sphere_center;
    root_node.sphere_radius = root_cluster.sphere_radius;

    virtual_mesh_lod_node detail_node;
    detail_node.first_cluster = 1u;
    detail_node.cluster_count = 1u;
    detail_node.parent = 0u;
    detail_node.page_index = 1u;
    detail_node.error = 4.0f;
    detail_node.sphere_center = detail_cluster.sphere_center;
    detail_node.sphere_radius = detail_cluster.sphere_radius;

    geometry.lod_nodes = {root_node, detail_node};
    geometry.hierarchy_children = {1u};
    geometry.root_nodes = {0u};
    geometry.pages = {{.first_cluster = 0u,
                       .cluster_count = 1u,
                       .uncompressed_size = root_gpu_bytes,
                       .compressed_size = root_cpu_bytes,
                       .root = true},
                      {.first_cluster = 1u,
                       .cluster_count = 1u,
                       .uncompressed_size = detail_gpu_bytes,
                       .compressed_size = detail_cpu_bytes}};
    return geometry;
}

} // namespace

TEST_CASE("M2.6 16 km terrain streaming remains bounded during rapid traversal")
{
    using namespace arc;
    using namespace arc::render;
    using namespace arc::scene;

    renderer target;
    const auto root_gpu_floor = static_cast<std::uint64_t>(stress_region_count) * root_gpu_bytes;
    const auto root_cpu_floor = static_cast<std::uint64_t>(stress_region_count) * root_cpu_bytes;
    constexpr std::uint32_t detail_budget_pages = 24u;
    const auto gpu_budget = root_gpu_floor + static_cast<std::uint64_t>(detail_budget_pages) * detail_gpu_bytes;
    const auto cpu_budget = root_cpu_floor + static_cast<std::uint64_t>(detail_budget_pages) * detail_cpu_bytes;
    target.virtual_geometry_residency().configure({.gpu_budget_bytes = gpu_budget,
                                                   .compressed_cpu_budget_bytes = cpu_budget,
                                                   .maximum_requests_per_frame = 16u,
                                                   .protected_frame_count = 2u,
                                                   .reload_cooldown_frames = 4u});

    terrain_render_proxy proxy;
    proxy.regions.reserve(stress_region_count);
    std::vector<virtual_mesh_handle> resources;
    resources.reserve(stress_region_count);
    for (std::uint32_t z = 0; z < stress_regions_per_axis; ++z)
        for (std::uint32_t x = 0; x < stress_regions_per_axis; ++x)
        {
            const auto minimum_x = static_cast<float>(x) * stress_region_size;
            const auto minimum_z = static_cast<float>(z) * stress_region_size;
            const auto resource = target.create_virtual_mesh(make_stress_region_geometry(minimum_x, minimum_z));
            REQUIRE(resource.valid());
            resources.push_back(resource);

            terrain_render_region_proxy region;
            region.id = {static_cast<std::int32_t>(x), static_cast<std::int32_t>(z)};
            region.geometry.virtualized = resource;
            proxy.regions.push_back(region);
        }

    REQUIRE(proxy.regions.size() == stress_region_count);
    REQUIRE(stress_world_size == stress_region_size * static_cast<float>(stress_regions_per_axis));

    terrain_streaming_predictor predictor({.prediction_horizon_seconds = 1.25f,
                                           .prefetch_distance = 1536.0f,
                                           .hysteresis_distance = 256.0f,
                                           .forward_bias = 1.0f,
                                           .streaming_importance = 1.0f,
                                           .maximum_prefetch_pages = 64u});

    bool saw_request_pressure{};
    bool saw_eviction{};
    std::uint32_t maximum_loads_per_frame{};
    constexpr std::uint64_t frame_count = 128u;
    for (std::uint64_t frame = 1u; frame <= frame_count; ++frame)
    {
        const auto travel = (static_cast<float>(frame - 1u) / static_cast<float>(frame_count - 1u)) * stress_world_size;
        terrain_streaming_prediction_view view;
        view.camera_position = {std::min(travel, stress_world_size - 1.0f), 64.0f, stress_world_size * 0.5f};
        view.camera_velocity = {1024.0f, 0.0f, 0.0f};
        view.camera_forward = {1.0f, 0.0f, 0.0f};

        auto& residency = target.virtual_geometry_residency();
        residency.begin_frame(frame);
        const auto prediction = predictor.update(proxy, target, view);
        CHECK(prediction.requested_pages <= 64u);

        const auto loads = residency.take_load_requests();
        maximum_loads_per_frame = std::max(maximum_loads_per_frame, static_cast<std::uint32_t>(loads.size()));
        const auto before_publish = residency.snapshot();
        saw_request_pressure = saw_request_pressure || before_publish.request_budget_overflow != 0u;

        for (const auto& load : loads)
        {
            const auto* geometry = target.virtual_mesh_data_for(load.resource);
            REQUIRE(geometry != nullptr);
            REQUIRE(load.page_index < geometry->pages.size());
            const auto& page = geometry->pages[load.page_index];
            residency.mark_loading(load.resource, load.resource_generation, load.page_index);
            residency.publish(load.resource, load.resource_generation, load.page_index, page.uncompressed_size,
                              page.compressed_size);
        }
        saw_eviction = saw_eviction || !residency.take_evictions().empty();

        const auto snapshot = residency.snapshot();
        CHECK(snapshot.resource_count == stress_region_count);
        CHECK(snapshot.root_gpu_resident_bytes == root_gpu_floor);
        CHECK(snapshot.root_compressed_cpu_resident_bytes == root_cpu_floor);
        CHECK(snapshot.gpu_resident_bytes <= snapshot.gpu_budget_bytes);
        CHECK(snapshot.compressed_cpu_resident_bytes <= snapshot.compressed_cpu_budget_bytes);
        CHECK(snapshot.gpu_budget_overflow_bytes == 0u);
        CHECK(snapshot.compressed_cpu_budget_overflow_bytes == 0u);
    }

    CHECK(maximum_loads_per_frame <= 16u);
    CHECK(saw_request_pressure);
    CHECK(saw_eviction);
    for (const auto resource : resources)
    {
        const auto generation = target.virtual_mesh_content_generation(resource);
        REQUIRE(generation != 0u);
        CHECK(target.virtual_geometry_residency().resident(resource, generation, 0u));
    }
}

TEST_CASE("M2.6 missing terrain detail keeps a resident parent fallback")
{
    using namespace arc::render;
    const auto geometry = make_fallback_geometry();
    const std::array<std::uint8_t, 2> root_only{1u, 0u};

    virtual_geometry_reference_view view;
    view.camera_position = {0.0f, 128.0f, 0.0f};
    view.projection_scale = 1000.0f;
    view.geometric_error_threshold = 1.0f;
    view.minimum_projected_radius = 0.0f;
    view.double_sided = true;

    const auto fallback = traverse_virtual_geometry_reference(geometry, root_only, view);
    REQUIRE(fallback.parent_fallbacks == 1u);
    REQUIRE(fallback.requested_pages == std::vector<std::uint32_t>{1u});
    REQUIRE(fallback.visible_clusters == std::vector<std::uint32_t>{0u});

    const std::array<std::uint8_t, 2> full_detail{1u, 1u};
    const auto refined = traverse_virtual_geometry_reference(geometry, full_detail, view);
    CHECK(refined.parent_fallbacks == 0u);
    CHECK(refined.visible_clusters == std::vector<std::uint32_t>{1u});
}
