#include <arc/render/events.h>
#include <arc/render/mesh.h>
#include <arc/render/virtual_geometry.h>
#include <arc/render/virtual_mesh.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{
arc::render::virtual_mesh_data make_two_level_geometry()
{
    using namespace arc::render;
    virtual_mesh_data geometry;
    geometry.pages = {{.uncompressed_size = 16u, .compressed_size = 8u, .root = true},
                      {.uncompressed_size = 16u, .compressed_size = 8u}};
    geometry.clusters = {{.index_count = 3u,
                          .triangle_count = 1u,
                          .vertex_count = 3u,
                          .sphere_center = {0.0f, 0.0f, 0.0f},
                          .sphere_radius = 1.0f,
                          .page_index = 0u},
                         {.index_count = 3u,
                          .triangle_count = 1u,
                          .vertex_count = 3u,
                          .sphere_center = {0.0f, 0.0f, 0.0f},
                          .sphere_radius = 1.0f,
                          .page_index = 1u}};
    geometry.lod_nodes = {{.first_cluster = 0u,
                           .cluster_count = 1u,
                           .first_child = 0u,
                           .child_count = 1u,
                           .page_index = 0u,
                           .error = 1.0f,
                           .sphere_center = {0.0f, 0.0f, 0.0f},
                           .sphere_radius = 1.0f},
                          {.first_cluster = 1u,
                           .cluster_count = 1u,
                           .parent = 0u,
                           .page_index = 1u,
                           .error = 0.0f,
                           .sphere_center = {0.0f, 0.0f, 0.0f},
                           .sphere_radius = 1.0f,
                           .level = 1u}};
    geometry.hierarchy_children = {1u};
    geometry.root_nodes = {0u};
    return geometry;
}

arc::render::virtual_geometry_reference_view open_view()
{
    arc::render::virtual_geometry_reference_view view;
    view.camera_position = {0.0f, 0.0f, 10.0f};
    view.projection_scale = 1.0f;
    view.geometric_error_threshold = 1.0f;
    view.minimum_projected_radius = 0.0f;
    view.double_sided = true;
    return view;
}

bool always_occluded(const arc::math::vector3f&, float, void*)
{
    return true;
}

arc::render::mesh_data make_terrain_grid(std::uint32_t side)
{
    arc::render::mesh_data mesh;
    mesh.name = "M1.6 terrain stress grid";
    mesh.vertices.resize(static_cast<std::size_t>(side) * side);
    for (std::uint32_t z = 0; z < side; ++z)
        for (std::uint32_t x = 0; x < side; ++x)
        {
            auto& vertex = mesh.vertices[static_cast<std::size_t>(z) * side + x];
            vertex.position[0] = static_cast<float>(x);
            vertex.position[1] =
                std::sin(static_cast<float>(x) * 0.11f) * std::cos(static_cast<float>(z) * 0.09f) * 3.0f;
            vertex.position[2] = static_cast<float>(z);
            vertex.normal[1] = 1.0f;
            vertex.tangent[0] = 1.0f;
            vertex.tangent[3] = 1.0f;
            vertex.texcoord[0] = static_cast<float>(x) / static_cast<float>(side - 1u);
            vertex.texcoord[1] = static_cast<float>(z) / static_cast<float>(side - 1u);
        }
    for (std::uint32_t z = 0; z + 1u < side; ++z)
        for (std::uint32_t x = 0; x + 1u < side; ++x)
        {
            const auto i0 = z * side + x;
            const auto i1 = i0 + 1u;
            const auto i2 = i0 + side;
            const auto i3 = i2 + 1u;
            mesh.indices.insert(mesh.indices.end(), {i0, i2, i1, i1, i2, i3});
        }
    return mesh;
}
} // namespace

TEST_CASE("virtual geometry does not stream detail below the projected-error threshold")
{
    const auto geometry = make_two_level_geometry();
    const std::vector<std::uint8_t> resident{1u, 0u};
    auto view = open_view();
    view.geometric_error_threshold = 1.0f;

    const auto coarse = arc::render::traverse_virtual_geometry_reference(geometry, resident, view);
    REQUIRE(coarse.requested_pages.empty());
    REQUIRE(coarse.parent_fallbacks == 0u);
    REQUIRE(coarse.visible_clusters == std::vector<std::uint32_t>{0u});

    view.geometric_error_threshold = 0.01f;
    const auto detailed = arc::render::traverse_virtual_geometry_reference(geometry, resident, view);
    REQUIRE(detailed.requested_pages == std::vector<std::uint32_t>{1u});
    REQUIRE(detailed.parent_fallbacks == 1u);
    REQUIRE(detailed.visible_clusters == std::vector<std::uint32_t>{0u});
}

TEST_CASE("virtual geometry skips HZB rejection on camera cuts")
{
    const auto geometry = make_two_level_geometry();
    const std::vector<std::uint8_t> resident{1u, 1u};
    auto view = open_view();
    view.occluded = &always_occluded;

    const auto stable = arc::render::traverse_virtual_geometry_reference(geometry, resident, view);
    REQUIRE(stable.hzb_rejected == 1u);
    REQUIRE(stable.visible_clusters.empty());

    view.camera_cut = true;
    const auto cut = arc::render::traverse_virtual_geometry_reference(geometry, resident, view);
    REQUIRE(cut.hzb_rejected == 0u);
    REQUIRE_FALSE(cut.visible_clusters.empty());
}

TEST_CASE("virtual geometry residency emits backend evictions and protects roots")
{
    using namespace arc::render;
    virtual_mesh_data geometry;
    geometry.pages = {{.uncompressed_size = 8u, .compressed_size = 4u, .root = true},
                      {.uncompressed_size = 8u, .compressed_size = 4u},
                      {.uncompressed_size = 8u, .compressed_size = 4u}};
    const virtual_mesh_handle handle{7u, 3u};
    virtual_geometry_residency_manager residency({.gpu_budget_bytes = 16u,
                                                  .compressed_cpu_budget_bytes = 8u,
                                                  .maximum_requests_per_frame = 16u,
                                                  .protected_frame_count = 0u});
    residency.register_resource(handle, geometry, 11u);
    residency.begin_frame(1u);
    residency.publish(handle, 11u, 1u, 8u, 4u);
    REQUIRE(residency.take_evictions().empty());

    residency.begin_frame(2u);
    residency.publish(handle, 11u, 2u, 8u, 4u);
    const auto evictions = residency.take_evictions();
    REQUIRE(evictions.size() == 1u);
    REQUIRE(evictions.front().resource == handle);
    REQUIRE(evictions.front().resource_generation == 11u);
    REQUIRE(evictions.front().page_index == 1u);
    REQUIRE(residency.resident(handle, 11u, 0u));
    REQUIRE_FALSE(residency.resident(handle, 11u, 1u));
    REQUIRE(residency.resident(handle, 11u, 2u));
    REQUIRE(residency.snapshot().evictions == 1u);
    REQUIRE(residency.take_evictions().empty());
}

TEST_CASE("virtual geometry eviction has a typed render event")
{
    using namespace arc::render;
    render_event_buffer buffer;
    render_event_writer writer(buffer);
    const virtual_geometry_page_eviction eviction{.resource = {5u, 2u}, .resource_generation = 9u, .page_index = 4u};
    writer.virtual_geometry_page_evict(eviction);
    REQUIRE(buffer.events().size() == 1u);
    REQUIRE(buffer.events().front().type() == render_event_type::virtual_geometry_page_evict);
    REQUIRE(std::get<virtual_geometry_page_evict_event>(buffer.events().front().payload).eviction.page_index == 4u);
}

TEST_CASE("large terrain-like virtual geometry changes detail with projected error")
{
    using namespace arc::render;
    const auto geometry = build_virtual_mesh(make_terrain_grid(65u), {.build_conventional_lods = false});
    REQUIRE(geometry.pages.size() > 1u);
    REQUIRE(geometry.lod_nodes.size() > geometry.root_nodes.size());
    REQUIRE_FALSE(geometry.root_nodes.empty());

    std::vector<std::uint8_t> resident(geometry.pages.size(), 1u);
    const auto& root = geometry.lod_nodes[geometry.root_nodes.front()];
    REQUIRE(root.child_count > 0u);
    REQUIRE(root.error > 0.0f);

    virtual_geometry_reference_view near_view;
    near_view.projection_scale = 1080.0f;
    near_view.minimum_projected_radius = 0.0f;
    near_view.double_sided = true;
    near_view.camera_position = root.sphere_center;
    near_view.camera_position[1] += std::max(root.sphere_radius * 2.0f, 10.0f);
    auto far_view = near_view;
    far_view.camera_position[1] += std::max(root.sphere_radius * 100.0f, 10000.0f);

    const auto projected_error = [&](const virtual_geometry_reference_view& view)
    {
        const auto delta = arc::math::sub(view.camera_position, root.sphere_center);
        const auto distance = std::sqrt(std::max(arc::math::length_squared(delta), 1.0e-12f));
        const auto nearest = std::max(distance - root.sphere_radius, 1.0e-4f);
        return root.error * view.projection_scale / nearest;
    };
    const auto near_error = projected_error(near_view);
    const auto far_error = projected_error(far_view);
    REQUIRE(near_error > far_error);
    near_view.geometric_error_threshold = (near_error + far_error) * 0.5f;
    far_view.geometric_error_threshold = near_view.geometric_error_threshold;

    const auto near_result = traverse_virtual_geometry_reference(geometry, resident, near_view);
    const auto far_result = traverse_virtual_geometry_reference(geometry, resident, far_view);
    REQUIRE(near_result.requested_pages.empty());
    REQUIRE(far_result.requested_pages.empty());
    REQUIRE_FALSE(near_result.visible_clusters.empty());
    REQUIRE_FALSE(far_result.visible_clusters.empty());
    REQUIRE(near_result.visible_clusters != far_result.visible_clusters);
}
