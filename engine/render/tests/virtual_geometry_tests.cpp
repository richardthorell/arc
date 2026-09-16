#include <arc/render/render.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <memory>
#include <vector>

#if !defined(ARC_RENDER_TEST_ASSET_ROOT)
#define ARC_RENDER_TEST_ASSET_ROOT "assets"
#endif

TEST_CASE("virtual mesh builder handles empty input")
{
    const arc::render::mesh_data source;
    const auto virtual_mesh = arc::render::build_virtual_mesh(source);

    REQUIRE(virtual_mesh.vertices.empty());
    REQUIRE(virtual_mesh.indices.empty());
    REQUIRE(virtual_mesh.clusters.empty());
    REQUIRE(virtual_mesh.lod_nodes.empty());
    REQUIRE(virtual_mesh.stats.source_vertex_count == 0);
    REQUIRE(virtual_mesh.stats.source_triangle_count == 0);
    REQUIRE(virtual_mesh.stats.cluster_count == 0);
    REQUIRE(virtual_mesh.stats.average_triangles_per_cluster == Catch::Approx(0.0f));
    REQUIRE(virtual_mesh.stats.material_group_count == 0);
    REQUIRE(virtual_mesh.stats.invalid_triangle_count == 0);
}

TEST_CASE("virtual mesh builder creates one bounded cluster for a triangle")
{
    arc::render::mesh_data source;
    source.material_index = 7;
    source.vertices.resize(3);
    source.vertices[0].position[0] = 0.0f;
    source.vertices[0].position[1] = 0.0f;
    source.vertices[0].position[2] = 0.0f;
    source.vertices[1].position[0] = 2.0f;
    source.vertices[1].position[1] = 0.0f;
    source.vertices[1].position[2] = 0.0f;
    source.vertices[2].position[0] = 0.0f;
    source.vertices[2].position[1] = 2.0f;
    source.vertices[2].position[2] = 0.0f;
    source.indices = {0, 1, 2};

    const auto virtual_mesh = arc::render::build_virtual_mesh(source);

    REQUIRE(virtual_mesh.vertices.size() == 3);
    REQUIRE(virtual_mesh.indices == source.indices);
    REQUIRE(virtual_mesh.clusters.size() == 1);
    const auto& cluster = virtual_mesh.clusters.front();
    REQUIRE(cluster.first_index == 0);
    REQUIRE(cluster.index_count == 3);
    REQUIRE(cluster.first_triangle == 0);
    REQUIRE(cluster.triangle_count == 1);
    REQUIRE(cluster.first_vertex == 0);
    REQUIRE(cluster.vertex_count == 3);
    REQUIRE(cluster.material_index == 7);
    REQUIRE(cluster.bounds_min[0] == Catch::Approx(0.0f));
    REQUIRE(cluster.bounds_min[1] == Catch::Approx(0.0f));
    REQUIRE(cluster.bounds_max[0] == Catch::Approx(2.0f));
    REQUIRE(cluster.bounds_max[1] == Catch::Approx(2.0f));
    REQUIRE(cluster.sphere_center[0] == Catch::Approx(1.0f));
    REQUIRE(cluster.sphere_center[1] == Catch::Approx(1.0f));
    REQUIRE(cluster.sphere_radius == Catch::Approx(std::sqrt(2.0f)));
    REQUIRE(virtual_mesh.stats.material_group_count == 1);
}

TEST_CASE("virtual mesh builder creates deterministic topology-aware clusters and hierarchy")
{
    arc::render::mesh_data source;
    source.material_index = 3;
    source.vertices.resize(390);
    source.indices.reserve(390);
    for (std::uint32_t triangle = 0; triangle < 130; ++triangle)
    {
        const std::uint32_t base = triangle * 3;
        source.vertices[base + 0].position[0] = static_cast<float>(triangle);
        source.vertices[base + 1].position[0] = static_cast<float>(triangle);
        source.vertices[base + 1].position[1] = 1.0f;
        source.vertices[base + 2].position[0] = static_cast<float>(triangle);
        source.vertices[base + 2].position[2] = 1.0f;
        source.indices.insert(source.indices.end(), {base, base + 1, base + 2});
    }

    const auto first = arc::render::build_virtual_mesh(source);
    const auto second = arc::render::build_virtual_mesh(source);

    REQUIRE(first.clusters.size() > 2);
    REQUIRE(std::all_of(first.clusters.begin(), first.clusters.end(),
                        [](const auto& cluster)
                        {
                            return cluster.vertex_count <= arc::render::virtual_geometry_max_vertices_per_cluster &&
                                   cluster.triangle_count <= arc::render::virtual_geometry_max_triangles_per_cluster;
                        }));
    REQUIRE_FALSE(first.root_nodes.empty());
    REQUIRE(first.root_nodes.size() <= 4);
    REQUIRE(first.lod_nodes.size() >= first.root_nodes.size());
    REQUIRE_FALSE(first.pages.empty());
    REQUIRE(first.stats.root_page_count > 0);
    REQUIRE(first.stats.source_triangle_count == 130);
    REQUIRE(first.stats.cluster_count == first.clusters.size());
    REQUIRE(first.stats.hierarchy_level_count >= 2);
    REQUIRE(first.stats.invalid_triangle_count == 0);
    REQUIRE(second.indices == first.indices);
    REQUIRE(second.page_payload == first.page_payload);
    REQUIRE(second.root_nodes == first.root_nodes);
    REQUIRE(second.clusters.size() == first.clusters.size());
    REQUIRE(second.clusters[0].first_index == first.clusters[0].first_index);
    REQUIRE(second.clusters[0].triangle_count == first.clusters[0].triangle_count);
    REQUIRE(second.clusters.back().sphere_radius == Catch::Approx(first.clusters.back().sphere_radius));
    REQUIRE(first.conventional_lods.size() == 4);
    REQUIRE(first.conventional_lods.front().indices.size() == source.indices.size());
    REQUIRE(first.conventional_lods.back().indices.size() < source.indices.size());

    std::vector<std::byte> decoded;
    REQUIRE(arc::render::decode_virtual_geometry_page(first, 0, decoded));
    REQUIRE_FALSE(decoded.empty());
}

TEST_CASE("virtual geometry graph selects mesh-shader rasterization without software passes")
{
    using namespace arc::render;
    resolved_render_config config;
    config.quality = render_quality_tier::ultra;
    config.path = render_path::deferred;
    config.features.gpu_driven_rendering = true;
    config.features.hzb_occlusion = true;
    config.features.virtual_geometry = true;
    config.features.virtual_geometry_path = virtual_geometry_raster_path::mesh_shader;

    const auto compiled = make_scene_draw_graph("viewport", config, true).compile().value();
    const auto contains = [&](builtin_render_pass expected)
    {
        return std::any_of(compiled.passes.begin(), compiled.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };
    REQUIRE(contains(builtin_render_pass::virtual_geometry_hierarchy_traversal));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_mesh_shader_visibility));
    REQUIRE_FALSE(contains(builtin_render_pass::virtual_geometry_cluster_binning));
    REQUIRE_FALSE(contains(builtin_render_pass::virtual_geometry_software_depth));
}

TEST_CASE("virtual geometry artifact is deterministic page aligned and integrity checked")
{
    using namespace arc::render;
    mesh_data source;
    source.name = "fixture";
    source.material_index = 19;
    source.vertices.resize(6);
    source.vertices[1].position[0] = 1.0f;
    source.vertices[2].position[1] = 1.0f;
    source.vertices[3].position[0] = 1.0f;
    source.vertices[4].position[0] = 1.0f;
    source.vertices[4].position[1] = 1.0f;
    source.vertices[5].position[1] = 1.0f;
    source.indices = {0, 1, 2, 3, 4, 5};
    const auto geometry = build_virtual_mesh(source, {.max_triangles_per_cluster = 1});
    const std::array inputs{virtual_geometry_artifact_source{
        .name = source.name, .material_index = source.material_index, .geometry = &geometry}};

    const auto first = encode_virtual_geometry_artifact(inputs, 0x12345678u);
    const auto second = encode_virtual_geometry_artifact(inputs, 0x12345678u);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(first.value() == second.value());

    const auto inspected = inspect_virtual_geometry_artifact(first.value());
    REQUIRE(inspected);
    REQUIRE(inspected.value().schema_version == virtual_geometry_artifact_schema_version);
    REQUIRE(inspected.value().conventional_artifact_hash == 0x12345678u);
    REQUIRE(inspected.value().meshes.size() == 1);
    REQUIRE(inspected.value().meshes[0].name == source.name);
    REQUIRE(inspected.value().meshes[0].material_index == source.material_index);
    REQUIRE(inspected.value().meshes[0].pages.size() == geometry.pages.size());
    REQUIRE(std::all_of(inspected.value().meshes[0].pages.begin(), inspected.value().meshes[0].pages.end(),
                        [](const auto& page) { return page.offset % virtual_geometry_artifact_page_alignment == 0; }));
    REQUIRE(std::any_of(inspected.value().meshes[0].pages.begin(), inspected.value().meshes[0].pages.end(),
                        [](const auto& page) { return page.root; }));

    auto corrupt = first.value();
    const auto page_offset = inspected.value().meshes[0].pages[0].offset;
    corrupt[static_cast<std::size_t>(page_offset)] ^= std::byte{1};
    const auto corrupt_index = inspect_virtual_geometry_artifact(corrupt);
    REQUIRE(corrupt_index);
    const auto rejected = read_virtual_geometry_artifact_page(corrupt, corrupt_index.value(), 0, 0);
    REQUIRE_FALSE(rejected);
    REQUIRE(rejected.error().code == virtual_geometry_artifact_error_code::integrity_failure);
}

TEST_CASE("virtual mesh builder honors custom cluster size and skips invalid triangles")
{
    arc::render::mesh_data source;
    source.material_index = 11;
    source.vertices.resize(6);
    source.vertices[0].position[0] = 0.0f;
    source.vertices[0].position[1] = 0.0f;
    source.vertices[1].position[0] = 1.0f;
    source.vertices[1].position[1] = 0.0f;
    source.vertices[2].position[0] = 0.0f;
    source.vertices[2].position[1] = 1.0f;
    source.vertices[3].position[0] = 2.0f;
    source.vertices[3].position[1] = 0.0f;
    source.vertices[4].position[0] = 3.0f;
    source.vertices[4].position[1] = 0.0f;
    source.vertices[5].position[0] = 2.0f;
    source.vertices[5].position[1] = 1.0f;
    source.indices = {0, 1, 2, 3, 4, 5, 0, 99, 1, 2};

    const auto virtual_mesh = arc::render::build_virtual_mesh(source, {.max_triangles_per_cluster = 1});

    REQUIRE(virtual_mesh.indices == std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5});
    REQUIRE(virtual_mesh.clusters.size() == 2);
    REQUIRE(virtual_mesh.clusters[0].triangle_count == 1);
    REQUIRE(virtual_mesh.clusters[1].triangle_count == 1);
    REQUIRE(virtual_mesh.clusters[0].material_index == 11);
    REQUIRE(virtual_mesh.clusters[1].material_index == 11);
    REQUIRE(virtual_mesh.clusters[0].page_byte_offset == 0);
    REQUIRE(virtual_mesh.clusters[1].page_byte_offset > virtual_mesh.clusters[0].page_byte_offset);
    REQUIRE(virtual_mesh.stats.source_vertex_count == 6);
    REQUIRE(virtual_mesh.stats.source_triangle_count == 3);
    REQUIRE(virtual_mesh.stats.invalid_triangle_count == 2);
    REQUIRE(virtual_mesh.stats.material_group_count == 1);
}

TEST_CASE("virtual geometry residency keeps roots and deduplicates prioritized page requests")
{
    arc::render::virtual_mesh_data geometry;
    geometry.pages = {{.uncompressed_size = 1024, .compressed_offset = 0, .compressed_size = 512, .root = true},
                      {.uncompressed_size = 768, .compressed_offset = 512, .compressed_size = 256}};
    const arc::render::virtual_mesh_handle handle{4, 2};
    arc::render::virtual_geometry_residency_manager residency(
        {.gpu_budget_bytes = 4096, .compressed_cpu_budget_bytes = 4096, .maximum_requests_per_frame = 8});
    residency.register_resource(handle, geometry, 7);
    residency.begin_frame(10);

    REQUIRE(residency.resident(handle, 7, 0));
    REQUIRE_FALSE(residency.resident(handle, 7, 1));
    const std::array requests{
        arc::render::virtual_geometry_page_request{.resource = handle,
                                                   .resource_generation = 7,
                                                   .page_index = 1,
                                                   .projected_error = 4.0f,
                                                   .visible_child = true},
        arc::render::virtual_geometry_page_request{
            .resource = handle, .resource_generation = 7, .page_index = 1, .projected_error = 2.0f}};
    residency.request(requests);
    const auto loads = residency.take_load_requests();
    REQUIRE(loads.size() == 1);
    REQUIRE(loads.front().byte_offset == 512);
    REQUIRE(residency.snapshot().deduplicated_requests == 1);

    residency.mark_loading(handle, 7, 1);
    residency.publish(handle, 7, 1, 768, 256);
    REQUIRE(residency.resident(handle, 7, 1));
    REQUIRE(residency.snapshot().resident_pages == 2);

    const std::array gpu_requests{arc::render::virtual_geometry_gpu_page_request{.resource_index = handle.index,
                                                                                 .handle_generation = handle.generation,
                                                                                 .resource_generation = 6,
                                                                                 .page_index = 1},
                                  arc::render::virtual_geometry_gpu_page_request{.resource_index = handle.index,
                                                                                 .handle_generation = handle.generation,
                                                                                 .resource_generation = 7,
                                                                                 .page_index = 1}};
    residency.request_gpu(gpu_requests);
    REQUIRE(residency.snapshot().stale_requests == 1);
}

TEST_CASE("virtual geometry GPU table update preserves hierarchy and page generations")
{
    using namespace arc::render;
    mesh_data source;
    source.material_index = 5;
    source.vertices.resize(3);
    source.vertices[1].position[0] = 1.0f;
    source.vertices[2].position[1] = 1.0f;
    source.indices = {0, 1, 2};
    const auto geometry = build_virtual_mesh(source);
    const virtual_mesh_handle handle{12, 4};

    const auto update = make_virtual_geometry_gpu_table_update(handle, geometry, 9);
    REQUIRE(update.resource == handle);
    REQUIRE(update.resource_generation == 9);
    REQUIRE(update.resources.size() == 1);
    REQUIRE(update.resources[0].node_count == geometry.lod_nodes.size());
    REQUIRE(update.resources[0].cluster_count == geometry.clusters.size());
    REQUIRE(update.nodes.size() == geometry.lod_nodes.size());
    REQUIRE(update.clusters.size() == geometry.clusters.size());
    REQUIRE(update.pages.size() == geometry.pages.size());
    REQUIRE(update.pages[0].resource_generation == 9);
    REQUIRE(update.clusters[0].page_byte_offset == geometry.clusters[0].page_byte_offset);
}

TEST_CASE("unified geometry binding selects cooked conventional LODs by geometric error")
{
    arc::render::geometry_resource_handle geometry{arc::render::mesh_handle{1, 1}};
    geometry.conventional_lods = {arc::render::mesh_handle{1, 1}, arc::render::mesh_handle{2, 1},
                                  arc::render::mesh_handle{3, 1}, arc::render::mesh_handle{4, 1}};
    geometry.conventional_lod_errors = {0.0f, 0.5f, 2.0f, 8.0f};
    geometry.conventional_lod_count = 4;

    REQUIRE(geometry.select_conventional_lod(0.25f) == arc::render::mesh_handle{1, 1});
    REQUIRE(geometry.select_conventional_lod(1.0f) == arc::render::mesh_handle{2, 1});
    REQUIRE(geometry.select_conventional_lod(3.0f) == arc::render::mesh_handle{3, 1});
    REQUIRE(geometry.select_conventional_lod(10.0f) == arc::render::mesh_handle{4, 1});
}

TEST_CASE("virtual geometry reference traversal selects resident children or a hole-free parent")
{
    using namespace arc::render;
    virtual_mesh_data geometry;
    geometry.pages.resize(3);
    geometry.pages[0].root = true;
    geometry.clusters.resize(3);
    geometry.clusters[0].page_index = 1;
    geometry.clusters[1].page_index = 2;
    geometry.clusters[2].page_index = 0;
    geometry.lod_nodes.resize(3);
    for (std::uint32_t index = 0; index < 2; ++index)
    {
        geometry.lod_nodes[index].first_cluster = index;
        geometry.lod_nodes[index].cluster_count = 1;
        geometry.lod_nodes[index].page_index = index + 1;
        geometry.lod_nodes[index].sphere_center[0] = index == 0 ? -0.5f : 0.5f;
        geometry.lod_nodes[index].sphere_radius = 0.5f;
    }
    auto& root = geometry.lod_nodes[2];
    root.first_cluster = 2;
    root.cluster_count = 1;
    root.first_child = 0;
    root.child_count = 2;
    root.page_index = 0;
    root.error = 1.0f;
    root.sphere_radius = 1.0f;
    geometry.hierarchy_children = {0, 1};
    geometry.root_nodes = {2};

    virtual_geometry_reference_view view;
    view.camera_position[2] = 10.0f;
    view.projection_scale = 100.0f;
    view.geometric_error_threshold = 1.0f;
    view.double_sided = true;

    const std::array<std::uint8_t, 3> root_only{1, 0, 0};
    const auto fallback = traverse_virtual_geometry_reference(geometry, root_only, view);
    REQUIRE(fallback.visible_clusters == std::vector<std::uint32_t>{2});
    REQUIRE(fallback.requested_pages == std::vector<std::uint32_t>{1, 2});
    REQUIRE(fallback.parent_fallbacks == 1);

    const std::array<std::uint8_t, 3> all_resident{1, 1, 1};
    const auto detailed = traverse_virtual_geometry_reference(geometry, all_resident, view);
    REQUIRE(detailed.visible_clusters == std::vector<std::uint32_t>{0, 1});
    REQUIRE(detailed.requested_pages.empty());
    REQUIRE(detailed.parent_fallbacks == 0);

    const auto gpu_fallback = traverse_virtual_geometry_gpu_reference(
        {7, 3}, 11, 23, 5, geometry, root_only, view, {.maximum_visible_clusters = 8, .maximum_page_requests = 1});
    REQUIRE(gpu_fallback.visible_clusters.size() == 1);
    REQUIRE(gpu_fallback.visible_clusters[0].instance_index == 23);
    REQUIRE(gpu_fallback.visible_clusters[0].resource_index == 7);
    REQUIRE(gpu_fallback.feedback.page_requests.size() == 1);
    REQUIRE(gpu_fallback.feedback.page_requests[0].handle_generation == 3);
    REQUIRE(gpu_fallback.feedback.page_requests[0].resource_generation == 11);
    REQUIRE(gpu_fallback.feedback.overflow.page_request_overflow == 1);

    const auto overflow = traverse_virtual_geometry_gpu_reference(
        {7, 3}, 11, 23, 5, geometry, all_resident, view, {.maximum_visible_clusters = 1, .maximum_page_requests = 8});
    REQUIRE(overflow.visible_clusters.empty());
    REQUIRE(overflow.feedback.overflow.visible_cluster_overflow == 1);
    REQUIRE(overflow.feedback.overflow.fallback_instance_count == 1);
}

TEST_CASE("virtual geometry software visibility resolves depth before stable primitive identity")
{
    using namespace arc::render;
    const std::array candidates{
        virtual_geometry_visibility_candidate{.depth = 0.75f, .visible_cluster = 8u, .triangle = 2u},
        virtual_geometry_visibility_candidate{.depth = 0.25f, .visible_cluster = 9u, .triangle = 4u},
        virtual_geometry_visibility_candidate{.depth = 0.25f, .visible_cluster = 3u, .triangle = 7u},
    };
    const auto result = resolve_virtual_geometry_visibility(candidates);
    REQUIRE(result.encoded_depth == encode_virtual_geometry_depth(0.25f));
    REQUIRE(result.identity == encode_virtual_geometry_visibility_id(3u, 7u));
    REQUIRE(encode_virtual_geometry_visibility_id(0x01ffffffu, 400u) == 0xffffffffu);
    REQUIRE(encode_virtual_geometry_depth(-1.0f) == encode_virtual_geometry_depth(0.0f));
    REQUIRE(encode_virtual_geometry_depth(2.0f) == encode_virtual_geometry_depth(1.0f));
}

TEST_CASE("virtual geometry material resolve reconstructs perspective-correct weights")
{
    using namespace arc::render;
    const auto weights = perspective_correct_virtual_geometry_barycentrics({0.25f, 0.25f, 0.5f}, {1.0f, 2.0f, 4.0f});
    REQUIRE(weights[0] == Catch::Approx(0.5f));
    REQUIRE(weights[1] == Catch::Approx(0.25f));
    REQUIRE(weights[2] == Catch::Approx(0.25f));
    REQUIRE(weights[0] + weights[1] + weights[2] == Catch::Approx(1.0f));

    const auto fallback = perspective_correct_virtual_geometry_barycentrics({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    REQUIRE(fallback == std::array{1.0f, 0.0f, 0.0f});
}

TEST_CASE("GLB mesh loader reads checked-in editor startup mesh")
{
    const std::filesystem::path path =
        std::filesystem::path(ARC_RENDER_TEST_ASSET_ROOT) / "models" / "UAL2_Standard.glb";
    REQUIRE(std::filesystem::exists(path));

    const auto result = arc::render::load_gltf_mesh(path);

    INFO(result.message);
    REQUIRE(result.succeeded());
    REQUIRE_FALSE(result.mesh.name.empty());
    REQUIRE_FALSE(result.mesh.vertices.empty());
    REQUIRE_FALSE(result.mesh.indices.empty());
}

TEST_CASE("lighting geometry cooking is deterministic and produces hole-free proxy data")
{
    const auto mesh = arc::render::make_cube_mesh(2.0f);
    const auto first = arc::render::build_lighting_geometry(mesh);
    const auto second = arc::render::build_lighting_geometry(mesh);

    REQUIRE(first.statistics.source_triangles == mesh.indices.size() / 3u);
    REQUIRE(first.geometry.cards.size() == 6u);
    REQUIRE_FALSE(first.geometry.distance_field.bricks.empty());
    REQUIRE_FALSE(first.geometry.distance_field.pages.empty());
    REQUIRE(first.geometry.distance_field.content_hash == second.geometry.distance_field.content_hash);
    REQUIRE(first.geometry.distance_field.pages == second.geometry.distance_field.pages);
    REQUIRE(first.geometry.cards.front().fallback_card < first.geometry.cards.size());

    const auto hit = arc::render::trace_mesh_distance_field(
        first.geometry.distance_field,
        {.origin = {0.0f, 0.0f, 3.0f}, .direction = {0.0f, 0.0f, -1.0f}, .maximum_distance = 8.0f});
    REQUIRE(hit.hit);
    REQUIRE(hit.source == arc::render::lighting_trace_source::software_distance_field);
    REQUIRE(hit.distance > 1.0f);
    REQUIRE(hit.distance < 3.0f);
}
