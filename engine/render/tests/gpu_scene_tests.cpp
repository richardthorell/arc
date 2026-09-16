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

#include "render_test_support.h"

using arc::render::tests::recording_backend;

TEST_CASE("render world preparation culls sorts batches and emits indirect commands")
{
    arc::render::render_world_packet packet;
    packet.camera.view_projection = arc::math::identity<float, 4>();
    packet.items.push_back({.mesh = {.index = 1, .generation = 1},
                            .material = {.index = 2, .generation = 1},
                            .world_bounds = arc::geometric::box3f{arc::geometric::point3f{-0.5f, -0.5f, -0.5f},
                                                                  arc::geometric::point3f{0.5f, 0.5f, 0.5f}},
                            .label = "A"});
    packet.items.push_back({.mesh = {.index = 1, .generation = 1},
                            .material = {.index = 2, .generation = 1},
                            .world_bounds = arc::geometric::box3f{arc::geometric::point3f{-0.25f, -0.25f, -0.25f},
                                                                  arc::geometric::point3f{0.25f, 0.25f, 0.25f}},
                            .label = "B"});
    packet.items.push_back({.mesh = {.index = 5, .generation = 1},
                            .material = {.index = 7, .generation = 1},
                            .world_bounds = arc::geometric::box3f{arc::geometric::point3f{4.0f, 4.0f, 4.0f},
                                                                  arc::geometric::point3f{5.0f, 5.0f, 5.0f}},
                            .label = "culled"});

    arc::render::prepare_render_world(packet);

    REQUIRE(packet.visible_items.size() == 2);
    REQUIRE(packet.culled_item_count == 1);
    REQUIRE(packet.instance_batches.size() == 1);
    REQUIRE(packet.instance_batches[0].item_count == 2);
    REQUIRE(packet.indirect_draws.size() == 1);
    REQUIRE(packet.indirect_draws[0].instance_count == 2);
}

TEST_CASE("GPU Scene keeps stable slots and emits precise incremental updates")
{
    using namespace arc::render;
    render_world_packet packet;
    packet.gpu_scene_world_id = 17;
    packet.world_epoch = 3;
    packet.items.push_back({.mesh = {.index = 1, .generation = 1},
                            .material = {.index = 2, .generation = 1},
                            .world_bounds = arc::geometric::box3f{arc::geometric::point3f{-1.0f, -1.0f, -1.0f},
                                                                  arc::geometric::point3f{1.0f, 1.0f, 1.0f}},
                            .object_id = {.index = 41, .generation = 1}});

    gpu_scene scene;
    const auto initial = scene.synchronize(packet, 1);
    REQUIRE(initial.active_instance_count == 1);
    REQUIRE(initial.updates.size() == 2); // epoch reset followed by first upload
    REQUIRE(initial.dirty_ranges == std::vector<gpu_table_dirty_range>{{.first = 0, .count = 1}});
    const auto handle = initial.updates.back().handle;
    REQUIRE(handle.valid());
    REQUIRE(scene.find(handle) != nullptr);

    auto moved = packet.items[0].model;
    moved(0, 3) = 4.0f;
    packet.items[0].model = moved;
    const auto update = scene.synchronize(packet, 2);
    REQUIRE(update.updates.size() == 1);
    REQUIRE(update.updates[0].handle == handle);
    REQUIRE(update.updates[0].dirty == gpu_scene_dirty::transform);
    REQUIRE(update.updates[0].instance.previous_model(0, 3) == Catch::Approx(0.0f));
    const auto second_view = scene.synchronize(packet, 2);
    REQUIRE(second_view.updates.empty());
    REQUIRE(contains(scene.find(handle)->flags, gpu_scene_instance_flag::recently_changed));

    const auto settled = scene.synchronize(packet, 3);
    REQUIRE(settled.updates.size() == 1);
    REQUIRE(settled.updates[0].dirty == (gpu_scene_dirty::transform | gpu_scene_dirty::flags));
    REQUIRE_FALSE(contains(settled.updates[0].instance.flags, gpu_scene_instance_flag::recently_changed));

    packet.items.clear();
    const auto removed = scene.synchronize(packet, 4);
    REQUIRE(removed.active_instance_count == 0);
    REQUIRE(removed.updates.size() == 1);
    REQUIRE(removed.updates[0].kind == gpu_scene_update_kind::destroy);
    REQUIRE(removed.updates[0].handle == handle);
    REQUIRE(scene.find(handle) == nullptr);

    packet.items.push_back({.mesh = {.index = 1, .generation = 1}, .object_id = {.index = 42, .generation = 1}});
    const auto before_retirement = scene.synchronize(packet, 5);
    const auto temporary_handle = before_retirement.updates.back().handle;
    REQUIRE(temporary_handle.index != handle.index);
    packet.items.clear();
    static_cast<void>(scene.synchronize(packet, 6));
    packet.items.push_back({.mesh = {.index = 1, .generation = 1}, .object_id = {.index = 43, .generation = 1}});
    const auto after_retirement = scene.synchronize(packet, 8);
    const auto recycled_handle = after_retirement.updates.back().handle;
    REQUIRE(recycled_handle.index == handle.index);
    REQUIRE(recycled_handle.generation != handle.generation);
}

TEST_CASE("GPU Scene preserves virtual material attribute references")
{
    using namespace arc::render;
    render_world_packet packet;
    packet.gpu_scene_world_id = 23;
    packet.world_epoch = 1;
    packet.virtual_items.push_back({.mesh = {.index = 4, .generation = 2},
                                    .material = {.index = 6, .generation = 3},
                                    .material_attribute_texture = {.index = 9, .generation = 5},
                                    .root_node = 7u,
                                    .object_id = {.index = 14, .generation = 1}});

    gpu_scene scene;
    const auto initial = scene.synchronize(packet, 1);
    REQUIRE(initial.active_instance_count == 1u);
    REQUIRE(packet.virtual_items.front().gpu_scene_instance.valid());
    const auto handle = packet.virtual_items.front().gpu_scene_instance;
    const auto* instance = scene.find(handle);
    REQUIRE(instance != nullptr);
    CHECK(instance->geometry_kind == gpu_scene_geometry_kind::virtual_mesh);
    CHECK(instance->material_attribute_texture == texture_handle{.index = 9, .generation = 5});

    // Let the one-frame recently-changed flag settle before checking a pure material update.
    (void)scene.synchronize(packet, 2);
    packet.virtual_items.front().material_attribute_texture = {.index = 9, .generation = 6};
    const auto updated = scene.synchronize(packet, 3);
    REQUIRE(updated.updates.size() == 1u);
    CHECK(updated.updates.front().dirty == gpu_scene_dirty::material);
    CHECK(updated.updates.front().instance.material_attribute_texture == texture_handle{.index = 9, .generation = 6});
}

TEST_CASE("GPU-driven preparation skips allocating CPU visibility unless validation requests it")
{
    arc::render::render_world_packet packet;
    packet.camera.view_projection = arc::math::identity<float, 4>();
    packet.items.push_back({.mesh = {.index = 1, .generation = 1},
                            .world_bounds = arc::geometric::box3f{arc::geometric::point3f{-0.5f, -0.5f, -0.5f},
                                                                  arc::geometric::point3f{0.5f, 0.5f, 0.5f}}});
    arc::render::prepare_render_world(packet, {.gpu_driven = true});
    REQUIRE(packet.visible_items.empty());
    arc::render::prepare_render_world(packet, {.gpu_driven = true, .retain_cpu_reference = true});
    REQUIRE(packet.visible_items == std::vector<std::uint32_t>{0});
}

TEST_CASE("GPU table dirty ranges are sorted coalesced and duplicate free")
{
    const std::array indices{9u, 2u, 3u, 9u, 4u, 12u};
    const auto ranges = arc::render::coalesce_gpu_table_dirty_ranges(indices);
    REQUIRE(ranges == std::vector<arc::render::gpu_table_dirty_range>{
                          {.first = 2, .count = 3}, {.first = 9, .count = 1}, {.first = 12, .count = 1}});
}

TEST_CASE("GPU draw compaction stably scatters bins and preserves overflow for CPU fallback")
{
    using namespace arc::render;
    const std::array records{
        gpu_draw_record{.instance_index = 10, .pipeline_bin = 2},
        gpu_draw_record{.instance_index = 11, .pipeline_bin = 0},
        gpu_draw_record{.instance_index = 12, .pipeline_bin = 2},
        gpu_draw_record{.instance_index = 13, .pipeline_bin = 7},
        gpu_draw_record{.instance_index = 14, .pipeline_bin = 1},
    };

    const auto compacted = compact_gpu_draw_records(records, 3, 3);
    REQUIRE(compacted.visible_draws.size() == 3);
    REQUIRE(compacted.visible_draws[0].instance_index == 11);
    REQUIRE(compacted.visible_draws[1].instance_index == 14);
    REQUIRE(compacted.visible_draws[2].instance_index == 10);
    REQUIRE(compacted.bin_offsets == std::vector<std::uint32_t>{0, 1, 2});
    REQUIRE(compacted.bin_counts == std::vector<std::uint32_t>{1, 1, 1});
    REQUIRE(compacted.overflow_draws.size() == 2);
    REQUIRE(compacted.overflow_draws[0].instance_index == 13);
    REQUIRE(compacted.overflow_draws[1].instance_index == 12);
    REQUIRE(compacted.statistics.candidates == records.size());
    REQUIRE(compacted.statistics.visible == 3);
    REQUIRE(compacted.statistics.active_bins == 3);
    REQUIRE(compacted.statistics.indirect_commands == 3);
    REQUIRE(compacted.statistics.overflow_records == 2);
    REQUIRE(compacted.statistics.cpu_submissions == 2);
}

TEST_CASE("GPU resource tables publish generational records and reusable shared heap ranges")
{
    using namespace arc::render;
    gpu_resource_tables tables;
    const resource_handle first{.index = 5, .generation = 2};
    const std::array<mesh_vertex, 3> vertices{};
    const std::array<std::uint32_t, 3> indices{0, 1, 2};

    const auto initial = tables.publish_geometry(first, std::as_bytes(std::span{vertices}), sizeof(mesh_vertex),
                                                 std::as_bytes(std::span{indices}), sizeof(std::uint32_t), 10);
    REQUIRE(initial.table == gpu_resource_table_kind::geometry);
    REQUIRE(initial.element_stride == sizeof(gpu_geometry_table_record));
    REQUIRE(initial.capacity >= 6);
    REQUIRE(initial.updates.size() == 1);
    REQUIRE(initial.heap_updates.size() == 2);
    REQUIRE(initial.reuse_after_frame == 10 + default_gpu_table_slot_reuse_delay_frames);
    REQUIRE(tables.find(gpu_resource_table_kind::geometry, first) ==
            gpu_resource_table_reference{.index = 5, .generation = 2});

    gpu_geometry_table_record first_record{};
    std::memcpy(&first_record, initial.payload.data(), sizeof(first_record));
    REQUIRE(first_record.generation == 2);
    REQUIRE(first_record.vertex_count == 3);
    REQUIRE(first_record.index_count == 3);
    const auto first_vertex_offset = first_record.vertex_offset;
    const auto first_index_offset = first_record.index_offset;

    const auto update = tables.publish_geometry(first, std::as_bytes(std::span{vertices}), sizeof(mesh_vertex),
                                                std::as_bytes(std::span{indices}), sizeof(std::uint32_t), 11);
    gpu_geometry_table_record updated_record{};
    std::memcpy(&updated_record, update.payload.data(), sizeof(updated_record));
    REQUIRE(updated_record.vertex_offset == first_vertex_offset);
    REQUIRE(updated_record.index_offset == first_index_offset);

    const auto retired = tables.tombstone(gpu_resource_table_kind::geometry, first, 12);
    REQUIRE(retired.updates.size() == 1);
    REQUIRE(retired.updates[0].kind == gpu_table_update_kind::tombstone);
    REQUIRE_FALSE(tables.find(gpu_resource_table_kind::geometry, first));
    REQUIRE(tables.snapshot(gpu_resource_table_kind::geometry).tombstones == 1);

    const resource_handle recycled{.index = 5, .generation = 3};
    const auto replacement = tables.publish_geometry(recycled, std::as_bytes(std::span{vertices}), sizeof(mesh_vertex),
                                                     std::as_bytes(std::span{indices}), sizeof(std::uint32_t), 13);
    gpu_geometry_table_record replacement_record{};
    std::memcpy(&replacement_record, replacement.payload.data(), sizeof(replacement_record));
    REQUIRE(replacement_record.generation == 3);
    REQUIRE(replacement_record.vertex_offset == first_vertex_offset);
    REQUIRE(replacement_record.index_offset == first_index_offset);
    REQUIRE(tables.snapshot(gpu_resource_table_kind::geometry).live_entries == 1);
    REQUIRE(tables.geometry_heap_snapshot().live_allocations == 1);
}

TEST_CASE("GPU material tables retain stable texture generations")
{
    using namespace arc::render;
    gpu_resource_tables tables;
    const texture_handle texture{.index = 7, .generation = 4};
    gpu_texture_table_record texture_record{.generation = texture.generation,
                                            .descriptor_index = 12,
                                            .descriptor_generation = 3,
                                            .mip_count = 8,
                                            .width = 2048,
                                            .height = 2048};
    const auto texture_update = tables.publish_texture(texture, texture_record, 1);
    REQUIRE(texture_update.updates.size() == 1);

    const material_handle material{.index = 2, .generation = 9};
    gpu_material_table_record material_record{};
    material_record.generation = material.generation;
    material_record.texture_indices.fill(resource_handle::invalid_index);
    material_record.texture_indices[0] = texture.index;
    material_record.texture_generations[0] = texture.generation;
    const auto material_update = tables.publish_material(material, material_record, 1);
    REQUIRE(material_update.updates.size() == 1);
    gpu_material_table_record published{};
    std::memcpy(&published, material_update.payload.data(), sizeof(published));
    REQUIRE(published.generation == material.generation);
    REQUIRE(published.texture_indices[0] == texture.index);
    REQUIRE(published.texture_generations[0] == texture.generation);
    REQUIRE(tables.snapshot(gpu_resource_table_kind::texture).live_entries == 1);
    REQUIRE(tables.snapshot(gpu_resource_table_kind::material).live_entries == 1);
}

TEST_CASE("renderer resource creation publishes GPU tables without changing public handles")
{
    using namespace arc::render;
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    renderer renderer;
    renderer.set_backend(std::move(backend));

    texture_data texture_data;
    texture_data.width = 1;
    texture_data.height = 1;
    texture_data.pixels.resize(4);
    const auto texture = renderer.create_texture(std::move(texture_data));

    material_descriptor material_data;
    material_data.base_color_texture = texture;
    const auto material = renderer.create_material(std::move(material_data));

    mesh_data mesh_data;
    mesh_data.usage = mesh_usage::dynamic_per_frame;
    mesh_data.vertices.resize(3);
    mesh_data.indices = {0, 1, 2};
    const auto mesh = renderer.create_mesh(std::move(mesh_data));

    REQUIRE(texture.valid());
    REQUIRE(material.valid());
    REQUIRE(mesh.valid());
    REQUIRE(renderer.gpu_resources().find(gpu_resource_table_kind::texture, texture));
    REQUIRE(renderer.gpu_resources().find(gpu_resource_table_kind::material, material));
    REQUIRE(renderer.gpu_resources().find(gpu_resource_table_kind::geometry, mesh));

    REQUIRE(renderer.render_frame(1, make_clear_present_graph("viewport")));
    REQUIRE(std::count(backend_ptr->last_event_types.begin(), backend_ptr->last_event_types.end(),
                       render_event_type::gpu_resource_table_update) == 3);
}

TEST_CASE("GPU transparent keys preserve bin then back-to-front depth and stable ties")
{
    using arc::render::make_gpu_transparent_sort_key;
    REQUIRE(make_gpu_transparent_sort_key(0.9f, 2u, 4u) < make_gpu_transparent_sort_key(0.1f, 2u, 4u));
    REQUIRE(make_gpu_transparent_sort_key(0.5f, 2u, 4u) < make_gpu_transparent_sort_key(0.5f, 3u, 1u));
    REQUIRE(make_gpu_transparent_sort_key(0.5f, 2u, 4u) < make_gpu_transparent_sort_key(0.5f, 2u, 5u));
}

TEST_CASE("GPU transparent reference sort is stable back to front within pipeline bins")
{
    using namespace arc::render;
    const std::array records{
        gpu_draw_record{
            .instance_index = 5u, .pipeline_bin = 2u, .sort_key = make_gpu_transparent_sort_key(0.2f, 2u, 5u)},
        gpu_draw_record{
            .instance_index = 8u, .pipeline_bin = 1u, .sort_key = make_gpu_transparent_sort_key(0.4f, 1u, 8u)},
        gpu_draw_record{
            .instance_index = 3u, .pipeline_bin = 2u, .sort_key = make_gpu_transparent_sort_key(0.8f, 2u, 3u)},
    };
    const auto sorted = sort_gpu_transparent_records(records);
    REQUIRE(sorted[0].pipeline_bin == 1u);
    REQUIRE(sorted[1].instance_index == 3u);
    REQUIRE(sorted[2].instance_index == 5u);
}
