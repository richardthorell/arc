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

TEST_CASE("CPU skinning is a normalized portable fallback", "[render][skinning]")
{
    using namespace arc;
    using namespace arc::render;

    mesh_vertex source{};
    source.position[0] = 1.0f;
    source.normal[1] = 1.0f;
    source.tangent[0] = 1.0f;
    source.tangent[3] = -1.0f;
    source.texcoord[0] = 0.25f;
    source.color[2] = 0.5f;
    mesh_skin_vertex influences{};
    influences.joint_indices[0] = 0u;
    influences.joint_indices[1] = 1u;
    influences.joint_indices[2] = 99u;
    influences.joint_weights[0] = 1.0f;
    influences.joint_weights[1] = 3.0f;
    influences.joint_weights[2] = 100.0f;
    const std::array joints{math::translation(math::vector3f{1.0f, 0.0f, 0.0f}),
                            math::translation(math::vector3f{0.0f, 2.0f, 0.0f})};
    mesh_vertex result{};

    REQUIRE(skin_mesh_vertices(std::span{&source, 1u}, std::span{&influences, 1u}, joints, std::span{&result, 1u}));
    REQUIRE(result.position[0] == Catch::Approx(1.25f));
    REQUIRE(result.position[1] == Catch::Approx(1.5f));
    REQUIRE(result.normal[1] == Catch::Approx(1.0f));
    REQUIRE(result.tangent[0] == Catch::Approx(1.0f));
    REQUIRE(result.tangent[3] == -1.0f);
    REQUIRE(result.texcoord[0] == source.texcoord[0]);
    REQUIRE(result.color[2] == source.color[2]);

    influences.joint_weights[0] = -1.0f;
    influences.joint_weights[1] = 0.0f;
    REQUIRE(skin_mesh_vertices(std::span{&source, 1u}, std::span{&influences, 1u}, joints, std::span{&result, 1u}));
    REQUIRE(result.position[0] == source.position[0]);
    REQUIRE_FALSE(skin_mesh_vertices(std::span{&source, 1u}, {}, joints, std::span{&result, 1u}));
}
TEST_CASE("viewport output metadata is backend neutral and unsupported by default")
{
    recording_backend backend;
    const arc::render::viewport_output_descriptor descriptor{
        .id = "viewport-a", .type = arc::render::viewport_output_type::shared_texture, .width = 1280, .height = 720};
    const auto created = backend.create_viewport_output(descriptor);
    REQUIRE_FALSE(created);
    REQUIRE(created.error().code == arc::render::surface_frame_error_code::unsupported);

    const arc::render::shared_viewport_frame frame{
        .viewport_id = "viewport-a",
        .frame_id = 7,
        .generation = 3,
        .width = 1280,
        .height = 720,
        .format = arc::render::viewport_pixel_format::bgra8_unorm,
        .texture = {.type = arc::render::external_gpu_handle_type::win32_nt_handle, .payload = 0x1234u},
        .synchronization = {.producer_complete = true, .value = 7}};
    REQUIRE(frame.texture.valid());
    REQUIRE(frame.generation == 3);
    REQUIRE(frame.synchronization.producer_complete);
}
TEST_CASE("render handles reject stale generations")
{
    arc::render::handle_pool pool;
    const auto first = pool.allocate();

    REQUIRE(first.valid());
    REQUIRE(pool.alive(first));
    REQUIRE(pool.live_count() == 1);
    REQUIRE(pool.release(first));
    REQUIRE_FALSE(pool.alive(first));

    const auto second = pool.allocate();
    REQUIRE(second.index == first.index);
    REQUIRE(second.generation != first.generation);
    REQUIRE(pool.alive(second));
    REQUIRE_FALSE(pool.release(first));
}

TEST_CASE("render frame queue commits buffers into frame packets")
{
    arc::render::render_frame_queue queue;

    arc::render::render_event_buffer first;
    arc::render::render_event_writer first_writer(first);
    first_writer.debug_marker("a");
    queue.submit(std::move(first));

    arc::render::render_event_buffer second;
    arc::render::render_event_writer second_writer(second);
    second_writer.viewport_resize(1920, 1080);
    queue.submit(std::move(second));

    REQUIRE(queue.pending_buffer_count() == 2);

    const auto packet = queue.commit(7);
    REQUIRE(packet.frame_index == 7);
    REQUIRE(packet.events.size() == 2);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::debug_marker);
    REQUIRE(std::get<arc::render::debug_marker_event>(packet.events[0].payload).label == "a");
    REQUIRE(packet.events[1].type() == arc::render::render_event_type::viewport_resize);
    const auto& resize = std::get<arc::render::viewport_resize_event>(packet.events[1].payload);
    REQUIRE(resize.width == 1920);
    REQUIRE(resize.height == 1080);
    REQUIRE(queue.pending_buffer_count() == 0);
}

TEST_CASE("render event writer emits mesh upload and draw events")
{
    arc::render::render_event_buffer buffer;
    arc::render::render_event_writer writer(buffer);
    arc::render::mesh_handle mesh{.index = 4, .generation = 2};
    arc::render::texture_handle texture{.index = 5, .generation = 1};
    arc::render::material_handle material{.index = 6, .generation = 1};
    auto mesh_data = std::make_shared<arc::render::mesh_data>();
    mesh_data->name = "triangle";
    auto texture_data = std::make_shared<arc::render::texture_data>();
    texture_data->name = "white";
    auto material_data = std::make_shared<arc::render::material_descriptor>();
    material_data->name = "default";

    writer.mesh_upload(mesh, mesh_data, "triangle");
    writer.texture_upload(texture, texture_data, "white");
    writer.material_upload(material, material_data, "default");
    writer.draw_mesh(mesh, material, arc::math::identity<float, 4>(), arc::math::identity<float, 4>(),
                     arc::render::render_mode::wireframe, arc::render::mesh_visualization_mode::world_normal, true,
                     arc::math::vector4f{1.0f, 0.5f, 0.0f, 1.0f}, "triangle");
    writer.draw_mesh_tinted(mesh, material, arc::math::identity<float, 4>(), arc::math::identity<float, 4>(),
                            arc::render::render_mode::shaded, arc::render::mesh_visualization_mode::standard, false,
                            arc::math::vector4f{0.25f, 0.5f, 0.75f, 1.0f}, arc::math::vector4f::one, "tinted");
    writer.directional_light({0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 3.0f, true, "Sun");

    REQUIRE(buffer.events().size() == 6);
    REQUIRE(buffer.events()[0].type() == arc::render::render_event_type::mesh_upload);
    const auto& upload = std::get<arc::render::mesh_upload_event>(buffer.events()[0].payload);
    REQUIRE(upload.handle == mesh);
    REQUIRE(upload.mesh == mesh_data);
    REQUIRE(buffer.events()[1].type() == arc::render::render_event_type::texture_upload);
    REQUIRE(std::get<arc::render::texture_upload_event>(buffer.events()[1].payload).texture == texture_data);
    REQUIRE(buffer.events()[2].type() == arc::render::render_event_type::material_upload);
    REQUIRE(std::get<arc::render::material_upload_event>(buffer.events()[2].payload).material == material_data);
    const auto& tinted = std::get<arc::render::draw_mesh_event>(buffer.events()[4].payload);
    REQUIRE(tinted.base_color_tint[0] == Catch::Approx(0.25f));
    REQUIRE(buffer.events()[3].type() == arc::render::render_event_type::draw);
    const auto& draw = std::get<arc::render::draw_mesh_event>(buffer.events()[3].payload);
    REQUIRE(draw.mesh == mesh);
    REQUIRE(draw.material == material);
    REQUIRE(draw.mode == arc::render::render_mode::wireframe);
    REQUIRE(draw.visualization == arc::render::mesh_visualization_mode::world_normal);
    REQUIRE(draw.selected);
    REQUIRE(draw.label == "triangle");
    REQUIRE(buffer.events()[5].type() == arc::render::render_event_type::directional_light);
    const auto& light = std::get<arc::render::directional_light_event>(buffer.events()[5].payload);
    REQUIRE(light.label == "Sun");
    REQUIRE(light.intensity == Catch::Approx(3.0f));
}

TEST_CASE("render frame queue accepts producer buffers from multiple threads")
{
    arc::render::render_frame_queue queue;
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;

    for (int index = 0; index < 4; ++index)
    {
        threads.emplace_back(
            [&, index]()
            {
                arc::render::render_event_buffer buffer;
                arc::render::render_event_writer writer(buffer);
                writer.debug_marker("producer " + std::to_string(index));
                ready.fetch_add(1);
                queue.submit(std::move(buffer));
            });
    }

    for (auto& thread : threads)
        thread.join();

    REQUIRE(ready.load() == 4);
    const auto packet = queue.commit(1);
    REQUIRE(packet.events.size() == 4);
}
