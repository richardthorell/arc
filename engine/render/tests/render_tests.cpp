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

namespace
{

class recording_backend final : public arc::render::render_backend
{
public:
    arc::render::render_backend_type type() const noexcept override
    {
        return arc::render::render_backend_type::vulkan;
    }

    const arc::render::render_capabilities& capabilities() const noexcept override
    {
        return capabilities_;
    }

    void configure(const arc::render::resolved_render_config& config) override
    {
        configured = config;
    }

    arc::render::render_submit_result submit(const arc::render::render_frame_packet& packet,
                                             const arc::render::compiled_render_graph& graph) override
    {
        last_frame = packet.frame_index;
        last_event_count = packet.events.size();
        last_event_types.clear();
        for (const auto& event : packet.events)
            last_event_types.push_back(event.type());
        last_pass_count = graph.passes.size();
        profile.frame_index = packet.frame_index;
        profile.graph = graph;
        profile.summary = "recorded";
        profile.clustered_lights = {.tile_size_pixels = 32,
                                    .tiles_x = 2,
                                    .tiles_y = 3,
                                    .depth_slices = 16,
                                    .cluster_count = 96,
                                    .point_light_references = 4,
                                    .spot_light_references = 2,
                                    .overflow_count = 1,
                                    .available = true};
        return arc::render::render_submit_result::success();
    }

    void resize_viewport(std::uint32_t width, std::uint32_t height) override
    {
        viewport_width = width;
        viewport_height = height;
    }

    arc::render::render_viewport_texture viewport_texture() const noexcept override
    {
        return {.id = texture_id, .width = viewport_width, .height = viewport_height};
    }

    arc::render::render_backend_frame_profile last_frame_profile() const override
    {
        return profile;
    }

    void request_object_pick(arc::render::render_object_pick_request request) override
    {
        pick_request = request;
        pick_requested = true;
    }
    void request_frame_capture(const arc::render::render_frame_capture_request& request) override
    {
        capture_request = request;
        capture_requested = true;
    }
    arc::render::render_frame_capture_result last_frame_capture() const override
    {
        return capture_result;
    }

    arc::render::render_capabilities capabilities_{};
    arc::render::resolved_render_config configured{};
    arc::render::render_backend_frame_profile profile{};
    arc::render::render_object_pick_request pick_request{};
    arc::render::render_frame_capture_request capture_request{};
    arc::render::render_frame_capture_result capture_result{};
    std::uint64_t last_frame{};
    std::size_t last_event_count{};
    std::size_t last_pass_count{};
    std::vector<arc::render::render_event_type> last_event_types;
    std::uint64_t texture_id{99};
    std::uint32_t viewport_width{};
    std::uint32_t viewport_height{};
    bool pick_requested{};
    bool capture_requested{};
};

class recording_command_encoder final : public arc::render::command_encoder
{
public:
    void begin_submission(const arc::render::compiled_queue_submission& submission) override
    {
        submissions.push_back(submission.queue);
    }

    void resource_barrier(const arc::render::render_resource_transition& transition) override
    {
        barriers.push_back(transition.resource);
    }

    void begin_pass(const arc::render::compiled_render_pass& pass) override
    {
        passes.push_back(pass.name);
    }

    void end_pass() override
    {
        ++ended_passes;
    }

    std::vector<std::string> barriers;
    std::vector<std::string> passes;
    std::vector<arc::render::render_queue_type> submissions;
    std::size_t ended_passes{};
};

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

void count_recorded_pass(arc::render::render_pass_context& context)
{
    ++*context.payload<std::uint32_t*>();
}

void append_u32(std::vector<std::byte>& bytes, std::uint32_t value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

void write_u32_at(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::vector<std::byte> make_dds_header(std::uint32_t width, std::uint32_t height, std::uint32_t mip_count,
                                       std::uint32_t pixel_flags, std::uint32_t four_cc,
                                       std::uint32_t rgb_bit_count = 0, std::uint32_t r_mask = 0,
                                       std::uint32_t g_mask = 0, std::uint32_t b_mask = 0, std::uint32_t a_mask = 0)
{
    std::vector<std::byte> bytes(128);
    write_u32_at(bytes, 0, 0x20534444);
    write_u32_at(bytes, 4, 124);
    write_u32_at(bytes, 8, 0x0002100Fu);
    write_u32_at(bytes, 12, height);
    write_u32_at(bytes, 16, width);
    write_u32_at(bytes, 28, mip_count);
    write_u32_at(bytes, 76, 32);
    write_u32_at(bytes, 80, pixel_flags);
    write_u32_at(bytes, 84, four_cc);
    write_u32_at(bytes, 88, rgb_bit_count);
    write_u32_at(bytes, 92, r_mask);
    write_u32_at(bytes, 96, g_mask);
    write_u32_at(bytes, 100, b_mask);
    write_u32_at(bytes, 104, a_mask);
    return bytes;
}

void append_f32(std::vector<std::byte>& bytes, float value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

void append_u16(std::vector<std::byte>& bytes, std::uint16_t value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

void pad4(std::vector<std::byte>& bytes, std::byte value)
{
    while ((bytes.size() % 4) != 0)
        bytes.push_back(value);
}

std::filesystem::path write_triangle_glb()
{
    std::vector<std::byte> bin;
    const std::size_t position_offset = bin.size();
    for (const float value : {0.0f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f})
        append_f32(bin, value);
    const std::size_t normal_offset = bin.size();
    for (int index = 0; index < 3; ++index)
    {
        append_f32(bin, 0.0f);
        append_f32(bin, 0.0f);
        append_f32(bin, 1.0f);
    }
    const std::size_t uv_offset = bin.size();
    for (const float value : {0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f})
        append_f32(bin, value);
    const std::size_t index_offset = bin.size();
    append_u16(bin, 0);
    append_u16(bin, 1);
    append_u16(bin, 2);
    const std::size_t image_offset = bin.size();
    for (const std::byte value : {std::byte{0x89}, std::byte{0x50}, std::byte{0x4e}, std::byte{0x47}})
        bin.push_back(value);
    pad4(bin, std::byte{0});

    const std::string json = "{\"asset\":{\"version\":\"2.0\"},"
                             "\"buffers\":[{\"byteLength\":" +
                             std::to_string(bin.size()) +
                             "}],"
                             "\"bufferViews\":["
                             "{\"buffer\":0,\"byteOffset\":" +
                             std::to_string(position_offset) +
                             ",\"byteLength\":36},"
                             "{\"buffer\":0,\"byteOffset\":" +
                             std::to_string(normal_offset) +
                             ",\"byteLength\":36},"
                             "{\"buffer\":0,\"byteOffset\":" +
                             std::to_string(uv_offset) +
                             ",\"byteLength\":24},"
                             "{\"buffer\":0,\"byteOffset\":" +
                             std::to_string(index_offset) +
                             ",\"byteLength\":6},"
                             "{\"buffer\":0,\"byteOffset\":" +
                             std::to_string(image_offset) +
                             ",\"byteLength\":4}],"
                             "\"accessors\":["
                             "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
                             "{\"bufferView\":1,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
                             "{\"bufferView\":2,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
                             "{\"bufferView\":3,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}],"
                             "\"images\":[{\"name\":\"BaseColor\",\"mimeType\":\"image/png\",\"bufferView\":4}],"
                             "\"textures\":[{\"source\":0}],"
                             "\"materials\":[{\"name\":\"TestMaterial\",\"alphaMode\":\"MASK\",\"alphaCutoff\":0.35,"
                             "\"doubleSided\":true,"
                             "\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.25,0.5,0.75,0.9],"
                             "\"metallicFactor\":0.2,\"roughnessFactor\":0.7,\"baseColorTexture\":{\"index\":0}},"
                             "\"normalTexture\":{\"index\":0,\"scale\":0.8},"
                             "\"occlusionTexture\":{\"index\":0,\"strength\":0.6},"
                             "\"emissiveTexture\":{\"index\":0},\"emissiveFactor\":[0.1,0.2,0.3]}],"
                             "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":"
                             "2},\"indices\":3,\"material\":0}]}]}";

    std::vector<std::byte> json_bytes(reinterpret_cast<const std::byte*>(json.data()),
                                      reinterpret_cast<const std::byte*>(json.data() + json.size()));
    pad4(json_bytes, std::byte{' '});

    std::vector<std::byte> glb;
    append_u32(glb, 0x46546C67);
    append_u32(glb, 2);
    append_u32(glb, static_cast<std::uint32_t>(12 + 8 + json_bytes.size() + 8 + bin.size()));
    append_u32(glb, static_cast<std::uint32_t>(json_bytes.size()));
    append_u32(glb, 0x4E4F534A);
    glb.insert(glb.end(), json_bytes.begin(), json_bytes.end());
    append_u32(glb, static_cast<std::uint32_t>(bin.size()));
    append_u32(glb, 0x004E4942);
    glb.insert(glb.end(), bin.begin(), bin.end());

    const auto path = std::filesystem::temp_directory_path() / "arc_triangle_mesh.glb";
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(glb.data()), static_cast<std::streamsize>(glb.size()));
    return path;
}

} // namespace

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

TEST_CASE("render graph orders passes by declared resources")
{
    arc::render::render_graph graph;
    const auto backbuffer = graph.add_resource({.name = "backbuffer",
                                                .kind = arc::render::render_resource_kind::color_texture,
                                                .format = arc::render::render_format::rgba8_unorm});
    graph.add_pass({.name = "clear",
                    .kind = arc::render::render_pass_kind::clear,
                    .writes = {{.handle = backbuffer,
                                .kind = arc::render::render_resource_kind::color_texture,
                                .usage = arc::render::render_resource_usage::color_attachment,
                                .write = true}}});
    graph.add_pass({.name = "present",
                    .kind = arc::render::render_pass_kind::present,
                    .reads = {{.handle = backbuffer,
                               .kind = arc::render::render_resource_kind::color_texture,
                               .usage = arc::render::render_resource_usage::sampled}}});

    const auto compiled = graph.compile().value();
    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.passes[0].name == "clear");
    REQUIRE(compiled.passes[1].name == "present");
}

TEST_CASE("render graph compiles typed resources and transitions")
{
    arc::render::render_graph graph;
    graph.add_resource({.name = "viewport",
                        .kind = arc::render::render_resource_kind::color_texture,
                        .extent = {.width = 1280, .height = 720},
                        .format = arc::render::render_format::rgba8_unorm,
                        .persistent = true});

    graph.add_pass({.name = "viewport clear",
                    .kind = arc::render::render_pass_kind::clear,
                    .writes = {{.resource = "viewport",
                                .kind = arc::render::render_resource_kind::color_texture,
                                .usage = arc::render::render_resource_usage::color_attachment,
                                .write = true,
                                .load_op = arc::render::render_load_op::clear}}});
    graph.add_pass({.name = "imgui sample",
                    .kind = arc::render::render_pass_kind::imgui,
                    .reads = {{.resource = "viewport",
                               .kind = arc::render::render_resource_kind::color_texture,
                               .usage = arc::render::render_resource_usage::sampled}},
                    .side_effect = true});

    const auto compiled = graph.compile().value();
    REQUIRE(compiled.resources.size() == 1);
    REQUIRE(compiled.resources[0].name == "viewport");
    REQUIRE(compiled.resources[0].format == arc::render::render_format::rgba8_unorm);
    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.passes[0].writes[0].usage == arc::render::render_resource_usage::color_attachment);
    REQUIRE(compiled.transitions.size() == 1);
    REQUIRE(compiled.transitions[0].resource == "viewport");
    REQUIRE(compiled.transitions[0].before == arc::render::render_resource_usage::color_attachment);
    REQUIRE(compiled.transitions[0].after == arc::render::render_resource_usage::sampled);
}

TEST_CASE("compiled render graph executes passes and barriers through a command encoder")
{
    arc::render::render_graph graph;
    const auto target = graph.add_resource({.name = "target",
                                            .kind = arc::render::render_resource_kind::color_texture,
                                            .format = arc::render::render_format::rgba8_unorm});
    std::uint32_t recorded{};
    graph.add_pass({.name = "produce",
                    .writes = {{.handle = target,
                                .kind = arc::render::render_resource_kind::color_texture,
                                .usage = arc::render::render_resource_usage::color_attachment,
                                .write = true}},
                    .record = count_recorded_pass,
                    .payload = arc::render::render_pass_payload::from(&recorded)});
    graph.add_pass({.name = "consume",
                    .reads = {{.handle = target,
                               .kind = arc::render::render_resource_kind::color_texture,
                               .usage = arc::render::render_resource_usage::sampled}},
                    .record = count_recorded_pass,
                    .payload = arc::render::render_pass_payload::from(&recorded),
                    .side_effect = true});

    recording_command_encoder encoder;
    arc::render::execute_render_graph(graph.compile().value(), encoder);

    REQUIRE(encoder.passes == std::vector<std::string>{"produce", "consume"});
    REQUIRE(encoder.barriers == std::vector<std::string>{"target"});
    REQUIRE(encoder.ended_passes == 2);
    REQUIRE(recorded == 2);
}

TEST_CASE("render graph schedules cross-queue waits and rotates persistent history")
{
    using namespace arc::render;
    render_graph graph;
    const auto seed = graph.add_resource({.name = "seed", .kind = render_resource_kind::buffer, .byte_size = 4096});
    const auto history =
        graph.add_resource({.name = "history",
                            .kind = render_resource_kind::color_texture,
                            .extent = {.width = 64, .height = 64},
                            .format = render_format::rgba16_float,
                            .persistent_key = "test.history",
                            .history_length = 2,
                            .history_reset = render_history_reset::camera_cut | render_history_reset::resize});
    graph.add_pass({.name = "graphics seed",
                    .queue = render_queue_type::graphics,
                    .writes = {{.handle = seed,
                                .kind = render_resource_kind::buffer,
                                .usage = render_resource_usage::storage_buffer,
                                .write = true}}});
    graph.add_pass({.name = "temporal compute",
                    .queue = render_queue_type::compute,
                    .reads = {{.handle = seed,
                               .kind = render_resource_kind::buffer,
                               .usage = render_resource_usage::storage_buffer},
                              {.handle = history,
                               .kind = render_resource_kind::color_texture,
                               .usage = render_resource_usage::sampled,
                               .history = render_history_access::previous}},
                    .writes = {{.handle = history,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::storage,
                                .write = true}}});

    const auto compiled = graph.compile().value();
    REQUIRE(compiled.submissions.size() == 2);
    REQUIRE(compiled.submissions[0].queue == render_queue_type::graphics);
    REQUIRE(compiled.submissions[1].queue == render_queue_type::compute);
    REQUIRE(compiled.submissions[1].waits.size() == 1);
    REQUIRE(compiled.submissions[1].waits[0].queue == render_queue_type::graphics);
    REQUIRE(compiled.history_rotations.size() == 1);
    REQUIRE(compiled.history_rotations[0].persistent_key == "test.history");
    REQUIRE(compiled.history_rotations[0].history_length == 2);
    REQUIRE(compiled.lifetimes[history.index].physical_resource != compiled.lifetimes[seed.index].physical_resource);

    recording_command_encoder encoder;
    execute_render_graph(compiled, encoder);
    REQUIRE(encoder.submissions ==
            std::vector<render_queue_type>{render_queue_type::graphics, render_queue_type::compute});
}

TEST_CASE("render graph specializes view extents queues and temporal resets")
{
    using namespace arc::render;
    render_graph graph;
    const auto history = graph.add_resource({.name = "history",
                                             .kind = render_resource_kind::color_texture,
                                             .extent_mode = render_extent_mode::relative_to_view,
                                             .width_scale = 0.5f,
                                             .height_scale = 0.5f,
                                             .format = render_format::rgba16_float,
                                             .mip_levels = 3,
                                             .persistent_key = "view.history",
                                             .history_length = 2,
                                             .history_reset = render_history_reset::camera_cut});
    graph.add_pass({.name = "resolve",
                    .queue = render_queue_type::compute,
                    .writes = {{.handle = history,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::storage,
                                .write = true}},
                    .side_effect = true});

    const auto compiled = graph
                              .compile({.view_id = 42,
                                        .output_extent = {1920, 1080, 1},
                                        .render_extent = {1280, 720, 1},
                                        .frame_index = 7,
                                        .world_epoch = 9,
                                        .temporal_reset = render_history_reset::camera_cut,
                                        .compute_queue_available = false})
                              .value();
    REQUIRE(compiled.view.view_id == 42);
    REQUIRE(compiled.resources[history.index].extent.width == 640);
    REQUIRE(compiled.resources[history.index].extent.height == 360);
    REQUIRE(compiled.passes[0].queue == render_queue_type::graphics);
    REQUIRE(compiled.submissions[0].queue == render_queue_type::graphics);
    REQUIRE(compiled.history_rotations[0].invalidated);
    REQUIRE(compiled.lifetimes[history.index].estimated_bytes ==
            (640ull * 360ull + 320ull * 180ull + 160ull * 90ull) * 8ull);
}

TEST_CASE("render graph culls dead work and aliases nonoverlapping transient resources")
{
    using namespace arc::render;
    render_graph graph;
    const auto dead = graph.add_resource({.name = "dead",
                                          .kind = render_resource_kind::color_texture,
                                          .extent = {64, 64, 1},
                                          .extent_mode = render_extent_mode::absolute,
                                          .format = render_format::rgba8_unorm});
    const auto first = graph.add_resource({.name = "first",
                                           .kind = render_resource_kind::color_texture,
                                           .extent = {64, 64, 1},
                                           .extent_mode = render_extent_mode::absolute,
                                           .format = render_format::rgba8_unorm});
    const auto gate = graph.add_resource({.name = "gate", .kind = render_resource_kind::buffer, .byte_size = 16});
    const auto second = graph.add_resource({.name = "second",
                                            .kind = render_resource_kind::color_texture,
                                            .extent = {64, 64, 1},
                                            .extent_mode = render_extent_mode::absolute,
                                            .format = render_format::rgba8_unorm});
    graph.add_pass({.name = "dead producer",
                    .writes = {{.handle = dead,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true}}});
    graph.add_pass({.name = "first producer",
                    .writes = {{.handle = first,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true}}});
    graph.add_pass({.name = "first consumer",
                    .reads = {{.handle = first,
                               .kind = render_resource_kind::color_texture,
                               .usage = render_resource_usage::sampled}},
                    .writes = {{.handle = gate,
                                .kind = render_resource_kind::buffer,
                                .usage = render_resource_usage::storage_buffer,
                                .write = true}}});
    graph.add_pass({.name = "second producer",
                    .reads = {{.handle = gate,
                               .kind = render_resource_kind::buffer,
                               .usage = render_resource_usage::storage_buffer}},
                    .writes = {{.handle = second,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true}},
                    .side_effect = true});

    const auto compiled = graph.compile().value();
    REQUIRE(compiled.culled_passes.size() == 1);
    REQUIRE(compiled.culled_passes[0].name == "dead producer");
    REQUIRE(compiled.lifetimes[dead.index].physical_resource == render_graph_resource_handle::invalid_index);
    REQUIRE(compiled.lifetimes[first.index].physical_resource == compiled.lifetimes[second.index].physical_resource);
    REQUIRE(compiled.lifetimes[second.index].aliased);
}

TEST_CASE("render graph rejects invalid resource declarations and internal reads")
{
    arc::render::render_graph undeclared;
    undeclared.add_pass(
        {.name = "bad read", .reads = {{.resource = "missing", .usage = arc::render::render_resource_usage::sampled}}});
    REQUIRE_FALSE(undeclared.compile());

    arc::render::render_graph read_before_write;
    const auto transient = read_before_write.add_resource({.name = "transient",
                                                           .kind = arc::render::render_resource_kind::color_texture,
                                                           .format = arc::render::render_format::rgba8_unorm});
    read_before_write.add_pass({.name = "bad read",
                                .reads = {{.handle = transient,
                                           .kind = arc::render::render_resource_kind::color_texture,
                                           .usage = arc::render::render_resource_usage::sampled}}});
    REQUIRE_FALSE(read_before_write.compile());

    arc::render::render_graph incompatible;
    const auto depth = incompatible.add_resource({.name = "depth",
                                                  .kind = arc::render::render_resource_kind::depth_texture,
                                                  .format = arc::render::render_format::d32_float});
    incompatible.add_pass({.name = "bad attachment",
                           .writes = {{.handle = depth,
                                       .kind = arc::render::render_resource_kind::depth_texture,
                                       .usage = arc::render::render_resource_usage::color_attachment,
                                       .write = true}}});
    REQUIRE_FALSE(incompatible.compile());
}

TEST_CASE("clear present graph declares the bring-up passes")
{
    const auto graph = arc::render::make_clear_present_graph("viewport");
    const auto compiled = graph.compile().value();

    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.resources.size() == 1);
    REQUIRE(compiled.passes[0].kind == arc::render::render_pass_kind::clear);
    REQUIRE(compiled.passes[1].kind == arc::render::render_pass_kind::present);
    REQUIRE_FALSE(compiled.transitions.empty());
}

TEST_CASE("scene draw graph selects only implemented deferred passes")
{
    const auto graph = arc::render::make_scene_draw_graph("viewport", arc::render::render_path::deferred);
    const auto compiled = graph.compile().value();

    REQUIRE(compiled.passes.size() >= 19);
    const auto pass_index = [&](std::string_view name)
    {
        for (std::size_t index = 0; index < compiled.passes.size(); ++index)
        {
            if (compiled.passes[index].name == name) return index;
        }
        return compiled.passes.size();
    };

    const std::size_t static_shadow_index = pass_index("directional static shadows");
    const std::size_t dynamic_shadow_index = pass_index("directional dynamic shadows");
    const std::size_t sky_index = pass_index("sky composite");
    const std::size_t depth_index = pass_index("depth prepass");
    const std::size_t gbuffer_index = pass_index("gbuffer pass");
    const std::size_t deferred_index = pass_index("deferred lighting");
    const std::size_t transparent_index = pass_index("forward transparent");
    for (std::size_t index = 0; index < compiled.passes.size(); ++index)
        REQUIRE_FALSE(compiled.passes[index].name.empty());

    REQUIRE(static_shadow_index < gbuffer_index);
    REQUIRE(dynamic_shadow_index < gbuffer_index);
    REQUIRE(depth_index < gbuffer_index);
    REQUIRE(gbuffer_index < deferred_index);
    REQUIRE(sky_index < deferred_index);
    REQUIRE(deferred_index < transparent_index);
    REQUIRE(compiled.passes[compiled.passes.size() - 5].builtin == arc::render::builtin_render_pass::debug_overlay);
    REQUIRE(compiled.passes[compiled.passes.size() - 4].builtin ==
            arc::render::builtin_render_pass::luminance_histogram);
    REQUIRE(compiled.passes[compiled.passes.size() - 3].builtin == arc::render::builtin_render_pass::exposure_resolve);
    REQUIRE(compiled.passes[compiled.passes.size() - 2].builtin == arc::render::builtin_render_pass::output_transform);
    REQUIRE(compiled.passes.back().builtin == arc::render::builtin_render_pass::editor_overlay);
    REQUIRE(compiled.resources.size() >= 18);
    REQUIRE(std::any_of(
        compiled.resources.begin(), compiled.resources.end(), [](const auto& resource)
        { return resource.name == "gbuffer_albedo" && resource.format == arc::render::render_format::rgba8_srgb; }));
    REQUIRE(compiled.lifetimes.size() == compiled.resources.size());
    REQUIRE_FALSE(compiled.transitions.empty());
}

TEST_CASE("scene draw graph provides a compact forward plus fallback")
{
    const auto compiled =
        arc::render::make_scene_draw_graph("viewport", arc::render::render_path::forward_plus, false).compile().value();

    REQUIRE(compiled.passes.size() >= 11);
    REQUIRE(compiled.resources.size() >= 8);
    REQUIRE(std::any_of(compiled.passes.begin(), compiled.passes.end(),
                        [](const auto& pass) { return pass.name == "forward opaque"; }));
    for (const auto& pass : compiled.passes)
        REQUIRE(pass.name != "gbuffer pass");
}

TEST_CASE("GPU-driven scene graph declares visibility indirect and temporal history work")
{
    using namespace arc::render;
    resolved_render_config config;
    config.quality = render_quality_tier::ultra;
    config.path = render_path::deferred;
    config.render_scale = 0.75f;
    config.features.gpu_driven_rendering = true;
    config.features.hzb_occlusion = true;
    config.features.temporal_antialiasing = true;
    config.features.temporal_upscaling = true;
    config.features.async_compute = true;
    config.features.virtual_geometry = true;
    config.features.virtual_geometry_path = virtual_geometry_raster_path::compute;
    config.features.submission = gpu_submission_path::indirect_count;

    const auto compiled = make_scene_draw_graph("viewport", config, true).compile().value();
    const auto contains = [&](builtin_render_pass expected)
    {
        return std::any_of(compiled.passes.begin(), compiled.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };
    REQUIRE(contains(builtin_render_pass::gpu_scene_upload));
    REQUIRE(contains(builtin_render_pass::gpu_frustum_distance_cull));
    REQUIRE(contains(builtin_render_pass::gpu_hzb_occlusion_cull));
    REQUIRE(contains(builtin_render_pass::gpu_lod_selection));
    REQUIRE(contains(builtin_render_pass::gpu_draw_bin_scatter));
    REQUIRE(contains(builtin_render_pass::gpu_indirect_command_generation));
    REQUIRE(contains(builtin_render_pass::gpu_visibility_overflow));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_hierarchy_traversal));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_page_requests));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_cluster_binning));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_software_depth));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_visibility_resolve));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_material_resolve));
    REQUIRE(contains(builtin_render_pass::virtual_geometry_shadow_traversal));
    REQUIRE_FALSE(contains(builtin_render_pass::virtual_geometry_mesh_shader_visibility));
    REQUIRE(contains(builtin_render_pass::depth_pyramid));
    REQUIRE(contains(builtin_render_pass::reactive_mask));
    REQUIRE(contains(builtin_render_pass::disocclusion_mask));
    REQUIRE(contains(builtin_render_pass::temporal_upscale));
    REQUIRE(contains(builtin_render_pass::spatial_sharpen));
    REQUIRE(std::any_of(compiled.submissions.begin(), compiled.submissions.end(),
                        [](const auto& submission) { return submission.queue == render_queue_type::compute; }));
    REQUIRE(std::any_of(compiled.history_rotations.begin(), compiled.history_rotations.end(),
                        [](const auto& history) { return history.persistent_key == "view.temporal_color"; }));
    REQUIRE(std::any_of(compiled.history_rotations.begin(), compiled.history_rotations.end(),
                        [](const auto& history) { return history.persistent_key == "view.depth_hzb"; }));
}

TEST_CASE("Ultra virtual shadow graph declares page feedback cache and lighting dependencies")
{
    using namespace arc::render;
    resolved_render_config config;
    config.quality = render_quality_tier::ultra;
    config.path = render_path::deferred;
    config.features.virtual_shadow_maps = true;
    config.features.screen_space_contact_shadows = true;
    config.screen_space_shadows = true;
    config.screen_space_shadow_scale = 1.0f;

    const auto compiled = make_scene_draw_graph("vsm", config, true).compile().value();
    const auto pass_index = [&](builtin_render_pass expected)
    {
        for (std::size_t index = 0; index < compiled.passes.size(); ++index)
            if (compiled.passes[index].builtin == expected) return index;
        return compiled.passes.size();
    };
    const auto marking = pass_index(builtin_render_pass::virtual_shadow_page_marking);
    const auto allocation = pass_index(builtin_render_pass::virtual_shadow_page_allocation);
    const auto culling = pass_index(builtin_render_pass::virtual_shadow_caster_culling);
    const auto static_render = pass_index(builtin_render_pass::virtual_shadow_static_render);
    const auto dynamic_render = pass_index(builtin_render_pass::virtual_shadow_dynamic_render);
    const auto publication = pass_index(builtin_render_pass::virtual_shadow_page_table_publication);
    const auto feedback = pass_index(builtin_render_pass::virtual_shadow_feedback_readback);
    const auto lighting = pass_index(builtin_render_pass::deferred_lighting);
    REQUIRE(marking < allocation);
    REQUIRE(allocation < culling);
    REQUIRE(culling < static_render);
    REQUIRE(culling < dynamic_render);
    REQUIRE(static_render < publication);
    REQUIRE(dynamic_render < publication);
    REQUIRE(publication < lighting);
    REQUIRE(feedback < compiled.passes.size());
    REQUIRE(pass_index(builtin_render_pass::screen_space_shadow) < lighting);
    REQUIRE(pass_index(builtin_render_pass::screen_space_shadow_filter) < lighting);

    const auto resource = [&](std::string_view name) -> const render_graph_resource*
    {
        const auto found = std::find_if(compiled.resources.begin(), compiled.resources.end(),
                                        [name](const auto& value) { return value.name == name; });
        return found == compiled.resources.end() ? nullptr : &*found;
    };
    const auto* static_pages = resource("virtual_shadow_static_pages");
    const auto* dynamic_pages = resource("virtual_shadow_dynamic_pages");
    const auto* page_table = resource("virtual_shadow_page_table");
    const auto* readback = resource("virtual_shadow_feedback_readback");
    REQUIRE(static_pages != nullptr);
    REQUIRE(dynamic_pages != nullptr);
    REQUIRE(page_table != nullptr);
    REQUIRE(readback != nullptr);
    REQUIRE(static_pages->format == render_format::d16_unorm);
    REQUIRE(static_pages->persistent);
    REQUIRE(dynamic_pages->persistent);
    REQUIRE(page_table->lifetime == render_resource_lifetime_class::per_world);
    REQUIRE(readback->memory == render_memory_class::readback);
    REQUIRE(readback->exported);
}

TEST_CASE("environment lighting graph schedules scalable IBL generation")
{
    arc::render::resolved_render_config config;
    config.quality = arc::render::render_quality_tier::medium;
    config.path = arc::render::render_path::deferred;
    arc::render::world_environment_data environment;
    environment.enabled = true;
    environment.sky_visible = true;
    environment.affect_lighting = true;
    environment.source = arc::render::sky_source_mode::hdri;
    environment.lighting.enabled = true;
    environment.lighting.source = arc::render::environment_lighting_source_mode::follow_sky;

    const auto compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile().value();
    const auto contains = [&](arc::render::builtin_render_pass expected)
    {
        return std::any_of(compiled.passes.begin(), compiled.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };
    REQUIRE(contains(arc::render::builtin_render_pass::environment_equirect_to_cube));
    REQUIRE(contains(arc::render::builtin_render_pass::environment_irradiance));
    REQUIRE(contains(arc::render::builtin_render_pass::environment_specular_prefilter));
    REQUIRE(contains(arc::render::builtin_render_pass::brdf_integration));
    REQUIRE(contains(arc::render::builtin_render_pass::luminance_histogram));
    REQUIRE(contains(arc::render::builtin_render_pass::exposure_resolve));
    REQUIRE(contains(arc::render::builtin_render_pass::output_transform));
    REQUIRE(std::any_of(compiled.resources.begin(), compiled.resources.end(),
                        [](const auto& resource)
                        {
                            return resource.name == "environment_specular" && resource.extent.width == 256 &&
                                   resource.array_layers == 6 && resource.mip_levels == 9;
                        }));
}

TEST_CASE("world environment graph selects scalable atmosphere and cloud passes")
{
    arc::render::resolved_render_config standard;
    standard.quality = arc::render::render_quality_tier::medium;
    standard.path = arc::render::render_path::deferred;
    arc::render::world_environment_data environment;
    environment.enabled = true;
    environment.sky_visible = true;
    environment.source = arc::render::sky_source_mode::physical_atmosphere;
    environment.atmosphere.enabled = true;
    environment.clouds.enabled = true;
    environment.clouds.cast_shadows = true;

    const auto compiled = arc::render::make_scene_draw_graph("viewport", standard, true, environment).compile().value();
    const auto contains = [&](arc::render::builtin_render_pass expected)
    {
        return std::any_of(compiled.passes.begin(), compiled.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_transmittance));
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_multi_scattering));
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_sky_view));
    REQUIRE(contains(arc::render::builtin_render_pass::cloud_shadow));
    REQUIRE(contains(arc::render::builtin_render_pass::sky_composite));
    REQUIRE(contains(arc::render::builtin_render_pass::debug_overlay));
    REQUIRE(contains(arc::render::builtin_render_pass::editor_overlay));

    standard.quality = arc::render::render_quality_tier::low;
    standard.path = arc::render::render_path::forward_plus;
    const auto low = arc::render::make_scene_draw_graph("viewport", standard, true, environment).compile().value();
    REQUIRE(std::none_of(low.passes.begin(), low.passes.end(),
                         [](const auto& pass)
                         {
                             return pass.builtin == arc::render::builtin_render_pass::atmosphere_transmittance ||
                                    pass.builtin == arc::render::builtin_render_pass::cloud_shadow;
                         }));
    REQUIRE(std::any_of(low.passes.begin(), low.passes.end(), [](const auto& pass)
                        { return pass.builtin == arc::render::builtin_render_pass::sky_composite; }));
    REQUIRE(std::any_of(low.passes.begin(), low.passes.end(), [](const auto& pass)
                        { return pass.builtin == arc::render::builtin_render_pass::debug_overlay; }));
    REQUIRE(std::any_of(low.passes.begin(), low.passes.end(), [](const auto& pass)
                        { return pass.builtin == arc::render::builtin_render_pass::editor_overlay; }));
}

TEST_CASE("world environment graph selects off solid and HDRI sky paths without atmosphere LUTs")
{
    arc::render::resolved_render_config config;
    config.quality = arc::render::render_quality_tier::medium;
    config.path = arc::render::render_path::deferred;
    const auto contains = [](const auto& graph, arc::render::builtin_render_pass expected)
    {
        return std::any_of(graph.passes.begin(), graph.passes.end(),
                           [expected](const auto& pass) { return pass.builtin == expected; });
    };

    arc::render::world_environment_data environment;
    environment.enabled = false;
    environment.sky_visible = false;
    environment.clouds.enabled = false;
    auto compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile().value();
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::sky_composite));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::atmosphere_transmittance));

    environment.enabled = true;
    environment.sky_visible = true;
    environment.source = arc::render::sky_source_mode::solid_color;
    compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile().value();
    REQUIRE(contains(compiled, arc::render::builtin_render_pass::sky_composite));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::atmosphere_transmittance));

    environment.source = arc::render::sky_source_mode::hdri;
    compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile().value();
    REQUIRE(contains(compiled, arc::render::builtin_render_pass::sky_composite));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::atmosphere_transmittance));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::environment_prefilter));
}

TEST_CASE("directional shadow cascade splits are deterministic and ordered")
{
    const auto splits = arc::render::cascade_splits(0.1f, 100.0f, 0.65f);

    REQUIRE(splits[0] > 0.1f);
    REQUIRE(splits[0] < splits[1]);
    REQUIRE(splits[1] < splits[2]);
    REQUIRE(splits[2] < splits[3]);
    REQUIRE(splits[3] == Catch::Approx(100.0f));
}

TEST_CASE("stable directional shadow fitting produces ordered blended cascades")
{
    arc::render::directional_shadow_camera camera{};
    camera.near_plane = 0.1f;
    camera.far_plane = 1000.0f;
    camera.inverse_view_projection = arc::math::identity<float, 4>();
    arc::render::directional_shadow_settings settings{};
    settings.cascade_count = 4;
    settings.maximum_distance = 200.0f;
    settings.blend_fraction = 0.1f;

    const auto first = arc::render::fit_directional_shadow_cascades(camera, {0.35f, -0.85f, -0.4f}, settings, 2048);
    const auto second = arc::render::fit_directional_shadow_cascades(camera, {0.35f, -0.85f, -0.4f}, settings, 2048);

    REQUIRE(first.cascade_count == 4);
    REQUIRE(first.cascades[3].split_depth == Catch::Approx(200.0f));
    for (std::uint32_t index = 0; index < first.cascade_count; ++index)
    {
        const auto& cascade = first.cascades[index];
        REQUIRE(cascade.radius > 0.0f);
        REQUIRE(cascade.texel_world_size > 0.0f);
        REQUIRE(cascade.blend_start_depth < cascade.split_depth);
        REQUIRE(std::memcmp(cascade.light_view_projection.data(), second.cascades[index].light_view_projection.data(),
                            sizeof(float) * 16u) == 0);
        if (index > 0) REQUIRE(cascade.split_depth > first.cascades[index - 1].split_depth);

        const auto light_direction = arc::math::normalize(arc::math::vector3f{0.35f, -0.85f, -0.4f});
        const auto light_right =
            arc::math::normalize(arc::math::cross(light_direction, arc::math::vector3f{0.0f, 1.0f, 0.0f}));
        const auto light_up = arc::math::cross(light_right, light_direction);
        const float snapped_x = arc::math::dot(light_right, cascade.center) / cascade.texel_world_size;
        const float snapped_y = arc::math::dot(light_up, cascade.center) / cascade.texel_world_size;
        REQUIRE(snapped_x == Catch::Approx(std::round(snapped_x)).margin(0.0001f));
        REQUIRE(snapped_y == Catch::Approx(std::round(snapped_y)).margin(0.0001f));
    }
}

TEST_CASE("shadow atlas allocates point faces atomically and invalidates released handles")
{
    arc::render::shadow_atlas_allocator atlas(2048, 128, 2);
    const auto point = atlas.allocate({.kind = arc::render::shadow_light_kind::point,
                                       .light_key = 42,
                                       .requested_resolution = 512,
                                       .minimum_resolution = 256,
                                       .priority = 200,
                                       .frame_index = 1});
    REQUIRE(point);
    REQUIRE(point->face_count == arc::render::point_shadow_face_count);
    REQUIRE(point->resolved_resolution == 512);
    for (const auto& face : point->faces)
        REQUIRE(face.valid());

    const auto handle = point->handle;
    REQUIRE(atlas.find(handle) != nullptr);
    REQUIRE(atlas.release(handle));
    REQUIRE(atlas.find(handle) == nullptr);
    REQUIRE_FALSE(atlas.release(handle));
}

TEST_CASE("shadow atlas reduces resolution and evicts lower priority allocations")
{
    arc::render::shadow_atlas_allocator atlas(512, 128, 2);
    for (std::uint64_t light = 1; light <= 4; ++light)
    {
        REQUIRE(atlas.allocate({.kind = arc::render::shadow_light_kind::spot,
                                .light_key = light,
                                .requested_resolution = 240,
                                .minimum_resolution = 120,
                                .priority = 10,
                                .frame_index = light}));
    }
    const auto important = atlas.allocate({.kind = arc::render::shadow_light_kind::spot,
                                           .light_key = 99,
                                           .requested_resolution = 240,
                                           .minimum_resolution = 120,
                                           .priority = 250,
                                           .frame_index = 10});
    REQUIRE(important);
    REQUIRE(atlas.statistics().eviction_count >= 1);
}

TEST_CASE("virtual shadow address spaces normalize light topology and invalidate generations")
{
    arc::render::virtual_shadow_cache cache(16ull * 1024ull * 1024ull);
    const auto directional = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::directional,
                                                         .light_key = 11,
                                                         .level_count = 2,
                                                         .face_count = 3});
    REQUIRE(directional);
    const auto* directional_descriptor = cache.address_space(*directional);
    REQUIRE(directional_descriptor != nullptr);
    REQUIRE(directional_descriptor->level_count == arc::render::virtual_shadow_directional_clip_levels);
    REQUIRE(directional_descriptor->face_count == 1);

    const auto point = cache.create_address_space(
        {.light_kind = arc::render::shadow_light_kind::point, .light_key = 12, .level_count = 4});
    REQUIRE(point);
    REQUIRE(cache.address_space(*point)->face_count == arc::render::point_shadow_face_count);

    const auto stale = *directional;
    REQUIRE(cache.destroy_address_space(*directional));
    REQUIRE(cache.address_space(stale) == nullptr);
    const auto replacement =
        cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot, .light_key = 13});
    REQUIRE(replacement);
    REQUIRE(replacement->index == stale.index);
    REQUIRE(replacement->generation != stale.generation);
}

TEST_CASE("virtual shadow page requests are deterministic and retain coarse fallback")
{
    constexpr std::uint64_t one_d16_page_pair =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_page_texels +
                                   arc::render::virtual_shadow_page_guard_texels * 2u) *
        (arc::render::virtual_shadow_page_texels + arc::render::virtual_shadow_page_guard_texels * 2u) * 4u;
    arc::render::virtual_shadow_cache cache(one_d16_page_pair * 2u);
    const auto light = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot,
                                                   .light_key = 42,
                                                   .virtual_resolution = 2048,
                                                   .level_count = 5});
    REQUIRE(light);
    const arc::render::virtual_shadow_page_key root{.address_space = *light,
                                                    .coordinate = {.x = 0, .y = 0, .level = 4, .face = 0}};
    const arc::render::virtual_shadow_page_key child{.address_space = *light,
                                                     .coordinate = {.x = 2, .y = 2, .level = 2, .face = 0}};
    const std::array requests{
        arc::render::virtual_shadow_page_request{
            .key = child, .frame_index = 1, .content_revision = 7, .projected_coverage = 10.0f, .light_priority = 200},
        arc::render::virtual_shadow_page_request{.key = root,
                                                 .frame_index = 1,
                                                 .content_revision = 7,
                                                 .projected_coverage = 1.0f,
                                                 .light_priority = 200,
                                                 .coarse_page = true}};
    const auto first = cache.resolve_requests(requests, 1);
    REQUIRE(first.render_pages.size() == 2);
    REQUIRE(first.render_pages.front().key == root);
    REQUIRE(cache.publish(root, 7));
    REQUIRE(cache.publish(child, 7));
    REQUIRE(cache.find_resident_or_ancestor(
                {.address_space = *light, .coordinate = {.x = 5, .y = 5, .level = 1, .face = 0}}) != nullptr);

    const auto second = cache.resolve_requests(requests, 2);
    REQUIRE(second.cache_hits == 2);
    REQUIRE(second.render_pages.empty());
    REQUIRE(cache.invalidate(*light, arc::render::virtual_shadow_invalidation_reason::material_alpha) == 2);
    REQUIRE(cache.statistics().dirty_pages == 2);
}

TEST_CASE("virtual shadow cache protects recent and pinned pages under pressure")
{
    constexpr std::uint64_t one_d16_page_pair =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_page_texels +
                                   arc::render::virtual_shadow_page_guard_texels * 2u) *
        (arc::render::virtual_shadow_page_texels + arc::render::virtual_shadow_page_guard_texels * 2u) * 4u;
    arc::render::virtual_shadow_cache cache(one_d16_page_pair * 2u);
    const auto light = cache.create_address_space(
        {.light_kind = arc::render::shadow_light_kind::spot, .light_key = 71, .level_count = 5});
    REQUIRE(light);
    const auto request = [&](std::uint16_t x, std::uint64_t frame, bool coarse)
    {
        const arc::render::virtual_shadow_page_request value{
            .key = {.address_space = *light, .coordinate = {.x = x, .y = 0, .level = 0, .face = 0}},
            .frame_index = frame,
            .content_revision = 1,
            .projected_coverage = 1.0f,
            .light_priority = 1,
            .coarse_page = coarse};
        return cache.resolve_requests(std::span{&value, 1}, frame);
    };
    REQUIRE(request(0, 1, true).render_pages.size() == 1);
    REQUIRE(request(1, 1, false).render_pages.size() == 1);
    REQUIRE(request(2, 2, false).failed_requests == 1);
    REQUIRE(request(2, 64, false).render_pages.size() == 1);
    REQUIRE(cache.statistics().eviction_count == 1);
    REQUIRE(cache.statistics().pinned_pages == 1);
}

TEST_CASE("virtual shadow clipmap origins snap at physical page granularity")
{
    const auto snapped = arc::render::snap_virtual_shadow_clipmap_origin({13.2f, -4.1f}, 0.125f);
    REQUIRE(snapped[0] == Catch::Approx(0.0f));
    REQUIRE(snapped[1] == Catch::Approx(-16.0f));
    REQUIRE(arc::render::virtual_shadow_pages_per_axis(16384, 0) == 128);
    REQUIRE(arc::render::virtual_shadow_pages_per_axis(16384, 4) == 8);
    REQUIRE(arc::render::virtual_shadow_parent_page({6, 10, 1, 2}) ==
            arc::render::virtual_shadow_page_coordinate{3, 5, 2, 2});
}

TEST_CASE("virtual shadow requests always retain a conventional executable fallback")
{
    using arc::render::resolve_shadow_map_method;
    using arc::render::shadow_map_method;

    REQUIRE(resolve_shadow_map_method(shadow_map_method::auto_select, true) == shadow_map_method::virtualized);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::virtualized, true) == shadow_map_method::virtualized);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::conventional, true) == shadow_map_method::conventional);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::auto_select, false) == shadow_map_method::conventional);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::virtualized, false) == shadow_map_method::conventional);
}

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

TEST_CASE("GPU Scene represents skinned meshes and terrain without CPU patch expansion")
{
    using namespace arc::render;
    render_world_packet packet;
    packet.gpu_scene_world_id = 9;
    packet.world_epoch = 1;
    packet.items.push_back({.mesh = {.index = 2, .generation = 3},
                            .object_id = {.index = 10, .generation = 1},
                            .skin_matrices = {.index = 5, .generation = 1},
                            .skin_joint_count = 72});
    packet.terrains.push_back({.terrain = {.index = 8, .generation = 2}, .object_id = {.index = 11, .generation = 1}});

    gpu_scene scene;
    const auto update = scene.synchronize(packet, 1);
    REQUIRE(update.active_instance_count == 2);
    REQUIRE(packet.items[0].gpu_scene_instance.valid());
    REQUIRE(packet.terrains[0].gpu_scene_instance.valid());
    const auto* skinned = scene.find(packet.items[0].gpu_scene_instance);
    const auto* terrain = scene.find(packet.terrains[0].gpu_scene_instance);
    REQUIRE(skinned != nullptr);
    REQUIRE(skinned->geometry_kind == gpu_scene_geometry_kind::skinned_mesh);
    REQUIRE(skinned->skin_palette == buffer_handle{.index = 5, .generation = 1});
    REQUIRE(skinned->skin_joint_count == 72);
    REQUIRE(terrain != nullptr);
    REQUIRE(terrain->geometry_kind == gpu_scene_geometry_kind::terrain);
    REQUIRE(terrain->terrain == terrain_handle{.index = 8, .generation = 2});
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

TEST_CASE("renderer submits committed packets to attached backend")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    arc::render::render_event_buffer buffer;
    arc::render::render_event_writer writer(buffer);
    writer.debug_marker("frame");
    renderer.frame_queue().submit(std::move(buffer));

    const auto result = renderer.render_frame(42, arc::render::make_clear_present_graph("viewport"));

    REQUIRE(result.has_value());
    REQUIRE(backend_ptr->last_frame == 42);
    REQUIRE(backend_ptr->last_event_count == 1);
    REQUIRE(backend_ptr->last_pass_count == 2);
}

TEST_CASE("renderer resolves low quality policy and optional feature overrides")
{
    arc::render::render_capabilities capabilities{};
    capabilities.dedicated_video_memory = 1024ull * 1024ull * 1024ull;
    capabilities.dynamic_rendering = true;
    capabilities.synchronization2 = true;
    capabilities.timeline_semaphores = true;
    capabilities.descriptor_indexing = true;
    capabilities.draw_indirect = true;
    capabilities.draw_indirect_count = true;
    capabilities.sampler_anisotropy = true;
    capabilities.texture_compression_bc = true;

    arc::render::renderer_config config{};
    config.force_disable_optional_features = true;
    const auto resolved = arc::render::resolve_render_config(config, capabilities);

    REQUIRE(resolved.quality == arc::render::render_quality_tier::low);
    REQUIRE(resolved.path == arc::render::render_path::forward_plus);
    REQUIRE(resolved.minimum_render_scale == Catch::Approx(0.5f));
    REQUIRE(resolved.max_point_lights == 32);
    REQUIRE(resolved.directional_shadow_cascades == 2);
    REQUIRE(resolved.directional_shadow_resolution == 1024);
    REQUIRE(resolved.features.draw_indirect);
    REQUIRE(resolved.features.texture_compression_bc);
    REQUIRE_FALSE(resolved.features.dynamic_rendering);
    REQUIRE_FALSE(resolved.features.timeline_semaphores);
    REQUIRE_FALSE(resolved.features.descriptor_indexing);
    REQUIRE_FALSE(resolved.fallback_reasons.empty());
}

TEST_CASE("render quality profiles expose immutable implemented tier policy")
{
    using namespace arc::render;

    STATIC_REQUIRE(default_target_frame_time_ms > 16.0f);
    STATIC_REQUIRE(default_target_frame_time_ms < 17.0f);
    STATIC_REQUIRE(dynamic_resolution_scale_step == 1.0f / 16.0f);
    STATIC_REQUIRE(low_render_quality_profile.default_path == render_path::forward_plus);
    STATIC_REQUIRE(standard_render_quality_profile.default_path == render_path::deferred);

    const auto& low = quality_profile(render_quality_tier::low);
    REQUIRE(low.minimum_render_scale == Catch::Approx(0.5f));
    REQUIRE(low.max_point_lights == 32);
    REQUIRE(low.directional_shadow_cascades == 2);

    const auto& high = quality_profile(render_quality_tier::high);
    REQUIRE(&high == &high_render_quality_profile);
    REQUIRE(high.minimum_render_scale == Catch::Approx(0.67f));
    REQUIRE(high.directional_shadow_resolution == 4096);
    REQUIRE(high.local_shadow_atlas_resolution == 8192);

    const auto& ultra = quality_profile(render_quality_tier::ultra);
    REQUIRE(&ultra == &ultra_render_quality_profile);
    REQUIRE(ultra.max_point_lights == 128);
    REQUIRE(ultra.gi_trace_budget == 4);
    REQUIRE(ultra.virtual_shadow_budget_bytes == 512ull * 1024ull * 1024ull);
    REQUIRE(ultra.virtual_shadow_page_render_budget == 2048);
    REQUIRE(ultra.target_frame_time_ms == Catch::Approx(1000.0f / 30.0f));
}

TEST_CASE("renderer resolves GPU-driven temporal features and their forced fallbacks")
{
    using namespace arc::render;
    render_capabilities capabilities{};
    capabilities.dedicated_video_memory = 16ull * 1024ull * 1024ull * 1024ull;
    capabilities.compute_queue = true;
    capabilities.dedicated_compute_queue = true;
    capabilities.compute_shaders = true;
    capabilities.storage_buffers = true;
    capabilities.storage_images = true;
    capabilities.shader_draw_parameters = true;
    capabilities.gpu_scene_indirect = true;
    capabilities.gpu_scene_indirect_count = true;
    capabilities.hzb_occlusion = true;
    capabilities.temporal_resolve = true;
    capabilities.temporal_upscale = true;
    capabilities.fxaa = true;
    capabilities.descriptor_indexing = true;
    capabilities.virtual_geometry_compute = true;
    capabilities.virtual_geometry_streaming = true;
    capabilities.virtual_shadow_allocation = true;
    capabilities.virtual_shadow_feedback = true;
    capabilities.virtual_shadow_rendering = true;
    capabilities.virtual_shadow_sampling = true;
    capabilities.virtual_shadow_virtual_geometry = true;
    capabilities.screen_space_contact_shadows = true;
    capabilities.screen_space_indirect_lighting = true;
    capabilities.surface_cache = true;
    capabilities.radiance_cache = true;
    capabilities.software_ray_tracing = true;
    capabilities.hardware_ray_query = true;
    capabilities.draw_indirect = true;
    capabilities.draw_indirect_count = true;
    capabilities.sparse_resources = true;
    capabilities.ray_tracing = true;

    renderer_config config{};
    config.quality = render_quality_tier::ultra;
    auto resolved = resolve_render_config(config, capabilities);
    REQUIRE(resolved.quality == render_quality_tier::ultra);
    REQUIRE(resolved.features.gpu_driven_rendering);
    REQUIRE(resolved.features.gpu_binding_model == gpu_resource_binding_model::classic);
    REQUIRE_FALSE(resolved.features.gpu_visibility_compaction);
    REQUIRE(resolved.features.hzb_occlusion);
    REQUIRE(resolved.features.temporal_antialiasing);
    REQUIRE_FALSE(resolved.features.temporal_upscaling);
    REQUIRE(resolved.anti_aliasing == anti_aliasing_method::taa);
    REQUIRE(resolved.features.async_compute);
    REQUIRE_FALSE(resolved.features.virtual_geometry);
    REQUIRE(resolved.features.virtual_geometry_path == virtual_geometry_raster_path::unavailable);
    REQUIRE(resolved.features.software_ray_tracing);
    REQUIRE(resolved.features.virtual_shadow_maps);
    REQUIRE_FALSE(resolved.features.virtual_shadow_virtual_geometry);
    REQUIRE(resolved.features.screen_space_contact_shadows);
    REQUIRE(resolved.virtual_shadow_budget_bytes == 512ull * 1024ull * 1024ull);
    REQUIRE(resolved.features.hardware_ray_tracing);
    REQUIRE(resolved.features.screen_space_gi);
    REQUIRE(resolved.features.screen_space_reflections);
    REQUIRE(resolved.features.surface_cache);
    REQUIRE(resolved.features.radiance_cache);
    REQUIRE(resolved.features.software_gi);
    REQUIRE(resolved.features.software_reflections);
    REQUIRE(resolved.features.hardware_gi);
    REQUIRE(resolved.features.hardware_reflections);
    REQUIRE(resolved.indirect_lighting_path == lighting_trace_path::hybrid_hardware);
    REQUIRE(resolved.lighting_scene_gpu_budget_bytes == 768ull * 1024ull * 1024ull);
    REQUIRE(resolved.features.submission == gpu_submission_path::indirect_count);

    config.quality = render_quality_tier::high;
    const auto high = resolve_render_config(config, capabilities);
    REQUIRE_FALSE(high.features.virtual_geometry);
    REQUIRE(high.features.virtual_geometry_path == virtual_geometry_raster_path::unavailable);
    REQUIRE_FALSE(high.features.virtual_shadow_maps);
    config.quality = render_quality_tier::ultra;

    capabilities.gpu_visibility_compaction = true;
    capabilities.bindless_sampled_images = true;
    capabilities.bindless_samplers = true;
    capabilities.bindless_material_tables = true;
    capabilities.bindless_geometry_tables = true;
    capabilities.gpu_transparent_sorting = true;
    capabilities.gpu_skinning = true;
    capabilities.gpu_terrain_traversal = true;
    resolved = resolve_render_config(config, capabilities);
    REQUIRE(resolved.features.gpu_binding_model == gpu_resource_binding_model::bindless);
    REQUIRE(resolved.features.gpu_visibility_compaction);
    REQUIRE(resolved.features.gpu_transparent_sorting);
    REQUIRE(resolved.features.gpu_skinning);
    REQUIRE(resolved.features.gpu_terrain_traversal);
    REQUIRE(resolved.features.virtual_geometry);
    REQUIRE(resolved.features.virtual_geometry_path == virtual_geometry_raster_path::compute);
    REQUIRE(resolved.features.virtual_shadow_virtual_geometry);

    config.force_disable_gpu_driven = true;
    config.force_disable_temporal = true;
    config.force_disable_async_compute = true;
    config.force_disable_dynamic_gi = true;
    config.force_disable_hardware_ray_tracing = true;
    config.force_cpu_submission = true;
    resolved = resolve_render_config(config, capabilities);
    REQUIRE_FALSE(resolved.features.gpu_driven_rendering);
    REQUIRE_FALSE(resolved.features.virtual_geometry);
    REQUIRE_FALSE(resolved.features.virtual_shadow_maps);
    REQUIRE_FALSE(resolved.features.temporal_antialiasing);
    REQUIRE_FALSE(resolved.features.async_compute);
    REQUIRE_FALSE(resolved.features.screen_space_gi);
    REQUIRE_FALSE(resolved.features.software_gi);
    REQUIRE_FALSE(resolved.features.hardware_gi);
    REQUIRE(resolved.features.submission == gpu_submission_path::cpu_direct);
}

TEST_CASE("renderer applies resolved configuration when attaching a backend")
{
    auto backend = std::make_unique<recording_backend>();
    backend->capabilities_.dedicated_video_memory = 4ull * 1024ull * 1024ull * 1024ull;
    backend->capabilities_.dynamic_rendering = true;
    auto* backend_ptr = backend.get();

    arc::render::renderer_config config{};
    config.quality = arc::render::render_quality_tier::high;
    arc::render::renderer renderer(config);
    renderer.set_backend(std::move(backend));

    REQUIRE(renderer.resolved_config().quality == arc::render::render_quality_tier::medium);
    REQUIRE(renderer.resolved_config().path == arc::render::render_path::deferred);
    REQUIRE(backend_ptr->configured.quality == arc::render::render_quality_tier::medium);
    REQUIRE_FALSE(backend_ptr->configured.fallback_reasons.empty());
}

TEST_CASE("frame budget controller scales expensive systems before resolution")
{
    arc::render::frame_budget_controller controller;
    controller.reset(arc::render::standard_render_quality_profile, arc::render::default_target_frame_time_ms);

    for (std::uint32_t index = 0; index < 12; ++index)
        controller.update(30.0f);
    const auto reduced = controller.settings();
    REQUIRE(reduced.radiance_probe_update_budget <
            arc::render::standard_render_quality_profile.radiance_probe_update_budget);
    REQUIRE(reduced.volumetric_resolution_scale == Catch::Approx(1.0f));
    REQUIRE(reduced.render_scale == Catch::Approx(1.0f));
    REQUIRE(controller.smoothed_frame_time_ms() > arc::render::default_target_frame_time_ms);

    for (std::uint32_t index = 0; index < 48; ++index)
        controller.update(5.0f);
    REQUIRE(controller.settings().radiance_probe_update_budget >= reduced.radiance_probe_update_budget);
    REQUIRE(controller.settings().render_scale <= 1.0f);
}

TEST_CASE("renderer exposes compiled render graph snapshots through frame profile")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    const auto result = renderer.render_frame(7, arc::render::make_scene_draw_graph("viewport"));

    REQUIRE(result.has_value());
    const auto profile = renderer.last_frame_profile();
    REQUIRE(profile.frame_index == 7);
    REQUIRE(profile.summary == "recorded");
    REQUIRE(profile.graph.passes.size() == backend_ptr->last_pass_count);
    REQUIRE_FALSE(profile.graph.resources.empty());
    REQUIRE(profile.graph.resources[2].name == "scene_color");
    REQUIRE(profile.graph.resources[2].format == arc::render::render_format::rgba16_float);
    REQUIRE(profile.clustered_lights.available);
    REQUIRE(profile.clustered_lights.cluster_count == 96);
    REQUIRE(profile.clustered_lights.overflow_count == 1);
}

TEST_CASE("renderer forwards ObjectID pick requests to backend")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    renderer.request_object_pick(7, 12, 34);

    REQUIRE(backend_ptr->pick_requested);
    REQUIRE(backend_ptr->pick_request.request_id == 7);
    REQUIRE(backend_ptr->pick_request.x == 12);
    REQUIRE(backend_ptr->pick_request.y == 34);
    REQUIRE_FALSE(renderer.last_object_pick().available);
}

TEST_CASE("renderer forwards coherent asynchronous frame capture requests")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    renderer.request_frame_capture({.capture_id = 31,
                                    .channels = {arc::render::render_capture_channel::output_color,
                                                 arc::render::render_capture_channel::object_id}});

    REQUIRE(backend_ptr->capture_requested);
    REQUIRE(backend_ptr->capture_request.capture_id == 31);
    REQUIRE(backend_ptr->capture_request.channels.size() == 2);
    REQUIRE_FALSE(renderer.last_frame_capture().available);
}

TEST_CASE("renderer forwards viewport resize events to backend")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    arc::render::render_event_buffer buffer;
    arc::render::render_event_writer writer(buffer);
    writer.viewport_resize(800, 450);
    renderer.frame_queue().submit(std::move(buffer));

    const auto result = renderer.render_frame(1, arc::render::make_clear_present_graph("viewport"));
    REQUIRE(result.has_value());
    REQUIRE(backend_ptr->viewport_width == 800);
    REQUIRE(backend_ptr->viewport_height == 450);
    REQUIRE(renderer.viewport_texture().valid());
}

TEST_CASE("renderer create mesh enqueues typed upload and tracks handle lifetime")
{
    arc::render::renderer renderer;
    arc::render::mesh_data mesh;
    mesh.name = "triangle";
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};

    const auto handle = renderer.create_mesh(std::move(mesh));
    REQUIRE(renderer.mesh_alive(handle));

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 3);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::mesh_upload);
    const auto& upload = std::get<arc::render::mesh_upload_event>(packet.events[0].payload);
    REQUIRE(upload.handle == handle);
    REQUIRE(upload.mesh->vertices.size() == 3);
    REQUIRE(upload.mesh->indices.size() == 3);
    REQUIRE(packet.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    REQUIRE(packet.events[2].type() == arc::render::render_event_type::lighting_geometry_upload);
    const auto& lighting_upload = std::get<arc::render::lighting_geometry_upload_event>(packet.events[2].payload);
    REQUIRE(lighting_upload.geometry->cards.size() == 6);
    REQUIRE(lighting_upload.geometry->distance_field.mode ==
            arc::render::distance_field_mode::two_sided_unsigned_distance);
}

TEST_CASE("renderer updates mesh vertices and retires stale handles")
{
    arc::render::renderer renderer;
    arc::render::mesh_data mesh;
    mesh.name = "dynamic terrain chunk";
    mesh.usage = arc::render::mesh_usage::dynamic_per_frame;
    mesh.vertices.resize(4);
    mesh.indices = {0, 1, 2, 0, 2, 3};
    const auto handle = renderer.create_mesh(std::move(mesh));
    renderer.frame_queue().commit(1);

    std::vector<arc::render::mesh_vertex> vertices(4);
    vertices[0].position[1] = 3.0f;
    REQUIRE(renderer.update_mesh_vertices(handle, vertices));
    auto update = renderer.frame_queue().commit(2);
    REQUIRE(update.events.size() == 2);
    REQUIRE(update.events[0].type() == arc::render::render_event_type::mesh_upload);
    REQUIRE(update.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->indices.size() == 6);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->usage ==
            arc::render::mesh_usage::dynamic_per_frame);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->vertices[0].position[1] == 3.0f);

    REQUIRE(renderer.destroy_mesh(handle));
    REQUIRE_FALSE(renderer.mesh_alive(handle));
    auto destroy = renderer.frame_queue().commit(3);
    REQUIRE(destroy.events.size() == 2);
    REQUIRE(destroy.events[0].type() == arc::render::render_event_type::mesh_destroy);
    REQUIRE(destroy.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    REQUIRE_FALSE(renderer.destroy_mesh(handle));
}

TEST_CASE("renderer create virtual mesh enqueues typed upload and keeps CPU cluster metadata")
{
    arc::render::renderer renderer;
    arc::render::virtual_mesh_data mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    mesh.clusters.push_back(
        {.first_index = 0, .index_count = 3, .triangle_count = 1, .vertex_count = 3, .material_index = 2});

    const auto handle = renderer.create_virtual_mesh(std::move(mesh));
    REQUIRE(renderer.virtual_mesh_alive(handle));
    REQUIRE(renderer.virtual_mesh_content_generation(handle) == 1);
    REQUIRE(renderer.virtual_mesh_data_for(handle) != nullptr);
    REQUIRE(renderer.virtual_mesh_data_for(handle)->clusters.size() == 1);

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 1);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::virtual_mesh_upload);
    const auto& upload = std::get<arc::render::virtual_mesh_upload_event>(packet.events[0].payload);
    REQUIRE(upload.handle == handle);
    REQUIRE(upload.mesh->vertices.size() == 3);
    REQUIRE(upload.mesh->indices.size() == 3);
    REQUIRE(upload.mesh->clusters.size() == 1);
    REQUIRE(upload.mesh->clusters[0].index_count == 3);

    auto updated = *upload.mesh;
    updated.clusters.push_back(updated.clusters.front());
    REQUIRE(renderer.update_virtual_mesh(handle, std::move(updated)));
    const auto update = renderer.frame_queue().commit(2);
    REQUIRE(update.events.size() == 1);
    REQUIRE(update.events[0].type() == arc::render::render_event_type::virtual_mesh_upload);
    REQUIRE(renderer.virtual_mesh_data_for(handle)->clusters.size() == 2);
    REQUIRE(renderer.virtual_mesh_content_generation(handle) == 2);

    REQUIRE(renderer.destroy_virtual_mesh(handle));
    REQUIRE_FALSE(renderer.virtual_mesh_alive(handle));
    REQUIRE(renderer.virtual_mesh_content_generation(handle) == 0);
    REQUIRE(renderer.virtual_mesh_data_for(handle) == nullptr);
    const auto destroy = renderer.frame_queue().commit(3);
    REQUIRE(destroy.events.size() == 1);
    REQUIRE(destroy.events[0].type() == arc::render::render_event_type::virtual_mesh_destroy);
    REQUIRE(std::get<arc::render::virtual_mesh_destroy_event>(destroy.events[0].payload).handle == handle);
    REQUIRE_FALSE(renderer.destroy_virtual_mesh(handle));
}

TEST_CASE("renderer realizes and retires one unified cooked geometry resource")
{
    arc::render::mesh_data source;
    source.vertices.resize(3);
    source.vertices[1].position[0] = 1.0f;
    source.vertices[2].position[1] = 1.0f;
    source.indices = {0, 1, 2};

    arc::render::renderer renderer;
    const auto geometry = renderer.create_geometry_resource(arc::render::build_virtual_mesh(source), 9);
    REQUIRE(geometry.valid());
    REQUIRE(geometry.asset_generation == 9);
    REQUIRE(geometry.conventional_lod_count == 4);
    REQUIRE(renderer.mesh_alive(geometry.conventional));
    REQUIRE(renderer.virtual_mesh_alive(geometry.virtualized));
    const auto uploads = renderer.frame_queue().commit(1);
    REQUIRE(uploads.events.size() == 13);
    REQUIRE(uploads.events.back().type() == arc::render::render_event_type::virtual_mesh_upload);

    REQUIRE(renderer.destroy_geometry_resource(geometry));
    REQUIRE_FALSE(renderer.mesh_alive(geometry.conventional));
    REQUIRE_FALSE(renderer.virtual_mesh_alive(geometry.virtualized));
    const auto destroys = renderer.frame_queue().commit(2);
    REQUIRE(destroys.events.size() == 13);
    REQUIRE(destroys.events.back().type() == arc::render::render_event_type::virtual_mesh_destroy);
}

TEST_CASE("renderer creates texture and material resources")
{
    arc::render::renderer renderer;
    arc::render::texture_data texture;
    texture.name = "encoded";
    texture.mime_type = "image/png";
    texture.encoded = {std::byte{1}, std::byte{2}};

    const auto texture_handle = renderer.create_texture(std::move(texture));
    REQUIRE(renderer.texture_alive(texture_handle));

    arc::render::material_descriptor material;
    material.name = "pbr";
    material.base_color_texture = texture_handle;
    material.metallic = 0.25f;
    material.roughness = 0.8f;
    material.alpha_mode = arc::render::material_alpha_mode::masked;

    const auto material_handle = renderer.create_material(material);
    REQUIRE(renderer.material_alive(material_handle));

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 4);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::texture_upload);
    REQUIRE(packet.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    REQUIRE(packet.events[2].type() == arc::render::render_event_type::material_upload);
    REQUIRE(packet.events[3].type() == arc::render::render_event_type::gpu_resource_table_update);
    const auto& uploaded = std::get<arc::render::material_upload_event>(packet.events[2].payload);
    REQUIRE(uploaded.handle == material_handle);
    REQUIRE(uploaded.material->handle == material_handle);
    REQUIRE(uploaded.material->base_color_texture == texture_handle);
    REQUIRE(uploaded.material->alpha_mode == arc::render::material_alpha_mode::masked);

    auto updated = material;
    updated.name = "pbr_updated";
    updated.roughness = 0.35f;
    updated.base_color = {0.25f, 0.5f, 0.75f, 1.0f};

    REQUIRE(renderer.update_material(material_handle, updated));
    const auto update_packet = renderer.frame_queue().commit(2);
    REQUIRE(update_packet.events.size() == 2);
    REQUIRE(update_packet.events[0].type() == arc::render::render_event_type::material_upload);
    REQUIRE(update_packet.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    const auto& material_update = std::get<arc::render::material_upload_event>(update_packet.events[0].payload);
    REQUIRE(material_update.handle == material_handle);
    REQUIRE(material_update.material->handle == material_handle);
    REQUIRE(material_update.material->roughness == Catch::Approx(0.35f));
    REQUIRE(material_update.material->base_color[2] == Catch::Approx(0.75f));

    arc::render::texture_data replacement;
    replacement.name = "environment replacement";
    replacement.width = 2;
    replacement.height = 1;
    replacement.pixels.resize(8);
    REQUIRE(renderer.update_texture(texture_handle, replacement));
    const auto texture_update_packet = renderer.frame_queue().commit(3);
    REQUIRE(texture_update_packet.events.size() == 2);
    REQUIRE(texture_update_packet.events[1].type() == arc::render::render_event_type::gpu_resource_table_update);
    const auto& texture_update = std::get<arc::render::texture_upload_event>(texture_update_packet.events[0].payload);
    REQUIRE(texture_update.handle == texture_handle);
    REQUIRE(texture_update.texture->width == 2);

    REQUIRE_FALSE(renderer.update_material({.index = 999, .generation = 1}, updated));
    REQUIRE_FALSE(renderer.update_texture({.index = 999, .generation = 1}, replacement));
}

TEST_CASE("renderer creates environment resources")
{
    arc::render::renderer renderer;
    arc::render::environment_descriptor environment;
    environment.name = "studio";
    environment.fallback_color = {0.20f, 0.22f, 0.25f};
    environment.intensity = 1.5f;

    const auto handle = renderer.create_environment(environment);
    REQUIRE(handle.valid());
    REQUIRE(renderer.environment_alive(handle));

    const auto packet = renderer.frame_queue().commit(12);
    REQUIRE(packet.events.size() == 1);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::environment_upload);
    const auto& upload = std::get<arc::render::environment_upload_event>(packet.events[0].payload);
    REQUIRE(upload.handle == handle);
    REQUIRE(upload.environment);
    REQUIRE(upload.environment->handle == handle);
    REQUIRE(upload.environment->intensity == Catch::Approx(1.5f));

    environment.intensity = 0.75f;
    REQUIRE(renderer.update_environment(handle, environment));
    REQUIRE(renderer.destroy_environment(handle));
    REQUIRE_FALSE(renderer.environment_alive(handle));
    const auto lifecycle = renderer.frame_queue().commit(13);
    REQUIRE(lifecycle.events.size() == 2);
    REQUIRE(lifecycle.events[0].type() == arc::render::render_event_type::environment_upload);
    REQUIRE(lifecycle.events[1].type() == arc::render::render_event_type::environment_destroy);
}

TEST_CASE("scene lighting data packs sorted capped light arrays")
{
    std::vector<arc::render::directional_light_event> directional;
    for (std::uint32_t index = 0; index < arc::render::max_directional_lights + 2; ++index)
    {
        directional.push_back({.direction = {0.0f, -1.0f, 0.0f},
                               .color = {1.0f, 1.0f, 1.0f},
                               .intensity = static_cast<float>(index + 1),
                               .label = "sun"});
    }

    std::vector<arc::render::point_light_event> points{
        {.object_id = {.index = 17, .generation = 3},
         .position = {1.0f, 2.0f, 3.0f},
         .color = {1.0f, 0.5f, 0.25f},
         .intensity = 80.0f,
         .range = 4.0f,
         .intensity_unit = arc::render::light_intensity_unit::lumen},
        {.position = {0.0f, 0.0f, 0.0f}, .color = {1.0f, 1.0f, 1.0f}, .intensity = 2.0f, .range = 8.0f}};
    std::vector<arc::render::spot_light_event> spots{{.position = {0.0f, 1.0f, 0.0f},
                                                      .direction = {0.0f, -1.0f, 0.0f},
                                                      .color = {0.8f, 0.9f, 1.0f},
                                                      .intensity = 3.0f,
                                                      .range = 10.0f,
                                                      .inner_angle = 0.2f,
                                                      .outer_angle = 0.7f}};

    arc::render::environment_descriptor environment;
    environment.fallback_color = {0.1f, 0.2f, 0.3f};
    environment.intensity = 1.25f;

    const auto data = arc::render::pack_scene_lighting(directional, points, spots, &environment);
    REQUIRE(data.directional_count == arc::render::max_directional_lights);
    REQUIRE(data.skipped_directional_count == 2);
    REQUIRE(data.directional_lights[0].direction_intensity[3] == Catch::Approx(6.0f));
    REQUIRE(data.point_count == 2);
    REQUIRE(data.point_lights[0].color_intensity[3] == Catch::Approx(80.0f / (4.0f * arc::math::pi<float>)));
    REQUIRE(data.point_lights[0].object_id_shadow[0] == Catch::Approx(17.0f));
    REQUIRE(data.point_lights[0].object_id_shadow[1] == Catch::Approx(3.0f));
    REQUIRE(data.point_lights[0].shadow_parameters[0] == Catch::Approx(-1.0f));
    REQUIRE(data.spot_count == 1);
    REQUIRE(data.spot_lights[0].params[0] == Catch::Approx(0.7f));
    REQUIRE(data.ambient_color_intensity[1] == Catch::Approx(0.2f));
    REQUIRE(data.local_shadow_face_count == 0);
    STATIC_REQUIRE(arc::render::max_local_shadow_faces == 144);

    environment.prefiltered = true;
    environment.diffuse_irradiance = {0.4f, 0.5f, 0.6f};
    environment.diffuse_intensity = 0.75f;
    const auto prefiltered = arc::render::pack_scene_lighting({}, {}, {}, &environment);
    REQUIRE(prefiltered.ambient_color_intensity[0] == Catch::Approx(0.4f));
    REQUIRE(prefiltered.ambient_color_intensity[2] == Catch::Approx(0.6f));
    REQUIRE(prefiltered.ambient_color_intensity[3] == Catch::Approx(0.75f));
}

TEST_CASE("light unit and temperature helpers provide stable defaults")
{
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::unitless, 2.0f, 4.0f) ==
            Catch::Approx(2.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::candela, 5.0f, 2.0f) ==
            Catch::Approx(5.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::lux, 3.0f, 2.0f) ==
            Catch::Approx(3.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::lumen, 4.0f * arc::math::pi<float>) ==
            Catch::Approx(1.0f));
    const auto warm = arc::render::color_temperature_rgb(3000.0f);
    const auto cool = arc::render::color_temperature_rgb(9000.0f);
    REQUIRE(warm[0] >= warm[2]);
    REQUIRE(cool[2] >= cool[0]);
}

TEST_CASE("PBR color transfer and material texture semantics are explicit")
{
    const arc::math::vector3f srgb{0.0f, 0.5f, 1.0f};
    const auto linear = arc::render::srgb_to_linear(srgb);
    const auto round_trip = arc::render::linear_to_srgb(linear);
    REQUIRE(round_trip[0] == Catch::Approx(srgb[0]).margin(1.0e-6f));
    REQUIRE(round_trip[1] == Catch::Approx(srgb[1]).margin(1.0e-5f));
    REQUIRE(round_trip[2] == Catch::Approx(srgb[2]).margin(1.0e-6f));
    REQUIRE(arc::render::texture_semantic_accepts(arc::render::texture_semantic::base_color,
                                                  arc::render::texture_color_space::srgb));
    REQUIRE_FALSE(arc::render::texture_semantic_accepts(arc::render::texture_semantic::normal,
                                                        arc::render::texture_color_space::srgb));
}

TEST_CASE("PBR reference functions stay finite and preserve physical limits")
{
    for (const float roughness : {0.04f, 0.25f, 0.6f, 1.0f})
    {
        const float distribution = arc::render::ggx_distribution(0.75f, roughness);
        const float visibility = arc::render::smith_ggx_correlated(0.6f, 0.7f, roughness);
        REQUIRE(std::isfinite(distribution));
        REQUIRE(std::isfinite(visibility));
        REQUIRE(distribution >= 0.0f);
        REQUIRE(visibility >= 0.0f);
    }
    const auto fresnel = arc::render::fresnel_schlick(0.0f, {0.04f, 0.04f, 0.04f});
    REQUIRE(fresnel[0] == Catch::Approx(1.0f));
    const auto absorption = arc::render::beer_lambert_attenuation({0.5f, 0.25f, 1.0f}, 2.0f, 2.0f);
    REQUIRE(absorption[0] == Catch::Approx(0.5f));
    REQUIRE(absorption[1] == Catch::Approx(0.25f));
    REQUIRE(absorption[2] == Catch::Approx(1.0f));
}

TEST_CASE("physical attenuation exposure and area light packing are stable")
{
    REQUIRE(arc::render::inverse_square_attenuation(2.0f, 0.0f) == Catch::Approx(0.25f));
    REQUIRE(arc::render::inverse_square_attenuation(10.0f, 5.0f) == Catch::Approx(0.0f));
    REQUIRE(arc::render::cone_solid_angle(arc::math::pi<float> * 0.5f) == Catch::Approx(2.0f * arc::math::pi<float>));

    arc::render::exposure_settings settings;
    settings.mode = arc::render::exposure_mode::automatic;
    settings.brighten_speed = 4.0f;
    settings.darken_speed = 2.0f;
    auto state = arc::render::adapt_exposure({}, settings, 5.0f, 1.0f / 60.0f, true);
    REQUIRE(state.valid);
    REQUIRE(state.ev100 == Catch::Approx(5.0f));
    const auto adapted = arc::render::adapt_exposure(state, settings, 8.0f, 1.0f, false);
    REQUIRE(adapted.ev100 > state.ev100);
    REQUIRE(adapted.ev100 < 8.0f);

    std::vector<arc::render::area_light_event> areas{{.intensity = 1000.0f,
                                                      .width = 2.0f,
                                                      .height = 1.0f,
                                                      .shape = arc::render::area_light_shape::rectangle,
                                                      .intensity_unit = arc::render::light_intensity_unit::lumen}};
    const auto lighting = arc::render::pack_scene_lighting({}, {}, {}, nullptr, 0, 0, areas);
    REQUIRE(lighting.area_count == 1);
    REQUIRE(lighting.area_lights[0].color_intensity[3] == Catch::Approx(1000.0f / (2.0f * arc::math::pi<float>)));
}

TEST_CASE("descriptor slots reject stale generations")
{
    arc::render::descriptor_slot_pool pool;
    const auto first = pool.allocate(arc::render::descriptor_resource_type::sampled_image);
    REQUIRE(first.valid());
    REQUIRE(pool.alive(first));

    REQUIRE(pool.release(first));
    REQUIRE_FALSE(pool.alive(first));

    const auto second = pool.allocate(arc::render::descriptor_resource_type::sampled_image);
    REQUIRE(second.index == first.index);
    REQUIRE(second.generation != first.generation);
    REQUIRE(pool.alive(second));
}

TEST_CASE("deferred resource releaser waits for completed frames")
{
    arc::render::deferred_resource_releaser releaser;
    int released = 0;
    releaser.defer(4, [&]() { released += 1; });
    releaser.defer(7, [&]() { released += 10; });

    REQUIRE(releaser.collect(3) == 0);
    REQUIRE(released == 0);
    REQUIRE(releaser.collect(4) == 1);
    REQUIRE(released == 1);
    REQUIRE(releaser.pending_count() == 1);
    REQUIRE(releaser.collect(8) == 1);
    REQUIRE(released == 11);
}

TEST_CASE("frame allocator resets transient allocations")
{
    arc::render::frame_allocator allocator(16);
    auto* first = static_cast<std::uint32_t*>(allocator.allocate(sizeof(std::uint32_t), alignof(std::uint32_t)));
    *first = 42;
    REQUIRE(allocator.used() >= sizeof(std::uint32_t));

    allocator.reset();
    REQUIRE(allocator.used() == 0);
    auto* second = static_cast<std::uint32_t*>(allocator.allocate(sizeof(std::uint32_t), alignof(std::uint32_t)));
    *second = 7;
    REQUIRE(*second == 7);
}

TEST_CASE("GPU upload arena retires ranges by completed frame")
{
    arc::render::gpu_upload_arena arena(256);
    arena.begin_frame(4);
    auto first = arena.try_allocate(80, 16);
    auto second = arena.try_allocate(80, 16);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(first.offset % 16 == 0);
    REQUIRE(arena.used() >= 160);

    arena.begin_frame(5);
    auto third = arena.try_allocate(80, 16);
    REQUIRE(third);
    REQUIRE_FALSE(arena.try_allocate(80, 16));
    REQUIRE(arena.retire_completed(3) == 0);
    REQUIRE(arena.retire_completed(4) == 2);

    auto wrapped = arena.try_allocate(80, 16);
    REQUIRE(wrapped);
    REQUIRE(wrapped.frame == 5);
    REQUIRE(arena.peak_used() >= 240);
    REQUIRE(arena.retire_completed(5) == 2);
    REQUIRE(arena.used() == 0);
}

TEST_CASE("GPU upload arena can suballocate persistently mapped backend storage")
{
    std::array<std::byte, 128> mapped{};
    arc::render::gpu_upload_arena arena(mapped);
    arena.begin_frame(9);

    auto allocation = arena.try_allocate(24, 32);
    REQUIRE(allocation);
    REQUIRE(allocation.offset % 32 == 0);
    REQUIRE(allocation.bytes.data() == mapped.data() + allocation.offset);
    allocation.bytes.front() = std::byte{0x5a};
    REQUIRE(mapped[allocation.offset] == std::byte{0x5a});

    REQUIRE(arena.retire_completed(8) == 0);
    REQUIRE(arena.retire_completed(9) == 1);
    REQUIRE(arena.used() == 0);
}

TEST_CASE("pipeline handle cache reuses equivalent keys")
{
    arc::render::pipeline_handle_cache cache;
    arc::render::graphics_pipeline_key key{.vertex_shader = {.index = 1, .generation = 1},
                                           .fragment_shader = {.index = 2, .generation = 1},
                                           .vertex_layout = "pnu",
                                           .color_format = "rgba16f",
                                           .depth_format = "d32",
                                           .depth_test = true,
                                           .depth_write = true};
    arc::render::pipeline_handle pipeline{.index = 9, .generation = 3};

    REQUIRE_FALSE(cache.find(key).valid());
    cache.insert(key, pipeline);
    REQUIRE(cache.find(key) == pipeline);
    key.wireframe = true;
    REQUIRE_FALSE(cache.find(key).valid());
    key.wireframe = false;
    key.permutation.has_normal_texture = true;
    REQUIRE_FALSE(cache.find(key).valid());
}

TEST_CASE("shader permutation keys capture material features")
{
    arc::render::material_descriptor material;
    material.alpha_mode = arc::render::material_alpha_mode::blend;
    material.normal_texture = {.index = 1, .generation = 1};
    material.emissive_texture = {.index = 2, .generation = 1};
    material.clear_coat_factor = 0.5f;

    const auto key = arc::render::make_shader_permutation_key(material, 3, true);
    REQUIRE(key.alpha_mode == arc::render::material_alpha_mode::blend);
    REQUIRE(key.debug_view == 3);
    REQUIRE(key.has_normal_texture);
    REQUIRE(key.has_emissive_texture);
    REQUIRE(key.clear_coat);
    REQUIRE(key.wireframe);

    auto other = key;
    other.wireframe = false;
    REQUIRE(hash_shader_permutation_key(key) != hash_shader_permutation_key(other));
}

namespace
{

class counting_shader_compiler final : public arc::render::shader_compiler
{
public:
    arc::render::shader_compile_result compile(const arc::render::shader_compile_request& request) override
    {
        ++count;
        return arc::render::shader_compile_result::success(
            {.bytecode = {std::uint8_t(count)},
             .reflection = {
                 .entry_points = {{.id = arc::render::make_shader_entry_point_id(request.entry_point, request.stage),
                                   .name = request.entry_point,
                                   .stage = request.stage,
                                   .profile = request.profile}}}});
    }

    std::string_view fingerprint() const noexcept override
    {
        return "arc.test-compiler/1";
    }

    int count{};
};

} // namespace

TEST_CASE("shader library cache reuses unchanged source requests")
{
    const auto path = std::filesystem::temp_directory_path() / "arc_shader_cache_test.slang";
    {
        std::ofstream file(path);
        file << "float4 main() : SV_Target { return 1; }";
    }

    counting_shader_compiler compiler;
    arc::render::shader_library_cache cache;
    arc::render::shader_compile_request request{.source_path = path.string(),
                                                .entry_point = "main",
                                                .profile = "fragment",
                                                .target = arc::render::shader_target::spirv};

    const auto first = cache.compile_or_get(compiler, request);
    const auto second = cache.compile_or_get(compiler, request);

    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(compiler.count == 1);
    REQUIRE(cache.size() == 1);
    REQUIRE_FALSE(cache.source_changed(request));

    std::filesystem::remove(path);
}

TEST_CASE("shader cache invalidates when an include changes")
{
    const auto directory = std::filesystem::temp_directory_path() / "arc_shader_include_cache_test";
    std::filesystem::create_directories(directory);
    const auto source_path = directory / "main.slang";
    const auto include_path = directory / "common.slang";
    {
        std::ofstream source(source_path);
        source << "#include \"common.slang\"\nfloat4 main() : SV_Target { return color(); }";
        std::ofstream include(include_path);
        include << "float4 color() { return 1; }";
    }

    counting_shader_compiler compiler;
    arc::render::shader_library_cache cache;
    arc::render::shader_compile_request request{.source_path = source_path.string(),
                                                .entry_point = "main",
                                                .profile = "spirv_1_5",
                                                .include_directories = {directory}};
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE_FALSE(cache.source_changed(request));
    {
        std::ofstream include(include_path, std::ios::trunc);
        include << "float4 color() { return 0; }";
    }
    REQUIRE(cache.source_changed(request));
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 2);
    std::filesystem::remove_all(directory);
}

TEST_CASE("shader cache tracks Slang module imports and shader library versions")
{
    const auto directory = std::filesystem::temp_directory_path() / "arc_shader_import_cache_test";
    std::filesystem::create_directories(directory);
    const auto source_path = directory / "main.slang";
    const auto module_path = directory / "arc_surface.slang";
    {
        std::ofstream source(source_path);
        source << "import arc.surface;\nfloat4 main() : SV_Target { return arcColor(); }";
        std::ofstream module(module_path);
        module << "module arc.surface;\nfloat4 arcColor() { return 1; }";
    }

    counting_shader_compiler compiler;
    arc::render::shader_library_cache cache;
    arc::render::shader_compile_request request{.source_path = source_path.string(),
                                                .entry_point = "main",
                                                .profile = "spirv_1_5",
                                                .include_directories = {directory}};
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE_FALSE(cache.source_changed(request));
    {
        std::ofstream module(module_path, std::ios::trunc);
        module << "module arc.surface;\nfloat4 arcColor() { return 0; }";
    }
    REQUIRE(cache.source_changed(request));
    REQUIRE(cache.compile_or_get(compiler, request));

    request.library_version = "arc-shader-library/2";
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 3);
    std::filesystem::remove_all(directory);
}

TEST_CASE("shader packages round trip reflection and reject corruption")
{
    arc::render::shader_package package{
        .id = {.high = 1, .low = 2},
        .generation = {3},
        .target = arc::render::shader_target::spirv,
        .permutation = {4},
        .compiled = {.bytecode = {3, 2, 35, 7},
                     .reflection = {.domain = arc::render::shader_domain::surface,
                                    .entry_points = {{.id = arc::render::make_shader_entry_point_id(
                                                          "main", arc::render::shader_stage::fragment),
                                                      .name = "main",
                                                      .stage = arc::render::shader_stage::fragment,
                                                      .profile = "spirv_1_5"}},
                                    .parameters = {{.id = arc::render::make_shader_parameter_id("baseColor"),
                                                    .name = "baseColor",
                                                    .type = arc::render::shader_parameter_type::float4,
                                                    .size = 16}},
                                    .passes = {{.pass = arc::render::material_pass::forward,
                                                .entry_point = arc::render::make_shader_entry_point_id(
                                                    "main", arc::render::shader_stage::fragment)}}},
                     .build_hash = {.bytes = {std::byte{1}}},
                     .dependencies = {{.path = "main.slang", .content_hash = {.bytes = {std::byte{2}}}}},
                     .source_map = {{.generated_line = 12,
                                     .source = {.path = "material.arcmat", .graph_node_id = "base-color"}}},
                     .diagnostics = {{.severity = arc::render::shader_diagnostic_severity::warning,
                                      .code = "W001",
                                      .message = "test warning",
                                      .location = {.path = "main.slang", .line = 4, .column = 2},
                                      .include_stack = {{.path = "shared.slang", .line = 1, .column = 1}},
                                      .permutation = arc::render::shader_permutation_id{4}}},
                     .compiler_fingerprint = "slang/2026.14.1"}};
    const auto encoded = arc::render::serialize_shader_package(package);
    REQUIRE(encoded);
    const auto decoded = arc::render::deserialize_shader_package(encoded.value());
    REQUIRE(decoded);
    REQUIRE(decoded.value().id == package.id);
    REQUIRE(decoded.value().compiled.bytecode == package.compiled.bytecode);
    REQUIRE(decoded.value().compiled.reflection.parameters.front().name == "baseColor");
    REQUIRE(decoded.value().compiled.source_map.front().source.graph_node_id == "base-color");
    REQUIRE(decoded.value().compiled.diagnostics.front().include_stack.front().path == "shared.slang");

    auto corrupt = encoded.value();
    corrupt.pop_back();
    REQUIRE_FALSE(arc::render::deserialize_shader_package(corrupt));
}

TEST_CASE("shader package publication preserves last good generations")
{
    arc::render::shader_package_library library;
    arc::render::shader_package first{
        .id = {.high = 7, .low = 9},
        .generation = {1},
        .permutation = {4},
        .compiled = {.bytecode = {1, 2, 3}},
    };
    first.compiled.build_hash.bytes[0] = std::byte{1};
    REQUIRE(library.publish(first, 3) == arc::render::shader_publication_status::published);

    auto conflicting = first;
    conflicting.compiled.build_hash.bytes[0] = std::byte{9};
    REQUIRE(library.publish(std::move(conflicting), 3) ==
            arc::render::shader_publication_status::rejected_stale_generation);

    auto second = first;
    second.generation = arc::render::shader_generation_id{2};
    second.compiled.build_hash.bytes[0] = std::byte{2};
    REQUIRE(library.publish(second, 8) == arc::render::shader_publication_status::published);
    REQUIRE(library.retired_count() == 1);
    REQUIRE(library.find(first.id, first.permutation)->generation == second.generation);

    library.report_failure(
        first.id, first.permutation,
        {.code = arc::render::shader_compile_error_code::compilation_failed, .message = "transient edit failed"});
    REQUIRE(library.snapshot(first.id, first.permutation)->last_error.has_value());
    REQUIRE(library.find(first.id, first.permutation)->generation == second.generation);

    library.collect(7);
    REQUIRE(library.retired_count() == 1);
    library.collect(8);
    REQUIRE(library.retired_count() == 0);
    REQUIRE(library.publish(first, 9) == arc::render::shader_publication_status::rejected_stale_generation);
}

TEST_CASE("material instances validate stable parameter overrides without changing permutations")
{
    const auto tint = arc::render::make_shader_parameter_id("tint");
    arc::render::material_definition_descriptor definition{
        .material = {.name = "Base", .shader_permutation = {99}},
        .parameter_layout = {
            {.id = tint, .name = "tint", .type = arc::render::shader_parameter_type::float3, .size = 12}}};
    arc::render::material_instance_descriptor instance{
        .parent = {.index = 1, .generation = 1},
        .name = "Blue",
        .overrides = {{.id = tint, .name = "tint", .value = arc::math::vector3f{0.1f, 0.2f, 1.0f}}}};
    const auto resolved = arc::render::resolve_material_instance(definition, instance);
    REQUIRE(resolved);
    REQUIRE(resolved.value().name == "Blue");
    REQUIRE(resolved.value().shader_permutation == definition.material.shader_permutation);
    REQUIRE(resolved.value().parameters.size() == 1);

    instance.overrides.front().value = 1.0f;
    const auto invalid = arc::render::resolve_material_instance(definition, instance);
    REQUIRE_FALSE(invalid);
    REQUIRE(invalid.error().code == arc::render::material_instance_error_code::incompatible_type);
}

TEST_CASE("material routing keeps standard surfaces deferred and custom contracts forward")
{
    arc::render::material_descriptor material;
    REQUIRE(arc::render::resolve_material_render_path(material) == arc::render::material_render_path::deferred);
    material.deferred_compatible = false;
    REQUIRE(arc::render::resolve_material_render_path(material) ==
            arc::render::material_render_path::clustered_forward);
    material.deferred_compatible = true;
    material.shading_model = arc::render::material_shading_model::unlit;
    REQUIRE(arc::render::resolve_material_render_path(material) ==
            arc::render::material_render_path::clustered_forward);
}

TEST_CASE("GLB mesh loader reads static triangle geometry")
{
    const auto path = write_triangle_glb();
    const auto result = arc::render::load_gltf_mesh(path);

    REQUIRE(result.succeeded());
    REQUIRE(result.mesh.vertices.size() == 3);
    REQUIRE(result.mesh.indices == std::vector<std::uint32_t>{0, 1, 2});
    REQUIRE(result.mesh.vertices[0].position[1] == 0.5f);
    REQUIRE(result.mesh.vertices[0].normal[2] == 1.0f);
    REQUIRE(result.mesh.vertices[0].tangent[0] == Catch::Approx(1.0f));
    REQUIRE(result.mesh.vertices[0].tangent[1] == Catch::Approx(0.0f));
    REQUIRE(result.mesh.vertices[0].tangent[2] == Catch::Approx(0.0f));
    REQUIRE(result.mesh.vertices[0].tangent[3] == Catch::Approx(1.0f));
    REQUIRE(result.mesh.vertices[1].texcoord[1] == 1.0f);
    REQUIRE(result.mesh.material_index == 0);
    REQUIRE(result.textures.size() == 1);
    REQUIRE(result.textures[0].mime_type == "image/png");
    REQUIRE(result.textures[0].encoded.size() == 4);
    REQUIRE(result.materials.size() == 1);
    REQUIRE(result.materials[0].material.name == "TestMaterial");
    REQUIRE(result.materials[0].material.alpha_mode == arc::render::material_alpha_mode::masked);
    REQUIRE(result.materials[0].material.alpha_cutoff == Catch::Approx(0.35f));
    REQUIRE(result.materials[0].material.base_color[2] == Catch::Approx(0.75f));
    REQUIRE(result.materials[0].material.metallic == Catch::Approx(0.2f));
    REQUIRE(result.materials[0].material.roughness == Catch::Approx(0.7f));
    REQUIRE(result.materials[0].material.double_sided);
    REQUIRE(result.materials[0].material.normal_scale == Catch::Approx(0.8f));
    REQUIRE(result.materials[0].material.occlusion_strength == Catch::Approx(0.6f));
    REQUIRE(result.materials[0].material.emissive_factor[1] == Catch::Approx(0.2f));
    REQUIRE(result.materials[0].textures.base_color == 0);
    REQUIRE(result.materials[0].textures.normal == 0);

    std::filesystem::remove(path);
}

TEST_CASE("DDS loader parses BC1 texture metadata")
{
    auto bytes = make_dds_header(4, 4, 1, 0x00000004u, 0x31545844u);
    bytes.resize(bytes.size() + 8);

    const auto result = arc::render::parse_dds_texture(bytes, "bc1.dds");

    INFO(result.message);
    REQUIRE(result.succeeded());
    REQUIRE(result.texture.dds);
    REQUIRE(result.texture.compressed);
    REQUIRE(result.texture.format == arc::render::texture_format::bc1_rgba_unorm);
    REQUIRE(result.texture.width == 4);
    REQUIRE(result.texture.height == 4);
    REQUIRE(result.texture.mips.size() == 1);
    REQUIRE(result.texture.mips[0].size == 8);
    REQUIRE(result.texture.encoded.size() == 8);
}

TEST_CASE("DDS loader parses uncompressed RGBA8 texture metadata")
{
    auto bytes = make_dds_header(2, 2, 1, 0x00000041u, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
    bytes.resize(bytes.size() + 16);

    const auto result = arc::render::parse_dds_texture(bytes, "rgba.dds");

    INFO(result.message);
    REQUIRE(result.succeeded());
    REQUIRE_FALSE(result.texture.compressed);
    REQUIRE(result.texture.format == arc::render::texture_format::rgba8_unorm);
    REQUIRE(result.texture.mips.size() == 1);
    REQUIRE(result.texture.mips[0].size == 16);
}

TEST_CASE("texture loader infers material texture color space from file names")
{
    auto bytes = make_dds_header(4, 4, 1, 0x00000004u, 0x31545844u);
    bytes.resize(bytes.size() + 8);

    const auto root = std::filesystem::temp_directory_path();
    const auto base_color_path = root / "MASTER_Stone_BaseColor.dds";
    const auto normal_path = root / "MASTER_Stone_Normal.dds";
    {
        std::ofstream file(base_color_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    {
        std::ofstream file(normal_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    const auto base_color = arc::render::load_texture_asset(base_color_path);
    const auto normal = arc::render::load_texture_asset(normal_path);

    REQUIRE(base_color.succeeded());
    REQUIRE(normal.succeeded());
    REQUIRE(base_color.texture.format == arc::render::texture_format::bc1_rgba_srgb);
    REQUIRE(normal.texture.format == arc::render::texture_format::bc1_rgba_unorm);

    std::filesystem::remove(base_color_path);
    std::filesystem::remove(normal_path);
}

TEST_CASE("texture loader prepares checked-in landscape maps for GPU upload")
{
    const auto path = std::filesystem::path(ARC_RENDER_TEST_ASSET_ROOT) / "textures" / "terrain" / "aerial_grass_rock" /
                      "aerial_grass_rock_diff_1k.jpg";
    const auto result = arc::render::load_texture_asset(path);

    INFO(result.message);
    REQUIRE(result.succeeded());
#if defined(ARC_RENDER_TEST_EXPECT_IMAGE_DECODER)
    REQUIRE(result.texture.width == 1024);
    REQUIRE(result.texture.height == 1024);
    REQUIRE(result.texture.has_pixels());
    REQUIRE(result.texture.mips.size() == 11);
    REQUIRE(result.texture.encoded.empty());
#else
    REQUIRE_FALSE(result.texture.encoded.empty());
#endif
}

TEST_CASE("DDS loader rejects invalid and truncated payloads")
{
    std::vector<std::byte> invalid(8);
    REQUIRE_FALSE(arc::render::parse_dds_texture(invalid, "bad.dds").succeeded());

    auto truncated = make_dds_header(8, 8, 1, 0x00000004u, 0x31545844u);
    truncated.resize(truncated.size() + 4);
    const auto result = arc::render::parse_dds_texture(truncated, "truncated.dds");
    REQUIRE_FALSE(result.succeeded());
}

TEST_CASE("scene asset loader wraps GLB meshes and reports scene import failures cleanly")
{
    const auto path = write_triangle_glb();
    const auto glb = arc::render::load_scene_asset(path);

    INFO(glb.message);
    REQUIRE(glb.succeeded());
    REQUIRE(glb.meshes.size() == 1);
    REQUIRE(glb.nodes.size() == 1);
    REQUIRE(glb.nodes[0].mesh_index == 0);
    REQUIRE(glb.materials.size() == 1);

    const auto fbx = arc::render::load_scene_asset(path.parent_path() / "missing.fbx");
    REQUIRE_FALSE(fbx.succeeded());
    REQUIRE_FALSE(fbx.message.empty());

    std::filesystem::remove(path);
}

#if defined(ARC_RENDER_TEST_UFBX_DATA_ROOT)
TEST_CASE("scene asset loader imports static FBX meshes with ufbx")
{
    const std::filesystem::path fixture =
        std::filesystem::path(ARC_RENDER_TEST_UFBX_DATA_ROOT) / "blender_279_nested_meshes_7400_binary.fbx";
    REQUIRE(std::filesystem::exists(fixture));

    const auto temp_root = std::filesystem::temp_directory_path() / "arc-render-fbx-import-test";
    std::error_code ec;
    std::filesystem::remove_all(temp_root, ec);
    std::filesystem::create_directories(temp_root, ec);

    arc::render::scene_import_options options;
    options.asset_root = temp_root;
    options.import_directory = temp_root / "imported" / "nested_meshes";

    std::vector<arc::render::scene_import_progress> progress;
    const auto result = arc::render::load_scene_asset(fixture, options,
                                                      [&](const arc::render::scene_import_progress& value)
                                                      {
                                                          progress.push_back(value);
                                                          return true;
                                                      });

    INFO(result.message);
    for (const auto& diagnostic : result.diagnostics)
        INFO(diagnostic);
    REQUIRE(result.succeeded());
    REQUIRE(result.meshes.size() >= 1);
    REQUIRE(result.nodes.size() >= 1);
    REQUIRE(result.nodes.front().mesh_index < result.meshes.size());
    REQUIRE(std::filesystem::exists(result.manifest_path));
    REQUIRE_FALSE(progress.empty());
    REQUIRE(progress.back().stage == arc::render::scene_import_stage::finalizing);

    std::filesystem::remove_all(temp_root, ec);
}

TEST_CASE("scene asset loader extracts FBX material assets and embedded textures")
{
    const std::filesystem::path fixture =
        std::filesystem::path(ARC_RENDER_TEST_UFBX_DATA_ROOT) / "blender_279_internal_textures_7400_binary.fbx";
    REQUIRE(std::filesystem::exists(fixture));

    const auto temp_root = std::filesystem::temp_directory_path() / "arc-render-fbx-texture-import-test";
    std::error_code ec;
    std::filesystem::remove_all(temp_root, ec);
    std::filesystem::create_directories(temp_root, ec);

    arc::render::scene_import_options options;
    options.asset_root = temp_root;
    options.import_directory = temp_root / "imported" / "internal_textures";

    const auto result = arc::render::load_scene_asset(fixture, options);

    INFO(result.message);
    for (const auto& diagnostic : result.diagnostics)
        INFO(diagnostic);
    REQUIRE(result.succeeded());
    REQUIRE_FALSE(result.materials.empty());
    REQUIRE_FALSE(result.textures.empty());
    REQUIRE(std::filesystem::exists(result.manifest_path));
    REQUIRE(std::filesystem::exists(result.materials.front().asset_path));
    REQUIRE_FALSE(result.textures.front().source_path.empty());
    REQUIRE(std::filesystem::exists(temp_root / result.textures.front().source_path));

    std::filesystem::remove_all(temp_root, ec);
}
#endif

TEST_CASE("primitive mesh builders create renderable geometry")
{
    const auto plane = arc::render::make_plane_mesh(2.0f);
    REQUIRE(plane.name == "Plane");
    REQUIRE(plane.vertices.size() == 4);
    REQUIRE(plane.indices == std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3});
    REQUIRE(plane.vertices[0].normal[1] == Catch::Approx(1.0f));

    const auto cube = arc::render::make_cube_mesh();
    REQUIRE(cube.vertices.size() == 24);
    REQUIRE(cube.indices.size() == 36);

    const auto sphere = arc::render::make_uv_sphere_mesh(0.5f, 8, 4);
    REQUIRE(sphere.vertices.size() == 45);
    REQUIRE(sphere.indices.size() == 8 * 4 * 6);

    const auto cylinder = arc::render::make_cylinder_mesh(0.5f, 1.0f, 8);
    REQUIRE(cylinder.vertices.size() == 20);
    REQUIRE(cylinder.indices.size() == 8 * 12);

    const auto cone = arc::render::make_cone_mesh(0.5f, 1.0f, 8);
    REQUIRE(cone.name == "Cone");
    REQUIRE(cone.vertices.size() == 28);
    REQUIRE(cone.indices.size() == 8 * 6);

    const auto capsule = arc::render::make_capsule_mesh(0.5f, 1.0f, 8, 4);
    REQUIRE(capsule.name == "Capsule");
    REQUIRE(capsule.vertices.size() == 90);
    REQUIRE(capsule.indices.size() == 9 * 8 * 6);

    const auto terrain = arc::render::make_terrain_grid_mesh(8.0f, 8, 1.0f);
    REQUIRE(terrain.name == "Terrain");
    REQUIRE(terrain.vertices.size() == 81);
    REQUIRE(terrain.indices.size() == 8 * 8 * 6);
    bool has_height_variation = false;
    bool has_tilted_normal = false;
    bool has_color_variation = false;
    for (const auto& vertex : terrain.vertices)
    {
        has_height_variation = has_height_variation || std::abs(vertex.position[1]) > 0.01f;
        has_tilted_normal = has_tilted_normal || vertex.normal[1] < 0.995f;
        has_color_variation = has_color_variation ||
                              std::abs(vertex.color[0] - terrain.vertices.front().color[0]) > 0.01f ||
                              std::abs(vertex.color[1] - terrain.vertices.front().color[1]) > 0.01f ||
                              std::abs(vertex.color[2] - terrain.vertices.front().color[2]) > 0.01f;
    }
    REQUIRE(has_height_variation);
    REQUIRE(has_tilted_normal);
    REQUIRE(has_color_variation);
    const auto& center = terrain.vertices[4 * 9 + 4];
    REQUIRE(center.position[1] == Catch::Approx(arc::render::sample_terrain_height(0.0f, 0.0f, 8.0f, 1.0f)));
    REQUIRE(terrain.vertices.back().texcoord[0] - terrain.vertices.front().texcoord[0] > 1.0f);
}

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

TEST_CASE("HZB min max reduction is conservative for odd extents")
{
    using namespace arc::render;
    REQUIRE(hzb_mip_count(1, 1) == 1);
    REQUIRE(hzb_mip_count(7, 3) == 3);
    REQUIRE(hzb_mip_count(1920, 1080) == 11);

    const auto reduced =
        reduce_hzb_depth(reduce_hzb_depth({0.2f, 0.8f}, {0.1f, 0.7f}), reduce_hzb_depth({0.4f, 0.9f}, {0.3f, 0.6f}));
    REQUIRE(reduced.nearest == Catch::Approx(0.1f));
    REQUIRE(reduced.farthest == Catch::Approx(0.9f));
}

TEST_CASE("anti aliasing policy resolves path aware auto and explicit fallbacks")
{
    using namespace arc::render;
    render_capabilities capabilities{};
    capabilities.fxaa = true;
    capabilities.temporal_resolve = true;
    capabilities.temporal_upscale = true;

    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::auto_select, render_path::forward_plus, 1.0f, capabilities) ==
            anti_aliasing_method::fxaa);
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::auto_select, render_path::deferred, 1.0f, capabilities) ==
            anti_aliasing_method::taa);
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::auto_select, render_path::deferred, 0.75f, capabilities) ==
            anti_aliasing_method::taau);
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::disabled, render_path::deferred, 0.5f, capabilities) ==
            anti_aliasing_method::disabled);

    capabilities.temporal_upscale = false;
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::taau, render_path::deferred, 0.5f, capabilities) ==
            anti_aliasing_method::taa);
    capabilities.temporal_resolve = false;
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::taa, render_path::deferred, 1.0f, capabilities) ==
            anti_aliasing_method::fxaa);
    capabilities.fxaa = false;
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::auto_select, render_path::deferred, 1.0f, capabilities) ==
            anti_aliasing_method::disabled);
}

TEST_CASE("terrain hierarchy is deterministic monotonic and incrementally updated")
{
    constexpr std::uint32_t resolution = 65u;
    std::vector<float> heights(static_cast<std::size_t>(resolution) * resolution);
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x)
            heights[static_cast<std::size_t>(z) * resolution + x] =
                std::sin(static_cast<float>(x) * 0.11f) * std::cos(static_cast<float>(z) * 0.07f);
    const auto first = arc::render::build_terrain_hierarchy(heights, resolution, 64.0f, 64.0f, {.patch_quads = 16u});
    auto second = arc::render::build_terrain_hierarchy(heights, resolution, 64.0f, 64.0f, {.patch_quads = 16u});
    REQUIRE(first.nodes.size() == second.nodes.size());
    REQUIRE(first.leaf_count == 16u);
    REQUIRE(first.nodes[first.root].geometric_error == Catch::Approx(second.nodes[second.root].geometric_error));
    for (const auto& node : first.nodes)
        for (const auto child : node.children)
            if (child != arc::render::invalid_terrain_node)
                REQUIRE(node.geometric_error >= first.nodes[child].geometric_error);

    const auto root_before = second.nodes[second.root].maximum_height;
    heights[32u * resolution + 32u] += 20.0f;
    REQUIRE(arc::render::update_terrain_hierarchy(second, heights, resolution, 64.0f, 64.0f, {31u, 31u, 33u, 33u},
                                                  {.patch_quads = 16u}));
    REQUIRE(second.nodes[second.root].maximum_height > root_before);
}

TEST_CASE("terrain stitched topology variants remain valid and deterministic")
{
    for (std::uint8_t mask = 0u; mask < 16u; ++mask)
    {
        const auto first = arc::render::make_terrain_patch_indices(32u, mask);
        const auto second = arc::render::make_terrain_patch_indices(32u, mask);
        REQUIRE(first == second);
        REQUIRE_FALSE(first.empty());
        REQUIRE(first.size() % 3u == 0u);
        for (const auto index : first)
            REQUIRE(index < 33u * 33u);
        for (std::size_t triangle = 0; triangle < first.size(); triangle += 3u)
        {
            REQUIRE(first[triangle] != first[triangle + 1u]);
            REQUIRE(first[triangle + 1u] != first[triangle + 2u]);
            REQUIRE(first[triangle] != first[triangle + 2u]);
        }
    }
}

TEST_CASE("terrain selection responds to projected error and balances neighboring LODs")
{
    constexpr std::uint32_t resolution = 129u;
    std::vector<float> heights(static_cast<std::size_t>(resolution) * resolution);
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x)
            heights[static_cast<std::size_t>(z) * resolution + x] =
                5.0f * std::sin(static_cast<float>(x) * 0.17f) * std::cos(static_cast<float>(z) * 0.13f);
    const auto hierarchy =
        arc::render::build_terrain_hierarchy(heights, resolution, 128.0f, 128.0f, {.patch_quads = 16u});
    arc::render::render_camera camera;
    camera.position = {0.0f, 5.0f, 96.0f};
    camera.render_width = 1920u;
    camera.render_height = 1080u;
    const float near_plane = 0.1f;
    const float far_plane = 500.0f;
    const float inverse_tangent = 1.0f / std::tan(arc::math::to_radians(60.0f) * 0.5f);
    camera.projection = {};
    camera.projection(0, 0) = inverse_tangent / (16.0f / 9.0f);
    camera.projection(1, 1) = inverse_tangent;
    camera.projection(2, 2) = far_plane / (near_plane - far_plane);
    camera.projection(2, 3) = far_plane * near_plane / (near_plane - far_plane);
    camera.projection(3, 2) = -1.0f;
    camera.view = arc::math::identity<float, 4>();
    camera.view(0, 3) = -camera.position[0];
    camera.view(1, 3) = -camera.position[1];
    camera.view(2, 3) = -camera.position[2];
    camera.view_projection = arc::math::matmul(camera.projection, camera.view);
    arc::render::terrain_selection_scratch scratch;
    const auto detailed = arc::render::select_terrain_patches({1u, 1u}, hierarchy, arc::math::identity<float, 4>(),
                                                              camera, 0.25f, 1.0f, &scratch);
    const auto coarse =
        arc::render::select_terrain_patches({1u, 1u}, hierarchy, arc::math::identity<float, 4>(), camera, 10000.0f);
    REQUIRE(detailed.patches.size() > coarse.patches.size());
    REQUIRE(detailed.statistics.rendered_triangles > coarse.statistics.rendered_triangles);
    for (std::size_t a = 0; a < detailed.patches.size(); ++a)
        for (std::size_t b = a + 1u; b < detailed.patches.size(); ++b)
        {
            const auto& left = detailed.patches[a];
            const auto& right = detailed.patches[b];
            const bool vertical =
                (left.samples.max_x == right.samples.min_x || right.samples.max_x == left.samples.min_x) &&
                left.samples.min_z < right.samples.max_z && right.samples.min_z < left.samples.max_z;
            const bool horizontal =
                (left.samples.max_z == right.samples.min_z || right.samples.max_z == left.samples.min_z) &&
                left.samples.min_x < right.samples.max_x && right.samples.min_x < left.samples.max_x;
            if (vertical || horizontal)
                REQUIRE(std::abs(static_cast<int>(left.lod) - static_cast<int>(right.lod)) <= 1);
        }
}

TEST_CASE("terrain renderer resources preserve weight-only hierarchy and emit partial events")
{
    arc::render::renderer renderer;
    arc::render::terrain_resource_descriptor descriptor;
    descriptor.sample_resolution = 33u;
    descriptor.width = 32.0f;
    descriptor.depth = 32.0f;
    descriptor.heights.resize(33u * 33u);
    descriptor.weights.resize(33u * 33u, {255u, 0u, 0u, 0u});
    descriptor.lod.patch_quads = 16u;
    const auto terrain = renderer.create_terrain(std::move(descriptor));
    REQUIRE(terrain.valid());
    const auto before = renderer.terrain_snapshot(terrain);
    (void)renderer.frame_queue().commit(1u);

    arc::render::terrain_weight_region_update weights;
    weights.region = {4u, 5u, 7u, 8u};
    weights.row_stride = weights.region.width();
    weights.values.resize(static_cast<std::size_t>(weights.row_stride) * weights.region.height(), {0u, 255u, 0u, 0u});
    weights.content_revision = 2u;
    REQUIRE(renderer.update_terrain_weights(terrain, std::move(weights)));
    const auto after = renderer.terrain_snapshot(terrain);
    REQUIRE(after.hierarchy_nodes == before.hierarchy_nodes);
    REQUIRE(after.uploaded_height_bytes == before.uploaded_height_bytes);
    REQUIRE(after.uploaded_weight_bytes == before.uploaded_weight_bytes + 4u * 4u * 4u);
    const auto packet = renderer.frame_queue().commit(2u);
    REQUIRE(packet.events.size() == 1u);
    REQUIRE(packet.events.front().type() == arc::render::render_event_type::terrain_weight_update);
    REQUIRE(renderer.destroy_terrain(terrain));
}

TEST_CASE("lighting scene emits precise incremental updates and rejects stale world generations")
{
    using namespace arc::render;
    lighting_scene scene;
    lighting_scene_instance instance{.stable_id = 42,
                                     .geometry = {2, 1},
                                     .material = {3, 1},
                                     .world_bounds = {{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}},
                                     .transform_revision = 1,
                                     .material_revision = 1};
    auto update = scene.synchronize(7, 1, 1, std::span(&instance, 1));
    REQUIRE(update.updates.size() == 1);
    REQUIRE(update.updates.front().kind == lighting_scene_update_kind::upsert);
    REQUIRE(update.updates.front().geometry_dirty);

    update = scene.synchronize(7, 1, 2, std::span(&instance, 1));
    REQUIRE(update.updates.empty());

    instance.transform_revision = 2;
    update = scene.synchronize(7, 1, 3, std::span(&instance, 1));
    REQUIRE(update.updates.size() == 1);
    REQUIRE(update.updates.front().transform_dirty);
    REQUIRE_FALSE(update.updates.front().material_dirty);

    update = scene.synchronize(7, 2, 4, std::span(&instance, 1));
    REQUIRE(update.updates.front().kind == lighting_scene_update_kind::reset);
    REQUIRE(scene.snapshot().world_epoch == 2);

    update = scene.synchronize(7, 2, 5, {});
    REQUIRE(update.updates.size() == 1);
    REQUIRE(update.updates.front().kind == lighting_scene_update_kind::destroy);
}

TEST_CASE("dynamic indirect lighting graph selects the resolved screen software and hardware hierarchy")
{
    using namespace arc::render;
    renderer_config renderer_settings;
    renderer_settings.quality = render_quality_tier::ultra;
    render_capabilities capabilities;
    capabilities.compute_shaders = true;
    capabilities.storage_buffers = true;
    capabilities.storage_images = true;
    capabilities.hzb_occlusion = true;
    capabilities.temporal_resolve = true;
    capabilities.screen_space_indirect_lighting = true;
    capabilities.surface_cache = true;
    capabilities.radiance_cache = true;
    capabilities.software_ray_tracing = true;
    capabilities.hardware_ray_query = true;
    capabilities.ray_tracing = true;
    const auto config = resolve_render_config(renderer_settings, capabilities);
    REQUIRE(config.indirect_lighting_path == lighting_trace_path::hybrid_hardware);

    world_environment_data environment;
    environment.enabled = true;
    environment.indirect_lighting.enabled = true;
    environment.indirect_lighting.method = indirect_lighting_method::auto_select;
    const auto graph = make_scene_draw_graph("gi-test", config, false, environment).compile().value();
    const auto contains = [&](builtin_render_pass pass)
    {
        return std::ranges::any_of(graph.passes,
                                   [pass](const compiled_render_pass& candidate) { return candidate.builtin == pass; });
    };
    REQUIRE(contains(builtin_render_pass::screen_space_gi));
    REQUIRE(contains(builtin_render_pass::software_gi_trace));
    REQUIRE(contains(builtin_render_pass::hardware_gi_trace));
    REQUIRE(contains(builtin_render_pass::screen_space_reflections));
    REQUIRE(contains(builtin_render_pass::software_reflections));
    REQUIRE(contains(builtin_render_pass::hardware_reflections));
    REQUIRE(contains(builtin_render_pass::indirect_lighting_temporal));
    REQUIRE(contains(builtin_render_pass::reflection_temporal));
    REQUIRE(contains(builtin_render_pass::indirect_lighting_composite));
}

TEST_CASE("OBJ scene import triangulates polygons and supports negative indices", "[render][mesh][obj]")
{
    const auto path = std::filesystem::temp_directory_path() / "arc-render-obj-import-test.obj";
    {
        std::ofstream output(path, std::ios::trunc);
        REQUIRE(output.good());
        output << "v 0 0 0\n"
               << "v 1 0 0\n"
               << "v 1 1 0\n"
               << "v 0 1 0\n"
               << "vt 0 0\n"
               << "vt 1 0\n"
               << "vt 1 1\n"
               << "vt 0 1\n"
               << "f -4/-4 -3/-3 -2/-2 -1/-1\n";
    }

    arc::render::scene_import_options options;
    const auto imported = arc::render::load_scene_asset(path, options);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    REQUIRE(imported.succeeded());
    REQUIRE(imported.meshes.size() == 1);
    REQUIRE(imported.nodes.size() == 1);
    CHECK(imported.meshes.front().indices.size() == 6);
    CHECK(imported.meshes.front().vertices.size() == 6);
    CHECK(imported.nodes.front().mesh_index == 0);
}
