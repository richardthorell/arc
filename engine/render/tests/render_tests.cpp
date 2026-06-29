#include <arc/render.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <array>
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

    arc::render::render_submit_result submit(
        const arc::render::render_frame_packet& packet,
        const arc::render::compiled_render_graph& graph) override
    {
        last_frame = packet.frame_index;
        last_event_count = packet.events.size();
        last_pass_count = graph.passes.size();
        return { .submitted = true, .message = "submitted" };
    }

    void resize_viewport(std::uint32_t width, std::uint32_t height) override
    {
        viewport_width = width;
        viewport_height = height;
    }

    arc::render::render_viewport_texture viewport_texture() const noexcept override
    {
        return { .id = texture_id, .width = viewport_width, .height = viewport_height };
    }

    arc::render::render_capabilities capabilities_{};
    std::uint64_t last_frame{};
    std::size_t last_event_count{};
    std::size_t last_pass_count{};
    std::uint64_t texture_id{ 99 };
    std::uint32_t viewport_width{};
    std::uint32_t viewport_height{};
};

void append_u32(std::vector<std::byte>& bytes, std::uint32_t value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
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
    for (const float value : { 0.0f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f })
        append_f32(bin, value);
    const std::size_t normal_offset = bin.size();
    for (int index = 0; index < 3; ++index)
    {
        append_f32(bin, 0.0f);
        append_f32(bin, 0.0f);
        append_f32(bin, 1.0f);
    }
    const std::size_t uv_offset = bin.size();
    for (const float value : { 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f })
        append_f32(bin, value);
    const std::size_t index_offset = bin.size();
    append_u16(bin, 0);
    append_u16(bin, 1);
    append_u16(bin, 2);
    pad4(bin, std::byte{ 0 });

    const std::string json =
        "{\"asset\":{\"version\":\"2.0\"},"
        "\"buffers\":[{\"byteLength\":" + std::to_string(bin.size()) + "}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(position_offset) + ",\"byteLength\":36},"
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(normal_offset) + ",\"byteLength\":36},"
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(uv_offset) + ",\"byteLength\":24},"
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(index_offset) + ",\"byteLength\":6}],"
        "\"accessors\":["
        "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
        "{\"bufferView\":1,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
        "{\"bufferView\":2,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
        "{\"bufferView\":3,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2},\"indices\":3}]}]}";

    std::vector<std::byte> json_bytes(reinterpret_cast<const std::byte*>(json.data()), reinterpret_cast<const std::byte*>(json.data() + json.size()));
    pad4(json_bytes, std::byte{ ' ' });

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
    arc::render::mesh_handle mesh{ .index = 4, .generation = 2 };
    auto mesh_data = std::make_shared<arc::render::mesh_data>();
    mesh_data->name = "triangle";

    writer.mesh_upload(mesh, mesh_data, "triangle");
    writer.draw_mesh(
        mesh,
        {},
        arc::math::identity<float, 4>(),
        arc::math::identity<float, 4>(),
        arc::render::render_mode::wireframe,
        arc::render::mesh_visualization_mode::world_normal,
        true,
        arc::math::vector4f{ 1.0f, 0.5f, 0.0f, 1.0f },
        "triangle");
    writer.directional_light({ 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 3.0f, true, "Sun");

    REQUIRE(buffer.events().size() == 3);
    REQUIRE(buffer.events()[0].type() == arc::render::render_event_type::mesh_upload);
    const auto& upload = std::get<arc::render::mesh_upload_event>(buffer.events()[0].payload);
    REQUIRE(upload.handle == mesh);
    REQUIRE(upload.mesh == mesh_data);
    REQUIRE(buffer.events()[1].type() == arc::render::render_event_type::draw);
    const auto& draw = std::get<arc::render::draw_mesh_event>(buffer.events()[1].payload);
    REQUIRE(draw.mesh == mesh);
    REQUIRE(draw.mode == arc::render::render_mode::wireframe);
    REQUIRE(draw.visualization == arc::render::mesh_visualization_mode::world_normal);
    REQUIRE(draw.selected);
    REQUIRE(draw.label == "triangle");
    REQUIRE(buffer.events()[2].type() == arc::render::render_event_type::directional_light);
    const auto& light = std::get<arc::render::directional_light_event>(buffer.events()[2].payload);
    REQUIRE(light.label == "Sun");
    REQUIRE(light.intensity == Catch::Approx(3.0f));
}

TEST_CASE("render frame queue accepts producer buffers from multiple threads")
{
    arc::render::render_frame_queue queue;
    std::atomic<int> ready{ 0 };
    std::vector<std::thread> threads;

    for (int index = 0; index < 4; ++index)
    {
        threads.emplace_back([&, index]() {
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
    graph.add_pass({
        .name = "present",
        .kind = arc::render::render_pass_kind::present,
        .reads = { { .resource = "backbuffer" } }
    });
    graph.add_pass({
        .name = "clear",
        .kind = arc::render::render_pass_kind::clear,
        .writes = { { .resource = "backbuffer", .write = true } }
    });

    const auto compiled = graph.compile();
    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.passes[0].name == "clear");
    REQUIRE(compiled.passes[1].name == "present");
}

TEST_CASE("clear present graph declares the bring-up passes")
{
    const auto graph = arc::render::make_clear_present_graph("viewport");
    const auto compiled = graph.compile();

    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.passes[0].kind == arc::render::render_pass_kind::clear);
    REQUIRE(compiled.passes[1].kind == arc::render::render_pass_kind::present);
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

    REQUIRE(result.submitted);
    REQUIRE(backend_ptr->last_frame == 42);
    REQUIRE(backend_ptr->last_event_count == 1);
    REQUIRE(backend_ptr->last_pass_count == 2);
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
    REQUIRE(result.submitted);
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
    mesh.indices = { 0, 1, 2 };

    const auto handle = renderer.create_mesh(std::move(mesh));
    REQUIRE(renderer.mesh_alive(handle));

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 1);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::mesh_upload);
    const auto& upload = std::get<arc::render::mesh_upload_event>(packet.events[0].payload);
    REQUIRE(upload.handle == handle);
    REQUIRE(upload.mesh->vertices.size() == 3);
    REQUIRE(upload.mesh->indices.size() == 3);
}

TEST_CASE("GLB mesh loader reads static triangle geometry")
{
    const auto path = write_triangle_glb();
    const auto result = arc::render::load_gltf_mesh(path);

    REQUIRE(result.succeeded());
    REQUIRE(result.mesh.vertices.size() == 3);
    REQUIRE(result.mesh.indices == std::vector<std::uint32_t>{ 0, 1, 2 });
    REQUIRE(result.mesh.vertices[0].position[1] == 0.5f);
    REQUIRE(result.mesh.vertices[0].normal[2] == 1.0f);
    REQUIRE(result.mesh.vertices[1].texcoord[1] == 1.0f);

    std::filesystem::remove(path);
}

TEST_CASE("GLB mesh loader reads checked-in editor startup mesh")
{
    const std::filesystem::path path = std::filesystem::path(ARC_RENDER_TEST_ASSET_ROOT) / "models" / "UAL2_Standard.glb";
    REQUIRE(std::filesystem::exists(path));

    const auto result = arc::render::load_gltf_mesh(path);

    INFO(result.message);
    REQUIRE(result.succeeded());
    REQUIRE_FALSE(result.mesh.name.empty());
    REQUIRE_FALSE(result.mesh.vertices.empty());
    REQUIRE_FALSE(result.mesh.indices.empty());
}
