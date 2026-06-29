#include <arc/render/render.h>

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
    const std::size_t image_offset = bin.size();
    for (const std::byte value : { std::byte{ 0x89 }, std::byte{ 0x50 }, std::byte{ 0x4e }, std::byte{ 0x47 } })
        bin.push_back(value);
    pad4(bin, std::byte{ 0 });

    const std::string json =
        "{\"asset\":{\"version\":\"2.0\"},"
        "\"buffers\":[{\"byteLength\":" + std::to_string(bin.size()) + "}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(position_offset) + ",\"byteLength\":36},"
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(normal_offset) + ",\"byteLength\":36},"
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(uv_offset) + ",\"byteLength\":24},"
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(index_offset) + ",\"byteLength\":6},"
        "{\"buffer\":0,\"byteOffset\":" + std::to_string(image_offset) + ",\"byteLength\":4}],"
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
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2},\"indices\":3,\"material\":0}]}]}";

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
    arc::render::texture_handle texture{ .index = 5, .generation = 1 };
    arc::render::material_handle material{ .index = 6, .generation = 1 };
    auto mesh_data = std::make_shared<arc::render::mesh_data>();
    mesh_data->name = "triangle";
    auto texture_data = std::make_shared<arc::render::texture_data>();
    texture_data->name = "white";
    auto material_data = std::make_shared<arc::render::material_desc>();
    material_data->name = "default";

    writer.mesh_upload(mesh, mesh_data, "triangle");
    writer.texture_upload(texture, texture_data, "white");
    writer.material_upload(material, material_data, "default");
    writer.draw_mesh(
        mesh,
        material,
        arc::math::identity<float, 4>(),
        arc::math::identity<float, 4>(),
        arc::render::render_mode::wireframe,
        arc::render::mesh_visualization_mode::world_normal,
        true,
        arc::math::vector4f{ 1.0f, 0.5f, 0.0f, 1.0f },
        "triangle");
    writer.directional_light({ 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 3.0f, true, "Sun");

    REQUIRE(buffer.events().size() == 5);
    REQUIRE(buffer.events()[0].type() == arc::render::render_event_type::mesh_upload);
    const auto& upload = std::get<arc::render::mesh_upload_event>(buffer.events()[0].payload);
    REQUIRE(upload.handle == mesh);
    REQUIRE(upload.mesh == mesh_data);
    REQUIRE(buffer.events()[1].type() == arc::render::render_event_type::texture_upload);
    REQUIRE(std::get<arc::render::texture_upload_event>(buffer.events()[1].payload).texture == texture_data);
    REQUIRE(buffer.events()[2].type() == arc::render::render_event_type::material_upload);
    REQUIRE(std::get<arc::render::material_upload_event>(buffer.events()[2].payload).material == material_data);
    REQUIRE(buffer.events()[3].type() == arc::render::render_event_type::draw);
    const auto& draw = std::get<arc::render::draw_mesh_event>(buffer.events()[3].payload);
    REQUIRE(draw.mesh == mesh);
    REQUIRE(draw.material == material);
    REQUIRE(draw.mode == arc::render::render_mode::wireframe);
    REQUIRE(draw.visualization == arc::render::mesh_visualization_mode::world_normal);
    REQUIRE(draw.selected);
    REQUIRE(draw.label == "triangle");
    REQUIRE(buffer.events()[4].type() == arc::render::render_event_type::directional_light);
    const auto& light = std::get<arc::render::directional_light_event>(buffer.events()[4].payload);
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

TEST_CASE("render graph compiles typed resources and transitions")
{
    arc::render::render_graph graph;
    graph.add_resource({
        .name = "viewport",
        .kind = arc::render::render_resource_kind::color_texture,
        .extent = { .width = 1280, .height = 720 },
        .format = "rgba8",
        .persistent = true
    });

    graph.add_pass({
        .name = "viewport clear",
        .kind = arc::render::render_pass_kind::clear,
        .writes = { {
            .resource = "viewport",
            .kind = arc::render::render_resource_kind::color_texture,
            .usage = arc::render::render_resource_usage::color_attachment,
            .write = true,
            .load_op = arc::render::render_load_op::clear
        } }
    });
    graph.add_pass({
        .name = "imgui sample",
        .kind = arc::render::render_pass_kind::imgui,
        .reads = { {
            .resource = "viewport",
            .kind = arc::render::render_resource_kind::color_texture,
            .usage = arc::render::render_resource_usage::sampled
        } }
    });

    const auto compiled = graph.compile();
    REQUIRE(compiled.resources.size() == 1);
    REQUIRE(compiled.resources[0].name == "viewport");
    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.passes[0].writes[0].usage == arc::render::render_resource_usage::color_attachment);
    REQUIRE(compiled.transitions.size() == 1);
    REQUIRE(compiled.transitions[0].resource == "viewport");
    REQUIRE(compiled.transitions[0].before == arc::render::render_resource_usage::color_attachment);
    REQUIRE(compiled.transitions[0].after == arc::render::render_resource_usage::sampled);
}

TEST_CASE("clear present graph declares the bring-up passes")
{
    const auto graph = arc::render::make_clear_present_graph("viewport");
    const auto compiled = graph.compile();

    REQUIRE(compiled.passes.size() == 2);
    REQUIRE(compiled.resources.size() == 1);
    REQUIRE(compiled.passes[0].kind == arc::render::render_pass_kind::clear);
    REQUIRE(compiled.passes[1].kind == arc::render::render_pass_kind::present);
    REQUIRE_FALSE(compiled.transitions.empty());
}

TEST_CASE("scene draw graph declares modern viewport pass order")
{
    const auto graph = arc::render::make_scene_draw_graph("viewport");
    const auto compiled = graph.compile();

    REQUIRE(compiled.passes.size() == 6);
    REQUIRE(compiled.passes[0].name == "depth prepass");
    REQUIRE(compiled.passes[1].name == "gbuffer pass");
    REQUIRE(compiled.passes[2].name == "forward transparent pass");
    REQUIRE(compiled.passes[3].name == "editor picking pass");
    REQUIRE(compiled.passes[4].name == "selection outline pass");
    REQUIRE(compiled.passes[5].name == "present viewport");
    REQUIRE(compiled.resources.size() == 6);
    REQUIRE_FALSE(compiled.transitions.empty());
}

TEST_CASE("render world preparation culls sorts batches and emits indirect commands")
{
    arc::render::render_world_packet packet;
    packet.camera.view_projection = arc::math::identity<float, 4>();
    packet.items.push_back({
        .mesh = { .index = 1, .generation = 1 },
        .material = { .index = 2, .generation = 1 },
        .world_bounds = arc::geometric::box3f{
            arc::geometric::point3f{ -0.5f, -0.5f, -0.5f },
            arc::geometric::point3f{ 0.5f, 0.5f, 0.5f } },
        .label = "A"
    });
    packet.items.push_back({
        .mesh = { .index = 1, .generation = 1 },
        .material = { .index = 2, .generation = 1 },
        .world_bounds = arc::geometric::box3f{
            arc::geometric::point3f{ -0.25f, -0.25f, -0.25f },
            arc::geometric::point3f{ 0.25f, 0.25f, 0.25f } },
        .label = "B"
    });
    packet.items.push_back({
        .mesh = { .index = 5, .generation = 1 },
        .material = { .index = 7, .generation = 1 },
        .world_bounds = arc::geometric::box3f{
            arc::geometric::point3f{ 4.0f, 4.0f, 4.0f },
            arc::geometric::point3f{ 5.0f, 5.0f, 5.0f } },
        .label = "culled"
    });

    arc::render::prepare_render_world(packet);

    REQUIRE(packet.visible_items.size() == 2);
    REQUIRE(packet.culled_item_count == 1);
    REQUIRE(packet.instance_batches.size() == 1);
    REQUIRE(packet.instance_batches[0].item_count == 2);
    REQUIRE(packet.indirect_draws.size() == 1);
    REQUIRE(packet.indirect_draws[0].instance_count == 2);
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

TEST_CASE("renderer creates texture and material resources")
{
    arc::render::renderer renderer;
    arc::render::texture_data texture;
    texture.name = "encoded";
    texture.mime_type = "image/png";
    texture.encoded = { std::byte{ 1 }, std::byte{ 2 } };

    const auto texture_handle = renderer.create_texture(std::move(texture));
    REQUIRE(renderer.texture_alive(texture_handle));

    arc::render::material_desc material;
    material.name = "pbr";
    material.base_color_texture = texture_handle;
    material.metallic = 0.25f;
    material.roughness = 0.8f;
    material.alpha_mode = arc::render::material_alpha_mode::masked;

    const auto material_handle = renderer.create_material(material);
    REQUIRE(renderer.material_alive(material_handle));

    const auto packet = renderer.frame_queue().commit(1);
    REQUIRE(packet.events.size() == 2);
    REQUIRE(packet.events[0].type() == arc::render::render_event_type::texture_upload);
    REQUIRE(packet.events[1].type() == arc::render::render_event_type::material_upload);
    const auto& uploaded = std::get<arc::render::material_upload_event>(packet.events[1].payload);
    REQUIRE(uploaded.handle == material_handle);
    REQUIRE(uploaded.material->handle == material_handle);
    REQUIRE(uploaded.material->base_color_texture == texture_handle);
    REQUIRE(uploaded.material->alpha_mode == arc::render::material_alpha_mode::masked);
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

TEST_CASE("pipeline handle cache reuses equivalent keys")
{
    arc::render::pipeline_handle_cache cache;
    arc::render::graphics_pipeline_key key{
        .vertex_shader = { .index = 1, .generation = 1 },
        .fragment_shader = { .index = 2, .generation = 1 },
        .vertex_layout = "pnu",
        .color_format = "rgba16f",
        .depth_format = "d32",
        .depth_test = true,
        .depth_write = true
    };
    arc::render::pipeline_handle pipeline{ .index = 9, .generation = 3 };

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
    arc::render::material_desc material;
    material.alpha_mode = arc::render::material_alpha_mode::blend;
    material.normal_texture = { .index = 1, .generation = 1 };
    material.emissive_texture = { .index = 2, .generation = 1 };
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
        return {
            .succeeded = true,
            .diagnostics = request.source_path,
            .bytecode = { std::uint8_t(count) },
            .reflection = { .entry_points = { request.entry_point } }
        };
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
    arc::render::shader_compile_request request{
        .source_path = path.string(),
        .entry_point = "main",
        .profile = "fragment",
        .target = arc::render::shader_target::spirv
    };

    const auto first = cache.compile_or_get(compiler, request);
    const auto second = cache.compile_or_get(compiler, request);

    REQUIRE(first.succeeded);
    REQUIRE(second.succeeded);
    REQUIRE(compiler.count == 1);
    REQUIRE(cache.size() == 1);
    REQUIRE_FALSE(cache.source_changed(request));

    std::filesystem::remove(path);
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
