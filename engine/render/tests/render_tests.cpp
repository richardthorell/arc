#include <arc/render/render.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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

    arc::render::render_submit_result submit(
        const arc::render::render_frame_packet& packet,
        const arc::render::compiled_render_graph& graph) override
    {
        last_frame = packet.frame_index;
        last_event_count = packet.events.size();
        last_pass_count = graph.passes.size();
        profile.frame_index = packet.frame_index;
        profile.graph = graph;
        profile.summary = "recorded";
        profile.clustered_lights = {
            .tile_size_pixels = 32,
            .tiles_x = 2,
            .tiles_y = 3,
            .depth_slices = 16,
            .cluster_count = 96,
            .point_light_references = 4,
            .spot_light_references = 2,
            .overflow_count = 1,
            .available = true
        };
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

    arc::render::render_backend_frame_profile last_frame_profile() const override
    {
        return profile;
    }

    void request_object_pick(arc::render::render_object_pick_request request) override
    {
        pick_request = request;
        pick_requested = true;
    }

    arc::render::render_capabilities capabilities_{};
    arc::render::resolved_render_config configured{};
    arc::render::render_backend_frame_profile profile{};
    arc::render::render_object_pick_request pick_request{};
    std::uint64_t last_frame{};
    std::size_t last_event_count{};
    std::size_t last_pass_count{};
    std::uint64_t texture_id{ 99 };
    std::uint32_t viewport_width{};
    std::uint32_t viewport_height{};
    bool pick_requested{};
};

class recording_command_encoder final : public arc::render::command_encoder
{
public:
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
    std::size_t ended_passes{};
};

void count_recorded_pass(arc::render::command_encoder&, void* user_data)
{
    ++*static_cast<std::uint32_t*>(user_data);
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

std::vector<std::byte> make_dds_header(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t mip_count,
    std::uint32_t pixel_flags,
    std::uint32_t four_cc,
    std::uint32_t rgb_bit_count = 0,
    std::uint32_t r_mask = 0,
    std::uint32_t g_mask = 0,
    std::uint32_t b_mask = 0,
    std::uint32_t a_mask = 0)
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
    writer.draw_mesh_tinted(
        mesh,
        material,
        arc::math::identity<float, 4>(),
        arc::math::identity<float, 4>(),
        arc::render::render_mode::shaded,
        arc::render::mesh_visualization_mode::standard,
        false,
        arc::math::vector4f{ 0.25f, 0.5f, 0.75f, 1.0f },
        arc::math::vector4f{ 1.0f, 1.0f, 1.0f, 1.0f },
        "tinted");
    writer.directional_light({ 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 3.0f, true, "Sun");

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
    const auto backbuffer = graph.add_resource({
        .name = "backbuffer",
        .kind = arc::render::render_resource_kind::color_texture,
        .format = arc::render::render_format::rgba8_unorm
    });
    graph.add_pass({
        .name = "clear",
        .kind = arc::render::render_pass_kind::clear,
        .writes = { { .handle = backbuffer, .kind = arc::render::render_resource_kind::color_texture,
            .usage = arc::render::render_resource_usage::color_attachment, .write = true } }
    });
    graph.add_pass({
        .name = "present",
        .kind = arc::render::render_pass_kind::present,
        .reads = { { .handle = backbuffer, .kind = arc::render::render_resource_kind::color_texture,
            .usage = arc::render::render_resource_usage::sampled } }
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
        .format = arc::render::render_format::rgba8_unorm,
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
    const auto target = graph.add_resource({
        .name = "target",
        .kind = arc::render::render_resource_kind::color_texture,
        .format = arc::render::render_format::rgba8_unorm
    });
    std::uint32_t recorded{};
    graph.add_pass({
        .name = "produce",
        .writes = { { .handle = target, .kind = arc::render::render_resource_kind::color_texture,
            .usage = arc::render::render_resource_usage::color_attachment, .write = true } },
        .record = count_recorded_pass,
        .user_data = &recorded
    });
    graph.add_pass({
        .name = "consume",
        .reads = { { .handle = target, .kind = arc::render::render_resource_kind::color_texture,
            .usage = arc::render::render_resource_usage::sampled } },
        .record = count_recorded_pass,
        .user_data = &recorded
    });

    recording_command_encoder encoder;
    arc::render::execute_render_graph(graph.compile(), encoder);

    REQUIRE(encoder.passes == std::vector<std::string>{ "produce", "consume" });
    REQUIRE(encoder.barriers == std::vector<std::string>{ "target" });
    REQUIRE(encoder.ended_passes == 2);
    REQUIRE(recorded == 2);
}

TEST_CASE("render graph rejects invalid resource declarations and internal reads")
{
    arc::render::render_graph undeclared;
    undeclared.add_pass({
        .name = "bad read",
        .reads = { { .resource = "missing", .usage = arc::render::render_resource_usage::sampled } }
    });
    REQUIRE_THROWS_AS(undeclared.compile(), std::invalid_argument);

    arc::render::render_graph read_before_write;
    const auto transient = read_before_write.add_resource({
        .name = "transient",
        .kind = arc::render::render_resource_kind::color_texture,
        .format = arc::render::render_format::rgba8_unorm
    });
    read_before_write.add_pass({
        .name = "bad read",
        .reads = { { .handle = transient, .kind = arc::render::render_resource_kind::color_texture,
            .usage = arc::render::render_resource_usage::sampled } }
    });
    REQUIRE_THROWS_AS(read_before_write.compile(), std::invalid_argument);

    arc::render::render_graph incompatible;
    const auto depth = incompatible.add_resource({
        .name = "depth",
        .kind = arc::render::render_resource_kind::depth_texture,
        .format = arc::render::render_format::d32_float
    });
    incompatible.add_pass({
        .name = "bad attachment",
        .writes = { { .handle = depth, .kind = arc::render::render_resource_kind::depth_texture,
            .usage = arc::render::render_resource_usage::color_attachment, .write = true } }
    });
    REQUIRE_THROWS_AS(incompatible.compile(), std::invalid_argument);
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

TEST_CASE("scene draw graph selects only implemented deferred passes")
{
    const auto graph = arc::render::make_scene_draw_graph("viewport", arc::render::render_path::deferred);
    const auto compiled = graph.compile();

    REQUIRE(compiled.passes.size() == 13);
    const auto pass_index = [&](std::string_view name) {
        for (std::size_t index = 0; index < compiled.passes.size(); ++index)
        {
            if (compiled.passes[index].name == name)
                return index;
        }
        return compiled.passes.size();
    };

    const std::size_t shadow_index = pass_index("directional shadow cascades");
    const std::size_t sky_index = pass_index("sky composite");
    const std::size_t depth_index = pass_index("depth prepass");
    const std::size_t gbuffer_index = pass_index("gbuffer pass");
    const std::size_t deferred_index = pass_index("deferred lighting");
    const std::size_t transparent_index = pass_index("forward transparent");
    for (std::size_t index = 0; index < compiled.passes.size(); ++index)
        REQUIRE_FALSE(compiled.passes[index].name.empty());

    REQUIRE(shadow_index < gbuffer_index);
    REQUIRE(depth_index < gbuffer_index);
    REQUIRE(gbuffer_index < deferred_index);
    REQUIRE(sky_index < deferred_index);
    REQUIRE(deferred_index < transparent_index);
    REQUIRE(compiled.passes[compiled.passes.size() - 4].builtin == arc::render::builtin_render_pass::debug_overlay);
    REQUIRE(compiled.passes[compiled.passes.size() - 3].builtin == arc::render::builtin_render_pass::luminance_histogram);
    REQUIRE(compiled.passes[compiled.passes.size() - 2].builtin == arc::render::builtin_render_pass::exposure_resolve);
    REQUIRE(compiled.passes.back().builtin == arc::render::builtin_render_pass::output_transform);
    REQUIRE(compiled.resources.size() == 15);
    REQUIRE(std::any_of(compiled.resources.begin(), compiled.resources.end(), [](const auto& resource) {
        return resource.name == "gbuffer_albedo" && resource.format == arc::render::render_format::rgba8_srgb;
    }));
    REQUIRE(compiled.lifetimes.size() == compiled.resources.size());
    REQUIRE_FALSE(compiled.transitions.empty());
}

TEST_CASE("scene draw graph provides a compact forward plus fallback")
{
    const auto compiled = arc::render::make_scene_draw_graph(
        "viewport", arc::render::render_path::forward_plus, false).compile();

    REQUIRE(compiled.passes.size() == 8);
    REQUIRE(compiled.resources.size() == 6);
    REQUIRE(compiled.passes[3].name == "forward opaque");
    for (const auto& pass : compiled.passes)
        REQUIRE(pass.name != "gbuffer pass");
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

    const auto compiled = arc::render::make_scene_draw_graph(
        "viewport", config, true, environment).compile();
    const auto contains = [&](arc::render::builtin_render_pass expected) {
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
    REQUIRE(std::any_of(compiled.resources.begin(), compiled.resources.end(), [](const auto& resource) {
        return resource.name == "environment_specular" &&
            resource.extent.width == 256 &&
            resource.array_layers == 6 &&
            resource.mip_levels == 9;
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

    const auto compiled = arc::render::make_scene_draw_graph("viewport", standard, true, environment).compile();
    const auto contains = [&](arc::render::builtin_render_pass expected) {
        return std::any_of(compiled.passes.begin(), compiled.passes.end(),
            [expected](const auto& pass) { return pass.builtin == expected; });
    };
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_transmittance));
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_multi_scattering));
    REQUIRE(contains(arc::render::builtin_render_pass::atmosphere_sky_view));
    REQUIRE(contains(arc::render::builtin_render_pass::cloud_shadow));
    REQUIRE(contains(arc::render::builtin_render_pass::sky_composite));
    REQUIRE(contains(arc::render::builtin_render_pass::debug_overlay));

    standard.quality = arc::render::render_quality_tier::low;
    standard.path = arc::render::render_path::forward_plus;
    const auto low = arc::render::make_scene_draw_graph("viewport", standard, true, environment).compile();
    REQUIRE(std::none_of(low.passes.begin(), low.passes.end(), [](const auto& pass) {
        return pass.builtin == arc::render::builtin_render_pass::atmosphere_transmittance ||
            pass.builtin == arc::render::builtin_render_pass::cloud_shadow;
    }));
    REQUIRE(std::any_of(low.passes.begin(), low.passes.end(), [](const auto& pass) {
        return pass.builtin == arc::render::builtin_render_pass::sky_composite;
    }));
    REQUIRE(std::any_of(low.passes.begin(), low.passes.end(), [](const auto& pass) {
        return pass.builtin == arc::render::builtin_render_pass::debug_overlay;
    }));
}

TEST_CASE("world environment graph selects off solid and HDRI sky paths without atmosphere LUTs")
{
    arc::render::resolved_render_config config;
    config.quality = arc::render::render_quality_tier::medium;
    config.path = arc::render::render_path::deferred;
    const auto contains = [](const auto& graph, arc::render::builtin_render_pass expected) {
        return std::any_of(graph.passes.begin(), graph.passes.end(),
            [expected](const auto& pass) { return pass.builtin == expected; });
    };

    arc::render::world_environment_data environment;
    environment.enabled = false;
    environment.sky_visible = false;
    environment.clouds.enabled = false;
    auto compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile();
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::sky_composite));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::atmosphere_transmittance));

    environment.enabled = true;
    environment.sky_visible = true;
    environment.source = arc::render::sky_source_mode::solid_color;
    compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile();
    REQUIRE(contains(compiled, arc::render::builtin_render_pass::sky_composite));
    REQUIRE_FALSE(contains(compiled, arc::render::builtin_render_pass::atmosphere_transmittance));

    environment.source = arc::render::sky_source_mode::hdri;
    compiled = arc::render::make_scene_draw_graph("viewport", config, true, environment).compile();
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
    REQUIRE(&high == &standard_render_quality_profile);
    REQUIRE(high.minimum_render_scale == Catch::Approx(0.67f));
    REQUIRE(high.directional_shadow_resolution == 2048);
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

TEST_CASE("dynamic resolution uses smoothed hysteresis and sixteenth steps")
{
    arc::render::dynamic_resolution_controller controller;
    controller.reset(
        arc::render::default_target_frame_time_ms,
        arc::render::low_render_quality_profile.minimum_render_scale,
        arc::render::low_render_quality_profile.maximum_render_scale);

    for (std::uint32_t index = 0; index < 12; ++index)
        controller.update(30.0f);
    const float reduced = controller.scale();
    REQUIRE(reduced < 1.0f);
    REQUIRE(reduced >= 0.5f);
    REQUIRE(std::round(reduced / arc::render::dynamic_resolution_scale_step) ==
        Catch::Approx(reduced / arc::render::dynamic_resolution_scale_step));

    for (std::uint32_t index = 0; index < 48; ++index)
        controller.update(5.0f);
    REQUIRE(controller.scale() > reduced);
    REQUIRE(controller.scale() <= 1.0f);
}

TEST_CASE("renderer exposes compiled render graph snapshots through frame profile")
{
    auto backend = std::make_unique<recording_backend>();
    auto* backend_ptr = backend.get();
    arc::render::renderer renderer;
    renderer.set_backend(std::move(backend));

    const auto result = renderer.render_frame(7, arc::render::make_scene_draw_graph("viewport"));

    REQUIRE(result.submitted);
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

TEST_CASE("renderer updates mesh vertices and retires stale handles")
{
    arc::render::renderer renderer;
    arc::render::mesh_data mesh;
    mesh.name = "dynamic terrain chunk";
    mesh.usage = arc::render::mesh_usage::dynamic_per_frame;
    mesh.vertices.resize(4);
    mesh.indices = { 0, 1, 2, 0, 2, 3 };
    const auto handle = renderer.create_mesh(std::move(mesh));
    renderer.frame_queue().commit(1);

    std::vector<arc::render::mesh_vertex> vertices(4);
    vertices[0].position[1] = 3.0f;
    REQUIRE(renderer.update_mesh_vertices(handle, vertices));
    auto update = renderer.frame_queue().commit(2);
    REQUIRE(update.events.size() == 1);
    REQUIRE(update.events[0].type() == arc::render::render_event_type::mesh_upload);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->indices.size() == 6);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->usage ==
        arc::render::mesh_usage::dynamic_per_frame);
    REQUIRE(std::get<arc::render::mesh_upload_event>(update.events[0].payload).mesh->vertices[0].position[1] == 3.0f);

    REQUIRE(renderer.destroy_mesh(handle));
    REQUIRE_FALSE(renderer.mesh_alive(handle));
    auto destroy = renderer.frame_queue().commit(3);
    REQUIRE(destroy.events.size() == 1);
    REQUIRE(destroy.events[0].type() == arc::render::render_event_type::mesh_destroy);
    REQUIRE_FALSE(renderer.destroy_mesh(handle));
}

TEST_CASE("renderer create virtual mesh enqueues typed upload and keeps CPU cluster metadata")
{
    arc::render::renderer renderer;
    arc::render::virtual_mesh_data mesh;
    mesh.vertices.resize(3);
    mesh.indices = { 0, 1, 2 };
    mesh.clusters.push_back({
        .first_index = 0,
        .index_count = 3,
        .triangle_count = 1,
        .vertex_count = 3,
        .material_index = 2
    });

    const auto handle = renderer.create_virtual_mesh(std::move(mesh));
    REQUIRE(renderer.virtual_mesh_alive(handle));
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

    auto updated = material;
    updated.name = "pbr_updated";
    updated.roughness = 0.35f;
    updated.base_color = { 0.25f, 0.5f, 0.75f, 1.0f };

    REQUIRE(renderer.update_material(material_handle, updated));
    const auto update_packet = renderer.frame_queue().commit(2);
    REQUIRE(update_packet.events.size() == 1);
    REQUIRE(update_packet.events[0].type() == arc::render::render_event_type::material_upload);
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
    REQUIRE(texture_update_packet.events.size() == 1);
    const auto& texture_update = std::get<arc::render::texture_upload_event>(texture_update_packet.events[0].payload);
    REQUIRE(texture_update.handle == texture_handle);
    REQUIRE(texture_update.texture->width == 2);

    REQUIRE_FALSE(renderer.update_material({ .index = 999, .generation = 1 }, updated));
    REQUIRE_FALSE(renderer.update_texture({ .index = 999, .generation = 1 }, replacement));
}

TEST_CASE("renderer creates environment resources")
{
    arc::render::renderer renderer;
    arc::render::environment_desc environment;
    environment.name = "studio";
    environment.fallback_color = { 0.20f, 0.22f, 0.25f };
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
        directional.push_back({
            .direction = { 0.0f, -1.0f, 0.0f },
            .color = { 1.0f, 1.0f, 1.0f },
            .intensity = static_cast<float>(index + 1),
            .label = "sun"
        });
    }

    std::vector<arc::render::point_light_event> points{
        { .position = { 1.0f, 2.0f, 3.0f }, .color = { 1.0f, 0.5f, 0.25f }, .intensity = 80.0f, .range = 4.0f, .intensity_unit = arc::render::light_intensity_unit::lumen },
        { .position = { 0.0f, 0.0f, 0.0f }, .color = { 1.0f, 1.0f, 1.0f }, .intensity = 2.0f, .range = 8.0f }
    };
    std::vector<arc::render::spot_light_event> spots{
        { .position = { 0.0f, 1.0f, 0.0f }, .direction = { 0.0f, -1.0f, 0.0f }, .color = { 0.8f, 0.9f, 1.0f }, .intensity = 3.0f, .range = 10.0f, .inner_angle = 0.2f, .outer_angle = 0.7f }
    };

    arc::render::environment_desc environment;
    environment.fallback_color = { 0.1f, 0.2f, 0.3f };
    environment.intensity = 1.25f;

    const auto data = arc::render::pack_scene_lighting(directional, points, spots, &environment);
    REQUIRE(data.directional_count == arc::render::max_directional_lights);
    REQUIRE(data.skipped_directional_count == 2);
    REQUIRE(data.directional_lights[0].direction_intensity[3] == Catch::Approx(6.0f));
    REQUIRE(data.point_count == 2);
    REQUIRE(data.point_lights[0].color_intensity[3] == Catch::Approx(80.0f / (4.0f * arc::math::pi<float>)));
    REQUIRE(data.spot_count == 1);
    REQUIRE(data.spot_lights[0].params[0] == Catch::Approx(0.7f));
    REQUIRE(data.ambient_color_intensity[1] == Catch::Approx(0.2f));

    environment.prefiltered = true;
    environment.diffuse_irradiance = { 0.4f, 0.5f, 0.6f };
    environment.diffuse_intensity = 0.75f;
    const auto prefiltered = arc::render::pack_scene_lighting({}, {}, {}, &environment);
    REQUIRE(prefiltered.ambient_color_intensity[0] == Catch::Approx(0.4f));
    REQUIRE(prefiltered.ambient_color_intensity[2] == Catch::Approx(0.6f));
    REQUIRE(prefiltered.ambient_color_intensity[3] == Catch::Approx(0.75f));
}

TEST_CASE("light unit and temperature helpers provide stable defaults")
{
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::unitless, 2.0f, 4.0f) == Catch::Approx(2.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::candela, 5.0f, 2.0f) == Catch::Approx(5.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::lux, 3.0f, 2.0f) == Catch::Approx(3.0f));
    REQUIRE(arc::render::light_intensity_scale(arc::render::light_intensity_unit::lumen, 4.0f * arc::math::pi<float>) == Catch::Approx(1.0f));
    const auto warm = arc::render::color_temperature_rgb(3000.0f);
    const auto cool = arc::render::color_temperature_rgb(9000.0f);
    REQUIRE(warm[0] >= warm[2]);
    REQUIRE(cool[2] >= cool[0]);
}

TEST_CASE("PBR color transfer and material texture semantics are explicit")
{
    const arc::math::vector3f srgb{ 0.0f, 0.5f, 1.0f };
    const auto linear = arc::render::srgb_to_linear(srgb);
    const auto round_trip = arc::render::linear_to_srgb(linear);
    REQUIRE(round_trip[0] == Catch::Approx(srgb[0]).margin(1.0e-6f));
    REQUIRE(round_trip[1] == Catch::Approx(srgb[1]).margin(1.0e-5f));
    REQUIRE(round_trip[2] == Catch::Approx(srgb[2]).margin(1.0e-6f));
    REQUIRE(arc::render::texture_semantic_accepts(
        arc::render::texture_semantic::base_color,
        arc::render::texture_color_space::srgb));
    REQUIRE_FALSE(arc::render::texture_semantic_accepts(
        arc::render::texture_semantic::normal,
        arc::render::texture_color_space::srgb));
}

TEST_CASE("PBR reference functions stay finite and preserve physical limits")
{
    for (const float roughness : { 0.04f, 0.25f, 0.6f, 1.0f })
    {
        const float distribution = arc::render::ggx_distribution(0.75f, roughness);
        const float visibility = arc::render::smith_ggx_correlated(0.6f, 0.7f, roughness);
        REQUIRE(std::isfinite(distribution));
        REQUIRE(std::isfinite(visibility));
        REQUIRE(distribution >= 0.0f);
        REQUIRE(visibility >= 0.0f);
    }
    const auto fresnel = arc::render::fresnel_schlick(0.0f, { 0.04f, 0.04f, 0.04f });
    REQUIRE(fresnel[0] == Catch::Approx(1.0f));
    const auto absorption = arc::render::beer_lambert_attenuation(
        { 0.5f, 0.25f, 1.0f }, 2.0f, 2.0f);
    REQUIRE(absorption[0] == Catch::Approx(0.5f));
    REQUIRE(absorption[1] == Catch::Approx(0.25f));
    REQUIRE(absorption[2] == Catch::Approx(1.0f));
}

TEST_CASE("physical attenuation exposure and area light packing are stable")
{
    REQUIRE(arc::render::inverse_square_attenuation(2.0f, 0.0f) == Catch::Approx(0.25f));
    REQUIRE(arc::render::inverse_square_attenuation(10.0f, 5.0f) == Catch::Approx(0.0f));
    REQUIRE(arc::render::cone_solid_angle(arc::math::pi<float> * 0.5f) ==
        Catch::Approx(2.0f * arc::math::pi<float>));

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

    std::vector<arc::render::area_light_event> areas{
        {
            .intensity = 1000.0f,
            .width = 2.0f,
            .height = 1.0f,
            .shape = arc::render::area_light_shape::rectangle,
            .intensity_unit = arc::render::light_intensity_unit::lumen
        }
    };
    const auto lighting = arc::render::pack_scene_lighting({}, {}, {}, nullptr, 0, 0, areas);
    REQUIRE(lighting.area_count == 1);
    REQUIRE(lighting.area_lights[0].color_intensity[3] ==
        Catch::Approx(1000.0f / (2.0f * arc::math::pi<float>)));
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
    allocation.bytes.front() = std::byte{ 0x5a };
    REQUIRE(mapped[allocation.offset] == std::byte{ 0x5a });

    REQUIRE(arena.retire_completed(8) == 0);
    REQUIRE(arena.retire_completed(9) == 1);
    REQUIRE(arena.used() == 0);
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
    auto bytes = make_dds_header(
        2,
        2,
        1,
        0x00000041u,
        0,
        32,
        0x000000ff,
        0x0000ff00,
        0x00ff0000,
        0xff000000);
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
    const auto path = std::filesystem::path(ARC_RENDER_TEST_ASSET_ROOT) /
        "textures" / "terrain" / "aerial_grass_rock" / "aerial_grass_rock_diff_1k.jpg";
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
    const auto result = arc::render::load_scene_asset(fixture, options, [&](const arc::render::scene_import_progress& value) {
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
    REQUIRE(plane.indices == std::vector<std::uint32_t>{ 0, 1, 2, 0, 2, 3 });
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
    source.indices = { 0, 1, 2 };

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

TEST_CASE("virtual mesh builder splits fixed-size clusters deterministically")
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
        source.indices.insert(source.indices.end(), { base, base + 1, base + 2 });
    }

    const auto first = arc::render::build_virtual_mesh(source);
    const auto second = arc::render::build_virtual_mesh(source);

    REQUIRE(first.clusters.size() == 2);
    REQUIRE(first.clusters[0].triangle_count == 128);
    REQUIRE(first.clusters[0].index_count == 384);
    REQUIRE(first.clusters[1].first_triangle == 128);
    REQUIRE(first.clusters[1].triangle_count == 2);
    REQUIRE(first.stats.source_triangle_count == 130);
    REQUIRE(first.stats.cluster_count == 2);
    REQUIRE(first.stats.average_triangles_per_cluster == Catch::Approx(65.0f));
    REQUIRE(first.stats.invalid_triangle_count == 0);
    REQUIRE(second.indices == first.indices);
    REQUIRE(second.clusters.size() == first.clusters.size());
    REQUIRE(second.clusters[0].first_index == first.clusters[0].first_index);
    REQUIRE(second.clusters[0].triangle_count == first.clusters[0].triangle_count);
    REQUIRE(second.clusters[1].sphere_radius == Catch::Approx(first.clusters[1].sphere_radius));
}

TEST_CASE("virtual mesh builder honors custom cluster size and skips invalid triangles")
{
    arc::render::mesh_data source;
    source.material_index = 11;
    source.vertices.resize(6);
    for (std::uint32_t index = 0; index < source.vertices.size(); ++index)
        source.vertices[index].position[0] = static_cast<float>(index);
    source.indices = {
        0, 1, 2,
        3, 4, 5,
        0, 99, 1,
        2
    };

    const auto virtual_mesh = arc::render::build_virtual_mesh(source, { .max_triangles_per_cluster = 1 });

    REQUIRE(virtual_mesh.indices == std::vector<std::uint32_t>{ 0, 1, 2, 3, 4, 5 });
    REQUIRE(virtual_mesh.clusters.size() == 2);
    REQUIRE(virtual_mesh.clusters[0].triangle_count == 1);
    REQUIRE(virtual_mesh.clusters[1].triangle_count == 1);
    REQUIRE(virtual_mesh.clusters[0].material_index == 11);
    REQUIRE(virtual_mesh.clusters[1].material_index == 11);
    REQUIRE(virtual_mesh.stats.source_vertex_count == 6);
    REQUIRE(virtual_mesh.stats.source_triangle_count == 3);
    REQUIRE(virtual_mesh.stats.invalid_triangle_count == 2);
    REQUIRE(virtual_mesh.stats.material_group_count == 1);
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
