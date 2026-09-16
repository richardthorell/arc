#pragma once

#include <arc/render/render.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::render::tests
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

    arc::render::virtual_geometry_feedback_readback take_virtual_geometry_feedback() override
    {
        auto result = std::move(virtual_feedback);
        virtual_feedback = {};
        return result;
    }

    std::vector<arc::render::virtual_geometry_page_upload_result> take_virtual_geometry_page_upload_results() override
    {
        auto result = std::move(virtual_upload_results);
        virtual_upload_results.clear();
        return result;
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
    arc::render::virtual_geometry_feedback_readback virtual_feedback{};
    std::vector<arc::render::virtual_geometry_page_upload_result> virtual_upload_results;
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

inline void count_recorded_pass(arc::render::render_pass_context& context)
{
    ++*context.payload<std::uint32_t*>();
}

inline void append_u32(std::vector<std::byte>& bytes, std::uint32_t value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

inline void write_u32_at(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

inline std::vector<std::byte> make_dds_header(std::uint32_t width, std::uint32_t height, std::uint32_t mip_count,
                                              std::uint32_t pixel_flags, std::uint32_t four_cc,
                                              std::uint32_t rgb_bit_count = 0, std::uint32_t r_mask = 0,
                                              std::uint32_t g_mask = 0, std::uint32_t b_mask = 0,
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

inline void append_f32(std::vector<std::byte>& bytes, float value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

inline void append_u16(std::vector<std::byte>& bytes, std::uint16_t value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

inline void pad4(std::vector<std::byte>& bytes, std::byte value)
{
    while ((bytes.size() % 4) != 0)
        bytes.push_back(value);
}

inline std::filesystem::path write_triangle_glb()
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

} // namespace arc::render::tests
