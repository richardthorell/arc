#pragma once

#include <arc/editor/arc_host.h>
#include <arc/render/render_backend.h>

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>

namespace arc::editor::tests
{

class pick_test_backend final : public render::render_backend
{
public:
    render::render_backend_type type() const noexcept override
    {
        return render::render_backend_type::vulkan;
    }

    const render::render_capabilities& capabilities() const noexcept override
    {
        return capabilities_;
    }

    render::render_submit_result submit(const render::render_frame_packet&,
                                        const render::compiled_render_graph&) override
    {
        return render::render_submit_result::success();
    }

    void request_object_pick(render::render_object_pick_request request) override
    {
        request_ = request;
    }

    render::render_object_pick_result last_object_pick() const override
    {
        return result;
    }

    void request_frame_capture(const render::render_frame_capture_request& request) override
    {
        capture_request = request;
    }

    render::render_frame_capture_result last_frame_capture() const override
    {
        return capture_result;
    }

    render::render_object_pick_request request_{};
    render::render_object_pick_result result{};
    render::render_frame_capture_request capture_request{};
    render::render_frame_capture_result capture_result{};

private:
    render::render_capabilities capabilities_{
        .api_major = 1, .api_minor = 2, .graphics_queue = true, .compute_queue = true, .presentation = true};
};

inline host_entity_id parse_entity_from_response(const std::string& line)
{
    host_entity_id entity;
    const auto index_pos = line.find("\"index\":");
    const auto generation_pos = line.find("\"generation\":");
    if (index_pos == std::string::npos || generation_pos == std::string::npos) return entity;

    const auto parse_value = [&line](const std::size_t position, const std::string_view prefix, std::uint32_t& value)
    {
        const char* begin = line.data() + position + prefix.size();
        const char* end = line.data() + line.size();
        const auto [parsed_end, error] = std::from_chars(begin, end, value);
        return error == std::errc{} && parsed_end != begin;
    };
    if (!parse_value(index_pos, "\"index\":", entity.index) ||
        !parse_value(generation_pos, "\"generation\":", entity.generation))
        return {};
    return entity;
}

} // namespace arc::editor::tests
