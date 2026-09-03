#include <arc/render/texture_streaming_io.h>

#include <arc/render/renderer.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::render
{

struct filesystem_texture_artifact_source::implementation
{
    struct range
    {
        std::filesystem::path path;
        std::uint64_t base{};
        std::uint64_t size{};
    };
    io::async_file_service* files{};
    std::unordered_map<texture_stream_source_id, range> sources;
};

filesystem_texture_artifact_source::filesystem_texture_artifact_source(io::async_file_service& files)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->files = &files;
}

filesystem_texture_artifact_source::~filesystem_texture_artifact_source() = default;
filesystem_texture_artifact_source::filesystem_texture_artifact_source(filesystem_texture_artifact_source&&) noexcept =
    default;
filesystem_texture_artifact_source&
filesystem_texture_artifact_source::operator=(filesystem_texture_artifact_source&&) noexcept = default;

void filesystem_texture_artifact_source::register_file(texture_stream_source_id source, std::filesystem::path path,
                                                       std::uint64_t size)
{
    register_package_range(source, std::move(path), 0, size);
}

void filesystem_texture_artifact_source::register_package_range(texture_stream_source_id source,
                                                                std::filesystem::path package,
                                                                std::uint64_t base_offset, std::uint64_t size)
{
    if (source == 0 || package.empty() || size == 0) return;
    implementation_->sources[source] = {.path = std::move(package), .base = base_offset, .size = size};
}

void filesystem_texture_artifact_source::unregister(texture_stream_source_id source)
{
    implementation_->sources.erase(source);
}

jobs::job_future<io::file_result<io::file_buffer>>
filesystem_texture_artifact_source::read_range(texture_stream_source_id source, std::uint64_t offset, std::size_t bytes,
                                               jobs::cancellation_token cancellation)
{
    const auto found = implementation_->sources.find(source);
    if (found == implementation_->sources.end() || bytes == 0 || offset > found->second.size ||
        bytes > found->second.size - offset || found->second.base > std::numeric_limits<std::uint64_t>::max() - offset)
    {
        return implementation_->files->scheduler().submit_future(
            {.name = "render.texture_range.invalid",
             .priority = jobs::job_priority::normal,
             .affinity = jobs::job_affinity::io_thread,
             .cancellation = cancellation},
            []
            {
                return io::file_result<io::file_buffer>::failure(
                    {.code = io::file_error_code::invalid_range,
                     .message = "texture artifact source or range is invalid"});
            });
    }
    return implementation_->files->read_range(found->second.path, found->second.base + offset, bytes, cancellation);
}

struct texture_streaming_controller::implementation
{
    struct pending_read
    {
        texture_stream_load load;
        jobs::job_future<io::file_result<io::file_buffer>> future;
        std::chrono::steady_clock::time_point started;
    };
    renderer* target{};
    texture_artifact_source* source{};
    std::uint32_t maximum_in_flight{2048};
    std::vector<pending_read> pending;
    texture_streaming_io_snapshot statistics;
};

texture_streaming_controller::texture_streaming_controller(renderer& renderer, texture_artifact_source& source,
                                                           std::uint32_t maximum_in_flight)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->target = &renderer;
    implementation_->source = &source;
    implementation_->maximum_in_flight = std::max(1u, maximum_in_flight);
}

texture_streaming_controller::~texture_streaming_controller() = default;
texture_streaming_controller::texture_streaming_controller(texture_streaming_controller&&) noexcept = default;
texture_streaming_controller&
texture_streaming_controller::operator=(texture_streaming_controller&&) noexcept = default;

void texture_streaming_controller::update(const jobs::cancellation_token& cancellation)
{
    auto& state = *implementation_;
    for (std::size_t index = 0; index < state.pending.size();)
    {
        auto& pending = state.pending[index];
        if (!pending.future.ready())
        {
            ++index;
            continue;
        }
        auto result = pending.future.get();
        state.statistics.total_read_latency_milliseconds +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pending.started).count();
        if (result)
        {
            state.statistics.read_bytes += result.value().size();
            ++state.statistics.completed_reads;
            texture_stream_upload upload{.resource = pending.load.resource,
                                         .content_generation = pending.load.content_generation,
                                         .kind = pending.load.kind,
                                         .mip = pending.load.mip,
                                         .x = pending.load.x,
                                         .y = pending.load.y,
                                         .bytes =
                                             std::make_shared<const std::vector<std::byte>>(std::move(result).value()),
                                         .stored_bytes = pending.load.byte_size};
            if (!state.target->publish_texture_subresource(std::move(upload)))
            {
                ++state.statistics.failed_reads;
                state.statistics.failed_bytes += pending.load.byte_size;
                state.target->fail_texture_subresource(pending.load);
            }
        }
        else
        {
            ++state.statistics.failed_reads;
            state.statistics.failed_bytes += pending.load.byte_size;
            state.target->fail_texture_subresource(pending.load);
        }
        state.pending[index] = std::move(state.pending.back());
        state.pending.pop_back();
    }

    const auto available = state.maximum_in_flight - static_cast<std::uint32_t>(state.pending.size());
    auto loads = state.target->take_texture_stream_loads(available);
    for (auto& load : loads)
    {
        auto future = state.source->read_range(load.source, load.byte_offset, load.byte_size, cancellation);
        state.pending.push_back(
            {.load = load, .future = std::move(future), .started = std::chrono::steady_clock::now()});
    }
    state.statistics.in_flight_reads = static_cast<std::uint32_t>(state.pending.size());
    state.target->report_texture_streaming_io(state.statistics);
}

texture_streaming_io_snapshot texture_streaming_controller::snapshot() const noexcept
{
    return implementation_->statistics;
}

} // namespace arc::render
