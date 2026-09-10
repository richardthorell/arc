#include <arc/render/virtual_geometry_streaming_io.h>

#include <arc/render/renderer.h>
#include <arc/render/virtual_mesh.h>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::render
{
namespace
{

std::uint64_t streaming_resource_key(virtual_mesh_handle handle) noexcept
{
    return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
}

io::file_result<io::file_buffer> invalid_page_read(std::string message)
{
    return io::file_result<io::file_buffer>::failure(
        {.code = io::file_error_code::invalid_range, .message = std::move(message)});
}

} // namespace

struct filesystem_virtual_geometry_artifact_source::implementation
{
    struct artifact_range
    {
        std::filesystem::path path;
        std::uint64_t base{};
        std::uint64_t size{};
        std::uint32_t resource_generation{};
        std::vector<virtual_geometry_artifact_page_range> pages;
    };

    io::async_file_service* files{};
    std::unordered_map<std::uint64_t, artifact_range> sources;
};

filesystem_virtual_geometry_artifact_source::filesystem_virtual_geometry_artifact_source(io::async_file_service& files)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->files = &files;
}

filesystem_virtual_geometry_artifact_source::~filesystem_virtual_geometry_artifact_source() = default;
filesystem_virtual_geometry_artifact_source::filesystem_virtual_geometry_artifact_source(
    filesystem_virtual_geometry_artifact_source&&) noexcept = default;
filesystem_virtual_geometry_artifact_source& filesystem_virtual_geometry_artifact_source::operator=(
    filesystem_virtual_geometry_artifact_source&&) noexcept = default;

void filesystem_virtual_geometry_artifact_source::register_package_range(
    virtual_mesh_handle resource, std::uint32_t resource_generation, std::filesystem::path package,
    std::uint64_t artifact_base_offset, std::uint64_t artifact_size,
    std::vector<virtual_geometry_artifact_page_range> pages)
{
    if (!resource.valid() || resource_generation == 0u || package.empty() || artifact_size == 0u || pages.empty())
    {
        unregister(resource);
        return;
    }
    const bool valid_ranges = std::all_of(pages.begin(), pages.end(),
                                          [artifact_size](const auto& page)
                                          {
                                              return page.stored_size != 0u && page.decoded_size != 0u &&
                                                     page.offset <= artifact_size &&
                                                     page.stored_size <= artifact_size - page.offset;
                                          });
    if (!valid_ranges)
    {
        unregister(resource);
        return;
    }
    implementation_->sources[streaming_resource_key(resource)] = {.path = std::move(package),
                                                                  .base = artifact_base_offset,
                                                                  .size = artifact_size,
                                                                  .resource_generation = resource_generation,
                                                                  .pages = std::move(pages)};
}

void filesystem_virtual_geometry_artifact_source::unregister(virtual_mesh_handle resource)
{
    implementation_->sources.erase(streaming_resource_key(resource));
}

jobs::job_future<io::file_result<io::file_buffer>>
filesystem_virtual_geometry_artifact_source::read_page(const virtual_geometry_page_load& load,
                                                       jobs::cancellation_token cancellation)
{
    const auto found = implementation_->sources.find(streaming_resource_key(load.resource));
    if (found == implementation_->sources.end() || found->second.resource_generation != load.resource_generation ||
        load.page_index >= found->second.pages.size())
    {
        return implementation_->files->scheduler().submit_future(
            {.name = "render.virtual_geometry_page.invalid_source",
             .priority = jobs::job_priority::normal,
             .affinity = jobs::job_affinity::io_thread,
             .cancellation = cancellation},
            [] { return invalid_page_read("virtual-geometry artifact source or generation is invalid"); });
    }

    const auto& source = found->second;
    const auto& page = source.pages[load.page_index];
    if (page.stored_size != load.byte_size || page.offset > source.size ||
        page.stored_size > source.size - page.offset ||
        source.base > std::numeric_limits<std::uint64_t>::max() - page.offset)
    {
        return implementation_->files->scheduler().submit_future(
            {.name = "render.virtual_geometry_page.invalid_range",
             .priority = jobs::job_priority::normal,
             .affinity = jobs::job_affinity::io_thread,
             .cancellation = cancellation},
            [] { return invalid_page_read("virtual-geometry cooked page range is invalid"); });
    }

    return implementation_->files->read_range(source.path, source.base + page.offset, page.stored_size, cancellation);
}

struct virtual_geometry_streaming_controller::implementation
{
    struct pending_read
    {
        virtual_geometry_page_load load;
        std::uint64_t reserved_bytes{};
        jobs::job_future<io::file_result<io::file_buffer>> future;
    };

    struct decode_result
    {
        bool succeeded{};
        std::vector<std::byte> bytes;
        std::uint32_t compressed_bytes{};
    };

    struct pending_decode
    {
        virtual_geometry_page_load load;
        std::uint64_t reserved_bytes{};
        jobs::job_future<decode_result> future;
    };

    renderer* target{};
    virtual_geometry_page_source* source{};
    jobs::job_system* jobs{};
    virtual_geometry_streaming_config config{};
    std::vector<virtual_geometry_page_load> queued;
    std::vector<pending_read> reads;
    std::vector<pending_decode> decodes;
    std::uint64_t in_flight_bytes{};
    virtual_geometry_streaming_io_snapshot statistics;

    [[nodiscard]] bool current(const virtual_geometry_page_load& load) const noexcept
    {
        return target->virtual_mesh_content_generation(load.resource) == load.resource_generation;
    }

    [[nodiscard]] std::uint64_t reservation(const virtual_geometry_page_load& load) const noexcept
    {
        const auto* geometry = target->virtual_mesh_data_for(load.resource);
        if (!geometry || load.page_index >= geometry->pages.size()) return load.byte_size;
        const auto decoded = static_cast<std::uint64_t>(geometry->pages[load.page_index].uncompressed_size);
        if (decoded > std::numeric_limits<std::uint64_t>::max() - load.byte_size)
            return std::numeric_limits<std::uint64_t>::max();
        return decoded + load.byte_size;
    }

    [[nodiscard]] std::size_t in_flight_count() const noexcept
    {
        return reads.size() + decodes.size();
    }

    void release(std::uint64_t bytes) noexcept
    {
        in_flight_bytes -= std::min(in_flight_bytes, bytes);
    }

    void fail_or_discard(const virtual_geometry_page_load& load)
    {
        if (!current(load))
        {
            ++statistics.stale_completions;
            return;
        }
        ++statistics.failed_pages;
        target->fail_virtual_geometry_page(load.resource, load.resource_generation, load.page_index);
    }
};

virtual_geometry_streaming_controller::virtual_geometry_streaming_controller(renderer& renderer,
                                                                             virtual_geometry_page_source& source,
                                                                             jobs::job_system& jobs,
                                                                             virtual_geometry_streaming_config config)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->target = &renderer;
    implementation_->source = &source;
    implementation_->jobs = &jobs;
    configure(config);
}

virtual_geometry_streaming_controller::virtual_geometry_streaming_controller(renderer& renderer,
                                                                             virtual_geometry_page_source& source,
                                                                             jobs::job_system& jobs,
                                                                             std::uint32_t maximum_in_flight)
    : virtual_geometry_streaming_controller(
          renderer, source, jobs, virtual_geometry_streaming_config{.maximum_in_flight_requests = maximum_in_flight})
{
}

virtual_geometry_streaming_controller::~virtual_geometry_streaming_controller() = default;
virtual_geometry_streaming_controller::virtual_geometry_streaming_controller(
    virtual_geometry_streaming_controller&&) noexcept = default;
virtual_geometry_streaming_controller&
virtual_geometry_streaming_controller::operator=(virtual_geometry_streaming_controller&&) noexcept = default;

void virtual_geometry_streaming_controller::configure(virtual_geometry_streaming_config config) noexcept
{
    config.maximum_in_flight_requests = std::max(1u, config.maximum_in_flight_requests);
    config.maximum_in_flight_bytes = std::max<std::uint64_t>(1u, config.maximum_in_flight_bytes);
    implementation_->config = config;
    implementation_->statistics.maximum_in_flight_requests = config.maximum_in_flight_requests;
    implementation_->statistics.maximum_in_flight_bytes = config.maximum_in_flight_bytes;
}

void virtual_geometry_streaming_controller::update(const jobs::cancellation_token& cancellation)
{
    auto& state = *implementation_;

    for (std::size_t index = 0; index < state.decodes.size();)
    {
        auto& pending = state.decodes[index];
        if (!pending.future.ready())
        {
            ++index;
            continue;
        }
        auto decoded = pending.future.get();
        if (!state.current(pending.load))
        {
            ++state.statistics.stale_completions;
        }
        else if (!decoded.succeeded)
        {
            ++state.statistics.failed_pages;
            state.target->fail_virtual_geometry_page(pending.load.resource, pending.load.resource_generation,
                                                     pending.load.page_index);
        }
        else
        {
            state.statistics.decoded_bytes += decoded.bytes.size();
            auto shared = std::make_shared<const std::vector<std::byte>>(std::move(decoded.bytes));
            virtual_geometry_page_upload upload{.resource = pending.load.resource,
                                                .resource_generation = pending.load.resource_generation,
                                                .page_index = pending.load.page_index,
                                                .decoded_bytes = std::move(shared),
                                                .compressed_cpu_bytes = decoded.compressed_bytes};
            if (state.target->publish_virtual_geometry_page(std::move(upload)))
                ++state.statistics.completed_pages;
            else
                state.fail_or_discard(pending.load);
        }
        state.release(pending.reserved_bytes);
        state.decodes[index] = std::move(state.decodes.back());
        state.decodes.pop_back();
    }

    for (std::size_t index = 0; index < state.reads.size();)
    {
        auto& pending = state.reads[index];
        if (!pending.future.ready())
        {
            ++index;
            continue;
        }
        auto result = pending.future.get();
        bool transferred_reservation{};
        if (!state.current(pending.load))
        {
            ++state.statistics.stale_completions;
        }
        else if (!result)
        {
            ++state.statistics.failed_pages;
            state.target->fail_virtual_geometry_page(pending.load.resource, pending.load.resource_generation,
                                                     pending.load.page_index);
        }
        else
        {
            auto* geometry = state.target->virtual_mesh_data_for(pending.load.resource);
            if (!geometry || pending.load.page_index >= geometry->pages.size())
            {
                state.fail_or_discard(pending.load);
            }
            else
            {
                auto compressed = std::move(result).value();
                state.statistics.read_bytes += compressed.size();
                const auto page = geometry->pages[pending.load.page_index];
                const auto compressed_bytes = static_cast<std::uint32_t>(compressed.size());
                auto future =
                    state.jobs->submit_future({.name = "render.virtual_geometry_page.decode",
                                               .priority = jobs::job_priority::high,
                                               .affinity = jobs::job_affinity::any_worker},
                                              [page, compressed = std::move(compressed), compressed_bytes]() mutable
                                              {
                                                  implementation::decode_result decoded;
                                                  decoded.compressed_bytes = compressed_bytes;
                                                  decoded.succeeded =
                                                      decode_virtual_geometry_page(page, compressed, decoded.bytes);
                                                  return decoded;
                                              });
                state.decodes.push_back(
                    {.load = pending.load, .reserved_bytes = pending.reserved_bytes, .future = std::move(future)});
                transferred_reservation = true;
            }
        }
        if (!transferred_reservation) state.release(pending.reserved_bytes);
        state.reads[index] = std::move(state.reads.back());
        state.reads.pop_back();
    }

    for (std::size_t index = 0; index < state.queued.size();)
    {
        if (state.current(state.queued[index]))
        {
            ++index;
            continue;
        }
        ++state.statistics.stale_completions;
        state.queued.erase(state.queued.begin() + static_cast<std::ptrdiff_t>(index));
    }

    if (!cancellation.stop_requested() && state.queued.empty() &&
        state.in_flight_count() < state.config.maximum_in_flight_requests)
        state.queued = state.target->take_virtual_geometry_page_loads();

    while (!cancellation.stop_requested() && !state.queued.empty() &&
           state.in_flight_count() < state.config.maximum_in_flight_requests)
    {
        std::size_t selected = state.queued.size();
        std::uint64_t selected_reservation{};
        for (std::size_t index = 0; index < state.queued.size(); ++index)
        {
            const auto reservation = state.reservation(state.queued[index]);
            if (reservation <= state.config.maximum_in_flight_bytes -
                                   std::min(state.config.maximum_in_flight_bytes, state.in_flight_bytes))
            {
                selected = index;
                selected_reservation = reservation;
                break;
            }
        }

        if (selected == state.queued.size())
        {
            if (state.in_flight_count() != 0u) break;
            selected = 0u;
            selected_reservation = state.reservation(state.queued.front());
            ++state.statistics.oversized_pages;
        }

        auto load = state.queued[selected];
        state.queued.erase(state.queued.begin() + static_cast<std::ptrdiff_t>(selected));
        if (!state.current(load))
        {
            ++state.statistics.stale_completions;
            continue;
        }
        auto future = state.source->read_page(load, cancellation);
        state.in_flight_bytes += selected_reservation;
        state.statistics.peak_in_flight_bytes = std::max(state.statistics.peak_in_flight_bytes, state.in_flight_bytes);
        state.reads.push_back({.load = load, .reserved_bytes = selected_reservation, .future = std::move(future)});
    }

    state.statistics.in_flight_reads = static_cast<std::uint32_t>(state.reads.size());
    state.statistics.in_flight_decodes = static_cast<std::uint32_t>(state.decodes.size());
    state.statistics.queued_pages = static_cast<std::uint32_t>(state.queued.size());
    state.statistics.in_flight_bytes = state.in_flight_bytes;
    state.statistics.deferred_pages = static_cast<std::uint32_t>(state.queued.size());
}

virtual_geometry_streaming_io_snapshot virtual_geometry_streaming_controller::snapshot() const noexcept
{
    return implementation_->statistics;
}

} // namespace arc::render