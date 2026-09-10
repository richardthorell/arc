#include <arc/io/io.h>
#include <arc/jobs/jobs.h>
#include <arc/render/renderer.h>
#include <arc/render/virtual_geometry.h>
#include <arc/render/virtual_geometry_streaming_io.h>
#include <arc/render/virtual_mesh.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace
{

arc::render::virtual_mesh_data make_budget_geometry()
{
    arc::render::virtual_mesh_data geometry;
    geometry.pages = {{.uncompressed_size = 8u, .compressed_offset = 0u, .compressed_size = 4u, .root = true},
                      {.uncompressed_size = 8u, .compressed_offset = 4u, .compressed_size = 4u},
                      {.uncompressed_size = 8u, .compressed_offset = 8u, .compressed_size = 4u},
                      {.uncompressed_size = 8u, .compressed_offset = 12u, .compressed_size = 4u}};
    geometry.page_payload.resize(16u);
    return geometry;
}

class delayed_multi_page_source final : public arc::render::virtual_geometry_page_source
{
public:
    delayed_multi_page_source(arc::jobs::job_system& jobs, const arc::render::virtual_mesh_data& geometry)
        : jobs_(jobs), payloads_(geometry.pages.size())
    {
        for (std::size_t index = 0; index < geometry.pages.size(); ++index)
        {
            const auto& page = geometry.pages[index];
            const auto bytes =
                std::span<const std::byte>(geometry.page_payload).subspan(page.compressed_offset, page.compressed_size);
            payloads_[index] = {bytes.begin(), bytes.end()};
        }
    }

    arc::jobs::job_future<arc::io::file_result<arc::io::file_buffer>>
    read_page(const arc::render::virtual_geometry_page_load& load, arc::jobs::cancellation_token cancellation) override
    {
        const auto payload = load.page_index < payloads_.size() ? payloads_[load.page_index] : std::vector<std::byte>{};
        return jobs_.submit_future({.name = "test.m2_5_delayed_virtual_geometry_read",
                                    .priority = arc::jobs::job_priority::normal,
                                    .affinity = arc::jobs::job_affinity::io_thread,
                                    .cancellation = cancellation},
                                   [payload]
                                   {
                                       std::this_thread::sleep_for(std::chrono::milliseconds(40));
                                       return arc::io::file_result<arc::io::file_buffer>::success(payload);
                                   });
    }

private:
    arc::jobs::job_system& jobs_;
    std::vector<std::vector<std::byte>> payloads_;
};

} // namespace

TEST_CASE("M2.5 hard budgets can evict recently used detail while roots remain resident")
{
    using namespace arc::render;
    const virtual_mesh_handle resource{7u, 3u};
    const auto geometry = make_budget_geometry();
    virtual_geometry_residency_manager residency({.gpu_budget_bytes = 16u,
                                                  .compressed_cpu_budget_bytes = 8u,
                                                  .maximum_requests_per_frame = 16u,
                                                  .protected_frame_count = 30u,
                                                  .reload_cooldown_frames = 4u});
    residency.register_resource(resource, geometry, 11u);

    residency.begin_frame(1u);
    residency.publish(resource, 11u, 1u, 8u, 4u);
    REQUIRE(residency.snapshot().gpu_resident_bytes == 16u);

    residency.begin_frame(2u);
    residency.publish(resource, 11u, 2u, 8u, 4u);
    const auto evictions = residency.take_evictions();
    REQUIRE(evictions.size() == 1u);
    CHECK(evictions.front().page_index == 1u);
    CHECK(residency.resident(resource, 11u, 0u));
    CHECK_FALSE(residency.resident(resource, 11u, 1u));
    CHECK(residency.resident(resource, 11u, 2u));

    const auto snapshot = residency.snapshot();
    CHECK(snapshot.gpu_resident_bytes <= snapshot.gpu_budget_bytes);
    CHECK(snapshot.compressed_cpu_resident_bytes <= snapshot.compressed_cpu_budget_bytes);
    CHECK(snapshot.root_gpu_resident_bytes == 8u);
    CHECK(snapshot.root_compressed_cpu_resident_bytes == 4u);
    CHECK(snapshot.forced_budget_evictions == 1u);
    CHECK(snapshot.gpu_budget_overflow_bytes == 0u);
    CHECK(snapshot.compressed_cpu_budget_overflow_bytes == 0u);
}

TEST_CASE("M2.5 post eviction cooldown suppresses prefetch thrash but not correctness demand")
{
    using namespace arc::render;
    const virtual_mesh_handle resource{9u, 2u};
    const auto geometry = make_budget_geometry();
    virtual_geometry_residency_manager residency({.gpu_budget_bytes = 16u,
                                                  .compressed_cpu_budget_bytes = 8u,
                                                  .maximum_requests_per_frame = 16u,
                                                  .protected_frame_count = 0u,
                                                  .reload_cooldown_frames = 4u});
    residency.register_resource(resource, geometry, 5u);
    residency.begin_frame(1u);
    residency.publish(resource, 5u, 1u, 8u, 4u);
    residency.begin_frame(2u);
    residency.publish(resource, 5u, 2u, 8u, 4u);
    REQUIRE_FALSE(residency.resident(resource, 5u, 1u));

    residency.begin_frame(3u);
    const virtual_geometry_page_request prefetch{.resource = resource,
                                                 .resource_generation = 5u,
                                                 .page_index = 1u,
                                                 .projected_error = 4.0f,
                                                 .screen_coverage = 0.5f,
                                                 .distance = 32.0f};
    residency.request(std::span(&prefetch, 1u));
    CHECK(residency.take_load_requests().empty());
    CHECK(residency.snapshot().cooldown_suppressed_requests == 1u);

    auto demand = prefetch;
    demand.visible_child = true;
    residency.request(std::span(&demand, 1u));
    const auto loads = residency.take_load_requests();
    REQUIRE(loads.size() == 1u);
    CHECK(loads.front().page_index == 1u);
}

TEST_CASE("M2.5 reports the non evictable root budget floor")
{
    using namespace arc::render;
    virtual_mesh_data geometry;
    geometry.pages = {{.uncompressed_size = 8u, .compressed_size = 4u, .root = true}};
    const virtual_mesh_handle resource{3u, 1u};
    virtual_geometry_residency_manager residency({.gpu_budget_bytes = 4u,
                                                  .compressed_cpu_budget_bytes = 2u,
                                                  .maximum_requests_per_frame = 1u,
                                                  .protected_frame_count = 0u,
                                                  .reload_cooldown_frames = 0u});
    residency.register_resource(resource, geometry, 2u);
    const auto snapshot = residency.snapshot();
    CHECK(residency.resident(resource, 2u, 0u));
    CHECK(snapshot.root_gpu_resident_bytes == 8u);
    CHECK(snapshot.root_compressed_cpu_resident_bytes == 4u);
    CHECK(snapshot.gpu_budget_overflow_bytes == 4u);
    CHECK(snapshot.compressed_cpu_budget_overflow_bytes == 2u);
}

TEST_CASE("M2.5 async streaming caps compressed plus decoded in flight bytes")
{
    using namespace arc;
    using namespace arc::render;

    const auto geometry = make_budget_geometry();
    jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    renderer target;
    const auto resource = target.create_virtual_mesh(geometry);
    REQUIRE(resource.valid());
    const auto generation = target.virtual_mesh_content_generation(resource);
    REQUIRE(generation != 0u);

    delayed_multi_page_source source(jobs, geometry);
    constexpr std::uint32_t first_page = 1u;
    const auto byte_budget = static_cast<std::uint64_t>(geometry.pages[first_page].compressed_size) +
                             geometry.pages[first_page].uncompressed_size;
    virtual_geometry_streaming_controller controller(
        target, source, jobs, {.maximum_in_flight_requests = 8u, .maximum_in_flight_bytes = byte_budget});

    target.virtual_geometry_residency().begin_frame(1u);
    std::vector<virtual_geometry_page_request> requests;
    for (std::uint32_t page_index = 1u; page_index <= 3u; ++page_index)
        requests.push_back({.resource = resource,
                            .resource_generation = generation,
                            .page_index = page_index,
                            .projected_error = 101.0f - static_cast<float>(page_index),
                            .screen_coverage = 1.0f,
                            .visible_child = true});
    target.virtual_geometry_residency().request(requests);

    controller.update();
    const auto snapshot = controller.snapshot();
    CHECK(snapshot.maximum_in_flight_requests == 8u);
    CHECK(snapshot.maximum_in_flight_bytes == byte_budget);
    CHECK(snapshot.in_flight_reads == 1u);
    CHECK(snapshot.in_flight_decodes == 0u);
    CHECK(snapshot.in_flight_bytes <= snapshot.maximum_in_flight_bytes);
    CHECK(snapshot.peak_in_flight_bytes <= snapshot.maximum_in_flight_bytes);
    CHECK(snapshot.queued_pages >= 2u);
    CHECK(snapshot.deferred_pages >= 2u);
    CHECK(snapshot.oversized_pages == 0u);
}
