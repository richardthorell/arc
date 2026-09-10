#include <arc/io/io.h>
#include <arc/jobs/jobs.h>
#include <arc/render/renderer.h>
#include <arc/render/virtual_geometry_streaming_io.h>
#include <arc/render/virtual_mesh.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <thread>
#include <vector>

namespace
{

arc::render::mesh_data make_streaming_grid(std::uint32_t side)
{
    arc::render::mesh_data mesh;
    mesh.name = "M2.3 streaming grid";
    mesh.vertices.resize(static_cast<std::size_t>(side) * side);
    for (std::uint32_t z = 0; z < side; ++z)
        for (std::uint32_t x = 0; x < side; ++x)
        {
            auto& vertex = mesh.vertices[static_cast<std::size_t>(z) * side + x];
            vertex.position[0] = static_cast<float>(x);
            vertex.position[1] = static_cast<float>((x * 7u + z * 11u) % 13u) * 0.1f;
            vertex.position[2] = static_cast<float>(z);
            vertex.normal[1] = 1.0f;
            vertex.tangent[0] = 1.0f;
            vertex.tangent[3] = 1.0f;
        }
    for (std::uint32_t z = 0; z + 1u < side; ++z)
        for (std::uint32_t x = 0; x + 1u < side; ++x)
        {
            const auto i0 = z * side + x;
            const auto i1 = i0 + 1u;
            const auto i2 = i0 + side;
            const auto i3 = i2 + 1u;
            mesh.indices.insert(mesh.indices.end(), {i0, i2, i1, i1, i2, i3});
        }
    return mesh;
}

std::uint32_t detail_page_index(const arc::render::virtual_mesh_data& geometry)
{
    const auto found = std::find_if(geometry.pages.begin(), geometry.pages.end(),
                                    [](const auto& page) { return !page.root; });
    return found == geometry.pages.end()
               ? arc::render::invalid_virtual_geometry_index
               : static_cast<std::uint32_t>(std::distance(geometry.pages.begin(), found));
}

std::vector<std::byte> page_bytes(const arc::render::virtual_mesh_data& geometry, std::uint32_t page_index)
{
    const auto& page = geometry.pages[page_index];
    const auto source = std::span<const std::byte>(geometry.page_payload)
                            .subspan(page.compressed_offset, page.compressed_size);
    return {source.begin(), source.end()};
}

class delayed_page_source final : public arc::render::virtual_geometry_artifact_source
{
public:
    delayed_page_source(arc::jobs::job_system& jobs, std::vector<std::byte> payload)
        : jobs_(jobs), payload_(std::move(payload))
    {
    }

    arc::jobs::job_future<arc::io::file_result<arc::io::file_buffer>>
    read_page(const arc::render::virtual_geometry_page_load&, arc::jobs::cancellation_token) override
    {
        return jobs_.submit_future(
            {.name = "test.delayed_virtual_geometry_read",
             .priority = arc::jobs::job_priority::normal,
             .affinity = arc::jobs::job_affinity::io_thread},
            [payload = payload_]
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                return arc::io::file_result<arc::io::file_buffer>::success(payload);
            });
    }

private:
    arc::jobs::job_system& jobs_;
    std::vector<std::byte> payload_;
};

} // namespace

TEST_CASE("externally read virtual geometry pages decode identically to retained pages")
{
    using namespace arc::render;
    const auto geometry = build_virtual_mesh(make_streaming_grid(65u), {.build_conventional_lods = false});
    const auto page_index = detail_page_index(geometry);
    REQUIRE(page_index != invalid_virtual_geometry_index);

    std::vector<std::byte> retained;
    std::vector<std::byte> external;
    REQUIRE(decode_virtual_geometry_page(geometry, page_index, retained));
    auto encoded = page_bytes(geometry, page_index);
    REQUIRE(decode_virtual_geometry_page(geometry.pages[page_index], encoded, external));
    REQUIRE(external == retained);

    encoded.front() ^= std::byte{0x1};
    REQUIRE_FALSE(decode_virtual_geometry_page(geometry.pages[page_index], encoded, external));
}

TEST_CASE("virtual geometry page IO and decode complete asynchronously from an external range")
{
    using namespace arc;
    using namespace arc::render;

    const auto root = std::filesystem::temp_directory_path() / "arc-m2-3-virtual-geometry-streaming.bin";
    struct cleanup
    {
        std::filesystem::path path;
        ~cleanup()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup_guard{root};

    const auto geometry = build_virtual_mesh(make_streaming_grid(65u), {.build_conventional_lods = false});
    const auto requested_page = detail_page_index(geometry);
    REQUIRE(requested_page != invalid_virtual_geometry_index);

    std::vector<virtual_geometry_artifact_page_range> ranges(geometry.pages.size());
    std::vector<std::byte> artifact(128u, std::byte{});
    for (std::size_t page_index = 0; page_index < geometry.pages.size(); ++page_index)
    {
        const auto& page = geometry.pages[page_index];
        while (artifact.size() % 64u != 0u) artifact.push_back(std::byte{});
        const auto offset = artifact.size();
        const auto encoded = page_bytes(geometry, static_cast<std::uint32_t>(page_index));
        artifact.insert(artifact.end(), encoded.begin(), encoded.end());
        ranges[page_index] = {.offset = offset,
                              .stored_size = page.compressed_size,
                              .decoded_size = page.uncompressed_size,
                              .content_hash = page.content_hash,
                              .root = page.root};
    }
    {
        std::ofstream stream(root, std::ios::binary | std::ios::trunc);
        REQUIRE(stream.good());
        stream.write(reinterpret_cast<const char*>(artifact.data()), static_cast<std::streamsize>(artifact.size()));
        REQUIRE(stream.good());
    }

    jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    io::async_file_service files(jobs);
    renderer target;
    const auto resource = target.create_virtual_mesh(geometry);
    REQUIRE(resource.valid());
    const auto generation = target.virtual_mesh_content_generation(resource);
    REQUIRE(generation != 0u);

    filesystem_virtual_geometry_artifact_source source(files);
    source.register_package_range(resource, generation, root, 0u, artifact.size(), ranges);
    virtual_geometry_streaming_controller controller(target, source, jobs, 1u);

    target.virtual_geometry_residency().begin_frame(1u);
    const virtual_geometry_page_request request{.resource = resource,
                                                .resource_generation = generation,
                                                .page_index = requested_page,
                                                .projected_error = 100.0f,
                                                .screen_coverage = 1.0f,
                                                .visible_child = true};
    target.virtual_geometry_residency().request(std::span(&request, 1u));

    for (std::uint32_t attempt = 0; attempt < 500u && controller.snapshot().completed_pages == 0u; ++attempt)
    {
        controller.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto snapshot = controller.snapshot();
    REQUIRE(snapshot.completed_pages == 1u);
    REQUIRE(snapshot.failed_pages == 0u);
    REQUIRE(snapshot.stale_completions == 0u);
    REQUIRE(snapshot.read_bytes == geometry.pages[requested_page].compressed_size);
    REQUIRE(snapshot.decoded_bytes == geometry.pages[requested_page].uncompressed_size);
}

TEST_CASE("virtual geometry streaming discards stale asynchronous completions after generation replacement")
{
    using namespace arc;
    using namespace arc::render;

    const auto geometry = build_virtual_mesh(make_streaming_grid(65u), {.build_conventional_lods = false});
    const auto requested_page = detail_page_index(geometry);
    REQUIRE(requested_page != invalid_virtual_geometry_index);

    jobs::job_system jobs({.worker_count = 1u, .io_worker_count = 1u, .enable_render_thread = false});
    renderer target;
    const auto resource = target.create_virtual_mesh(geometry);
    const auto old_generation = target.virtual_mesh_content_generation(resource);
    REQUIRE(old_generation != 0u);

    delayed_page_source source(jobs, page_bytes(geometry, requested_page));
    virtual_geometry_streaming_controller controller(target, source, jobs, 1u);
    target.virtual_geometry_residency().begin_frame(1u);
    const virtual_geometry_page_request request{.resource = resource,
                                                .resource_generation = old_generation,
                                                .page_index = requested_page,
                                                .projected_error = 100.0f,
                                                .screen_coverage = 1.0f,
                                                .visible_child = true};
    target.virtual_geometry_residency().request(std::span(&request, 1u));
    controller.update();

    REQUIRE(target.update_virtual_mesh(resource, geometry));
    REQUIRE(target.virtual_mesh_content_generation(resource) != old_generation);

    for (std::uint32_t attempt = 0; attempt < 500u && controller.snapshot().stale_completions == 0u; ++attempt)
    {
        controller.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto snapshot = controller.snapshot();
    REQUIRE(snapshot.stale_completions == 1u);
    REQUIRE(snapshot.completed_pages == 0u);
    REQUIRE(snapshot.failed_pages == 0u);
}
