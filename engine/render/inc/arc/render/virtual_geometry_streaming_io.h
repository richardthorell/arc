#pragma once

#include <arc/io/io.h>
#include <arc/render/virtual_geometry.h>
#include <arc/render/virtual_geometry_artifact.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace arc::render
{

class renderer;

/** @brief Non-blocking virtual-geometry IO/decode diagnostics. */
struct virtual_geometry_streaming_io_snapshot
{
    std::uint32_t in_flight_reads{};
    std::uint32_t in_flight_decodes{};
    std::uint64_t read_bytes{};
    std::uint64_t decoded_bytes{};
    std::uint32_t completed_pages{};
    std::uint32_t failed_pages{};
    std::uint32_t stale_completions{};
};

/** @brief Opaque asynchronous source for one requested virtual-geometry page. */
class virtual_geometry_page_source
{
public:
    virtual ~virtual_geometry_page_source() = default;

    [[nodiscard]] virtual jobs::job_future<io::file_result<io::file_buffer>>
    read_page(const virtual_geometry_page_load& load, jobs::cancellation_token cancellation = {}) = 0;
};

/**
 * @brief Async file-backed virtual-geometry source using artifact-relative page ranges.
 *
 * One registration represents a single cooked virtual-geometry artifact, including its physical package range and
 * validated M2.2 page table. Renderer load offsets are never treated as file offsets; logical page IDs are resolved
 * through the cooked artifact table so physical Morton ordering remains transparent to residency.
 */
class filesystem_virtual_geometry_artifact_source final : public virtual_geometry_page_source
{
public:
    explicit filesystem_virtual_geometry_artifact_source(io::async_file_service& files);
    ~filesystem_virtual_geometry_artifact_source();
    filesystem_virtual_geometry_artifact_source(filesystem_virtual_geometry_artifact_source&&) noexcept;
    filesystem_virtual_geometry_artifact_source& operator=(filesystem_virtual_geometry_artifact_source&&) noexcept;
    filesystem_virtual_geometry_artifact_source(const filesystem_virtual_geometry_artifact_source&) = delete;
    filesystem_virtual_geometry_artifact_source& operator=(const filesystem_virtual_geometry_artifact_source&) = delete;

    void register_package_range(virtual_mesh_handle resource, std::uint32_t resource_generation,
                                std::filesystem::path package, std::uint64_t artifact_base_offset,
                                std::uint64_t artifact_size,
                                std::vector<virtual_geometry_artifact_page_range> pages);
    void unregister(virtual_mesh_handle resource);

    [[nodiscard]] jobs::job_future<io::file_result<io::file_buffer>>
    read_page(const virtual_geometry_page_load& load, jobs::cancellation_token cancellation = {}) override;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

/**
 * @brief Non-blocking bridge from renderer residency requests to IO-thread reads, worker decompression and upload.
 *
 * update() only polls ready futures. It never waits for file IO or decompression and therefore remains safe to call
 * from the render loop. Generation changes turn outstanding work into stale completions instead of publishing old
 * terrain/geometry data into a replacement resource.
 */
class virtual_geometry_streaming_controller
{
public:
    virtual_geometry_streaming_controller(renderer& renderer, virtual_geometry_page_source& source,
                                          jobs::job_system& jobs, std::uint32_t maximum_in_flight = 2048);
    ~virtual_geometry_streaming_controller();
    virtual_geometry_streaming_controller(virtual_geometry_streaming_controller&&) noexcept;
    virtual_geometry_streaming_controller& operator=(virtual_geometry_streaming_controller&&) noexcept;
    virtual_geometry_streaming_controller(const virtual_geometry_streaming_controller&) = delete;
    virtual_geometry_streaming_controller& operator=(const virtual_geometry_streaming_controller&) = delete;

    void update(const jobs::cancellation_token& cancellation = {});
    [[nodiscard]] virtual_geometry_streaming_io_snapshot snapshot() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace arc::render
