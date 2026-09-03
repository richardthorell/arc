#pragma once

#include <arc/io/io.h>
#include <arc/render/texture_streaming.h>

#include <cstdint>
#include <filesystem>
#include <memory>

namespace arc::render
{

class renderer;

/** @brief Opaque range source used by texture policy without exposing asset paths. */
class texture_artifact_source
{
public:
    virtual ~texture_artifact_source() = default;
    [[nodiscard]] virtual jobs::job_future<io::file_result<io::file_buffer>>
    read_range(texture_stream_source_id source, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {}) = 0;
};

/**
 * @brief Async file-backed source map for loose artifacts and package subranges.
 *
 * The renderer sees only source IDs and artifact-relative ranges. Package records
 * are represented by a file plus a validated base offset and extent.
 */
class filesystem_texture_artifact_source final : public texture_artifact_source
{
public:
    explicit filesystem_texture_artifact_source(io::async_file_service& files);
    ~filesystem_texture_artifact_source();
    filesystem_texture_artifact_source(filesystem_texture_artifact_source&&) noexcept;
    filesystem_texture_artifact_source& operator=(filesystem_texture_artifact_source&&) noexcept;
    filesystem_texture_artifact_source(const filesystem_texture_artifact_source&) = delete;
    filesystem_texture_artifact_source& operator=(const filesystem_texture_artifact_source&) = delete;
    void register_file(texture_stream_source_id source, std::filesystem::path path, std::uint64_t size);
    void register_package_range(texture_stream_source_id source, std::filesystem::path package,
                                std::uint64_t base_offset, std::uint64_t size);
    void unregister(texture_stream_source_id source);
    [[nodiscard]] jobs::job_future<io::file_result<io::file_buffer>>
    read_range(texture_stream_source_id source, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {}) override;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

/** @brief Non-blocking bridge from prioritized renderer loads to range I/O and publication. */
class texture_streaming_controller
{
public:
    texture_streaming_controller(renderer& renderer, texture_artifact_source& source,
                                 std::uint32_t maximum_in_flight = 2048);
    ~texture_streaming_controller();
    texture_streaming_controller(texture_streaming_controller&&) noexcept;
    texture_streaming_controller& operator=(texture_streaming_controller&&) noexcept;
    texture_streaming_controller(const texture_streaming_controller&) = delete;
    texture_streaming_controller& operator=(const texture_streaming_controller&) = delete;
    void update(const jobs::cancellation_token& cancellation = {});
    [[nodiscard]] texture_streaming_io_snapshot snapshot() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace arc::render
