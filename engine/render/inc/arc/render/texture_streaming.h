#pragma once

#include <arc/render/handles.h>
#include <arc/render/texture_artifact.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace arc::render
{

using texture_stream_source_id = std::uint64_t;

struct texture_streaming_capabilities
{
    bool mip_streaming{true};
    bool virtual_textures{};
};

[[nodiscard]] texture_streaming_mode
resolve_texture_streaming_mode(texture_streaming_mode authored, texture_streaming_capabilities capabilities) noexcept;

[[nodiscard]] std::uint32_t texture_requested_mip(std::uint32_t width, std::uint32_t height, std::uint32_t mip_count,
                                                  float projected_texel_extent, float lod_bias = 0.0f) noexcept;

enum class texture_subresource_kind : std::uint8_t
{
    mip,
    tile
};

enum class texture_residency_state : std::uint8_t
{
    nonresident,
    requested,
    loading,
    uploading,
    resident,
    failed
};

/** @brief Immutable renderer registration for a cooked streamable texture. */
struct streamed_texture_descriptor
{
    texture_descriptor texture;
    texture_streaming_mode mode{texture_streaming_mode::streamed_mips};
    texture_stream_source_id source{};
    texture_artifact_index artifact;
    std::uint32_t content_generation{1};
};

struct texture_mip_feedback
{
    texture_handle resource{};
    std::uint32_t content_generation{};
    std::uint32_t desired_mip{};
    float screen_coverage{};
};

struct texture_tile_feedback
{
    texture_handle resource{};
    std::uint32_t content_generation{};
    std::uint32_t mip{};
    std::uint32_t x{};
    std::uint32_t y{};
    float screen_coverage{};
};

struct texture_feedback_readback
{
    std::uint64_t frame_index{};
    std::vector<texture_mip_feedback> mips;
    std::vector<texture_tile_feedback> tiles;
    std::uint32_t overflow{};
};

/** @brief One prioritized range request for a cooked mip or tile. */
struct texture_stream_load
{
    texture_handle resource{};
    std::uint32_t content_generation{};
    texture_stream_source_id source{};
    texture_subresource_kind kind{texture_subresource_kind::mip};
    std::uint32_t mip{};
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint64_t byte_offset{};
    std::uint32_t byte_size{};
    std::uint64_t content_hash{};
    float priority{};
};

/** @brief Bytes read and validated by the streaming controller, ready for backend publication. */
struct texture_stream_upload
{
    texture_handle resource{};
    std::uint32_t content_generation{};
    texture_subresource_kind kind{texture_subresource_kind::mip};
    std::uint32_t mip{};
    std::uint32_t x{};
    std::uint32_t y{};
    std::shared_ptr<const std::vector<std::byte>> bytes;
    std::uint32_t stored_bytes{};
};

/** @brief Backend acknowledgement for a generation-safe upload. */
struct [[nodiscard]] texture_stream_upload_result
{
    texture_handle resource{};
    std::uint32_t content_generation{};
    texture_subresource_kind kind{texture_subresource_kind::mip};
    std::uint32_t mip{};
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t gpu_bytes{};
    bool succeeded{};
};

/** @brief Backend retirement request selected by the common budget policy. */
struct texture_stream_eviction
{
    texture_handle resource{};
    std::uint32_t content_generation{};
    texture_subresource_kind kind{texture_subresource_kind::mip};
    std::uint32_t mip{};
    std::uint32_t x{};
    std::uint32_t y{};
};

struct texture_residency_config
{
    std::uint64_t gpu_budget_bytes{512ull * 1024ull * 1024ull};
    std::uint64_t cpu_cache_budget_bytes{128ull * 1024ull * 1024ull};
    std::uint64_t upload_budget_per_frame{64ull * 1024ull * 1024ull};
    std::uint32_t maximum_requests_per_frame{2048};
    std::uint32_t protected_frame_count{30};
};

struct texture_streaming_resource_snapshot
{
    texture_handle resource{};
    std::uint32_t content_generation{};
    texture_streaming_mode authored_mode{texture_streaming_mode::resident};
    texture_streaming_mode resolved_mode{texture_streaming_mode::resident};
    std::uint32_t requested_mip{};
    std::uint32_t resident_first_mip{};
    std::uint32_t tail_first_mip{};
    std::uint64_t resident_bytes{};
    std::optional<std::uint32_t> forced_mip;
};

struct texture_residency_snapshot
{
    std::uint64_t frame_index{};
    std::uint64_t gpu_budget_bytes{};
    std::uint64_t gpu_resident_bytes{};
    std::uint64_t cpu_cache_budget_bytes{};
    std::uint64_t cpu_cached_bytes{};
    std::uint64_t upload_budget_per_frame{};
    std::uint64_t uploaded_bytes{};
    std::uint32_t resource_count{};
    std::uint32_t streamed_mip_resources{};
    std::uint32_t virtual_texture_resources{};
    std::uint32_t resident_mips{};
    std::uint32_t resident_tiles{};
    std::uint32_t requested_subresources{};
    std::uint32_t failed_subresources{};
    std::uint32_t evictions{};
    std::uint32_t deduplicated_requests{};
    std::uint32_t stale_requests{};
    std::uint32_t feedback_overflow{};
    std::uint32_t parent_fallbacks{};
    bool over_budget{};
};

/** @brief Backend-neutral authority for mip and virtual-tile demand, publication, and eviction. */
class texture_residency_manager
{
public:
    explicit texture_residency_manager(texture_residency_config config = {},
                                       texture_streaming_capabilities capabilities = {});
    ~texture_residency_manager();
    texture_residency_manager(texture_residency_manager&&) noexcept;
    texture_residency_manager& operator=(texture_residency_manager&&) noexcept;
    texture_residency_manager(const texture_residency_manager&) = delete;
    texture_residency_manager& operator=(const texture_residency_manager&) = delete;

    void configure(texture_residency_config config);
    void set_capabilities(texture_streaming_capabilities capabilities);
    void register_resource(texture_handle resource, const streamed_texture_descriptor& descriptor);
    void unregister_resource(texture_handle resource);
    void begin_frame(std::uint64_t frame_index);
    void request(std::span<const texture_mip_feedback> mips, std::span<const texture_tile_feedback> tiles);
    void note_feedback_overflow(std::uint32_t count) noexcept;
    [[nodiscard]] std::vector<texture_stream_load>
    take_load_requests(std::uint32_t maximum_requests = std::numeric_limits<std::uint32_t>::max());
    void mark_loading(const texture_stream_load& load);
    void mark_uploading(const texture_stream_upload& upload);
    void complete(const texture_stream_upload_result& result);
    void fail(const texture_stream_load& load);
    [[nodiscard]] std::vector<texture_stream_eviction> take_evictions();
    [[nodiscard]] bool resident(texture_handle resource, std::uint32_t generation, texture_subresource_kind kind,
                                std::uint32_t mip, std::uint32_t x = 0, std::uint32_t y = 0) const noexcept;
    void note_parent_fallback() noexcept;
    void set_forced_mip(texture_handle resource, std::uint32_t generation, std::optional<std::uint32_t> mip) noexcept;
    [[nodiscard]] texture_residency_snapshot snapshot() const noexcept;
    [[nodiscard]] std::vector<texture_streaming_resource_snapshot> resource_snapshots() const;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace arc::render
