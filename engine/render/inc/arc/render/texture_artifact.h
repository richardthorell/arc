#pragma once

#include <arc/core/result.h>
#include <arc/render/material.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace arc::render
{

inline constexpr std::uint32_t texture_artifact_schema_version = 3;
inline constexpr std::uint32_t texture_artifact_alignment = 4096;
inline constexpr std::uint32_t virtual_texture_tile_size = 128;
inline constexpr std::uint32_t virtual_texture_tile_border = 4;

/** @brief Runtime storage policy authored for a conventional 2D texture. */
enum class texture_streaming_mode : std::uint8_t
{
    resident,
    streamed_mips,
    virtual_tiles
};

enum class texture_filter_mode : std::uint8_t
{
    nearest,
    linear
};

enum class texture_mip_filter_mode : std::uint8_t
{
    nearest,
    linear
};

enum class texture_address_mode : std::uint8_t
{
    repeat,
    clamp_to_edge,
    mirrored_repeat
};

enum class texture_power_of_two_policy : std::uint8_t
{
    preserve,
    resize_down,
    resize_up
};

enum class texture_compression_policy : std::uint8_t
{
    automatic,
    color,
    normal,
    mask,
    hdr,
    uncompressed
};

/** @brief Resolved deterministic import policy embedded into a cooked texture artifact. */
struct texture_artifact_metadata
{
    std::uint32_t source_width{};
    std::uint32_t source_height{};
    std::uint32_t requested_max_size{};
    std::uint32_t resolved_max_size{};
    texture_power_of_two_policy power_of_two{texture_power_of_two_policy::preserve};
    texture_compression_policy compression{texture_compression_policy::automatic};
    texture_filter_mode min_filter{texture_filter_mode::linear};
    texture_filter_mode mag_filter{texture_filter_mode::linear};
    texture_mip_filter_mode mip_filter{texture_mip_filter_mode::linear};
    texture_address_mode wrap_u{texture_address_mode::repeat};
    texture_address_mode wrap_v{texture_address_mode::repeat};
    float anisotropy{1.0f};
    float lod_bias{};
    float minimum_lod{};
    float maximum_lod{1000.0f};
    float alpha_coverage_threshold{0.5f};
    bool generated_mips{};
    bool resized{};
    bool power_of_two_adjusted{};
    bool normal_mips_renormalized{};
    bool alpha_coverage_preserved{};
};

enum class texture_artifact_error_code : std::uint8_t
{
    invalid_data,
    unsupported_version,
    unsupported_texture,
    out_of_bounds,
    integrity_failure,
    size_overflow
};

struct texture_artifact_error
{
    texture_artifact_error_code code{texture_artifact_error_code::invalid_data};
    std::string message;
};

/** @brief Independently readable conventional mip payload. */
struct texture_artifact_mip_range
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t offset{};
    std::uint32_t stored_size{};
    std::uint32_t decoded_size{};
    std::uint64_t content_hash{};
};

/** @brief Independently readable virtual-texture tile including its cooked gutter. */
struct texture_artifact_tile_range
{
    std::uint32_t mip{};
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t offset{};
    std::uint32_t stored_size{};
    std::uint32_t decoded_size{};
    std::uint64_t content_hash{};
};

/** @brief Validated index for one range-readable `.arctex` artifact. */
struct texture_artifact_index
{
    std::uint32_t schema_version{};
    texture_streaming_mode mode{texture_streaming_mode::resident};
    texture_format format{texture_format::rgba8_srgb};
    texture_color_space color_space{texture_color_space::srgb};
    texture_semantic semantic{texture_semantic::generic_color};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t mip_count{};
    std::uint32_t tail_first_mip{};
    std::uint32_t tile_size{};
    std::uint32_t tile_border{};
    std::uint64_t table_end{};
    std::uint64_t artifact_size{};
    std::vector<texture_artifact_mip_range> mips;
    std::vector<texture_artifact_tile_range> tiles;
    texture_artifact_metadata metadata{};
};

using texture_artifact_bytes_result = core::result<std::vector<std::byte>, texture_artifact_error>;
using texture_artifact_index_result = core::result<texture_artifact_index, texture_artifact_error>;

/** @brief Encode a deterministic range-readable texture artifact. */
[[nodiscard]] texture_artifact_bytes_result encode_texture_artifact(const texture_data& texture,
                                                                    texture_streaming_mode mode,
                                                                    texture_artifact_metadata metadata = {});

/** @brief Validate a complete texture artifact and return its range index. */
[[nodiscard]] texture_artifact_index_result inspect_texture_artifact(std::span<const std::byte> bytes);

/** @brief Integrity-check and copy one conventional mip payload. */
[[nodiscard]] texture_artifact_bytes_result
read_texture_artifact_mip(std::span<const std::byte> bytes, const texture_artifact_index& index, std::uint32_t mip);

/** @brief Integrity-check and copy one virtual tile payload. */
[[nodiscard]] texture_artifact_bytes_result
read_texture_artifact_tile(std::span<const std::byte> bytes, const texture_artifact_index& index, std::uint32_t tile);

} // namespace arc::render
