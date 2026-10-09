#pragma once

#include <arc/assets/assets.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace arc::assets
{

enum class wav_encoding : std::uint16_t
{
    pcm = 1,
    ieee_float = 3
};

struct wav_info
{
    wav_encoding encoding{wav_encoding::pcm};
    std::uint16_t channels{};
    std::uint32_t sample_rate{};
    std::uint32_t byte_rate{};
    std::uint16_t block_align{};
    std::uint16_t bits_per_sample{};
    std::uint64_t frame_count{};
    std::size_t data_offset{};
    std::size_t data_size{};

    [[nodiscard]] double duration_seconds() const noexcept
    {
        return sample_rate == 0 ? 0.0 : static_cast<double>(frame_count) / static_cast<double>(sample_rate);
    }
};

enum class wav_error_code : std::uint8_t
{
    none,
    truncated,
    invalid_riff,
    missing_format,
    missing_data,
    unsupported_encoding,
    invalid_format
};

struct wav_parse_result
{
    wav_info info;
    wav_error_code code{wav_error_code::none};
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return code == wav_error_code::none;
    }
};

[[nodiscard]] wav_parse_result parse_wav(std::span<const std::byte> bytes);

struct wav_source_data
{
    wav_info info;
    std::vector<std::byte> bytes;
};

struct sound_asset
{
    static constexpr std::uint32_t current_version = 1;

    std::uint32_t version{current_version};
    std::string source;
    bool loop{};
    float volume{1.0f};
    float pitch{1.0f};
    bool spatial{true};
    float minimum_distance{1.0f};
    float maximum_distance{25.0f};
};

enum class sound_asset_error_code : std::uint8_t
{
    none,
    invalid_json,
    unsupported_version,
    missing_source,
    unsupported_source,
    invalid_playback,
    invalid_spatial
};

struct sound_asset_parse_result
{
    sound_asset asset;
    sound_asset_error_code code{sound_asset_error_code::none};
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return code == sound_asset_error_code::none;
    }
};

[[nodiscard]] sound_asset_parse_result parse_sound_asset_json(std::string_view source);

} // namespace arc::assets
