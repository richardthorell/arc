#include <arc/assets/audio.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <utility>

namespace arc::assets
{
namespace
{

std::uint16_t read_u16(std::span<const std::byte> bytes, std::size_t offset) noexcept
{
    return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[offset])) |
           static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[offset + 1])) << 8u;
}

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) noexcept
{
    return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset])) |
           static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset + 1])) << 8u |
           static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset + 2])) << 16u |
           static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset + 3])) << 24u;
}

bool fourcc(std::span<const std::byte> bytes, std::size_t offset, const char (&value)[5]) noexcept
{
    if (offset + 4 > bytes.size()) return false;
    for (std::size_t index = 0; index < 4; ++index)
        if (std::to_integer<unsigned char>(bytes[offset + index]) != static_cast<unsigned char>(value[index]))
            return false;
    return true;
}

sound_asset_parse_result sound_failure(sound_asset_error_code code, std::string message)
{
    return {.code = code, .message = std::move(message)};
}

} // namespace

wav_parse_result parse_wav(std::span<const std::byte> bytes)
{
    const auto failure = [](wav_error_code code, const char* message)
    { return wav_parse_result{.code = code, .message = message}; };

    if (bytes.size() < 12) return failure(wav_error_code::truncated, "WAV source is smaller than the RIFF header");
    if (!fourcc(bytes, 0, "RIFF") || !fourcc(bytes, 8, "WAVE"))
        return failure(wav_error_code::invalid_riff, "WAV source must use a RIFF/WAVE container");

    bool found_format{};
    bool found_data{};
    wav_info info;
    std::size_t offset = 12;
    while (offset + 8 <= bytes.size())
    {
        const std::uint32_t chunk_size = read_u32(bytes, offset + 4);
        const std::size_t payload = offset + 8;
        if (chunk_size > bytes.size() - payload)
            return failure(wav_error_code::truncated, "WAV chunk extends beyond the source buffer");

        if (fourcc(bytes, offset, "fmt "))
        {
            if (chunk_size < 16) return failure(wav_error_code::invalid_format, "WAV fmt chunk is too small");
            const std::uint16_t raw_encoding = read_u16(bytes, payload);
            if (raw_encoding != static_cast<std::uint16_t>(wav_encoding::pcm) &&
                raw_encoding != static_cast<std::uint16_t>(wav_encoding::ieee_float))
                return failure(wav_error_code::unsupported_encoding,
                               "WAV encoding is not PCM or IEEE floating point");

            info.encoding = static_cast<wav_encoding>(raw_encoding);
            info.channels = read_u16(bytes, payload + 2);
            info.sample_rate = read_u32(bytes, payload + 4);
            info.byte_rate = read_u32(bytes, payload + 8);
            info.block_align = read_u16(bytes, payload + 12);
            info.bits_per_sample = read_u16(bytes, payload + 14);
            found_format = true;
        }
        else if (fourcc(bytes, offset, "data"))
        {
            info.data_offset = payload;
            info.data_size = chunk_size;
            found_data = true;
        }

        const std::size_t padded_size = static_cast<std::size_t>(chunk_size) + (chunk_size & 1u);
        if (padded_size > bytes.size() - payload) break;
        offset = payload + padded_size;
    }

    if (!found_format) return failure(wav_error_code::missing_format, "WAV source does not contain a fmt chunk");
    if (!found_data) return failure(wav_error_code::missing_data, "WAV source does not contain a data chunk");
    if (info.channels == 0 || info.sample_rate == 0 || info.bits_per_sample == 0 || info.bits_per_sample % 8 != 0)
        return failure(wav_error_code::invalid_format, "WAV format contains invalid channel/sample metadata");

    const std::uint32_t expected_block_align =
        static_cast<std::uint32_t>(info.channels) * static_cast<std::uint32_t>(info.bits_per_sample / 8u);
    if (info.block_align != expected_block_align || info.byte_rate != info.sample_rate * info.block_align)
        return failure(wav_error_code::invalid_format, "WAV block alignment or byte rate is inconsistent");
    if (info.data_size % info.block_align != 0)
        return failure(wav_error_code::invalid_format, "WAV data size is not aligned to complete sample frames");

    if (info.encoding == wav_encoding::pcm &&
        info.bits_per_sample != 8 && info.bits_per_sample != 16 && info.bits_per_sample != 24 &&
        info.bits_per_sample != 32)
        return failure(wav_error_code::unsupported_encoding, "PCM WAV must use 8, 16, 24, or 32 bits per sample");
    if (info.encoding == wav_encoding::ieee_float && info.bits_per_sample != 32 && info.bits_per_sample != 64)
        return failure(wav_error_code::unsupported_encoding, "Float WAV must use 32 or 64 bits per sample");

    info.frame_count = info.data_size / info.block_align;
    return {.info = info};
}

sound_asset_parse_result parse_sound_asset_json(std::string_view source)
{
    const auto document = nlohmann::json::parse(source.begin(), source.end(), nullptr, false);
    if (document.is_discarded() || !document.is_object())
        return sound_failure(sound_asset_error_code::invalid_json, "Sound asset must be a JSON object");
    if (document.value("kind", std::string{}) != "sound")
        return sound_failure(sound_asset_error_code::invalid_json, "Sound asset kind must be 'sound'");
    if (document.value("version", 0u) != sound_asset::current_version)
        return sound_failure(sound_asset_error_code::unsupported_version, "Unsupported .arcsound schema version");

    sound_asset asset;
    asset.source = document.value("source", std::string{});
    if (asset.source.empty())
        return sound_failure(sound_asset_error_code::missing_source, "Sound asset requires a WAV source");

    const std::filesystem::path source_path(asset.source);
    if (source_path.is_absolute() || source_path.has_root_name())
        return sound_failure(sound_asset_error_code::unsupported_source,
                             "Sound source must use an asset-root-relative path");
    const auto normalized = source_path.lexically_normal();
    auto extension = normalized.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (normalized.empty() || normalized.native().starts_with(std::filesystem::path("..").native()) ||
        extension != ".wav")
        return sound_failure(sound_asset_error_code::unsupported_source,
                             "Sound source must reference a .wav file inside the asset root");

    if (const auto playback = document.find("playback"); playback != document.end())
    {
        if (!playback->is_object())
            return sound_failure(sound_asset_error_code::invalid_playback, "Sound playback settings must be an object");
        asset.loop = playback->value("loop", false);
        asset.volume = playback->value("volume", 1.0f);
        asset.pitch = playback->value("pitch", 1.0f);
    }
    if (!std::isfinite(asset.volume) || asset.volume < 0.0f || !std::isfinite(asset.pitch) || asset.pitch <= 0.0f)
        return sound_failure(sound_asset_error_code::invalid_playback,
                             "Sound volume must be non-negative and pitch must be positive");

    if (const auto spatial = document.find("spatial"); spatial != document.end())
    {
        if (!spatial->is_object())
            return sound_failure(sound_asset_error_code::invalid_spatial, "Sound spatial settings must be an object");
        asset.spatial = spatial->value("enabled", true);
        asset.minimum_distance = spatial->value("minDistance", 1.0f);
        asset.maximum_distance = spatial->value("maxDistance", 25.0f);
    }
    if (!std::isfinite(asset.minimum_distance) || !std::isfinite(asset.maximum_distance) ||
        asset.minimum_distance < 0.0f || asset.maximum_distance <= asset.minimum_distance)
        return sound_failure(sound_asset_error_code::invalid_spatial,
                             "Sound attenuation requires maxDistance greater than non-negative minDistance");

    return {.asset = std::move(asset)};
}

} // namespace arc::assets
