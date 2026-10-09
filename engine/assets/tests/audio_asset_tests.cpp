#include <arc/assets/audio.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace
{

void append_u16(std::vector<std::byte>& bytes, std::uint16_t value)
{
    bytes.push_back(static_cast<std::byte>(value & 0xffu));
    bytes.push_back(static_cast<std::byte>((value >> 8u) & 0xffu));
}

void append_u32(std::vector<std::byte>& bytes, std::uint32_t value)
{
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<std::byte>((value >> shift) & 0xffu));
}

void append_fourcc(std::vector<std::byte>& bytes, std::string_view value)
{
    for (char character : value)
        bytes.push_back(static_cast<std::byte>(character));
}

std::vector<std::byte> pcm_wav()
{
    constexpr std::uint16_t channels = 2;
    constexpr std::uint32_t sample_rate = 48000;
    constexpr std::uint16_t bits = 16;
    constexpr std::uint16_t block_align = channels * (bits / 8);
    constexpr std::uint32_t frame_count = 480;
    constexpr std::uint32_t data_size = frame_count * block_align;

    std::vector<std::byte> bytes;
    append_fourcc(bytes, "RIFF");
    append_u32(bytes, 36u + data_size);
    append_fourcc(bytes, "WAVE");
    append_fourcc(bytes, "fmt ");
    append_u32(bytes, 16);
    append_u16(bytes, 1);
    append_u16(bytes, channels);
    append_u32(bytes, sample_rate);
    append_u32(bytes, sample_rate * block_align);
    append_u16(bytes, block_align);
    append_u16(bytes, bits);
    append_fourcc(bytes, "data");
    append_u32(bytes, data_size);
    bytes.resize(bytes.size() + data_size);
    return bytes;
}

} // namespace

TEST_CASE("WAV parser reads PCM source metadata without decoding")
{
    const auto bytes = pcm_wav();
    const auto result = arc::assets::parse_wav(bytes);

    REQUIRE(result.succeeded());
    CHECK(result.info.encoding == arc::assets::wav_encoding::pcm);
    CHECK(result.info.channels == 2);
    CHECK(result.info.sample_rate == 48000);
    CHECK(result.info.bits_per_sample == 16);
    CHECK(result.info.frame_count == 480);
    CHECK(result.info.data_size == 1920);
    CHECK(result.info.duration_seconds() == 0.01);
}

TEST_CASE("WAV parser rejects malformed and unsupported sources")
{
    const std::array<std::byte, 12> invalid{};
    const auto not_riff = arc::assets::parse_wav(invalid);
    CHECK_FALSE(not_riff.succeeded());
    CHECK(not_riff.code == arc::assets::wav_error_code::invalid_riff);

    auto truncated = pcm_wav();
    truncated.resize(40);
    const auto truncated_result = arc::assets::parse_wav(truncated);
    CHECK_FALSE(truncated_result.succeeded());
    CHECK(truncated_result.code == arc::assets::wav_error_code::truncated);
}

TEST_CASE(".arcsound schema captures source playback and spatial authoring")
{
    constexpr std::string_view source = R"({
      "kind": "sound",
      "version": 1,
      "source": "Audio/Footsteps/footstep_01.wav",
      "playback": { "loop": true, "volume": 0.75, "pitch": 1.1 },
      "spatial": { "enabled": true, "minDistance": 2.0, "maxDistance": 30.0 }
    })";

    const auto result = arc::assets::parse_sound_asset_json(source);
    REQUIRE(result.succeeded());
    CHECK(result.asset.source == "Audio/Footsteps/footstep_01.wav");
    CHECK(result.asset.loop);
    CHECK(result.asset.volume == 0.75f);
    CHECK(result.asset.pitch == 1.1f);
    CHECK(result.asset.spatial);
    CHECK(result.asset.minimum_distance == 2.0f);
    CHECK(result.asset.maximum_distance == 30.0f);
}

TEST_CASE(".arcsound schema rejects unsafe or non-WAV source references")
{
    const auto missing =
        arc::assets::parse_sound_asset_json(R"({"kind":"sound","version":1,"source":""})");
    CHECK_FALSE(missing.succeeded());
    CHECK(missing.code == arc::assets::sound_asset_error_code::missing_source);

    const auto escaped =
        arc::assets::parse_sound_asset_json(R"({"kind":"sound","version":1,"source":"../outside.wav"})");
    CHECK_FALSE(escaped.succeeded());
    CHECK(escaped.code == arc::assets::sound_asset_error_code::unsupported_source);

    const auto compressed =
        arc::assets::parse_sound_asset_json(R"({"kind":"sound","version":1,"source":"Audio/music.ogg"})");
    CHECK_FALSE(compressed.succeeded());
    CHECK(compressed.code == arc::assets::sound_asset_error_code::unsupported_source);
}
