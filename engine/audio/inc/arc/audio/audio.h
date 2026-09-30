#pragma once

#include <arc/math/vector.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

namespace arc::audio
{

using device_id = std::uint64_t;
using clip_id = std::uint64_t;
using source_id = std::uint64_t;
using bus_id = std::uint64_t;
using listener_id = std::uint64_t;
inline constexpr device_id invalid_device_id = 0;
inline constexpr clip_id invalid_clip_id = 0;
inline constexpr source_id invalid_source_id = 0;
inline constexpr bus_id invalid_bus_id = 0;
inline constexpr listener_id invalid_listener_id = 0;
inline constexpr bus_id master_bus_id = 1;

enum class sample_format : std::uint8_t
{
    signed_16,
    signed_24,
    float_32
};

struct stream_format
{
    std::uint32_t sample_rate = 48000;
    std::uint16_t channel_count = 2;
    sample_format format = sample_format::float_32;
};

struct source_definition
{
    bus_id bus = master_bus_id;
    float gain = 1.0F;
    float pitch = 1.0F;
    bool looping = false;
    bool spatialized = false;
    float min_distance = 1.0F;
    float max_distance = 100.0F;
};

struct listener_state
{
    math::vector<float, 3> position{};
    math::vector<float, 3> forward{0.0F, 0.0F, -1.0F};
    math::vector<float, 3> up{0.0F, 1.0F, 0.0F};
    math::vector<float, 3> velocity{};
};

struct source_state
{
    math::vector<float, 3> position{};
    math::vector<float, 3> velocity{};
};

struct bus_definition
{
    bus_id id = master_bus_id;
    bus_id parent = invalid_bus_id;
    std::string_view name = "Master";
    float gain = 1.0F;
};

enum class validation_error : std::uint8_t
{
    none,
    invalid_stream_format,
    invalid_source,
    invalid_listener,
    invalid_bus
};

enum class backend_type : std::uint8_t
{
    miniaudio
};

enum class device_mode : std::uint8_t
{
    default_output,
    null_output
};

enum class runtime_error : std::uint8_t
{
    none,
    invalid_configuration,
    backend_unavailable,
    backend_initialization_failed,
    device_initialization_failed,
    device_start_failed
};

struct runtime_config
{
    stream_format output{};
    device_mode device = device_mode::default_output;
};

// Gameplay and editor code cross this ARC-owned boundary rather than touching
// backend objects directly. The backend remains private so later playback and
// mixer commands can be queued without exposing real-time implementation state.
class audio_runtime final
{
public:
    explicit audio_runtime(backend_type backend = backend_type::miniaudio);
    ~audio_runtime();

    audio_runtime(const audio_runtime&) = delete;
    audio_runtime& operator=(const audio_runtime&) = delete;
    audio_runtime(audio_runtime&&) noexcept;
    audio_runtime& operator=(audio_runtime&&) noexcept;

    [[nodiscard]] runtime_error initialize(const runtime_config& config = {}) noexcept;
    void shutdown() noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] backend_type backend() const noexcept;
    [[nodiscard]] device_id playback_device() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

[[nodiscard]] validation_error validate(const stream_format& format) noexcept;
[[nodiscard]] validation_error validate(const source_definition& definition) noexcept;
[[nodiscard]] validation_error validate(const listener_state& listener) noexcept;
[[nodiscard]] validation_error validate(const bus_definition& bus) noexcept;

} // namespace arc::audio
