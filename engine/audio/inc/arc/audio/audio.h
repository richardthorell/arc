#pragma once

#include <arc/math/vector.h>

#include <cstdint>
#include <limits>
#include <string_view>

namespace arc::audio
{

using source_id = std::uint64_t;
using bus_id = std::uint64_t;
inline constexpr source_id invalid_source_id = 0;
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
    bus_id parent = 0;
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

[[nodiscard]] validation_error validate(const stream_format& format) noexcept;
[[nodiscard]] validation_error validate(const source_definition& definition) noexcept;
[[nodiscard]] validation_error validate(const listener_state& listener) noexcept;
[[nodiscard]] validation_error validate(const bus_definition& bus) noexcept;

} // namespace arc::audio
