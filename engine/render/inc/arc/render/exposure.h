#pragma once

#include <cstdint>

namespace arc::render
{

enum class exposure_mode : std::uint8_t
{
    manual,
    automatic
};

enum class exposure_metering_mode : std::uint8_t
{
    average,
    center_weighted
};

struct exposure_settings
{
    exposure_mode mode{ exposure_mode::automatic };
    exposure_metering_mode metering{ exposure_metering_mode::average };
    float manual_ev100{ 10.0f };
    float compensation_ev{};
    float minimum_ev100{ -8.0f };
    float maximum_ev100{ 20.0f };
    float brighten_speed{ 3.0f };
    float darken_speed{ 1.0f };
};

struct exposure_state
{
    float ev100{ 10.0f };
    float multiplier{ 1.0f };
    bool valid{};
};

} // namespace arc::render
