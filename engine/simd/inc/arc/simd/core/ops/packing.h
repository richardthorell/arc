#pragma once

#include <arc/simd/core/ops/scalar.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace arc
{

namespace detail
{

inline std::uint32_t pack_unorm8_lane(float value) noexcept
{
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint32_t>(std::round(clamped * 255.0f));
}

inline std::uint32_t pack_snorm8_lane(float value) noexcept
{
    const float clamped = std::clamp(value, -1.0f, 1.0f);
    const auto signed_value = static_cast<std::int32_t>(std::round(clamped * 127.0f));
    return static_cast<std::uint32_t>(static_cast<std::uint8_t>(signed_value));
}

inline std::uint16_t float_to_half_lane(float value) noexcept
{
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::uint32_t mantissa = bits & 0x007FFFFFu;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;

    if (exponent <= 0)
    {
        if (exponent < -10)
            return static_cast<std::uint16_t>(sign);
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000u) >> 13));
    }

    if (exponent >= 31)
        return static_cast<std::uint16_t>(sign | 0x7C00u);

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | ((mantissa + 0x00001000u) >> 13));
}

inline float half_to_float_lane(std::uint16_t value) noexcept
{
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::int32_t exponent = static_cast<std::int32_t>((value >> 10) & 0x1Fu);
    std::uint32_t mantissa = value & 0x03FFu;

    if (exponent == 0)
    {
        if (mantissa == 0)
            return std::bit_cast<float>(sign);
        while ((mantissa & 0x0400u) == 0)
        {
            mantissa <<= 1;
            --exponent;
        }
        ++exponent;
        mantissa &= ~0x0400u;
    }
    else if (exponent == 31)
    {
        return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
    }

    exponent = exponent + (127 - 15);
    return std::bit_cast<float>(sign | (static_cast<std::uint32_t>(exponent) << 23) | (mantissa << 13));
}

} // namespace detail

template <std::size_t N>
inline simd<uint32_t, N> pack_unorm8(const simd<float, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<uint32_t, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = detail::pack_unorm8_lane(lanes[i]);
    return detail::simd_from_array<uint32_t, N>(result);
}

template <std::size_t N>
inline simd<float, N> unpack_unorm8(const simd<uint32_t, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<float, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = static_cast<float>(lanes[i] & 0xFFu) / 255.0f;
    return detail::simd_from_array<float, N>(result);
}

template <std::size_t N>
inline simd<uint32_t, N> pack_snorm8(const simd<float, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<uint32_t, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = detail::pack_snorm8_lane(lanes[i]);
    return detail::simd_from_array<uint32_t, N>(result);
}

template <std::size_t N>
inline simd<float, N> unpack_snorm8(const simd<uint32_t, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<float, N> result{};
    for (std::size_t i = 0; i < N; ++i)
    {
        const auto signed_value = static_cast<int8_t>(lanes[i] & 0xFFu);
        result[i] = std::max(-1.0f, static_cast<float>(signed_value) / 127.0f);
    }
    return detail::simd_from_array<float, N>(result);
}

template <std::size_t N>
inline simd<uint32_t, N> pack_rgba8(
    const simd<float, N>& r,
    const simd<float, N>& g,
    const simd<float, N>& b,
    const simd<float, N>& a) noexcept
{
    auto rv = detail::simd_to_array(r);
    auto gv = detail::simd_to_array(g);
    auto bv = detail::simd_to_array(b);
    auto av = detail::simd_to_array(a);
    std::array<uint32_t, N> result{};

    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] =
            (detail::pack_unorm8_lane(rv[i]) << 0) |
            (detail::pack_unorm8_lane(gv[i]) << 8) |
            (detail::pack_unorm8_lane(bv[i]) << 16) |
            (detail::pack_unorm8_lane(av[i]) << 24);
    }

    return detail::simd_from_array<uint32_t, N>(result);
}

template <std::size_t N>
inline std::array<simd<float, N>, 4> unpack_rgba8(const simd<uint32_t, N>& rgba) noexcept
{
    auto lanes = detail::simd_to_array(rgba);
    std::array<float, N> r{};
    std::array<float, N> g{};
    std::array<float, N> b{};
    std::array<float, N> a{};

    for (std::size_t i = 0; i < N; ++i)
    {
        r[i] = static_cast<float>((lanes[i] >> 0) & 0xFFu) / 255.0f;
        g[i] = static_cast<float>((lanes[i] >> 8) & 0xFFu) / 255.0f;
        b[i] = static_cast<float>((lanes[i] >> 16) & 0xFFu) / 255.0f;
        a[i] = static_cast<float>((lanes[i] >> 24) & 0xFFu) / 255.0f;
    }

    return {
        detail::simd_from_array<float, N>(r),
        detail::simd_from_array<float, N>(g),
        detail::simd_from_array<float, N>(b),
        detail::simd_from_array<float, N>(a)
    };
}

template <std::size_t N>
inline simd<uint32_t, N> float_to_half(const simd<float, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<uint32_t, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = detail::float_to_half_lane(lanes[i]);
    return detail::simd_from_array<uint32_t, N>(result);
}

template <std::size_t N>
inline simd<float, N> half_to_float(const simd<uint32_t, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<float, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = detail::half_to_float_lane(static_cast<std::uint16_t>(lanes[i] & 0xFFFFu));
    return detail::simd_from_array<float, N>(result);
}

template <std::size_t N>
inline simd<uint32_t, N> pack_bgra8(
    const simd<float, N>& r,
    const simd<float, N>& g,
    const simd<float, N>& b,
    const simd<float, N>& a) noexcept
{
    return pack_rgba8(b, g, r, a);
}

template <std::size_t N>
inline simd<uint32_t, N> pack_argb8(
    const simd<float, N>& r,
    const simd<float, N>& g,
    const simd<float, N>& b,
    const simd<float, N>& a) noexcept
{
    auto rv = detail::simd_to_array(r);
    auto gv = detail::simd_to_array(g);
    auto bv = detail::simd_to_array(b);
    auto av = detail::simd_to_array(a);
    std::array<uint32_t, N> result{};

    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] =
            (detail::pack_unorm8_lane(av[i]) << 24) |
            (detail::pack_unorm8_lane(rv[i]) << 16) |
            (detail::pack_unorm8_lane(gv[i]) << 8) |
            (detail::pack_unorm8_lane(bv[i]) << 0);
    }

    return detail::simd_from_array<uint32_t, N>(result);
}

} // namespace arc
