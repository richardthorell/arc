#pragma once

#include <arc/simd/core/ops/bitwise.h>
#include <arc/simd/core/ops/mask_helpers.h>
#include <arc/simd/core/ops/scalar.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace arc
{

template <class To, class From, std::size_t N>
constexpr simd<To, N> convert(const simd<From, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<To, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = static_cast<To>(lanes[i]);
    return detail::simd_from_array<To, N>(result);
}

template <class T, std::size_t N>
constexpr simd<float, N> to_float(const simd<T, N>& value) noexcept
{
    return convert<float>(value);
}

template <class T, std::size_t N>
constexpr simd<int32_t, N> to_int32(const simd<T, N>& value) noexcept
{
    return convert<int32_t>(value);
}

template <class T, std::size_t N>
constexpr simd<uint32_t, N> to_uint32(const simd<T, N>& value) noexcept
{
    return convert<uint32_t>(value);
}

template <class To, class From, std::size_t N>
    requires (sizeof(To) == sizeof(From)) && std::is_trivially_copyable_v<To> && std::is_trivially_copyable_v<From>
constexpr simd<To, N> bit_cast(const simd<From, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<To, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = std::bit_cast<To>(lanes[i]);
    return detail::simd_from_array<To, N>(result);
}

template <class T, std::size_t N>
inline simd_mask<N> is_nan(const simd<T, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    auto mask = simd_mask<N>(false);
    for (std::size_t i = 0; i < N; ++i)
        if constexpr (std::is_floating_point_v<T>)
            if (std::isnan(lanes[i]))
                mask = bitwise_or(mask, range_mask<N>(i, i + 1));
    return mask;
}

template <class T, std::size_t N>
inline simd_mask<N> is_finite(const simd<T, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    auto mask = simd_mask<N>(false);
    for (std::size_t i = 0; i < N; ++i)
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            if (std::isfinite(lanes[i]))
                mask = bitwise_or(mask, range_mask<N>(i, i + 1));
        }
        else
        {
            mask = bitwise_or(mask, range_mask<N>(i, i + 1));
        }
    }
    return mask;
}

template <class T, std::size_t N>
inline simd_mask<N> signbit(const simd<T, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    auto mask = simd_mask<N>(false);
    for (std::size_t i = 0; i < N; ++i)
        if (std::signbit(lanes[i]))
            mask = bitwise_or(mask, range_mask<N>(i, i + 1));
    return mask;
}

template <class T, std::size_t N>
inline simd<T, N> copysign(const simd<T, N>& magnitude, const simd<T, N>& sign) noexcept
{
    auto mag = detail::simd_to_array(magnitude);
    auto sgn = detail::simd_to_array(sign);
    std::array<T, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = std::copysign(mag[i], sgn[i]);
    return detail::simd_from_array<T, N>(result);
}

} // namespace arc
