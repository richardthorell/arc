#pragma once

#include <arc/simd/core/ops/conversions.h>
#include <arc/simd/core/ops/scalar.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace arc
{

template <class T, std::size_t N>
    requires std::is_integral_v<T>
constexpr simd<T, N> saturating_add(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    auto av = detail::simd_to_array(a);
    auto bv = detail::simd_to_array(b);
    std::array<T, N> result{};

    for (std::size_t i = 0; i < N; ++i)
    {
        if constexpr (std::is_unsigned_v<T>)
        {
            const T sum = static_cast<T>(av[i] + bv[i]);
            result[i] = sum < av[i] ? std::numeric_limits<T>::max() : sum;
        }
        else
        {
            using wide_t = std::conditional_t<(sizeof(T) < sizeof(int64_t)), int64_t, long long>;
            const auto sum = static_cast<wide_t>(av[i]) + static_cast<wide_t>(bv[i]);
            result[i] = static_cast<T>(std::clamp<wide_t>(
                sum,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max()));
        }
    }

    return detail::simd_from_array<T, N>(result);
}

template <class T, std::size_t N>
    requires std::is_integral_v<T>
constexpr simd<T, N> saturating_sub(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    auto av = detail::simd_to_array(a);
    auto bv = detail::simd_to_array(b);
    std::array<T, N> result{};

    for (std::size_t i = 0; i < N; ++i)
    {
        if constexpr (std::is_unsigned_v<T>)
        {
            result[i] = av[i] < bv[i] ? T{} : static_cast<T>(av[i] - bv[i]);
        }
        else
        {
            using wide_t = std::conditional_t<(sizeof(T) < sizeof(int64_t)), int64_t, long long>;
            const auto diff = static_cast<wide_t>(av[i]) - static_cast<wide_t>(bv[i]);
            result[i] = static_cast<T>(std::clamp<wide_t>(
                diff,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max()));
        }
    }

    return detail::simd_from_array<T, N>(result);
}

template <class To, class From, std::size_t N>
    requires std::is_arithmetic_v<To> && std::is_arithmetic_v<From>
constexpr simd<To, N> narrow(const simd<From, N>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<To, N> result{};
    for (std::size_t i = 0; i < N; ++i)
    {
        const auto clamped = std::clamp(
            lanes[i],
            static_cast<From>(std::numeric_limits<To>::lowest()),
            static_cast<From>(std::numeric_limits<To>::max()));
        result[i] = static_cast<To>(clamped);
    }
    return detail::simd_from_array<To, N>(result);
}

template <class To, class From, std::size_t N>
    requires std::is_arithmetic_v<To> && std::is_arithmetic_v<From>
constexpr simd<To, N> widen(const simd<From, N>& value) noexcept
{
    return convert<To>(value);
}

} // namespace arc
