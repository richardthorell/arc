#pragma once

#include <arc/simd/core/ops/scalar.h>

#include <array>
#include <cmath>

namespace arc
{

template <class T, std::size_t N>
inline simd<T, N> sin(const simd<T, N>& value) noexcept
{
    return detail::simd_map(value, [](T lane) { return std::sin(lane); });
}

template <class T, std::size_t N>
inline simd<T, N> cos(const simd<T, N>& value) noexcept
{
    return detail::simd_map(value, [](T lane) { return std::cos(lane); });
}

template <class T, std::size_t N>
inline std::array<simd<T, N>, 2> sincos(const simd<T, N>& value) noexcept
{
    return { sin(value), cos(value) };
}

template <class T, std::size_t N>
inline simd<T, N> tan(const simd<T, N>& value) noexcept
{
    return detail::simd_map(value, [](T lane) { return std::tan(lane); });
}

template <class T, std::size_t N>
inline simd<T, N> exp(const simd<T, N>& value) noexcept
{
    return detail::simd_map(value, [](T lane) { return std::exp(lane); });
}

template <class T, std::size_t N>
inline simd<T, N> log(const simd<T, N>& value) noexcept
{
    return detail::simd_map(value, [](T lane) { return std::log(lane); });
}

template <class T, std::size_t N>
inline simd<T, N> pow(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return detail::simd_map(a, b, [](T x, T y) { return std::pow(x, y); });
}

template <class T, std::size_t N>
inline simd<T, N> atan2(const simd<T, N>& y, const simd<T, N>& x) noexcept
{
    return detail::simd_map(y, x, [](T a, T b) { return std::atan2(a, b); });
}

} // namespace arc
