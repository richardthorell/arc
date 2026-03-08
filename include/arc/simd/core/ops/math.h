#pragma once

#include <arc/simd/core/ops/arithmetic.h>
#include <arc/simd/core/ops/blend.h>
#include <arc/simd/core/ops/memory.h>
#include <arc/simd/core/ops/minmax.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> abs(const simd<T, N>& value) noexcept
{
    return max(value, neg(value));
}

template <class T, std::size_t N>
constexpr simd<T, N> clamp(const simd<T, N>& value, const simd<T, N>& min_value, const simd<T, N>& max_value) noexcept
{
    return max(min_value, min(max_value, value));
}

template <class T, std::size_t N>
constexpr simd<T, N> clamp_min(const simd<T, N>& value, const simd<T, N>& min_value) noexcept
{
    return max(min_value, value);
}

template <class T, std::size_t N>
constexpr simd<T, N> clamp_max(const simd<T, N>& value, const simd<T, N>& max_value) noexcept
{
    return min(max_value, value);
}

template <class T, std::size_t N>
constexpr simd<T, N> saturate(const simd<T, N>& value) noexcept
{
    return clamp(value, fill<T, N>(0), fill<T, N>(1));
}

template <class T, std::size_t N>
constexpr simd<T, N> sqrt(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::sqrt);
}

template <class T, std::size_t N>
constexpr simd<T, N> rsqrt(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::rsqrt);
}

template <class T, std::size_t N>
constexpr simd<T, N> reciprocal(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::reciprocal);
}

template <class T, std::size_t N>
constexpr simd<T, N> fma(const simd<T, N>& a, const simd<T, N>& b, const simd<T, N>& c) noexcept
{
    return apply(a, b, c, ops_for<simd<T, N>>::fma);
}

template <class T, std::size_t N>
constexpr simd<T, N> round(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::round);
}

template <class T, std::size_t N>
constexpr simd<T, N> floor(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::floor);
}

template <class T, std::size_t N>
constexpr simd<T, N> ceil(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::ceil);
}

template <class T, std::size_t N>
constexpr simd<T, N> trunc(const simd<T, N>& value) noexcept
{
    return apply(value, ops_for<simd<T, N>>::trunc);
}

} // namespace arc
