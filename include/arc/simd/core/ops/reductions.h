#pragma once

#include <arc/simd/core/ops/detail.h>

namespace arc
{

template <class T, std::size_t N>
constexpr T sum(const simd<T, N>& a) noexcept
{
    return reduce(
        a,
        ops_for<simd<T, N>>::sum,
        [](T x, T y)
        {
            return x + y;
        }
    );
}

template <class T, std::size_t N>
constexpr T dot(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return reduce(
        a,
        b,
        ops_for<simd<T, N>>::dot,
        [](T x, T y)
        {
            return x + y;
        }
    );
}

template <class T, std::size_t N>
constexpr T min_element(const simd<T, N>& a) noexcept
{
    return reduce(
        a,
        ops_for<simd<T, N>>::min_element,
        [](T x, T y)
        {
            return x < y ? x : y;
        }
    );
}

template <class T, std::size_t N>
constexpr T max_element(const simd<T, N>& a) noexcept
{
    return reduce(
        a,
        ops_for<simd<T, N>>::max_element,
        [](T x, T y)
        {
            return x > y ? x : y;
        }
    );
}

} // namespace arc
