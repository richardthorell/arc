#pragma once

#include <arc/simd/core/ops/detail.h>
#include <arc/simd/core/ops/scalar.h>

#include <type_traits>

namespace arc
{

/// @brief Return the elementwise sum of two SIMD vectors.
template <class T, std::size_t N>
constexpr simd<T, N> add(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::add, a, b);
}

/// @brief Return the elementwise difference of two SIMD vectors.
template <class T, std::size_t N>
constexpr simd<T, N> sub(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::sub, a, b);
}

/// @brief Return the elementwise product of two SIMD vectors.
template <class T, std::size_t N>
constexpr simd<T, N> mul(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_integral_v<T>)
        return detail::simd_map(a, b, [](T x, T y) { return static_cast<T>(x * y); });
    else
        return apply<simd<T, N>>(ops_for<simd<T, N>>::mul, a, b);
}

/// @brief Return the elementwise quotient of two SIMD vectors.
template <class T, std::size_t N>
constexpr simd<T, N> div(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_integral_v<T>)
        return detail::simd_map(a, b, [](T x, T y) { return static_cast<T>(x / y); });
    else
        return apply<simd<T, N>>(ops_for<simd<T, N>>::div, a, b);
}

/// @brief Return the elementwise negation of a SIMD vector.
template <class T, std::size_t N>
constexpr simd<T, N> neg(const simd<T, N>& a) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::neg, a);
}

} // namespace arc
