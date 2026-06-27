#pragma once

#include <arc/simd/core/ops/arithmetic.h>
#include <arc/simd/core/ops/blend.h>
#include <arc/simd/core/ops/memory.h>
#include <arc/simd/core/ops/minmax.h>

namespace arc
{

/**
 * @brief Computes the absolute value of each element in the SIMD vector.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to compute absolute values for.
 * 
 * @return A SIMD vector containing the absolute values.
 */
template <class T, std::size_t N>
constexpr simd<T, N> abs(const simd<T, N>& value) noexcept
{
    return max(value, neg(value));
}

/**
 * @brief Clamps each element of the SIMD vector between the corresponding elements of min_value and max_value.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to clamp.
 * @param min_value The minimum values for clamping.
 * @param max_value The maximum values for clamping.
 * 
 * @return A SIMD vector with elements clamped to [min_value, max_value].
 */
template <class T, std::size_t N>
constexpr simd<T, N> clamp(const simd<T, N>& value, const simd<T, N>& min_value, const simd<T, N>& max_value) noexcept
{
    return max(min_value, min(max_value, value));
}

/**
 * @brief Clamps each element of the SIMD vector to be at least the corresponding element of min_value.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to clamp.
 * @param min_value The minimum values for clamping.
 * 
 * @return A SIMD vector with elements clamped to [min_value, infinity).
 */
template <class T, std::size_t N>
constexpr simd<T, N> clamp_min(const simd<T, N>& value, const simd<T, N>& min_value) noexcept
{
    return max(min_value, value);
}

/**
 * @brief Clamps each element of the SIMD vector to be at most the corresponding element of max_value.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to clamp.
 * @param max_value The maximum values for clamping.
 * 
 * @return A SIMD vector with elements clamped to (-infinity, max_value].
 */
template <class T, std::size_t N>
constexpr simd<T, N> clamp_max(const simd<T, N>& value, const simd<T, N>& max_value) noexcept
{
    return min(max_value, value);
}

/**
 * @brief Saturates each element of the SIMD vector to the range [0, 1].
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to saturate.
 * 
 * @return A SIMD vector with elements clamped to [0, 1].
 */
template <class T, std::size_t N>
constexpr simd<T, N> saturate(const simd<T, N>& value) noexcept
{
    return clamp(value, fill<T, N>(0), fill<T, N>(1));
}

/**
 * @brief Computes the square root of each element in the SIMD vector.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to compute square roots for.
 * 
 * @return A SIMD vector containing the square roots.
 */
template <class T, std::size_t N>
constexpr simd<T, N> sqrt(const simd<T, N>& value) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::sqrt, value);
}

/**
 * @brief Computes the reciprocal square root (1/sqrt(x)) of each element in the SIMD vector.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to compute reciprocal square roots for.
 * 
 * @return A SIMD vector containing the reciprocal square roots.
 */
template <class T, std::size_t N>
constexpr simd<T, N> rsqrt(const simd<T, N>& value) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::rsqrt, value);
}

/**
 * @brief Computes the reciprocal (1/x) of each element in the SIMD vector.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to compute reciprocals for.
 * 
 * @return A SIMD vector containing the reciprocals.
 */
template <class T, std::size_t N>
constexpr simd<T, N> reciprocal(const simd<T, N>& value) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::reciprocal, value);
}

/**
 * @brief Computes the fused multiply-add (a * b + c) for each corresponding element in the SIMD vectors.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param a The first SIMD vector (multiplicand).
 * @param b The second SIMD vector (multiplier).
 * @param c The third SIMD vector (addend).
 * 
 * @return A SIMD vector containing the results of a * b + c.
 */
template <class T, std::size_t N>
constexpr simd<T, N> fma(const simd<T, N>& a, const simd<T, N>& b, const simd<T, N>& c) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::fma, a, b, c);
}

/**
 * @brief Rounds each element of the SIMD vector to the nearest integer.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to round.
 * 
 * @return A SIMD vector containing the rounded values.
 */
template <class T, std::size_t N>
constexpr simd<T, N> round(const simd<T, N>& value) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::round, value);
}

/**
 * @brief Computes the floor (largest integer less than or equal to) of each element in the SIMD vector.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to compute floors for.
 * 
 * @return A SIMD vector containing the floor values.
 */
template <class T, std::size_t N>
constexpr simd<T, N> floor(const simd<T, N>& value) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::floor, value);
}

/**
 * @brief Computes the ceiling (smallest integer greater than or equal to) of each element in the SIMD vector.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to compute ceilings for.
 * 
 * @return A SIMD vector containing the ceiling values.
 */
template <class T, std::size_t N>
constexpr simd<T, N> ceil(const simd<T, N>& value) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::ceil, value);
}

/**
 * @brief Truncates each element of the SIMD vector towards zero.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to truncate.
 * 
 * @return A SIMD vector containing the truncated values.
 */
template <class T, std::size_t N>
constexpr simd<T, N> trunc(const simd<T, N>& value) noexcept
{
    return apply<simd<T, N>>(ops_for<simd<T, N>>::trunc, value);
}

} // namespace arc
