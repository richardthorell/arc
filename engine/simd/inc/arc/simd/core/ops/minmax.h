#pragma once

#include <arc/simd/core/ops/detail.h>
#include <arc/simd/core/ops/scalar.h>

#include <type_traits>

namespace arc
{

/**
 * @brief Computes the element-wise minimum of two SIMD vectors.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param a The first SIMD vector.
 * @param b The second SIMD vector.
 * 
 * @return A SIMD vector containing the minimum of corresponding elements from a and b.
 */
template <class T, std::size_t N>
constexpr simd<T, N> min(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
        return detail::simd_map(a, b, [](T x, T y) { return x < y ? x : y; });
    else
        return apply<simd<T, N>>(ops_for<simd<T, N>>::min, a, b);
}

/**
 * @brief Computes the element-wise maximum of two SIMD vectors.
 * 
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param a The first SIMD vector.
 * @param b The second SIMD vector.
 * 
 * @return A SIMD vector containing the maximum of corresponding elements from a and b.
 */
template <class T, std::size_t N>
constexpr simd<T, N> max(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
        return detail::simd_map(a, b, [](T x, T y) { return x > y ? x : y; });
    else
        return apply<simd<T, N>>(ops_for<simd<T, N>>::max, a, b);
}

} // namespace arc
