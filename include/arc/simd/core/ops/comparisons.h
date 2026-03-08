#pragma once

#include <arc/simd/core/ops/detail.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_eq(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_eq);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_ne(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_ne);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_lt(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_lt);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_le(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_le);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_gt(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_gt);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_ge(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return compare(a, b, ops_for<simd<T, N>>::cmp_ge);
}

} // namespace arc
