#pragma once

#include <arc/simd/core/ops/detail.h>
#include <arc/simd/core/ops/bitwise.h>
#include <arc/simd/core/ops/mask_helpers.h>
#include <arc/simd/core/ops/scalar.h>

#include <type_traits>

namespace arc
{

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_eq(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
    {
        auto av = detail::simd_to_array(a);
        auto bv = detail::simd_to_array(b);
        auto mask = simd_mask<N>(false);
        for (std::size_t i = 0; i < N; ++i)
            if (av[i] == bv[i])
                mask = bitwise_or(mask, range_mask<N>(i, i + 1));
        return mask;
    }
    else
    {
        return compare(a, b, ops_for<simd<T, N>>::cmp_eq);
    }
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_ne(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
        return bitwise_not(cmp_eq(a, b));
    else
        return compare(a, b, ops_for<simd<T, N>>::cmp_ne);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_lt(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
    {
        auto av = detail::simd_to_array(a);
        auto bv = detail::simd_to_array(b);
        auto mask = simd_mask<N>(false);
        for (std::size_t i = 0; i < N; ++i)
            if (av[i] < bv[i])
                mask = bitwise_or(mask, range_mask<N>(i, i + 1));
        return mask;
    }
    else
    {
        return compare(a, b, ops_for<simd<T, N>>::cmp_lt);
    }
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_le(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
        return bitwise_or(cmp_lt(a, b), cmp_eq(a, b));
    else
        return compare(a, b, ops_for<simd<T, N>>::cmp_le);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_gt(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
        return cmp_lt(b, a);
    else
        return compare(a, b, ops_for<simd<T, N>>::cmp_gt);
}

template <class T, std::size_t N>
constexpr simd_mask<N> cmp_ge(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
        return bitwise_or(cmp_gt(a, b), cmp_eq(a, b));
    else
        return compare(a, b, ops_for<simd<T, N>>::cmp_ge);
}

} // namespace arc
