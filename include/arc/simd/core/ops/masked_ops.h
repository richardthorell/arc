#pragma once

#include <arc/simd/core/ops/arithmetic.h>
#include <arc/simd/core/ops/bitwise.h>
#include <arc/simd/core/ops/comparisons.h>
#include <arc/simd/core/ops/math.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> masked_fill(const simd_mask<N>& mask, const simd<T, N>& a, T value) noexcept
{
    return masked(mask, a, [&](auto) { return fill<T, N>(value); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_fill(const simd_mask<N>& mask, T value) noexcept
{
    return masked_fill(mask, fill<T, N>(T(0)), value);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_add(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, add<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_sub(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, sub<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_mul(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, mul<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_div(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, div<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_min(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, [](auto x, auto y) { return select(x < y, x, y); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_max(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, [](auto x, auto y) { return select(x > y, x, y); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_clamp(const simd_mask<N>& mask, const simd<T, N>& value, const simd<T, N>& min_value, const simd<T, N>& max_value) noexcept
{
    return masked(mask, min_value, max_value, [&](auto lo, auto hi) { return clamp(value, lo, hi); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_saturate(const simd_mask<N>& mask, const simd<T, N>& value) noexcept
{
    return masked(mask, fill<T, N>(0), fill<T, N>(1), [&](auto zero, auto one) { return clamp(value, zero, one); });
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_neg(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, neg<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_abs(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, abs<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_sqrt(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, sqrt<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_round(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, round<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_floor(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, floor<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_ceil(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, ceil<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_trunc(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, trunc<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_bitwise_and(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, bitwise_and<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_bitwise_or(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, bitwise_or<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_bitwise_xor(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, bitwise_xor<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_bitwise_not(const simd_mask<N>& mask, const simd<T, N>& a) noexcept
{
    return masked(mask, a, a, bitwise_not<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_eq(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_eq<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_ne(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_ne<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_lt(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_lt<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_le(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_le<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_gt(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_gt<T, N>);
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_compare_ge(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return masked(mask, a, b, cmp_ge<T, N>);
}

} // namespace arc
