#pragma once

#include <arc/simd/core/ops/arithmetic.h>
#include <arc/simd/core/ops/bitwise.h>
#include <arc/simd/core/ops/comparisons.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> operator+(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return add(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator-(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return sub(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator*(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return mul(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator/(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return div(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator-(const simd<T, N>& a) noexcept
{
    return neg(a);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator&(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_and(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator|(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_or(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator^(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_xor(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator~(const simd<T, N>& a) noexcept
{
    return bitwise_not(a);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator==(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_eq(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator!=(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_ne(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator<(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_lt(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator<=(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_le(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator>(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_gt(a, b);
}

template <class T, std::size_t N>
constexpr simd_mask<N> operator>=(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return cmp_ge(a, b);
}

template <std::size_t N>
constexpr simd_mask<N> operator&(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return bitwise_and(a, b);
}

template <std::size_t N>
constexpr simd_mask<N> operator|(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return bitwise_or(a, b);
}

template <std::size_t N>
constexpr simd_mask<N> operator^(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return bitwise_xor(a, b);
}

template <std::size_t N>
constexpr simd_mask<N> operator~(const simd_mask<N>& a) noexcept
{
    return bitwise_not(a);
}

template <std::size_t N>
constexpr simd_mask<N>& operator&=(simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return (a = a & b);
}

template <std::size_t N>
simd_mask<N>& operator|=(simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    a = a | b;
    return a;
}

template <std::size_t N>
simd_mask<N>& operator^=(simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    a = a ^ b;
    return a;
}

} // namespace arc
