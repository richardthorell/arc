#pragma once

#include <arc/simd/core/ops/detail.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_and(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::bitwise_and);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_or(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::bitwise_or);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_xor(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::bitwise_xor);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_not(const simd<T, N>& a) noexcept
{
    return apply(a, ops_for<simd<T, N>>::bitwise_not);
}

template <std::size_t N>
constexpr simd_mask<N> bitwise_and(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return apply(a, b, ops_for<simd_mask<N>>::bitwise_and);
}

template <std::size_t N>
constexpr simd_mask<N> bitwise_or(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return apply(a, b, ops_for<simd_mask<N>>::bitwise_or);
}

template <std::size_t N>
constexpr simd_mask<N> bitwise_xor(const simd_mask<N>& a, const simd_mask<N>& b) noexcept
{
    return apply(a, b, ops_for<simd_mask<N>>::bitwise_xor);
}

template <std::size_t N>
constexpr simd_mask<N> bitwise_not(const simd_mask<N>& a) noexcept
{
    return apply(a, ops_for<simd_mask<N>>::bitwise_not);
}

} // namespace arc
