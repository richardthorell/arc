#pragma once

#include <arc/simd/core/ops/detail.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> min(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::min);
}

template <class T, std::size_t N>
constexpr simd<T, N> max(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::max);
}

} // namespace arc
