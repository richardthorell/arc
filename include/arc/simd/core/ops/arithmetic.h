#pragma once

#include <arc/simd/core/ops/detail.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> add(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::add);
}

template <class T, std::size_t N>
constexpr simd<T, N> sub(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::sub);
}

template <class T, std::size_t N>
constexpr simd<T, N> mul(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::mul);
}

template <class T, std::size_t N>
constexpr simd<T, N> div(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, ops_for<simd<T, N>>::div);
}

template <class T, std::size_t N>
constexpr simd<T, N> neg(const simd<T, N>& a) noexcept
{
    return apply(a, ops_for<simd<T, N>>::neg);
}

} // namespace arc
