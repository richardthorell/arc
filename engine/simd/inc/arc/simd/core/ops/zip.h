#pragma once

#include <arc/simd/core/ops/permute.h>

#include <utility>

namespace arc
{
namespace detail
{

template <class T, std::size_t N, std::size_t... Index>
constexpr simd<T, N> zip_lo_impl(const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>) noexcept
{
    return shuffle<(Index % 2 == 0 ? Index / 2 : N + Index / 2)...>(a, b);
}

template <class T, std::size_t N, std::size_t... Index>
constexpr simd<T, N> zip_hi_impl(const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>) noexcept
{
    return shuffle<(Index % 2 == 0 ? N / 2 + Index / 2 : N + N / 2 + Index / 2)...>(a, b);
}

} // namespace detail

template <class T, std::size_t N>
constexpr simd<T, N> zip_lo(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    static_assert(N % 2 == 0, "zip_lo requires an even lane count");
    return detail::zip_lo_impl(a, b, std::make_index_sequence<N>{});
}

template <class T, std::size_t N>
constexpr simd<T, N> zip_hi(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    static_assert(N % 2 == 0, "zip_hi requires an even lane count");
    return detail::zip_hi_impl(a, b, std::make_index_sequence<N>{});
}

} // namespace arc
