#pragma once

#include <arc/simd/core/ops/permute.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> zip_lo(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return permute<Index * 2, (Index * 2 + 1)...>(a, b);
    }(std::make_index_sequence<N / 2>{});
}

template <class T, std::size_t N>
constexpr simd<T, N> zip_hi(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return permute<(N / 2 + Index) * 2, ((N / 2 + Index) * 2 + 1)...>(a, b);
    }(std::make_index_sequence<N / 2>{});
}

} // namespace arc