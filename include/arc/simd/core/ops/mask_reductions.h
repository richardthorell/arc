#pragma once

#include <arc/simd/core/ops/detail.h>

namespace arc
{

template <std::size_t N>
constexpr bool any(const simd_mask<N>& mask) noexcept
{
    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return (... || ops_for<simd_mask<N>>::any(mask.data[Index]));
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

template <std::size_t N>
constexpr bool all(const simd_mask<N>& mask) noexcept
{
    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return (... && ops_for<simd_mask<N>>::all(mask.data[Index]));
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

template <std::size_t N>
constexpr bool none(const simd_mask<N>& mask) noexcept
{
    return !any(mask);
}

} // namespace arc
