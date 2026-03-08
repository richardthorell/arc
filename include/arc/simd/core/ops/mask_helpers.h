#pragma once

#include <arc/simd/core/mask.h>

namespace arc
{

template <std::size_t N>
constexpr simd_mask<N> prefix_mask(std::size_t count) noexcept
{
    return [count]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return simd_mask<N>{ (Index < count)... };
    }(std::make_index_sequence<N>{});
}

template <std::size_t N>
constexpr simd_mask<N> suffix_mask(std::size_t count) noexcept
{
    return ~prefix_mask<N>(N - count);
}

template <std::size_t N>
constexpr simd_mask<N> range_mask(std::size_t first, std::size_t last) noexcept
{
    return prefix_mask<N>(last) & ~prefix_mask<N>(first);
}

} // namespace arc
