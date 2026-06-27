#pragma once

#include <arc/simd/core/ops/detail.h>

#include <array>

namespace arc
{

namespace detail
{

template <std::size_t N, class Predicate>
constexpr auto make_mask_block(std::size_t block_index, Predicate pred) noexcept
{
    using mask_block = simd_block<uint32_t, N>;
    using ops = ops_for<simd_mask<N>>;

    std::array<uint32_t, mask_block::lanes> lanes{};
    for (std::size_t i = 0; i < mask_block::lanes; ++i)
    {
        const std::size_t lane_index = block_index * mask_block::lanes + i;
        lanes[i] = lane_index < N && pred(lane_index) ? 0xFFFFFFFFu : 0u;
    }

    return ops::load_unaligned(lanes.data());
}

} // namespace detail

template <std::size_t N>
constexpr simd_mask<N> prefix_mask(std::size_t count) noexcept
{
    return [count]<std::size_t... Block>(std::index_sequence<Block...>)
    {
        return detail::simd_access::template make_mask<N>(
            detail::make_mask_block<N>(Block, [count](std::size_t lane) { return lane < count; })...
        );
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

template <std::size_t N>
constexpr simd_mask<N> suffix_mask(std::size_t count) noexcept
{
    return [count]<std::size_t... Block>(std::index_sequence<Block...>)
    {
        return detail::simd_access::template make_mask<N>(
            detail::make_mask_block<N>(Block, [count](std::size_t lane) { return lane >= N - count; })...
        );
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

template <std::size_t N>
constexpr simd_mask<N> range_mask(std::size_t first, std::size_t last) noexcept
{
    return [first, last]<std::size_t... Block>(std::index_sequence<Block...>)
    {
        return detail::simd_access::template make_mask<N>(
            detail::make_mask_block<N>(Block, [first, last](std::size_t lane) { return lane >= first && lane < last; })...
        );
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

} // namespace arc
