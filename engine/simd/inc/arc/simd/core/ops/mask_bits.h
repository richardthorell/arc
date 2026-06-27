#pragma once

#include <arc/simd/core/ops/bitwise.h>
#include <arc/simd/core/ops/mask_helpers.h>
#include <arc/simd/core/ops/mask_reductions.h>
#include <arc/simd/core/ops/scalar.h>

#include <bit>
#include <cstdint>
#include <limits>

namespace arc
{

template <std::size_t N>
constexpr std::uint64_t mask_to_bits(const simd_mask<N>& mask) noexcept
{
    static_assert(N <= 64, "mask_to_bits supports up to 64 lanes");

    std::uint64_t bits = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        if (any(bitwise_and(mask, range_mask<N>(i, i + 1))))
            bits |= std::uint64_t{ 1 } << i;
    }
    return bits;
}

template <std::size_t N>
constexpr simd_mask<N> bits_to_mask(std::uint64_t bits) noexcept
{
    static_assert(N <= 64, "bits_to_mask supports up to 64 lanes");

    auto mask = simd_mask<N>(false);
    for (std::size_t i = 0; i < N; ++i)
    {
        if ((bits & (std::uint64_t{ 1 } << i)) != 0)
            mask = bitwise_or(mask, range_mask<N>(i, i + 1));
    }
    return mask;
}

template <std::size_t N>
constexpr int popcount(const simd_mask<N>& mask) noexcept
{
    return std::popcount(mask_to_bits(mask));
}

template <std::size_t N>
constexpr int first_active_lane(const simd_mask<N>& mask) noexcept
{
    const auto bits = mask_to_bits(mask);
    if (bits == 0)
        return -1;
    return std::countr_zero(bits);
}

template <class T, std::size_t N>
constexpr simd<T, N> compress(const simd<T, N>& value, const simd_mask<N>& mask, T fill_value = T{}) noexcept
{
    auto lanes = detail::simd_to_array(value);
    std::array<T, N> result{};
    result.fill(fill_value);

    std::size_t out = 0;
    const auto bits = mask_to_bits(mask);
    for (std::size_t i = 0; i < N; ++i)
    {
        if ((bits & (std::uint64_t{ 1 } << i)) != 0)
            result[out++] = lanes[i];
    }

    return detail::simd_from_array<T, N>(result);
}

template <class T, std::size_t N>
constexpr simd<T, N> expand(const simd<T, N>& compacted, const simd_mask<N>& mask, T fill_value = T{}) noexcept
{
    auto lanes = detail::simd_to_array(compacted);
    std::array<T, N> result{};
    result.fill(fill_value);

    std::size_t in = 0;
    const auto bits = mask_to_bits(mask);
    for (std::size_t i = 0; i < N; ++i)
    {
        if ((bits & (std::uint64_t{ 1 } << i)) != 0)
            result[i] = lanes[in++];
    }

    return detail::simd_from_array<T, N>(result);
}

} // namespace arc
