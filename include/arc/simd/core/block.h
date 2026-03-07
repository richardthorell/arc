#pragma once

#include <arc/simd/arch/arch.h>

#include <array>

namespace arc
{
namespace detail
{

template <class T, std::size_t N>
concept simd_register_available = requires
{
    typename simd_register<T, N>::type;
};

// Helper to find the largest power-of-two ≤ N
consteval std::size_t largest_power_of_two(std::size_t n, std::size_t candidate = 1)
{
    if (candidate * 2 > n)
        return candidate;

    return largest_power_of_two(n, candidate * 2);
}

// Recursive optimal lanes selection
template <typename T, std::size_t Candidate, std::size_t MaxLanes>
consteval std::size_t optimal_lanes_impl()
{
    if constexpr (Candidate == 0)
    {
        return 1;
    }
    else if constexpr (Candidate > MaxLanes)
    {
        return optimal_lanes_impl<T, Candidate / 2, MaxLanes>();
    }
    else if constexpr (simd_register_available<T, Candidate>)
    {
        return Candidate;
    }
    else
    {
        return optimal_lanes_impl<T, Candidate / 2, MaxLanes>();
    }
}

// Entry point
template <typename T, std::size_t N>
consteval std::size_t optimal_lanes()
{
    constexpr std::size_t max_lanes = simd_max_lanes<T>;
    constexpr std::size_t candidate = largest_power_of_two(N);

    return optimal_lanes_impl<T, candidate, max_lanes>();
}

} // namespace detail


/**
 * @brief Storage block for SIMD types.
 * 
 * @tparam T The base data type (e.g., float, int32_t).
 * @tparam N The total number of lanes in the SIMD type.
 */
template <class T, std::size_t N>
struct simd_block
{
    /**
     * @brief Number of lanes per SIMD register block.
     */
    static constexpr std::size_t lanes = detail::optimal_lanes<T, N>();

    /**
     * @brief Number of SIMD register blocks needed to store N lanes.
     */
    static constexpr std::size_t blocks = (N + lanes - 1) / lanes;

    /**
     * @brief The base data type.
     */
    using value_type = T;

    /**
     * @brief The SIMD register type for the given data type and lanes.
     */
    using register_type = simd_register_t<value_type, lanes>;

    /**
     * @brief The data storage type, an array of SIMD register blocks.
     */
    using data_type = std::array<register_type, blocks>;
};

} // namespace arc