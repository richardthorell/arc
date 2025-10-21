#pragma once

#include <arc/simd/concepts.h>

// TODO: Include instructions headers based on detected architecture
#include <arc/simd/arch/x64.h>

#include <array>

namespace arc
{

template <class T, std::size_t N>
concept simd_register_available = requires
{
    typename simd_register<T, N>::type;
};

template <class T, std::size_t N>
consteval std::size_t optimal_lanes()
{
    if constexpr (N == 1)
    {
        return 1;
    }
    else if constexpr (simd_register_available<T, N>)
    {
        return N;
    }
    else
    {
        return optimal_lanes<T, N / 2>();
    }
}

template <class T, std::size_t N>
struct simd_storage
{
    static constexpr std::size_t lanes  = optimal_lanes<T, N>();
    static constexpr std::size_t blocks = (N + lanes - 1) / lanes;

    using value_type = T;
    using register_type = typename simd_register_t<value_type, lanes>;
    using data_type = std::array<register_type, blocks>;
};

}