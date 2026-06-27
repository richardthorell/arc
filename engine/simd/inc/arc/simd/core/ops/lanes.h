#pragma once

#include <arc/simd/core/ops/memory.h>

namespace arc
{

/**
 * @brief Extracts a single lane value from a SIMD vector at compile-time index I.
 * 
 * @tparam I The lane index to extract (0-based, must be less than N).
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to extract from.
 * 
 * @return The value at lane I.
 */
template <std::size_t I, class T, std::size_t N>
constexpr T extract(const simd<T, N>& value) noexcept
{
    using detail::simd_access;

    static_assert(I < N, "Index out of bounds");
    
    constexpr std::size_t block_index = I / simd_block<T, N>::lanes;
    constexpr std::size_t lane_index = I % simd_block<T, N>::lanes;

    return ops_for<simd<T, N>>::template extract<lane_index>(simd_access::block(value, block_index));
}

/**
 * @brief Inserts a value into a specific lane of a SIMD vector at compile-time index I.
 * 
 * @tparam I The lane index to insert into (0-based, must be less than N).
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to modify.
 * @param lane_value The value to insert into lane I.
 * 
 * @return A new SIMD vector with the value inserted at lane I, other lanes unchanged.
 */
template <std::size_t I, class T, std::size_t N>
constexpr simd<T, N> insert(const simd<T, N>& value, T lane_value) noexcept
{
    using detail::simd_access;

    static_assert(I < N, "Index out of bounds");
    
    constexpr std::size_t block_index = I / simd_block<T, N>::lanes;
    constexpr std::size_t lane_index = I % simd_block<T, N>::lanes;

    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return simd_access::template make_simd<T, N>(
            (Index == block_index ? ops_for<simd<T, N>>::template insert<lane_index>(simd_access::block(value, Index), lane_value) : simd_access::block(value, Index))...
        );
    }(std::make_index_sequence<simd<T, N>::blocks()>{});
}

/**
 * @brief Broadcasts the value from a specific lane to all lanes of a new SIMD vector.
 * 
 * @tparam I The lane index to extract and broadcast (0-based, must be less than N).
 * @tparam T The element type of the SIMD vector.
 * @tparam N The number of elements in the SIMD vector.
 * 
 * @param value The SIMD vector to extract from.
 * 
 * @return A new SIMD vector where all lanes contain the value from lane I of the input.
 */
template <std::size_t I, class T, std::size_t N>
constexpr simd<T, N> broadcast(const simd<T, N>& value) noexcept
{
    return fill<T, N>(extract<I>(value));
}

} // namespace arc