#pragma once

#include <arc/simd/core/ops/detail.h>

namespace arc
{

/**
 * @brief Checks if any element in the SIMD mask is true.
 * 
 * @tparam N The number of elements in the SIMD mask.
 * 
 * @param mask The SIMD mask to check.
 * 
 * @return true if at least one element is true, false otherwise.
 */
template <std::size_t N>
constexpr bool any(const simd_mask<N>& mask) noexcept
{
    using detail::simd_access;

    return[&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return (... || ops_for<simd_mask<N>>::any(simd_access::block(mask, Index)));
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

/**
 * @brief Checks if all elements in the SIMD mask are true.
 * 
 * @tparam N The number of elements in the SIMD mask.
 * 
 * @param mask The SIMD mask to check.
 * 
 * @return true if all elements are true, false otherwise.
 */
template <std::size_t N>
constexpr bool all(const simd_mask<N>& mask) noexcept
{
    using detail::simd_access;

    return[&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return (... && ops_for<simd_mask<N>>::all(simd_access::block(mask, Index)));
    }(std::make_index_sequence<simd_mask<N>::blocks()>{});
}

/**
 * @brief Checks if no elements in the SIMD mask are true.
 * 
 * @tparam N The number of elements in the SIMD mask.
 * 
 * @param mask The SIMD mask to check.
 * 
 * @return true if no elements are true, false otherwise.
 */
template <std::size_t N>
constexpr bool none(const simd_mask<N>& mask) noexcept
{
    return !any(mask);
}

} // namespace arc
