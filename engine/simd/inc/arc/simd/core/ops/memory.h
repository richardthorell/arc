#pragma once

#include <arc/simd/core/ops/detail.h>
#include <arc/simd/core/ops/mask_helpers.h>

namespace arc
{

/**
 * @brief Load `N` values from an aligned pointer into a SIMD vector.
 *
 * The pointer must satisfy the active backend's alignment requirements.
 */
template <class T, std::size_t N>
constexpr simd<T, N> load_aligned(const T* ptr) noexcept
{
    return apply<simd<T, N>>(
        [&](std::size_t block_index)
        {
            return ops_for<simd<T, N>>::load_aligned(ptr + block_index * simd_block<T, N>::lanes);
        }
    );
}

/**
 * @brief Store a SIMD vector to an aligned pointer.
 *
 * The pointer must satisfy the active backend's alignment requirements.
 */
template <class T, std::size_t N>
constexpr void store_aligned(T* ptr, const simd<T, N>& value) noexcept
{
    using detail::simd_access;

    apply<simd<T, N>>(
        [&](std::size_t block_index)
        {
            ops_for<simd<T, N>>::store_aligned(ptr + block_index * simd_block<T, N>::lanes, simd_access::block(value, block_index));
        }
    );
}

/// @brief Load `N` values from a pointer with no alignment requirement.
template <class T, std::size_t N>
constexpr simd<T, N> load_unaligned(const T* ptr) noexcept
{
    return apply<simd<T, N>>(
        [&](std::size_t block_index)
        {
            return ops_for<simd<T, N>>::load_unaligned(ptr + block_index * simd_block<T, N>::lanes);
        }
    );
}

/// @brief Store a SIMD vector to a pointer with no alignment requirement.
template <class T, std::size_t N>
constexpr void store_unaligned(T* ptr, const simd<T, N>& value) noexcept
{
    using detail::simd_access;

    apply<simd<T, N>>(
        [&](std::size_t block_index)
        {
            ops_for<simd<T, N>>::store_unaligned(ptr + block_index * simd_block<T, N>::lanes, simd_access::block(value, block_index));
        }
    );
}

/// @brief Load active lanes selected by a mask and use `default_value` elsewhere.
template <class T, std::size_t N>
constexpr simd<T, N> masked_load(const T* ptr, const simd_mask<N>& mask, const simd<T, N>& default_value = simd<T, N>{}) noexcept
{
    using detail::simd_access;

    return apply<simd<T, N>>(
        [&](std::size_t block_index)
        {
            return ops_for<simd<T, N>>::masked_load(
                ptr + block_index * simd_block<T, N>::lanes,
                simd_access::block(mask, block_index),
                simd_access::block(default_value, block_index)
            );
        }
    );
}

/// @brief Store only lanes selected by a mask.
template <class T, std::size_t N>
constexpr void masked_store(T* ptr, const simd<T, N>& value, const simd_mask<N>& mask) noexcept
{
    using detail::simd_access;

    apply<simd<T, N>>(
        [&](std::size_t block_index)
        {
            ops_for<simd<T, N>>::masked_store(
                ptr + block_index * simd_block<T, N>::lanes,
                simd_access::block(value, block_index),
                simd_access::block(mask, block_index)
            );
        }
    );
}

/// @brief Load up to `count` lanes and zero-fill inactive lanes.
template <class T, std::size_t N>
constexpr simd<T, N> load_partial(const T* ptr, std::size_t count) noexcept
{
    return masked_load(ptr, prefix_mask<N>(count));
}

/// @brief Store up to `count` lanes from a SIMD vector.
template <class T, std::size_t N>
constexpr void store_partial(T* ptr, const simd<T, N>& v, std::size_t count) noexcept
{
    masked_store(ptr, v, prefix_mask<N>(count));
}

/// @brief Return a SIMD vector with every lane set to `value`.
template <class T, std::size_t N>
constexpr simd<T, N> fill(T value) noexcept
{
    return apply<simd<T, N>>(
        [&](std::size_t)
        {
            return ops_for<simd<T, N>>::fill(value);
        }
    );
}

} // namespace arc
