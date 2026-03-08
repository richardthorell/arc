#pragma once

#include <arc/simd/core/ops/detail.h>
#include <arc/simd/core/ops/mask_helpers.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> load_aligned(const T* ptr) noexcept
{
    return apply<T, N>(
        [&](std::size_t block_index)
        {
            return ops_for<simd<T, N>>::load_aligned(ptr + block_index * simd_block<T, N>::lanes);
        }
    );
}

template <class T, std::size_t N>
constexpr void store_aligned(T* ptr, const simd<T, N>& value) noexcept
{
    apply<T, N>(
        [&](std::size_t block_index)
        {
            ops_for<simd<T, N>>::store_aligned(ptr + block_index * simd_block<T, N>::lanes, value.data[block_index]);
        }
    );
}

template <class T, std::size_t N>
constexpr simd<T, N> load_unaligned(const T* ptr) noexcept
{
    return apply<T, N>(
        [&](std::size_t block_index)
        {
            return ops_for<simd<T, N>>::load_unaligned(ptr + block_index * simd_block<T, N>::lanes);
        }
    );
}

template <class T, std::size_t N>
constexpr void store_unaligned(T* ptr, const simd<T, N>& value) noexcept
{
    apply<T, N>(
        [&](std::size_t block_index)
        {
            ops_for<simd<T, N>>::store_unaligned(ptr + block_index * simd_block<T, N>::lanes, value.data[block_index]);
        }
    );
}

template <class T, std::size_t N>
constexpr simd<T, N> masked_load(const T* ptr, const simd_mask<N>& mask, const simd<T, N>& default_value = simd<T, N>{}) noexcept
{
    return apply<T, N>(
        [&](std::size_t block_index)
        {
            return ops_for<simd<T, N>>::masked_load(
                ptr + block_index * simd_block<T, N>::lanes,
                mask.data[block_index],
                default_value.data[block_index]
            );
        }
    );
}

template <class T, std::size_t N>
constexpr void masked_store(T* ptr, const simd<T, N>& value, const simd_mask<N>& mask) noexcept
{
    apply<T, N>(
        [&](std::size_t block_index)
        {
            ops_for<simd<T, N>>::masked_store(
                ptr + block_index * simd_block<T, N>::lanes,
                value.data[block_index],
                mask.data[block_index]
            );
        }
    );
}

template <class T, std::size_t N>
constexpr simd<T, N> load_partial(const T* ptr, std::size_t count) noexcept
{
    return masked_load(ptr, prefix_mask<N>(count));
}

template <class T, std::size_t N>
constexpr void store_partial(T* ptr, const simd<T, N>& v, std::size_t count) noexcept
{
    masked_store(ptr, v, prefix_mask<N>(count));
}

template <class T, std::size_t N>
constexpr simd<T, N> fill(T value) noexcept
{
    return apply<T, N>(
        [&](std::size_t)
        {
            return ops_for<simd<T, N>>::fill(value);
        }
    );
}

} // namespace arc
