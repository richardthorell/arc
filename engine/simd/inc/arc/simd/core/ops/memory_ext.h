#pragma once

#include <arc/simd/core/ops/mask_bits.h>
#include <arc/simd/core/ops/scalar.h>

#include <array>
#include <cstdint>

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    #include <xmmintrin.h>
#endif

namespace arc
{

enum class prefetch_hint
{
    temporal0,
    temporal1,
    temporal2,
    non_temporal
};

inline void prefetch(const void* ptr, prefetch_hint hint = prefetch_hint::temporal0) noexcept
{
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    switch (hint)
    {
    case prefetch_hint::temporal0:
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
        break;
    case prefetch_hint::temporal1:
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T1);
        break;
    case prefetch_hint::temporal2:
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T2);
        break;
    case prefetch_hint::non_temporal:
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_NTA);
        break;
    }
#else
    (void)ptr;
    (void)hint;
#endif
}

template <class T, std::size_t N>
constexpr void stream_store(T* ptr, const simd<T, N>& value) noexcept
{
    store_aligned(ptr, value);
}

template <class T, std::size_t N>
constexpr simd<T, N> gather(const T* base, const simd<int32_t, N>& indices) noexcept
{
    auto index_values = detail::simd_to_array(indices);
    std::array<T, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = base[index_values[i]];
    return detail::simd_from_array<T, N>(result);
}

template <class T, std::size_t N>
constexpr simd<T, N> gather(const T* base, const simd<int32_t, N>& indices, const simd_mask<N>& mask, T default_value = T{}) noexcept
{
    auto index_values = detail::simd_to_array(indices);
    std::array<T, N> result{};
    const auto bits = mask_to_bits(mask);
    for (std::size_t i = 0; i < N; ++i)
        result[i] = (bits & (std::uint64_t{ 1 } << i)) != 0 ? base[index_values[i]] : default_value;
    return detail::simd_from_array<T, N>(result);
}

template <class T, std::size_t N>
constexpr void scatter(T* base, const simd<int32_t, N>& indices, const simd<T, N>& value) noexcept
{
    auto index_values = detail::simd_to_array(indices);
    auto values = detail::simd_to_array(value);
    for (std::size_t i = 0; i < N; ++i)
        base[index_values[i]] = values[i];
}

template <class T, std::size_t N>
constexpr void scatter(T* base, const simd<int32_t, N>& indices, const simd<T, N>& value, const simd_mask<N>& mask) noexcept
{
    auto index_values = detail::simd_to_array(indices);
    auto values = detail::simd_to_array(value);
    const auto bits = mask_to_bits(mask);
    for (std::size_t i = 0; i < N; ++i)
        if ((bits & (std::uint64_t{ 1 } << i)) != 0)
            base[index_values[i]] = values[i];
}

template <class T>
constexpr simd<T, 4> load3(const T* ptr, T w = T{}) noexcept
{
    return detail::simd_from_array<T, 4>({ ptr[0], ptr[1], ptr[2], w });
}

template <class T>
constexpr void store3(T* ptr, const simd<T, 4>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    ptr[0] = lanes[0];
    ptr[1] = lanes[1];
    ptr[2] = lanes[2];
}

template <class T>
constexpr simd<T, 4> load4(const T* ptr) noexcept
{
    return detail::simd_from_array<T, 4>({ ptr[0], ptr[1], ptr[2], ptr[3] });
}

template <class T>
constexpr void store4(T* ptr, const simd<T, 4>& value) noexcept
{
    auto lanes = detail::simd_to_array(value);
    ptr[0] = lanes[0];
    ptr[1] = lanes[1];
    ptr[2] = lanes[2];
    ptr[3] = lanes[3];
}

} // namespace arc
