#pragma once

#include <immintrin.h>

namespace arc
{
template <class T, std::size_t N>
struct simd_register;

template <class T>
struct simd_op;

// Float specializations
template <>
struct simd_register<float, 4>
{
    using type = __m128;
};

template <>
struct simd_register<float, 8>
{
    using type = __m256;
};
#if 0
template <>
struct simd_register<float, 16>
{
    using type = __m512;
};


// Double specializations
template <>
struct simd_register<double, 2>
{
    using type = __m128d;
};

template <>
struct simd_register<double, 4>
{
    using type = __m256d;
};

template <>
struct simd_register<double, 8>
{
    using type = __m512d;
};

// Int8 specializations
template <>
struct simd_register<int8_t, 16>
{
    using type = __m128i;
};

template <>
struct simd_register<int8_t, 32>
{
    using type = __m256i;
};

template <>
struct simd_register<int8_t, 64>
{
    using type = __m512i;
};

// Int16 specializations
template <>
struct simd_register<int16_t, 8>
{
    using type = __m128i;
};

template <>
struct simd_register<int16_t, 16>
{
    using type = __m256i;
};

template <>
struct simd_register<int16_t, 32>
{
    using type = __m512i;
};

// Int32 specializations
template <>
struct simd_register<int32_t, 4>
{
    using type = __m128i;
};

template <>
struct simd_register<int32_t, 8>
{
    using type = __m256i;
};

template <>
struct simd_register<int32_t, 16>
{
    using type = __m512i;
};
#endif

template <class T, std::size_t N>
using simd_register_t = typename simd_register<T, N>::type;

// Float operations
template <>
struct simd_op<__m128>
{
    // Load/Store
    static inline __m128 load(const float* ptr) noexcept
    {
        return _mm_loadu_ps(ptr);
    }

    static inline void store(float* ptr, __m128 value) noexcept
    {
        _mm_storeu_ps(ptr, value);
    }

    static inline __m128 fill(float value) noexcept
    {
        return _mm_set1_ps(value);
    }

    // Arithmetic
    static inline __m128 add(__m128 a, __m128 b) noexcept
    {
        return _mm_add_ps(a, b);
    }

    static inline __m128 sub(__m128 a, __m128 b) noexcept
    {
        return _mm_sub_ps(a, b);
    }

    static inline __m128 mul(__m128 a, __m128 b) noexcept
    {
        return _mm_mul_ps(a, b);
    }

    static inline __m128 div(__m128 a, __m128 b) noexcept
    {
        return _mm_div_ps(a, b);
    }

    static inline __m128 neg(__m128 a) noexcept
    {
        return _mm_sub_ps(_mm_setzero_ps(), a);
    }

    // Min/Max
    static inline __m128 min(__m128 a, __m128 b) noexcept 
    { 
        return _mm_min_ps(a, b); 
    }

    static inline __m128 max(__m128 a, __m128 b) noexcept 
    { 
        return _mm_max_ps(a, b); 
    }

    // Bitwise/logical
    static inline __m128 bitwise_and(__m128 a, __m128 b) noexcept 
    { 
        return _mm_and_ps(a, b); 
    }

    static inline __m128 bitwise_or(__m128 a, __m128 b) noexcept 
    { 
        return _mm_or_ps(a, b); 
    }

    static inline __m128 bitwise_xor(__m128 a, __m128 b) noexcept 
    { 
        return _mm_xor_ps(a, b); 
    }

    static inline __m128 bitwise_not(__m128 a) noexcept 
    { 
        return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1))); 
    }
};


template <>
struct simd_op<__m256>
{
    // Load/Store
    static inline __m256 load(const float* ptr) noexcept
    {
        return _mm256_loadu_ps(ptr);
    }

    static inline void store(float* ptr, __m256 value) noexcept
    {
        _mm256_storeu_ps(ptr, value);
    }

    static inline __m256 fill(float value) noexcept
    {
        return _mm256_set1_ps(value);
    }

    // Arithmetic
    static inline __m256 add(__m256 a, __m256 b) noexcept
    {
        return _mm256_add_ps(a, b);
    }

    static inline __m256 sub(__m256 a, __m256 b) noexcept
    {
        return _mm256_sub_ps(a, b);
    }

    static inline __m256 mul(__m256 a, __m256 b) noexcept
    {
        return _mm256_mul_ps(a, b);
    }

    static inline __m256 div(__m256 a, __m256 b) noexcept
    {
        return _mm256_div_ps(a, b);
    }

    static inline __m256 neg(__m256 a) noexcept
    {
        return _mm256_sub_ps(_mm256_setzero_ps(), a);
    }

    // Min/Max

    // Bitwise/logical
};
}