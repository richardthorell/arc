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

// UInt32 specializations
template <>
struct simd_register<uint32_t, 4>
{
    using type = __m128i;
};

template <>
struct simd_register<uint32_t, 8>
{
    using type = __m256i;
};

template <>
struct simd_register<uint32_t, 16>
{
    using type = __m512i;
};


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

    // Comparisons
    static inline __m128i cmp_eq(__m128 a, __m128 b) noexcept
    {
        return _mm_castps_si128(_mm_cmpeq_ps(a, b));
    }

    static inline __m128i cmp_ne(__m128 a, __m128 b) noexcept
    {
        return _mm_castps_si128(_mm_cmpneq_ps(a, b));
    }

    static inline __m128i cmp_lt(__m128 a, __m128 b) noexcept
    {
        return _mm_castps_si128(_mm_cmplt_ps(a, b));
    }

    static inline __m128i cmp_le(__m128 a, __m128 b) noexcept
    {
        return _mm_castps_si128(_mm_cmple_ps(a, b));
    }

    static inline __m128i cmp_gt(__m128 a, __m128 b) noexcept
    {
        return _mm_castps_si128(_mm_cmpgt_ps(a, b));
    }

    static inline __m128i cmp_ge(__m128 a, __m128 b) noexcept
    {
        return _mm_castps_si128(_mm_cmpge_ps(a, b));
    }

    // Blending
    static inline __m128 blend(__m128 a, __m128 b, __m128i mask) noexcept
    {
        return _mm_blendv_ps(a, b, _mm_castsi128_ps(mask));
    }

    // Operations
    static inline __m128 sqrt(__m128 a) noexcept
    {
        return _mm_sqrt_ps(a);
    }

    static inline __m128 round(__m128 a) noexcept
    {
        return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    static inline __m128 floor(__m128 a) noexcept
    {
        return _mm_floor_ps(a);
    }

    static inline __m128 ceil(__m128 a) noexcept
    {
        return _mm_ceil_ps(a);
    }

    static inline __m128 trunc(__m128 a) noexcept
    {
        return _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
};

template <>
struct simd_op<__m128i>
{
    // Load/Store
    static inline __m128i load(const int32_t* ptr) noexcept
    {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    }

    static inline void store(int32_t* ptr, __m128i value) noexcept
    {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), value);
    }

    static inline __m128i fill(int32_t value) noexcept
    {
        return _mm_set1_epi32(value);
    }

    // Arithmetic
    static inline __m128i add(__m128i a, __m128i b) noexcept
    {
        return _mm_add_epi32(a, b);
    }

    static inline __m128i sub(__m128i a, __m128i b) noexcept
    {
        return _mm_sub_epi32(a, b);
    }

    static inline __m128i neg(__m128i a) noexcept
    {
        return _mm_sub_epi32(_mm_setzero_si128(), a);
    }

    // Bitwise/logical
    static inline __m128i bitwise_and(__m128i a, __m128i b) noexcept
    { 
        return _mm_and_si128(a, b); 
    }

    static inline __m128i bitwise_or(__m128i a, __m128i b) noexcept
    { 
        return _mm_or_si128(a, b); 
    }

    static inline __m128i bitwise_xor(__m128i a, __m128i b) noexcept
    { 
        return _mm_xor_si128(a, b); 
    }

    static inline __m128i bitwise_not(__m128i a) noexcept
    { 
        return _mm_xor_si128(a, _mm_set1_epi32(-1)); 
    }

    // Reductions
    static inline bool any(__m128i a) noexcept
    {
        return !_mm_testz_si128(a, a);
    }

    static inline bool all(__m128i a) noexcept
    {
        return _mm_testc_si128(a, _mm_set1_epi32(-1));
    }
};
}