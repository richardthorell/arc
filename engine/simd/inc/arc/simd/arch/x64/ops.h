#pragma once

#include "arc/simd/arch/x64/detect.h"

namespace arc
{

#if defined(ARC_SIMD_SSE)
template <>
struct simd_op<__m128>
{
    static inline __m128 load_aligned(const float* ptr) noexcept
    {
        return _mm_load_ps(ptr);
    }

    static inline void store_aligned(float* ptr, __m128 value) noexcept
    {
        _mm_store_ps(ptr, value);
    }

    static inline __m128 load_unaligned(const float* ptr) noexcept
    {
        return _mm_loadu_ps(ptr);
    }

    static inline void store_unaligned(float* ptr, __m128 value) noexcept
    {
        _mm_storeu_ps(ptr, value);
    }

    static inline __m128 masked_load(const float* ptr, __m128i mask, __m128 default_value) noexcept
    {
        return _mm_blendv_ps(default_value, _mm_loadu_ps(ptr), _mm_castsi128_ps(mask));
    }

    static inline void masked_store(float* ptr, __m128 value, __m128i mask) noexcept
    {
        _mm_storeu_ps(ptr, _mm_blendv_ps(_mm_loadu_ps(ptr), value, _mm_castsi128_ps(mask)));
    }

    template <std::size_t I>
    static inline float extract(__m128 value) noexcept
    {
        return _mm_cvtss_f32(_mm_shuffle_ps(value, value, _MM_SHUFFLE(I, I, I, I)));
    }

    template <std::size_t I>
    static inline __m128 insert(__m128 value, float element) noexcept
    {
        return _mm_insert_ps(value, _mm_set_ss(element), I << 4);
    }

    static inline __m128 fill(float value) noexcept
    {
        return _mm_set1_ps(value);
    }

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

    static inline __m128 min(__m128 a, __m128 b) noexcept
    {
        return _mm_min_ps(a, b);
    }

    static inline __m128 max(__m128 a, __m128 b) noexcept
    {
        return _mm_max_ps(a, b);
    }

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

    static inline __m128 blend(__m128 a, __m128 b, __m128i mask) noexcept
    {
        return _mm_blendv_ps(a, b, _mm_castsi128_ps(mask));
    }

    static inline __m128 sqrt(__m128 a) noexcept
    {
        return _mm_sqrt_ps(a);
    }

    static inline __m128 rsqrt(__m128 a) noexcept
    {
        return _mm_rsqrt_ps(a);
    }

    static inline __m128 reciprocal(__m128 a) noexcept
    {
        return _mm_rcp_ps(a);
    }

    static inline __m128 fma(__m128 a, __m128 b, __m128 c) noexcept
    {
        return _mm_fmadd_ps(a, b, c);
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

    static inline float sum(__m128 a) noexcept
    {
        __m128 sum = _mm_add_ps(a, _mm_movehl_ps(a, a));
        sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
        return _mm_cvtss_f32(sum);
    }

    static inline float dot(__m128 a, __m128 b) noexcept
    {
        return sum(_mm_mul_ps(a, b));
    }

    static inline float min_element(__m128 a) noexcept
    {
        __m128 min_val = _mm_min_ps(a, _mm_movehl_ps(a, a));
        min_val = _mm_min_ss(min_val, _mm_shuffle_ps(min_val, min_val, 1));
        return _mm_cvtss_f32(min_val);
    }

    static inline float max_element(__m128 a) noexcept
    {
        __m128 max_val = _mm_max_ps(a, _mm_movehl_ps(a, a));
        max_val = _mm_max_ss(max_val, _mm_shuffle_ps(max_val, max_val, 1));
        return _mm_cvtss_f32(max_val);
    }
};


template <>
struct simd_op<__m128d>
{
    static inline __m128d load_aligned(const double* ptr) noexcept
    {
        return _mm_load_pd(ptr);
    }

    static inline void store_aligned(double* ptr, __m128d value) noexcept
    {
        _mm_store_pd(ptr, value);
    }

    static inline __m128d load_unaligned(const double* ptr) noexcept
    {
        return _mm_loadu_pd(ptr);
    }

    static inline void store_unaligned(double* ptr, __m128d value) noexcept
    {
        _mm_storeu_pd(ptr, value);
    }

    static inline __m128d masked_load(const double* ptr, __m128i mask, __m128d default_value) noexcept
    {
        return _mm_blendv_pd(default_value, _mm_loadu_pd(ptr), _mm_castsi128_pd(mask));
    }

    static inline void masked_store(double* ptr, __m128d value, __m128i mask) noexcept
    {
        _mm_storeu_pd(ptr, _mm_blendv_pd(_mm_loadu_pd(ptr), value, _mm_castsi128_pd(mask)));
    }

    template <std::size_t I>
    static inline double extract(__m128d value) noexcept
    {
        return _mm_cvtsd_f64(_mm_shuffle_pd(value, value, I));
    }

    template <std::size_t I>
    static inline __m128d insert(__m128d value, double element) noexcept
    {
        return _mm_blend_pd(value, _mm_set1_pd(element), 1 << I);
    }

    static inline __m128d fill(double value) noexcept
    {
        return _mm_set1_pd(value);
    }

    static inline __m128d add(__m128d a, __m128d b) noexcept
    {
        return _mm_add_pd(a, b);
    }

    static inline __m128d sub(__m128d a, __m128d b) noexcept
    {
        return _mm_sub_pd(a, b);
    }

    static inline __m128d mul(__m128d a, __m128d b) noexcept
    {
        return _mm_mul_pd(a, b);
    }

    static inline __m128d div(__m128d a, __m128d b) noexcept
    {
        return _mm_div_pd(a, b);
    }

    static inline __m128d neg(__m128d a) noexcept
    {
        return _mm_sub_pd(_mm_setzero_pd(), a);
    }

    static inline __m128d min(__m128d a, __m128d b) noexcept
    {
        return _mm_min_pd(a, b);
    }

    static inline __m128d max(__m128d a, __m128d b) noexcept
    {
        return _mm_max_pd(a, b);
    }

    static inline __m128d bitwise_and(__m128d a, __m128d b) noexcept
    {
        return _mm_and_pd(a, b);
    }

    static inline __m128d bitwise_or(__m128d a, __m128d b) noexcept
    {
        return _mm_or_pd(a, b);
    }

    static inline __m128d bitwise_xor(__m128d a, __m128d b) noexcept
    {
        return _mm_xor_pd(a, b);
    }

    static inline __m128d bitwise_not(__m128d a) noexcept
    {
        return _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(-1)));
    }

    static inline __m128i cmp_eq(__m128d a, __m128d b) noexcept
    {
        return _mm_castpd_si128(_mm_cmpeq_pd(a, b));
    }

    static inline __m128i cmp_ne(__m128d a, __m128d b) noexcept
    {
        return _mm_castpd_si128(_mm_cmpneq_pd(a, b));
    }

    static inline __m128i cmp_lt(__m128d a, __m128d b) noexcept
    {
        return _mm_castpd_si128(_mm_cmplt_pd(a, b));
    }

    static inline __m128i cmp_le(__m128d a, __m128d b) noexcept
    {
        return _mm_castpd_si128(_mm_cmple_pd(a, b));
    }

    static inline __m128i cmp_gt(__m128d a, __m128d b) noexcept
    {
        return _mm_castpd_si128(_mm_cmpgt_pd(a, b));
    }

    static inline __m128i cmp_ge(__m128d a, __m128d b) noexcept
    {
        return _mm_castpd_si128(_mm_cmpge_pd(a, b));
    }

    static inline __m128d blend(__m128d a, __m128d b, __m128i mask) noexcept
    {
        return _mm_blendv_pd(a, b, _mm_castsi128_pd(mask));
    }

    static inline __m128d sqrt(__m128d a) noexcept
    {
        return _mm_sqrt_pd(a);
    }

    static inline __m128d rsqrt(__m128d a) noexcept
    {
        return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(a));
    }

    static inline __m128d reciprocal(__m128d a) noexcept
    {
        return _mm_div_pd(_mm_set1_pd(1.0), a);
    }

    static inline __m128d fma(__m128d a, __m128d b, __m128d c) noexcept
    {
        return _mm_fmadd_pd(a, b, c);
    }

    static inline __m128d round(__m128d a) noexcept
    {
        return _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    static inline __m128d floor(__m128d a) noexcept
    {
        return _mm_floor_pd(a);
    }

    static inline __m128d ceil(__m128d a) noexcept
    {
        return _mm_ceil_pd(a);
    }

    static inline __m128d trunc(__m128d a) noexcept
    {
        return _mm_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    static inline double sum(__m128d a) noexcept
    {
        __m128d sum = _mm_add_sd(a, _mm_unpackhi_pd(a, a));
        return _mm_cvtsd_f64(sum);
    }

    static inline double dot(__m128d a, __m128d b) noexcept
    {
        return sum(_mm_mul_pd(a, b));
    }

    static inline double min_element(__m128d a) noexcept
    {
        __m128d min_val = _mm_min_sd(a, _mm_unpackhi_pd(a, a));
        return _mm_cvtsd_f64(min_val);
    }

    static inline double max_element(__m128d a) noexcept
    {
        __m128d max_val = _mm_max_sd(a, _mm_unpackhi_pd(a, a));
        return _mm_cvtsd_f64(max_val);
    }
};


template <>
struct simd_op<__m128i>
{
    static inline __m128i load_aligned(const int32_t* ptr) noexcept
    {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    }

    static inline __m128i load_aligned(const uint32_t* ptr) noexcept
    {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    }

    static inline void store_aligned(int32_t* ptr, __m128i value) noexcept
    {
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr), value);
    }

    static inline void store_aligned(uint32_t* ptr, __m128i value) noexcept
    {
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr), value);
    }

    static inline __m128i load_unaligned(const int32_t* ptr) noexcept
    {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    }

    static inline __m128i load_unaligned(const uint32_t* ptr) noexcept
    {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    }

    static inline void store_unaligned(int32_t* ptr, __m128i value) noexcept
    {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), value);
    }

    static inline void store_unaligned(uint32_t* ptr, __m128i value) noexcept
    {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), value);
    }

    static inline __m128i masked_load(const int32_t* ptr, __m128i mask, __m128i default_value) noexcept
    {
        return _mm_blendv_epi8(default_value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)), mask);
    }

    static inline void masked_store(int32_t* ptr, __m128i value, __m128i mask) noexcept
    {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), _mm_blendv_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)), value, mask));
    }

    template <std::size_t I>
    static inline int32_t extract(__m128i value) noexcept
    {
        return _mm_cvtsi128_si32(_mm_shuffle_epi32(value, _MM_SHUFFLE(I, I, I, I)));
    }

    template <std::size_t I>
    static inline __m128i insert(__m128i value, int32_t element) noexcept
    {
        return _mm_insert_epi32(value, element, I);
    }

    static inline __m128i fill(int32_t value) noexcept
    {
        return _mm_set1_epi32(value);
    }

    static inline __m128i fill(uint32_t value) noexcept
    {
        return _mm_set1_epi32(static_cast<int32_t>(value));
    }

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

    static inline __m128i min(__m128i a, __m128i b) noexcept
    {
        return _mm_min_epi32(a, b);
    }

    static inline __m128i max(__m128i a, __m128i b) noexcept
    {
        return _mm_max_epi32(a, b);
    }

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

    static inline __m128i cmp_eq(__m128i a, __m128i b) noexcept
    {
        return _mm_cmpeq_epi32(a, b);
    }

    static inline __m128i cmp_lt(__m128i a, __m128i b) noexcept
    {
        return _mm_cmpgt_epi32(b, a);
    }

    static inline __m128i cmp_le(__m128i a, __m128i b) noexcept
    {
        return _mm_or_si128(cmp_lt(a, b), cmp_eq(a, b));
    }

    static inline __m128i cmp_gt(__m128i a, __m128i b) noexcept
    {
        return _mm_cmpgt_epi32(a, b);
    }

    static inline __m128i cmp_ge(__m128i a, __m128i b) noexcept
    {
        return _mm_or_si128(cmp_gt(a, b), cmp_eq(a, b));
    }

    static inline __m128i blend(__m128i a, __m128i b, __m128i mask) noexcept
    {
        return _mm_or_si128(_mm_and_si128(a, _mm_xor_si128(mask, _mm_set1_epi32(-1))), _mm_and_si128(b, mask));
    }

    static inline bool any(__m128i a) noexcept
    {
        return !_mm_testz_si128(a, a);
    }

    static inline bool all(__m128i a) noexcept
    {
        return _mm_testc_si128(a, _mm_set1_epi32(-1));
    }

    static inline int32_t sum(__m128i a) noexcept
    {
        __m128i sum = _mm_add_epi32(a, _mm_srli_si128(a, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        return _mm_cvtsi128_si32(sum);
    }

    static inline int32_t dot(__m128i a, __m128i b) noexcept
    {
        return sum(_mm_mullo_epi32(a, b));
    }

    static inline int32_t min_element(__m128i a) noexcept
    {
        __m128i min_val = _mm_min_epi32(a, _mm_srli_si128(a, 8));
        min_val = _mm_min_epi32(min_val, _mm_srli_si128(min_val, 4));
        return _mm_cvtsi128_si32(min_val);
    }

    static inline int32_t max_element(__m128i a) noexcept
    {
        __m128i max_val = _mm_max_epi32(a, _mm_srli_si128(a, 8));
        max_val = _mm_max_epi32(max_val, _mm_srli_si128(max_val, 4));
        return _mm_cvtsi128_si32(max_val);
    }
};
#endif


#if defined(ARC_SIMD_AVX)
template <>
struct simd_op<__m256>
{
    static inline __m256 load_aligned(const float* ptr) noexcept
    {
        return _mm256_load_ps(ptr);
    }

    static inline void store_aligned(float* ptr, __m256 value) noexcept
    {
        _mm256_store_ps(ptr, value);
    }

    static inline __m256 load_unaligned(const float* ptr) noexcept
    {
        return _mm256_loadu_ps(ptr);
    }

    static inline void store_unaligned(float* ptr, __m256 value) noexcept
    {
        _mm256_storeu_ps(ptr, value);
    }

    static inline __m256 masked_load(const float* ptr, __m256i mask, __m256 default_value) noexcept
    {
        return _mm256_blendv_ps(default_value, _mm256_loadu_ps(ptr), _mm256_castsi256_ps(mask));
    }

    static inline void masked_store(float* ptr, __m256 value, __m256i mask) noexcept
    {
        _mm256_storeu_ps(ptr, _mm256_blendv_ps(_mm256_loadu_ps(ptr), value, _mm256_castsi256_ps(mask)));
    }

    template <std::size_t I>
    static inline float extract(__m256 value) noexcept
    {
        return _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_permutevar8x32_ps(value, _mm256_set1_epi32(I))));
    }

    template <std::size_t I>
    static inline __m256 insert(__m256 value, float element) noexcept
    {
        return _mm256_blend_ps(value, _mm256_set1_ps(element), 1 << I);
    }

    static inline __m256 fill(float value) noexcept
    {
        return _mm256_set1_ps(value);
    }

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

    static inline __m256 min(__m256 a, __m256 b) noexcept
    {
        return _mm256_min_ps(a, b);
    }

    static inline __m256 max(__m256 a, __m256 b) noexcept
    {
        return _mm256_max_ps(a, b);
    }

    static inline __m256 bitwise_and(__m256 a, __m256 b) noexcept
    {
        return _mm256_and_ps(a, b);
    }

    static inline __m256 bitwise_or(__m256 a, __m256 b) noexcept
    {
        return _mm256_or_ps(a, b);
    }

    static inline __m256 bitwise_xor(__m256 a, __m256 b) noexcept
    {
        return _mm256_xor_ps(a, b);
    }

    static inline __m256 bitwise_not(__m256 a) noexcept
    {
        return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }

    static inline __m256i cmp_eq(__m256 a, __m256 b) noexcept
    {
        return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_EQ_OQ));
    }

    static inline __m256i cmp_ne(__m256 a, __m256 b) noexcept
    {
        return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_NEQ_OQ));
    }

    static inline __m256i cmp_lt(__m256 a, __m256 b) noexcept
    {
        return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LT_OQ));
    }

    static inline __m256i cmp_le(__m256 a, __m256 b) noexcept
    {
        return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LE_OQ));
    }

    static inline __m256i cmp_gt(__m256 a, __m256 b) noexcept
    {
        return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GT_OQ));
    }

    static inline __m256i cmp_ge(__m256 a, __m256 b) noexcept
    {
        return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GE_OQ));
    }

    static inline __m256 blend(__m256 a, __m256 b, __m256i mask) noexcept
    {
        return _mm256_blendv_ps(a, b, _mm256_castsi256_ps(mask));
    }

    static inline __m256 sqrt(__m256 a) noexcept
    {
        return _mm256_sqrt_ps(a);
    }

    static inline __m256 rsqrt(__m256 a) noexcept
    {
        return _mm256_rsqrt_ps(a);
    }

    static inline __m256 reciprocal(__m256 a) noexcept
    {
        return _mm256_rcp_ps(a);
    }

    static inline __m256 fma(__m256 a, __m256 b, __m256 c) noexcept
    {
        return _mm256_fmadd_ps(a, b, c);
    }

    static inline __m256 round(__m256 a) noexcept
    {
        return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    static inline __m256 floor(__m256 a) noexcept
    {
        return _mm256_floor_ps(a);
    }

    static inline __m256 ceil(__m256 a) noexcept
    {
        return _mm256_ceil_ps(a);
    }

    static inline __m256 trunc(__m256 a) noexcept
    {
        return _mm256_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    static inline float sum(__m256 a) noexcept
    {
        return simd_op<__m128>::sum(_mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1)));
    }

    static inline float dot(__m256 a, __m256 b) noexcept
    {
        return sum(_mm256_mul_ps(a, b));
    }

    static inline float min_element(__m256 a) noexcept
    {
        return simd_op<__m128>::min_element(_mm_min_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1)));
    }

    static inline float max_element(__m256 a) noexcept
    {
        return simd_op<__m128>::max_element(_mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1)));
    }
};


template <>
struct simd_op<__m256d>
{
    static inline __m256d load_aligned(const double* ptr) noexcept
    {
        return _mm256_load_pd(ptr);
    }

    static inline void store_aligned(double* ptr, __m256d value) noexcept
    {
        _mm256_store_pd(ptr, value);
    }

    static inline __m256d load_unaligned(const double* ptr) noexcept
    {
        return _mm256_loadu_pd(ptr);
    }

    static inline void store_unaligned(double* ptr, __m256d value) noexcept
    {
        _mm256_storeu_pd(ptr, value);
    }

    static inline __m256d masked_load(const double* ptr, __m256i mask, __m256d default_value) noexcept
    {
        return _mm256_blendv_pd(default_value, _mm256_loadu_pd(ptr), _mm256_castsi256_pd(mask));
    }

    static inline void masked_store(double* ptr, __m256d value, __m256i mask) noexcept
    {
        _mm256_storeu_pd(ptr, _mm256_blendv_pd(_mm256_loadu_pd(ptr), value, _mm256_castsi256_pd(mask)));
    }

    template <std::size_t I>
    static inline double extract(__m256d value) noexcept
    {
        return _mm_cvtsd_f64(_mm256_castpd256_pd128(_mm256_permutevar_pd(value, _mm256_set1_epi64x(I))));
    }

    template <std::size_t I>
    static inline __m256d insert(__m256d value, double element) noexcept
    {
        return _mm256_blend_pd(value, _mm256_set1_pd(element), 1 << I);
    }

    static inline __m256d fill(double value) noexcept
    {
        return _mm256_set1_pd(value);
    }

    static inline __m256d add(__m256d a, __m256d b) noexcept
    {
        return _mm256_add_pd(a, b);
    }

    static inline __m256d sub(__m256d a, __m256d b) noexcept
    {
        return _mm256_sub_pd(a, b);
    }

    static inline __m256d mul(__m256d a, __m256d b) noexcept
    {
        return _mm256_mul_pd(a, b);
    }

    static inline __m256d div(__m256d a, __m256d b) noexcept
    {
        return _mm256_div_pd(a, b);
    }

    static inline __m256d neg(__m256d a) noexcept
    {
        return _mm256_sub_pd(_mm256_setzero_pd(), a);
    }

    static inline __m256d min(__m256d a, __m256d b) noexcept
    {
        return _mm256_min_pd(a, b);
    }

    static inline __m256d max(__m256d a, __m256d b) noexcept
    {
        return _mm256_max_pd(a, b);
    }

    static inline __m256d bitwise_and(__m256d a, __m256d b) noexcept
    {
        return _mm256_and_pd(a, b);
    }

    static inline __m256d bitwise_or(__m256d a, __m256d b) noexcept
    {
        return _mm256_or_pd(a, b);
    }

    static inline __m256d bitwise_xor(__m256d a, __m256d b) noexcept
    {
        return _mm256_xor_pd(a, b);
    }

    static inline __m256d bitwise_not(__m256d a) noexcept
    {
        return _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
    }

    static inline __m256i cmp_eq(__m256d a, __m256d b) noexcept
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
    }

    static inline __m256i cmp_ne(__m256d a, __m256d b) noexcept
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OQ));
    }

    static inline __m256i cmp_lt(__m256d a, __m256d b) noexcept
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
    }

    static inline __m256i cmp_le(__m256d a, __m256d b) noexcept
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
    }

    static inline __m256i cmp_gt(__m256d a, __m256d b) noexcept
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OQ));
    }

    static inline __m256i cmp_ge(__m256d a, __m256d b) noexcept
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OQ));
    }

    static inline __m256d blend(__m256d a, __m256d b, __m256i mask) noexcept
    {
        return _mm256_blendv_pd(a, b, _mm256_castsi256_pd(mask));
    }

    static inline __m256d sqrt(__m256d a) noexcept
    {
        return _mm256_sqrt_pd(a);
    }

    static inline __m256d rsqrt(__m256d a) noexcept
    {
        return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(a));
    }

    static inline __m256d reciprocal(__m256d a) noexcept
    {
        return _mm256_div_pd(_mm256_set1_pd(1.0), a);
    }

    static inline __m256d fma(__m256d a, __m256d b, __m256d c) noexcept
    {
        return _mm256_fmadd_pd(a, b, c);
    }

    static inline __m256d round(__m256d a) noexcept
    {
        return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    static inline __m256d floor(__m256d a) noexcept
    {
        return _mm256_floor_pd(a);
    }

    static inline __m256d ceil(__m256d a) noexcept
    {
        return _mm256_ceil_pd(a);
    }

    static inline __m256d trunc(__m256d a) noexcept
    {
        return _mm256_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    static inline double sum(__m256d a) noexcept
    {
        return simd_op< __m128d>::sum(_mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1)));
    }

    static inline double dot(__m256d a, __m256d b) noexcept
    {
        return sum(_mm256_mul_pd(a, b));
    }

    static inline double min_element(__m256d a) noexcept
    {
        return simd_op< __m128d>::min_element(_mm_min_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1)));
    }

    static inline double max_element(__m256d a) noexcept
    {
        return simd_op< __m128d>::max_element(_mm_max_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1)));
    }
};


template <>
struct simd_op<__m256i>
{
    static inline __m256i load_aligned(const int32_t* ptr) noexcept
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static inline __m256i load_aligned(const uint32_t* ptr) noexcept
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static inline void store_aligned(int32_t* ptr, __m256i value) noexcept
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), value);
    }

    static inline void store_aligned(uint32_t* ptr, __m256i value) noexcept
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), value);
    }

    static inline __m256i load_unaligned(const int32_t* ptr) noexcept
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static inline __m256i load_unaligned(const uint32_t* ptr) noexcept
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static inline void store_unaligned(int32_t* ptr, __m256i value) noexcept
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), value);
    }

    static inline void store_unaligned(uint32_t* ptr, __m256i value) noexcept
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), value);
    }

    static inline __m256i masked_load(const int32_t* ptr, __m256i mask, __m256i default_value) noexcept
    {
        return _mm256_blendv_epi8(default_value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)), mask);
    }

    static inline void masked_store(int32_t* ptr, __m256i value, __m256i mask) noexcept
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), _mm256_blendv_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)), value, mask));
    }

    template <std::size_t I>
    static inline int32_t extract(__m256i value) noexcept
    {
        return _mm256_extract_epi32(value, I);
    }

    template <std::size_t I>
    static inline __m256i insert(__m256i value, int32_t element) noexcept
    {
        return _mm256_insert_epi32(value, element, I);
    }

    static inline __m256i fill(int32_t value) noexcept
    {
        return _mm256_set1_epi32(value);
    }

    static inline __m256i fill(uint32_t value) noexcept
    {
        return _mm256_set1_epi32(static_cast<int32_t>(value));
    }

    static inline __m256i add(__m256i a, __m256i b) noexcept
    {
        return _mm256_add_epi32(a, b);
    }

    static inline __m256i sub(__m256i a, __m256i b) noexcept
    {
        return _mm256_sub_epi32(a, b);
    }

    static inline __m256i neg(__m256i a) noexcept
    {
        return _mm256_sub_epi32(_mm256_setzero_si256(), a);
    }

    static inline __m256i min(__m256i a, __m256i b) noexcept
    {
        return _mm256_min_epi32(a, b);
    }

    static inline __m256i max(__m256i a, __m256i b) noexcept
    {
        return _mm256_max_epi32(a, b);
    }

    static inline __m256i bitwise_and(__m256i a, __m256i b) noexcept
    {
        return _mm256_and_si256(a, b);
    }

    static inline __m256i bitwise_or(__m256i a, __m256i b) noexcept
    {
        return _mm256_or_si256(a, b);
    }

    static inline __m256i bitwise_xor(__m256i a, __m256i b) noexcept
    {
        return _mm256_xor_si256(a, b);
    }

    static inline __m256i bitwise_not(__m256i a) noexcept
    {
        return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
    }

    static inline __m256i cmp_eq(__m256i a, __m256i b) noexcept
    {
        return _mm256_cmpeq_epi32(a, b);
    }

    static inline __m256i cmp_lt(__m256i a, __m256i b) noexcept
    {
        return _mm256_cmpgt_epi32(b, a);
    }

    static inline __m256i cmp_le(__m256i a, __m256i b) noexcept
    {
        return _mm256_or_si256(cmp_lt(a, b), cmp_eq(a, b));
    }

    static inline __m256i cmp_gt(__m256i a, __m256i b) noexcept
    {
        return _mm256_cmpgt_epi32(a, b);
    }

    static inline __m256i cmp_ge(__m256i a, __m256i b) noexcept
    {
        return _mm256_or_si256(cmp_gt(a, b), cmp_eq(a, b));
    }

    static inline __m256i blend(__m256i a, __m256i b, __m256i mask) noexcept
    {
        return _mm256_or_si256(_mm256_and_si256(a, _mm256_xor_si256(mask, _mm256_set1_epi32(-1))), _mm256_and_si256(b, mask));
    }

    static inline bool any(__m256i a) noexcept
    {
        return !_mm256_testz_si256(a, a);
    }

    static inline bool all(__m256i a) noexcept
    {
        return _mm256_testc_si256(a, _mm256_set1_epi32(-1));
    }

    static inline int32_t sum(__m256i a) noexcept
    {
        return simd_op<__m128i>::sum(_mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)));
    }

    static inline int32_t dot(__m256i a, __m256i b) noexcept
    {
        return sum(_mm256_mullo_epi32(a, b));
    }

    static inline int32_t min_element(__m256i a) noexcept
    {
        return simd_op<__m128i>::min_element(_mm_min_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)));
    }

    static inline int32_t max_element(__m256i a) noexcept
    {
        return simd_op<__m128i>::max_element(_mm_max_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)));
    }
};
#endif


#if defined(ARC_SIMD_AVX512)
template <>
struct simd_op<__m512>
{
    static inline __m512 load_aligned(const float* ptr) noexcept
    {
        return _mm512_load_ps(ptr);
    }

    static inline void store_aligned(float* ptr, __m512 value) noexcept
    {
        _mm512_store_ps(ptr, value);
    }

    static inline __m512 load_unaligned(const float* ptr) noexcept
    {
        return _mm512_loadu_ps(ptr);
    }

    static inline void store_unaligned(float* ptr, __m512 value) noexcept
    {
        _mm512_storeu_ps(ptr, value);
    }

    static inline __m512 masked_load(const float* ptr, __mmask16 mask, __m512 default_value) noexcept
    {
        return _mm512_mask_blend_ps(mask, default_value, _mm512_loadu_ps(ptr));
    }

    static inline void masked_store(float* ptr, __m512 value, __mmask16 mask) noexcept
    {
        _mm512_mask_storeu_ps(ptr, mask, value);
    }

    template <std::size_t I>
    static inline float extract(__m512 value) noexcept
    {
        return _mm512_cvtss_f32(_mm512_permute_ps(value, _MM_SHUFFLE(I, I, I, I)));
    }

    template <std::size_t I>
    static inline __m512 insert(__m512 value, float element) noexcept
    {
        return _mm512_mask_blend_ps(1 << I, value, _mm512_set1_ps(element));
    }

    static inline __m512 fill(float value) noexcept
    {
        return _mm512_set1_ps(value);
    }

    static inline __m512 add(__m512 a, __m512 b) noexcept
    {
        return _mm512_add_ps(a, b);
    }

    static inline __m512 sub(__m512 a, __m512 b) noexcept
    {
        return _mm512_sub_ps(a, b);
    }

    static inline __m512 mul(__m512 a, __m512 b) noexcept
    {
        return _mm512_mul_ps(a, b);
    }

    static inline __m512 div(__m512 a, __m512 b) noexcept
    {
        return _mm512_div_ps(a, b);
    }

    static inline __m512 neg(__m512 a) noexcept
    {
        return _mm512_sub_ps(_mm512_setzero_ps(), a);
    }

    static inline __m512 min(__m512 a, __m512 b) noexcept
    {
        return _mm512_min_ps(a, b);
    }

    static inline __m512 max(__m512 a, __m512 b) noexcept
    {
        return _mm512_max_ps(a, b);
    }

    static inline __m512 bitwise_and(__m512 a, __m512 b) noexcept
    {
        return _mm512_and_ps(a, b);
    }

    static inline __m512 bitwise_or(__m512 a, __m512 b) noexcept
    {
        return _mm512_or_ps(a, b);
    }

    static inline __m512 bitwise_xor(__m512 a, __m512 b) noexcept
    {
        return _mm512_xor_ps(a, b);
    }

    static inline __m512 bitwise_not(__m512 a) noexcept
    {
        return _mm512_xor_ps(a, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
    }

    static inline __mmask16 cmp_eq(__m512 a, __m512 b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
    }

    static inline __mmask16 cmp_ne(__m512 a, __m512 b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
    }

    static inline __mmask16 cmp_lt(__m512 a, __m512 b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
    }

    static inline __mmask16 cmp_le(__m512 a, __m512 b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
    }

    static inline __mmask16 cmp_gt(__m512 a, __m512 b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
    }

    static inline __mmask16 cmp_ge(__m512 a, __m512 b) noexcept
    {
        return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
    }

    static inline __m512 blend(__m512 a, __m512 b, __mmask16 mask) noexcept
    {
        return _mm512_mask_blend_ps(mask, a, b);
    }

    static inline __m512 sqrt(__m512 a) noexcept
    {
        return _mm512_sqrt_ps(a);
    }

    static inline __m512 rsqrt(__m512 a) noexcept
    {
        return _mm512_rsqrt14_ps(a);
    }

    static inline __m512 reciprocal(__m512 a) noexcept
    {
        return _mm512_rcp14_ps(a);
    }

    static inline __m512 fma(__m512 a, __m512 b, __m512 c) noexcept
    {
        return _mm512_fmadd_ps(a, b, c);
    }

    static inline __m512 round(__m512 a) noexcept
    {
        return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    static inline __m512 floor(__m512 a) noexcept
    {
        return _mm512_floor_ps(a);
    }

    static inline __m512 ceil(__m512 a) noexcept
    {
        return _mm512_ceil_ps(a);
    }

    static inline __m512 trunc(__m512 a) noexcept
    {
        return _mm512_roundscale_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
};


template <>
struct simd_op<__m512d>
{
    static inline __m512d load_aligned(const double* ptr) noexcept
    {
        return _mm512_load_pd(ptr);
    }

    static inline void store_aligned(double* ptr, __m512d value) noexcept
    {
        _mm512_store_pd(ptr, value);
    }

    static inline __m512d load_unaligned(const double* ptr) noexcept
    {
        return _mm512_loadu_pd(ptr);
    }

    static inline void store_unaligned(double* ptr, __m512d value) noexcept
    {
        _mm512_storeu_pd(ptr, value);
    }

    static inline __m512d masked_load(const double* ptr, __mmask8 mask, __m512d default_value) noexcept
    {
        return _mm512_mask_blend_pd(mask, default_value, _mm512_loadu_pd(ptr));
    }

    static inline void masked_store(double* ptr, __m512d value, __mmask8 mask) noexcept
    {
        _mm512_mask_storeu_pd(ptr, mask, value);
    }

    template <std::size_t I>
    static inline double extract(__m512d value) noexcept
    {
        return _mm512_cvtsd_f64(_mm512_permute_pd(value, 1 << I));
    }

    template <std::size_t I>
    static inline __m512d insert(__m512d value, double element) noexcept
    {
        return _mm512_mask_blend_pd(1 << I, value, _mm512_set1_pd(element));
    }

    static inline __m512d fill(double value) noexcept
    {
        return _mm512_set1_pd(value);
    }

    static inline __m512d add(__m512d a, __m512d b) noexcept
    {
        return _mm512_add_pd(a, b);
    }

    static inline __m512d sub(__m512d a, __m512d b) noexcept
    {
        return _mm512_sub_pd(a, b);
    }

    static inline __m512d mul(__m512d a, __m512d b) noexcept
    {
        return _mm512_mul_pd(a, b);
    }

    static inline __m512d div(__m512d a, __m512d b) noexcept
    {
        return _mm512_div_pd(a, b);
    }

    static inline __m512d neg(__m512d a) noexcept
    {
        return _mm512_sub_pd(_mm512_setzero_pd(), a);
    }

    static inline __m512d min(__m512d a, __m512d b) noexcept
    {
        return _mm512_min_pd(a, b);
    }

    static inline __m512d max(__m512d a, __m512d b) noexcept
    {
        return _mm512_max_pd(a, b);
    }

    static inline __m512d bitwise_and(__m512d a, __m512d b) noexcept
    {
        return _mm512_and_pd(a, b);
    }

    static inline __m512d bitwise_or(__m512d a, __m512d b) noexcept
    {
        return _mm512_or_pd(a, b);
    }

    static inline __m512d bitwise_xor(__m512d a, __m512d b) noexcept
    {
        return _mm512_xor_pd(a, b);
    }

    static inline __m512d bitwise_not(__m512d a) noexcept
    {
        return _mm512_xor_pd(a, _mm512_castsi512_pd(_mm512_set1_epi64(-1)));
    }

    static inline __mmask8 cmp_eq(__m512d a, __m512d b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
    }

    static inline __mmask8 cmp_ne(__m512d a, __m512d b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
    }

    static inline __mmask8 cmp_lt(__m512d a, __m512d b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
    }

    static inline __mmask8 cmp_le(__m512d a, __m512d b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
    }

    static inline __mmask8 cmp_gt(__m512d a, __m512d b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
    }

    static inline __mmask8 cmp_ge(__m512d a, __m512d b) noexcept
    {
        return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
    }

    static inline __m512d blend(__m512d a, __m512d b, __mmask8 mask) noexcept
    {
        return _mm512_mask_blend_pd(mask, a, b);
    }

    static inline __m512d sqrt(__m512d a) noexcept
    {
        return _mm512_sqrt_pd(a);
    }

    static inline __m512d rsqrt(__m512d a) noexcept
    {
        return _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(a));
    }

    static inline __m512d reciprocal(__m512d a) noexcept
    {
        return _mm512_div_pd(_mm512_set1_pd(1.0), a);
    }

    static inline __m512d fma(__m512d a, __m512d b, __m512d c) noexcept
    {
        return _mm512_fmadd_pd(a, b, c);
    }

    static inline __m512d round(__m512d a) noexcept
    {
        return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    static inline __m512d floor(__m512d a) noexcept
    {
        return _mm512_floor_pd(a);
    }

    static inline __m512d ceil(__m512d a) noexcept
    {
        return _mm512_ceil_pd(a);
    }

    static inline __m512d trunc(__m512d a) noexcept
    {
        return _mm512_roundscale_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
};


template <>
struct simd_op<__m512i>
{
    static inline __m512i load_aligned(const int32_t* ptr) noexcept
    {
        return _mm512_load_si512(reinterpret_cast<const void*>(ptr));
    }

    static inline __m512i load_aligned(const uint32_t* ptr) noexcept
    {
        return _mm512_load_si512(reinterpret_cast<const void*>(ptr));
    }

    static inline void store_aligned(int32_t* ptr, __m512i value) noexcept
    {
        _mm512_store_si512(reinterpret_cast<void*>(ptr), value);
    }

    static inline void store_aligned(uint32_t* ptr, __m512i value) noexcept
    {
        _mm512_store_si512(reinterpret_cast<void*>(ptr), value);
    }

    static inline __m512i load_unaligned(const int32_t* ptr) noexcept
    {
        return _mm512_loadu_si512(reinterpret_cast<const void*>(ptr));
    }

    static inline __m512i load_unaligned(const uint32_t* ptr) noexcept
    {
        return _mm512_loadu_si512(reinterpret_cast<const void*>(ptr));
    }

    static inline void store_unaligned(int32_t* ptr, __m512i value) noexcept
    {
        _mm512_storeu_si512(reinterpret_cast<void*>(ptr), value);
    }

    static inline void store_unaligned(uint32_t* ptr, __m512i value) noexcept
    {
        _mm512_storeu_si512(reinterpret_cast<void*>(ptr), value);
    }

    static inline __m512i masked_load(const int32_t* ptr, __mmask16 mask, __m512i default_value) noexcept
    {
        return _mm512_mask_blend_epi32(mask, default_value, _mm512_loadu_si512(reinterpret_cast<const void*>(ptr)));
    }

    static inline void masked_store(int32_t* ptr, __m512i value, __mmask16 mask) noexcept
    {
        _mm512_mask_storeu_epi32(reinterpret_cast<void*>(ptr), mask, value);
    }

    template <std::size_t I>
    static inline int32_t extract(__m512i value) noexcept
    {
        return _mm512_extract_epi32(value, I);
    }

    template <std::size_t I>
    static inline __m512i insert(__m512i value, int32_t element) noexcept
    {
        return _mm512_insert_epi32(value, element, I);
    }

    static inline __m512i fill(int32_t value) noexcept
    {
        return _mm512_set1_epi32(value);
    }

    static inline __m512i fill(uint32_t value) noexcept
    {
        return _mm512_set1_epi32(static_cast<int32_t>(value));
    }

    static inline __m512i add(__m512i a, __m512i b) noexcept
    {
        return _mm512_add_epi32(a, b);
    }

    static inline __m512i sub(__m512i a, __m512i b) noexcept
    {
        return _mm512_sub_epi32(a, b);
    }

    static inline __m512i neg(__m512i a) noexcept
    {
        return _mm512_sub_epi32(_mm512_setzero_si512(), a);
    }

    static inline __m512i min(__m512i a, __m512i b) noexcept
    {
        return _mm512_min_epi32(a, b);
    }

    static inline __m512i max(__m512i a, __m512i b) noexcept
    {
        return _mm512_max_epi32(a, b);
    }

    static inline __m512i bitwise_and(__m512i a, __m512i b) noexcept
    {
        return _mm512_and_si512(a, b);
    }

    static inline __m512i bitwise_or(__m512i a, __m512i b) noexcept
    {
        return _mm512_or_si512(a, b);
    }

    static inline __m512i bitwise_xor(__m512i a, __m512i b) noexcept
    {
        return _mm512_xor_si512(a, b);
    }

    static inline __m512i bitwise_not(__m512i a) noexcept
    {
        return _mm512_xor_si512(a, _mm512_set1_epi32(-1));
    }

    static inline __mmask16 cmp_eq(__m512i a, __m512i b) noexcept
    {
        return _mm512_cmpeq_epi32_mask(a, b);
    }

    static inline __mmask16 cmp_lt(__m512i a, __m512i b) noexcept
    {
        return _mm512_cmpgt_epi32_mask(b, a);
    }

    static inline __mmask16 cmp_le(__m512i a, __m512i b) noexcept
    {
        return cmp_lt(a, b) | cmp_eq(a, b);
    }

    static inline __mmask16 cmp_gt(__m512i a, __m512i b) noexcept
    {
        return _mm512_cmpgt_epi32_mask(a, b);
    }

    static inline __mmask16 cmp_ge(__m512i a, __m512i b) noexcept
    {
        return cmp_gt(a, b) | cmp_eq(a, b);
    }

    static inline __m512i blend(__m512i a, __m512i b, __mmask16 mask) noexcept
    {
        return _mm512_mask_blend_epi32(mask, a, b);
    }

    static inline bool any(__m512i a) noexcept
    {
        return !_mm512_test_epi32_mask(a, a) == 0;
    }

    static inline bool all(__m512i a) noexcept
    {
        return _mm512_test_epi32_mask(a, _mm512_set1_epi32(-1)) != 0;
    }
};
#endif
}
