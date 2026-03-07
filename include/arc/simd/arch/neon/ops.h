#pragma once

#include "arc/simd/arch/neon/detect.h"

namespace arc
{

#if defined(ARC_SIMD_NEON)
template <>
struct simd_op<float32x4_t>
{
    static inline float32x4_t load(const float* ptr) noexcept
    {
        return vld1q_f32(ptr);
    }

    static inline void store(float* ptr, float32x4_t value) noexcept
    {
        vst1q_f32(ptr, value);
    }

    static inline float32x4_t load_unaligned(const float* ptr) noexcept
    {
        return vld1q_f32(ptr);
    }

    static inline void store_unaligned(float* ptr, float32x4_t value) noexcept
    {
        vst1q_f32(ptr, value);
    }

    static inline float32x4_t fill(float value) noexcept
    {
        return vdupq_n_f32(value);
    }

    static inline float32x4_t add(float32x4_t a, float32x4_t b) noexcept
    {
        return vaddq_f32(a, b);
    }

    static inline float32x4_t sub(float32x4_t a, float32x4_t b) noexcept
    {
        return vsubq_f32(a, b);
    }

    static inline float32x4_t mul(float32x4_t a, float32x4_t b) noexcept
    {
        return vmulq_f32(a, b);
    }

    static inline float32x4_t div(float32x4_t a, float32x4_t b) noexcept
    {
        return vdivq_f32(a, b);
    }

    static inline float32x4_t neg(float32x4_t a) noexcept
    {
        return vnegq_f32(a);
    }

    static inline float32x4_t min(float32x4_t a, float32x4_t b) noexcept
    {
        return vminq_f32(a, b);
    }

    static inline float32x4_t max(float32x4_t a, float32x4_t b) noexcept
    {
        return vmaxq_f32(a, b);
    }

    static inline float32x4_t bitwise_and(float32x4_t a, float32x4_t b) noexcept
    {
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
    }

    static inline float32x4_t bitwise_or(float32x4_t a, float32x4_t b) noexcept
    {
        return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
    }

    static inline float32x4_t bitwise_xor(float32x4_t a, float32x4_t b) noexcept
    {
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
    }

    static inline float32x4_t bitwise_not(float32x4_t a) noexcept
    {
        return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a)));
    }

    static inline uint32x4_t cmp_eq(float32x4_t a, float32x4_t b) noexcept
    {
        return vceqq_f32(a, b);
    }

    static inline uint32x4_t cmp_ne(float32x4_t a, float32x4_t b) noexcept
    {
        return vmvnq_u32(vceqq_f32(a, b));
    }

    static inline uint32x4_t cmp_lt(float32x4_t a, float32x4_t b) noexcept
    {
        return vcltq_f32(a, b);
    }

    static inline uint32x4_t cmp_le(float32x4_t a, float32x4_t b) noexcept
    {
        return vcleq_f32(a, b);
    }

    static inline uint32x4_t cmp_gt(float32x4_t a, float32x4_t b) noexcept
    {
        return vcgtq_f32(a, b);
    }

    static inline uint32x4_t cmp_ge(float32x4_t a, float32x4_t b) noexcept
    {
        return vcgeq_f32(a, b);
    }

    static inline float32x4_t blend(float32x4_t a, float32x4_t b, uint32x4_t mask) noexcept
    {
        return vbslq_f32(mask, b, a);
    }

    static inline float32x4_t sqrt(float32x4_t a) noexcept
    {
        return vsqrtq_f32(a);
    }

    static inline float32x4_t round(float32x4_t a) noexcept
    {
        return vrndnq_f32(a);
    }

    static inline float32x4_t floor(float32x4_t a) noexcept
    {
        return vrndmq_f32(a);
    }

    static inline float32x4_t ceil(float32x4_t a) noexcept
    {
        return vrndpq_f32(a);
    }

    static inline float32x4_t trunc(float32x4_t a) noexcept
    {
        return vrndq_f32(a);
    }

    static inline float sum(float32x4_t a) noexcept
    {
        float32x2_t sum = vadd_f32(vget_low_f32(a), vget_high_f32(a));
        return vget_lane_f32(vpadd_f32(sum, sum), 0);
    }

    static inline float dot(float32x4_t a, float32x4_t b) noexcept
    {
        return sum(vmulq_f32(a, b));
    }

    static inline float min_element(float32x4_t a) noexcept
    {
        float32x2_t min_val = vpmin_f32(vget_low_f32(a), vget_high_f32(a));
        return vget_lane_f32(vpmin_f32(min_val, min_val), 0);
    }

    static inline float max_element(float32x4_t a) noexcept
    {
        float32x2_t max_val = vpmax_f32(vget_low_f32(a), vget_high_f32(a));
        return vget_lane_f32(vpmax_f32(max_val, max_val), 0);
    }
};


template <>
struct simd_op<int32x4_t>
{
    static inline int32x4_t load(const int32_t* ptr) noexcept
    {
        return vld1q_s32(ptr);
    }

    static inline void store(int32_t* ptr, int32x4_t value) noexcept
    {
        vst1q_s32(ptr, value);
    }

    static inline int32x4_t load_unaligned(const int32_t* ptr) noexcept
    {
        return vld1q_s32(ptr);
    }

    static inline void store_unaligned(int32_t* ptr, int32x4_t value) noexcept
    {
        vst1q_s32(ptr, value);
    }

    static inline int32x4_t fill(int32_t value) noexcept
    {
        return vdupq_n_s32(value);
    }

    static inline int32x4_t add(int32x4_t a, int32x4_t b) noexcept
    {
        return vaddq_s32(a, b);
    }

    static inline int32x4_t sub(int32x4_t a, int32x4_t b) noexcept
    {
        return vsubq_s32(a, b);
    }

    static inline int32x4_t neg(int32x4_t a) noexcept
    {
        return vnegq_s32(a);
    }

    static inline int32x4_t min(int32x4_t a, int32x4_t b) noexcept
    {
        return vminq_s32(a, b);
    }

    static inline int32x4_t max(int32x4_t a, int32x4_t b) noexcept
    {
        return vmaxq_s32(a, b);
    }

    static inline int32x4_t bitwise_and(int32x4_t a, int32x4_t b) noexcept
    {
        return vandq_s32(a, b);
    }

    static inline int32x4_t bitwise_or(int32x4_t a, int32x4_t b) noexcept
    {
        return vorrq_s32(a, b);
    }

    static inline int32x4_t bitwise_xor(int32x4_t a, int32x4_t b) noexcept
    {
        return veorq_s32(a, b);
    }

    static inline int32x4_t bitwise_not(int32x4_t a) noexcept
    {
        return vmvnq_s32(a);
    }

    static inline uint32x4_t cmp_eq(int32x4_t a, int32x4_t b) noexcept
    {
        return vceqq_s32(a, b);
    }

    static inline uint32x4_t cmp_lt(int32x4_t a, int32x4_t b) noexcept
    {
        return vcltq_s32(a, b);
    }

    static inline uint32x4_t cmp_le(int32x4_t a, int32x4_t b) noexcept
    {
        return vcleq_s32(a, b);
    }

    static inline uint32x4_t cmp_gt(int32x4_t a, int32x4_t b) noexcept
    {
        return vcgtq_s32(a, b);
    }

    static inline uint32x4_t cmp_ge(int32x4_t a, int32x4_t b) noexcept
    {
        return vcgeq_s32(a, b);
    }

    static inline int32x4_t blend(int32x4_t a, int32x4_t b, uint32x4_t mask) noexcept
    {
        return vbslq_s32(mask, b, a);
    }

    static inline bool any(uint32x4_t a) noexcept
    {
        uint32x2_t tmp = vorr_u32(vget_low_u32(a), vget_high_u32(a));
        return vget_lane_u32(vpmax_u32(tmp, tmp), 0) != 0;
    }

    static inline bool all(uint32x4_t a) noexcept
    {
        uint32x2_t tmp = vand_u32(vget_low_u32(a), vget_high_u32(a));
        return vget_lane_u32(vpmin_u32(tmp, tmp), 0) != 0;
    }

    static inline int32_t sum(int32x4_t a) noexcept
    {
        int32x2_t sum = vadd_s32(vget_low_s32(a), vget_high_s32(a));
        return vget_lane_s32(vpadd_s32(sum, sum), 0);
    }

    static inline int32_t dot(int32x4_t a, int32x4_t b) noexcept
    {
        return sum(vmulq_s32(a, b));
    }

    static inline int32_t min_element(int32x4_t a) noexcept
    {
        int32x2_t min_val = vpmin_s32(vget_low_s32(a), vget_high_s32(a));
        return vget_lane_s32(vpmin_s32(min_val, min_val), 0);
    }

    static inline int32_t max_element(int32x4_t a) noexcept
    {
        int32x2_t max_val = vpmax_s32(vget_low_s32(a), vget_high_s32(a));
        return vget_lane_s32(vpmax_s32(max_val, max_val), 0);
    }
};
#endif


#if defined(ARC_SIMD_NEON_FP64)
template <>
struct simd_op<float64x2_t>
{
    static inline float64x2_t load(const double* ptr) noexcept
    {
        return vld1q_f64(ptr);
    }

    static inline void store(double* ptr, float64x2_t value) noexcept
    {
        vst1q_f64(ptr, value);
    }

    static inline float64x2_t load_unaligned(const double* ptr) noexcept
    {
        return vld1q_f64(ptr);
    }

    static inline void store_unaligned(double* ptr, float64x2_t value) noexcept
    {
        vst1q_f64(ptr, value);
    }

    static inline float64x2_t fill(double value) noexcept
    {
        return vdupq_n_f64(value);
    }

    static inline float64x2_t add(float64x2_t a, float64x2_t b) noexcept
    {
        return vaddq_f64(a, b);
    }

    static inline float64x2_t sub(float64x2_t a, float64x2_t b) noexcept
    {
        return vsubq_f64(a, b);
    }

    static inline float64x2_t mul(float64x2_t a, float64x2_t b) noexcept
    {
        return vmulq_f64(a, b);
    }

    static inline float64x2_t div(float64x2_t a, float64x2_t b) noexcept
    {
        return vdivq_f64(a, b);
    }

    static inline float64x2_t neg(float64x2_t a) noexcept
    {
        return vnegq_f64(a);
    }

    static inline float64x2_t min(float64x2_t a, float64x2_t b) noexcept
    {
        return vminq_f64(a, b);
    }

    static inline float64x2_t max(float64x2_t a, float64x2_t b) noexcept
    {
        return vmaxq_f64(a, b);
    }

    static inline float64x2_t bitwise_and(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
    }

    static inline float64x2_t bitwise_or(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
    }

    static inline float64x2_t bitwise_xor(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
    }

    static inline float64x2_t bitwise_not(float64x2_t a) noexcept
    {
        return vreinterpretq_f64_u64(vmvnq_u64(vreinterpretq_u64_f64(a)));
    }

    static inline uint32x4_t cmp_eq(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_u32_u64(vceqq_f64(a, b));
    }

    static inline uint32x4_t cmp_ne(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_u32_u64(vmvnq_u64(vceqq_f64(a, b)));
    }

    static inline uint32x4_t cmp_lt(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_u32_u64(vcltq_f64(a, b));
    }

    static inline uint32x4_t cmp_le(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_u32_u64(vcleq_f64(a, b));
    }

    static inline uint32x4_t cmp_gt(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_u32_u64(vcgtq_f64(a, b));
    }

    static inline uint32x4_t cmp_ge(float64x2_t a, float64x2_t b) noexcept
    {
        return vreinterpretq_u32_u64(vcgeq_f64(a, b));
    }

    static inline float64x2_t blend(float64x2_t a, float64x2_t b, uint32x4_t mask) noexcept
    {
        return vbslq_f64(vreinterpretq_u64_u32(mask), b, a);
    }

    static inline float64x2_t sqrt(float64x2_t a) noexcept
    {
        return vsqrtq_f64(a);
    }

    static inline float64x2_t round(float64x2_t a) noexcept
    {
        return vrndnq_f64(a);
    }

    static inline float64x2_t floor(float64x2_t a) noexcept
    {
        return vrndmq_f64(a);
    }

    static inline float64x2_t ceil(float64x2_t a) noexcept
    {
        return vrndpq_f64(a);
    }

    static inline float64x2_t trunc(float64x2_t a) noexcept
    {
        return vrndq_f64(a);
    }

    static inline double sum(float64x2_t a) noexcept
    {
        return vget_lane_f64(vpaddq_f64(a, a), 0);
    }

    static inline double dot(float64x2_t a, float64x2_t b) noexcept
    {
        return sum(vmulq_f64(a, b));
    }

    static inline double min_element(float64x2_t a) noexcept
    {
        return vget_lane_f64(vpminq_f64(a, a), 0);
    }

    static inline double max_element(float64x2_t a) noexcept
    {
        return vget_lane_f64(vpmaxq_f64(a, a), 0);
    }
};
#endif

} // namespace arc