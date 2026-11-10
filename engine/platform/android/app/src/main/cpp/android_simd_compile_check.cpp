// Compiled into the Android native host to catch cross-architecture header
// leaks and missing NEON specializations before APK linking.
#include <arc/simd/arch/arch.h>

#include <type_traits>

#if !defined(__aarch64__)
#error "Android arm64 host must compile for AArch64"
#endif

#if !defined(ARC_SIMD_NEON)
#error "Android arm64 build must select the NEON SIMD backend"
#endif

#if defined(ARC_SIMD_SSE) || defined(ARC_SIMD_AVX) || defined(ARC_SIMD_AVX2)
#error "x64 SIMD feature macros leaked into Android arm64"
#endif

static_assert(std::is_same_v<arc::simd::simd_register_t<float, 4>, float32x4_t>);
static_assert(std::is_same_v<arc::simd::simd_register_t<std::int32_t, 4>, int32x4_t>);
static_assert(std::is_same_v<arc::simd::simd_register_t<std::uint32_t, 4>, uint32x4_t>);
static_assert(arc::simd::simd_max_lanes<float> == 4);
static_assert(arc::simd::simd_max_lanes<std::int32_t> == 4);

// Compile representative operations, not just the register aliases. This
// exercises NEON intrinsics during every Android arm64 APK build.
[[maybe_unused]] double check_neon_operations_compile(const float* floats, const std::int32_t* ints,
                                                       const double* doubles) noexcept
{
    using float_ops = arc::simd::simd_op<float32x4_t>;
    using int_ops = arc::simd::simd_op<int32x4_t>;

    const auto f = float_ops::load_unaligned(floats);
    const auto i = int_ops::load_unaligned(ints);
    const auto f_mask = float_ops::cmp_ne(f, float_ops::fill(0.0f));
    const auto f_bits = float_ops::bitwise_not(f);
    const auto f_value = float_ops::sum(float_ops::blend(f, f_bits, f_mask));
    const auto i_value = int_ops::sum(i);

#if defined(ARC_SIMD_NEON_FP64)
    static_assert(std::is_same_v<arc::simd::simd_register_t<double, 2>, float64x2_t>);
    static_assert(arc::simd::simd_max_lanes<double> == 2);

    using double_ops = arc::simd::simd_op<float64x2_t>;
    const auto d = double_ops::load_unaligned(doubles);
    const auto d_bits = double_ops::bitwise_not(d);
    const auto d_eq = double_ops::cmp_eq(d, d_bits);
    const auto d_ne = double_ops::cmp_ne(d, d_bits);
    const auto d_value = double_ops::sum(d) + double_ops::dot(d, d) + double_ops::min_element(d) +
                         double_ops::max_element(d);
    return f_value + static_cast<double>(i_value) + d_value +
           static_cast<double>(vgetq_lane_u32(d_eq, 0) + vgetq_lane_u32(d_ne, 0));
#else
    (void)doubles;
    return f_value + static_cast<double>(i_value);
#endif
}
