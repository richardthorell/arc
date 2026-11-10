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
