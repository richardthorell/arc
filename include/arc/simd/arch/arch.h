#pragma once

/**
 * @brief Map data types and sizes to SIMD register types.
 * 
 * @tparam T The base data type (e.g., float, int32_t).
 * @tparam N The number of lanes in the SIMD register.
 */
template <class T, std::size_t N>
struct simd_register;


/**
 * @brief Define SIMD operations for specific SIMD register types.
 * 
 * @tparam T The SIMD register type (e.g., __m128, __m256i).
 */
template <class T>
struct simd_op;


#if defined(__x86_64__) || defined(_M_X64)
    #include "arc/simd/arch/x64/registers.h"
    #include "arc/simd/arch/x64/ops.h"
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include "arc/simd/arch/neon/registers.h"
    #include "arc/simd/arch/neon/ops.h"
#else
    #error "Unsupported SIMD architecture"
#endif


/**
 * @brief Helper alias to get the SIMD register type for a given type and size.
 * 
 * @tparam T The base data type (e.g., float, int32_t).
 * @tparam N The number of lanes in the SIMD register.
 */
template <class T, std::size_t N>
using simd_register_t = typename simd_register<T, N>::type;
