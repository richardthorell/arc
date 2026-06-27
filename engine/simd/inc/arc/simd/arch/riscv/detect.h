#pragma once

#if defined(__riscv)
    #define ARC_SIMD_RISCV
#endif

#if defined(__riscv_vector)
    #define ARC_SIMD_RISCV_VECTOR
#endif
