arc_add_module(
    NAME arc-simd
    TYPE INTERFACE
    NO_TESTS
)

option(ARC_SIMD_ENABLE_NATIVE "Enable native host SIMD code generation for arc-simd" OFF)

# Optional: enable host-native SIMD instruction sets for local performance builds.
# Keep this opt-in so CI and distributed builds do not depend on the runner CPU.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(arc-simd INTERFACE -Wno-ignored-attributes)

    if(ARC_SIMD_ENABLE_NATIVE)
        target_compile_options(arc-simd INTERFACE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(arc-simd INTERFACE /arch:AVX2)
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
