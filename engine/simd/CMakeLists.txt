arc_add_module(
    NAME arc-simd
    TYPE INTERFACE
    NO_TESTS
)

# Optional: enable SIMD instruction sets
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(arc-simd INTERFACE -march=native -Wno-ignored-attributes)
elseif(MSVC)
    target_compile_options(arc-simd INTERFACE /arch:AVX2)
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
