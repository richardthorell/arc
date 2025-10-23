cmake_minimum_required(VERSION 3.20)
project(arc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only library
add_library(arc INTERFACE)
target_include_directories(arc INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optional: enable SIMD instruction sets
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(arc INTERFACE -march=native)
elseif (MSVC)
    target_compile_options(arc INTERFACE /arch:AVX2)
endif()
