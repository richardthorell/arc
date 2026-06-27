cmake_minimum_required(VERSION 3.20)
project(arc_simd_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
include(FetchContent)

if(NOT TARGET Catch2::Catch2WithMain)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.6.0
  )
  FetchContent_MakeAvailable(Catch2)
endif()

FetchContent_GetProperties(Catch2 SOURCE_DIR catch2_SOURCE_DIR)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")

if(NOT TARGET arc-simd)
  set(ARC_BUILD_ENGINE_TESTS OFF CACHE BOOL "Build ARC engine unit tests" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_BINARY_DIR}/arc EXCLUDE_FROM_ALL)
endif()

# Collect all test source files automatically
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Define test executable
add_executable(arc_simd_tests ${TEST_SOURCES})
target_link_libraries(arc_simd_tests PRIVATE arc-simd Catch2::Catch2WithMain)

# Enable test discovery
include(CTest)
include(Catch)
catch_discover_tests(arc_simd_tests)
