cmake_minimum_required(VERSION 3.20)
project(arc_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch Catch2 (header-only)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(Catch2)

# Include header-only library from one level up
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_BINARY_DIR}/arc EXCLUDE_FROM_ALL)

# Collect all test source files automatically
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Define test executable
add_executable(arc_tests ${TEST_SOURCES})
target_link_libraries(arc_tests PRIVATE arc Catch2::Catch2WithMain)

# Enable test discovery
include(CTest)
include(Catch)
catch_discover_tests(arc_tests)
