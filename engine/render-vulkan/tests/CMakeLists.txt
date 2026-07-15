cmake_minimum_required(VERSION 3.20)
project(arc_render_vulkan_tests LANGUAGES CXX)

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

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(arc_render_vulkan_tests ${TEST_SOURCES})
target_link_libraries(arc_render_vulkan_tests PRIVATE arc-render-vulkan Catch2::Catch2WithMain)
target_include_directories(arc_render_vulkan_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")

include(Catch)
catch_discover_tests(arc_render_vulkan_tests)
