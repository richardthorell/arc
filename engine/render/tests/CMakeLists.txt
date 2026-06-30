cmake_minimum_required(VERSION 3.20)
project(arc_render_tests LANGUAGES CXX)

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

if(NOT TARGET arc-render)
  set(ARC_BUILD_ENGINE_TESTS OFF CACHE BOOL "Build ARC engine unit tests" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_BINARY_DIR}/arc EXCLUDE_FROM_ALL)
endif()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(arc_render_tests ${TEST_SOURCES})
target_link_libraries(arc_render_tests PRIVATE arc-render Catch2::Catch2WithMain)
target_compile_definitions(arc_render_tests
  PRIVATE
    "ARC_RENDER_TEST_ASSET_ROOT=\"${CMAKE_SOURCE_DIR}/assets\""
)
if(ARC_THIRD_PARTY_UFBX_AVAILABLE)
  FetchContent_GetProperties(ufbx SOURCE_DIR ufbx_SOURCE_DIR)
endif()
if(ARC_THIRD_PARTY_UFBX_AVAILABLE AND DEFINED ufbx_SOURCE_DIR AND EXISTS "${ufbx_SOURCE_DIR}/data")
  target_compile_definitions(arc_render_tests
    PRIVATE
      "ARC_RENDER_TEST_UFBX_DATA_ROOT=\"${ufbx_SOURCE_DIR}/data\""
  )
endif()

include(Catch)
catch_discover_tests(arc_render_tests)
