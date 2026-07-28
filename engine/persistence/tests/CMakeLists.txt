cmake_minimum_required(VERSION 3.20)
project(arc_persistence_tests LANGUAGES CXX)

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

add_executable(arc_persistence_tests persistence_tests.cpp)
target_link_libraries(arc_persistence_tests PRIVATE arc-persistence Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(arc_persistence_tests)
