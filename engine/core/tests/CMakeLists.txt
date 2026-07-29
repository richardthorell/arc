cmake_minimum_required(VERSION 3.20)
project(arc_core_tests LANGUAGES CXX)

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

add_executable(arc_core_tests core_tests.cpp)
arc_configure_first_party_target(arc_core_tests)
target_link_libraries(arc_core_tests PRIVATE arc-core Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(arc_core_tests)
