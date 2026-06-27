cmake_minimum_required(VERSION 3.20)
project(arc_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif()

option(ARC_BUILD_ENGINE_TESTS "Build ARC engine unit tests" ON)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/arc_module.cmake")

add_subdirectory(simd)
add_subdirectory(math)

add_library(arc INTERFACE)
target_link_libraries(arc
    INTERFACE
        arc-math
)

if(NOT TARGET arc::arc)
    add_library(arc::arc ALIAS arc)
endif()

if(NOT TARGET arc::engine)
    add_library(arc::engine ALIAS arc)
endif()
