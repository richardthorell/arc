cmake_minimum_required(VERSION 3.20)
project(arc_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif()

option(ARC_BUILD_ENGINE_TESTS "Build ARC engine unit tests" ON)

if(WIN32)
    set(_arc_build_platform_windows_default ON)
else()
    set(_arc_build_platform_windows_default OFF)
endif()

option(ARC_BUILD_PLATFORM_WINDOWS "Build ARC Windows platform entry targets" ${_arc_build_platform_windows_default})

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/arc_module.cmake")

add_subdirectory(simd)
add_subdirectory(math)
add_subdirectory(geometric)
add_subdirectory(diagnostics)
add_subdirectory(jobs)
add_subdirectory(memory)
add_subdirectory(io)
add_subdirectory(framework)
add_subdirectory(input)
add_subdirectory(render)

if(ARC_BUILD_RENDER_VULKAN)
    add_subdirectory(render-vulkan)
endif()

add_subdirectory(scene)

add_library(arc INTERFACE)
target_link_libraries(arc
    INTERFACE
        arc-scene
        arc-input
)

if(NOT TARGET arc::arc)
    add_library(arc::arc ALIAS arc)
endif()

if(NOT TARGET arc::engine)
    add_library(arc::engine ALIAS arc)
endif()

if(ARC_BUILD_PLATFORM_WINDOWS)
    if(NOT WIN32)
        message(FATAL_ERROR "ARC_BUILD_PLATFORM_WINDOWS is ON, but the current platform is not Windows")
    endif()
    add_subdirectory(platform/windows)
endif()
