cmake_minimum_required(VERSION 3.20)

include(FetchContent)

option(ARC_USE_SYSTEM_TINYEXR "Use a system-provided TinyEXR header" OFF)
option(ARC_FETCH_TINYEXR "Fetch TinyEXR when ARC_USE_SYSTEM_TINYEXR is OFF" ON)

add_library(arc-third-party-tinyexr INTERFACE)
if(NOT TARGET arc::third-party-tinyexr)
    add_library(arc::third-party-tinyexr ALIAS arc-third-party-tinyexr)
endif()

if(NOT TARGET arc-third-party-stb)
    message(FATAL_ERROR "ARC TinyEXR integration requires the existing stb dependency")
endif()

if(ARC_USE_SYSTEM_TINYEXR)
    find_path(ARC_TINYEXR_INCLUDE_DIR tinyexr.h REQUIRED)
    target_include_directories(arc-third-party-tinyexr SYSTEM INTERFACE "${ARC_TINYEXR_INCLUDE_DIR}")
    target_compile_definitions(arc-third-party-tinyexr INTERFACE ARC_RENDER_HAS_TINYEXR=1)
elseif(ARC_FETCH_TINYEXR AND ARC_FETCH_THIRD_PARTY)
    FetchContent_Declare(
        tinyexr
        GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
        GIT_TAG v1.0.13
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR arc-no-cmake
    )
    FetchContent_MakeAvailable(tinyexr)
    target_include_directories(arc-third-party-tinyexr SYSTEM INTERFACE "${tinyexr_SOURCE_DIR}")
    target_compile_definitions(arc-third-party-tinyexr INTERFACE ARC_RENDER_HAS_TINYEXR=1)
else()
    target_compile_definitions(arc-third-party-tinyexr INTERFACE ARC_RENDER_TINYEXR_UNAVAILABLE=1)
endif()

target_link_libraries(arc-third-party-tinyexr INTERFACE arc-third-party-stb)
