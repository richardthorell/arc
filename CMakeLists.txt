cmake_minimum_required(VERSION 3.20)
project(arc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

option(ARC_BUILD_EDITOR "Build ARC editor targets" OFF)
option(ARC_BUILD_SAMPLES "Build ARC sample targets" OFF)
option(ARC_BUILD_HIGH_LEVEL_TESTS "Build ARC high-level integration tests" OFF)

add_subdirectory(engine)

if(ARC_BUILD_EDITOR)
    add_subdirectory(third_party)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/editor/CMakeLists.txt")
        add_subdirectory(editor)
    else()
        message(FATAL_ERROR "ARC_BUILD_EDITOR is ON, but editor/CMakeLists.txt does not exist yet")
    endif()
endif()

if(ARC_BUILD_SAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/samples/CMakeLists.txt")
        add_subdirectory(samples)
    else()
        message(FATAL_ERROR "ARC_BUILD_SAMPLES is ON, but samples/CMakeLists.txt does not exist yet")
    endif()
endif()

if(ARC_BUILD_HIGH_LEVEL_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        message(FATAL_ERROR "ARC_BUILD_HIGH_LEVEL_TESTS is ON, but tests/CMakeLists.txt does not exist yet")
    endif()
endif()
