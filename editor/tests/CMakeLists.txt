cmake_minimum_required(VERSION 3.20)
project(arc_editor_tests LANGUAGES CXX)

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

add_executable(arc_editor_tests ${TEST_SOURCES})
target_link_libraries(arc_editor_tests PRIVATE arc-editor-core Catch2::Catch2WithMain)
if(TARGET arc_host_process)
  add_dependencies(arc_editor_tests arc_host_process)
  target_compile_definitions(arc_editor_tests PRIVATE "ARC_HOST_PROCESS_PATH=\"$<TARGET_FILE:arc_host_process>\"")
endif()

include(Catch)
catch_discover_tests(arc_editor_tests)
