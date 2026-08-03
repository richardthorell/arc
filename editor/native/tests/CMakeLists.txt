cmake_minimum_required(VERSION 3.20)
project(arc_editor_host_tests LANGUAGES CXX)

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
list(FILTER TEST_SOURCES EXCLUDE REGEX "game_module_fixture_.*\\.cpp$")

add_library(arc_test_game_module_v1 MODULE game_module_fixture_v1.cpp)
add_library(arc_test_game_module_v2 MODULE game_module_fixture_v2.cpp)
add_library(arc_test_game_module_v3 MODULE game_module_fixture_v3.cpp)
add_library(arc_test_game_module_v4 MODULE game_module_fixture_v4.cpp)
add_library(arc_test_game_module_rejected MODULE game_module_fixture_rejected.cpp)
foreach(module_target IN ITEMS arc_test_game_module_v1 arc_test_game_module_v2 arc_test_game_module_v3
                               arc_test_game_module_v4 arc_test_game_module_rejected)
  arc_configure_first_party_target(${module_target})
  target_link_libraries(${module_target} PRIVATE arc-project)
  set_target_properties(${module_target} PROPERTIES PREFIX "")
endforeach()

add_executable(arc_editor_host_tests ${TEST_SOURCES})
arc_configure_first_party_target(arc_editor_host_tests)
target_link_libraries(arc_editor_host_tests PRIVATE arc-editor-host Catch2::Catch2WithMain)
target_compile_definitions(arc_editor_host_tests PRIVATE "ARC_SOURCE_ROOT=\"${CMAKE_SOURCE_DIR}\"")
target_compile_definitions(arc_editor_host_tests PRIVATE
  "ARC_TEST_GAME_MODULE_V1=\"$<TARGET_FILE:arc_test_game_module_v1>\""
  "ARC_TEST_GAME_MODULE_V2=\"$<TARGET_FILE:arc_test_game_module_v2>\""
  "ARC_TEST_GAME_MODULE_V3=\"$<TARGET_FILE:arc_test_game_module_v3>\""
  "ARC_TEST_GAME_MODULE_V4=\"$<TARGET_FILE:arc_test_game_module_v4>\""
  "ARC_TEST_GAME_MODULE_REJECTED=\"$<TARGET_FILE:arc_test_game_module_rejected>\"")
add_dependencies(arc_editor_host_tests arc_test_game_module_v1 arc_test_game_module_v2 arc_test_game_module_v3
                 arc_test_game_module_v4 arc_test_game_module_rejected)
if(TARGET arc_host_process)
  add_dependencies(arc_editor_host_tests arc_host_process)
  target_compile_definitions(arc_editor_host_tests PRIVATE "ARC_HOST_PROCESS_PATH=\"$<TARGET_FILE:arc_host_process>\"")
endif()

include(Catch)
catch_discover_tests(arc_editor_host_tests)
