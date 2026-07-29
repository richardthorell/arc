cmake_minimum_required(VERSION 3.20)
project(arc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

option(ARC_BUILD_EDITOR "Build ARC editor targets" OFF)
option(ARC_BUILD_SAMPLES "Build ARC sample targets" OFF)
option(ARC_BUILD_HIGH_LEVEL_TESTS "Build ARC high-level integration tests" OFF)
option(ARC_BUILD_RENDER_VULKAN "Build ARC Vulkan render backend" OFF)
option(ARC_BUILD_ASSET_COOKER "Build the headless ARC asset cooker" ON)
option(ARC_BUILD_BENCHMARKS "Build deterministic ARC subsystem benchmarks" OFF)
option(ARC_ENABLE_COVERAGE "Instrument Clang builds for LLVM source coverage" OFF)
set(ARC_SHIPPING_CONTENT_DIR "" CACHE PATH
    "Optional directory containing a verified cook manifest and package chunks to install")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ArcBuildPolicy)

file(GLOB_RECURSE ARC_BUILTIN_SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/*.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/*.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/include/*.glsl")
set(ARC_GLSLC_EXECUTABLE "glslc" CACHE FILEPATH "glslc used to validate built-in shaders")
set(ARC_SPIRV_VAL_EXECUTABLE "spirv-val" CACHE FILEPATH "spirv-val used to validate built-in shaders")
add_custom_target(arc-shaders-check
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/compile_builtin_shaders.py"
        --source-root "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders"
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders"
        --header "${CMAKE_CURRENT_SOURCE_DIR}/engine/render-vulkan/src/builtin_shaders.h"
        --glslc "${ARC_GLSLC_EXECUTABLE}"
        --spirv-val "${ARC_SPIRV_VAL_EXECUTABLE}"
        --check
    DEPENDS
        ${ARC_BUILTIN_SHADER_SOURCES}
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/compile_builtin_shaders.py"
    VERBATIM)

if(ARC_ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "ARC_ENABLE_COVERAGE requires Clang or AppleClang")
    endif()
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

add_subdirectory(third_party)

add_subdirectory(engine)

if(BUILD_TESTING)
    add_subdirectory(tests/public_api)
endif()

if(ARC_BUILD_ASSET_COOKER)
    add_subdirectory(tools/asset_cooker)
endif()

if(ARC_BUILD_EDITOR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/editor/CMakeLists.txt")
        add_subdirectory(editor)
    else()
        message(FATAL_ERROR "ARC_BUILD_EDITOR is ON, but editor/CMakeLists.txt does not exist")
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

if(ARC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

set(_arc_build_commit "$ENV{GITHUB_SHA}")
if(NOT _arc_build_commit)
    set(_arc_build_commit "local")
endif()
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build-manifest-$<CONFIG>.json"
    CONTENT
"{
  \"format\": \"arc-build-manifest\",
  \"version\": 1,
  \"commit\": \"${_arc_build_commit}\",
  \"configuration\": \"$<CONFIG>\",
  \"target\": \"${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}\",
  \"system\": \"${CMAKE_SYSTEM_NAME}\",
  \"architecture\": \"${CMAKE_SYSTEM_PROCESSOR}\",
  \"compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\",
  \"generator\": \"${CMAKE_GENERATOR}\"
}
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build-manifest-$<CONFIG>.json"
    DESTINATION .
    RENAME build-manifest.json
    COMPONENT manifests)
if(ARC_SHIPPING_CONTENT_DIR)
    if(NOT EXISTS "${ARC_SHIPPING_CONTENT_DIR}")
        message(FATAL_ERROR "ARC_SHIPPING_CONTENT_DIR does not exist: ${ARC_SHIPPING_CONTENT_DIR}")
    endif()
    install(DIRECTORY "${ARC_SHIPPING_CONTENT_DIR}/"
        DESTINATION content
        COMPONENT cooked-content
        FILES_MATCHING
            PATTERN "*.arccookmanifest"
            PATTERN "*.arcpak")
endif()

include(ArcPackaging)
