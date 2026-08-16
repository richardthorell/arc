cmake_minimum_required(VERSION 3.20)
project(arc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
include(CMakePackageConfigHelpers)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

option(ARC_BUILD_EDITOR "Build ARC editor targets" OFF)
option(ARC_BUILD_SAMPLES "Build ARC sample targets" OFF)
option(ARC_BUILD_HIGH_LEVEL_TESTS "Build ARC high-level integration tests" OFF)
option(ARC_BUILD_RENDER_VULKAN "Build ARC Vulkan render backend" OFF)
option(ARC_BUILD_ASSET_COOKER "Build the headless ARC asset cooker" ON)
option(ARC_BUILD_PROJECT_TOOL "Build the ARC external project command-line tool" ON)
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

if(ARC_BUILD_PROJECT_TOOL)
    add_subdirectory(tools/project_cli)
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

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ARCConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ARCConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/ARC"
    NO_SET_AND_CHECK_MACRO)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ARCConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY ExactVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ARCConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ARCConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ARCProjectReflection.cmake"
    DESTINATION "lib/cmake/ARC"
    COMPONENT sdk)
install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tools/project_reflection.py"
    DESTINATION tools
    COMPONENT sdk)
install(EXPORT ARCTargets
    FILE ARCTargets.cmake
    NAMESPACE ARC::
    DESTINATION "lib/cmake/ARC"
    COMPONENT sdk)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/templates/"
    DESTINATION "share/arc/templates"
    COMPONENT project-templates)

if(WIN32)
    if(ARC_BUILD_EDITOR)
        set(_arc_installation_editor "editor/arc-editor.exe")
    else()
        set(_arc_installation_editor "")
    endif()
    set(_arc_installation_platforms "\"windows-x64-vulkan\", \"windows-x64-headless\"")
else()
    set(_arc_installation_editor "")
    set(_arc_installation_platforms "\"linux-x64-headless\"")
endif()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/arc-installation.json" CONTENT
"{
  \"format\": \"arc-installation\",
  \"formatVersion\": 1,
  \"installationId\": \"arc-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}\",
  \"engineVersion\": \"${PROJECT_VERSION}\",
  \"editor\": \"${_arc_installation_editor}\",
  \"sdk\": \".\",
  \"cooker\": \"bin/arc-cook${CMAKE_EXECUTABLE_SUFFIX}\",
  \"projectTool\": \"bin/arc-project${CMAKE_EXECUTABLE_SUFFIX}\",
  \"supportedPlatforms\": [${_arc_installation_platforms}],
  \"configurations\": [\"Debug\", \"RelWithDebInfo\", \"Shipping\"],
  \"plugins\": [{\"id\":\"arc.renderer.vulkan\",\"version\":\"${PROJECT_VERSION}\",\"platforms\":[\"windows-x64-vulkan\"]}],
  \"templates\": [
    {\"id\":\"blank-3d\",\"name\":\"Blank 3D\",\"path\":\"share/arc/templates/blank-3d\"},
    {\"id\":\"blank-headless\",\"name\":\"Blank Headless\",\"path\":\"share/arc/templates/blank-headless\"},
    {\"id\":\"rendering-sample\",\"name\":\"Rendering Sample\",\"path\":\"share/arc/templates/rendering-sample\"},
    {\"id\":\"empty-cpp\",\"name\":\"Empty C++ Project\",\"path\":\"share/arc/templates/empty-cpp\"}
  ],
  \"toolchain\": {\"compiler\":\"auto\",\"minimumVersion\":\"\",\"generator\":\"auto\",\"architecture\":\"x86_64\",\"cppStandard\":20}
}
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/arc-installation.json"
    DESTINATION . COMPONENT manifests)

include(ArcPackaging)
