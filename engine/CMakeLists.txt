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

add_subdirectory(core)
add_subdirectory(simd)
add_subdirectory(math)
add_subdirectory(geometric)
add_subdirectory(diagnostics)
add_subdirectory(jobs)
add_subdirectory(memory)
add_subdirectory(io)
add_subdirectory(framework)
add_subdirectory(assets)
add_subdirectory(input)
add_subdirectory(ecs)
add_subdirectory(persistence)
add_subdirectory(project)
add_subdirectory(render)

if(ARC_BUILD_ASSET_COOKER OR ARC_BUILD_EDITOR)
    add_subdirectory(render-tools)
endif()

if(ARC_BUILD_RENDER_VULKAN)
    add_subdirectory(render-vulkan)
endif()

add_subdirectory(scene)

add_library(arc INTERFACE)
target_link_libraries(arc
    INTERFACE
        arc-scene
        arc-assets
        arc-persistence
        arc-project
        arc-input
)

if(NOT TARGET arc::arc)
    add_library(arc::arc ALIAS arc)
endif()

set_target_properties(arc PROPERTIES EXPORT_NAME Engine)
install(TARGETS arc EXPORT ARCTargets COMPONENT sdk)

add_library(arc-runtime INTERFACE)
target_link_libraries(arc-runtime INTERFACE
    arc-framework arc-assets arc-persistence arc-input arc-ecs)
set_target_properties(arc-runtime PROPERTIES EXPORT_NAME Runtime)
install(TARGETS arc-runtime EXPORT ARCTargets COMPONENT sdk)
add_library(arc::runtime ALIAS arc-runtime)

add_library(arc-server INTERFACE)
target_link_libraries(arc-server INTERFACE
    arc-framework arc-assets arc-persistence arc-ecs)
set_target_properties(arc-server PROPERTIES EXPORT_NAME Server)
install(TARGETS arc-server EXPORT ARCTargets COMPONENT sdk)
add_library(arc::server ALIAS arc-server)

add_library(arc-editor-module-sdk INTERFACE)
target_link_libraries(arc-editor-module-sdk INTERFACE arc-project arc-runtime)
set_target_properties(arc-editor-module-sdk PROPERTIES EXPORT_NAME EditorModuleSDK)
install(TARGETS arc-editor-module-sdk EXPORT ARCTargets COMPONENT sdk)
add_library(arc::editor-module-sdk ALIAS arc-editor-module-sdk)

if(NOT TARGET arc::engine)
    add_library(arc::engine ALIAS arc)
endif()

if(ARC_BUILD_PLATFORM_WINDOWS)
    if(NOT WIN32)
        message(FATAL_ERROR "ARC_BUILD_PLATFORM_WINDOWS is ON, but the current platform is not Windows")
    endif()
    add_subdirectory(platform/windows)
endif()
