cmake_minimum_required(VERSION 3.22.1)
project(arc_android_host LANGUAGES C CXX)

if(NOT ANDROID)
    message(FATAL_ERROR "arc_android_host requires an Android NDK toolchain")
endif()

if(NOT DEFINED ANDROID_NDK)
    message(FATAL_ERROR "ANDROID_NDK must be provided by the Android Gradle Plugin")
endif()

set(ARC_NATIVE_APP_GLUE "${ANDROID_NDK}/sources/android/native_app_glue")
if(NOT EXISTS "${ARC_NATIVE_APP_GLUE}/android_native_app_glue.c")
    message(FATAL_ERROR "Android NDK native_app_glue not found: ${ARC_NATIVE_APP_GLUE}")
endif()

add_library(arc_native_app_glue STATIC "${ARC_NATIVE_APP_GLUE}/android_native_app_glue.c")
target_include_directories(arc_native_app_glue PUBLIC "${ARC_NATIVE_APP_GLUE}")
set_target_properties(arc_native_app_glue PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link the real ARC runtime bootstrap implementation, not an Android copy.
# Configure only the core dependency graph needed by the VFS and D1 boot.
get_filename_component(ARC_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../../.." ABSOLUTE)
if(NOT EXISTS "${ARC_REPO_ROOT}/engine/project/src/common/runtime_boot.cpp")
    message(FATAL_ERROR "Full ARC engine source is required for Android A3: ${ARC_REPO_ROOT}")
endif()

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ARC_BUILD_ENGINE_TESTS OFF CACHE BOOL "" FORCE)
set(ARC_BUILD_RENDER_VULKAN ON CACHE BOOL "" FORCE)
set(ARC_BUILD_ASSET_COOKER OFF CACHE BOOL "" FORCE)
set(ARC_BUILD_EDITOR OFF CACHE BOOL "" FORCE)
set(ARC_FETCH_STB OFF CACHE BOOL "" FORCE)
set(ARC_FETCH_UFBX OFF CACHE BOOL "" FORCE)
include("${ARC_REPO_ROOT}/cmake/ArcBuildPolicy.cmake")
include("${ARC_REPO_ROOT}/engine/cmake/arc_module.cmake")

add_subdirectory("${ARC_REPO_ROOT}/third_party" "${CMAKE_CURRENT_BINARY_DIR}/arc-third-party")
foreach(module IN ITEMS core simd math geometric diagnostics memory jobs io input ecs framework assets persistence project water render scene)
    add_subdirectory("${ARC_REPO_ROOT}/engine/${module}" "${CMAKE_CURRENT_BINARY_DIR}/arc-${module}")
endforeach()
add_subdirectory("${ARC_REPO_ROOT}/engine/render-vulkan" "${CMAKE_CURRENT_BINARY_DIR}/arc-render-vulkan")

add_library(arc_android_host SHARED
    android_host.cpp android_lifecycle.cpp android_log_sink.cpp
    android_package_assets.cpp android_package_assets_ndk.cpp
    android_apk_provider.cpp android_runtime_bridge.cpp
    android_vulkan_surface.cpp android_vulkan_presenter.cpp
    android_simd_compile_check.cpp)
target_compile_features(arc_android_host PRIVATE cxx_std_20)
target_compile_definitions(arc_android_host PRIVATE VK_USE_PLATFORM_ANDROID_KHR=1)
target_link_libraries(arc_android_host PRIVATE
    arc_native_app_glue arc-project arc-io arc-jobs arc-diagnostics arc-render-vulkan arc-scene android log)
target_compile_definitions(arc_android_host PRIVATE
    ARC_ANDROID_ENGINE_VERSION="0.1.0"
    ARC_ANDROID_SDK_VERSION="0.1.0"
    "$<$<CONFIG:Debug>:ARC_ANDROID_BUILD_CONFIGURATION=\"Debug\">"
    "$<$<NOT:$<CONFIG:Debug>>:ARC_ANDROID_BUILD_CONFIGURATION=\"Shipping\">")
target_link_options(arc_android_host PRIVATE "-Wl,-u,ANativeActivity_onCreate")
