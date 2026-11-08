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

# Build ARC diagnostics as a local library for this isolated Android fixture.
# The full engine-integrated host (A3) will link arc-diagnostics directly.
get_filename_component(ARC_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../../.." ABSOLUTE)
set(ARC_DIAGNOSTICS_ROOT "${ARC_REPO_ROOT}/engine/diagnostics")
if(NOT EXISTS "${ARC_DIAGNOSTICS_ROOT}/src/common/log.cpp")
    message(FATAL_ERROR "ARC diagnostics sources missing: ${ARC_DIAGNOSTICS_ROOT}")
endif()
add_library(arc_android_diagnostics STATIC "${ARC_DIAGNOSTICS_ROOT}/src/common/log.cpp")
target_include_directories(arc_android_diagnostics PUBLIC "${ARC_DIAGNOSTICS_ROOT}/inc")
target_compile_features(arc_android_diagnostics PUBLIC cxx_std_20)
set_target_properties(arc_android_diagnostics PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(arc_android_host SHARED android_host.cpp)
target_compile_features(arc_android_host PRIVATE cxx_std_20)
target_link_libraries(arc_android_host PRIVATE arc_native_app_glue arc_android_diagnostics android log)
target_link_options(arc_android_host PRIVATE "-Wl,-u,ANativeActivity_onCreate")
