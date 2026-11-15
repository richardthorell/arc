# Native Android module: project-owned code compiled against ARC's prebuilt
# Android host with the same runtime-module reflection and lifecycle ABI.
if(NOT ANDROID OR NOT ARC_ANDROID_PROJECT_BUILD OR NOT ARC_ANDROID_SDK_ROOT)
    message(FATAL_ERROR "This project Android module must be invoked by ARC's Android NDK launcher")
endif()

set(_sdk "${ARC_ANDROID_SDK_ROOT}")
set(_host_variant debug)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_host_variant release)
endif()
set(_jni_root "${_sdk}/share/arc/android/host/${_host_variant}/jniLibs/arm64-v8a")
set(_native_host "${_jni_root}/libarc_android_host.so")
if(NOT EXISTS "${_native_host}")
    message(FATAL_ERROR "Installed ARC SDK does not contain the selected Android native host: ${_native_host}")
endif()
if(NOT EXISTS "${_sdk}/include/arc/project/project_module.h")
    message(FATAL_ERROR "Installed ARC SDK lacks project module headers")
endif()

add_library(arc_android_imported_host SHARED IMPORTED GLOBAL)
set_target_properties(arc_android_imported_host PROPERTIES
    IMPORTED_LOCATION "${_native_host}"
    INTERFACE_INCLUDE_DIRECTORIES "${_sdk}/include")
# Export the other A1 host JNI dependencies too. The Android host can depend on
# render backends loaded from this same native-library namespace.
file(GLOB _host_dependencies "${_jni_root}/lib*.so")
foreach(_path IN LISTS _host_dependencies)
    if(_path STREQUAL "${_native_host}")
        continue()
    endif()
    get_filename_component(_name "${_path}" NAME_WE)
    string(MAKE_C_IDENTIFIER "${_name}" _target_suffix)
    set(_target "arc_android_imported_${_target_suffix}")
    add_library("${_target}" SHARED IMPORTED GLOBAL)
    set_target_properties("${_target}" PROPERTIES IMPORTED_LOCATION "${_path}")
    list(APPEND _arc_android_dependencies "${_target}")
endforeach()

add_library({{PROJECT_TOKEN}}Runtime STATIC
    Source/{{PROJECT_TOKEN}}Runtime/Runtime.cpp)
set_target_properties({{PROJECT_TOKEN}}Runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features({{PROJECT_TOKEN}}Runtime PUBLIC cxx_std_20)
target_include_directories({{PROJECT_TOKEN}}Runtime PUBLIC "${_sdk}/include")
arc_generate_reflection(
    TARGET {{PROJECT_TOKEN}}Runtime
    NAMESPACE {{PROJECT_TOKEN}}::generated
    COMPONENT_NAMESPACE {{PROJECT_TOKEN}}
    HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/Source/{{PROJECT_TOKEN}}Runtime/Components.h")

add_library(arc_android_project_runtime SHARED
    Source/{{PROJECT_TOKEN}}Runtime/AndroidModule.cpp)
target_compile_features(arc_android_project_runtime PRIVATE cxx_std_20)
target_link_libraries(arc_android_project_runtime PRIVATE
    {{PROJECT_TOKEN}}Runtime arc_android_imported_host ${_arc_android_dependencies} android log)
set_target_properties(arc_android_project_runtime PROPERTIES OUTPUT_NAME arc_project_runtime)
