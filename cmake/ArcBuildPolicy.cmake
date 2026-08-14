include_guard(GLOBAL)

option(ARC_WARNINGS_AS_ERRORS "Treat warnings in first-party ARC targets as errors" OFF)
option(ARC_ENABLE_ADDRESS_SANITIZER "Enable AddressSanitizer for first-party targets" OFF)
option(ARC_ENABLE_UNDEFINED_SANITIZER "Enable UndefinedBehaviorSanitizer for first-party targets" OFF)
option(ARC_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer for first-party targets" OFF)
option(ARC_ENABLE_SHIPPING_LTO "Enable link-time optimization for Shipping builds" ON)

if(ARC_ENABLE_THREAD_SANITIZER AND
   (ARC_ENABLE_ADDRESS_SANITIZER OR ARC_ENABLE_UNDEFINED_SANITIZER))
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer or UndefinedBehaviorSanitizer")
endif()

if((ARC_ENABLE_ADDRESS_SANITIZER OR ARC_ENABLE_UNDEFINED_SANITIZER OR
    ARC_ENABLE_THREAD_SANITIZER) AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "ARC sanitizer configurations require Clang or GCC")
endif()

if(CMAKE_CONFIGURATION_TYPES)
    list(FIND CMAKE_CONFIGURATION_TYPES Shipping _arc_shipping_index)
    if(_arc_shipping_index EQUAL -1)
        list(APPEND CMAKE_CONFIGURATION_TYPES Shipping)
        set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
            "Available build configurations" FORCE)
    endif()
endif()

if(MSVC)
    set(_arc_shipping_cxx_flags "${CMAKE_CXX_FLAGS_RELEASE}")
    set(_arc_shipping_exe_linker_flags
        "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(_arc_shipping_shared_linker_flags
        "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    if(ARC_ENABLE_SHIPPING_LTO)
        string(APPEND _arc_shipping_cxx_flags " /GL")
        string(APPEND _arc_shipping_exe_linker_flags " /LTCG")
        string(APPEND _arc_shipping_shared_linker_flags " /LTCG")
    endif()
    set(CMAKE_CXX_FLAGS_SHIPPING "${_arc_shipping_cxx_flags}" CACHE STRING
        "C++ flags used by Shipping builds" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_SHIPPING "${_arc_shipping_exe_linker_flags}"
        CACHE STRING "Executable linker flags used by Shipping builds" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_SHIPPING "${_arc_shipping_shared_linker_flags}"
        CACHE STRING "Shared-library linker flags used by Shipping builds" FORCE)
else()
    set(_arc_shipping_cxx_flags "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(_arc_shipping_exe_linker_flags "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    set(_arc_shipping_shared_linker_flags "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    if(ARC_ENABLE_SHIPPING_LTO)
        string(APPEND _arc_shipping_cxx_flags " -flto")
        string(APPEND _arc_shipping_exe_linker_flags " -flto")
        string(APPEND _arc_shipping_shared_linker_flags " -flto")
    endif()
    set(CMAKE_CXX_FLAGS_SHIPPING "${_arc_shipping_cxx_flags}"
        CACHE STRING "C++ flags used by Shipping builds" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_SHIPPING "${_arc_shipping_exe_linker_flags}"
        CACHE STRING "Executable linker flags used by Shipping builds" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_SHIPPING "${_arc_shipping_shared_linker_flags}"
        CACHE STRING "Shared-library linker flags used by Shipping builds" FORCE)
endif()
unset(_arc_shipping_cxx_flags)
unset(_arc_shipping_exe_linker_flags)
unset(_arc_shipping_shared_linker_flags)

if(CMAKE_BUILD_TYPE STREQUAL "Shipping")
    set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
    set(ARC_BUILD_EDITOR OFF CACHE BOOL "Build ARC editor targets" FORCE)
    set(ARC_BUILD_ASSET_COOKER OFF CACHE BOOL "Build the headless ARC asset cooker" FORCE)
endif()

add_library(arc-build-config INTERFACE)
add_library(arc::build-config ALIAS arc-build-config)
set_target_properties(arc-build-config PROPERTIES EXPORT_NAME BuildConfig)
install(TARGETS arc-build-config EXPORT ARCTargets COMPONENT sdk)

target_compile_definitions(arc-build-config INTERFACE
    "$<$<CONFIG:Debug>:ARC_BUILD_DEBUG=1;ARC_BUILD_DEVELOPMENT=1>"
    "$<$<CONFIG:RelWithDebInfo>:ARC_BUILD_DEVELOPMENT=1>"
    "$<$<CONFIG:Shipping>:ARC_BUILD_SHIPPING=1>")

set(_arc_sanitizer_compile_options)
set(_arc_sanitizer_link_options)
if(ARC_ENABLE_ADDRESS_SANITIZER)
    list(APPEND _arc_sanitizer_compile_options -fsanitize=address -fno-omit-frame-pointer)
    list(APPEND _arc_sanitizer_link_options -fsanitize=address)
endif()
if(ARC_ENABLE_UNDEFINED_SANITIZER)
    list(APPEND _arc_sanitizer_compile_options -fsanitize=undefined -fno-omit-frame-pointer)
    list(APPEND _arc_sanitizer_link_options -fsanitize=undefined)
endif()
if(ARC_ENABLE_THREAD_SANITIZER)
    list(APPEND _arc_sanitizer_compile_options -fsanitize=thread -fno-omit-frame-pointer)
    list(APPEND _arc_sanitizer_link_options -fsanitize=thread)
endif()
target_compile_options(arc-build-config INTERFACE ${_arc_sanitizer_compile_options})
target_link_options(arc-build-config INTERFACE ${_arc_sanitizer_link_options})
if(MSVC)
    target_compile_options(arc-build-config INTERFACE
        "$<BUILD_INTERFACE:$<$<CONFIG:Shipping>:/Brepro;/experimental:deterministic;/pathmap:${CMAKE_SOURCE_DIR}=.>>"
        "$<INSTALL_INTERFACE:$<$<CONFIG:Shipping>:/Brepro;/experimental:deterministic>>")
    target_link_options(arc-build-config INTERFACE
        "$<$<CONFIG:Shipping>:/Brepro;/PDBALTPATH:%_PDB%>")
else()
    target_compile_options(arc-build-config INTERFACE
        "$<BUILD_INTERFACE:$<$<CONFIG:Shipping>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.;-fdebug-prefix-map=${CMAKE_SOURCE_DIR}=.>>")
endif()

function(arc_configure_first_party_target target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Cannot configure missing first-party target '${target}'")
    endif()

    get_target_property(_arc_target_type "${target}" TYPE)
    if(_arc_target_type STREQUAL "INTERFACE_LIBRARY")
        target_link_libraries("${target}" INTERFACE arc-build-config)
        return()
    endif()

    if(_arc_target_type STREQUAL "STATIC_LIBRARY" OR _arc_target_type STREQUAL "OBJECT_LIBRARY")
        # First-party static/object libraries can be linked into shared modules
        # such as arc-render-vulkan, so keep their objects relocatable on ELF platforms.
        set_target_properties("${target}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    if(_arc_target_type STREQUAL "EXECUTABLE")
        target_link_libraries("${target}" PRIVATE arc-build-config)
    else()
        target_link_libraries("${target}" PUBLIC arc-build-config)
    endif()

    if(MSVC)
        target_compile_options("${target}" PRIVATE /W4 /permissive-)
        if(ARC_WARNINGS_AS_ERRORS)
            target_compile_options("${target}" PRIVATE /WX)
        endif()
    else()
        # ARC descriptors intentionally use partial designated initialization so
        # omitted fields retain their declared defaults. GCC and Clang diagnose
        # that valid pattern through -Wmissing-field-initializers, making the
        # warning incompatible with the public descriptor style.
        target_compile_options("${target}" PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
        if(ARC_WARNINGS_AS_ERRORS)
            target_compile_options("${target}" PRIVATE -Werror)
        endif()
    endif()
endfunction()
