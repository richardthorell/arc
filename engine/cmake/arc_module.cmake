include_guard(GLOBAL)

include(CMakeParseArguments)

function(arc_collect_module_public_headers OUT_VAR MODULE_DIR)
    set(_headers)
    set(_inc_dir "${MODULE_DIR}/inc")

    if(EXISTS "${_inc_dir}")
        file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
            "${_inc_dir}/*.h"
            "${_inc_dir}/*.hh"
            "${_inc_dir}/*.hpp"
            "${_inc_dir}/*.hxx"
            "${_inc_dir}/*.inl"
        )
    endif()

    set(${OUT_VAR} "${_headers}" PARENT_SCOPE)
endfunction()

function(arc_collect_module_private_headers OUT_VAR MODULE_DIR)
    set(_headers)
    set(_src_dir "${MODULE_DIR}/src")

    if(EXISTS "${_src_dir}")
        file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
            "${_src_dir}/*.h"
            "${_src_dir}/*.hh"
            "${_src_dir}/*.hpp"
            "${_src_dir}/*.hxx"
            "${_src_dir}/*.inl"
            "${_src_dir}/*.ipp"
            "${_src_dir}/*.tpp"
        )
    endif()

    set(${OUT_VAR} "${_headers}" PARENT_SCOPE)
endfunction()

function(arc_collect_module_source_dirs OUT_VAR MODULE_DIR)
    set(_dirs
        "${MODULE_DIR}/src/common"
    )

    #
    # Console / special platforms first where useful.
    #
    if(CMAKE_SYSTEM_NAME STREQUAL "Xbox" OR CMAKE_SYSTEM_NAME STREQUAL "Durango" OR CMAKE_SYSTEM_NAME STREQUAL "Scarlett")
        list(APPEND _dirs
            "${MODULE_DIR}/src/msft"
            "${MODULE_DIR}/src/xbox"
        )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "PS5")
        list(APPEND _dirs
            "${MODULE_DIR}/src/sony"
            "${MODULE_DIR}/src/ps5"
        )
    else()
        #
        # Mainstream platforms
        #
        if(WIN32)
            list(APPEND _dirs
                "${MODULE_DIR}/src/msft"
                "${MODULE_DIR}/src/windows"
            )
        endif()

        if(ANDROID)
            list(APPEND _dirs
                "${MODULE_DIR}/src/posix"
                "${MODULE_DIR}/src/android"
            )
        endif()

        if(APPLE)
            list(APPEND _dirs
                "${MODULE_DIR}/src/apple"
            )

            #
            # CMake Apple platform detection varies by generator/toolchain.
            # These checks cover common cases.
            #
            if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "tvOS" OR CMAKE_OSX_SYSROOT MATCHES "iphone")
                list(APPEND _dirs
                    "${MODULE_DIR}/src/ios"
                )
            elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
                list(APPEND _dirs
                    "${MODULE_DIR}/src/macos"
                )
            endif()
        endif()

        if(UNIX AND NOT APPLE AND NOT ANDROID)
            list(APPEND _dirs
                "${MODULE_DIR}/src/posix"
                "${MODULE_DIR}/src/linux"
            )
        endif()
    endif()

    set(${OUT_VAR} "${_dirs}" PARENT_SCOPE)
endfunction()

function(arc_collect_module_sources OUT_VAR MODULE_DIR)
    arc_collect_module_source_dirs(_candidate_dirs "${MODULE_DIR}")

    set(_sources)

    foreach(_dir IN LISTS _candidate_dirs)
        if(EXISTS "${_dir}")
            file(GLOB_RECURSE _dir_sources CONFIGURE_DEPENDS
                "${_dir}/*.c"
                "${_dir}/*.cc"
                "${_dir}/*.cpp"
                "${_dir}/*.cxx"
                "${_dir}/*.m"
                "${_dir}/*.mm"
                "${_dir}/*.ixx"
                "${_dir}/*.cppm"
            )
            list(APPEND _sources ${_dir_sources})
        endif()
    endforeach()

    set(${OUT_VAR} "${_sources}" PARENT_SCOPE)
endfunction()

function(arc_add_module)
    set(options NO_TESTS)
    set(oneValueArgs NAME TYPE)
    set(multiValueArgs DEPS TEST_DEPS)
    cmake_parse_arguments(ARC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARC_NAME)
        message(FATAL_ERROR "arc_add_module requires NAME")
    endif()

    if(NOT ARC_TYPE)
        message(FATAL_ERROR "arc_add_module requires TYPE")
    endif()

    set(_module_dir "${CMAKE_CURRENT_SOURCE_DIR}")

    arc_collect_module_public_headers(_public_headers "${_module_dir}")
    arc_collect_module_private_headers(_private_headers "${_module_dir}")
    arc_collect_module_sources(_sources "${_module_dir}")

    if(ARC_TYPE STREQUAL "INTERFACE")
        add_library(${ARC_NAME} INTERFACE)

        target_include_directories(${ARC_NAME}
            INTERFACE
                "${_module_dir}/inc"
        )

        target_compile_features(${ARC_NAME}
            INTERFACE
                cxx_std_20
        )

        if(ARC_DEPS)
            target_link_libraries(${ARC_NAME}
                INTERFACE
                    ${ARC_DEPS}
            )
        endif()

        #
        # Interface libraries cannot own source files in the same way as compiled
        # targets, but these help IDE organization in some generators.
        #
        target_sources(${ARC_NAME}
            INTERFACE
                ${_public_headers}
                ${_private_headers}
        )
    else()
        add_library(${ARC_NAME} ${ARC_TYPE}
            ${_public_headers}
            ${_private_headers}
            ${_sources}
        )

        target_include_directories(${ARC_NAME}
            PUBLIC
                "${_module_dir}/inc"
            PRIVATE
                "${_module_dir}/src"
        )

        target_compile_features(${ARC_NAME}
            PUBLIC
                cxx_std_20
        )

        if(ARC_DEPS)
            target_link_libraries(${ARC_NAME}
                PUBLIC
                    ${ARC_DEPS}
            )
        endif()
    endif()

    #
    # Nice IDE folder layout
    #
    source_group(TREE "${_module_dir}" FILES
        ${_public_headers}
        ${_private_headers}
        ${_sources}
    )

    #
    # Optional namespaced alias: arc::geometry for arc-geometry, etc.
    #
    string(REPLACE "arc-" "" _alias_suffix "${ARC_NAME}")
    if(NOT TARGET "arc::${_alias_suffix}")
        add_library("arc::${_alias_suffix}" ALIAS ${ARC_NAME})
    endif()

    if(BUILD_TESTING AND NOT ARC_NO_TESTS)
        set(_tests_dir "${_module_dir}/tests")

        if(EXISTS "${_tests_dir}")
            file(GLOB_RECURSE _test_sources CONFIGURE_DEPENDS
                "${_tests_dir}/*.c"
                "${_tests_dir}/*.cc"
                "${_tests_dir}/*.cpp"
                "${_tests_dir}/*.cxx"
                "${_tests_dir}/*.m"
                "${_tests_dir}/*.mm"
            )

            if(_test_sources)
                set(_test_target "${ARC_NAME}-tests")

                add_executable(${_test_target}
                    ${_test_sources}
                )

                target_compile_features(${_test_target}
                    PRIVATE
                        cxx_std_20
                )

                target_include_directories(${_test_target}
                    PRIVATE
                        "${_module_dir}/src"
                )

                target_link_libraries(${_test_target}
                    PRIVATE
                        ${ARC_NAME}
                        ${ARC_TEST_DEPS}
                )

                add_test(NAME ${_test_target} COMMAND ${_test_target})

                source_group(TREE "${_module_dir}" FILES ${_test_sources})
            endif()
        endif()
    endif()
endfunction()