include_guard(GLOBAL)

if(NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

function(arc_generate_reflection)
    cmake_parse_arguments(ARG "" "TARGET;NAMESPACE;COMPONENT_NAMESPACE;OUTPUT_DIRECTORY" "HEADERS" ${ARGN})
    if(NOT ARG_TARGET OR NOT ARG_HEADERS)
        message(FATAL_ERROR "arc_generate_reflection requires TARGET and HEADERS")
    endif()
    if(NOT ARG_NAMESPACE)
        set(ARG_NAMESPACE arc_project_generated)
    endif()
    if(NOT ARG_OUTPUT_DIRECTORY)
        set(ARG_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    endif()
    set(_generated_header "${ARG_OUTPUT_DIRECTORY}/${ARG_TARGET}.reflection.h")
    set(_generated_json "${ARG_OUTPUT_DIRECTORY}/${ARG_TARGET}.schema.json")
    set(_header_args)
    foreach(_header IN LISTS ARG_HEADERS)
        list(APPEND _header_args --header "${_header}")
    endforeach()
    add_custom_command(OUTPUT "${_generated_header}" "${_generated_json}"
        COMMAND "${Python3_EXECUTABLE}" "${ARC_REFLECTION_GENERATOR}" ${_header_args}
            --cpp-output "${_generated_header}" --json-output "${_generated_json}" --namespace "${ARG_NAMESPACE}"
            --component-namespace "${ARG_COMPONENT_NAMESPACE}"
        DEPENDS ${ARG_HEADERS} "${ARC_REFLECTION_GENERATOR}"
        COMMENT "Generating ARC reflection for ${ARG_TARGET}" VERBATIM)
    target_sources(${ARG_TARGET} PRIVATE "${_generated_header}" "${_generated_json}")
    target_include_directories(${ARG_TARGET} PUBLIC "${ARG_OUTPUT_DIRECTORY}")
    set_property(TARGET ${ARG_TARGET} PROPERTY ARC_REFLECTION_HEADER "${_generated_header}")
    set_property(TARGET ${ARG_TARGET} PROPERTY ARC_REFLECTION_SCHEMA "${_generated_json}")
endfunction()
