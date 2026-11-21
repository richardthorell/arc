set(_arc_dev_root "${CMAKE_BINARY_DIR}/development-sdk")

# Interface libraries have no build rule. Their concrete link dependencies
# provide the SDK closure without building tests, benchmarks or samples.
add_custom_target(arc-development-sdk
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${_arc_dev_root}/$<CONFIG>" --component sdk
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${_arc_dev_root}/$<CONFIG>" --component sdk-development
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${_arc_dev_root}/$<CONFIG>" --component sdk-headers
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${_arc_dev_root}/$<CONFIG>" --component sdk-private
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${_arc_dev_root}/$<CONFIG>" --component project-tools
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${_arc_dev_root}/$<CONFIG>" --component tools
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${_arc_dev_root}/$<CONFIG>" --component project-templates
    VERBATIM)
foreach(_interface arc arc-runtime arc-server arc-editor-module-sdk)
    get_target_property(_dependencies ${_interface} INTERFACE_LINK_LIBRARIES)
    foreach(_dependency IN LISTS _dependencies)
        if(TARGET "${_dependency}" AND NOT _dependency STREQUAL "arc-runtime")
            add_dependencies(arc-development-sdk ${_dependency})
        endif()
    endforeach()
endforeach()
# Every exported SDK target must have its archive present when CMake imports
# ARCTargets.cmake. Live Play is optional, but configured development archives
# must be built and staged with the rest of the SDK.
foreach(_target arc-project-cli arc-cook arc-platform-windows arc-platform-windows-packaged-entry
                arc-platform-windows-live-play-entry arc-live-play-dev arc-render-vulkan)
    if(TARGET ${_target})
        add_dependencies(arc-development-sdk ${_target})
    endif()
endforeach()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arc_dev_arch)
if(CMAKE_GENERATOR_PLATFORM MATCHES "^[Aa][Rr][Mm]64" OR _arc_dev_arch MATCHES "^(aarch64|arm64)$")
    set(_arc_dev_arch arm64)
elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64" OR _arc_dev_arch MATCHES "^(amd64|x86_64)$")
    set(_arc_dev_arch x86_64)
endif()
file(GENERATE OUTPUT "${_arc_dev_root}/$<CONFIG>/arc-development-sdk.json" CONTENT
    "{\"version\":1,\"engineVersion\":\"${PROJECT_VERSION}\",\"architecture\":\"${_arc_dev_arch}\",\"buildDirectory\":\"${CMAKE_BINARY_DIR}\",\"configuration\":\"$<CONFIG>\",\"projectTool\":\"$<TARGET_FILE:arc-project-cli>\"}\n")

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_arc_dev_templates blank-headless)
    if(ARC_BUILD_RENDER_VULKAN)
        list(APPEND _arc_dev_templates blank-3d)
    endif()
    add_test(NAME arc-development-sdk-projects
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/project_cli/development_sdk_smoke.py"
            "$<TARGET_FILE:arc-project-cli>" "${CMAKE_SOURCE_DIR}/templates" "${PROJECT_VERSION}"
            "${_arc_dev_root}" "$<CONFIG>" "${CMAKE_COMMAND}" ${_arc_dev_templates})
    set_tests_properties(arc-development-sdk-projects PROPERTIES TIMEOUT 3600 RUN_SERIAL TRUE)
endif()
