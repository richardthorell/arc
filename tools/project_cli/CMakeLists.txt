add_executable(arc-project-cli main.cpp package_command.cpp workflow_command.cpp runtime_build_manifest.cpp)
arc_configure_first_party_target(arc-project-cli)
target_compile_features(arc-project-cli PRIVATE cxx_std_20)
target_link_libraries(arc-project-cli PRIVATE arc-project arc-third-party-json)
set_target_properties(arc-project-cli PROPERTIES FOLDER "Tools" OUTPUT_NAME "arc-project")

install(TARGETS arc-project-cli
    RUNTIME DESTINATION bin
    COMPONENT project-tools)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME arc-project-settings
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/project_settings_tests.py"
            $<TARGET_FILE:arc-project-cli> "${CMAKE_SOURCE_DIR}/templates" "${PROJECT_VERSION}")
    set_tests_properties(arc-project-settings PROPERTIES TIMEOUT 30)
    add_test(NAME arc-project-runtime-build-manifest
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/runtime_build_tests.py"
            $<TARGET_FILE:arc-project-cli> "${CMAKE_SOURCE_DIR}/templates" "${PROJECT_VERSION}"
            "${CMAKE_COMMAND}" "${CMAKE_SOURCE_DIR}/cmake/ARCRuntimeOutput.cmake")
    set_tests_properties(arc-project-runtime-build-manifest PROPERTIES TIMEOUT 120)
    add_test(NAME arc-project-workflow-plan
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/workflow_plan_tests.py"
            $<TARGET_FILE:arc-project-cli> "${CMAKE_SOURCE_DIR}/templates" "${PROJECT_VERSION}")
    set_tests_properties(arc-project-workflow-plan PROPERTIES TIMEOUT 30)
    add_test(NAME arc-project-template-smoke
        COMMAND "${CMAKE_COMMAND}"
            -DARC_PROJECT_TOOL=$<TARGET_FILE:arc-project-cli>
            -DARC_TEMPLATE_ROOT=${CMAKE_SOURCE_DIR}/templates
            -DARC_TEMPLATE_DESTINATION=${CMAKE_CURRENT_BINARY_DIR}/template-smoke
            -DARC_ENGINE_VERSION=${PROJECT_VERSION}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/template_smoke.cmake")
    set_tests_properties(arc-project-template-smoke PROPERTIES
        FIXTURES_SETUP arc_project_template
        TIMEOUT 30)
    add_test(NAME arc-project-template-validate
        COMMAND arc-project-cli validate
            --project "${CMAKE_CURRENT_BINARY_DIR}/template-smoke/ExternalGame.arcproject")
    set_tests_properties(arc-project-template-validate PROPERTIES
        FIXTURES_REQUIRED arc_project_template
        TIMEOUT 30)
endif()
