add_executable(arc-project-cli main.cpp)
arc_configure_first_party_target(arc-project-cli)
target_compile_features(arc-project-cli PRIVATE cxx_std_20)
target_link_libraries(arc-project-cli PRIVATE arc-project arc-third-party-json)
set_target_properties(arc-project-cli PROPERTIES FOLDER "Tools" OUTPUT_NAME "arc-project")

install(TARGETS arc-project-cli
    RUNTIME DESTINATION bin
    COMPONENT project-tools)

if(BUILD_TESTING)
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
