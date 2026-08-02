file(REMOVE_RECURSE "${ARC_TEMPLATE_DESTINATION}")
execute_process(
    COMMAND "${ARC_PROJECT_TOOL}" create
        --name ExternalGame
        --destination "${ARC_TEMPLATE_DESTINATION}"
        --template blank-headless
        --templates "${ARC_TEMPLATE_ROOT}"
        --engine "${ARC_ENGINE_VERSION}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "arc-project failed to generate the template fixture")
endif()
