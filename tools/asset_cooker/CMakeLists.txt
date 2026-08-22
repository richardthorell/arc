add_executable(arc-cook
    main.cpp
)
arc_configure_first_party_target(arc-cook)

target_compile_features(arc-cook PRIVATE cxx_std_20)
target_link_libraries(arc-cook
    PRIVATE
        arc-assets
        arc-persistence
        arc-project
        arc-render
        arc-render-tools
        arc-scene
        arc-third-party-json
)

set_target_properties(arc-cook PROPERTIES FOLDER "Tools")
install(TARGETS arc-cook
    RUNTIME DESTINATION bin
    COMPONENT tools)

if(BUILD_TESTING)
    add_test(
        NAME arc-cook-clean-checkout
        COMMAND "${CMAKE_COMMAND}"
            -DARC_COOK=$<TARGET_FILE:arc-cook>
            -DARC_SOURCE_ROOT=${CMAKE_SOURCE_DIR}
            -DARC_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/clean-checkout
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/clean_checkout.cmake
    )
    set_tests_properties(arc-cook-clean-checkout PROPERTIES
        LABELS "integration;assets;persistence"
        TIMEOUT 600
    )
endif()
