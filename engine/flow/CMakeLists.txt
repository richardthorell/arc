arc_add_module(
    NAME arc-flow
    TYPE STATIC
    TEST_DEPS arc-project
)

target_link_libraries(arc-flow
    PRIVATE
        "$<BUILD_INTERFACE:arc-project>"
        "$<BUILD_INTERFACE:arc-third-party-json>"
)

if(BUILD_TESTING)
    add_executable(arc-flow-world-tests
        test_support/world_tests_main.cpp
        tests/flow_world_tests.cpp
    )
    arc_configure_first_party_target(arc-flow-world-tests)
    target_compile_features(arc-flow-world-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-flow-world-tests PRIVATE arc-flow arc-project)
    add_test(NAME arc-flow-world-tests COMMAND arc-flow-world-tests)

    add_executable(arc-flow-gameplay-compile-tests
        test_support/gameplay_compile_tests_main.cpp
        tests/flow_gameplay_compile_tests.cpp
    )
    arc_configure_first_party_target(arc-flow-gameplay-compile-tests)
    target_compile_features(arc-flow-gameplay-compile-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-flow-gameplay-compile-tests PRIVATE arc-flow arc-project)
    add_test(NAME arc-flow-gameplay-compile-tests COMMAND arc-flow-gameplay-compile-tests)
endif()
