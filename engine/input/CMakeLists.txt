arc_add_module(
    NAME arc-input
    TYPE STATIC
    DEPS arc-math
)

if(BUILD_TESTING)
    add_executable(arc-input-rumble-tests
        test_support/rumble_tests.cpp
    )
    arc_configure_first_party_target(arc-input-rumble-tests)

    target_compile_features(arc-input-rumble-tests
        PRIVATE
            cxx_std_20
    )

    target_link_libraries(arc-input-rumble-tests
        PRIVATE
            arc-input
    )

    add_test(NAME arc-input-rumble-tests COMMAND arc-input-rumble-tests)

    add_executable(arc-input-battery-tests
        test_support/battery_tests.cpp
    )
    arc_configure_first_party_target(arc-input-battery-tests)

    target_compile_features(arc-input-battery-tests
        PRIVATE
            cxx_std_20
    )

    target_link_libraries(arc-input-battery-tests
        PRIVATE
            arc-input
    )

    add_test(NAME arc-input-battery-tests COMMAND arc-input-battery-tests)

    add_executable(arc-input-touch-tests
        test_support/touch_tests.cpp
    )
    arc_configure_first_party_target(arc-input-touch-tests)

    target_compile_features(arc-input-touch-tests
        PRIVATE
            cxx_std_20
    )

    target_link_libraries(arc-input-touch-tests
        PRIVATE
            arc-input
    )

    add_test(NAME arc-input-touch-tests COMMAND arc-input-touch-tests)

    add_executable(arc-input-advanced-output-tests
        test_support/advanced_output_tests.cpp
    )
    arc_configure_first_party_target(arc-input-advanced-output-tests)

    target_compile_features(arc-input-advanced-output-tests
        PRIVATE
            cxx_std_20
    )

    target_link_libraries(arc-input-advanced-output-tests
        PRIVATE
            arc-input
    )

    add_test(NAME arc-input-advanced-output-tests COMMAND arc-input-advanced-output-tests)
endif()
