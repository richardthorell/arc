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
endif()
