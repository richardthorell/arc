arc_add_module(
    NAME arc-navigation
    TYPE STATIC
    DEPS arc-math
)

if(BUILD_TESTING)
    add_executable(arc-navigation-foundation-tests
        test_support/foundation_tests.cpp
    )
    arc_configure_first_party_target(arc-navigation-foundation-tests)
    target_compile_features(arc-navigation-foundation-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-navigation-foundation-tests PRIVATE arc-navigation)
    add_test(NAME arc-navigation-foundation-tests COMMAND arc-navigation-foundation-tests)
endif()
