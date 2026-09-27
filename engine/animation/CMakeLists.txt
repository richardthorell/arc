arc_add_module(
    NAME arc-animation
    TYPE STATIC
    DEPS arc-math
)

if(BUILD_TESTING)
    add_executable(arc-animation-foundation-tests
        test_support/foundation_tests.cpp
    )
    arc_configure_first_party_target(arc-animation-foundation-tests)
    target_compile_features(arc-animation-foundation-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-animation-foundation-tests PRIVATE arc-animation)
    add_test(NAME arc-animation-foundation-tests COMMAND arc-animation-foundation-tests)
endif()
