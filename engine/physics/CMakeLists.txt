arc_add_module(
    NAME arc-physics
    TYPE STATIC
    DEPS arc-math
)

if(BUILD_TESTING)
    add_executable(arc-physics-foundation-tests
        test_support/foundation_tests.cpp
    )
    arc_configure_first_party_target(arc-physics-foundation-tests)
    target_compile_features(arc-physics-foundation-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-physics-foundation-tests PRIVATE arc-physics)
    add_test(NAME arc-physics-foundation-tests COMMAND arc-physics-foundation-tests)
endif()
