arc_add_module(
    NAME arc-vfx
    TYPE STATIC
    DEPS arc-math
)

if(BUILD_TESTING)
    add_executable(arc-vfx-foundation-tests
        test_support/foundation_tests.cpp
    )
    arc_configure_first_party_target(arc-vfx-foundation-tests)
    target_compile_features(arc-vfx-foundation-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-vfx-foundation-tests PRIVATE arc-vfx)
    add_test(NAME arc-vfx-foundation-tests COMMAND arc-vfx-foundation-tests)
endif()
