arc_add_module(
    NAME arc-network
    TYPE STATIC
)

if(BUILD_TESTING)
    add_executable(arc-network-foundation-tests
        test_support/foundation_tests.cpp
    )
    arc_configure_first_party_target(arc-network-foundation-tests)
    target_compile_features(arc-network-foundation-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-network-foundation-tests PRIVATE arc-network)
    add_test(NAME arc-network-foundation-tests COMMAND arc-network-foundation-tests)
endif()
