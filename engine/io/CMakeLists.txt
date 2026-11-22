arc_add_module(
    NAME arc-io
    TYPE STATIC
    DEPS arc-jobs arc-memory arc-diagnostics
    NO_TESTS
)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
