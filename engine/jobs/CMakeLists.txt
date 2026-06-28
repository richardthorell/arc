arc_add_module(
    NAME arc-jobs
    TYPE STATIC
    DEPS arc-diagnostics
    NO_TESTS
)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
