arc_add_module(
    NAME arc-framework
    TYPE STATIC
    DEPS arc-geometric arc-jobs arc-memory arc-ecs arc-diagnostics
    NO_TESTS
)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
