arc_add_module(
    NAME arc-framework
    TYPE STATIC
    DEPS arc-geometric
    NO_TESTS
)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
