arc_add_module(
    NAME arc-core
    TYPE INTERFACE
    NO_TESTS
)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
