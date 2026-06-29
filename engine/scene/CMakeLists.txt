arc_add_module(
    NAME arc-scene
    TYPE STATIC
    DEPS arc-render arc-math arc-framework arc-diagnostics arc-memory
    NO_TESTS
)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
