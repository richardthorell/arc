arc_add_module(
    NAME arc-render
    TYPE STATIC
    DEPS arc-framework arc-math arc-jobs arc-memory arc-diagnostics
    NO_TESTS
)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()

if(ARC_BUILD_RENDER_VULKAN)
    add_subdirectory(backends/vulkan)
endif()
