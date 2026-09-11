arc_add_module(
    NAME arc-water
    TYPE STATIC
    DEPS arc-assets arc-math
    NO_TESTS
)

target_link_libraries(arc-water PRIVATE "$<BUILD_INTERFACE:arc-third-party-json>")

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
