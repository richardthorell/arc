arc_add_module(
    NAME arc-persistence
    TYPE STATIC
    DEPS arc-ecs arc-assets arc-io arc-diagnostics
    NO_TESTS
)

target_link_libraries(arc-persistence PRIVATE arc-third-party-json)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
