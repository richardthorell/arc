arc_add_module(
    NAME arc-project
    TYPE STATIC
    DEPS arc-core arc-assets arc-persistence arc-diagnostics
    NO_TESTS
)

target_link_libraries(arc-project PRIVATE "$<BUILD_INTERFACE:arc-third-party-json>")

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
