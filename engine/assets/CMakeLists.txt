arc_add_module(
    NAME arc-assets
    TYPE STATIC
    DEPS arc-framework arc-io arc-jobs arc-memory arc-diagnostics
    NO_TESTS
)

target_link_libraries(arc-assets
    PRIVATE
        arc-third-party-json
        arc-third-party-sqlite
)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
