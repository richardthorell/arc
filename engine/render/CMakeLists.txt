arc_add_module(
    NAME arc-render
    TYPE STATIC
    DEPS arc-framework arc-math arc-jobs arc-memory arc-io arc-diagnostics
    NO_TESTS
)

if(TARGET arc-third-party-ufbx)
    target_link_libraries(arc-render PUBLIC arc-third-party-ufbx)
    if(ARC_THIRD_PARTY_UFBX_AVAILABLE)
        target_compile_definitions(arc-render PRIVATE ARC_RENDER_HAS_UFBX=1)
    endif()
endif()

if(TARGET arc-third-party-stb)
    target_link_libraries(arc-render PRIVATE arc-third-party-stb)
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
