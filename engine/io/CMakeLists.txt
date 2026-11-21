arc_add_module(
    NAME arc-io
    TYPE STATIC
    DEPS arc-jobs arc-memory arc-diagnostics
    NO_TESTS
)

# WinHTTP is an OS-provided TLS client; no new third-party downloader is shipped.
if(WIN32)
    target_link_libraries(arc-io PRIVATE winhttp)
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
