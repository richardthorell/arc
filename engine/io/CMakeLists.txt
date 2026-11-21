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

# Linux uses libcurl for system-CA HTTPS and cancellable bounded transfers.
# Android NDK does not bundle libcurl; that backend remains a separate slice.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(CURL QUIET)
    if(CURL_FOUND)
        target_link_libraries(arc-io PRIVATE CURL::libcurl)
        target_compile_definitions(arc-io PUBLIC "$<BUILD_INTERFACE:ARC_CONTENT_HTTP_CURL=1>")
        set_property(GLOBAL PROPERTY ARC_IO_NEEDS_CURL TRUE)
    else()
        message(STATUS "ARC Linux remote HTTP disabled (libcurl development package not found)")
    endif()
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
