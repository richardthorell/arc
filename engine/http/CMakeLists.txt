arc_add_module(
    NAME arc-http
    TYPE STATIC
    DEPS arc-io arc-jobs
    NO_TESTS
)

# Native HTTP transport is shared by games and editor hosts, not editor-only.
if(WIN32)
    target_link_libraries(arc-http PRIVATE winhttp)
endif()

# Linux uses the system CA store through libcurl. Android gets its own backend.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(CURL QUIET)
    if(CURL_FOUND)
        target_link_libraries(arc-http PRIVATE CURL::libcurl)
        target_compile_definitions(arc-http PUBLIC "$<BUILD_INTERFACE:ARC_CONTENT_HTTP_CURL=1>")
        set_property(GLOBAL PROPERTY ARC_HTTP_NEEDS_CURL TRUE)
    else()
        message(STATUS "ARC Linux remote HTTP disabled (libcurl development package not found)")
    endif()
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
