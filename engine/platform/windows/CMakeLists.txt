if(NOT WIN32)
    message(FATAL_ERROR "arc-platform-windows can only be built on Windows")
endif()

add_library(arc-platform-windows STATIC
    src/windows_entry.cpp
)

target_compile_features(arc-platform-windows
    PUBLIC
        cxx_std_20
)

target_link_libraries(arc-platform-windows
    PUBLIC
        arc
)

target_compile_definitions(arc-platform-windows
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
)

if(NOT TARGET arc::platform-windows)
    add_library(arc::platform-windows ALIAS arc-platform-windows)
endif()

function(arc_add_windows_app)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES DEPS)
    cmake_parse_arguments(ARC_WINDOWS_APP "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARC_WINDOWS_APP_NAME)
        message(FATAL_ERROR "arc_add_windows_app requires NAME")
    endif()

    if(NOT ARC_WINDOWS_APP_SOURCES)
        message(FATAL_ERROR "arc_add_windows_app requires SOURCES")
    endif()

    add_executable(${ARC_WINDOWS_APP_NAME} WIN32
        ${ARC_WINDOWS_APP_SOURCES}
    )

    target_compile_features(${ARC_WINDOWS_APP_NAME}
        PRIVATE
            cxx_std_20
    )

    target_link_libraries(${ARC_WINDOWS_APP_NAME}
        PRIVATE
            arc-platform-windows
            ${ARC_WINDOWS_APP_DEPS}
    )
endfunction()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    arc_add_windows_app(
        NAME arc_windows_smoke_app
        SOURCES tests/windows_smoke_app.cpp
    )
endif()
