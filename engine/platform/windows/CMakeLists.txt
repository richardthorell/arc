if(NOT WIN32)
    message(FATAL_ERROR "arc-platform-windows can only be built on Windows")
endif()

add_library(arc-platform-windows STATIC
    src/windows_entry.cpp
)
arc_configure_first_party_target(arc-platform-windows)

target_compile_features(arc-platform-windows
    PUBLIC
        cxx_std_20
)

target_include_directories(arc-platform-windows PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>"
    "$<INSTALL_INTERFACE:include>")

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

set_target_properties(arc-platform-windows PROPERTIES EXPORT_NAME PlatformWindows)
install(TARGETS arc-platform-windows
    EXPORT ARCTargets
    ARCHIVE DESTINATION "lib/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
    COMPONENT sdk)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/inc")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/" DESTINATION include COMPONENT sdk-headers)
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
    arc_configure_first_party_target(${ARC_WINDOWS_APP_NAME})

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
