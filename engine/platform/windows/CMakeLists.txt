if(NOT WIN32)
    message(FATAL_ERROR "arc-platform-windows can only be built on Windows")
endif()

include(FetchContent)

option(ARC_WINDOWS_ENABLE_GAMEINPUT "Prefer Microsoft GameInput for Windows controllers when available" ON)

add_library(arc-third-party-gameinput INTERFACE)
if(NOT TARGET arc::third-party-gameinput)
    add_library(arc::third-party-gameinput ALIAS arc-third-party-gameinput)
endif()

set(ARC_WINDOWS_HAS_GAMEINPUT OFF)
if(ARC_WINDOWS_ENABLE_GAMEINPUT AND ARC_FETCH_THIRD_PARTY)
    FetchContent_Declare(
        arc_gameinput
        GIT_REPOSITORY https://github.com/microsoftconnect/GameInput.git
        GIT_TAG v3.3.195.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(arc_gameinput)

    set(_arc_gameinput_header "${arc_gameinput_SOURCE_DIR}/include/GameInput.h")
    set(_arc_gameinput_loader "${arc_gameinput_SOURCE_DIR}/releases/v3.3.195/lib/GameInput.cpp")
    if(EXISTS "${_arc_gameinput_header}" AND EXISTS "${_arc_gameinput_loader}")
        add_library(arc-third-party-gameinput-loader STATIC "${_arc_gameinput_loader}")
        target_include_directories(arc-third-party-gameinput-loader
            PUBLIC
                "$<BUILD_INTERFACE:${arc_gameinput_SOURCE_DIR}/include>"
                "$<INSTALL_INTERFACE:include/arc/private/gameinput>"
        )
        target_compile_features(arc-third-party-gameinput-loader PUBLIC cxx_std_20)
        if(MSVC)
            target_compile_options(arc-third-party-gameinput-loader PRIVATE /W0)
        else()
            target_compile_options(arc-third-party-gameinput-loader PRIVATE -w)
        endif()

        target_link_libraries(arc-third-party-gameinput INTERFACE arc-third-party-gameinput-loader)
        target_compile_definitions(arc-third-party-gameinput INTERFACE ARC_WINDOWS_HAS_GAMEINPUT=1)
        set_target_properties(arc-third-party-gameinput PROPERTIES EXPORT_NAME _GameInput)
        set_target_properties(arc-third-party-gameinput-loader PROPERTIES EXPORT_NAME _GameInputLoader)
        install(TARGETS arc-third-party-gameinput arc-third-party-gameinput-loader
            EXPORT ARCTargets
            ARCHIVE DESTINATION "lib/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
            COMPONENT sdk-private)
        install(FILES "${_arc_gameinput_header}"
            DESTINATION include/arc/private/gameinput
            COMPONENT sdk-private)
        set(ARC_WINDOWS_HAS_GAMEINPUT ON)
    endif()
endif()

if(NOT ARC_WINDOWS_HAS_GAMEINPUT)
    target_compile_definitions(arc-third-party-gameinput INTERFACE ARC_WINDOWS_HAS_GAMEINPUT=0)
    set_target_properties(arc-third-party-gameinput PROPERTIES EXPORT_NAME _GameInput)
    install(TARGETS arc-third-party-gameinput EXPORT ARCTargets COMPONENT sdk-private)
    message(STATUS "ARC Windows input: GameInput disabled; using XInput fallback")
else()
    message(STATUS "ARC Windows input: Microsoft GameInput provider enabled")
endif()

add_library(arc-platform-windows STATIC
    src/windows_controller_manager.cpp
    src/windows_entry.cpp
    src/windows_game_input_backend.cpp
    src/windows_gamepad_backend.cpp
    src/windows_input_backend.cpp
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
    PRIVATE
        arc-third-party-gameinput
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

    add_executable(arc_windows_controller_manager_tests
        tests/windows_controller_manager_tests.cpp
    )
    arc_configure_first_party_target(arc_windows_controller_manager_tests)
    target_compile_features(arc_windows_controller_manager_tests PRIVATE cxx_std_20)
    target_include_directories(arc_windows_controller_manager_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(arc_windows_controller_manager_tests PRIVATE arc-platform-windows)
    add_test(NAME arc_windows_controller_manager_tests COMMAND arc_windows_controller_manager_tests)
endif()
