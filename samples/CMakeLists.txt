cmake_minimum_required(VERSION 3.20)
project(arc_samples LANGUAGES CXX)

add_executable(arc_headless
    headless/main.cpp
)
arc_configure_first_party_target(arc_headless)
target_compile_features(arc_headless PRIVATE cxx_std_20)
target_link_libraries(arc_headless PRIVATE arc-framework)
install(TARGETS arc_headless
    RUNTIME DESTINATION bin
    COMPONENT headless)
if(MSVC)
    install(FILES "$<TARGET_PDB_FILE:arc_headless>"
        DESTINATION symbols
        COMPONENT debug-symbols
        CONFIGURATIONS RelWithDebInfo Shipping
        OPTIONAL)
endif()

# Compile the actual blank-headless server entrypoint with sample identities.
# This keeps the SDK/template bootstrap under native CI rather than relying only
# on isolated API tests. Real packaged content is supplied by the D0 finalizer.
set(_arc_headless_template "${CMAKE_SOURCE_DIR}/templates/blank-headless/Source/__PROJECT__Server/Main.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_arc_headless_template}")
file(READ "${_arc_headless_template}" _arc_headless_main)
string(REPLACE "{{PROJECT_TOKEN}}" "arc_packaged_sample" _arc_headless_main "${_arc_headless_main}")
string(REPLACE "{{ENGINE_VERSION}}" "${PROJECT_VERSION}" _arc_headless_main "${_arc_headless_main}")
string(REPLACE "{{PROJECT_GUID}}" "11111111-1111-1111-1111-111111111111" _arc_headless_main "${_arc_headless_main}")
set(_arc_headless_generated "${CMAKE_CURRENT_BINARY_DIR}/generated/packaged_server_main.cpp")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(WRITE "${_arc_headless_generated}" "${_arc_headless_main}")

add_executable(arc_packaged_server
    "${_arc_headless_generated}"
    headless/packaged_server_registration.cpp)
arc_configure_first_party_target(arc_packaged_server)
target_compile_features(arc_packaged_server PRIVATE cxx_std_20)
target_compile_definitions(arc_packaged_server PRIVATE
    ARC_SERVER_BUILD_CONFIGURATION="$<CONFIG>")
target_link_libraries(arc_packaged_server PRIVATE arc-server)
install(TARGETS arc_packaged_server
    RUNTIME DESTINATION bin
    COMPONENT headless)

if(BUILD_TESTING)
    add_test(NAME arc-packaged-server-cli-arguments
        COMMAND $<TARGET_FILE:arc_packaged_server> --ticks nope)
    set_tests_properties(arc-packaged-server-cli-arguments PROPERTIES WILL_FAIL TRUE)
endif()

if(WIN32 AND COMMAND arc_add_windows_app)
    arc_add_windows_app(
        NAME arc_sample_game
        SOURCES
            boot_app/main.cpp
    )
    install(TARGETS arc_sample_game
        RUNTIME DESTINATION bin
        COMPONENT sample-game)
    if(MSVC)
        install(FILES "$<TARGET_PDB_FILE:arc_sample_game>"
            DESTINATION symbols
            COMPONENT debug-symbols
            CONFIGURATIONS RelWithDebInfo Shipping
            OPTIONAL)
    endif()
else()
    message(STATUS "arc_sample_game is skipped because no supported platform entry is available")
endif()
