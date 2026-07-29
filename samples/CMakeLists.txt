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
