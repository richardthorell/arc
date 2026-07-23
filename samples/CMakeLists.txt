cmake_minimum_required(VERSION 3.20)
project(arc_samples LANGUAGES CXX)

add_executable(arc_headless
    headless/main.cpp
)
target_compile_features(arc_headless PRIVATE cxx_std_20)
target_link_libraries(arc_headless PRIVATE arc-framework)

if(WIN32 AND COMMAND arc_add_windows_app)
    arc_add_windows_app(
        NAME arc_boot_test_app
        SOURCES
            boot_app/main.cpp
    )
else()
    message(STATUS "arc_boot_test_app is skipped because no supported platform entry is available")
endif()
