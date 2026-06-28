cmake_minimum_required(VERSION 3.20)
project(arc_samples LANGUAGES CXX)

if(WIN32 AND COMMAND arc_add_windows_app)
    arc_add_windows_app(
        NAME arc_boot_test_app
        SOURCES
            boot_app/main.cpp
    )
else()
    message(STATUS "arc_boot_test_app is skipped because no supported platform entry is available")
endif()
