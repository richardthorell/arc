cmake_minimum_required(VERSION 3.20)

include(FetchContent)

option(ARC_FETCH_THIRD_PARTY "Fetch third-party dependencies when system packages are not requested" ON)
option(ARC_USE_SYSTEM_SDL "Use a system-provided SDL3 package" OFF)
option(ARC_USE_SYSTEM_IMGUI "Use a system-provided Dear ImGui package" OFF)

add_library(arc-third-party-sdl INTERFACE)
add_library(arc-third-party-imgui STATIC)

if(NOT TARGET arc::third-party-sdl)
    add_library(arc::third-party-sdl ALIAS arc-third-party-sdl)
endif()

if(NOT TARGET arc::third-party-imgui)
    add_library(arc::third-party-imgui ALIAS arc-third-party-imgui)
endif()

if(ARC_USE_SYSTEM_SDL)
    find_package(SDL3 CONFIG REQUIRED)
elseif(ARC_FETCH_THIRD_PARTY)
    set(SDL_SHARED OFF CACHE BOOL "Build SDL shared library" FORCE)
    set(SDL_STATIC ON CACHE BOOL "Build SDL static library" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "Build SDL test library" FORCE)
    set(SDL_DISABLE_INSTALL ON CACHE BOOL "Disable SDL install rules" FORCE)

    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-3.4.10
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(SDL3)
else()
    message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_SDL is OFF")
endif()

if(TARGET SDL3::SDL3-static)
    target_link_libraries(arc-third-party-sdl INTERFACE SDL3::SDL3-static)
elseif(TARGET SDL3::SDL3)
    target_link_libraries(arc-third-party-sdl INTERFACE SDL3::SDL3)
else()
    message(FATAL_ERROR "SDL3 target was not found")
endif()

if(ARC_USE_SYSTEM_IMGUI)
    find_package(imgui CONFIG REQUIRED)
    target_link_libraries(arc-third-party-imgui PUBLIC imgui::imgui arc-third-party-sdl)
elseif(ARC_FETCH_THIRD_PARTY)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.92.8-docking
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(imgui)

    target_sources(arc-third-party-imgui
        PRIVATE
            "${imgui_SOURCE_DIR}/imgui.cpp"
            "${imgui_SOURCE_DIR}/imgui_draw.cpp"
            "${imgui_SOURCE_DIR}/imgui_tables.cpp"
            "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
            "${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp"
            "${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp"
    )

    target_include_directories(arc-third-party-imgui
        PUBLIC
            "${imgui_SOURCE_DIR}"
            "${imgui_SOURCE_DIR}/backends"
    )

    target_link_libraries(arc-third-party-imgui
        PUBLIC
            arc-third-party-sdl
    )
else()
    message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_IMGUI is OFF")
endif()

target_compile_features(arc-third-party-imgui
    PUBLIC
        cxx_std_20
)
