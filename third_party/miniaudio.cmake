include_guard(GLOBAL)

include(FetchContent)

option(ARC_FETCH_THIRD_PARTY "Fetch third-party dependencies when system packages are not requested" ON)
option(ARC_USE_SYSTEM_MINIAUDIO "Use a system-provided miniaudio source/header pair" OFF)
option(ARC_FETCH_MINIAUDIO "Fetch miniaudio when ARC_USE_SYSTEM_MINIAUDIO is OFF" ON)

if(TARGET arc::third-party-miniaudio)
    return()
endif()

if(ARC_USE_SYSTEM_MINIAUDIO)
    find_path(ARC_MINIAUDIO_INCLUDE_DIR miniaudio.h REQUIRED)
    find_file(ARC_MINIAUDIO_SOURCE_FILE miniaudio.c
        HINTS "${ARC_MINIAUDIO_INCLUDE_DIR}"
        REQUIRED)
    set(_arc_miniaudio_include_dir "${ARC_MINIAUDIO_INCLUDE_DIR}")
    set(_arc_miniaudio_source "${ARC_MINIAUDIO_SOURCE_FILE}")
    set(_arc_miniaudio_header "${ARC_MINIAUDIO_INCLUDE_DIR}/miniaudio.h")
elseif(ARC_FETCH_THIRD_PARTY AND ARC_FETCH_MINIAUDIO)
    FetchContent_Declare(
        arc_miniaudio
        GIT_REPOSITORY https://github.com/mackron/miniaudio.git
        GIT_TAG 0.11.25
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR cmake/arc-no-subdir
    )
    FetchContent_MakeAvailable(arc_miniaudio)
    set(_arc_miniaudio_include_dir "${arc_miniaudio_SOURCE_DIR}")
    set(_arc_miniaudio_source "${arc_miniaudio_SOURCE_DIR}/miniaudio.c")
    set(_arc_miniaudio_header "${arc_miniaudio_SOURCE_DIR}/miniaudio.h")
else()
    message(FATAL_ERROR
        "miniaudio is required by arc-audio. Enable ARC_FETCH_THIRD_PARTY/ARC_FETCH_MINIAUDIO or ARC_USE_SYSTEM_MINIAUDIO.")
endif()

enable_language(C)

add_library(arc-third-party-miniaudio-static STATIC
    "${_arc_miniaudio_source}"
)
set_target_properties(arc-third-party-miniaudio-static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME _MiniaudioStatic)
target_include_directories(arc-third-party-miniaudio-static SYSTEM
    PUBLIC
        "$<BUILD_INTERFACE:${_arc_miniaudio_include_dir}>"
        "$<INSTALL_INTERFACE:include/arc/private/miniaudio>"
)
if(MSVC)
    target_compile_options(arc-third-party-miniaudio-static PRIVATE /W0)
else()
    target_compile_options(arc-third-party-miniaudio-static PRIVATE -w)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(arc-third-party-miniaudio-static PUBLIC pthread m)
    if(CMAKE_DL_LIBS)
        target_link_libraries(arc-third-party-miniaudio-static PUBLIC ${CMAKE_DL_LIBS})
    endif()
endif()

add_library(arc-third-party-miniaudio INTERFACE)
add_library(arc::third-party-miniaudio ALIAS arc-third-party-miniaudio)
target_link_libraries(arc-third-party-miniaudio INTERFACE arc-third-party-miniaudio-static)
set_target_properties(arc-third-party-miniaudio PROPERTIES EXPORT_NAME _Miniaudio)

install(TARGETS arc-third-party-miniaudio arc-third-party-miniaudio-static
    EXPORT ARCTargets
    ARCHIVE DESTINATION "lib/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
    COMPONENT sdk-private)
install(FILES "${_arc_miniaudio_header}"
    DESTINATION include/arc/private/miniaudio
    COMPONENT sdk-private)
