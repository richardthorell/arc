cmake_minimum_required(VERSION 3.20)

include(FetchContent)

option(ARC_USE_SYSTEM_TIFF "Use a system-provided libtiff package" OFF)
option(ARC_FETCH_TIFF "Fetch libtiff when ARC_USE_SYSTEM_TIFF is OFF" ON)

add_library(arc-third-party-tiff INTERFACE)
if(NOT TARGET arc::third-party-tiff)
    add_library(arc::third-party-tiff ALIAS arc-third-party-tiff)
endif()

if(ARC_USE_SYSTEM_TIFF)
    find_package(TIFF REQUIRED)
    target_link_libraries(arc-third-party-tiff INTERFACE TIFF::TIFF)
    target_compile_definitions(arc-third-party-tiff INTERFACE ARC_RENDER_HAS_TIFF=1)
elseif(ARC_FETCH_TIFF AND ARC_FETCH_THIRD_PARTY)
    set(tiff-static ON CACHE BOOL "" FORCE)
    set(tiff-tools OFF CACHE BOOL "" FORCE)
    set(tiff-tests OFF CACHE BOOL "" FORCE)
    set(tiff-contrib OFF CACHE BOOL "" FORCE)
    set(tiff-docs OFF CACHE BOOL "" FORCE)
    set(tiff-install OFF CACHE BOOL "" FORCE)
    set(jpeg OFF CACHE BOOL "" FORCE)
    set(zlib OFF CACHE BOOL "" FORCE)
    set(lzma OFF CACHE BOOL "" FORCE)
    set(zstd OFF CACHE BOOL "" FORCE)
    set(webp OFF CACHE BOOL "" FORCE)
    set(jbig OFF CACHE BOOL "" FORCE)
    set(lerc OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        libtiff
        URL https://download.osgeo.org/libtiff/tiff-4.7.2.tar.gz
        URL_HASH SHA256=672bd7d10aee4606171afb864f3570b83340f6a33e2c186dc0512f7145ffdf6a
    )
    FetchContent_MakeAvailable(libtiff)

    if(TARGET TIFF::TIFF)
        target_link_libraries(arc-third-party-tiff INTERFACE TIFF::TIFF)
    elseif(TARGET tiff)
        target_link_libraries(arc-third-party-tiff INTERFACE tiff)
    else()
        message(FATAL_ERROR "Fetched libtiff did not provide a usable library target")
    endif()
    target_compile_definitions(arc-third-party-tiff INTERFACE ARC_RENDER_HAS_TIFF=1)

    if(TARGET tiff)
        set_target_properties(tiff PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(MSVC)
            target_compile_options(tiff PRIVATE /W0)
        else()
            target_compile_options(tiff PRIVATE -w)
        endif()
    endif()
else()
    target_compile_definitions(arc-third-party-tiff INTERFACE ARC_RENDER_TIFF_UNAVAILABLE=1)
endif()
