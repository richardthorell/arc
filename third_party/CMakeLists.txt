cmake_minimum_required(VERSION 3.20)

include(FetchContent)

option(ARC_FETCH_THIRD_PARTY "Fetch third-party dependencies when system packages are not requested" ON)
option(ARC_USE_SYSTEM_IMGUI "Use a system-provided Dear ImGui package for Vulkan presentation helpers" OFF)
option(ARC_USE_SYSTEM_VULKAN "Use a system-provided Vulkan package" OFF)
option(ARC_USE_SYSTEM_VOLK "Use a system-provided volk package" OFF)
option(ARC_USE_SYSTEM_VMA "Use a system-provided Vulkan Memory Allocator package" OFF)
option(ARC_USE_SYSTEM_SLANG "Use a system-provided Slang package" OFF)
option(ARC_USE_SYSTEM_STB "Use a system-provided stb package" OFF)
option(ARC_FETCH_STB "Fetch stb when ARC_USE_SYSTEM_STB is OFF" ON)
option(ARC_USE_SYSTEM_UFBX "Use a system-provided ufbx package" OFF)
option(ARC_FETCH_UFBX "Fetch ufbx when ARC_USE_SYSTEM_UFBX is OFF" ON)
option(ARC_USE_SYSTEM_NLOHMANN_JSON "Use a system-provided nlohmann_json package" OFF)
option(ARC_USE_SYSTEM_SQLITE "Use a system-provided SQLite package" OFF)
option(ARC_USE_SYSTEM_MESHOPTIMIZER "Use a system-provided meshoptimizer package" OFF)
option(ARC_USE_SYSTEM_PNG "Use a system-provided libpng package" OFF)

if(ARC_BUILD_EDITOR)
    add_library(arc-third-party-png INTERFACE)
    if(NOT TARGET arc::third-party-png)
        add_library(arc::third-party-png ALIAS arc-third-party-png)
    endif()
    if(ARC_USE_SYSTEM_PNG)
        find_package(PNG REQUIRED)
        target_link_libraries(arc-third-party-png INTERFACE PNG::PNG)
    elseif(ARC_FETCH_THIRD_PARTY)
        set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            zlib
            GIT_REPOSITORY https://github.com/madler/zlib.git
            GIT_TAG v1.3.1
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(zlib)
        set(ZLIB_INCLUDE_DIR "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}" CACHE STRING "" FORCE)
        if(TARGET ZLIB::ZLIB)
            set_property(TARGET ZLIB::ZLIB PROPERTY INTERFACE_LINK_LIBRARIES zlibstatic)
        else()
            add_library(ZLIB::ZLIB ALIAS zlibstatic)
        endif()
        list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        set(PNG_SHARED OFF CACHE BOOL "" FORCE)
        set(PNG_STATIC ON CACHE BOOL "" FORCE)
        set(PNG_TESTS OFF CACHE BOOL "" FORCE)
        set(PNG_TOOLS OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            libpng
            GIT_REPOSITORY https://github.com/pnggroup/libpng.git
            GIT_TAG v1.6.50
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(libpng)
        if(TARGET png_static)
            target_link_libraries(arc-third-party-png INTERFACE png_static zlibstatic)
        elseif(TARGET PNG::PNG)
            target_link_libraries(arc-third-party-png INTERFACE PNG::PNG)
        endif()
        foreach(target zlibstatic png_static)
            if(TARGET ${target})
                if(MSVC)
                    target_compile_options(${target} PRIVATE /W0)
                else()
                    target_compile_options(${target} PRIVATE -w)
                endif()
            endif()
        endforeach()
    else()
        message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_PNG is OFF")
    endif()
endif()

add_library(arc-third-party-meshoptimizer INTERFACE)
if(NOT TARGET arc::third-party-meshoptimizer)
    add_library(arc::third-party-meshoptimizer ALIAS arc-third-party-meshoptimizer)
endif()

if(ARC_USE_SYSTEM_MESHOPTIMIZER)
    find_package(meshoptimizer CONFIG REQUIRED)
    target_link_libraries(arc-third-party-meshoptimizer INTERFACE meshoptimizer::meshoptimizer)
elseif(ARC_FETCH_THIRD_PARTY)
    set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "" FORCE)
    set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
    set(MESHOPT_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(MESHOPT_WERROR OFF CACHE BOOL "" FORCE)
    set(MESHOPT_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        meshoptimizer
        GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
        GIT_TAG 9d9890c73011d75920af614485296d1e03e95448
    )
    FetchContent_MakeAvailable(meshoptimizer)
    target_include_directories(meshoptimizer PUBLIC "$<INSTALL_INTERFACE:include/arc/private/meshoptimizer>")
    if(MSVC)
        target_compile_options(meshoptimizer PRIVATE /W0)
    else()
        target_compile_options(meshoptimizer PRIVATE -w)
    endif()
    target_link_libraries(arc-third-party-meshoptimizer INTERFACE meshoptimizer)
    set_target_properties(arc-third-party-meshoptimizer PROPERTIES EXPORT_NAME _Meshoptimizer)
    set_target_properties(meshoptimizer PROPERTIES EXPORT_NAME _MeshoptimizerStatic)
    install(TARGETS arc-third-party-meshoptimizer meshoptimizer
        EXPORT ARCTargets
        ARCHIVE DESTINATION "lib/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
        COMPONENT sdk-private)
    install(FILES "${meshoptimizer_SOURCE_DIR}/src/meshoptimizer.h"
        DESTINATION include/arc/private/meshoptimizer COMPONENT sdk-private)
else()
    message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_MESHOPTIMIZER is OFF")
endif()

add_library(arc-third-party-json INTERFACE)
if(NOT TARGET arc::third-party-json)
    add_library(arc::third-party-json ALIAS arc-third-party-json)
endif()

if(ARC_USE_SYSTEM_NLOHMANN_JSON)
    find_package(nlohmann_json CONFIG REQUIRED)
    target_link_libraries(arc-third-party-json INTERFACE nlohmann_json::nlohmann_json)
elseif(ARC_FETCH_THIRD_PARTY)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)
    target_link_libraries(arc-third-party-json INTERFACE nlohmann_json::nlohmann_json)
else()
    message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_NLOHMANN_JSON is OFF")
endif()

add_library(arc-third-party-sqlite INTERFACE)
if(NOT TARGET arc::third-party-sqlite)
    add_library(arc::third-party-sqlite ALIAS arc-third-party-sqlite)
endif()

if(ARC_USE_SYSTEM_SQLITE)
    find_package(SQLite3 REQUIRED)
    target_link_libraries(arc-third-party-sqlite INTERFACE SQLite::SQLite3)
elseif(ARC_FETCH_THIRD_PARTY)
    enable_language(C)
    FetchContent_Declare(
        sqlite_amalgamation
        URL https://www.sqlite.org/2026/sqlite-amalgamation-3530300.zip
        URL_HASH SHA256=646421e12aac110282ef8cc68f1a62d4bb15fc7b8f09da0b53e29ee690500431
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(sqlite_amalgamation)
    add_library(arc-third-party-sqlite-static STATIC
        "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c"
    )
    target_include_directories(arc-third-party-sqlite-static
        PUBLIC
            "$<BUILD_INTERFACE:${sqlite_amalgamation_SOURCE_DIR}>"
            "$<INSTALL_INTERFACE:include/arc/private/sqlite>"
    )
    target_compile_definitions(arc-third-party-sqlite-static
        PUBLIC
            SQLITE_THREADSAFE=1
            SQLITE_ENABLE_FTS5
            SQLITE_ENABLE_JSON1
        PRIVATE
            SQLITE_DEFAULT_MEMSTATUS=0
    )
    if(MSVC)
        target_compile_options(arc-third-party-sqlite-static PRIVATE /W0)
    else()
        target_compile_options(arc-third-party-sqlite-static PRIVATE -w)
    endif()
    target_link_libraries(arc-third-party-sqlite INTERFACE arc-third-party-sqlite-static)
    set_target_properties(arc-third-party-sqlite PROPERTIES EXPORT_NAME _SQLite)
    set_target_properties(arc-third-party-sqlite-static PROPERTIES EXPORT_NAME _SQLiteStatic)
    install(TARGETS arc-third-party-sqlite arc-third-party-sqlite-static
        EXPORT ARCTargets
        ARCHIVE DESTINATION "lib/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
        COMPONENT sdk-private)
    install(FILES "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.h"
        DESTINATION include/arc/private/sqlite COMPONENT sdk-private)
else()
    message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_SQLITE is OFF")
endif()

add_library(arc-third-party-stb INTERFACE)
if(NOT TARGET arc::third-party-stb)
    add_library(arc::third-party-stb ALIAS arc-third-party-stb)
endif()

if(ARC_USE_SYSTEM_STB)
    find_path(ARC_STB_INCLUDE_DIR stb_image.h REQUIRED)
    target_include_directories(arc-third-party-stb INTERFACE "${ARC_STB_INCLUDE_DIR}")
    target_compile_definitions(arc-third-party-stb INTERFACE ARC_RENDER_HAS_STB=1)
elseif(ARC_FETCH_STB)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4
    )
    FetchContent_MakeAvailable(stb)
    target_include_directories(arc-third-party-stb INTERFACE "${stb_SOURCE_DIR}")
    target_compile_definitions(arc-third-party-stb INTERFACE ARC_RENDER_HAS_STB=1)
else()
    target_compile_definitions(arc-third-party-stb INTERFACE ARC_RENDER_STB_UNAVAILABLE=1)
endif()

if(ARC_BUILD_EDITOR OR ARC_BUILD_RENDER_VULKAN)
add_library(arc-third-party-ufbx INTERFACE)
if(NOT TARGET arc::third-party-ufbx)
    add_library(arc::third-party-ufbx ALIAS arc-third-party-ufbx)
endif()
set(ARC_THIRD_PARTY_UFBX_AVAILABLE OFF CACHE INTERNAL "Whether ufbx is available for ARC importers")

if(ARC_USE_SYSTEM_UFBX)
    find_path(ARC_UFBX_INCLUDE_DIR ufbx.h REQUIRED)
    find_library(ARC_UFBX_LIBRARY ufbx REQUIRED)
    target_include_directories(arc-third-party-ufbx INTERFACE "${ARC_UFBX_INCLUDE_DIR}")
    target_link_libraries(arc-third-party-ufbx INTERFACE "${ARC_UFBX_LIBRARY}")
    set(ARC_THIRD_PARTY_UFBX_AVAILABLE ON CACHE INTERNAL "Whether ufbx is available for ARC importers" FORCE)
elseif(ARC_FETCH_THIRD_PARTY AND ARC_FETCH_UFBX)
    enable_language(C)
    FetchContent_Declare(
        ufbx
        GIT_REPOSITORY https://github.com/ufbx/ufbx.git
        GIT_TAG fcc5d6ba444cfd3eb80677dba5e37e493941abe5
    )
    FetchContent_MakeAvailable(ufbx)

    if(EXISTS "${ufbx_SOURCE_DIR}/ufbx.c" AND EXISTS "${ufbx_SOURCE_DIR}/ufbx.h")
        add_library(arc-third-party-ufbx-static STATIC "${ufbx_SOURCE_DIR}/ufbx.c")
        target_include_directories(arc-third-party-ufbx-static PUBLIC
            "$<BUILD_INTERFACE:${ufbx_SOURCE_DIR}>"
            "$<INSTALL_INTERFACE:include/arc/private/ufbx>")
        target_link_libraries(arc-third-party-ufbx INTERFACE arc-third-party-ufbx-static)
        set_target_properties(arc-third-party-ufbx PROPERTIES EXPORT_NAME _Ufbx)
        set_target_properties(arc-third-party-ufbx-static PROPERTIES EXPORT_NAME _UfbxStatic)
        install(TARGETS arc-third-party-ufbx arc-third-party-ufbx-static
            EXPORT ARCTargets
            ARCHIVE DESTINATION "lib/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
            COMPONENT sdk-private)
        install(FILES "${ufbx_SOURCE_DIR}/ufbx.h" DESTINATION include/arc/private/ufbx COMPONENT sdk-private)
        set(ARC_THIRD_PARTY_UFBX_AVAILABLE ON CACHE INTERNAL "Whether ufbx is available for ARC importers" FORCE)
    endif()
else()
    target_compile_definitions(arc-third-party-ufbx INTERFACE ARC_RENDER_UFBX_UNAVAILABLE=1)
endif()
endif()

if(ARC_BUILD_RENDER_VULKAN)
    add_library(arc-third-party-vulkan INTERFACE)
    add_library(arc-third-party-volk INTERFACE)
    add_library(arc-third-party-vma INTERFACE)
    add_library(arc-third-party-slang INTERFACE)

    if(NOT TARGET arc::third-party-vulkan)
        add_library(arc::third-party-vulkan ALIAS arc-third-party-vulkan)
    endif()
    if(NOT TARGET arc::third-party-volk)
        add_library(arc::third-party-volk ALIAS arc-third-party-volk)
    endif()
    if(NOT TARGET arc::third-party-vma)
        add_library(arc::third-party-vma ALIAS arc-third-party-vma)
    endif()
    if(NOT TARGET arc::third-party-slang)
        add_library(arc::third-party-slang ALIAS arc-third-party-slang)
    endif()

    if(ARC_USE_SYSTEM_VULKAN)
        find_package(Vulkan REQUIRED)
        target_link_libraries(arc-third-party-vulkan INTERFACE Vulkan::Vulkan)
    elseif(ARC_FETCH_THIRD_PARTY)
        FetchContent_Declare(
            VulkanHeaders
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG v1.4.304
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(VulkanHeaders)
        if(TARGET Vulkan::Headers)
            target_link_libraries(arc-third-party-vulkan INTERFACE Vulkan::Headers)
        elseif(TARGET Vulkan-Headers)
            target_link_libraries(arc-third-party-vulkan INTERFACE Vulkan-Headers)
        else()
            target_include_directories(arc-third-party-vulkan INTERFACE "${vulkanheaders_SOURCE_DIR}/include")
        endif()
    else()
        message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_VULKAN is OFF")
    endif()

    if(ARC_USE_SYSTEM_VOLK)
        find_package(volk CONFIG REQUIRED)
        target_link_libraries(arc-third-party-volk INTERFACE volk::volk)
    elseif(ARC_FETCH_THIRD_PARTY)
        FetchContent_Declare(
            volk
            GIT_REPOSITORY https://github.com/zeux/volk.git
            GIT_TAG 1.4.304
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(volk)
        if(TARGET volk)
            target_link_libraries(volk PUBLIC arc-third-party-vulkan)
        endif()
        if(TARGET volk::volk)
            target_link_libraries(arc-third-party-volk INTERFACE volk::volk)
        elseif(TARGET volk)
            target_link_libraries(arc-third-party-volk INTERFACE volk)
        else()
            target_include_directories(arc-third-party-volk INTERFACE "${volk_SOURCE_DIR}")
            target_sources(arc-third-party-volk INTERFACE "${volk_SOURCE_DIR}/volk.c")
        endif()
    else()
        message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_VOLK is OFF")
    endif()

    target_link_libraries(arc-third-party-volk INTERFACE arc-third-party-vulkan)

    if(ARC_USE_SYSTEM_VMA)
        find_package(VulkanMemoryAllocator CONFIG REQUIRED)
        target_link_libraries(arc-third-party-vma INTERFACE VulkanMemoryAllocator::VulkanMemoryAllocator)
    elseif(ARC_FETCH_THIRD_PARTY)
        FetchContent_Declare(
            vma
            GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
            GIT_TAG v3.3.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(vma)
        if(TARGET GPUOpen::VulkanMemoryAllocator)
            target_link_libraries(arc-third-party-vma INTERFACE GPUOpen::VulkanMemoryAllocator)
        elseif(TARGET VulkanMemoryAllocator)
            target_link_libraries(arc-third-party-vma INTERFACE VulkanMemoryAllocator)
        else()
            target_include_directories(arc-third-party-vma INTERFACE "${vma_SOURCE_DIR}/include")
        endif()
    else()
        message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_VMA is OFF")
    endif()

    target_link_libraries(arc-third-party-vma INTERFACE arc-third-party-vulkan)

    if(ARC_USE_SYSTEM_SLANG)
        find_package(slang CONFIG REQUIRED)
        if(TARGET slang::slang)
            target_link_libraries(arc-third-party-slang INTERFACE slang::slang)
        elseif(TARGET slang)
            target_link_libraries(arc-third-party-slang INTERFACE slang)
        endif()
    else()
    target_compile_definitions(arc-third-party-slang INTERFACE ARC_RENDER_SLANG_UNAVAILABLE=1)
    endif()

    if(ARC_BUILD_EDITOR)
        add_library(arc-third-party-imgui STATIC)
        set_target_properties(arc-third-party-imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(NOT TARGET arc::third-party-imgui)
            add_library(arc::third-party-imgui ALIAS arc-third-party-imgui)
        endif()

        if(ARC_USE_SYSTEM_IMGUI)
            find_package(imgui CONFIG REQUIRED)
            target_link_libraries(arc-third-party-imgui PUBLIC imgui::imgui arc-third-party-volk)
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
                    "${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp"
            )
            target_include_directories(arc-third-party-imgui
                PUBLIC
                    "${imgui_SOURCE_DIR}"
                    "${imgui_SOURCE_DIR}/backends"
            )
            target_link_libraries(arc-third-party-imgui PUBLIC arc-third-party-volk)
        else()
            message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_IMGUI is OFF")
        endif()

        target_compile_features(arc-third-party-imgui PUBLIC cxx_std_20)
        target_compile_definitions(arc-third-party-imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
    endif()

endif()
