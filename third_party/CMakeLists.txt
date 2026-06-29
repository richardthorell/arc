cmake_minimum_required(VERSION 3.20)

include(FetchContent)

option(ARC_FETCH_THIRD_PARTY "Fetch third-party dependencies when system packages are not requested" ON)
option(ARC_USE_SYSTEM_SDL "Use a system-provided SDL3 package" OFF)
option(ARC_USE_SYSTEM_IMGUI "Use a system-provided Dear ImGui package" OFF)
option(ARC_USE_SYSTEM_IMGUIZMO "Use a system-provided ImGuizmo package" OFF)
option(ARC_USE_SYSTEM_VULKAN "Use a system-provided Vulkan package" OFF)
option(ARC_USE_SYSTEM_VOLK "Use a system-provided volk package" OFF)
option(ARC_USE_SYSTEM_VMA "Use a system-provided Vulkan Memory Allocator package" OFF)
option(ARC_USE_SYSTEM_SLANG "Use a system-provided Slang package" OFF)
option(ARC_USE_SYSTEM_STB "Use a system-provided stb package" OFF)
option(ARC_FETCH_STB "Fetch stb when ARC_USE_SYSTEM_STB is OFF" OFF)

if(ARC_BUILD_EDITOR)
    add_library(arc-third-party-sdl INTERFACE)
    add_library(arc-third-party-imgui STATIC)
    add_library(arc-third-party-imgizmo STATIC)

    if(NOT TARGET arc::third-party-sdl)
        add_library(arc::third-party-sdl ALIAS arc-third-party-sdl)
    endif()

    if(NOT TARGET arc::third-party-imgui)
        add_library(arc::third-party-imgui ALIAS arc-third-party-imgui)
    endif()
    if(NOT TARGET arc::third-party-imgizmo)
        add_library(arc::third-party-imgizmo ALIAS arc-third-party-imgizmo)
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

    if(ARC_USE_SYSTEM_IMGUIZMO)
        find_package(ImGuizmo CONFIG REQUIRED)
        if(TARGET ImGuizmo::ImGuizmo)
            target_link_libraries(arc-third-party-imgizmo PUBLIC ImGuizmo::ImGuizmo arc-third-party-imgui)
        elseif(TARGET ImGuizmo)
            target_link_libraries(arc-third-party-imgizmo PUBLIC ImGuizmo arc-third-party-imgui)
        endif()
    elseif(ARC_FETCH_THIRD_PARTY)
        FetchContent_Declare(
            imguizmo
            GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
            GIT_TAG master
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(imguizmo)

        if(EXISTS "${imguizmo_SOURCE_DIR}/ImGuizmo.cpp")
            set(_arc_imguizmo_source_dir "${imguizmo_SOURCE_DIR}")
        else()
            set(_arc_imguizmo_source_dir "${imguizmo_SOURCE_DIR}/src")
        endif()

        target_sources(arc-third-party-imgizmo
            PRIVATE
                "${_arc_imguizmo_source_dir}/ImGuizmo.cpp"
        )

        target_include_directories(arc-third-party-imgizmo
            PUBLIC
                "${_arc_imguizmo_source_dir}"
        )

        target_link_libraries(arc-third-party-imgizmo
            PUBLIC
                arc-third-party-imgui
        )
    else()
        message(FATAL_ERROR "ARC_FETCH_THIRD_PARTY is OFF and ARC_USE_SYSTEM_IMGUIZMO is OFF")
    endif()

    target_compile_features(arc-third-party-imgizmo
        PUBLIC
            cxx_std_20
    )
endif()

if(ARC_BUILD_RENDER_VULKAN)
    add_library(arc-third-party-vulkan INTERFACE)
    add_library(arc-third-party-volk INTERFACE)
    add_library(arc-third-party-vma INTERFACE)
    add_library(arc-third-party-slang INTERFACE)
    add_library(arc-third-party-stb INTERFACE)

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
    if(NOT TARGET arc::third-party-stb)
        add_library(arc::third-party-stb ALIAS arc-third-party-stb)
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

    if(ARC_USE_SYSTEM_STB)
        find_path(ARC_STB_INCLUDE_DIR stb_image.h REQUIRED)
        target_include_directories(arc-third-party-stb INTERFACE "${ARC_STB_INCLUDE_DIR}")
    elseif(ARC_FETCH_STB)
        FetchContent_Declare(
            stb
            GIT_REPOSITORY https://github.com/nothings/stb.git
            GIT_TAG master
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(stb)
        target_include_directories(arc-third-party-stb INTERFACE "${stb_SOURCE_DIR}")
    else()
        target_compile_definitions(arc-third-party-stb INTERFACE ARC_RENDER_STB_UNAVAILABLE=1)
    endif()

    if(ARC_BUILD_EDITOR AND TARGET arc-third-party-imgui)
        target_sources(arc-third-party-imgui
            PRIVATE
                "${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp"
        )
        target_compile_definitions(arc-third-party-imgui
            PUBLIC
                IMGUI_IMPL_VULKAN_USE_VOLK
        )
        target_link_libraries(arc-third-party-imgui
            PUBLIC
                arc-third-party-volk
        )
    endif()
endif()
