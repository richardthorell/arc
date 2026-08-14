add_library(arc-render-vulkan SHARED
    inc/arc/render/vulkan/vulkan_backend.h
    src/vma_impl.cpp
    src/vulkan_backend.cpp
    src/vulkan_pick_utils.h
    src/vulkan_sky_constants.cpp
    src/vulkan_sky_constants.h
    src/vulkan_swapchain.cpp
    src/vulkan_swapchain.h
)
arc_configure_first_party_target(arc-render-vulkan)

# The implementation translation unit is entirely third-party VMA code. Keep
# ARC's warning gate on the Vulkan backend while exempting this vendor source.
if(MSVC)
    set_source_files_properties(src/vma_impl.cpp PROPERTIES COMPILE_OPTIONS "/W0;/WX-")
else()
    set_source_files_properties(src/vma_impl.cpp PROPERTIES COMPILE_OPTIONS "-w;-Wno-error")
endif()

# VMA 3.3 annotates pointers with Clang's _Nonnull/_Nullable extensions. ARC
# builds first-party code with -Wpedantic -Werror, so suppress only that vendor
# header diagnostic in the translation unit that consumes VMA's public API.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(src/vulkan_backend.cpp PROPERTIES COMPILE_OPTIONS "-Wno-nullability-extension")
endif()

target_compile_features(arc-render-vulkan
    PUBLIC
        cxx_std_20
)

target_include_directories(arc-render-vulkan
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>"
        "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(arc-render-vulkan
    PUBLIC
        arc-render
        "$<BUILD_INTERFACE:arc-third-party-vulkan>"
        "$<INSTALL_INTERFACE:Vulkan::Vulkan>"
    PRIVATE
        arc-third-party-volk
        arc-third-party-vma
        arc-third-party-slang
        arc-third-party-stb
)

target_compile_definitions(arc-render-vulkan
    PRIVATE
        VK_NO_PROTOTYPES=1
        VMA_STATIC_VULKAN_FUNCTIONS=1
        VMA_DYNAMIC_VULKAN_FUNCTIONS=0
)
if(WIN32)
    target_compile_definitions(arc-render-vulkan PRIVATE VK_USE_PLATFORM_WIN32_KHR=1 WIN32_LEAN_AND_MEAN=1 NOMINMAX=1)
endif()

# D3D11 is used only to allocate NT-handle textures for Electron's shared
# editor viewport. Runtime/Shipping builds keep the Vulkan backend free of
# D3D and DXGI link dependencies.
if(WIN32 AND ARC_BUILD_EDITOR)
    target_compile_definitions(arc-render-vulkan PRIVATE ARC_EDITOR=1)
    target_link_libraries(arc-render-vulkan PRIVATE d3d11 dxgi)
endif()

if(NOT TARGET arc::render-vulkan)
    add_library(arc::render-vulkan ALIAS arc-render-vulkan)
endif()

set_target_properties(arc-render-vulkan PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    EXPORT_NAME RenderVulkan)
install(TARGETS arc-render-vulkan
    EXPORT ARCTargets
    ARCHIVE DESTINATION "lib/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
    LIBRARY DESTINATION "lib/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
    RUNTIME DESTINATION "bin/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
    COMPONENT sdk)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/" DESTINATION include COMPONENT sdk-headers)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
