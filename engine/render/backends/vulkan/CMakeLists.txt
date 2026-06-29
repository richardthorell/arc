add_library(arc-render-vulkan STATIC
    inc/arc/render/vulkan/vulkan_backend.h
    src/vma_impl.cpp
    src/vulkan_backend.cpp
)

target_compile_features(arc-render-vulkan
    PUBLIC
        cxx_std_20
)

target_include_directories(arc-render-vulkan
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

target_link_libraries(arc-render-vulkan
    PUBLIC
        arc-render
        arc-third-party-vulkan
        arc-third-party-volk
        arc-third-party-vma
        arc-third-party-slang
)

target_compile_definitions(arc-render-vulkan
    PRIVATE
        VMA_STATIC_VULKAN_FUNCTIONS=1
        VMA_DYNAMIC_VULKAN_FUNCTIONS=0
)

if(ARC_BUILD_EDITOR AND TARGET arc-third-party-imgui)
    target_link_libraries(arc-render-vulkan
        PUBLIC
            arc-third-party-imgui
    )
    target_compile_definitions(arc-render-vulkan
        PRIVATE
            ARC_RENDER_VULKAN_ENABLE_IMGUI=1
    )
endif()

if(NOT TARGET arc::render-vulkan)
    add_library(arc::render-vulkan ALIAS arc-render-vulkan)
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
