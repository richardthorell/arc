add_library(arc-editor-host STATIC
    src/arc_host.cpp
    src/editor_console.cpp
    src/editor_gizmo.cpp
    src/editor_defaults.cpp
    src/editor_interaction.cpp
    src/editor_history.cpp
    src/editor_state.cpp
    src/editor_viewport.cpp
    src/host_protocol.cpp
    src/material_asset.cpp
    src/material_library.cpp
    src/material_preview.cpp
    src/prefab_document.cpp
    src/scene_document.cpp
    src/world_environment_host.cpp
)

target_compile_features(arc-editor-host
    PUBLIC
        cxx_std_20
)

target_include_directories(arc-editor-host
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

target_link_libraries(arc-editor-host
    PUBLIC
        arc
    PRIVATE
        arc-third-party-json
)

if(NOT TARGET arc::editor-host)
    add_library(arc::editor-host ALIAS arc-editor-host)
endif()

add_executable(arc_host_process
    src/arc_host_process_main.cpp
)

target_compile_features(arc_host_process
    PRIVATE
        cxx_std_20
)

target_link_libraries(arc_host_process
    PRIVATE
        arc-editor-host
)

if(ARC_BUILD_RENDER_VULKAN AND TARGET arc-render-vulkan)
    target_link_libraries(arc_host_process
        PRIVATE
            arc-render-vulkan
    )
    target_compile_definitions(arc_host_process
        PRIVATE
            ARC_EDITOR_HOST_ENABLE_VULKAN_RENDER=1
    )
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
