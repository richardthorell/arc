add_library(arc-editor-core STATIC
    src/arc_host.cpp
    src/asset_drag_drop.cpp
    src/editor_console.cpp
    src/editor_defaults.cpp
    src/editor_interaction.cpp
    src/editor_state.cpp
    src/editor_viewport.cpp
    src/host_protocol.cpp
    src/material_asset.cpp
    src/material_library.cpp
    src/panels/console_panel.cpp
    src/panels/content_browser_panel.cpp
    src/panels/inspector_panel.cpp
    src/panels/material_editor_panel.cpp
    src/panels/profiler_panel.cpp
    src/panels/render_graph_panel.cpp
    src/panels/scene_hierarchy_panel.cpp
    src/panels/shader_graph_panel.cpp
    src/panels/status_bar.cpp
    src/panels/toolbar_panel.cpp
    src/sdl_events.cpp
    src/ui/theme.cpp
    src/ui/widgets.cpp
    src/world_environment_host.cpp
)

target_compile_features(arc-editor-core
    PUBLIC
        cxx_std_20
)

target_include_directories(arc-editor-core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

target_link_libraries(arc-editor-core
    PUBLIC
        arc
        arc-third-party-sdl
        arc-third-party-imgui
)

if(NOT TARGET arc::editor-core)
    add_library(arc::editor-core ALIAS arc-editor-core)
endif()

add_executable(arc_editor
    src/editor_application.cpp
    src/main.cpp
)

target_compile_features(arc_editor
    PRIVATE
        cxx_std_20
)

target_compile_definitions(arc_editor
    PRIVATE
        "ARC_EDITOR_ASSET_ROOT=\"${CMAKE_SOURCE_DIR}/assets\""
)

target_link_libraries(arc_editor
    PRIVATE
        arc-editor-core
        arc-third-party-imgui
        arc-third-party-imgizmo
)

if(ARC_BUILD_RENDER_VULKAN AND TARGET arc-render-vulkan)
    target_link_libraries(arc_editor
        PRIVATE
            arc-render-vulkan
    )
    target_compile_definitions(arc_editor
        PRIVATE
            ARC_EDITOR_ENABLE_VULKAN_RENDER=1
    )
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
        arc-editor-core
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
