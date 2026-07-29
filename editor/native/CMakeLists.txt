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
arc_configure_first_party_target(arc-editor-host)

target_compile_features(arc-editor-host
    PUBLIC
        cxx_std_20
)

target_include_directories(arc-editor-host
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

if(MSVC)
    # The host protocol dispatch intentionally centralizes many command
    # specializations in arc_host.cpp; debug builds can exceed COFF's default
    # section count as editor capabilities grow.
    target_compile_options(arc-editor-host PRIVATE /bigobj)
endif()

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
arc_configure_first_party_target(arc_host_process)

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

install(TARGETS arc_host_process
    RUNTIME DESTINATION native
    COMPONENT editor-host)
if(MSVC)
    install(FILES "$<TARGET_PDB_FILE:arc_host_process>"
        DESTINATION symbols
        COMPONENT debug-symbols
        CONFIGURATIONS RelWithDebInfo Shipping
        OPTIONAL)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
