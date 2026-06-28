add_library(arc-editor-core STATIC
    src/editor_console.cpp
    src/editor_viewport.cpp
    src/sdl_events.cpp
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

target_link_libraries(arc_editor
    PRIVATE
        arc-editor-core
        arc-third-party-imgui
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
