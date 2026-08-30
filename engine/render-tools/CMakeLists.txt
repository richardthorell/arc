arc_add_module(
    NAME arc-render-tools
    TYPE STATIC
    DEPS arc-render arc-assets
    NO_TESTS
)

target_link_libraries(arc-render-tools PRIVATE "$<BUILD_INTERFACE:arc-third-party-json>")
target_sources(arc-render-tools PRIVATE
    src/common/material_asset.cpp
    src/common/material_function.cpp
    src/common/material_graph_codegen.cpp
    src/common/material_graph_ir.cpp
    src/common/material_pass_codegen.cpp
    src/common/texture_cooker.cpp
)
target_compile_definitions(arc-render-tools PRIVATE ARC_PINNED_SLANG_VERSION="2026.14.1")

set(ARC_SLANGC_EXECUTABLE "" CACHE FILEPATH
    "Path to the pinned Slang 2026.14.1 command-line compiler used by editor/cooker tooling")
if(ARC_SLANGC_EXECUTABLE)
    target_compile_definitions(arc-render-tools PRIVATE
        ARC_DEFAULT_SLANGC_EXECUTABLE="${ARC_SLANGC_EXECUTABLE}")
endif()

install(DIRECTORY shaders/
    DESTINATION share/arc/shaders
    COMPONENT sdk)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
