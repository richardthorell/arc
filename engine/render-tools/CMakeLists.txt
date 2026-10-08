arc_add_module(
    NAME arc-render-tools
    TYPE STATIC
    DEPS arc-render arc-assets
    NO_TESTS
)

target_link_libraries(arc-render-tools PRIVATE
    "$<BUILD_INTERFACE:arc-third-party-json>"
    "$<BUILD_INTERFACE:arc-third-party-stb>"
)
target_sources(arc-render-tools PRIVATE
    src/common/material_asset.cpp
    src/common/material_function.cpp
    src/common/material_graph_codegen.cpp
    src/common/material_graph_ir.cpp
    src/common/material_pass_codegen.cpp
    src/common/texture_cooker.cpp
)
target_compile_definitions(arc-render-tools PRIVATE ARC_PINNED_SLANG_VERSION="2026.14.1")

# GLSL and composed Slang passes use the same generation-safe lookup and PCF policy.
set(arc_vsm_shader_include "${CMAKE_CURRENT_SOURCE_DIR}/../../assets/shaders/include")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${arc_vsm_shader_include}/arc_virtual_shadow_sampling.slang"
    "${arc_vsm_shader_include}/arc_virtual_shadow_sampling.shared")
file(READ "${arc_vsm_shader_include}/arc_virtual_shadow_sampling.slang" ARC_VSM_SLANG_ADAPTER)
file(READ "${arc_vsm_shader_include}/arc_virtual_shadow_sampling.shared" ARC_VSM_SHARED_LOOKUP)
configure_file(src/common/virtual_shadow_sampling_source.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/virtual_shadow_sampling_source.h" @ONLY)
target_include_directories(arc-render-tools PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

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
